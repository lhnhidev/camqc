"""Command-line entry point: run checks on image files or folders, and query stored history and statistics.

Commands::

    camqc init-db --db camqc.db
    camqc run --model CAM_A --serial S001 --station ST01 --normal n.png --dark d.png --white w.png --db camqc.db
    camqc batch --dir samples/ --model CAM_A --db camqc.db
    camqc history --serial S001
    camqc stats --from 2026-11-01 --to 2026-11-30

Exit codes: ``0`` PASS (and successful queries), ``1`` FAIL or INCOMPLETE, ``2`` errors
(bad arguments, missing files, invalid config, database problems).
"""

from __future__ import annotations

import argparse
import logging
import logging.handlers
import os
import sqlite3
import sys
from collections.abc import Mapping, Sequence
from datetime import date, datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from camqc import db
from camqc.pipeline import (
    FRAME_TYPES,
    PASS,
    Config,
    ConfigError,
    RunResult,
    load_config,
    run_pipeline,
)

logger = logging.getLogger(__name__)

EXIT_OK, EXIT_NOT_PASS, EXIT_ERROR = 0, 1, 2
DEFAULT_DB = "camqc.db"
DEFAULT_LOG = Path("logs") / "camqc.log"
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"

ANSI_RESET = "\x1b[0m"
STATUS_COLORS = {
    "PASS": "\x1b[32m",
    "FAIL": "\x1b[31m",
    "INCOMPLETE": "\x1b[33m",
    "SKIPPED": "\x1b[33m",
    "ERROR": "\x1b[31m",
}


class CliError(Exception):
    """A user-facing error: printed as ``error: ...`` and mapped to exit code 2."""


# --- output ----------------------------------------------------------------------------


def _enable_windows_ansi() -> bool:
    """Turn on ANSI escape processing for the Windows console; return True on success."""
    try:
        import ctypes

        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        handle = kernel32.GetStdHandle(-11)
        mode = ctypes.c_ulong()
        if not kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
            return False
        return bool(kernel32.SetConsoleMode(handle, mode.value | 0x0004))
    except (AttributeError, OSError):
        return False


def use_color(stream: Any) -> bool:
    """Return True if ``stream`` is a TTY that should receive ANSI colors.

    Honors the ``NO_COLOR`` environment variable and enables VT processing on Windows.
    """
    if os.environ.get("NO_COLOR") or not getattr(stream, "isatty", lambda: False)():
        return False
    return _enable_windows_ansi() if sys.platform == "win32" else True


def format_table(
    headers: Sequence[str],
    rows: Sequence[Sequence[Any]],
    colorize: bool = False,
    align: str = "",
) -> str:
    """Return an aligned text table.

    Args:
        headers: Column titles.
        rows: Row cells; converted with ``str``.
        colorize: Wrap cells that equal a status word (PASS, FAIL, ...) in ANSI colors.
            Widths are computed on the plain text, so colors never break alignment.
        align: One ``l`` or ``r`` per column (default all left).
    """
    text = [[str(c) for c in row] for row in rows]
    widths = [max([len(h), *(len(r[i]) for r in text)]) for i, h in enumerate(headers)]
    aligns = (align + "l" * len(headers))[: len(headers)]

    def cell(value: str, i: int) -> str:
        pad = " " * (widths[i] - len(value))
        shown = value
        if colorize and value in STATUS_COLORS:
            shown = f"{STATUS_COLORS[value]}{value}{ANSI_RESET}"
        return pad + shown if aligns[i] == "r" else shown + pad

    def line(cells: Sequence[str]) -> str:
        return "  ".join(cell(c, i) for i, c in enumerate(cells)).rstrip()

    return "\n".join([line(headers), "  ".join("-" * w for w in widths), *(line(r) for r in text)])


def emit(text: str = "") -> None:
    """Write user-facing output to stdout (resolved at call time so tests can capture it)."""
    print(text)


def fail(message: str) -> None:
    """Write an error line to stderr."""
    print(f"error: {message}", file=sys.stderr)


def status_word(verdict: str, colorize: bool) -> str:
    """Return ``verdict``, wrapped in its ANSI color when ``colorize`` is set."""
    return f"{STATUS_COLORS[verdict]}{verdict}{ANSI_RESET}" if colorize and verdict in STATUS_COLORS else verdict


def format_limits(low: float | None, high: float | None) -> str:
    """Describe an acceptance range, e.g. ``100 .. 160``, ``>= 100``, ``<= 5`` or ``-``."""
    if low is not None and high is not None:
        return f"{low:g} .. {high:g}"
    if low is not None:
        return f">= {low:g}"
    if high is not None:
        return f"<= {high:g}"
    return "-"


def results_table(run: RunResult, colorize: bool) -> str:
    """Return the per-check table of a pipeline run."""
    rows = []
    for o in run.results:
        res = o.result
        reason = o.reason or (res.details.get("reason", "") if res else "")
        rows.append([
            o.name,
            o.status,
            f"{res.value:.5g} {res.unit}" if res else "-",
            format_limits(res.low, res.high) if res else "-",
            o.frame or "-",
            reason,
        ])
    return format_table(["check", "status", "value", "limits", "frame", "note"], rows, colorize, "ll")


# --- helpers ---------------------------------------------------------------------------


def exit_code_for(verdict: str) -> int:
    """Return 0 for PASS and 1 for FAIL or INCOMPLETE."""
    return EXIT_OK if verdict == PASS else EXIT_NOT_PASS


def default_config_path() -> Path:
    """Return ``config/models.yaml`` from the working directory, else from the source tree."""
    local = Path("config") / "models.yaml"
    return local if local.is_file() else Path(__file__).resolve().parents[2] / "config" / "models.yaml"


def load_image(path: str | Path) -> np.ndarray:
    """Read an image as BGR (works with non-ASCII Windows paths).

    Raises:
        CliError: If the file is missing or cannot be decoded as an image.
    """
    path = Path(path)
    if not path.is_file():
        raise CliError(f"image not found: {path}")
    data = np.fromfile(str(path), dtype=np.uint8)
    try:
        image = cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None
    except cv2.error:
        image = None
    if image is None:
        raise CliError(f"cannot decode image: {path}")
    return image


def load_config_for(args: argparse.Namespace) -> Config:
    """Load the configuration named by ``--config`` and check that ``--model`` exists in it."""
    config = load_config(args.config or default_config_path())
    if args.model not in config.models:
        raise CliError(f"unknown model {args.model!r}; available: {', '.join(sorted(config.models))}")
    return config


def open_existing_db(path: str | Path) -> sqlite3.Connection:
    """Open a database for a read-only command without creating it as a side effect.

    Raises:
        CliError: If the file does not exist (usually a mistyped ``--db``).
    """
    if not Path(path).is_file():
        raise CliError(f"database not found: {path} (create it with 'camqc init-db')")
    return db.init_db(path)


def parse_day(text: str) -> date:
    """argparse type: parse ``YYYY-MM-DD``."""
    try:
        return date.fromisoformat(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"invalid date {text!r}, expected YYYY-MM-DD") from None


def setup_logging(log_file: Path, verbose: bool) -> logging.Handler | None:
    """Log the ``camqc`` loggers to a rotating file; return the handler (or None on failure)."""
    try:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handler = logging.handlers.RotatingFileHandler(
            log_file, maxBytes=1_000_000, backupCount=3, encoding="utf-8"
        )
    except OSError as exc:
        print(f"warning: cannot write log file {log_file}: {exc}", file=sys.stderr)
        return None
    handler.setFormatter(logging.Formatter(LOG_FORMAT))
    root = logging.getLogger("camqc")
    root.setLevel(logging.DEBUG if verbose else logging.INFO)
    root.addHandler(handler)
    return handler


def teardown_logging(handler: logging.Handler | None) -> None:
    """Detach and close the handler created by :func:`setup_logging`."""
    if handler is not None:
        logging.getLogger("camqc").removeHandler(handler)
        handler.close()


def execute(
    conn: sqlite3.Connection,
    config: Config,
    model: str,
    serial: str,
    station: str,
    frames: Mapping[str, np.ndarray],
) -> tuple[int, RunResult]:
    """Run the pipeline, store the run, and return ``(run_id, result)``."""
    result = run_pipeline(model, frames, config)
    run_id = db.save_run(conn, db.RunInput(serial, model, station, datetime.now()), result)
    logger.info("run %d serial=%s model=%s verdict=%s failed=%s",
                run_id, serial, model, result.verdict, list(result.failed_checks))
    return run_id, result
