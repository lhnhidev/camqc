"""Export test results: a CSV of runs/checks for a date range, and a self-contained HTML report per run.

The HTML report is one file (inline CSS, images as base64 data URIs, no external requests) with
the run header, the per-check results table and the run's input images with the checks'
overlays drawn on them via OpenCV. Everything user-supplied (serial, model, ...) is escaped.
"""

from __future__ import annotations

import base64
import csv
import html
import logging
import sqlite3
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from camqc import db
from camqc.checks.base import format_limits

logger = logging.getLogger(__name__)

CSV_COLUMNS = (
    "run_id", "serial", "model", "station", "started_at", "verdict", "duration_ms",
    "check", "value", "unit", "low", "high", "status", "reason",
)
CSV_SOURCE_KEYS = {"check": "check_name"}  # CSV column -> db.export_rows key when they differ
FORMULA_PREFIXES = ("=", "+", "-", "@", "\t", "\r")

REPORT_MAX_WIDTH = 1600  # wider images are downscaled before embedding
REPORT_JPEG_QUALITY = 85
FRAME_ORDER = ("normal", "flat", "dark", "white")

VERDICT_COLORS: Mapping[str, tuple[str, str]] = {  # status -> (background, text)
    "PASS": ("#c6efce", "#006100"),
    "FAIL": ("#ffc7ce", "#9c0006"),
    "INCOMPLETE": ("#ffeb9c", "#9c5700"),
    "SKIPPED": ("#ffeb9c", "#9c5700"),
    "ERROR": ("#ffc7ce", "#9c0006"),
}
NEUTRAL_COLORS = ("#e0e0e0", "#404040")
OVERLAY_HEX: Mapping[str, str] = {"PASS": "#2ecc40", "FAIL": "#ff4136"}  # same colors as the GUI viewer
OVERLAY_DEFAULT_HEX = "#ff851b"


class RunNotFoundError(LookupError):
    """Raised when a report is requested for a run id that does not exist."""


# --- CSV ---------------------------------------------------------------------------------


def _is_number(text: str) -> bool:
    try:
        float(text)
    except ValueError:
        return False
    return True


def csv_safe(value: Any) -> Any:
    """Neutralize spreadsheet formulas in a text cell (CSV injection).

    A text starting with ``= + - @``, a tab or a CR would be run as a formula by Excel
    and friends; such texts get a leading ``'``. Plain numbers (``-5``) are left alone.
    """
    if isinstance(value, str) and value.startswith(FORMULA_PREFIXES) and not _is_number(value):
        return "'" + value
    return value


def export_csv(
    conn: sqlite3.Connection, date_from: db.DateLike, date_to: db.DateLike, path: str | Path
) -> int:
    """Write a CSV with one row per check result of every run in the inclusive day range.

    Args:
        conn: Database connection.
        date_from: First day, or ``None`` for no lower bound.
        date_to: Last day (inclusive), or ``None`` for no upper bound.
        path: Output file (UTF-8 with BOM so Excel opens it correctly); parent folders
            are created.

    Returns:
        The number of data rows written (a run without results counts as one row).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(CSV_COLUMNS)
        for row in db.export_rows(conn, date_from, date_to):
            writer.writerow([csv_safe(row[CSV_SOURCE_KEYS.get(c, c)]) for c in CSV_COLUMNS])
            count += 1
    logger.info("exported %d rows to %s", count, path)
    return count


# --- images --------------------------------------------------------------------------------


def draw_overlays(image_bgr: np.ndarray, shapes: Sequence[tuple[str, Mapping[str, Any]]]) -> np.ndarray:
    """Return a BGR copy of ``image_bgr`` with check overlays drawn on it with OpenCV.

    Args:
        image_bgr: The input image (gray images are converted to BGR).
        shapes: ``(check status, shape)`` pairs; shapes are the ``CheckResult.overlay``
            dicts in image pixel coordinates (circle: center x, y and radius r; rect:
            top-left x, y and size w, h). PASS is drawn green, FAIL red, others orange.
    """
    out = image_bgr.copy() if image_bgr.ndim == 3 else cv2.cvtColor(image_bgr, cv2.COLOR_GRAY2BGR)
    thickness = max(2, round(max(out.shape[:2]) / 600))
    for status, shape in shapes:
        color = _bgr(OVERLAY_HEX.get(status, OVERLAY_DEFAULT_HEX))
        kind = shape.get("type")
        if kind == "circle":
            cv2.circle(out, (round(shape["x"]), round(shape["y"])), max(round(shape["r"]), 1),
                       color, thickness, cv2.LINE_AA)
        elif kind == "rect":
            x, y = round(shape["x"]), round(shape["y"])
            cv2.rectangle(out, (x, y), (x + round(shape["w"]) - 1, y + round(shape["h"]) - 1),
                          color, thickness, cv2.LINE_AA)
        else:
            logger.debug("ignoring overlay shape of unknown type: %r", shape)
    return out


def _bgr(hex_color: str) -> tuple[int, int, int]:
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (1, 3, 5))
    return b, g, r


def image_data_uri(image_bgr: np.ndarray) -> str:
    """Return the image as a ``data:image/jpeg;base64,...`` URI (downscaled if very wide)."""
    h, w = image_bgr.shape[:2]
    if w > REPORT_MAX_WIDTH:
        image_bgr = cv2.resize(image_bgr, (REPORT_MAX_WIDTH, round(h * REPORT_MAX_WIDTH / w)),
                               interpolation=cv2.INTER_AREA)
    ok, buffer = cv2.imencode(".jpg", image_bgr, [cv2.IMWRITE_JPEG_QUALITY, REPORT_JPEG_QUALITY])
    if not ok:
        raise ValueError("cannot encode the image")
    return "data:image/jpeg;base64," + base64.b64encode(buffer.tobytes()).decode("ascii")


# --- HTML ----------------------------------------------------------------------------------

STYLE = """
body { font-family: system-ui, "Segoe UI", sans-serif; margin: 0; padding: 24px; background: #f6f6f4; color: #1b1b1a; }
main { max-width: 1100px; margin: 0 auto; background: #fff; padding: 28px 32px; border-radius: 8px; }
h1 { margin: 0 0 4px; font-size: 1.5rem; } h2 { margin-top: 32px; font-size: 1.1rem; }
.meta { color: #52514e; margin: 0 0 18px; }
.verdict { display: inline-block; padding: 6px 18px; border-radius: 6px; font-size: 1.6rem; font-weight: 700; }
dl { display: grid; grid-template-columns: max-content 1fr; gap: 4px 18px; margin: 18px 0; }
dt { color: #52514e; } dd { margin: 0; font-weight: 600; }
table { border-collapse: collapse; width: 100%; font-variant-numeric: tabular-nums; }
th, td { text-align: left; padding: 6px 10px; border-bottom: 1px solid #e1e0d9; }
th { background: #f0efec; font-weight: 600; }
td.num { text-align: right; }
figure { margin: 18px 0; } figcaption { color: #52514e; margin-bottom: 6px; font-weight: 600; }
img { max-width: 100%; height: auto; border: 1px solid #c3c2b7; }
ul.legend { list-style: none; padding: 0; margin: 8px 0 0; display: flex; flex-wrap: wrap; gap: 6px 18px; }
.swatch { display: inline-block; width: 12px; height: 12px; margin-right: 6px; vertical-align: -1px; }
.note { color: #52514e; font-style: italic; }
footer { margin-top: 32px; color: #898781; font-size: 0.85rem; }
@media print { body { background: #fff; padding: 0; } main { padding: 0; } }
"""


def _colors(status: str) -> tuple[str, str]:
    return VERDICT_COLORS.get(status, NEUTRAL_COLORS)


def _result_cells(result: Mapping[str, Any]) -> list[str]:
    has_value = result["value"] is not None
    note = result.get("reason") or result.get("details", {}).get("reason", "")
    return [
        result["check_name"],
        f"{result['value']:.5g}" if has_value else "-",
        (result["unit"] or "-") if has_value else "-",
        format_limits(result["low"], result["high"]) if has_value else "-",
        result["status"],
        note,
    ]


def _results_table(detail: Mapping[str, Any]) -> str:
    esc = html.escape
    rows = []
    for result in detail["results"]:
        bg, fg = _colors(result["status"])
        cells = _result_cells(result)
        tds = "".join(
            f'<td class="num">{esc(str(c))}</td>' if i == 1 else f"<td>{esc(str(c))}</td>" for i, c in enumerate(cells)
        )
        rows.append(f'<tr style="background:{bg}; color:{fg}">{tds}</tr>')
    head = "".join(f"<th>{h}</th>" for h in ("Check", "Value", "Unit", "Limits", "Status", "Note"))
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(rows)}</tbody></table>"


def _frame_figures(detail: Mapping[str, Any], images: Mapping[str, np.ndarray]) -> str:
    esc = html.escape
    if not images:
        return '<p class="note">No input image was stored for this run.</p>'
    figures = []
    for frame in sorted(images, key=lambda f: (FRAME_ORDER.index(f) if f in FRAME_ORDER else len(FRAME_ORDER), f)):
        checks = [r for r in detail["results"] if r.get("frame") == frame]
        shapes = [(r["status"], shape) for r in checks for shape in r.get("overlay", [])]
        uri = image_data_uri(draw_overlays(images[frame], shapes))
        legend = "".join(
            f'<li><span class="swatch" style="background:{OVERLAY_HEX.get(r["status"], OVERLAY_DEFAULT_HEX)}"></span>'
            f'{esc(r["check_name"])} ({esc(r["status"])}, {len(r.get("overlay", []))} marks)</li>'
            for r in checks if r.get("overlay")
        )
        alt = f"{frame} frame with {len(shapes)} overlay marks"
        figures.append(
            f"<figure><figcaption>{esc(frame.capitalize())} frame</figcaption>"
            f'<img alt="{esc(alt)}" src="{uri}">'
            + (f'<ul class="legend">{legend}</ul>' if legend else "") + "</figure>"
        )
    return "".join(figures)


def render_html(detail: Mapping[str, Any], images: Mapping[str, np.ndarray]) -> str:
    """Return the complete HTML report for a run.

    Args:
        detail: A :func:`camqc.db.get_run_detail` result.
        images: ``{frame type: BGR image}`` from :func:`camqc.db.get_run_images` (may be empty).
    """
    esc = html.escape
    bg, fg = _colors(detail["verdict"])
    title = f"CamQC report - run #{detail['id']} - {detail['serial']}"
    facts = "".join(
        f"<dt>{label}</dt><dd>{esc(str(detail[key]))}</dd>"
        for label, key in (("Serial", "serial"), ("Model", "model"), ("Station", "station"),
                           ("Started", "started_at"))
    ) + f"<dt>Duration</dt><dd>{esc(str(detail['duration_ms']))} ms</dd>"
    return (
        "<!DOCTYPE html>\n<html lang=\"en\"><head><meta charset=\"utf-8\">"
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">"
        f"<title>{esc(title)}</title><style>{STYLE}</style></head><body><main>"
        f"<h1>{esc(title)}</h1><p class=\"meta\">Camera module quality check</p>"
        f'<span class="verdict" style="background:{bg}; color:{fg}">{esc(detail["verdict"])}</span>'
        f"<dl>{facts}</dl>"
        f"<h2>Check results</h2>{_results_table(detail)}"
        f"<h2>Input images</h2>{_frame_figures(detail, images)}"
        "<footer>Generated by CamQC</footer></main></body></html>\n"
    )


def export_html(conn: sqlite3.Connection, run_id: int, path: str | Path) -> Path:
    """Write the self-contained HTML report of one run.

    Args:
        conn: Database connection.
        run_id: The ``runs.id`` to report on.
        path: Output file; parent folders are created.

    Returns:
        The path written.

    Raises:
        RunNotFoundError: If there is no such run.
    """
    detail = db.get_run_detail(conn, run_id)
    if detail is None:
        raise RunNotFoundError(f"run #{run_id} not found")
    text = render_html(detail, db.get_run_images(conn, run_id))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    logger.info("wrote report for run %d to %s", run_id, path)
    return path
