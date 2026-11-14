"""SQLite persistence: store and query test results keyed by module serial number.

Plain ``sqlite3`` with parameterized queries only. Every value that reaches SQL is
bound as a parameter; the only SQL assembled dynamically is fixed ``WHERE`` fragments.

Dates are inclusive calendar days (``date``, ``datetime`` or ``"YYYY-MM-DD"``);
``None`` means unbounded. ``started_at`` is stored as naive local ISO-8601 text
(``YYYY-MM-DDTHH:MM:SS``), so text order is chronological order. Stations should run in a
time zone without DST changes (or on a fixed clock): during a repeated local hour,
ordering and day buckets can be off.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np

from camqc.pipeline import CheckOutcome, RunResult

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id          INTEGER PRIMARY KEY,
    serial      TEXT NOT NULL,
    model       TEXT NOT NULL,
    station     TEXT NOT NULL,
    started_at  TEXT NOT NULL,
    verdict     TEXT NOT NULL,
    duration_ms INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS results (
    id           INTEGER PRIMARY KEY,
    run_id       INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    check_name   TEXT NOT NULL,
    value        REAL,
    unit         TEXT,
    low          REAL,
    high         REAL,
    status       TEXT NOT NULL,
    details_json TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_runs_serial ON runs(serial);
CREATE INDEX IF NOT EXISTS idx_runs_started_at ON runs(started_at);
CREATE INDEX IF NOT EXISTS idx_results_check_status ON results(check_name, status);
CREATE INDEX IF NOT EXISTS idx_results_run_id ON results(run_id);
"""

DateLike = date | datetime | str | None


@dataclass(frozen=True)
class RunInput:
    """Identity of a test run, supplied by the caller.

    Attributes:
        serial: Module serial number (non-empty); a retest reuses the same serial.
        model: Camera model name, e.g. ``"CAM_A"``.
        station: Test station identifier.
        started_at: When the run started (aware datetimes are converted to local time).
    """

    serial: str
    model: str
    station: str
    started_at: datetime


# --- helpers ---------------------------------------------------------------------------


def _json_default(obj: Any) -> Any:
    """Serialize NumPy scalars/arrays that checks may leave in ``details``."""
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"{type(obj).__name__} is not JSON serializable")


def _started_at_text(value: datetime) -> str:
    """Return ``value`` as naive local ISO-8601 text with second precision."""
    if value.tzinfo is not None:
        value = value.astimezone().replace(tzinfo=None)
    return value.isoformat(timespec="seconds")


def _to_day(value: date | datetime | str) -> date:
    """Convert a date-like value to a ``date``."""
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return datetime.fromisoformat(value).date()


def _range_clause(date_from: DateLike, date_to: DateLike) -> tuple[str, list[str]]:
    """Return an ``AND ...`` fragment on ``runs.started_at`` and its bound parameters."""
    clause, params = "", []
    if date_from is not None:
        clause += " AND runs.started_at >= ?"
        params.append(_to_day(date_from).isoformat())
    if date_to is not None:
        clause += " AND runs.started_at < ?"
        params.append((_to_day(date_to) + timedelta(days=1)).isoformat())
    return clause, params


def _result_row(run_id: int, outcome: CheckOutcome) -> tuple[Any, ...]:
    """Build the ``results`` row for one :class:`~camqc.pipeline.CheckOutcome`."""
    res = outcome.result
    payload = {
        "frame": outcome.frame,
        "reason": outcome.reason,
        "details": res.details if res is not None else {},
    }
    return (
        run_id,
        outcome.name,
        res.value if res is not None else None,
        res.unit if res is not None else None,
        res.low if res is not None else None,
        res.high if res is not None else None,
        outcome.status,
        json.dumps(payload, default=_json_default),
    )


# --- public API ------------------------------------------------------------------------


def init_db(path: str | Path) -> sqlite3.Connection:
    """Open (creating if needed) the database at ``path`` and ensure the schema exists.

    Args:
        path: File path, or ``":memory:"`` for a throwaway database. Missing parent
            directories are created.

    Returns:
        A connection with ``sqlite3.Row`` rows and foreign keys enforced.
    """
    if str(path) != ":memory:":
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA busy_timeout = 5000")
    conn.executescript(SCHEMA)
    conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
    conn.commit()
    return conn


def save_run(conn: sqlite3.Connection, run_input: RunInput, run_result: RunResult) -> int:
    """Store a run and all its check results in a single transaction.

    Args:
        conn: Connection from :func:`init_db`.
        run_input: Serial, model, station and start time.
        run_result: Pipeline output; every check outcome (including SKIPPED and
            ERROR) becomes a ``results`` row.

    Returns:
        The new ``runs.id``.

    Raises:
        ValueError: If the serial is empty or blank.
        Exception: Anything raised while writing (the transaction is rolled back,
            so no partial run is left behind).

    Note:
        This function owns the transaction: it commits on success, and any
        transaction the caller left open on ``conn`` is committed with it.
        Surrounding whitespace is stripped from the serial before storing.
    """
    serial = run_input.serial.strip()
    if not serial:
        raise ValueError("serial must not be empty")
    with conn:  # commits on success, rolls back on any exception
        cur = conn.execute(
            "INSERT INTO runs (serial, model, station, started_at, verdict, duration_ms)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            (
                serial,
                run_input.model,
                run_input.station,
                _started_at_text(run_input.started_at),
                run_result.verdict,
                int(round(run_result.duration_ms)),
            ),
        )
        run_id = int(cur.lastrowid)
        conn.executemany(
            "INSERT INTO results (run_id, check_name, value, unit, low, high, status, details_json)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            [_result_row(run_id, o) for o in run_result.results],
        )
    logger.info("saved run %d serial=%s verdict=%s", run_id, serial, run_result.verdict)
    return run_id


def find_by_serial(conn: sqlite3.Connection, serial: str) -> list[dict[str, Any]]:
    """Return all runs of ``serial`` (initial test and retests), newest first."""
    rows = conn.execute(
        "SELECT * FROM runs WHERE serial = ? ORDER BY started_at DESC, id DESC", (serial,)
    ).fetchall()
    return [dict(r) for r in rows]


def list_runs(
    conn: sqlite3.Connection,
    date_from: DateLike,
    date_to: DateLike,
    verdict: str | None = None,
) -> list[dict[str, Any]]:
    """Return runs started within the inclusive day range, newest first.

    Args:
        conn: Database connection.
        date_from: First day, or ``None`` for no lower bound.
        date_to: Last day (inclusive), or ``None`` for no upper bound.
        verdict: If given, only runs with this verdict.
    """
    clause, params = _range_clause(date_from, date_to)
    if verdict is not None:
        clause += " AND runs.verdict = ?"
        params.append(verdict)
    rows = conn.execute(
        "SELECT * FROM runs WHERE 1 = 1" + clause + " ORDER BY started_at DESC, id DESC", params
    ).fetchall()
    return [dict(r) for r in rows]


LIKE_ESCAPE = "!"


def _like_pattern(text: str) -> str:
    """Return a ``LIKE`` pattern matching ``text`` as a substring (``%`` and ``_`` are literal)."""
    for char in (LIKE_ESCAPE, "%", "_"):
        text = text.replace(char, LIKE_ESCAPE + char)
    return f"%{text}%"


def search_runs(
    conn: sqlite3.Connection,
    serial_contains: str | None = None,
    date_from: DateLike = None,
    date_to: DateLike = None,
    verdict: str | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Return runs whose serial contains ``serial_contains``, newest first.

    Args:
        conn: Database connection.
        serial_contains: Substring to look for in the serial (ASCII case-insensitive,
            wildcards in the text are matched literally); ``None`` or ``""`` matches all.
        date_from: First day, or ``None`` for no lower bound.
        date_to: Last day (inclusive), or ``None`` for no upper bound.
        verdict: If given, only runs with this verdict.
        limit: Maximum number of rows, or ``None`` for all.
    """
    clause, params = _range_clause(date_from, date_to)
    if serial_contains:
        clause += f" AND runs.serial LIKE ? ESCAPE '{LIKE_ESCAPE}'"
        params.append(_like_pattern(serial_contains))
    if verdict is not None:
        clause += " AND runs.verdict = ?"
        params.append(verdict)
    sql = "SELECT * FROM runs WHERE 1 = 1" + clause + " ORDER BY started_at DESC, id DESC"
    if limit is not None:
        sql += " LIMIT ?"
        params.append(limit)
    return [dict(r) for r in conn.execute(sql, params).fetchall()]


def get_run_detail(conn: sqlite3.Connection, run_id: int) -> dict[str, Any] | None:
    """Return a run with its ``results`` list, or ``None`` if the id is unknown.

    Each result dict has the table columns, with ``details_json`` replaced by
    ``frame``, ``reason`` and ``details`` (parsed JSON). Results keep insertion order.
    """
    run = conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
    if run is None:
        return None
    detail = dict(run)
    detail["results"] = []
    for row in conn.execute("SELECT * FROM results WHERE run_id = ? ORDER BY id", (run_id,)):
        item = dict(row)
        item.update(json.loads(item.pop("details_json")))
        detail["results"].append(item)
    return detail


def yield_by_day(conn: sqlite3.Connection, date_from: DateLike, date_to: DateLike) -> list[dict[str, Any]]:
    """Return per-day run counts and yield, oldest day first.

    Every run counts, so a retest is a separate run (this is run yield, not first-pass
    yield). Days without runs are omitted.

    Returns:
        Dicts with ``day``, ``total``, ``passed``, ``failed``, ``incomplete`` and
        ``yield_pct`` (``passed / total * 100``).
    """
    clause, params = _range_clause(date_from, date_to)
    rows = conn.execute(
        "SELECT substr(started_at, 1, 10) AS day, COUNT(*) AS total,"
        " SUM(verdict = 'PASS') AS passed, SUM(verdict = 'FAIL') AS failed,"
        " SUM(verdict = 'INCOMPLETE') AS incomplete"
        " FROM runs WHERE 1 = 1" + clause + " GROUP BY day ORDER BY day",
        params,
    ).fetchall()
    return [{**dict(r), "yield_pct": r["passed"] / r["total"] * 100.0} for r in rows]


def pareto_failures(conn: sqlite3.Connection, date_from: DateLike, date_to: DateLike) -> list[dict[str, Any]]:
    """Return failing checks ranked by how often they failed (Pareto order).

    Only results with status ``FAIL`` count (not SKIPPED or ERROR).

    Returns:
        Dicts with ``check_name``, ``fail_count``, ``pct`` (share of all check
        failures) and ``cumulative_pct``, most frequent first (ties by name).
    """
    clause, params = _range_clause(date_from, date_to)
    rows = conn.execute(
        "SELECT results.check_name AS check_name, COUNT(*) AS fail_count"
        " FROM results JOIN runs ON runs.id = results.run_id"
        " WHERE results.status = 'FAIL'" + clause +
        " GROUP BY results.check_name ORDER BY fail_count DESC, results.check_name ASC",
        params,
    ).fetchall()
    total = sum(r["fail_count"] for r in rows)
    out: list[dict[str, Any]] = []
    running = 0
    for r in rows:
        running += r["fail_count"]
        out.append({
            "check_name": r["check_name"],
            "fail_count": r["fail_count"],
            "pct": r["fail_count"] / total * 100.0,
            "cumulative_pct": running / total * 100.0,
        })
    return out
