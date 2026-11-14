"""Export test results as CSV: one row per check result of every run in a date range."""

from __future__ import annotations

import csv
import logging
import sqlite3
from pathlib import Path
from typing import Any

from camqc import db

logger = logging.getLogger(__name__)

CSV_COLUMNS = (
    "run_id", "serial", "model", "station", "started_at", "verdict", "duration_ms",
    "check", "value", "unit", "low", "high", "status", "reason",
)
CSV_SOURCE_KEYS = {"check": "check_name"}  # CSV column -> db.export_rows key when they differ
FORMULA_PREFIXES = ("=", "+", "-", "@", "\t", "\r")


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
