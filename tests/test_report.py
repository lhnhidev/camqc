"""Tests for the CSV export and the self-contained HTML report."""

from __future__ import annotations

import base64
import csv
import re
import sqlite3
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest
from seed_support import PASSING, add_run, run_result, seed

from camqc import db, report
from camqc.checks.base import make_result
from camqc.pipeline import FAIL, PASS, SKIPPED, CheckOutcome, RunResult

CIRCLE = {"type": "circle", "x": 60, "y": 50, "r": 20}
RECT = {"type": "rect", "x": 150, "y": 100, "w": 80, "h": 40}
RED, GREEN = (54, 65, 255), (64, 204, 46)  # BGR of the overlay colors


@pytest.fixture
def conn() -> Iterator[sqlite3.Connection]:
    connection = db.init_db(":memory:")
    yield connection
    connection.close()


def gray_image(value: int = 100, shape: tuple[int, int] = (200, 300)) -> np.ndarray:
    return np.full((*shape, 3), value, np.uint8)


def checked(name: str, status: str, frame: str, overlay: list[dict[str, Any]], reason: str = "") -> CheckOutcome:
    details = {"reason": reason} if reason else {}
    return CheckOutcome(name, status, make_result(name, 5.0, "u", 1.0, 10.0, details, overlay), None, frame, 1.0)


def has_color(image: np.ndarray, xy: tuple[int, int], bgr: tuple[int, int, int], tol: int = 40) -> bool:
    """True if a pixel within 1 px of ``xy`` is close to ``bgr`` (edges are anti-aliased)."""
    x, y = xy
    patch = image[max(y - 1, 0): y + 2, max(x - 1, 0): x + 2].reshape(-1, 3).astype(int)
    return bool((np.abs(patch - np.array(bgr)).max(axis=1) <= tol).any())


@pytest.fixture
def overlay_run(conn: sqlite3.Connection) -> int:
    """A run with a failing check (circle) and a passing one (rect) on the flat frame, plus a skipped one."""
    result = run_result(
        FAIL,
        checked("blemish", FAIL, "flat", [CIRCLE], reason="1 blemishes above max 0"),
        checked("shading", PASS, "flat", [RECT]),
        checked("brightness", PASS, "normal", []),
        CheckOutcome("pixels_dead", SKIPPED, None, "required frame missing", None, 0.1),
    )
    frames = {"flat": gray_image(100), "normal": gray_image(150), "dark": gray_image(5)}  # dark is unused
    return db.save_run(conn, db.RunInput("SN-0042", "CAM_B", "ST9", datetime(2026, 11, 14, 9, 30)), result, frames)


def html_of(conn: sqlite3.Connection, run_id: int) -> str:
    detail = db.get_run_detail(conn, run_id)
    assert detail is not None
    return report.render_html(detail, db.get_run_images(conn, run_id))


def embedded_images(text: str) -> list[np.ndarray]:
    images = []
    for b64 in re.findall(r'src="data:image/jpeg;base64,([A-Za-z0-9+/=]+)"', text):
        images.append(cv2.imdecode(np.frombuffer(base64.b64decode(b64), np.uint8), cv2.IMREAD_COLOR))
    return images


def read_csv(path: Path) -> list[list[str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.reader(handle))


def test_export_csv_writes_one_row_per_check_with_a_bom(conn: sqlite3.Connection, tmp_path: Path) -> None:
    seed(conn)
    out = tmp_path / "out" / "runs.csv"
    assert report.export_csv(conn, None, None, out) == 12
    assert out.read_bytes().startswith(b"\xef\xbb\xbf")  # Excel needs the BOM for UTF-8
    rows = read_csv(out)
    assert tuple(rows[0]) == report.CSV_COLUMNS and len(rows) == 13
    first = dict(zip(rows[0], rows[1]))
    assert (first["serial"], first["model"], first["station"], first["verdict"]) == ("A1", "CAM_A", "ST1", "PASS")
    assert (first["started_at"], first["check"], first["status"], first["unit"]) == ("2026-11-14T08:00:00", "brightness", "PASS", "u")
    assert (first["low"], first["high"], first["duration_ms"]) == ("0.0", "10.0", "120")
    fail = next(dict(zip(rows[0], r)) for r in rows[1:] if r[1] == "A2" and r[7] == "blemish")
    assert fail["status"] == "FAIL" and fail["reason"] == "blemish out of range"
    skipped = next(dict(zip(rows[0], r)) for r in rows[1:] if r[7] == "pixels_dead")
    assert skipped["status"] == "SKIPPED" and skipped["value"] == "" and skipped["reason"] == "required frame missing"
    assert [r[0] for r in rows[1:]] == sorted((r[0] for r in rows[1:]), key=int)  # grouped by run, in order


def test_export_csv_date_range_and_empty_result(conn: sqlite3.Connection, tmp_path: Path) -> None:
    seed(conn)
    out = tmp_path / "range.csv"
    assert report.export_csv(conn, "2026-11-15", "2026-11-15", out) == 4
    assert {r[1] for r in read_csv(out)[1:]} == {"B1", "B2"}
    assert report.export_csv(conn, "2027-01-01", None, out) == 0
    assert read_csv(out) == [list(report.CSV_COLUMNS)]  # header only


def test_export_csv_keeps_runs_without_results(conn: sqlite3.Connection, tmp_path: Path) -> None:
    db.save_run(conn, db.RunInput("EMPTY", "CAM_A", "ST1", datetime(2026, 11, 14, 8)), RunResult("PASS", (), (), 1.0))
    out = tmp_path / "e.csv"
    assert report.export_csv(conn, None, None, out) == 1
    row = dict(zip(*read_csv(out)))
    assert row["serial"] == "EMPTY" and row["check"] == "" and row["status"] == ""


def test_export_csv_quotes_commas_and_quotes_and_neutralizes_formulas(conn: sqlite3.Connection, tmp_path: Path) -> None:
    nasty = ['He said "hi", ok', "=HYPERLINK(\"http://evil\")", "@SUM(1)", "-5", "+1", "-cmd"]
    for i, serial in enumerate(nasty):
        add_run(conn, serial, datetime(2026, 11, 14, 8, i), run_result(PASS, checked("c", PASS, "flat", [])))
    out = tmp_path / "n.csv"
    report.export_csv(conn, None, None, out)
    serials = [r[1] for r in read_csv(out)[1:]]
    assert serials == ['He said "hi", ok', "'=HYPERLINK(\"http://evil\")", "'@SUM(1)", "-5", "+1", "'-cmd"]


@pytest.mark.parametrize(
    ("value", "expected"),
    [("=1+1", "'=1+1"), ("+x", "'+x"), ("-x", "'-x"), ("@x", "'@x"), ("\tx", "'\tx"), ("\rx", "'\rx"), ("-", "'-"),
     ("-5", "-5"), ("+1.5", "+1.5"), ("1e3", "1e3"), ("plain", "plain"), ("", ""), (None, None), (3.5, 3.5)],
)
def test_csv_safe(value: Any, expected: Any) -> None:
    assert report.csv_safe(value) == expected
