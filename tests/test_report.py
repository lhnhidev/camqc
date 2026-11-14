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


# --- drawing with OpenCV -------------------------------------------------------------------


def test_draw_overlays_puts_shapes_at_their_coordinates() -> None:
    image = gray_image(0)
    out = report.draw_overlays(image, [(FAIL, CIRCLE), (PASS, RECT)])
    assert has_color(out, (80, 50), RED, tol=10) and has_color(out, (40, 50), RED, tol=10)  # circle: x +- r
    assert has_color(out, (60, 30), RED, tol=10) and has_color(out, (60, 70), RED, tol=10)  # circle: y +- r
    assert has_color(out, (190, 100), GREEN, tol=10) and has_color(out, (190, 139), GREEN, tol=10)  # rect top/bottom
    assert has_color(out, (150, 120), GREEN, tol=10) and has_color(out, (229, 120), GREEN, tol=10)  # rect left/right
    assert not out[50, 60].any() and not out[120, 190].any()  # centers/interiors are not filled
    assert not image.any()  # the input is not modified


def test_draw_overlays_colors_and_ignored_shapes() -> None:
    out = report.draw_overlays(gray_image(0), [("ERROR", CIRCLE), (FAIL, {"type": "polygon"})])
    assert has_color(out, (80, 50), (27, 133, 255), tol=10)  # other statuses are orange (#ff851b, in BGR)
    assert report.draw_overlays(gray_image(0), []).shape == (200, 300, 3)


def test_draw_overlays_accepts_gray_images_and_scales_the_line_width() -> None:
    gray = np.zeros((200, 300), np.uint8)
    out = report.draw_overlays(gray, [(FAIL, RECT)])
    assert out.shape == (200, 300, 3) and has_color(out, (190, 100), RED, tol=10)
    big = report.draw_overlays(np.zeros((1500, 3000, 3), np.uint8),
                               [(FAIL, {"type": "rect", "x": 100, "y": 100, "w": 500, "h": 300})])
    thin = report.draw_overlays(np.zeros((200, 300, 3), np.uint8), [(FAIL, RECT)])
    big_width = int((big[250, 90:112] > 0).any(axis=1).sum())  # the rect's left edge, crossed at mid-height
    thin_width = int((thin[120, 140:162] > 0).any(axis=1).sum())
    assert big_width > thin_width >= 2  # lines get thicker with the image so they stay visible after scaling


def test_image_data_uri_round_trips_and_downscales_wide_images() -> None:
    uri = report.image_data_uri(gray_image(120))
    assert uri.startswith("data:image/jpeg;base64,")
    decoded = cv2.imdecode(np.frombuffer(base64.b64decode(uri.split(",", 1)[1]), np.uint8), cv2.IMREAD_COLOR)
    assert decoded.shape == (200, 300, 3) and abs(int(decoded.mean()) - 120) <= 2
    wide = cv2.imdecode(np.frombuffer(base64.b64decode(report.image_data_uri(gray_image(0, (1600, 3200))).split(",", 1)[1]),
                                      np.uint8), cv2.IMREAD_COLOR)
    assert wide.shape == (report.REPORT_MAX_WIDTH // 2, report.REPORT_MAX_WIDTH, 3)


# --- the HTML report -----------------------------------------------------------------------


def test_html_has_header_results_and_images(conn: sqlite3.Connection, overlay_run: int) -> None:
    text = html_of(conn, overlay_run)
    assert text.startswith("<!DOCTYPE html>") and "<title>CamQC report - run #1 - SN-0042</title>" in text
    for fact in ("SN-0042", "CAM_B", "ST9", "2026-11-14T09:30:00"):
        assert f"<dd>{fact}</dd>" in text
    assert '<span class="verdict" style="background:#ffc7ce; color:#9c0006">FAIL</span>' in text
    for name in ("blemish", "shading", "brightness", "pixels_dead"):
        assert f"<td>{name}</td>" in text
    assert "1 blemishes above max 0" in text and "required frame missing" in text
    assert re.search(r"<td>pixels_dead</td><td class=\"num\">-</td><td>-</td><td>-</td><td>SKIPPED</td>", text)
    assert len(embedded_images(text)) == 2  # flat and normal; the unused dark frame was never stored


def test_embedded_images_carry_the_overlays(conn: sqlite3.Connection, overlay_run: int) -> None:
    text = html_of(conn, overlay_run)
    normal, flat = embedded_images(text)  # normal comes before flat
    assert text.index("Normal frame") < text.index("Flat frame")
    assert has_color(flat, (80, 50), RED, tol=45) and has_color(flat, (190, 100), GREEN, tol=45)
    assert abs(int(normal.mean()) - 150) <= 2  # brightness has no overlay: the image is untouched
    assert "blemish (FAIL, 1 marks)" in text and "shading (PASS, 1 marks)" in text
    assert "brightness (PASS" not in text  # no marks, no legend entry


def test_html_is_self_contained(conn: sqlite3.Connection, overlay_run: int) -> None:
    text = html_of(conn, overlay_run)
    assert not re.search(r"<(script|link|iframe)\b", text, re.IGNORECASE)
    assert not re.search(r"(href|src)=\"(?!data:image/jpeg;base64,)", text)
    assert "http://" not in text and "https://" not in text and "url(" not in text


def test_html_escapes_user_supplied_text(conn: sqlite3.Connection) -> None:
    result = run_result(FAIL, checked("sharp<b>", FAIL, "flat", [], reason="<script>alert(1)</script> & co"))
    run_id = db.save_run(conn, db.RunInput('<img src=x onerror=1> "&"', "M<1>", "S&T", datetime(2026, 11, 14, 9)), result)
    text = html_of(conn, run_id)
    assert "<script>" not in text and "<img src=x" not in text and "onerror=1>" not in text
    assert "&lt;img src=x onerror=1&gt; &quot;&amp;&quot;" in text and "M&lt;1&gt;" in text and "S&amp;T" in text
    assert "&lt;script&gt;alert(1)&lt;/script&gt; &amp; co" in text and "sharp&lt;b&gt;" in text


def test_report_without_stored_images_says_so(conn: sqlite3.Connection) -> None:
    run_id = add_run(conn, "NOIMG", datetime(2026, 11, 14, 9), PASSING)  # saved without frames
    text = html_of(conn, run_id)
    assert "No input image was stored for this run." in text and embedded_images(text) == []
    assert "<td>brightness</td>" in text


def test_report_of_a_run_saved_before_overlays_were_stored(conn: sqlite3.Connection) -> None:
    run_id = add_run(conn, "OLD", datetime(2026, 11, 14, 9), PASSING)
    conn.execute("UPDATE results SET details_json = '{\"frame\": \"flat\", \"reason\": null, \"details\": {}}' WHERE run_id = ?",
                 (run_id,))  # the old payload had no "overlay" key
    conn.commit()
    conn.execute("INSERT INTO run_images (run_id, frame, format, data) VALUES (?, 'flat', 'jpg', ?)",
                 (run_id, cv2.imencode(".jpg", gray_image())[1].tobytes()))
    conn.commit()
    text = html_of(conn, run_id)
    assert len(embedded_images(text)) == 1 and "marks)" not in text


# --- export_html ---------------------------------------------------------------------------


def test_export_html_writes_a_file_containing_the_serial(conn: sqlite3.Connection, overlay_run: int, tmp_path: Path) -> None:
    out = tmp_path / "deep" / "dir" / "report.html"
    assert report.export_html(conn, overlay_run, out) == out
    text = out.read_text(encoding="utf-8")
    assert "SN-0042" in text and "CAM_B" in text and out.stat().st_size > 1000
    assert text == html_of(conn, overlay_run)  # the file is exactly the rendered report


def test_export_html_unknown_run_raises_and_writes_nothing(conn: sqlite3.Connection, tmp_path: Path) -> None:
    out = tmp_path / "none.html"
    with pytest.raises(report.RunNotFoundError, match="run #99 not found"):
        report.export_html(conn, 99, out)
    assert not out.exists() and isinstance(report.RunNotFoundError("x"), LookupError)


# --- export_csv ----------------------------------------------------------------------------


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


def test_non_text_values_in_a_result_do_not_break_rendering(conn: sqlite3.Connection) -> None:
    run_id = add_run(conn, "ODD", datetime(2026, 11, 14, 9), PASSING)
    conn.execute("UPDATE results SET details_json = '{\"frame\": \"flat\", \"reason\": 42, \"details\": {}}' WHERE run_id = ?",
                 (run_id,))
    conn.commit()
    assert "<td>42</td>" in html_of(conn, run_id)
