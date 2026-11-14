"""Tests for the PySide6 main window (headless via QT_QPA_PLATFORM=offscreen)."""

from __future__ import annotations

import time
import tomllib
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QSize, Qt, QTimer  # noqa: E402
from PySide6.QtGui import QPixmap  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QApplication, QFileDialog  # noqa: E402

from camqc import db  # noqa: E402
from camqc.gui import main_window  # noqa: E402
from camqc.gui.main_window import MainWindow, ScaledImageLabel  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent

CONFIG = """
models:
  M:
    checks:
      brightness:
        frame: normal
        params: {mean_low: 100, mean_high: 160}
  N:
    checks:
      brightness:
        frame: normal
        params: {mean_low: 100, mean_high: 160}
      pixels_dead:
        check: pixels
        frame: white
        params: {mode: dead, diff_threshold: 50, max_defects: 5}
"""


@pytest.fixture(scope="session")
def qapp() -> Iterator[QApplication]:
    yield QApplication.instance() or QApplication([])


def wait_for(condition: Callable[[], bool], timeout: float = 20.0) -> None:
    """Pump the Qt event loop until ``condition()`` is true."""
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        QApplication.processEvents()
        if condition():
            return
        time.sleep(0.005)
    raise AssertionError("timed out waiting for condition")


def png(path: Path, value: int, size: tuple[int, int] = (60, 80)) -> Path:
    assert cv2.imwrite(str(path), np.full((*size, 3), value, np.uint8))
    return path


@pytest.fixture
def paths(tmp_path: Path) -> dict[str, Path]:
    config = tmp_path / "models.yaml"
    config.write_text(CONFIG, encoding="utf-8")
    return {
        "config": config,
        "db": tmp_path / "t.db",
        "good": png(tmp_path / "good.png", 128),
        "dark": png(tmp_path / "dark.png", 20),
    }


@pytest.fixture
def window(qapp: QApplication, paths: dict[str, Path]) -> Iterator[MainWindow]:
    win = MainWindow(paths["config"], paths["db"])
    win.show()
    yield win
    win.close()
    QApplication.processEvents()


def click(widget: Any) -> None:
    QTest.mouseClick(widget, Qt.MouseButton.LeftButton)


def run_and_wait(win: MainWindow, serial: str = "S001") -> None:
    tab = win.test_tab
    tab.serial_edit.setText(serial)
    click(tab.run_button)
    wait_for(lambda: tab.run_button.isEnabled() and not tab.is_running)


def runs(paths: dict[str, Path], serial: str) -> list[dict[str, Any]]:
    conn = db.init_db(paths["db"])
    try:
        return db.find_by_serial(conn, serial)
    finally:
        conn.close()


def row_color(win: MainWindow, row: int) -> str:
    return win.test_tab.table.item(row, 0).background().color().name()


def test_choose_button_loads_frame_and_shows_it(
    window: MainWindow, paths: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *a, **k: (str(paths["good"]), ""))
    tab = window.test_tab
    click(tab.frame_buttons["normal"])
    assert tab.frame_labels["normal"].text() == "good.png"
    assert tab.frame_labels["normal"].toolTip() == str(paths["good"])
    assert "Normal: good.png  (80x60)" in tab.caption.text()
    assert tab.viewer.pixmap() is not None and not tab.viewer.pixmap().isNull()

    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *a, **k: ("", ""))  # cancelled
    click(tab.frame_buttons["dark"])
    assert tab.frame_labels["dark"].text() == "not selected"


def test_bad_image_keeps_previous_frame_and_reports(window: MainWindow, paths: dict[str, Path], tmp_path: Path) -> None:
    tab = window.test_tab
    assert tab.set_frame("normal", paths["good"])
    garbage = tmp_path / "bad.png"
    garbage.write_bytes(b"nope")
    assert not tab.set_frame("normal", garbage)
    assert not tab.set_frame("dark", tmp_path / "missing.png")
    assert tab.frame_labels["normal"].text() == "good.png"
    assert "image not found" in tab.message_label.text()


def test_viewer_scales_to_fit_keeping_aspect_ratio(qapp: QApplication) -> None:
    label = ScaledImageLabel()
    label.show()  # resize events are only delivered to visible widgets
    label.resize(300, 200)
    pixmap = QPixmap(600, 300)
    pixmap.fill(Qt.GlobalColor.red)
    label.set_image(pixmap)
    QApplication.processEvents()
    assert label.pixmap().size() == QSize(300, 150)
    label.resize(240, 400)
    QApplication.processEvents()
    assert label.pixmap().size() == QSize(240, 120)  # width-limited, aspect ratio kept
    label.set_image(None)
    assert label.pixmap().isNull() and label.text() == "No image"
    assert label.sizePolicy().horizontalPolicy().name == "Ignored"  # big images never grow the window
    label.close()


def test_bgr_to_pixmap_keeps_colors_and_accepts_gray(qapp: QApplication) -> None:
    bgr = np.zeros((2, 3, 3), np.uint8)
    bgr[..., 0] = 255  # pure blue in BGR
    image = main_window.bgr_to_pixmap(bgr).toImage()
    assert (image.width(), image.height()) == (3, 2)
    assert image.pixelColor(0, 0).name() == "#0000ff"
    gray = main_window.bgr_to_pixmap(np.full((4, 5), 100, np.uint8)).toImage()
    assert (gray.width(), gray.height()) == (5, 4) and gray.pixelColor(0, 0).red() == 100


def test_run_pass_fills_table_verdict_and_database(window: MainWindow, paths: dict[str, Path]) -> None:
    tab = window.test_tab
    tab.model_combo.setCurrentText("M")
    tab.station_edit.setText("ST07")
    tab.set_frame("normal", paths["good"])
    tab.serial_edit.setText("S001")
    click(tab.run_button)
    assert not tab.run_button.isEnabled() and tab.is_running  # disabled while running
    wait_for(lambda: tab.run_button.isEnabled())

    assert tab.verdict_label.text() == "PASS"
    assert "#c6efce" in tab.verdict_label.styleSheet()
    assert tab.table.rowCount() == 1
    assert [tab.table.item(0, c).text() for c in range(5)] == ["brightness", "128", "gray", "100 .. 160", "PASS"]
    assert row_color(window, 0) == "#c6efce"
    (stored,) = runs(paths, "S001")
    assert (stored["model"], stored["station"], stored["verdict"]) == ("M", "ST07", "PASS")
    assert f"Saved as run #{stored['id']}" in tab.message_label.text()


def test_run_fail_is_red_and_incomplete_is_amber(window: MainWindow, paths: dict[str, Path]) -> None:
    tab = window.test_tab
    tab.model_combo.setCurrentText("M")
    tab.set_frame("normal", paths["dark"])
    run_and_wait(window)
    assert tab.verdict_label.text() == "FAIL" and row_color(window, 0) == "#ffc7ce"
    assert "below 100" in tab.table.item(0, 4).toolTip()

    tab.model_combo.setCurrentText("N")  # needs a white frame we do not have
    tab.set_frame("normal", paths["good"])
    run_and_wait(window, "S002")
    assert tab.verdict_label.text() == "INCOMPLETE" and "#ffeb9c" in tab.verdict_label.styleSheet()
    assert [tab.table.item(r, 4).text() for r in range(2)] == ["PASS", "SKIPPED"]
    assert row_color(window, 0) == "#c6efce" and row_color(window, 1) == "#ffeb9c"
    assert runs(paths, "S002")[0]["verdict"] == "INCOMPLETE"


def test_frames_are_cleared_after_a_run_so_a_scan_cannot_reuse_them(
    window: MainWindow, paths: dict[str, Path]
) -> None:
    tab = window.test_tab
    tab.model_combo.setCurrentText("M")
    tab.set_frame("normal", paths["good"])
    run_and_wait(window, "FIRST")
    assert tab.frame_labels["normal"].text() == "not selected" and tab.frame_labels["normal"].toolTip() == ""
    tab.serial_edit.setText("SECOND")
    QTest.keyClick(tab.serial_edit, Qt.Key.Key_Return)  # barcode scan + Enter
    assert not tab.is_running and "Choose at least one frame" in tab.message_label.text()
    assert runs(paths, "SECOND") == [] and len(runs(paths, "FIRST")) == 1


def test_frames_are_kept_when_the_pipeline_itself_fails(
    window: MainWindow, paths: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(main_window, "run_pipeline", lambda *a: (_ for _ in ()).throw(RuntimeError("x")))
    tab = window.test_tab
    tab.model_combo.setCurrentText("M")
    tab.set_frame("normal", paths["good"])
    run_and_wait(window)
    assert tab.frame_labels["normal"].text() == "good.png"  # nothing was consumed, so retry is possible


def test_retest_of_same_serial_creates_two_runs(window: MainWindow, paths: dict[str, Path]) -> None:
    tab = window.test_tab
    tab.model_combo.setCurrentText("M")
    tab.set_frame("normal", paths["dark"])
    run_and_wait(window, "SN9")
    tab.set_frame("normal", paths["good"])
    run_and_wait(window, "SN9")
    assert [r["verdict"] for r in runs(paths, "SN9")] == ["PASS", "FAIL"]


def test_enter_in_serial_field_starts_a_run(window: MainWindow, paths: dict[str, Path]) -> None:
    tab = window.test_tab
    tab.model_combo.setCurrentText("M")
    tab.set_frame("normal", paths["good"])
    tab.serial_edit.setText("SCAN1")
    QTest.keyClick(tab.serial_edit, Qt.Key.Key_Return)
    assert tab.is_running
    wait_for(lambda: tab.run_button.isEnabled())
    assert len(runs(paths, "SCAN1")) == 1


def test_blank_serial_or_missing_frames_do_not_start(window: MainWindow, paths: dict[str, Path]) -> None:
    tab = window.test_tab
    click(tab.run_button)  # nothing entered: the serial is checked first
    assert not tab.is_running and "serial" in tab.message_label.text()
    tab.serial_edit.setText("S1")
    click(tab.run_button)
    assert not tab.is_running and "Choose at least one frame" in tab.message_label.text()
    tab.set_frame("normal", paths["good"])
    tab.serial_edit.setText("   ")
    click(tab.run_button)
    assert not tab.is_running and "serial" in tab.message_label.text()
    assert tab.run_button.isEnabled() and not paths["db"].exists()


def test_ui_stays_responsive_while_pipeline_runs(
    window: MainWindow, paths: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    real = main_window.run_pipeline

    def slow(model: str, frames: Any, config: Any) -> Any:
        time.sleep(0.5)
        return real(model, frames, config)

    monkeypatch.setattr(main_window, "run_pipeline", slow)
    tab = window.test_tab
    tab.model_combo.setCurrentText("M")
    tab.set_frame("normal", paths["good"])
    tab.serial_edit.setText("SLOW")
    ticks: list[int] = []
    timer = QTimer()
    timer.timeout.connect(lambda: ticks.append(1))
    timer.start(20)
    click(tab.run_button)
    assert not tab.run_button.isEnabled()
    click(tab.run_button)  # second click while running is ignored
    wait_for(lambda: tab.run_button.isEnabled())
    timer.stop()
    assert len(ticks) >= 8, f"event loop was blocked, only {len(ticks)} ticks"
    assert len(runs(paths, "SLOW")) == 1


def test_pipeline_exception_is_reported_and_run_reenabled(
    window: MainWindow, paths: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("kaboom")

    monkeypatch.setattr(main_window, "run_pipeline", boom)
    tab = window.test_tab
    tab.model_combo.setCurrentText("M")
    tab.set_frame("normal", paths["good"])
    run_and_wait(window)
    assert tab.verdict_label.text() == "ERROR"
    assert "Pipeline failed: RuntimeError: kaboom" in tab.message_label.text()
    assert tab.table.rowCount() == 0 and runs(paths, "S001") == []


def test_database_failure_still_shows_the_result(qapp: QApplication, paths: dict[str, Path], tmp_path: Path) -> None:
    win = MainWindow(paths["config"], tmp_path)  # a directory cannot be opened as a database
    win.show()
    tab = win.test_tab
    tab.model_combo.setCurrentText("M")
    tab.set_frame("normal", paths["good"])
    run_and_wait(win)
    assert tab.verdict_label.text() == "PASS" and tab.table.rowCount() == 1
    assert "Result NOT saved" in tab.message_label.text()
    win.close()


def test_result_rows_and_colors() -> None:
    from camqc.checks.base import make_result
    from camqc.pipeline import CheckOutcome, RunResult

    ok = CheckOutcome("a", "PASS", make_result("a", 1234.5678, "u", None, 5), None, "flat", 1.0)
    skipped = CheckOutcome("b", "SKIPPED", None, "required frame missing", None, 0.1)
    rows = main_window.result_rows(RunResult("INCOMPLETE", (ok, skipped), (), 2.0))
    assert rows[0] == (["a", "1234.6", "u", "<= 5", "PASS"], "PASS", "")
    assert rows[1] == (["b", "-", "-", "-", "SKIPPED"], "SKIPPED", "required frame missing")
    assert main_window.colors_for("PASS")[0] == "#c6efce" and main_window.colors_for("FAIL")[0] == "#ffc7ce"
    assert main_window.colors_for("-") == main_window.NEUTRAL_COLORS
