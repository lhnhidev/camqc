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
from camqc.gui.main_window import ScaledImageLabel  # noqa: E402

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
