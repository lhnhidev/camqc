"""Window-level tests: overlays + row highlighting, webcam capture, and folder mode."""

from __future__ import annotations

import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import Qt, QTimer  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QApplication, QFileDialog  # noqa: E402

from camqc import db  # noqa: E402
from camqc.gui import viewer, workers  # noqa: E402
from camqc.gui.main_window import MainWindow, TestTab  # noqa: E402
from gui_support import click, png, wait_for  # noqa: E402

CONFIG = """
models:
  M:
    checks:
      brightness:
        frame: normal
        params: {mean_low: 100, mean_high: 160}
  O:
    checks:
      sharpness: {frame: normal, params: {min_center: 1}}
      shading: {frame: flat, params: {min_corner_pct: 50}}
      brightness: {frame: [normal, flat], params: {}}
  P:
    checks:
      sharpness: {frame: flat, params: {}}
      shading: {frame: flat, params: {min_corner_pct: 50}}
  N:
    checks:
      shading: {frame: flat, params: {min_corner_pct: 50}}
      pixels_dead: {check: pixels, frame: white, params: {mode: dead, max_defects: 5}}
"""


def chart(size: int = 120) -> np.ndarray:
    y, x = np.mgrid[0:size, 0:size]
    gray = np.where((x // 10 + y // 10) % 2 == 0, 30, 220).astype(np.uint8)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def write(path: Path, image: np.ndarray) -> Path:
    assert cv2.imwrite(str(path), image)
    return path


@pytest.fixture
def paths(tmp_path: Path) -> dict[str, Path]:
    config = tmp_path / "models.yaml"
    config.write_text(CONFIG, encoding="utf-8")
    return {
        "config": config,
        "db": tmp_path / "t.db",
        "chart": write(tmp_path / "chart.png", chart()),
        "flat": png(tmp_path / "flat.png", 128, (120, 120)),
        "good": png(tmp_path / "good.png", 128),
        "dark": png(tmp_path / "dark.png", 20),
    }


@pytest.fixture
def window(qapp: QApplication, paths: dict[str, Path]) -> Iterator[MainWindow]:
    win = MainWindow(paths["config"], paths["db"])
    win.resize(1400, 900)
    win.show()
    yield win
    win.close()
    QApplication.processEvents()


def runs(paths: dict[str, Path], serial: str) -> list[dict[str, Any]]:
    conn = db.init_db(paths["db"])
    try:
        return db.find_by_serial(conn, serial)
    finally:
        conn.close()


def run_model(win: MainWindow, model: str, serial: str = "S1") -> TestTab:
    tab = win.test_tab
    tab.model_combo.setCurrentText(model)
    tab.serial_edit.setText(serial)
    click(tab.run_button)
    wait_for(lambda: tab.run_button.isEnabled() and not tab.is_running)
    return tab


def click_row(tab: TestTab, row: int) -> None:
    """Click the first cell of a results row with the mouse."""
    rect = tab.table.visualItemRect(tab.table.item(row, 0))
    QTest.mouseClick(tab.table.viewport(), Qt.MouseButton.LeftButton, pos=rect.center())


def row_of(tab: TestTab, name: str) -> int:
    return next(r for r in range(tab.table.rowCount()) if tab.table.item(r, 0).text() == name)


def overlay_of(tab: TestTab, name: str) -> list[dict[str, Any]]:
    outcome = next(o for o in tab._last_result.results if o.name == name)  # type: ignore[union-attr]
    return outcome.result.overlay  # type: ignore[union-attr]


def test_run_draws_every_overlay_of_the_shown_frame(window: MainWindow, paths: dict[str, Path]) -> None:
    tab = window.test_tab
    tab.set_frame("normal", paths["chart"])
    tab.set_frame("flat", paths["flat"])  # the viewer shows the last chosen frame
    run_model(window, "O")
    shapes = tab.viewer.shapes
    assert [s.shape for s in shapes] == overlay_of(tab, "shading")  # only shading used the flat frame
    assert len(shapes) == 5 and {s.width for s in shapes} == {viewer.THIN_WIDTH}
    assert "Flat frame - 5 overlay shape(s)" in tab.caption.text()


def test_clicking_a_row_highlights_only_that_checks_overlay(window: MainWindow, paths: dict[str, Path]) -> None:
    tab = window.test_tab
    tab.set_frame("flat", paths["flat"])
    run_model(window, "P")  # sharpness and shading both draw 5 rects on the flat frame
    sharp, shade = overlay_of(tab, "sharpness"), overlay_of(tab, "shading")
    assert len(tab.viewer.shapes) == 10 and sharp != shade

    click_row(tab, row_of(tab, "sharpness"))
    shapes = tab.viewer.shapes
    assert [s.shape for s in shapes] == sharp
    assert {(s.color, s.width) for s in shapes} == {(viewer.HIGHLIGHT_COLOR, viewer.HIGHLIGHT_WIDTH)}
    assert "sharpness: 5 overlay shape(s)" in tab.caption.text()

    click_row(tab, row_of(tab, "shading"))
    assert [s.shape for s in tab.viewer.shapes] == shade  # switched, not accumulated

    tab.table.clearSelection()
    assert len(tab.viewer.shapes) == 10 and {s.width for s in tab.viewer.shapes} == {viewer.THIN_WIDTH}


def test_clicking_a_row_switches_to_the_frame_that_check_used(window: MainWindow, paths: dict[str, Path]) -> None:
    tab = window.test_tab
    tab.set_frame("normal", paths["chart"])
    tab.set_frame("flat", paths["flat"])
    run_model(window, "O")
    assert tab._viewer_frame == "flat"
    click_row(tab, row_of(tab, "sharpness"))  # sharpness used the chart (normal frame)
    assert tab._viewer_frame == "normal" and [s.shape for s in tab.viewer.shapes] == overlay_of(tab, "sharpness")
    assert tab.viewer.pixmap().width() > 0 and "Normal frame" in tab.caption.text()
    click_row(tab, row_of(tab, "brightness"))  # brightness has no overlay
    assert tab.viewer.shapes == [] and "0 overlay shape(s)" in tab.caption.text()


def test_overlay_is_painted_at_the_scaled_position(window: MainWindow, paths: dict[str, Path]) -> None:
    tab = window.test_tab
    tab.set_frame("flat", paths["flat"])
    run_model(window, "P")
    click_row(tab, row_of(tab, "shading"))
    rect = overlay_of(tab, "shading")[0]  # the center patch, in image coordinates
    pixmap = tab.viewer.pixmap()
    scale = pixmap.width() / 120
    x, y = int((rect["x"] + rect["w"] / 2) * scale), int(rect["y"] * scale)  # middle of its top edge
    c = pixmap.toImage().pixelColor(x, y)
    assert c.red() > 200 and c.green() > 160 and c.blue() < 120, c.name()  # highlight yellow on a gray image


def test_skipped_check_has_nothing_to_draw(window: MainWindow, paths: dict[str, Path]) -> None:
    tab = window.test_tab
    tab.set_frame("flat", paths["flat"])
    run_model(window, "N")  # pixels_dead needs a white frame we do not have
    click_row(tab, row_of(tab, "pixels_dead"))
    assert tab.viewer.shapes == [] and "SKIPPED - nothing to draw" in tab.caption.text()


def test_choosing_a_new_frame_hides_old_overlays_but_the_table_still_reviews_the_old_run(
    window: MainWindow, paths: dict[str, Path]
) -> None:
    tab = window.test_tab
    tab.set_frame("flat", paths["flat"])
    run_model(window, "P")
    assert tab.viewer.shapes
    tab.set_frame("flat", paths["good"])  # a new, not yet tested image
    assert tab.viewer.shapes == [] and "good.png" in tab.caption.text()
    click_row(tab, row_of(tab, "shading"))  # the table belongs to the previous run: review that image
    assert len(tab.viewer.shapes) == 5


def test_frame_buttons_are_disabled_while_a_job_runs(
    window: MainWindow, paths: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    real = workers.run_pipeline
    monkeypatch.setattr(workers, "run_pipeline", lambda *a: (time.sleep(0.3), real(*a))[1])
    tab = window.test_tab
    assert all(b.isEnabled() for b in tab.frame_buttons.values())
    tab.set_frame("normal", paths["good"])
    tab.model_combo.setCurrentText("M")
    tab.serial_edit.setText("BUSY")
    click(tab.run_button)
    assert tab.is_running and not any(b.isEnabled() for b in tab.frame_buttons.values())
    wait_for(lambda: tab.run_button.isEnabled() and not tab.is_running)
    assert all(b.isEnabled() for b in tab.frame_buttons.values())


def test_clicking_a_skipped_row_keeps_the_caption_of_a_newly_chosen_frame(
    window: MainWindow, paths: dict[str, Path]
) -> None:
    tab = window.test_tab
    tab.set_frame("flat", paths["flat"])
    run_model(window, "N")
    tab.set_frame("flat", paths["good"])  # not yet tested
    click_row(tab, row_of(tab, "pixels_dead"))
    assert "good.png" in tab.caption.text() and "nothing to draw" not in tab.caption.text()
