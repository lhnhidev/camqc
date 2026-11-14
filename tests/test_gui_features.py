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

from gui_support import click, png, wait_for  # noqa: E402
from PySide6.QtCore import Qt, QTimer  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QApplication, QFileDialog  # noqa: E402

from camqc import db  # noqa: E402
from camqc.gui import viewer, workers  # noqa: E402
from camqc.gui.main_window import MainWindow, TestTab  # noqa: E402

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


# --- overlays and row highlighting -----------------------------------------------------


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


# --- webcam ----------------------------------------------------------------------------


def test_webcam_capture_becomes_the_normal_frame(
    window: MainWindow, paths: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    shot = np.full((48, 64, 3), 90, np.uint8)
    monkeypatch.setattr(workers, "capture_frame", lambda: (time.sleep(0.1), shot)[1])
    tab = window.test_tab
    click(tab.capture_button)
    assert tab.is_running and not tab.capture_button.isEnabled() and not tab.run_button.isEnabled()
    wait_for(lambda: tab.capture_button.isEnabled() and not tab.is_running)
    assert tab.frame_labels["normal"].text().startswith("webcam ")
    assert tab.frame_labels["normal"].toolTip() == "Captured from the webcam"
    assert np.array_equal(tab._frames["normal"], shot)
    assert "(64x48)" in tab.caption.text() and tab.run_button.isEnabled()
    tab.model_combo.setCurrentText("M")  # the captured frame is usable like a file
    tab.serial_edit.setText("CAMSHOT")
    click(tab.run_button)
    wait_for(lambda: tab.run_button.isEnabled())
    assert runs(paths, "CAMSHOT")[0]["verdict"] == "FAIL"  # 90 is darker than mean_low 100


def test_missing_camera_shows_a_friendly_error(window: MainWindow, monkeypatch: pytest.MonkeyPatch) -> None:
    class NoCamera:
        def __init__(self, index: int) -> None: ...
        def isOpened(self) -> bool: return False  # noqa: N802, E704
        def release(self) -> None: ...

    monkeypatch.setattr(workers.cv2, "VideoCapture", NoCamera)  # the real capture_frame, no real camera
    tab = window.test_tab
    click(tab.capture_button)
    wait_for(lambda: tab.capture_button.isEnabled() and not tab.is_running)
    assert "No camera found" in tab.message_label.text()
    assert "#c00000" in tab.message_label.styleSheet()  # shown as an error
    assert "Traceback" not in tab.message_label.text() and "normal" not in tab._frames
    assert tab.run_button.isEnabled()


def test_unexpected_camera_failure_is_reported_not_raised(window: MainWindow, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(workers, "capture_frame", lambda: (_ for _ in ()).throw(RuntimeError("driver crashed")))
    tab = window.test_tab
    click(tab.capture_button)
    wait_for(lambda: tab.capture_button.isEnabled() and not tab.is_running)
    assert "Camera error: RuntimeError: driver crashed" in tab.message_label.text()


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


def test_ui_stays_responsive_during_a_slow_capture(window: MainWindow, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(workers, "capture_frame", lambda: (time.sleep(0.5), np.zeros((8, 8, 3), np.uint8))[1])
    tab = window.test_tab
    ticks: list[int] = []
    timer = QTimer()
    timer.timeout.connect(lambda: ticks.append(1))
    timer.start(20)
    click(tab.capture_button)
    tab.capture_from_webcam()  # ignored while a capture is running
    wait_for(lambda: tab.capture_button.isEnabled() and not tab.is_running)
    timer.stop()
    assert len(ticks) >= 8, f"event loop was blocked, only {len(ticks)} ticks"


# --- folder mode -----------------------------------------------------------------------


@pytest.fixture
def folder(tmp_path: Path) -> Path:
    root = tmp_path / "imgs"
    root.mkdir()
    png(root / "S001.png", 128)
    png(root / "S002.png", 20)
    (root / "bad.png").write_bytes(b"garbage")
    png(root / "S003.png", 130)
    (root / "notes.txt").write_text("ignored", encoding="utf-8")
    return root


def start_folder(win: MainWindow, folder: Path, monkeypatch: pytest.MonkeyPatch, model: str = "M") -> TestTab:
    monkeypatch.setattr(QFileDialog, "getExistingDirectory", lambda *a, **k: str(folder))
    tab = win.test_tab
    tab.model_combo.setCurrentText(model)
    click(tab.folder_button)
    return tab


def test_folder_mode_processes_every_image_with_a_progress_bar(
    window: MainWindow, paths: dict[str, Path], folder: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tab = window.test_tab
    assert not tab.progress_bar.isVisibleTo(tab) and not tab.cancel_button.isVisibleTo(tab)
    tab = start_folder(window, folder, monkeypatch)
    assert tab.is_running and not tab.run_button.isEnabled() and not tab.folder_button.isEnabled()
    assert tab.progress_bar.isVisibleTo(tab) and tab.progress_bar.maximum() == 4
    assert tab.cancel_button.isVisibleTo(tab)
    wait_for(lambda: tab.folder_button.isEnabled() and not tab.is_running)

    assert tab.progress_bar.value() == 4 and tab.progress_bar.maximum() == 4
    assert not tab.cancel_button.isVisibleTo(tab)
    assert "Folder done: 4 of 4 images - 2 PASS, 1 not PASS, 1 errors" in tab.message_label.text()
    conn = db.init_db(paths["db"])
    verdicts = {s: db.find_by_serial(conn, s)[0]["verdict"] for s in ("S001", "S002", "S003")}
    assert verdicts == {"S001": "PASS", "S002": "FAIL", "S003": "PASS"}  # serial = file name stem
    assert db.find_by_serial(conn, "bad") == [] and db.find_by_serial(conn, "notes") == []
    conn.close()
    assert tab.verdict_label.text() in {"PASS", "FAIL"} and tab.table.rowCount() == 1  # last result shown


def test_folder_items_are_shown_with_their_overlays(
    window: MainWindow, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "charts"
    root.mkdir()
    write(root / "U1.png", chart())
    tab = start_folder(window, root, monkeypatch, model="O")  # sharpness draws 5 rects on the normal frame
    wait_for(lambda: tab.folder_button.isEnabled() and not tab.is_running)
    assert len(tab.viewer.shapes) == 5 and tab._viewer_frame == "normal"
    click_row(tab, row_of(tab, "sharpness"))
    assert {s.color for s in tab.viewer.shapes} == {viewer.HIGHLIGHT_COLOR}


def test_folder_frame_type_comes_from_the_combo(
    window: MainWindow, paths: dict[str, Path], folder: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tab = window.test_tab
    tab.folder_frame_combo.setCurrentText("flat")  # config M needs a 'normal' frame -> INCOMPLETE
    start_folder(window, folder, monkeypatch)
    wait_for(lambda: tab.folder_button.isEnabled() and not tab.is_running)
    assert runs(paths, "S001")[0]["verdict"] == "INCOMPLETE"


def test_empty_or_missing_folder_does_not_start(window: MainWindow, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "empty").mkdir()
    tab = start_folder(window, tmp_path / "empty", monkeypatch)
    assert not tab.is_running and "no images found" in tab.message_label.text()
    assert not tab.progress_bar.isVisibleTo(tab) and tab.folder_button.isEnabled()
    monkeypatch.setattr(QFileDialog, "getExistingDirectory", lambda *a, **k: "")  # dialog cancelled
    click(tab.folder_button)
    assert not tab.is_running


def test_running_a_unit_hides_the_folder_progress(
    window: MainWindow, paths: dict[str, Path], folder: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tab = start_folder(window, folder, monkeypatch)
    wait_for(lambda: tab.folder_button.isEnabled() and not tab.is_running)
    assert tab.progress_bar.isVisibleTo(tab)
    tab.set_frame("normal", paths["good"])
    run_model(window, "M", "AFTER")
    assert not tab.progress_bar.isVisibleTo(tab)


@pytest.fixture
def slow_folder(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "many"
    root.mkdir()
    for i in range(10):
        png(root / f"U{i:02d}.png", 128)
    real = workers.run_pipeline
    monkeypatch.setattr(workers, "run_pipeline", lambda *a: (time.sleep(0.15), real(*a))[1])
    return root


def test_cancel_stops_the_folder_batch_early(
    window: MainWindow, paths: dict[str, Path], slow_folder: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tab = start_folder(window, slow_folder, monkeypatch)
    wait_for(lambda: tab.progress_bar.value() >= 1)
    click(tab.cancel_button)
    assert not tab.cancel_button.isEnabled()
    wait_for(lambda: tab.folder_button.isEnabled() and not tab.is_running)
    saved = [s for i in range(10) if (s := runs(paths, f"U{i:02d}"))]
    assert 1 <= len(saved) < 10
    assert "Folder cancelled" in tab.message_label.text() and not tab.cancel_button.isVisibleTo(tab)
    assert tab.progress_bar.value() < 10


def test_run_and_capture_are_ignored_while_a_folder_runs(
    window: MainWindow, slow_folder: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tab = start_folder(window, slow_folder, monkeypatch)
    tab.serial_edit.setText("X")
    tab.start_run()
    tab.capture_from_webcam()
    tab.start_folder(slow_folder)  # none of these may start a second job
    assert tab._worker is not None and type(tab._worker).__name__ == "FolderWorker"
    tab.cancel_folder()
    wait_for(lambda: not tab.is_running)


def test_closing_during_a_folder_batch_cancels_it_and_keeps_finished_runs(
    qapp: QApplication, paths: dict[str, Path], slow_folder: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    win = MainWindow(paths["config"], paths["db"])
    win.show()
    tab = start_folder(win, slow_folder, monkeypatch)
    wait_for(lambda: tab.progress_bar.value() >= 1)
    assert win.close()  # cancels the batch, waits for the image in progress, delivers its result
    assert not tab.is_running
    saved = [s for i in range(10) if (s := runs(paths, f"U{i:02d}"))]
    assert 1 <= len(saved) < 10


def test_progress_bar_widget_text_format(window: MainWindow) -> None:
    assert window.test_tab.progress_bar.format() == "%v / %m"
    assert window.test_tab.folder_frame_combo.count() == 4
    assert window.test_tab.capture_button.text() == "Capture from webcam"


# --- History / Statistics are wired to new runs ------------------------------------------


def test_tabs_are_the_real_history_and_statistics_tabs(window: MainWindow) -> None:
    from camqc.gui.history_tab import HistoryTab
    from camqc.gui.stats_tab import StatsTab

    assert isinstance(window.history_tab, HistoryTab) and isinstance(window.stats_tab, StatsTab)
    assert [window.tabs.tabText(i) for i in range(3)] == ["Test", "History", "Statistics"]
    assert window.history_tab.is_stale and window.stats_tab.is_stale  # nothing is queried until shown


def test_a_saved_run_refreshes_the_visible_statistics_and_the_history_when_opened(
    window: MainWindow, paths: dict[str, Path]
) -> None:
    window.tabs.setCurrentWidget(window.stats_tab)  # first show loads an empty database
    assert window.stats_tab.days == [] and window.stats_tab.summary_label.text() == "No runs"
    assert window.history_tab.is_stale and window.history_tab.proxy.rowCount() == 0

    window.tabs.setCurrentWidget(window.test_tab)
    window.test_tab.set_frame("normal", paths["good"])
    run_model(window, "M", "WIRED1")
    assert window.stats_tab.is_stale  # hidden while the run finished

    window.tabs.setCurrentWidget(window.stats_tab)  # showing it loads the new run
    assert window.stats_tab.days[0]["total"] == 1 and "1 runs - overall yield 100.0%" in window.stats_tab.summary_label.text()
    window.tabs.setCurrentWidget(window.history_tab)
    assert [window.history_tab.proxy.index(r, 2).data() for r in range(window.history_tab.proxy.rowCount())] == ["WIRED1"]


def test_statistics_updates_by_itself_while_it_is_shown(window: MainWindow, paths: dict[str, Path]) -> None:
    window.tabs.setCurrentWidget(window.stats_tab)
    window.test_tab.set_frame("normal", paths["good"])
    window.test_tab.model_combo.setCurrentText("M")
    window.test_tab.serial_edit.setText("LIVE1")
    window.test_tab.start_run()  # the Test tab is hidden, but the run still works
    wait_for(lambda: not window.test_tab.is_running)
    wait_for(lambda: window.stats_tab.days != [])  # the visible tab refreshed after the coalescing delay
    assert window.stats_tab.days[0]["passed"] == 1


def test_a_folder_batch_causes_few_reloads_of_the_visible_tabs(
    window: MainWindow, folder: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    reloads: list[int] = []
    real = window.stats_tab.reload
    monkeypatch.setattr(window.stats_tab, "reload", lambda: (reloads.append(1), real())[1])
    window.tabs.setCurrentWidget(window.stats_tab)
    reloads.clear()
    tab = start_folder(window, folder, monkeypatch)
    wait_for(lambda: tab.folder_button.isEnabled() and not tab.is_running)
    wait_for(lambda: window.stats_tab.days != [] and window.stats_tab.days[0]["total"] == 3)
    assert 1 <= len(reloads) <= 2  # 3 saved images, but the reloads were coalesced


# --- Export report -----------------------------------------------------------------------


def stub_save_dialog(monkeypatch: pytest.MonkeyPatch, answer: str) -> dict[str, str]:
    """Replace QFileDialog.getSaveFileName; returns what the dialog was asked."""
    asked: dict[str, str] = {}

    def fake(parent: object, caption: str, directory: str, file_filter: str) -> tuple[str, str]:
        asked.update(caption=caption, directory=directory, file_filter=file_filter)
        return answer, ""

    monkeypatch.setattr(QFileDialog, "getSaveFileName", fake)
    return asked


def saved_images(paths: dict[str, Path], run_id: int) -> dict[str, np.ndarray]:
    conn = db.init_db(paths["db"])
    try:
        return db.get_run_images(conn, run_id)
    finally:
        conn.close()


def test_report_filename_is_safe_and_descriptive() -> None:
    from camqc.gui.main_window import report_filename

    assert report_filename("SN-001", 7) == "report_SN-001_run7.html"
    assert report_filename("a/b c:d", 3) == "report_a_b_c_d_run3.html"
    assert report_filename("   ", 4) == "report_unit_run4.html"


def test_export_button_is_enabled_only_for_a_saved_run(window: MainWindow, paths: dict[str, Path]) -> None:
    tab = window.test_tab
    assert tab.export_button.text() == "Export report" and not tab.export_button.isEnabled()
    tab.set_frame("normal", paths["good"])
    run_model(window, "M", "S1")
    assert tab.export_button.isEnabled()
    tab.set_frame("normal", paths["good"])
    tab.serial_edit.setText("S2")
    tab.model_combo.setCurrentText("M")
    click(tab.run_button)
    assert tab.is_running and not tab.export_button.isEnabled()  # a new run is under way: nothing to export yet
    wait_for(lambda: tab.run_button.isEnabled() and not tab.is_running)
    assert tab.export_button.isEnabled()


def test_a_gui_run_stores_its_input_images(window: MainWindow, paths: dict[str, Path]) -> None:
    window.test_tab.set_frame("normal", paths["good"])
    run_model(window, "M", "S1")
    assert set(saved_images(paths, 1)) == {"normal"}


def test_export_writes_a_report_with_the_serial_and_the_image(
    window: MainWindow, paths: dict[str, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tab = window.test_tab
    tab.set_frame("normal", paths["good"])
    run_model(window, "M", "SN-GUI-1")
    out = tmp_path / "out" / "mine.html"
    asked = stub_save_dialog(monkeypatch, str(out))
    click(tab.export_button)
    assert asked["caption"] == "Export report" and asked["directory"].endswith("report_SN-GUI-1_run1.html")
    assert "HTML" in asked["file_filter"]
    text = out.read_text(encoding="utf-8")
    assert "SN-GUI-1" in text and "<td>brightness</td>" in text and "data:image/jpeg;base64," in text
    assert f"Report written: {out}" in tab.message_label.text() and "#c00000" not in tab.message_label.styleSheet()


def test_export_adds_the_html_extension_and_ignores_a_cancelled_dialog(
    window: MainWindow, paths: dict[str, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tab = window.test_tab
    tab.set_frame("normal", paths["good"])
    run_model(window, "M", "SN-GUI-2")
    stub_save_dialog(monkeypatch, "")  # the user cancels
    message = tab.message_label.text()
    click(tab.export_button)
    assert tab.message_label.text() == message and not list(tmp_path.glob("**/*.html"))
    stub_save_dialog(monkeypatch, str(tmp_path / "noext"))
    click(tab.export_button)
    assert "SN-GUI-2" in (tmp_path / "noext.html").read_text(encoding="utf-8")


def test_export_failure_is_shown_not_raised(window: MainWindow, paths: dict[str, Path], tmp_path: Path) -> None:
    tab = window.test_tab
    tab.set_frame("normal", paths["good"])
    run_model(window, "M", "S1")
    assert not tab.export_report_to(1, tmp_path)  # a directory cannot be written as a file
    assert "Report NOT written" in tab.message_label.text() and "#c00000" in tab.message_label.styleSheet()
    assert not tab.export_report_to(999, tmp_path / "x.html")
    assert "run #999 not found" in tab.message_label.text()


def test_nothing_to_export_if_the_run_could_not_be_saved(qapp: QApplication, paths: dict[str, Path], tmp_path: Path) -> None:
    win = MainWindow(paths["config"], tmp_path)  # a directory cannot be opened as a database
    win.show()
    win.test_tab.set_frame("normal", paths["good"])
    run_model(win, "M", "S1")
    assert "Result NOT saved" in win.test_tab.message_label.text()
    assert not win.test_tab.export_button.isEnabled()
    win.close()


def test_after_a_folder_batch_the_report_is_for_the_last_saved_image(
    window: MainWindow, paths: dict[str, Path], folder: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tab = start_folder(window, folder, monkeypatch)  # S001, S002, S003 saved; bad.png fails to load
    wait_for(lambda: tab.folder_button.isEnabled() and not tab.is_running)
    assert tab.export_button.isEnabled()
    out = tmp_path / "last.html"
    stub_save_dialog(monkeypatch, str(out))
    click(tab.export_button)
    text = out.read_text(encoding="utf-8")
    assert "S003" in text and "S001" not in text and "data:image/jpeg;base64," in text


def test_a_failed_pipeline_does_not_leave_the_previous_report_exportable(
    window: MainWindow, paths: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    tab = window.test_tab
    tab.set_frame("normal", paths["good"])
    run_model(window, "M", "S1")
    assert tab.export_button.isEnabled()
    monkeypatch.setattr(workers, "run_pipeline", lambda *a: (_ for _ in ()).throw(RuntimeError("boom")))
    tab.set_frame("normal", paths["good"])
    run_model(window, "M", "S2")
    assert tab.verdict_label.text() == "ERROR" and tab._last_run_id is None
    assert not tab.export_button.isEnabled()  # it would have exported S1 while the screen shows an error


def test_starting_a_folder_batch_clears_the_exportable_run(
    window: MainWindow, paths: dict[str, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tab = window.test_tab
    tab.set_frame("normal", paths["good"])
    run_model(window, "M", "S1")
    assert tab._last_run_id == 1
    only_bad = tmp_path / "bad_only"
    only_bad.mkdir()
    (only_bad / "x.png").write_bytes(b"garbage")  # every image of the batch fails to load
    start_folder(window, only_bad, monkeypatch)
    wait_for(lambda: tab.folder_button.isEnabled() and not tab.is_running)
    assert tab._last_run_id is None and not tab.export_button.isEnabled()


def test_unexpected_report_errors_are_shown_not_raised(
    window: MainWindow, paths: dict[str, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tab = window.test_tab
    tab.set_frame("normal", paths["good"])
    run_model(window, "M", "S1")
    from camqc import report

    for error in (ValueError("cannot encode the image"), KeyError("x")):
        monkeypatch.setattr(report, "export_html", lambda *a, _e=error: (_ for _ in ()).throw(_e))
        assert not tab.export_report_to(1, tmp_path / "x.html")
        assert "Report NOT written" in tab.message_label.text()
