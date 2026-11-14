"""Main window: image source selection, live preview, run checks, and per-check PASS/FAIL display.

A ``QMainWindow`` with three tabs: *Test* (this module), *History* and *Statistics* (see
:mod:`camqc.gui.history_tab` and :mod:`camqc.gui.stats_tab`), which refresh when a run is
saved. Anything slow (pipeline, webcam capture, folder batches) runs in a ``QThread``
worker (see :mod:`camqc.gui.workers`) so the UI never freezes; runs are saved to SQLite on the
GUI thread because sqlite3 connections must not cross threads.
"""

from __future__ import annotations

import logging
import re
import sqlite3
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
from PySide6.QtCore import (
    QCoreApplication,
    QObject,
    Qt,
    QThread,
    Signal,
    SignalInstance,
    Slot,
)
from PySide6.QtGui import QBrush, QCloseEvent, QColor, QFont, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QProgressBar,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from camqc import db, report
from camqc.cli import (
    DEFAULT_DB,
    DEFAULT_LOG,
    CliError,
    default_config_path,
    format_limits,
    list_images,
    load_image,
    setup_logging,
    teardown_logging,
)
from camqc.gui.common import NEUTRAL_COLORS, colors_for  # noqa: F401 - re-exported
from camqc.gui.history_tab import HistoryTab
from camqc.gui.stats_tab import StatsTab
from camqc.gui.viewer import ScaledImageLabel, bgr_to_pixmap, overlay_shapes
from camqc.gui.workers import CaptureWorker, FolderWorker, PipelineWorker
from camqc.pipeline import (
    PASS,
    Config,
    ConfigError,
    RunResult,
    load_config,
)

logger = logging.getLogger(__name__)

IMAGE_FILTER = "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff);;All files (*)"
DEFAULT_STATION = "UNKNOWN"
FRAME_BUTTON_ORDER = ("normal", "dark", "white", "flat")  # flat last: only some checks need it

SHUTDOWN_WAIT_MS = 30_000  # how long closing the window waits for a running job
TABLE_HEADERS = ("Check", "Value", "Unit", "Limits", "Status")


def report_filename(serial: str, run_id: int) -> str:
    """Return a safe default file name such as ``report_SN-001_run7.html``."""
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", serial).strip("_") or "unit"
    return f"report_{safe}_run{run_id}.html"


# --- helpers ---------------------------------------------------------------------------


def result_rows(result: RunResult) -> list[tuple[list[str], str, str]]:
    """Return ``(cells, status, tooltip)`` per check for the results table."""
    rows = []
    for o in result.results:
        res = o.result
        note = o.reason or (res.details.get("reason", "") if res else "")
        cells = [
            o.name,
            f"{res.value:.5g}" if res else "-",
            res.unit if res else "-",
            format_limits(res.low, res.high) if res else "-",
            o.status,
        ]
        rows.append((cells, o.status, note))
    return rows


@dataclass
class FolderRun:
    """Settings and running totals of the folder batch in progress."""

    model: str
    station: str
    total: int
    passed: int = 0
    not_passed: int = 0
    errors: int = 0


# --- Test tab --------------------------------------------------------------------------


class TestTab(QWidget):
    """The *Test* tab: choose model/serial/frames, run, and see per-check results with overlays."""

    __test__ = False  # not a pytest test class despite the name

    run_saved = Signal(int)  # emitted with the new run id after each run is stored

    def __init__(self, config: Config | None, db_path: str | Path, config_error: str = "") -> None:
        super().__init__()
        self._config = config
        self._db_path = db_path
        self._frames: dict[str, np.ndarray] = {}
        self._pixmaps: dict[str, QPixmap] = {}  # pixmaps of the chosen frames
        self._frame_paths: dict[str, Path] = {}
        self._last_dir = ""
        self._thread: QThread | None = None
        self._worker: QObject | None = None
        self._pending: tuple[str, str, str] | None = None  # (model, serial, station) of the active run
        self._run_pixmaps: dict[str, QPixmap] = {}  # snapshot taken when a run starts
        self._run_frames: dict[str, np.ndarray] = {}  # the run's input images, stored with the run
        self._last_run_id: int | None = None  # the saved run whose result is shown (for the report)
        self._last_serial = ""
        self._review_pixmaps: dict[str, QPixmap] = {}  # images that belong to the shown result
        self._last_result: RunResult | None = None
        self._viewer_frame: str | None = None  # frame type shown in the viewer
        self._viewer_shows_review = False  # True if the viewer shows an image of the shown result
        self._folder: FolderRun | None = None
        self._build_ui()
        if config_error:
            self.show_message(config_error, error=True)
            self._set_busy(False)

    # -- construction --

    def _build_ui(self) -> None:
        """Create the widgets and lay them out."""
        self.model_combo = QComboBox(objectName="model_combo")
        if self._config:
            self.model_combo.addItems(list(self._config.models))
        self.serial_edit = QLineEdit(objectName="serial_edit", placeholderText="scan or type serial")
        self.station_edit = QLineEdit(DEFAULT_STATION, objectName="station_edit")
        self.serial_edit.returnPressed.connect(self.start_run)  # barcode scanners send Enter

        unit_box = QGroupBox("Unit")
        form = QFormLayout(unit_box)
        form.addRow("Model", self.model_combo)
        form.addRow("Serial", self.serial_edit)
        form.addRow("Station", self.station_edit)

        frames_box = QGroupBox("Frames")
        grid = QGridLayout(frames_box)
        self.frame_buttons: dict[str, QPushButton] = {}
        self.frame_labels: dict[str, QLabel] = {}
        for row, frame in enumerate(FRAME_BUTTON_ORDER):
            button = QPushButton(f"{frame.capitalize()}...", objectName=f"choose_{frame}")
            button.clicked.connect(lambda _checked=False, f=frame: self.choose_frame(f))
            label = QLabel("not selected", objectName=f"path_{frame}")
            label.setStyleSheet("color: #808080;")
            grid.addWidget(button, row, 0)
            grid.addWidget(label, row, 1)
            self.frame_buttons[frame], self.frame_labels[frame] = button, label
        self.capture_button = QPushButton("Capture from webcam", objectName="capture_button")
        self.capture_button.clicked.connect(self.capture_from_webcam)
        grid.addWidget(self.capture_button, len(FRAME_BUTTON_ORDER), 0, 1, 2)
        self.folder_button = QPushButton("Load folder...", objectName="folder_button")
        self.folder_button.clicked.connect(self.choose_folder)
        self.folder_frame_combo = QComboBox(objectName="folder_frame_combo")
        self.folder_frame_combo.addItems(list(FRAME_BUTTON_ORDER))
        self.folder_frame_combo.setToolTip("Frame type given to every image in the folder")
        folder_row = QHBoxLayout()
        folder_row.addWidget(QLabel("as frame:"))
        folder_row.addWidget(self.folder_frame_combo, 1)
        grid.addWidget(self.folder_button, len(FRAME_BUTTON_ORDER) + 1, 0)
        grid.addLayout(folder_row, len(FRAME_BUTTON_ORDER) + 1, 1)
        grid.setColumnStretch(1, 1)

        self.run_button = QPushButton("Run", objectName="run_button")
        self.run_button.setMinimumHeight(60)
        self.run_button.setMinimumWidth(120)
        self.run_button.setFont(self._font(16, bold=True))
        self.run_button.clicked.connect(self.start_run)

        top = QHBoxLayout()
        top.addWidget(unit_box, 2)
        top.addWidget(frames_box, 3)
        top.addWidget(self.run_button, 0)

        self.viewer = ScaledImageLabel("No image selected")
        self.viewer.setObjectName("viewer")
        self.caption = QLabel("", objectName="caption")

        self.verdict_label = QLabel("-", objectName="verdict_label")
        self.verdict_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.verdict_label.setFont(self._font(40, bold=True))
        self.verdict_label.setMinimumHeight(90)
        self._style_verdict("-")

        self.table = QTableWidget(0, len(TABLE_HEADERS), objectName="result_table")
        self.table.setHorizontalHeaderLabels(list(TABLE_HEADERS))
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.itemSelectionChanged.connect(self._on_selection_changed)

        self.progress_bar = QProgressBar(objectName="progress_bar")
        self.progress_bar.setFormat("%v / %m")
        self.cancel_button = QPushButton("Cancel", objectName="cancel_button")
        self.cancel_button.clicked.connect(self.cancel_folder)
        progress_row = QHBoxLayout()
        progress_row.addWidget(self.progress_bar, 1)
        progress_row.addWidget(self.cancel_button)
        self.progress_bar.setVisible(False)
        self.cancel_button.setVisible(False)

        self.export_button = QPushButton("Export report", objectName="export_button")
        self.export_button.setToolTip("Save an HTML report (results, input images with overlays) of the shown run")
        self.export_button.setEnabled(False)
        self.export_button.clicked.connect(self.export_report)
        export_row = QHBoxLayout()
        export_row.addStretch(1)
        export_row.addWidget(self.export_button)

        self.message_label = QLabel("", objectName="message_label", wordWrap=True)

        left = QVBoxLayout()
        left.addWidget(self.viewer, 1)
        left.addWidget(self.caption)
        right = QVBoxLayout()
        right.addWidget(self.verdict_label)
        right.addWidget(self.table, 1)
        right.addLayout(export_row)
        right.addLayout(progress_row)
        right.addWidget(self.message_label)
        body = QHBoxLayout()
        body.addLayout(left, 1)
        body.addLayout(right, 1)

        layout = QVBoxLayout(self)
        layout.addLayout(top)
        layout.addLayout(body, 1)

    @staticmethod
    def _font(size: int, bold: bool = False) -> QFont:
        """Return a font of the given point size."""
        font = QFont()
        font.setPointSize(size)
        font.setBold(bold)
        return font

    # -- state --

    @property
    def is_running(self) -> bool:
        """True while a background job (run, capture or folder batch) is active."""
        return self._thread is not None

    def show_message(self, text: str, error: bool = False) -> None:
        """Show a non-blocking message under the results table."""
        self.message_label.setStyleSheet(f"color: {'#c00000' if error else '#404040'};")
        self.message_label.setText(text)

    def _style_verdict(self, verdict: str) -> None:
        bg, fg = colors_for(verdict)
        self.verdict_label.setText(verdict)
        self.verdict_label.setStyleSheet(
            f"background: {bg}; color: {fg}; border-radius: 8px; padding: 8px;"
        )

    def _set_busy(self, busy: bool) -> None:
        """Enable or disable the buttons that start jobs; show Cancel during folder batches.

        The frame buttons are disabled too: a frame chosen mid-run would never be tested.
        """
        enabled = not busy and self._config is not None
        for button in (self.run_button, self.capture_button, self.folder_button):
            button.setEnabled(enabled)
        for button in self.frame_buttons.values():
            button.setEnabled(not busy)
        self.export_button.setEnabled(not busy and self._last_run_id is not None)
        folder_active = busy and isinstance(self._worker, FolderWorker)
        self.cancel_button.setVisible(folder_active)
        self.cancel_button.setEnabled(folder_active)

    def _launch(self, worker: QObject, *end_signals: SignalInstance) -> None:
        """Run ``worker`` in a new thread that quits when any of ``end_signals`` fires.

        Result slots must be connected to the signals *before* calling this, so that they
        run before the thread is told to quit.
        """
        thread = QThread(self)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)  # type: ignore[attr-defined]
        for signal in end_signals:
            signal.connect(thread.quit)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(self._on_thread_done)
        self._thread, self._worker = thread, worker
        self._set_busy(True)
        thread.start()

    @Slot()
    def _on_thread_done(self) -> None:
        """Release the finished thread and re-enable the buttons."""
        thread, self._thread, self._worker = self._thread, None, None
        if thread is not None:
            thread.deleteLater()
        self._pending = None
        self._folder = None
        self._set_busy(False)

    # -- frames --

    @Slot(str)
    def choose_frame(self, frame: str) -> None:
        """Ask for an image file for ``frame`` and load it."""
        path, _ = QFileDialog.getOpenFileName(
            self, f"Choose {frame} frame", self._last_dir, IMAGE_FILTER
        )
        if path:
            self.set_frame(frame, path)

    def set_frame(self, frame: str, path: str | Path) -> bool:
        """Load ``path`` as the ``frame`` image and show it in the viewer.

        Returns:
            True on success; on failure the previous frame is kept and an error
            message is shown.
        """
        try:
            image = load_image(path)
        except CliError as exc:
            self.show_message(str(exc), error=True)
            return False
        path = Path(path)
        self._frame_paths[frame] = path
        self._last_dir = str(path.parent)
        self._install_frame(frame, image, path.name, str(path))
        return True

    def _install_frame(self, frame: str, image: np.ndarray, source: str, tooltip: str) -> None:
        """Store ``image`` as the ``frame`` input and show it (without overlays) in the viewer."""
        pixmap = bgr_to_pixmap(image)
        self._frames[frame], self._pixmaps[frame] = image, pixmap
        label = self.frame_labels[frame]
        label.setText(source)
        label.setToolTip(tooltip)
        label.setStyleSheet("")
        self.viewer.set_image(pixmap)
        self._viewer_frame, self._viewer_shows_review = frame, False
        h, w = image.shape[:2]
        self.caption.setText(f"{frame.capitalize()}: {source}  ({w}x{h})")
        self.show_message("")

    def _clear_frames(self) -> None:
        """Forget the chosen frames (the viewer keeps showing the last image for review)."""
        self._frames.clear()
        self._pixmaps.clear()
        self._frame_paths.clear()
        for label in self.frame_labels.values():
            label.setText("not selected")
            label.setToolTip("")
            label.setStyleSheet("color: #808080;")

    # -- webcam --

    @Slot()
    def capture_from_webcam(self) -> None:
        """Grab one webcam frame (in a worker thread) and use it as the Normal frame."""
        if self.is_running:
            return
        self.show_message("Capturing from webcam...")
        worker = CaptureWorker()
        worker.captured.connect(self._on_captured)
        worker.failed.connect(self._on_capture_failed)
        self._launch(worker, worker.captured, worker.failed)

    @Slot(object)
    def _on_captured(self, image: np.ndarray) -> None:
        """Use the captured frame as the Normal frame."""
        stamp = datetime.now().strftime("%H:%M:%S")
        self._install_frame("normal", image, f"webcam {stamp}", "Captured from the webcam")
        self.show_message("Captured a webcam frame as the Normal frame.")

    @Slot(str)
    def _on_capture_failed(self, message: str) -> None:
        """Show the (already friendly) camera error."""
        self.show_message(message, error=True)

    # -- running one unit --

    @Slot()
    def start_run(self) -> None:
        """Validate the inputs and start the pipeline in a worker thread."""
        if self.is_running or self._config is None:
            return
        serial = self.serial_edit.text().strip()
        if not serial:
            self.show_message("Enter or scan a serial number first.", error=True)
            self.serial_edit.setFocus()
            return
        if not self._frames:
            self.show_message("Choose at least one frame image first.", error=True)
            return
        model = self.model_combo.currentText()
        station = self.station_edit.text().strip() or DEFAULT_STATION
        self._pending = (model, serial, station)
        self._last_run_id = None  # a failed pipeline must not leave the previous unit's report exportable
        self._run_pixmaps = dict(self._pixmaps)
        self._run_frames = dict(self._frames)
        self.progress_bar.setVisible(False)
        self.show_message(f"Running {model} on {serial}...")
        self._style_verdict("RUNNING")

        worker = PipelineWorker(model, self._frames, self._config)
        worker.finished.connect(self._on_finished)
        worker.failed.connect(self._on_failed)
        self._launch(worker, worker.finished, worker.failed)

    def _save_run(
        self, model: str, serial: str, station: str, result: RunResult,
        frames: Mapping[str, np.ndarray] | None = None,
    ) -> int | None:
        """Save a run (and its input images) on the GUI thread; return its id, or ``None`` after showing the error."""
        self._last_run_id = None  # until this run is saved, there is nothing to report on
        try:
            conn = db.init_db(self._db_path)
            try:
                run_id = db.save_run(conn, db.RunInput(serial, model, station, datetime.now()), result, frames)
            finally:
                conn.close()
        except (sqlite3.Error, OSError, ValueError) as exc:
            logger.exception("could not save run")
            self.show_message(f"Result NOT saved: {exc}", error=True)
            return None
        self._last_run_id, self._last_serial = run_id, serial
        self.run_saved.emit(run_id)
        return run_id

    @Slot(object)
    def _on_finished(self, result: RunResult) -> None:
        """Show the result and save it to the database (on the GUI thread)."""
        self._show_run(result, self._run_pixmaps)
        self._clear_frames()  # a scan + Enter must never re-run the previous unit's images
        if self._pending is None:
            logger.error("run finished without pending run info; result not saved")
            return
        model, serial, station = self._pending
        run_id = self._save_run(model, serial, station, result, self._run_frames)
        if run_id is not None:
            self.show_message(f"Saved as run #{run_id} ({result.duration_ms:.0f} ms)")

    @Slot(str)
    def _on_failed(self, message: str) -> None:
        """Report an exception raised by the pipeline itself."""
        self._style_verdict("ERROR")
        self.table.setRowCount(0)
        self.show_message(f"Pipeline failed: {message}", error=True)

    # -- folder mode --

    @Slot()
    def choose_folder(self) -> None:
        """Ask for a folder and process every image in it."""
        folder = QFileDialog.getExistingDirectory(self, "Choose folder of images", self._last_dir)
        if folder:
            self.start_folder(folder)

    def start_folder(self, folder: str | Path) -> None:
        """Process every image in ``folder`` as its own unit (serial = file name stem).

        Each image is given the frame type chosen next to the *Load folder* button, runs
        through the pipeline in a worker thread, and is saved as it completes.
        """
        if self.is_running or self._config is None:
            return
        try:
            files = list_images(Path(folder))
        except CliError as exc:
            self.show_message(str(exc), error=True)
            return
        self._last_dir = str(folder)
        model = self.model_combo.currentText()
        station = self.station_edit.text().strip() or DEFAULT_STATION
        self._folder = FolderRun(model, station, len(files))
        self._last_run_id = None  # until an image of this batch is saved
        self.progress_bar.setRange(0, len(files))
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)
        self.show_message(f"Processing {len(files)} images from {folder}...")

        worker = FolderWorker(files, model, self.folder_frame_combo.currentText(), self._config)
        worker.progress.connect(self._on_folder_progress)
        worker.item_done.connect(self._on_folder_item)
        worker.item_failed.connect(self._on_folder_item_failed)
        worker.finished.connect(self._on_folder_finished)
        self._launch(worker, worker.finished)

    @Slot(int, int)
    def _on_folder_progress(self, done: int, total: int) -> None:
        """Advance the progress bar."""
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(done)

    @Slot(str, object, object)
    def _on_folder_item(self, name: str, result: RunResult, frames: Mapping[str, np.ndarray]) -> None:
        """Save one processed image and show its result."""
        folder = self._folder
        if folder is None:
            return
        pixmaps = {frame: bgr_to_pixmap(image) for frame, image in frames.items()}
        self._show_run(result, pixmaps)
        run_id = self._save_run(folder.model, Path(name).stem, folder.station, result, frames)
        if run_id is None:
            folder.errors += 1
            return
        if result.verdict == PASS:
            folder.passed += 1
        else:
            folder.not_passed += 1
        self.show_message(f"{name}: {result.verdict} (run #{run_id})")

    @Slot(str, str)
    def _on_folder_item_failed(self, name: str, message: str) -> None:
        """Count and show an image that could not be processed."""
        if self._folder is not None:
            self._folder.errors += 1
        self.show_message(f"{name}: {message}", error=True)

    @Slot(bool)
    def _on_folder_finished(self, cancelled: bool) -> None:
        """Summarize the batch."""
        folder = self._folder
        if folder is None:
            return
        done = folder.passed + folder.not_passed + folder.errors
        text = (f"Folder {'cancelled' if cancelled else 'done'}: {done} of {folder.total} images - "
                f"{folder.passed} PASS, {folder.not_passed} not PASS, {folder.errors} errors")
        self.show_message(text, error=folder.errors > 0)

    @Slot()
    def cancel_folder(self) -> None:
        """Stop the folder batch after the image being processed."""
        if isinstance(self._worker, FolderWorker):
            self._worker.cancel()
            self.cancel_button.setEnabled(False)
            self.show_message("Cancelling after the current image...")

    # -- report --

    @Slot()
    def export_report(self) -> None:
        """Ask where to save and write the HTML report of the shown (saved) run."""
        run_id = self._last_run_id
        if run_id is None or self.is_running:
            return
        default = Path(self._last_dir or ".") / report_filename(self._last_serial, run_id)
        path, _ = QFileDialog.getSaveFileName(self, "Export report", str(default), "HTML report (*.html)")
        if path:
            self.export_report_to(run_id, path if Path(path).suffix else path + ".html")

    def export_report_to(self, run_id: int, path: str | Path) -> bool:
        """Write the HTML report of ``run_id`` to ``path``; return False after showing the error."""
        try:
            conn = db.init_db(self._db_path)
            try:
                report.export_html(conn, run_id, path)
            finally:
                conn.close()
        except (report.RunNotFoundError, sqlite3.Error, OSError, ValueError, KeyError) as exc:
            logger.exception("could not write report for run %s", run_id)
            self.show_message(f"Report NOT written: {exc}", error=True)
            return False
        self._last_dir = str(Path(path).parent)
        self.show_message(f"Report written: {path}")
        return True

    # -- results and overlays --

    def _show_run(self, result: RunResult, pixmaps: Mapping[str, QPixmap]) -> None:
        """Show a result: table, verdict, and its images (with overlays) for review."""
        self._last_result = result
        self._review_pixmaps = dict(pixmaps)
        self.show_result(result)
        frame = self._viewer_frame if self._viewer_frame in self._review_pixmaps else next(
            (o.frame for o in result.results if o.frame in self._review_pixmaps), None
        )
        if frame is not None:
            self._show_frame(frame)

    def _show_frame(self, frame: str, selected: str | None = None) -> None:
        """Show the review image of ``frame`` with overlays (only ``selected``'s if given)."""
        pixmap = self._review_pixmaps.get(frame)
        if pixmap is None or self._last_result is None:
            return
        shapes = overlay_shapes(self._last_result, frame, selected)
        self.viewer.set_image(pixmap)
        self.viewer.set_overlays(shapes)
        self._viewer_frame, self._viewer_shows_review = frame, True
        what = f"{selected}: {len(shapes)} overlay shape(s)" if selected else f"{len(shapes)} overlay shape(s)"
        self.caption.setText(f"{frame.capitalize()} frame - {what}")

    @Slot()
    def _on_selection_changed(self) -> None:
        """Highlight only the clicked check's overlay (or show all again when nothing is selected)."""
        result = self._last_result
        rows = self.table.selectionModel().selectedRows()
        if result is None:
            return
        if not rows:
            if self._viewer_shows_review and self._viewer_frame:
                self._show_frame(self._viewer_frame)
            return
        name = self.table.item(rows[0].row(), 0).text()
        outcome = next((o for o in result.results if o.name == name), None)
        if outcome is None or outcome.frame is None:  # skipped: it has no image
            self.viewer.set_overlays([])
            if self._viewer_shows_review:  # keep the caption of a freshly chosen frame
                self.caption.setText(f"{name}: {outcome.status if outcome else '?'} - nothing to draw")
            return
        self._show_frame(outcome.frame, selected=name)

    def show_result(self, result: RunResult) -> None:
        """Fill the table (green/red/amber rows) and the big verdict label."""
        rows = result_rows(result)
        self.table.blockSignals(True)
        self.table.setRowCount(len(rows))
        for r, (cells, status, note) in enumerate(rows):
            bg, fg = colors_for(status)
            for c, text in enumerate(cells):
                item = QTableWidgetItem(text)
                item.setBackground(QBrush(QColor(bg)))
                item.setForeground(QBrush(QColor(fg)))
                if note:
                    item.setToolTip(note)
                self.table.setItem(r, c, item)
        self.table.clearSelection()
        self.table.blockSignals(False)
        self._style_verdict(result.verdict)

    def shutdown(self) -> bool:
        """Wait for a running job and deliver its queued results (so the run is saved).

        A folder batch is cancelled first (it stops after the current image).

        Returns:
            True if no job is running any more; False if it did not finish within
            ``SHUTDOWN_WAIT_MS`` (the thread must then be kept alive, so do not close).
        """
        if self._thread is None:
            return True
        if isinstance(self._worker, FolderWorker):
            self._worker.cancel()
        self._thread.quit()
        if not self._thread.wait(SHUTDOWN_WAIT_MS):
            logger.warning("job still running after %d ms; not closing", SHUTDOWN_WAIT_MS)
            return False
        QCoreApplication.sendPostedEvents()  # run the queued result slots (DB saves) before exiting
        return True


# --- main window -----------------------------------------------------------------------


class MainWindow(QMainWindow):
    """Top-level window with *Test*, *History* and *Statistics* tabs."""

    def __init__(self, config_path: str | Path | None = None, db_path: str | Path = DEFAULT_DB) -> None:
        super().__init__()
        self.setWindowTitle("CamQC")
        self.resize(1280, 800)
        config: Config | None = None
        error = ""
        try:
            config = load_config(config_path or default_config_path())
        except ConfigError as exc:
            logger.error("%s", exc)
            error = f"Configuration error: {exc}"
        self.test_tab = TestTab(config, db_path, error)
        self.history_tab = HistoryTab(db_path)
        self.stats_tab = StatsTab(db_path)
        self.tabs = QTabWidget()
        self.tabs.addTab(self.test_tab, "Test")
        self.tabs.addTab(self.history_tab, "History")
        self.tabs.addTab(self.stats_tab, "Statistics")
        self.setCentralWidget(self.tabs)
        # new runs refresh the other tabs (immediately if shown, else when next shown)
        self.test_tab.run_saved.connect(lambda _run_id: self.history_tab.request_refresh())
        self.test_tab.run_saved.connect(lambda _run_id: self.stats_tab.request_refresh())

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802 - Qt override
        """Let a running job finish (and be saved) before the window and its thread go away."""
        if not self.test_tab.shutdown():
            self.test_tab.show_message("A job is still running - wait for it to finish.", error=True)
            event.ignore()
            return
        super().closeEvent(event)


def main(argv: Sequence[str] | None = None) -> int:
    """Start the desktop application (the ``camqc-gui`` entry point)."""
    handler = setup_logging(DEFAULT_LOG, verbose=False)
    try:
        app = QApplication.instance() or QApplication(list(argv if argv is not None else sys.argv))
        window = MainWindow()
        window.show()
        return int(app.exec())
    finally:
        teardown_logging(handler)


if __name__ == "__main__":
    sys.exit(main())
