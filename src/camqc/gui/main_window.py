"""Main window: image source selection, live preview, run checks, and per-check PASS/FAIL display.

A ``QMainWindow`` with three tabs: *Test* (this module), and *History* / *Statistics*
placeholders. The pipeline runs in a ``QThread`` worker so the UI never freezes; the run
is saved to SQLite on the GUI thread when the worker finishes (sqlite3 connections must
not cross threads).
"""

from __future__ import annotations

import logging
import sqlite3
import sys
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path

import numpy as np
from PySide6.QtCore import QCoreApplication, QObject, QSize, Qt, QThread, Signal, Slot
from PySide6.QtGui import QBrush, QCloseEvent, QColor, QFont, QImage, QPixmap, QResizeEvent
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
    QPushButton,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from camqc import db
from camqc.cli import (
    DEFAULT_DB,
    DEFAULT_LOG,
    CliError,
    default_config_path,
    format_limits,
    load_image,
    setup_logging,
    teardown_logging,
)
from camqc.pipeline import (
    FAIL,
    FRAME_TYPES,
    PASS,
    Config,
    ConfigError,
    RunResult,
    load_config,
    run_pipeline,
)

logger = logging.getLogger(__name__)

IMAGE_FILTER = "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff);;All files (*)"
DEFAULT_STATION = "UNKNOWN"
FRAME_BUTTON_ORDER = ("normal", "dark", "white", "flat")  # flat last: only some checks need it

GREEN, RED, AMBER, GRAY = "#c6efce", "#ffc7ce", "#ffeb9c", "#e0e0e0"
STATUS_COLORS: Mapping[str, tuple[str, str]] = {  # status -> (background, text)
    "PASS": (GREEN, "#006100"),
    "FAIL": (RED, "#9c0006"),
    "INCOMPLETE": (AMBER, "#9c5700"),
    "SKIPPED": (AMBER, "#9c5700"),
    "ERROR": (RED, "#9c0006"),
}
NEUTRAL_COLORS = (GRAY, "#404040")
SHUTDOWN_WAIT_MS = 30_000  # how long closing the window waits for a running pipeline
TABLE_HEADERS = ("Check", "Value", "Unit", "Limits", "Status")


# --- helpers ---------------------------------------------------------------------------


def bgr_to_pixmap(image_bgr: np.ndarray) -> QPixmap:
    """Convert a BGR (or gray) uint8 image to a ``QPixmap`` that owns its pixel data."""
    rgb = np.ascontiguousarray(image_bgr[..., ::-1]) if image_bgr.ndim == 3 else np.ascontiguousarray(image_bgr)
    h, w = rgb.shape[:2]
    fmt = QImage.Format.Format_RGB888 if rgb.ndim == 3 else QImage.Format.Format_Grayscale8
    return QPixmap.fromImage(QImage(rgb.data, w, h, rgb.strides[0], fmt).copy())


def colors_for(status: str) -> tuple[str, str]:
    """Return ``(background, text)`` colors for a status or verdict word."""
    return STATUS_COLORS.get(status, NEUTRAL_COLORS)


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


class ScaledImageLabel(QLabel):
    """A label that shows an image scaled to fit its size, keeping the aspect ratio.

    It ignores the pixmap's size in layouts, so large images never enlarge the window.
    """

    def __init__(self, placeholder: str = "No image", parent: QWidget | None = None) -> None:
        super().__init__(placeholder, parent)
        self._source: QPixmap | None = None
        self._placeholder = placeholder
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(QSize(240, 160))
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self.setStyleSheet("background: #202020; color: #909090; border: 1px solid #505050;")

    def set_image(self, pixmap: QPixmap | None) -> None:
        """Show ``pixmap`` (or the placeholder text when ``None``)."""
        self._source = pixmap
        self._refresh()

    def _refresh(self) -> None:
        """Scale the source image to the current size (or show the placeholder)."""
        if self._source is None or self._source.isNull():
            self.setPixmap(QPixmap())
            self.setText(self._placeholder)
            return
        self.setPixmap(self._source.scaled(
            self.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
        ))

    def resizeEvent(self, event: QResizeEvent) -> None:  # noqa: N802 - Qt override
        """Rescale the image whenever the label is resized."""
        super().resizeEvent(event)
        if self._source is not None:
            self._refresh()


class PipelineWorker(QObject):
    """Runs the pipeline off the GUI thread.

    Signals:
        finished(RunResult): the run completed.
        failed(str): the pipeline itself raised (individual check errors are not failures).
    """

    finished = Signal(object)
    failed = Signal(str)

    def __init__(self, model: str, frames: Mapping[str, np.ndarray], config: Config) -> None:
        super().__init__()
        self._model, self._frames, self._config = model, dict(frames), config

    @Slot()
    def run(self) -> None:
        """Execute the pipeline and emit the outcome."""
        try:
            result = run_pipeline(self._model, self._frames, self._config)
        except Exception as exc:  # noqa: BLE001 - never let a worker exception kill the thread silently
            logger.exception("pipeline failed")
            self.failed.emit(f"{type(exc).__name__}: {exc}")
        else:
            self.finished.emit(result)


# --- Test tab --------------------------------------------------------------------------


class TestTab(QWidget):
    """The *Test* tab: choose model/serial/frames, run, and see per-check results."""

    __test__ = False  # not a pytest test class despite the name

    def __init__(self, config: Config | None, db_path: str | Path, config_error: str = "") -> None:
        super().__init__()
        self._config = config
        self._db_path = db_path
        self._frames: dict[str, np.ndarray] = {}
        self._frame_paths: dict[str, Path] = {}
        self._last_dir = ""
        self._thread: QThread | None = None
        self._worker: PipelineWorker | None = None
        self._pending: tuple[str, str, str] | None = None  # (model, serial, station) of the active run
        self._build_ui()
        if config_error:
            self.show_message(config_error, error=True)
            self.run_button.setEnabled(False)

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
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setStretchLastSection(True)

        self.message_label = QLabel("", objectName="message_label", wordWrap=True)

        left = QVBoxLayout()
        left.addWidget(self.viewer, 1)
        left.addWidget(self.caption)
        right = QVBoxLayout()
        right.addWidget(self.verdict_label)
        right.addWidget(self.table, 1)
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
        """True while a worker thread is active."""
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
        self._frames[frame], self._frame_paths[frame] = image, path
        self._last_dir = str(path.parent)
        label = self.frame_labels[frame]
        label.setText(path.name)
        label.setToolTip(str(path))
        label.setStyleSheet("")
        self.viewer.set_image(bgr_to_pixmap(image))
        h, w = image.shape[:2]
        self.caption.setText(f"{frame.capitalize()}: {path.name}  ({w}x{h})")
        self.show_message("")
        return True

    # -- running --

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

        self.run_button.setEnabled(False)
        self.show_message(f"Running {model} on {serial}...")
        self._style_verdict("RUNNING")

        thread = QThread(self)
        worker = PipelineWorker(model, self._frames, self._config)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(self._on_finished)
        worker.failed.connect(self._on_failed)
        worker.finished.connect(thread.quit)
        worker.failed.connect(thread.quit)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(self._on_thread_done)
        self._thread, self._worker = thread, worker
        thread.start()

    @Slot(object)
    def _on_finished(self, result: RunResult) -> None:
        """Show the result and save it to the database (on the GUI thread)."""
        self.show_result(result)
        self._clear_frames()  # a scan + Enter must never re-run the previous unit's images
        if self._pending is None:
            logger.error("run finished without pending run info; result not saved")
            return
        model, serial, station = self._pending
        try:
            conn = db.init_db(self._db_path)
            try:
                run_id = db.save_run(conn, db.RunInput(serial, model, station, datetime.now()), result)
            finally:
                conn.close()
        except (sqlite3.Error, OSError, ValueError) as exc:
            logger.exception("could not save run")
            self.show_message(f"Result NOT saved: {exc}", error=True)
            return
        self.show_message(f"Saved as run #{run_id} ({result.duration_ms:.0f} ms)")

    @Slot(str)
    def _on_failed(self, message: str) -> None:
        """Report an exception raised by the pipeline itself."""
        self._style_verdict("ERROR")
        self.table.setRowCount(0)
        self.show_message(f"Pipeline failed: {message}", error=True)

    @Slot()
    def _on_thread_done(self) -> None:
        """Release the finished thread and re-enable Run."""
        thread, self._thread, self._worker = self._thread, None, None
        if thread is not None:
            thread.deleteLater()
        self._pending = None
        self.run_button.setEnabled(self._config is not None)

    def _clear_frames(self) -> None:
        """Forget the chosen frames (the viewer keeps showing the last image for review)."""
        self._frames.clear()
        self._frame_paths.clear()
        for label in self.frame_labels.values():
            label.setText("not selected")
            label.setToolTip("")
            label.setStyleSheet("color: #808080;")

    def show_result(self, result: RunResult) -> None:
        """Fill the table (green/red/amber rows) and the big verdict label."""
        rows = result_rows(result)
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
        self._style_verdict(result.verdict)

    def shutdown(self) -> bool:
        """Wait for a running worker and deliver its queued result (so the run is saved).

        Returns:
            True if no worker is running any more; False if it did not finish within
            ``SHUTDOWN_WAIT_MS`` (the thread must then be kept alive, so do not close).
        """
        if self._thread is None:
            return True
        self._thread.quit()
        if not self._thread.wait(SHUTDOWN_WAIT_MS):
            logger.warning("pipeline still running after %d ms; not closing", SHUTDOWN_WAIT_MS)
            return False
        QCoreApplication.sendPostedEvents()  # run the queued _on_finished (DB save) before exiting
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
        self.tabs = QTabWidget()
        self.tabs.addTab(self.test_tab, "Test")
        self.tabs.addTab(self._placeholder("History"), "History")
        self.tabs.addTab(self._placeholder("Statistics"), "Statistics")
        self.setCentralWidget(self.tabs)

    @staticmethod
    def _placeholder(name: str) -> QWidget:
        label = QLabel(f"{name} - coming soon")
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label.setStyleSheet("color: #808080; font-size: 18pt;")
        return label

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802 - Qt override
        """Let a running test finish (and be saved) before the window and its thread go away."""
        if not self.test_tab.shutdown():
            self.test_tab.show_message("A test is still running - wait for it to finish.", error=True)
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
