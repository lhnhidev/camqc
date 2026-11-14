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
