"""Image viewer widget: scaled-to-fit display with check overlays drawn by QPainter.

Overlay shapes come from ``CheckResult.overlay`` and are in *image pixel coordinates*
(``{"type": "circle", "x", "y", "r"}`` centered on pixel (x, y); ``{"type": "rect", "x", "y", "w", "h"}``
with top-left x, y). They are drawn on the already-scaled pixmap, multiplied by the display
scale, so line widths stay constant on screen at any zoom.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from PySide6.QtCore import QPointF, QRectF, QSize, Qt
from PySide6.QtGui import QColor, QImage, QPainter, QPen, QPixmap, QResizeEvent
from PySide6.QtWidgets import QLabel, QSizePolicy, QWidget

from camqc.pipeline import FAIL, PASS, RunResult

logger = logging.getLogger(__name__)

HIGHLIGHT_COLOR = "#ffd400"
HIGHLIGHT_WIDTH = 3
OVERLAY_COLORS: Mapping[str, str] = {PASS: "#2ecc40", FAIL: "#ff4136"}
DEFAULT_OVERLAY_COLOR = "#ff851b"  # orange: clearly not the yellow highlight
THIN_WIDTH = 1


@dataclass(frozen=True)
class StyledShape:
    """An overlay shape (image coordinates) with its pen color and width in screen pixels."""

    shape: Mapping[str, Any]
    color: str
    width: int


def bgr_to_pixmap(image_bgr: np.ndarray) -> QPixmap:
    """Convert a BGR (or gray) uint8 image to a ``QPixmap`` that owns its pixel data."""
    rgb = np.ascontiguousarray(image_bgr[..., ::-1]) if image_bgr.ndim == 3 else np.ascontiguousarray(image_bgr)
    h, w = rgb.shape[:2]
    fmt = QImage.Format.Format_RGB888 if rgb.ndim == 3 else QImage.Format.Format_Grayscale8
    return QPixmap.fromImage(QImage(rgb.data, w, h, rgb.strides[0], fmt).copy())


def overlay_shapes(result: RunResult, frame: str, selected: str | None = None) -> list[StyledShape]:
    """Return the overlay shapes to draw on the ``frame`` image of a run.

    Args:
        result: The pipeline result.
        frame: Frame type being displayed; only checks that used this frame contribute.
        selected: Name of the clicked check. If given, only that check's overlay is
            returned, highlighted; otherwise every check's overlay is returned thinly
            in its status color.
    """
    shapes: list[StyledShape] = []
    for outcome in result.results:
        res = outcome.result
        if res is None or outcome.frame != frame or not res.overlay:
            continue
        if selected is not None:
            if outcome.name == selected:
                shapes += [StyledShape(s, HIGHLIGHT_COLOR, HIGHLIGHT_WIDTH) for s in res.overlay]
        else:
            color = OVERLAY_COLORS.get(outcome.status, DEFAULT_OVERLAY_COLOR)
            shapes += [StyledShape(s, color, THIN_WIDTH) for s in res.overlay]
    return shapes


def draw_shapes(pixmap: QPixmap, shapes: Sequence[StyledShape], scale_x: float, scale_y: float) -> None:
    """Paint ``shapes`` onto ``pixmap`` in place, scaling image coordinates by the given factors."""
    painter = QPainter(pixmap)
    try:
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        for styled in shapes:
            s = styled.shape
            painter.setPen(QPen(QColor(styled.color), styled.width))
            kind = s.get("type")
            if kind == "circle":
                # circle centers are pixel indices (e.g. blob centroids): the pixel's middle is +0.5
                center = QPointF((s["x"] + 0.5) * scale_x, (s["y"] + 0.5) * scale_y)
                painter.drawEllipse(center, s["r"] * scale_x, s["r"] * scale_y)
            elif kind == "rect":
                painter.drawRect(QRectF(s["x"] * scale_x, s["y"] * scale_y, s["w"] * scale_x, s["h"] * scale_y))
            else:
                logger.debug("ignoring overlay shape of unknown type: %r", s)
    finally:
        painter.end()


class ScaledImageLabel(QLabel):
    """A label that shows an image scaled to fit its size, keeping the aspect ratio.

    It ignores the pixmap's size in layouts, so large images never enlarge the window.
    Overlay shapes set with :meth:`set_overlays` are redrawn on every resize.
    """

    def __init__(self, placeholder: str = "No image", parent: QWidget | None = None) -> None:
        super().__init__(placeholder, parent)
        self._source: QPixmap | None = None
        self._shapes: list[StyledShape] = []
        self._placeholder = placeholder
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(QSize(240, 160))
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self.setStyleSheet("background: #202020; color: #909090; border: 1px solid #505050;")

    @property
    def shapes(self) -> list[StyledShape]:
        """The overlay shapes currently drawn (a copy)."""
        return list(self._shapes)

    def set_image(self, pixmap: QPixmap | None) -> None:
        """Show ``pixmap`` (or the placeholder text when ``None``); this clears any overlays."""
        self._source = pixmap
        self._shapes = []
        self._refresh()

    def set_overlays(self, shapes: Sequence[StyledShape]) -> None:
        """Draw ``shapes`` (image coordinates) over the current image."""
        self._shapes = list(shapes)
        self._refresh()

    def _refresh(self) -> None:
        """Scale the source image to the current size, draw the overlays, and show it."""
        if self._source is None or self._source.isNull():
            self.setPixmap(QPixmap())
            self.setText(self._placeholder)
            return
        scaled = self._source.scaled(
            self.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
        )
        if self._shapes:
            draw_shapes(
                scaled,
                self._shapes,
                scaled.width() / self._source.width(),
                scaled.height() / self._source.height(),
            )
        self.setPixmap(scaled)

    def resizeEvent(self, event: QResizeEvent) -> None:  # noqa: N802 - Qt override
        """Rescale the image and overlays whenever the label is resized."""
        super().resizeEvent(event)
        if self._source is not None:
            self._refresh()
