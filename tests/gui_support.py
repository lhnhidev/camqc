"""Helpers shared by the Qt GUI tests (import only after ``pytest.importorskip("PySide6")``)."""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication


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
    """Write a solid-gray PNG and return its path."""
    assert cv2.imwrite(str(path), np.full((*size, 3), value, np.uint8))
    return path


def click(widget: Any) -> None:
    """Left-click the center of ``widget``."""
    QTest.mouseClick(widget, Qt.MouseButton.LeftButton)
