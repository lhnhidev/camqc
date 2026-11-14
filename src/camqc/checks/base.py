"""Shared check interface: the result type (value, threshold, PASS/FAIL) and the base contract every check follows.

Convention: every check module exposes
``run(image_bgr: np.ndarray, params: dict) -> CheckResult``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np


@dataclass(frozen=True)
class CheckResult:
    """Outcome of a single check on one image.

    Attributes:
        name: Check name, e.g. ``"brightness"``.
        value: The main measured value.
        unit: Unit of ``value`` (e.g. ``"gray"``, ``"score"``).
        low: Minimum accepted value, or ``None`` if unbounded.
        high: Maximum accepted value, or ``None`` if unbounded.
        passed: Whether the image passed this check.
        details: Extra measurements and, on failure, ``details["reason"]``.
        overlay: Shapes to draw on the image, e.g.
            ``{"type": "circle", "x": .., "y": .., "r": ..}`` or
            ``{"type": "rect", "x": .., "y": .., "w": .., "h": ..}``.
    """

    name: str
    value: float
    unit: str
    low: float | None
    high: float | None
    passed: bool
    details: dict[str, Any] = field(default_factory=dict)
    overlay: list[dict[str, Any]] = field(default_factory=list)


def to_gray(image_bgr: np.ndarray) -> np.ndarray:
    """Return a single-channel uint8 view of a BGR (or already gray) image."""
    if image_bgr.ndim == 2:
        return image_bgr
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)


Rect = tuple[int, int, int, int]  # (x, y, w, h)


def center_region(h: int, w: int, fraction: float = 0.5) -> Rect:
    """Return ``(x, y, w, h)`` of the centered region covering ``fraction`` of each side."""
    rw, rh = int(w * fraction), int(h * fraction)
    return (w - rw) // 2, (h - rh) // 2, rw, rh


def corner_regions(
    h: int, w: int, fraction: float = 0.2
) -> dict[str, Rect]:
    """Return ``(x, y, w, h)`` for the four corner regions, each ``fraction`` of a side."""
    rw, rh = int(w * fraction), int(h * fraction)
    return {
        "top_left": (0, 0, rw, rh),
        "top_right": (w - rw, 0, rw, rh),
        "bottom_left": (0, h - rh, rw, rh),
        "bottom_right": (w - rw, h - rh, rw, rh),
    }


def crop(img: np.ndarray, rect: Rect) -> np.ndarray:
    """Return the sub-image for an ``(x, y, w, h)`` rectangle."""
    x, y, w, h = rect
    return img[y : y + h, x : x + w]


def format_limits(low: float | None, high: float | None) -> str:
    """Describe an acceptance range, e.g. ``100 .. 160``, ``>= 100``, ``<= 5`` or ``-``."""
    if low is not None and high is not None:
        return f"{low:g} .. {high:g}"
    if low is not None:
        return f">= {low:g}"
    if high is not None:
        return f"<= {high:g}"
    return "-"


def in_range(value: float, low: float | None, high: float | None) -> bool:
    """Return True if ``low <= value <= high``; a ``None`` bound is unbounded."""
    return (low is None or value >= low) and (high is None or value <= high)


def make_result(
    name: str,
    value: float,
    unit: str,
    low: float | None,
    high: float | None,
    details: dict[str, Any] | None = None,
    overlay: list[dict[str, Any]] | None = None,
) -> CheckResult:
    """Build a :class:`CheckResult`, computing ``passed`` from ``value`` and the limits.

    Args:
        name: Check name.
        value: Measured value.
        unit: Unit of ``value``.
        low: Minimum accepted value, or ``None``.
        high: Maximum accepted value, or ``None``.
        details: Extra data; copied so the caller's dict is not shared.
        overlay: Overlay shapes; copied likewise.

    Returns:
        A result with ``passed = low <= value <= high``. A NaN ``value`` fails
        whenever any limit is set.
    """
    return CheckResult(
        name=name,
        value=float(value),
        unit=unit,
        low=low,
        high=high,
        passed=in_range(float(value), low, high),
        details=dict(details or {}),
        overlay=list(overlay or []),
    )
