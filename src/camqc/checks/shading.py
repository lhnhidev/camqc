"""Lens shading check: corner-to-center brightness ratio (vignetting) must stay above a configured minimum.

Mean luminance is measured on a center patch and the four corner patches, each
10% of the image width and height. The value is ``min(corner) / center * 100``.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from camqc.checks.base import (
    CheckResult,
    Rect,
    center_region,
    corner_regions,
    crop,
    make_result,
    to_gray,
)

PATCH_FRACTION = 0.1
MIN_SIDE_PX = 10


def patch_mean(gray: np.ndarray, rect: Rect) -> float:
    """Return the mean luminance of the ``(x, y, w, h)`` patch of ``gray``."""
    return float(crop(gray, rect).mean())


def corner_to_center_pct(corners: dict[str, float], center: float) -> float:
    """Return ``min(corners) / center * 100``; 0.0 if the center is black (ratio undefined)."""
    if center <= 0:
        return 0.0
    return min(corners.values()) / center * 100.0


def run(image_bgr: np.ndarray, params: dict[str, Any]) -> CheckResult:
    """Measure lens shading (vignetting).

    Args:
        image_bgr: Flat-field image in BGR order (uint8).
        params: ``min_corner_pct`` is the minimum accepted corner/center
            percentage. A missing key means no limit.

    Returns:
        A result whose value is ``min(corner) / center * 100`` in percent.
        Details hold ``center``, ``corners`` (the four patch means),
        ``min_corner`` and ``min_corner_name``. The overlay marks the five
        patches as rectangles. Fails with ``details["reason"]`` when the value is
        below ``min_corner_pct``.

    Raises:
        ValueError: If the image is smaller than 10 px on a side.
    """
    gray = to_gray(image_bgr)
    h, w = gray.shape
    if min(h, w) < MIN_SIDE_PX:
        raise ValueError(f"image too small for shading check: {w}x{h}")
    center_rect = center_region(h, w, PATCH_FRACTION)
    corner_rects = corner_regions(h, w, PATCH_FRACTION)

    center = patch_mean(gray, center_rect)
    corners = {name: patch_mean(gray, r) for name, r in corner_rects.items()}
    value = corner_to_center_pct(corners, center)
    worst = min(corners, key=corners.__getitem__)

    low = params.get("min_corner_pct")
    details: dict[str, Any] = {
        "center": center,
        "corners": corners,
        "min_corner": corners[worst],
        "min_corner_name": worst,
    }
    if low is not None and value < low:
        details["reason"] = f"{worst} is {value:.1f}% of center, below {low}%"
    overlay = [
        {"type": "rect", "x": x, "y": y, "w": rw, "h": rh}
        for x, y, rw, rh in [center_rect, *corner_rects.values()]
    ]
    return make_result("shading", value, "%", low, None, details, overlay)
