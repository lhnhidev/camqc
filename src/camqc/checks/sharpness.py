"""Sharpness check: focus measure (variance of Laplacian) must exceed a configured minimum.

The score is computed on the central 50% region (the check's value) and on the
four corner regions (each 20% x 20% of the image, reported in details).
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from camqc.checks.base import (
    CheckResult,
    center_region,
    corner_regions,
    crop,
    make_result,
    to_gray,
)

MIN_SIDE_PX = 10


def laplacian(gray: np.ndarray) -> np.ndarray:
    """Return the Laplacian of ``gray`` as float64."""
    return cv2.Laplacian(gray, cv2.CV_64F)


def run(image_bgr: np.ndarray, params: dict[str, Any]) -> CheckResult:
    """Measure sharpness at the center and corners.

    Args:
        image_bgr: Image in BGR order (uint8).
        params: ``min_center`` is the minimum accepted center score and
            ``max_center`` an optional maximum. Missing keys are unbounded.

    Returns:
        A result whose value is the center variance of Laplacian. Details hold
        ``center`` and ``corners`` (a dict of the four corner scores). The
        overlay marks the center and corner regions as rectangles. Fails with
        ``details["reason"]`` when the center score is out of range.
    """
    gray = to_gray(image_bgr)
    h, w = gray.shape
    if min(h, w) < MIN_SIDE_PX:
        raise ValueError(f"image too small for sharpness check: {w}x{h}")
    center_rect = center_region(h, w)
    corner_rects = corner_regions(h, w)

    lap = laplacian(gray)  # computed once so crop edges are not reflected
    center = float(crop(lap, center_rect).var())
    corners = {name: float(crop(lap, r).var()) for name, r in corner_rects.items()}

    low = params.get("min_center")
    high = params.get("max_center")
    overlay = [
        {"type": "rect", "x": x, "y": y, "w": rw, "h": rh}
        for x, y, rw, rh in [center_rect, *corner_rects.values()]
    ]
    details: dict[str, Any] = {"center": center, "corners": corners}
    if low is not None and center < low:
        details["reason"] = f"center score {center:.1f} below {low}"
    elif high is not None and center > high:
        details["reason"] = f"center score {center:.1f} above {high}"
    return make_result("sharpness", center, "score", low, high, details, overlay)
