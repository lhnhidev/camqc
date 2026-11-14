"""Defective pixel check: count dead (stuck dark) and hot (stuck bright) pixels against configured limits.

Expects a dark frame in ``"hot"`` mode or a white/bright flat frame in ``"dead"`` mode.
A pixel is defective when it deviates from its 5x5 local median by more than
``diff_threshold`` in the direction given by the mode.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from camqc.checks.base import CheckResult, make_result, to_gray

MODES = ("hot", "dead")
MEDIAN_KSIZE = 5
DEFAULT_DIFF_THRESHOLD = 30
MAX_LISTED_DEFECTS = 200
OVERLAY_RADIUS = 8


def defect_mask(gray: np.ndarray, mode: str, diff_threshold: float) -> np.ndarray:
    """Return a boolean mask of pixels deviating from the local median past the threshold.

    Args:
        gray: uint8 single-channel image.
        mode: ``"hot"`` flags pixels brighter than the median, ``"dead"`` darker ones.
        diff_threshold: Minimum absolute difference from the median to count.

    Returns:
        Boolean array, True where a defect was detected.
    """
    median = cv2.medianBlur(gray, MEDIAN_KSIZE)
    signed = gray.astype(np.int16) - median.astype(np.int16)
    diff = signed if mode == "hot" else -signed
    return diff > diff_threshold


def defect_centers(mask: np.ndarray) -> list[tuple[int, int]]:
    """Return one ``(x, y)`` per connected component of ``mask``, in raster order."""
    count, _, _, centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    points = [(round(cx), round(cy)) for cx, cy in centroids[1:count]]  # 0 is background
    return sorted(points, key=lambda p: (p[1], p[0]))


def run(image_bgr: np.ndarray, params: dict[str, Any]) -> CheckResult:
    """Count hot or dead pixels.

    Args:
        image_bgr: Image in BGR order (uint8): a dark frame for ``"hot"``, a
            white frame for ``"dead"``.
        params: ``mode`` (required, ``"hot"`` or ``"dead"``), ``diff_threshold``
            (default 30) and ``max_defects`` (maximum allowed count; missing
            means no limit).

    Returns:
        A result whose value is the number of defective pixels (connected
        components). Details hold ``mode``, ``count``, ``defects`` (list of
        ``(x, y)``, capped at 200) and ``truncated``. The overlay has a circle of
        radius 8 per listed defect. Fails with ``details["reason"]`` when
        ``count > max_defects``.

    Raises:
        ValueError: If ``mode`` is missing or not ``"hot"`` / ``"dead"``.
    """
    mode = params.get("mode")
    if mode not in MODES:
        raise ValueError(f"params['mode'] must be one of {MODES}, got {mode!r}")
    threshold = params.get("diff_threshold", DEFAULT_DIFF_THRESHOLD)
    max_defects = params.get("max_defects")

    centers = defect_centers(defect_mask(to_gray(image_bgr), mode, threshold))
    count = len(centers)
    listed = centers[:MAX_LISTED_DEFECTS]

    details: dict[str, Any] = {
        "mode": mode,
        "count": count,
        "defects": listed,
        "truncated": count > len(listed),
    }
    if max_defects is not None and count > max_defects:
        details["reason"] = f"{count} {mode} pixels above max {max_defects}"
    overlay = [{"type": "circle", "x": x, "y": y, "r": OVERLAY_RADIUS} for x, y in listed]
    return make_result("pixels", count, "count", None, max_defects, details, overlay)
