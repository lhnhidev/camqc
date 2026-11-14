"""Blemish check: detect dust/stain regions on a flat-field image and fail if the count exceeds the limit.

The background is estimated with a large Gaussian blur (sigma = image width / 20)
and divided out; pixels darker than ``1 - contrast`` of the background form a mask
that is cleaned by morphological opening. Each connected component of at least
``min_area`` pixels is one blemish.

Assumes a well-exposed, smoothly lit flat field: dark/noisy frames and steep vignetting
produce false blobs (use the brightness and shading checks for those defects).
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from camqc.checks.base import CheckResult, make_result, to_gray

BACKGROUND_SIGMA_DIVISOR = 20
OPEN_KSIZE = 3
DEFAULT_CONTRAST = 0.05
DEFAULT_MIN_AREA = 30
MAX_LISTED_BLOBS = 200
MIN_SIDE_PX = 40
OVERLAY_PADDING = 4


def flat_field_ratio(gray: np.ndarray) -> np.ndarray:
    """Return ``gray / background`` where the background is a large Gaussian blur of ``gray``.

    The result is ~1.0 on a clean flat field (including smooth vignetting) and
    below 1.0 on dark blemishes.
    """
    g = gray.astype(np.float32)
    background = cv2.GaussianBlur(g, (0, 0), g.shape[1] / BACKGROUND_SIGMA_DIVISOR)
    return g / np.maximum(background, 1e-3)


def blemish_mask(ratio: np.ndarray, contrast: float) -> np.ndarray:
    """Return a uint8 mask of pixels with ``ratio < 1 - contrast``, cleaned by opening."""
    mask = (ratio < 1.0 - contrast).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (OPEN_KSIZE, OPEN_KSIZE))
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)


def find_blobs(mask: np.ndarray, ratio: np.ndarray, min_area: int) -> list[dict[str, Any]]:
    """Return blobs of at least ``min_area`` pixels, largest first.

    Each blob is ``{"x", "y", "area", "contrast"}`` where ``x, y`` is the rounded
    centroid and ``contrast`` is ``1 - mean(ratio)`` over the blob's pixels.
    """
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    ratio_sums = np.bincount(labels.ravel(), weights=ratio.ravel(), minlength=count)
    blobs = []
    for i in range(1, count):  # 0 is the background
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        blobs.append({
            "x": round(centroids[i][0]),
            "y": round(centroids[i][1]),
            "area": area,
            "contrast": float(1.0 - ratio_sums[i] / area),
        })
    return sorted(blobs, key=lambda b: -b["area"])


def run(image_bgr: np.ndarray, params: dict[str, Any]) -> CheckResult:
    """Count dark blemishes on a flat-field image.

    Args:
        image_bgr: Flat-field image in BGR order (uint8).
        params: ``contrast`` (default 0.05) is the minimum relative darkness vs.
            the background, ``min_area`` (default 30) the minimum blob size in
            pixels, and ``max_blobs`` the maximum allowed count (missing means
            no limit).

    Returns:
        A result whose value is the blob count. Details hold ``count``,
        ``blobs`` (list of ``{"x", "y", "area", "contrast"}``, largest first,
        capped at 200), ``truncated`` and, on failure, ``reason``. The overlay
        has one circle per listed blob. Fails when ``count > max_blobs``.

    Raises:
        ValueError: If the image is smaller than 40 px on a side.
    """
    gray = to_gray(image_bgr)
    h, w = gray.shape
    if min(h, w) < MIN_SIDE_PX:
        raise ValueError(f"image too small for blemish check: {w}x{h}")
    contrast = params.get("contrast", DEFAULT_CONTRAST)
    min_area = params.get("min_area", DEFAULT_MIN_AREA)
    max_blobs = params.get("max_blobs")

    ratio = flat_field_ratio(gray)
    blobs = find_blobs(blemish_mask(ratio, contrast), ratio, min_area)
    listed = blobs[:MAX_LISTED_BLOBS]

    details: dict[str, Any] = {
        "count": len(blobs),
        "blobs": listed,
        "truncated": len(blobs) > len(listed),
    }
    if max_blobs is not None and len(blobs) > max_blobs:
        details["reason"] = f"{len(blobs)} blemishes above max {max_blobs}"
    overlay = [
        {
            "type": "circle",
            "x": b["x"],
            "y": b["y"],
            "r": round(np.sqrt(b["area"] / np.pi)) + OVERLAY_PADDING,
        }
        for b in listed
    ]
    return make_result("blemish", len(blobs), "count", None, max_blobs, details, overlay)
