"""Color check: white balance / channel ratios on a neutral target must stay within configured tolerances.

The image is split into an 8x8 grid and the mean color of each cell is measured.
Color uniformity is the maximum CIE76 delta E (in CIELAB) of any cell relative to
the center of the image; white balance is the global R/G and B/G channel ratio.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import cv2
import numpy as np

from camqc.checks.base import CheckResult, in_range, make_result

GRID = 8
BLUE, GREEN, RED = 0, 1, 2  # BGR channel indices


def grid_edges(size: int, n: int = GRID) -> np.ndarray:
    """Return the ``n + 1`` integer pixel borders that split ``size`` pixels into ``n`` cells."""
    return np.rint(np.linspace(0, size, n + 1)).astype(int)


def grid_means(image_bgr: np.ndarray, n: int = GRID) -> np.ndarray:
    """Return the mean B, G, R of each cell of an ``n`` x ``n`` grid as an ``(n, n, 3)`` array.

    Cell borders are rounded to whole pixels, so cells differ by at most one pixel in size.
    """
    h, w = image_bgr.shape[:2]
    ys, xs = grid_edges(h, n), grid_edges(w, n)
    means = np.empty((n, n, 3), dtype=np.float64)
    for r in range(n):
        for c in range(n):
            means[r, c] = image_bgr[ys[r] : ys[r + 1], xs[c] : xs[c + 1]].reshape(-1, 3).mean(axis=0)
    return means


def bgr_to_lab(means_bgr: np.ndarray) -> np.ndarray:
    """Convert an ``(..., 3)`` array of 0-255 BGR values to CIELAB (L 0-100) with OpenCV."""
    flat = (means_bgr.reshape(-1, 1, 3) / 255.0).astype(np.float32)
    return cv2.cvtColor(flat, cv2.COLOR_BGR2Lab).reshape(means_bgr.shape).astype(np.float64)


def delta_e76(lab: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Return the CIE76 delta E (Euclidean Lab distance) of each ``lab`` entry to ``reference``."""
    return np.linalg.norm(lab - reference, axis=-1)


def center_reference_bgr(means: np.ndarray) -> np.ndarray:
    """Return the mean BGR of the central 2x2 cells (an even grid has no single center cell)."""
    n = means.shape[0]
    lo = n // 2 - 1
    return means[lo : lo + 2, lo : lo + 2].reshape(-1, 3).mean(axis=0)


def channel_ratios(image_bgr: np.ndarray) -> tuple[float, float]:
    """Return the global ``(R/G, B/G)`` ratios; both NaN if the green mean is zero."""
    b, g, r = (float(v) for v in cv2.mean(image_bgr)[:3])
    if g <= 0:
        return float("nan"), float("nan")
    return r / g, b / g


def run(image_bgr: np.ndarray, params: dict[str, Any]) -> CheckResult:
    """Check color uniformity (delta E) and white balance (channel ratios).

    Args:
        image_bgr: Neutral (gray/white) target in BGR order (uint8). A 2D gray
            image is accepted and treated as perfectly neutral.
        params: ``ratio_low`` / ``ratio_high`` bound both R/G and B/G, and
            ``max_delta_e`` caps the maximum cell delta E. Missing keys disable
            that condition.

    Returns:
        A result whose value is the maximum delta E of any cell relative to the
        center. Details hold ``r_g``, ``b_g``, the ``ratio_low`` / ``ratio_high`` limits,
        ``worst_cell`` (row * 8 + col),
        ``worst_cell_rc`` (``[row, col]``), ``delta_e`` (64 values, row-major)
        and, on failure, ``reason``. The overlay marks the worst cell when its
        delta E exceeds ``max_delta_e``. Fails if delta E or either ratio is
        out of range.

    Raises:
        ValueError: If the image is smaller than 8 px on a side.
    """
    if image_bgr.ndim == 2:
        image_bgr = cv2.cvtColor(image_bgr, cv2.COLOR_GRAY2BGR)
    h, w = image_bgr.shape[:2]
    if min(h, w) < GRID:
        raise ValueError(f"image too small for color check: {w}x{h}")

    means = grid_means(image_bgr)
    lab = bgr_to_lab(means)
    reference = bgr_to_lab(center_reference_bgr(means))
    delta = delta_e76(lab, reference)
    flat_idx = int(np.argmax(delta))
    row, col = divmod(flat_idx, GRID)
    max_de = float(delta[row, col])
    r_g, b_g = channel_ratios(image_bgr)

    ratio_low, ratio_high = params.get("ratio_low"), params.get("ratio_high")
    max_delta_e = params.get("max_delta_e")

    reasons: list[str] = []
    de_bad = max_delta_e is not None and max_de > max_delta_e
    if de_bad:
        reasons.append(f"cell {flat_idx} delta E {max_de:.1f} above {max_delta_e}")
    for label, ratio in (("R/G", r_g), ("B/G", b_g)):
        if not in_range(ratio, ratio_low, ratio_high):  # NaN is never in range once a bound is set
            reasons.append(f"{label} {ratio:.3f} outside [{ratio_low}, {ratio_high}]")

    details: dict[str, Any] = {
        "r_g": r_g,
        "b_g": b_g,
        "ratio_low": ratio_low,
        "ratio_high": ratio_high,
        "worst_cell": flat_idx,
        "worst_cell_rc": [row, col],
        "delta_e": delta.reshape(-1).tolist(),
    }
    if reasons:
        details["reason"] = "; ".join(reasons)
    overlay: list[dict[str, Any]] = []
    if de_bad:
        ys, xs = grid_edges(h), grid_edges(w)
        overlay.append({
            "type": "rect",
            "x": int(xs[col]),
            "y": int(ys[row]),
            "w": int(xs[col + 1] - xs[col]),
            "h": int(ys[row + 1] - ys[row]),
        })
    result = make_result("color", max_de, "dE76", None, max_delta_e, details, overlay)
    return replace(result, passed=False) if reasons else result
