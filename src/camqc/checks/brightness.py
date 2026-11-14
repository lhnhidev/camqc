"""Brightness check: mean luminance of the image must fall within a configured range.

Also reports the share of clipped (>250) and crushed (<5) pixels; clipping above
``max_clipped_pct`` fails the check.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np

from camqc.checks.base import CheckResult, make_result, to_gray

CLIPPED_ABOVE = 250
CRUSHED_BELOW = 5


def pixel_percent(mask: np.ndarray) -> float:
    """Return the percentage (0-100) of True entries in a boolean mask."""
    return float(mask.mean() * 100.0)


def run(image_bgr: np.ndarray, params: dict[str, Any]) -> CheckResult:
    """Check mean luminance and clipping.

    Args:
        image_bgr: Image in BGR order (uint8).
        params: ``mean_low`` / ``mean_high`` bound the mean luminance and
            ``max_clipped_pct`` caps the percent of pixels > 250. Missing keys
            disable that condition.

    Returns:
        A result whose value is the mean luminance. Fails if any condition is
        violated; the reasons are joined into ``details["reason"]``. Details
        always include ``mean``, ``clipped_pct`` and ``crushed_pct``.
    """
    gray = to_gray(image_bgr)
    mean = float(gray.mean())
    clipped_pct = pixel_percent(gray > CLIPPED_ABOVE)
    crushed_pct = pixel_percent(gray < CRUSHED_BELOW)

    low = params.get("mean_low")
    high = params.get("mean_high")
    max_clipped = params.get("max_clipped_pct")

    details: dict[str, Any] = {
        "mean": mean,
        "clipped_pct": clipped_pct,
        "crushed_pct": crushed_pct,
    }
    reasons: list[str] = []
    if low is not None and mean < low:
        reasons.append(f"mean {mean:.1f} below {low}")
    if high is not None and mean > high:
        reasons.append(f"mean {mean:.1f} above {high}")
    if max_clipped is not None and clipped_pct > max_clipped:
        reasons.append(f"clipped {clipped_pct:.2f}% above {max_clipped}%")

    if reasons:
        details["reason"] = "; ".join(reasons)
    result = make_result("brightness", mean, "gray", low, high, details)
    return replace(result, passed=False) if reasons else result
