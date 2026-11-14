"""Tests for the sharpness check."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from camqc.checks import sharpness

H, W = 200, 300


def checker(square: int = 10) -> np.ndarray:
    y, x = np.mgrid[0:H, 0:W]
    gray = np.where(((x // square) + (y // square)) % 2 == 0, 30, 220).astype(np.uint8)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def test_regions_geometry() -> None:
    assert sharpness.center_region(200, 300) == (75, 50, 150, 100)
    corners = sharpness.corner_regions(200, 300)
    assert corners["top_left"] == (0, 0, 60, 40)
    assert corners["bottom_right"] == (240, 160, 60, 40)
    assert set(corners) == {"top_left", "top_right", "bottom_left", "bottom_right"}


def test_sharp_beats_blurry_and_flat_is_zero() -> None:
    sharp = sharpness.run(checker(), {})
    blurry = sharpness.run(cv2.GaussianBlur(checker(), (0, 0), 3), {})
    flat = sharpness.run(np.full((H, W, 3), 128, np.uint8), {})
    assert sharp.value > 20 * blurry.value > 0
    assert flat.value == 0.0
    assert set(sharp.details["corners"]) == {"top_left", "top_right", "bottom_left", "bottom_right"}
    assert sharp.details["center"] == sharp.value


def test_threshold_pass_fail_and_reason() -> None:
    img = cv2.GaussianBlur(checker(), (0, 0), 3)
    score = sharpness.run(img, {}).value
    ok = sharpness.run(img, {"min_center": score * 0.5})
    assert ok.passed and "reason" not in ok.details
    failed = sharpness.run(img, {"min_center": score * 2})
    assert not failed.passed
    assert "below" in failed.details["reason"]
    over = sharpness.run(img, {"max_center": score * 0.5})
    assert not over.passed and "above" in over.details["reason"]


def test_corner_scores_localize_blur() -> None:
    img = checker()
    img[:40, :60] = cv2.GaussianBlur(img, (0, 0), 4)[:40, :60]  # blur top-left only
    result = sharpness.run(img, {})
    corners = result.details["corners"]
    assert corners["top_left"] < 0.2 * corners["bottom_right"]
    assert abs(corners["top_right"] - corners["bottom_right"]) < 0.5 * corners["top_right"]


def test_tiny_image_raises_value_error() -> None:
    with pytest.raises(ValueError):
        sharpness.run(np.zeros((4, 4, 3), np.uint8), {})


def test_overlay_marks_five_regions() -> None:
    overlay = sharpness.run(checker(), {}).overlay
    assert len(overlay) == 5
    assert all(o["type"] == "rect" and {"x", "y", "w", "h"} <= set(o) for o in overlay)
