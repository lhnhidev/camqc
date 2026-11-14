"""Tests for the lens shading check."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from camqc.checks import shading

PARAMS = {"min_corner_pct": 80}


def test_good_flat_passes(make_sample: Callable[[str], np.ndarray]) -> None:
    result = shading.run(make_sample("good_flat.png"), PARAMS)
    assert result.passed and "reason" not in result.details
    assert 85 < result.value < 100
    assert result.name == "shading" and result.unit == "%" and result.low == 80


def test_strong_vignette_fails(make_sample: Callable[[str], np.ndarray]) -> None:
    result = shading.run(make_sample("strong_vignette.png"), PARAMS)
    assert not result.passed
    assert result.value < 80
    assert "below 80" in result.details["reason"]


def test_value_is_min_corner_over_center() -> None:
    img = np.full((100, 200, 3), 100, np.uint8)
    img[:10, :20] = 60  # top-left patch (10% x 10%) darker
    result = shading.run(img, PARAMS)
    d = result.details
    assert d["center"] == 100.0
    assert d["corners"] == {"top_left": 60.0, "top_right": 100.0, "bottom_left": 100.0, "bottom_right": 100.0}
    assert result.value == pytest.approx(60.0)
    assert d["min_corner"] == 60.0 and d["min_corner_name"] == "top_left"
    assert not result.passed


def test_value_can_exceed_100_when_corners_brighter() -> None:
    img = np.full((100, 100), 100, np.uint8)
    img[45:55, 45:55] = 50
    assert shading.run(img, PARAMS).value == pytest.approx(200.0)


def test_threshold_edge_is_inclusive() -> None:
    img = np.full((100, 100), 100, np.uint8)
    img[:10, :10] = 80
    assert shading.run(img, {"min_corner_pct": 80}).passed
    assert not shading.run(img, {"min_corner_pct": 80.1}).passed


def test_missing_param_means_no_limit_and_black_center_fails_when_limited() -> None:
    assert shading.run(np.full((50, 50), 10, np.uint8), {}).passed
    black = np.zeros((50, 50), np.uint8)
    result = shading.run(black, PARAMS)
    assert result.value == 0.0 and not result.passed


def test_overlay_marks_five_patches() -> None:
    overlay = shading.run(np.full((100, 200), 128, np.uint8), PARAMS).overlay
    assert len(overlay) == 5
    assert {"type": "rect", "x": 90, "y": 45, "w": 20, "h": 10} in overlay
    assert {"type": "rect", "x": 180, "y": 90, "w": 20, "h": 10} in overlay


def test_tiny_image_raises_value_error() -> None:
    with pytest.raises(ValueError):
        shading.run(np.zeros((5, 5, 3), np.uint8), PARAMS)
