"""Tests for the brightness check."""

from __future__ import annotations

import numpy as np

from camqc.checks import brightness

PARAMS = {"mean_low": 100, "mean_high": 160, "max_clipped_pct": 1.0}


def solid(value: int, shape: tuple[int, int] = (100, 160)) -> np.ndarray:
    return np.full((*shape, 3), value, dtype=np.uint8)


def test_good_frame_passes() -> None:
    result = brightness.run(solid(128), PARAMS)
    assert result.name == "brightness" and result.unit == "gray"
    assert result.passed and "reason" not in result.details
    assert result.value == 128.0
    assert (result.low, result.high) == (100, 160)
    assert result.details["clipped_pct"] == 0.0 and result.details["crushed_pct"] == 0.0


def test_too_dark_fails_with_reason() -> None:
    result = brightness.run(solid(3), PARAMS)
    assert not result.passed
    assert "below" in result.details["reason"]
    assert not brightness.run(solid(20), PARAMS).passed
    assert result.details["crushed_pct"] == 100.0


def test_too_bright_fails_on_mean_and_clipping() -> None:
    result = brightness.run(solid(255), PARAMS)
    assert not result.passed
    reason = result.details["reason"]
    assert "above 160" in reason and "clipped" in reason
    assert result.details["clipped_pct"] == 100.0


def test_clipping_alone_fails_even_with_mean_in_range() -> None:
    img = solid(120)
    img[:10] = 255  # 10% rows clipped, mean stays in range
    result = brightness.run(img, PARAMS)
    assert 100 <= result.value <= 160
    assert not result.passed
    assert result.details["reason"].startswith("clipped")
    assert abs(result.details["clipped_pct"] - 10.0) < 1e-9


def test_threshold_edges_are_exclusive_for_clipping() -> None:
    img = solid(128)
    img.reshape(-1, 3)[: img.shape[0] * img.shape[1] // 100] = 255  # exactly 1%
    assert brightness.run(img, PARAMS).passed  # 1.0% is not above max 1.0
    assert not brightness.run(img, {**PARAMS, "max_clipped_pct": 0.5}).passed


def test_thresholds_are_strict_to_250_and_5() -> None:
    assert brightness.run(solid(250), {}).details["clipped_pct"] == 0.0
    assert brightness.run(solid(251), {}).details["clipped_pct"] == 100.0
    assert brightness.run(solid(5), {}).details["crushed_pct"] == 0.0
    assert brightness.run(solid(4), {}).details["crushed_pct"] == 100.0


def test_missing_params_disable_conditions() -> None:
    assert brightness.run(solid(255), {}).passed


def test_accepts_gray_input() -> None:
    gray = np.full((50, 50), 128, dtype=np.uint8)
    assert brightness.run(gray, PARAMS).passed


def test_uses_luminance_not_channel_mean() -> None:
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    img[..., 1] = 200  # pure green: luma ~ 0.587 * 200
    assert abs(brightness.run(img, {}).value - 117.4) < 1.0
