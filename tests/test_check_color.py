"""Tests for the color check."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from camqc.checks import color

PARAMS = {"ratio_low": 0.9, "ratio_high": 1.1, "max_delta_e": 10}


def gray_frame(value: int = 128, shape: tuple[int, int] = (160, 160)) -> np.ndarray:
    return np.full((*shape, 3), value, np.uint8)


def test_good_flat_passes(make_sample: Callable[[str], np.ndarray]) -> None:
    result = color.run(make_sample("good_flat.png"), PARAMS)
    assert result.passed and "reason" not in result.details
    assert result.name == "color" and result.unit == "dE76"
    assert result.details["r_g"] == pytest.approx(1.0, abs=0.01)
    assert result.details["b_g"] == pytest.approx(1.0, abs=0.01)
    assert 0 < result.value < 10 and result.high == 10


def test_color_cast_fails_on_blue_ratio(make_sample: Callable[[str], np.ndarray]) -> None:
    result = color.run(make_sample("color_cast.png"), PARAMS)
    assert not result.passed
    assert result.details["b_g"] == pytest.approx(153 / 128, abs=0.01)
    assert result.details["r_g"] == pytest.approx(1.0, abs=0.01)
    assert "B/G" in result.details["reason"] and "R/G" not in result.details["reason"]
    assert result.value < 10  # a global cast is uniform, so delta E alone would miss it


def test_uniform_gray_has_zero_delta_e() -> None:
    result = color.run(gray_frame(), PARAMS)
    assert result.passed
    assert result.value == pytest.approx(0.0, abs=1e-4)
    assert result.details["r_g"] == pytest.approx(1.0) and result.details["b_g"] == pytest.approx(1.0)
    assert len(result.details["delta_e"]) == 64


def test_tinted_cell_is_worst_cell_and_fails_delta_e() -> None:
    img = gray_frame()
    img[40:60, 100:120, 2] = 255  # cell row 2, col 5 (20x20 px cells): red patch
    result = color.run(img, {"max_delta_e": 5})
    assert result.details["worst_cell"] == 2 * 8 + 5
    assert result.details["worst_cell_rc"] == [2, 5]
    assert not result.passed and "cell 21" in result.details["reason"]
    assert result.value > 5
    assert result.overlay == [{"type": "rect", "x": 100, "y": 40, "w": 20, "h": 20}]


def test_reference_is_center_not_global() -> None:
    img = gray_frame(100)
    img[:60, :60] = 200  # cells (0..2, 0..2) brighter; the center cells (3..4, 3..4) stay at 100
    result = color.run(img, {})
    delta = result.details["delta_e"]
    assert delta[0] > 25  # bright corner cell is far from the center
    assert delta[3 * 8 + 3] == pytest.approx(0.0, abs=1e-4)  # center cells match the reference
    assert delta[4 * 8 + 4] == pytest.approx(0.0, abs=1e-4)
    assert result.details["worst_cell"] in {r * 8 + c for r in range(3) for c in range(3)}


@pytest.mark.parametrize(
    ("bgr", "reason_has"),
    [((128, 128, 150), "R/G"), ((100, 128, 128), "B/G")],
)
def test_each_ratio_is_checked(bgr: tuple[int, int, int], reason_has: str) -> None:
    img = np.empty((64, 64, 3), np.uint8)
    img[:] = bgr
    result = color.run(img, {"ratio_low": 0.9, "ratio_high": 1.1})
    assert not result.passed and reason_has in result.details["reason"]


def test_ratio_bounds_are_inclusive_and_missing_params_disable() -> None:
    img = np.empty((64, 64, 3), np.uint8)
    img[:] = (128, 128, 140)  # R/G = 1.09375
    assert color.run(img, {"ratio_low": 1.0, "ratio_high": 1.09375}).passed
    assert not color.run(img, {"ratio_low": 1.0, "ratio_high": 1.09}).passed
    assert color.run(img, {}).passed


def test_zero_green_fails_when_ratio_limited() -> None:
    img = np.zeros((64, 64, 3), np.uint8)
    img[..., 2] = 100
    result = color.run(img, {"ratio_low": 0.9, "ratio_high": 1.1})
    assert not result.passed and "R/G" in result.details["reason"]
    assert result.details["ratio_low"] == 0.9
    assert color.run(img, {}).passed


def test_accepts_gray_input() -> None:
    assert color.run(np.full((64, 64), 128, np.uint8), PARAMS).passed


def test_uneven_size_grid_is_covered() -> None:
    edges = color.grid_edges(1080)
    assert edges[0] == 0 and edges[-1] == 1080 and len(edges) == 9
    assert color.run(gray_frame(shape=(77, 101)), PARAMS).passed


def test_tiny_image_raises_value_error() -> None:
    with pytest.raises(ValueError):
        color.run(np.zeros((7, 64, 3), np.uint8), PARAMS)
