"""Tests for the defective pixel check."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from camqc.checks import pixels

HOT = {"mode": "hot", "diff_threshold": 50, "max_defects": 20}
DEAD = {"mode": "dead", "diff_threshold": 50, "max_defects": 5}


def test_dark_bad_finds_about_15_hot_pixels(make_sample: Callable[[str], np.ndarray]) -> None:
    result = pixels.run(make_sample("dark_bad.png"), {**HOT, "max_defects": 5})
    assert 13 <= result.value <= 17
    assert not result.passed and "hot" in result.details["reason"]
    assert result.details["count"] == result.value == len(result.details["defects"])
    assert result.unit == "count" and result.high == 5


def test_dark_good_passes(make_sample: Callable[[str], np.ndarray]) -> None:
    result = pixels.run(make_sample("dark_good.png"), HOT)
    assert result.passed and result.value == 0
    assert result.overlay == [] and "reason" not in result.details


def test_white_bad_fails_in_dead_mode(make_sample: Callable[[str], np.ndarray]) -> None:
    result = pixels.run(make_sample("white_bad.png"), DEAD)
    assert not result.passed
    assert 10 <= result.value <= 12  # 12 injected, adjacent ones may merge


def test_mode_selects_direction(make_sample: Callable[[str], np.ndarray]) -> None:
    assert pixels.run(make_sample("dark_bad.png"), {**DEAD, "max_defects": 0}).value == 0
    assert pixels.run(make_sample("white_bad.png"), {**HOT, "max_defects": 0}).value == 0


def test_coordinates_and_overlay() -> None:
    img = np.zeros((100, 120, 3), np.uint8)
    img[30, 40] = 255
    img[70, 90] = 255
    result = pixels.run(img, {"mode": "hot", "diff_threshold": 50, "max_defects": 5})
    assert result.details["defects"] == [(40, 30), (90, 70)]
    assert result.overlay == [
        {"type": "circle", "x": 40, "y": 30, "r": 8},
        {"type": "circle", "x": 90, "y": 70, "r": 8},
    ]


def test_adjacent_pixels_count_once() -> None:
    img = np.zeros((50, 50), np.uint8)
    img[20, 20] = img[20, 21] = 255  # 2-px cluster: median ignores it, one component
    assert pixels.run(img, {"mode": "hot"}).value == 1


def test_cap_at_200_listed_but_count_is_total() -> None:
    img = np.zeros((200, 200), np.uint8)
    img[4::8, 4::8] = 255  # 25 x 25 = 625 isolated hot pixels
    result = pixels.run(img, {"mode": "hot", "max_defects": 10})
    assert result.value == 625
    assert len(result.details["defects"]) == 200 and len(result.overlay) == 200
    assert result.details["truncated"] is True


def test_no_max_defects_means_unbounded() -> None:
    img = np.zeros((50, 50), np.uint8)
    img[25, 25] = 255
    result = pixels.run(img, {"mode": "hot"})
    assert result.passed and result.value == 1 and result.high is None


@pytest.mark.parametrize("params", [{}, {"mode": "warm"}, {"mode": None}])
def test_invalid_mode_raises(params: dict) -> None:
    with pytest.raises(ValueError):
        pixels.run(np.zeros((20, 20), np.uint8), params)
