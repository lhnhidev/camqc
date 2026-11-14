"""Tests for the blemish check."""

from __future__ import annotations

from collections.abc import Callable

import cv2
import numpy as np
import pytest

from camqc.checks import blemish

PARAMS = {"contrast": 0.05, "min_area": 30, "max_blobs": 2}


def flat(value: int = 128, shape: tuple[int, int] = (200, 400)) -> np.ndarray:
    return np.full((*shape, 3), value, np.uint8)


def with_disc(img: np.ndarray, x: int, y: int, r: int, scale: float) -> np.ndarray:
    out = img.copy()
    cv2.circle(out, (x, y), r, tuple(int(c * scale) for c in img[0, 0]), -1)
    return out


def test_dirty_has_six_blobs_and_fails(make_sample: Callable[[str], np.ndarray]) -> None:
    result = blemish.run(make_sample("dirty.png"), PARAMS)
    assert 5 <= result.value <= 7
    assert not result.passed and "blemishes above max 2" in result.details["reason"]
    assert result.details["count"] == result.value == len(result.details["blobs"])
    assert result.name == "blemish" and result.unit == "count" and result.high == 2
    assert len(result.overlay) == result.value
    assert all(b["contrast"] > 0.05 and b["area"] >= 30 for b in result.details["blobs"])


def test_good_flat_passes(make_sample: Callable[[str], np.ndarray]) -> None:
    result = blemish.run(make_sample("good_flat.png"), PARAMS)
    assert result.passed and result.value == 0
    assert result.overlay == [] and "reason" not in result.details


def test_blob_position_area_and_contrast() -> None:
    img = with_disc(flat(), x=150, y=80, r=10, scale=0.8)  # 20% darker disc, area ~ 314
    result = blemish.run(img, {"max_blobs": 5})
    (blob,) = result.details["blobs"]
    assert abs(blob["x"] - 150) <= 1 and abs(blob["y"] - 80) <= 1
    assert blob["area"] == pytest.approx(314, rel=0.15)
    assert blob["contrast"] == pytest.approx(0.2, abs=0.03)
    (circle,) = result.overlay
    assert circle["type"] == "circle" and (circle["x"], circle["y"]) == (blob["x"], blob["y"])
    assert circle["r"] >= 10
    assert result.passed


def test_contrast_threshold_filters_faint_blobs() -> None:
    img = with_disc(flat(), 200, 100, 12, 0.93)  # 7% darker
    assert blemish.run(img, {"contrast": 0.05}).value == 1
    assert blemish.run(img, {"contrast": 0.10}).value == 0


def test_min_area_filters_small_blobs() -> None:
    img = with_disc(flat(), 200, 100, 5, 0.7)  # area ~ 80
    assert blemish.run(img, {"min_area": 30}).value == 1
    assert blemish.run(img, {"min_area": 200}).value == 0


def test_single_dead_pixel_is_removed_by_opening() -> None:
    img = flat()
    img[100, 200] = 0  # one dead pixel is not a blemish
    assert blemish.run(img, {"min_area": 1}).value == 0


def test_bright_spots_and_smooth_gradient_are_ignored() -> None:
    img = with_disc(flat(), 100, 100, 10, 1.4)  # brighter disc
    assert blemish.run(img, {}).value == 0
    ramp = np.tile(np.linspace(110, 140, 400, dtype=np.float32), (200, 1))
    gradient = np.dstack([ramp] * 3).astype(np.uint8)
    assert blemish.run(gradient, {}).value == 0


def test_max_blobs_boundary_and_unbounded() -> None:
    img = flat()
    for x in (80, 200, 320):
        img = with_disc(img, x, 100, 10, 0.8)
    assert blemish.run(img, {"max_blobs": 3}).passed
    assert not blemish.run(img, {"max_blobs": 2}).passed
    unbounded = blemish.run(img, {})
    assert unbounded.passed and unbounded.value == 3 and unbounded.high is None


def test_blobs_listed_largest_first_and_capped() -> None:
    img = with_disc(with_disc(flat(), 100, 100, 6, 0.8), 300, 100, 12, 0.8)
    areas = [b["area"] for b in blemish.run(img, {}).details["blobs"]]
    assert areas == sorted(areas, reverse=True) and len(areas) == 2

    big = flat(shape=(600, 1200))
    for r in range(20):  # 20 x 15 = 300 well-separated discs
        for c in range(15):
            big = with_disc(big, 40 + c * 80, 30 + r * 28, 5, 0.7)
    result = blemish.run(big, {"max_blobs": 10})
    assert result.value == 300 and len(result.details["blobs"]) == 200
    assert result.details["truncated"] is True and len(result.overlay) == 200


def test_accepts_gray_input() -> None:
    gray = np.full((200, 400), 128, np.uint8)
    assert blemish.run(gray, PARAMS).passed


def test_tiny_image_raises_value_error() -> None:
    with pytest.raises(ValueError):
        blemish.run(np.zeros((30, 30, 3), np.uint8), PARAMS)
