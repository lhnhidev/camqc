"""Tests for tools/gen_samples.py: files, manifest, determinism, and per-image statistics."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import cv2
import numpy as np
import pytest

TOOL_PATH = Path(__file__).resolve().parent.parent / "tools" / "gen_samples.py"


def _load_tool() -> ModuleType:
    spec = importlib.util.spec_from_file_location("gen_samples", TOOL_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


gs = _load_tool()


@pytest.fixture(scope="module")
def samples(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Generate the full sample set once at full resolution."""
    out = tmp_path_factory.mktemp("samples")
    assert gs.main(["--out", str(out), "--seed", "42"]) == 0
    return out


def _gray(samples: Path, name: str) -> np.ndarray:
    img = cv2.imread(str(samples / name), cv2.IMREAD_GRAYSCALE)
    assert img is not None
    return img.astype(np.float64)


def _center_corner_ratio(img: np.ndarray, patch: int = 40) -> float:
    h, w = img.shape
    center = img[h // 2 - patch // 2: h // 2 + patch // 2, w // 2 - patch // 2: w // 2 + patch // 2]
    corners = [img[:patch, :patch], img[:patch, -patch:], img[-patch:, :patch], img[-patch:, -patch:]]
    return float(np.mean([c.mean() for c in corners]) / center.mean())


def test_all_files_written_with_correct_shape(samples: Path) -> None:
    manifest = json.loads((samples / "manifest.json").read_text(encoding="utf-8"))
    assert manifest == gs.build_manifest()
    assert len(manifest) == 11
    for name in manifest:
        img = cv2.imread(str(samples / name), cv2.IMREAD_UNCHANGED)
        assert img is not None, name
        assert img.shape == (1080, 1920, 3), name
        assert img.dtype == np.uint8


def test_seed_is_deterministic(tmp_path: Path) -> None:
    size = (60, 80)
    a = gs.generate_all(tmp_path / "a", seed=7, size=size)
    b = gs.generate_all(tmp_path / "b", seed=7, size=size)
    c = gs.generate_all(tmp_path / "c", seed=8, size=size)
    assert a == b == c
    def img(d: str) -> np.ndarray:
        return cv2.imread(str(tmp_path / d / "dark_bad.png"))

    assert np.array_equal(img("a"), img("b"))
    assert not np.array_equal(img("a"), img("c"))


def test_good_flat(samples: Path) -> None:
    img = _gray(samples, "good_flat.png")
    h, w = img.shape
    assert img[h // 2 - 20: h // 2 + 20, w // 2 - 20: w // 2 + 20].mean() == pytest.approx(128, abs=1.5)
    assert _center_corner_ratio(img) == pytest.approx(0.90, abs=0.03)


def test_strong_vignette(samples: Path) -> None:
    assert _center_corner_ratio(_gray(samples, "strong_vignette.png")) == pytest.approx(0.55, abs=0.04)


def test_dark_frames_and_hot_pixels(samples: Path) -> None:
    good = _gray(samples, "dark_good.png")
    bad = _gray(samples, "dark_bad.png")
    assert good.mean() == pytest.approx(2, abs=0.3)
    assert good.max() < 20
    assert int((bad > 150).sum()) == 15


def test_white_bad_dead_pixels(samples: Path) -> None:
    img = _gray(samples, "white_bad.png")
    assert img.mean() == pytest.approx(200, abs=1)
    assert int((img < 50).sum()) == 12


def test_blurry_is_much_less_sharp(samples: Path) -> None:
    def lap(name: str) -> float:
        return float(cv2.Laplacian(_gray(samples, name), cv2.CV_64F).var())

    assert lap("sharp.png") > 20 * lap("blurry.png")


def test_dirty_has_dark_blobs(samples: Path) -> None:
    smooth = cv2.GaussianBlur(_gray(samples, "dirty.png"), (0, 0), 3)
    assert smooth.min() < 128 * 0.92  # at least one blob >= ~8% darker after smoothing
    assert np.median(smooth) == pytest.approx(128, abs=1)


def test_color_cast_blue(samples: Path) -> None:
    b, g, r = cv2.split(cv2.imread(str(samples / "color_cast.png")).astype(np.float64))
    assert b.mean() - g.mean() == pytest.approx(25, abs=1)
    assert g.mean() == pytest.approx(r.mean(), abs=0.5)


def test_exposure_extremes(samples: Path) -> None:
    assert _gray(samples, "too_dark.png").mean() == pytest.approx(20, abs=1)
    over = _gray(samples, "overexposed.png")
    assert over.mean() == pytest.approx(245, abs=2)
    assert (over >= 255).mean() > 0.1  # many clipped pixels (~17% with mean 247, sigma 8)


def test_random_pixels_rejects_impossible_request() -> None:
    with pytest.raises(ValueError):
        gs.random_pixels(np.random.default_rng(0), (10, 10), 5, margin=5)


def test_format_table_lists_every_file() -> None:
    table = gs.format_table(gs.build_manifest())
    for name, defect in gs.build_manifest().items():
        assert any(name in line and defect in line for line in table.splitlines())
