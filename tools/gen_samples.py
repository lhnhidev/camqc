"""Generate synthetic sample images (good and with injected defects) for development and tests.

Writes 1920x1080 BGR PNGs plus ``manifest.json`` (filename -> expected defect) into an
output directory (default: ``<project>/samples``). Output is deterministic for a given seed.

Usage:
    python tools/gen_samples.py [--seed 42] [--out samples]
"""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Callable, Sequence
from pathlib import Path
from types import MappingProxyType

import cv2
import numpy as np

logger = logging.getLogger(__name__)

WIDTH, HEIGHT = 1920, 1080
DEFAULT_OUT = Path(__file__).resolve().parent.parent / "samples"

Size = tuple[int, int]  # (height, width)
SampleFn = Callable[[np.random.Generator, Size], np.ndarray]


# --- primitives (pure, float32 HxW or HxWx3) -------------------------------------------


def flat(value: float, size: Size) -> np.ndarray:
    """Return a uniform single-channel float image of the given value."""
    return np.full(size, value, dtype=np.float32)


def vignette_mask(size: Size, corner_falloff: float) -> np.ndarray:
    """Return a radial gain mask: 1.0 at the center, ``1 - corner_falloff`` at the corners.

    Gain falls off quadratically with distance normalized so the corners have radius 1.
    """
    h, w = size
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    r2 = ((x - (w - 1) / 2) / ((w - 1) / 2)) ** 2 + ((y - (h - 1) / 2) / ((h - 1) / 2)) ** 2
    return 1.0 - corner_falloff * (r2 / 2.0)


def add_noise(img: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    """Return ``img`` plus zero-mean Gaussian noise with standard deviation ``sigma``."""
    return img + rng.normal(0.0, sigma, img.shape).astype(np.float32)


def random_pixels(rng: np.random.Generator, size: Size, count: int, margin: int = 0) -> np.ndarray:
    """Return ``count`` unique (row, col) positions at least ``margin`` px from the border."""
    h, w = size
    inner_h, inner_w = h - 2 * margin, w - 2 * margin
    if inner_h <= 0 or inner_w <= 0 or count > inner_h * inner_w:
        raise ValueError(f"cannot place {count} pixels in {size} with margin {margin}")
    flat_idx = rng.choice(inner_h * inner_w, size=count, replace=False)
    rows, cols = np.divmod(flat_idx, inner_w)
    return np.stack([rows + margin, cols + margin], axis=1)


def set_pixels(img: np.ndarray, positions: np.ndarray, value: float) -> np.ndarray:
    """Return a copy of ``img`` with the given (row, col) positions set to ``value``."""
    out = img.copy()
    out[positions[:, 0], positions[:, 1]] = value
    return out


def checker_chart(size: Size, square: int = 60, angle_deg: float = 5.0,
                  low: float = 30.0, high: float = 220.0) -> np.ndarray:
    """Return a checkerboard rotated by ``angle_deg`` (slanted edges, hard transitions)."""
    h, w = size
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    a = np.deg2rad(angle_deg)
    u = x * np.cos(a) - y * np.sin(a)
    v = x * np.sin(a) + y * np.cos(a)
    parity = (np.floor(u / square) + np.floor(v / square)) % 2
    return np.where(parity == 0, low, high).astype(np.float32)


def dust_blobs(img: np.ndarray, rng: np.random.Generator, count: int,
               radius_range: tuple[int, int] = (6, 20),
               depth_range: tuple[float, float] = (0.10, 0.25)) -> np.ndarray:
    """Return ``img`` darkened by ``count`` soft circular blobs simulating sensor dust.

    Each blob has a smooth (cosine) falloff from ``depth`` darker at its center to 0 at its radius.
    """
    h, w = img.shape[:2]
    margin = min(radius_range[1] * 2, h // 4, w // 4)
    centers = random_pixels(rng, (h, w), count, margin=margin)
    gain = np.ones((h, w), dtype=np.float32)
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    for (cy, cx) in centers:
        radius = rng.uniform(*radius_range)
        depth = rng.uniform(*depth_range)
        d = np.hypot(x - cx, y - cy) / radius
        profile = np.where(d < 1.0, 0.5 * (1 + np.cos(np.pi * np.clip(d, 0, 1))), 0.0)
        gain *= 1.0 - depth * profile.astype(np.float32)
    return img * gain


def to_bgr_u8(img: np.ndarray) -> np.ndarray:
    """Clip a float image to [0, 255] and return it as uint8 BGR (gray is replicated)."""
    out = np.clip(np.rint(img), 0, 255).astype(np.uint8)
    return out if out.ndim == 3 else cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)


# --- samples ---------------------------------------------------------------------------


def gen_good_flat(rng: np.random.Generator, size: Size) -> np.ndarray:
    """Gray ~128, mild noise (sigma 2), corners ~10% darker."""
    return to_bgr_u8(add_noise(flat(128, size) * vignette_mask(size, 0.10), 2.0, rng))


def gen_dark_good(rng: np.random.Generator, size: Size) -> np.ndarray:
    """Near-black dark frame: mean ~2, noise sigma 0.8."""
    return to_bgr_u8(add_noise(flat(2, size), 0.8, rng))


def gen_dark_bad(rng: np.random.Generator, size: Size) -> np.ndarray:
    """Dark frame with 15 hot pixels (~200)."""
    base = add_noise(flat(2, size), 0.8, rng)
    return to_bgr_u8(set_pixels(base, random_pixels(rng, size, 15), 200))


def gen_white_bad(rng: np.random.Generator, size: Size) -> np.ndarray:
    """Bright flat frame (~200) with 12 dead pixels (~0)."""
    base = add_noise(flat(200, size), 2.0, rng)
    return to_bgr_u8(set_pixels(base, random_pixels(rng, size, 12), 0))


def gen_sharp(rng: np.random.Generator, size: Size) -> np.ndarray:
    """Slanted checkerboard chart with hard edges."""
    return to_bgr_u8(add_noise(checker_chart(size), 1.0, rng))


def gen_blurry(rng: np.random.Generator, size: Size) -> np.ndarray:
    """Same chart as ``sharp`` with Gaussian blur sigma 6."""
    blurred = cv2.GaussianBlur(checker_chart(size), (0, 0), sigmaX=6.0)
    return to_bgr_u8(add_noise(blurred, 1.0, rng))


def gen_dirty(rng: np.random.Generator, size: Size) -> np.ndarray:
    """Flat gray with 6 soft dark dust blobs (radius 6-20 px, 10-25% darker)."""
    return to_bgr_u8(add_noise(dust_blobs(flat(128, size), rng, 6), 2.0, rng))


def gen_color_cast(rng: np.random.Generator, size: Size) -> np.ndarray:
    """Flat gray with a blue tint (B channel +25)."""
    bgr = np.dstack([flat(128 + 25, size), flat(128, size), flat(128, size)])
    return to_bgr_u8(add_noise(bgr, 2.0, rng))


def gen_too_dark(rng: np.random.Generator, size: Size) -> np.ndarray:
    """Underexposed flat frame, mean ~20."""
    return to_bgr_u8(add_noise(flat(20, size), 2.0, rng))


def gen_overexposed(rng: np.random.Generator, size: Size) -> np.ndarray:
    """Overexposed flat frame, mean ~245 with many pixels clipped at 255."""
    return to_bgr_u8(add_noise(flat(247, size), 8.0, rng))


def gen_strong_vignette(rng: np.random.Generator, size: Size) -> np.ndarray:
    """Gray ~128 with corners ~45% darker."""
    return to_bgr_u8(add_noise(flat(128, size) * vignette_mask(size, 0.45), 2.0, rng))


SAMPLES: MappingProxyType[str, tuple[SampleFn, str]] = MappingProxyType({
    "good_flat.png": (gen_good_flat, "none"),
    "dark_good.png": (gen_dark_good, "none"),
    "dark_bad.png": (gen_dark_bad, "hot_pixels"),
    "white_bad.png": (gen_white_bad, "dead_pixels"),
    "sharp.png": (gen_sharp, "none"),
    "blurry.png": (gen_blurry, "blur"),
    "dirty.png": (gen_dirty, "blemish"),
    "color_cast.png": (gen_color_cast, "color_cast"),
    "too_dark.png": (gen_too_dark, "underexposed"),
    "overexposed.png": (gen_overexposed, "overexposed"),
    "strong_vignette.png": (gen_strong_vignette, "vignetting"),
})


def build_manifest() -> dict[str, str]:
    """Return the filename -> expected defect mapping."""
    return {name: defect for name, (_, defect) in SAMPLES.items()}


def gen_sample(name: str, seed: int = 42, size: Size = (HEIGHT, WIDTH)) -> np.ndarray:
    """Return the image for sample ``name`` exactly as :func:`generate_all` writes it.

    Each image gets its own RNG derived from ``seed`` and the sample's index, so one
    image's output does not depend on which others are generated.
    """
    idx = list(SAMPLES).index(name)
    return SAMPLES[name][0](np.random.default_rng([seed, idx]), size)


def generate_all(out_dir: Path, seed: int, size: Size = (HEIGHT, WIDTH)) -> dict[str, str]:
    """Write every sample PNG and ``manifest.json`` into ``out_dir``; return the manifest.

    Images are produced by :func:`gen_sample`.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in SAMPLES:
        path = out_dir / name
        if not cv2.imwrite(str(path), gen_sample(name, seed, size)):
            raise OSError(f"failed to write {path}")
        logger.info("wrote %s", path)
    manifest = build_manifest()
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def format_table(manifest: dict[str, str]) -> str:
    """Return the manifest as a two-column text table."""
    width = max(len(n) for n in manifest)
    lines = [f"{'filename':<{width}}  expected defect", f"{'-' * width}  {'-' * 15}"]
    lines += [f"{name:<{width}}  {defect}" for name, defect in manifest.items()]
    return "\n".join(lines)


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Generate synthetic CamQC sample images.")
    parser.add_argument("--seed", type=int, default=42, help="random seed (default: 42)")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="output directory")
    parser.add_argument("-v", "--verbose", action="store_true", help="log each written file")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point: generate samples and print the filename -> defect table."""
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s")
    manifest = generate_all(args.out, args.seed)
    print(format_table(manifest))
    print(f"\n{len(manifest)} images + manifest.json written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
