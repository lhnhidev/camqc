"""Shared pytest fixtures."""

from __future__ import annotations

import importlib.util
from collections.abc import Callable
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

TOOL_PATH = Path(__file__).resolve().parent.parent / "tools" / "gen_samples.py"


def load_gen_samples() -> ModuleType:
    """Import tools/gen_samples.py (it is not part of the installed package)."""
    spec = importlib.util.spec_from_file_location("gen_samples", TOOL_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="session")
def make_sample() -> Callable[[str], np.ndarray]:
    """Return ``make(name)``: the full-size sample image exactly as the tool writes it (seed 42).

    Images are generated once per session and returned read-only; copy before mutating.
    """
    gs = load_gen_samples()
    cache: dict[str, np.ndarray] = {}

    def make(name: str) -> np.ndarray:
        if name not in cache:
            img = gs.gen_sample(name, seed=42)
            img.setflags(write=False)
            cache[name] = img
        return cache[name]

    return make
