"""Shared pytest fixtures."""

from __future__ import annotations

import importlib.util
import os
from collections.abc import Callable
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

# Run any Qt widget tests headless (must be set before PySide6 creates a QApplication).
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

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


@pytest.fixture(scope="session")
def qapp():  # noqa: ANN201 - yields a QApplication; PySide6 is imported lazily
    """The single QApplication for GUI tests (skips them if PySide6 is missing)."""
    pytest.importorskip("PySide6")
    import gc

    from PySide6.QtCore import QCoreApplication, QEvent
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    yield app
    # Destroy every widget while the QApplication is still alive; otherwise Qt can abort
    # at interpreter exit when leftover widgets outlive the application object.
    for widget in app.topLevelWidgets():
        widget.close()
        widget.deleteLater()
    app.processEvents()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete.value)
    gc.collect()
