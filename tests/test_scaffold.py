"""Smoke test: every scaffolded module is importable."""

import importlib

import pytest

MODULES = [
    "camqc",
    "camqc.pipeline",
    "camqc.db",
    "camqc.report",
    "camqc.cli",
    "camqc.checks",
    "camqc.checks.base",
    "camqc.checks.brightness",
    "camqc.checks.sharpness",
    "camqc.checks.pixels",
    "camqc.checks.shading",
    "camqc.checks.color",
    "camqc.checks.noise",
    "camqc.checks.blemish",
    "camqc.gui",
    "camqc.gui.main_window",
    "camqc.gui.history_tab",
    "camqc.gui.stats_tab",
]


@pytest.mark.parametrize("name", MODULES)
def test_module_importable(name: str) -> None:
    """Each module imports and has a docstring."""
    module = importlib.import_module(name)
    assert module.__doc__
