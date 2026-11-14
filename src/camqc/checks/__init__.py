"""Image quality checks and the registry mapping check names to implementations."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from types import MappingProxyType
from typing import Any

import numpy as np

from camqc.checks import blemish, brightness, color, noise, pixels, shading, sharpness
from camqc.checks.base import CheckResult

CheckFn = Callable[[np.ndarray, dict[str, Any]], CheckResult]

CHECK_MODULES: Mapping[str, Any] = MappingProxyType({
    "brightness": brightness,
    "sharpness": sharpness,
    "pixels": pixels,
    "shading": shading,
    "color": color,
    "noise": noise,
    "blemish": blemish,
})


def get_runner(name: str) -> CheckFn | None:
    """Return the ``run`` function of check ``name``.

    Returns:
        The function, or ``None`` if the module exists but has no ``run`` yet
        (not implemented).

    Raises:
        KeyError: If ``name`` is not a known check.
    """
    return getattr(CHECK_MODULES[name], "run", None)
