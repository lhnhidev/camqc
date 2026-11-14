"""Run the configured checks for a camera model on an image and aggregate an overall PASS/FAIL verdict.

``load_config`` validates ``config/models.yaml``; ``run_pipeline`` runs every enabled
check of a model against the supplied frames and never lets one check crash the run.

Verdict rules:
    * ``FAIL``       - at least one check failed.
    * ``INCOMPLETE`` - nothing failed, but some enabled check was SKIPPED (frame
      missing) or ERROR (exception / not implemented), so the unit was not fully verified.
    * ``PASS``       - every enabled check ran and passed.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from camqc.checks import CHECK_MODULES, get_runner
from camqc.checks.base import CheckResult

logger = logging.getLogger(__name__)

FRAME_TYPES = ("normal", "flat", "dark", "white")

PASS, FAIL, SKIPPED, ERROR = "PASS", "FAIL", "SKIPPED", "ERROR"
INCOMPLETE = "INCOMPLETE"

_CHECK_KEYS = {"check", "enabled", "frame", "params"}
_MODEL_KEYS = {"description", "checks"}


class ConfigError(ValueError):
    """Raised when the configuration file is missing, unreadable or invalid."""


@dataclass(frozen=True)
class CheckSpec:
    """One configured check instance.

    Attributes:
        name: Unique key of this instance within the model (used in reports).
        check: Check module name in the registry (e.g. ``"pixels"``).
        enabled: Disabled specs are ignored by the pipeline.
        frames: Acceptable frame types; the first one supplied is used.
        params: Parameters passed to the check's ``run``.
    """

    name: str
    check: str
    enabled: bool
    frames: tuple[str, ...]
    params: Mapping[str, Any]


@dataclass(frozen=True)
class ModelConfig:
    """Configuration of one camera model."""

    name: str
    description: str
    checks: tuple[CheckSpec, ...]


@dataclass(frozen=True)
class Config:
    """All camera models from the configuration file."""

    models: Mapping[str, ModelConfig]


@dataclass(frozen=True)
class CheckOutcome:
    """What happened to one enabled check in a run.

    Attributes:
        name: The check instance name from the config.
        status: ``PASS``, ``FAIL``, ``SKIPPED`` or ``ERROR``.
        result: The check's result for PASS/FAIL, else ``None``.
        reason: Why it was skipped or errored, else ``None`` (a FAIL reason is in
            ``result.details["reason"]``).
        frame: Frame type used, or ``None`` if skipped.
        duration_ms: Time spent in the check.
    """

    name: str
    status: str
    result: CheckResult | None
    reason: str | None
    frame: str | None
    duration_ms: float


@dataclass(frozen=True)
class RunResult:
    """Outcome of running a model's checks.

    Attributes:
        verdict: ``PASS``, ``FAIL`` or ``INCOMPLETE``.
        results: One outcome per enabled check, in config order.
        failed_checks: Names of checks whose status is ``FAIL``.
        duration_ms: Total wall time of the run.
    """

    verdict: str
    results: tuple[CheckOutcome, ...]
    failed_checks: tuple[str, ...]
    duration_ms: float


# --- configuration ---------------------------------------------------------------------


def _mapping(value: Any, where: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or not value:
        raise ConfigError(f"{where}: expected a non-empty mapping")
    return value


def _unknown_keys(data: Mapping[str, Any], allowed: set[str], where: str) -> None:
    extra = sorted(str(k) for k in data if k not in allowed)
    if extra:
        raise ConfigError(f"{where}: unknown key(s) {extra}; allowed: {sorted(allowed)}")


def _parse_frames(value: Any, where: str) -> tuple[str, ...]:
    frames = (value,) if isinstance(value, str) else tuple(value) if isinstance(value, list) else ()
    if not frames or any(f not in FRAME_TYPES for f in frames):
        raise ConfigError(f"{where}.frame: must be one of {list(FRAME_TYPES)} or a non-empty list of them, got {value!r}")
    return frames


def _parse_check(name: str, data: Any, where: str) -> CheckSpec:
    if not isinstance(data, Mapping):
        raise ConfigError(f"{where}: expected a mapping, got {type(data).__name__}")
    _unknown_keys(data, _CHECK_KEYS, where)
    if "frame" not in data:
        raise ConfigError(f"{where}: missing required key 'frame'")
    check = data.get("check", name)
    if check not in CHECK_MODULES:
        raise ConfigError(f"{where}: unknown check {check!r}; known: {sorted(CHECK_MODULES)}")
    enabled = data.get("enabled", True)
    if not isinstance(enabled, bool):
        raise ConfigError(f"{where}.enabled: must be true or false, got {enabled!r}")
    params = data.get("params", {})
    if not isinstance(params, Mapping):
        raise ConfigError(f"{where}.params: expected a mapping, got {type(params).__name__}")
    return CheckSpec(name, check, enabled, _parse_frames(data["frame"], where), dict(params))


def _parse_model(name: str, data: Any) -> ModelConfig:
    where = f"models.{name}"
    if not isinstance(data, Mapping):
        raise ConfigError(f"{where}: expected a mapping, got {type(data).__name__}")
    _unknown_keys(data, _MODEL_KEYS, where)
    if "checks" not in data:
        raise ConfigError(f"{where}: missing required key 'checks'")
    checks = tuple(
        _parse_check(str(key), value, f"{where}.checks.{key}")
        for key, value in _mapping(data["checks"], f"{where}.checks").items()
    )
    if not any(c.enabled for c in checks):
        raise ConfigError(f"{where}.checks: at least one check must be enabled")
    return ModelConfig(name, str(data.get("description", "")), checks)


def parse_config(data: Any, source: str = "config") -> Config:
    """Validate already-loaded YAML data and build a :class:`Config`.

    Raises:
        ConfigError: With the path of the offending key, e.g.
            ``models.CAM_A.checks.sharpness: missing required key 'frame'``.
    """
    try:
        if not isinstance(data, Mapping) or "models" not in data:
            raise ConfigError("missing required top-level key 'models'")
        _unknown_keys(data, {"models"}, "top level")
        models = {
            str(name): _parse_model(str(name), value)
            for name, value in _mapping(data["models"], "models").items()
        }
    except ConfigError as exc:
        raise ConfigError(f"{source}: {exc}") from None
    return Config(models=models)


def load_config(path: str | Path) -> Config:
    """Read and validate a models YAML file.

    Raises:
        ConfigError: If the file is missing, is not valid YAML, or fails validation.
    """
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError(f"{path}: cannot read config file ({exc.strerror or exc})") from exc
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ConfigError(f"{path}: invalid YAML: {exc}") from exc
    return parse_config(data, source=str(path))


# --- running ---------------------------------------------------------------------------


def _elapsed_ms(start: float) -> float:
    return (time.perf_counter() - start) * 1000.0


def _run_check(spec: CheckSpec, frames: Mapping[str, np.ndarray]) -> CheckOutcome:
    """Run one check and convert every outcome (including exceptions) to a CheckOutcome."""
    start = time.perf_counter()
    frame_type = next((f for f in spec.frames if frames.get(f) is not None), None)
    if frame_type is None:
        reason = f"required frame missing (need one of: {', '.join(spec.frames)})"
        return CheckOutcome(spec.name, SKIPPED, None, reason, None, _elapsed_ms(start))
    try:
        runner = get_runner(spec.check)
        if runner is None:
            raise NotImplementedError(f"check {spec.check!r} is not implemented")
        result = runner(frames[frame_type], dict(spec.params))
    except Exception as exc:  # noqa: BLE001 - one broken check must not crash the run
        logger.exception("check %s failed with an exception", spec.name)
        reason = f"{type(exc).__name__}: {exc}"
        return CheckOutcome(spec.name, ERROR, None, reason, frame_type, _elapsed_ms(start))
    status = PASS if result.passed else FAIL
    return CheckOutcome(spec.name, status, result, None, frame_type, _elapsed_ms(start))


def decide_verdict(outcomes: tuple[CheckOutcome, ...]) -> str:
    """Return FAIL if any check failed, else INCOMPLETE if any was skipped/errored, else PASS."""
    statuses = {o.status for o in outcomes}
    if FAIL in statuses:
        return FAIL
    if statuses & {SKIPPED, ERROR}:
        return INCOMPLETE
    return PASS


def run_pipeline(model: str, frames: Mapping[str, np.ndarray], config: Config) -> RunResult:
    """Run all enabled checks of ``model`` against ``frames``.

    Args:
        model: Camera model name as in the config (e.g. ``"CAM_A"``).
        frames: Maps frame type (``normal``/``flat``/``dark``/``white``) to a BGR image.
        config: Loaded configuration.

    Returns:
        A :class:`RunResult`. A check whose frame is missing is SKIPPED; a check
        that raises is ERROR; neither stops the other checks.

    Raises:
        ValueError: If ``model`` is not in the config.
    """
    if model not in config.models:
        raise ValueError(f"unknown model {model!r}; available: {sorted(config.models)}")
    start = time.perf_counter()
    outcomes = tuple(_run_check(spec, frames) for spec in config.models[model].checks if spec.enabled)
    return RunResult(
        verdict=decide_verdict(outcomes),
        results=outcomes,
        failed_checks=tuple(o.name for o in outcomes if o.status == FAIL),
        duration_ms=_elapsed_ms(start),
    )
