"""Tests for config loading/validation and run_pipeline verdicts."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest

from camqc import pipeline
from camqc.pipeline import (
    ERROR,
    FAIL,
    INCOMPLETE,
    PASS,
    SKIPPED,
    Config,
    ConfigError,
    load_config,
    parse_config,
    run_pipeline,
)

MODELS_YAML = Path(__file__).resolve().parent.parent / "config" / "models.yaml"
Sample = Callable[[str], np.ndarray]


@pytest.fixture(scope="module")
def config() -> Config:
    return load_config(MODELS_YAML)


@pytest.fixture(scope="module")
def white_good() -> np.ndarray:
    """Bright flat frame with mild noise and no dead pixels."""
    noise = np.random.default_rng(0).normal(200, 2, (1080, 1920, 1)).clip(0, 255)
    return np.repeat(noise, 3, axis=2).astype(np.uint8)


def good_frames(make_sample: Sample, white: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "normal": make_sample("sharp.png"),
        "flat": make_sample("good_flat.png"),
        "dark": make_sample("dark_good.png"),
        "white": white,
    }


def status_of(result: pipeline.RunResult) -> dict[str, str]:
    return {o.name: o.status for o in result.results}


def mini(checks: dict[str, Any]) -> Config:
    """Build a one-model config from a dict of check entries."""
    return parse_config({"models": {"M": {"checks": checks}}})


BRIGHT = {"frame": "flat", "params": {"mean_low": 50, "mean_high": 200}}
SHARP = {"frame": "normal", "params": {"min_center": 10}}


def small(value: int = 128, size: int = 120) -> np.ndarray:
    return np.full((size, size, 3), value, np.uint8)


def chart() -> np.ndarray:
    y, x = np.mgrid[0:120, 0:120]
    return cv2.cvtColor(np.where((x // 10 + y // 10) % 2 == 0, 30, 220).astype(np.uint8), cv2.COLOR_GRAY2BGR)


# --- shipped config ----------------------------------------------------------------------


def test_shipped_config_loads_and_matches_spec(config: Config) -> None:
    assert set(config.models) == {"CAM_A", "CAM_B"}
    for model in config.models.values():
        by_name = {c.name: c for c in model.checks}
        assert by_name["pixels_hot"].frames == ("dark",) and by_name["pixels_hot"].params["mode"] == "hot"
        assert by_name["pixels_dead"].frames == ("white",) and by_name["pixels_dead"].params["mode"] == "dead"
        assert by_name["pixels_hot"].check == by_name["pixels_dead"].check == "pixels"
        assert set(by_name["brightness"].frames) == {"normal", "flat"}
        assert not by_name["noise"].enabled
        assert all(c.enabled for n, c in by_name.items() if n != "noise")
    a = {c.name: c for c in config.models["CAM_A"].checks}
    b = {c.name: c for c in config.models["CAM_B"].checks}
    assert b["sharpness"].params["min_center"] > a["sharpness"].params["min_center"]
    assert b["blemish"].params["max_blobs"] < a["blemish"].params["max_blobs"]
    assert b["blemish"].params["contrast"] < a["blemish"].params["contrast"]


# --- verdict paths with the shipped config and real samples --------------------------------


def test_pass(config: Config, make_sample: Sample, white_good: np.ndarray) -> None:
    result = run_pipeline("CAM_A", good_frames(make_sample, white_good), config)
    assert result.verdict == PASS, [(o.name, o.status, o.reason) for o in result.results]
    assert result.failed_checks == ()
    assert len(result.results) == 7  # noise is disabled
    assert all(o.status == PASS and o.result is not None and o.reason is None for o in result.results)
    assert {o.name: o.frame for o in result.results}["brightness"] == "normal"
    assert result.duration_ms >= sum(o.duration_ms for o in result.results) > 0


def test_fail_on_dirty_sensor(config: Config, make_sample: Sample, white_good: np.ndarray) -> None:
    frames = {**good_frames(make_sample, white_good), "flat": make_sample("dirty.png")}
    result = run_pipeline("CAM_A", frames, config)
    assert result.verdict == FAIL
    assert result.failed_checks == ("blemish",)
    assert "blemishes" in result.results[-1].result.details["reason"]  # type: ignore[union-attr]


def test_fail_on_hot_and_dead_pixels(config: Config, make_sample: Sample, white_good: np.ndarray) -> None:
    frames = {**good_frames(make_sample, white_good), "dark": make_sample("dark_bad.png"), "white": make_sample("white_bad.png")}
    result = run_pipeline("CAM_A", frames, config)
    assert result.verdict == FAIL
    assert result.failed_checks == ("pixels_hot", "pixels_dead")


def test_cam_b_is_stricter_than_cam_a(config: Config, make_sample: Sample, white_good: np.ndarray) -> None:
    frames = good_frames(make_sample, white_good)
    frames["normal"] = cv2.GaussianBlur(make_sample("sharp.png"), (0, 0), 1)  # slightly soft
    assert run_pipeline("CAM_A", frames, config).verdict == PASS
    b = run_pipeline("CAM_B", frames, config)
    assert b.verdict == FAIL and b.failed_checks == ("sharpness",)
    assert run_pipeline("CAM_B", good_frames(make_sample, white_good), config).verdict == PASS


def test_incomplete_when_a_frame_is_missing(config: Config, make_sample: Sample, white_good: np.ndarray) -> None:
    frames = good_frames(make_sample, white_good)
    del frames["white"]
    result = run_pipeline("CAM_A", frames, config)
    assert result.verdict == INCOMPLETE
    assert result.failed_checks == ()
    skipped = [o for o in result.results if o.status == SKIPPED]
    assert [o.name for o in skipped] == ["pixels_dead"]
    assert "white" in skipped[0].reason and skipped[0].result is None and skipped[0].frame is None  # type: ignore[operator]
    assert status_of(result)["pixels_hot"] == PASS


def test_fail_takes_precedence_over_skipped(config: Config, make_sample: Sample, white_good: np.ndarray) -> None:
    frames = {**good_frames(make_sample, white_good), "flat": make_sample("dirty.png")}
    del frames["white"]
    result = run_pipeline("CAM_A", frames, config)
    assert result.verdict == FAIL and status_of(result)["pixels_dead"] == SKIPPED


# --- logic with small configs and frames ---------------------------------------------------


def test_frame_list_falls_back_to_next_type() -> None:
    cfg = mini({"brightness": {"frame": ["normal", "flat"], "params": {}}})
    only_flat = run_pipeline("M", {"flat": small()}, cfg)
    assert only_flat.results[0].frame == "flat" and only_flat.verdict == PASS
    both = run_pipeline("M", {"flat": small(), "normal": small(100)}, cfg)
    assert both.results[0].frame == "normal"


def test_none_frame_counts_as_missing() -> None:
    result = run_pipeline("M", {"flat": None}, mini({"brightness": BRIGHT}))  # type: ignore[dict-item]
    assert result.verdict == INCOMPLETE and result.results[0].status == SKIPPED


def test_error_from_bad_input_is_recorded_and_run_continues() -> None:
    cfg = mini({"brightness": BRIGHT, "sharpness": SHARP})
    result = run_pipeline("M", {"flat": small(), "normal": np.zeros((4, 4, 3), np.uint8)}, cfg)
    outcome = {o.name: o for o in result.results}
    assert outcome["sharpness"].status == ERROR
    assert outcome["sharpness"].reason.startswith("ValueError")  # type: ignore[union-attr]
    assert outcome["sharpness"].result is None
    assert outcome["brightness"].status == PASS
    assert result.verdict == INCOMPLETE and result.failed_checks == ()


def test_error_does_not_hide_a_failure() -> None:
    cfg = mini({"brightness": {"frame": "flat", "params": {"mean_low": 200}}, "sharpness": SHARP})
    result = run_pipeline("M", {"flat": small(), "normal": np.zeros((4, 4, 3), np.uint8)}, cfg)
    assert result.verdict == FAIL and result.failed_checks == ("brightness",)
    assert status_of(result)["sharpness"] == ERROR


def test_unexpected_exception_in_a_check_is_caught(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(image: np.ndarray, params: dict[str, Any]) -> None:
        raise RuntimeError("camera exploded")

    real = pipeline.get_runner
    monkeypatch.setattr(pipeline, "get_runner", lambda name: boom if name == "sharpness" else real(name))
    cfg = mini({"brightness": BRIGHT, "sharpness": SHARP})
    result = run_pipeline("M", {"flat": small(), "normal": chart()}, cfg)
    assert result.results[1].status == ERROR
    assert result.results[1].reason == "RuntimeError: camera exploded"
    assert result.results[0].status == PASS and result.verdict == INCOMPLETE


def test_unimplemented_check_is_an_error() -> None:
    cfg = mini({"noise": {"frame": "flat", "params": {}}})  # checks/noise.py has no run() yet
    result = run_pipeline("M", {"flat": small()}, cfg)
    assert result.results[0].status == ERROR
    assert "not implemented" in result.results[0].reason  # type: ignore[operator]
    assert result.verdict == INCOMPLETE


def test_disabled_checks_are_not_run() -> None:
    cfg = mini({"brightness": BRIGHT, "sharpness": {**SHARP, "enabled": False}})
    result = run_pipeline("M", {"flat": small()}, cfg)
    assert [o.name for o in result.results] == ["brightness"] and result.verdict == PASS


def test_unknown_model_raises(config: Config) -> None:
    with pytest.raises(ValueError, match="CAM_A"):
        run_pipeline("CAM_Z", {}, config)


# --- config validation ---------------------------------------------------------------------


def test_load_config_errors(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="cannot read config file"):
        load_config(tmp_path / "nope.yaml")
    bad = tmp_path / "bad.yaml"
    bad.write_text("models: [unclosed", encoding="utf-8")
    with pytest.raises(ConfigError, match="invalid YAML"):
        load_config(bad)
    empty = tmp_path / "empty.yaml"
    empty.write_text("", encoding="utf-8")
    with pytest.raises(ConfigError, match="missing required top-level key 'models'"):
        load_config(empty)


def valid_check(**over: Any) -> dict[str, Any]:
    return {"frame": "flat", "params": {}, **over}


@pytest.mark.parametrize(
    ("data", "message"),
    [
        ({"models": {"M": {"checks": {"sharpness": {"params": {}}}}}}, "models.M.checks.sharpness: missing required key 'frame'"),
        ({"models": {"M": {}}}, "models.M: missing required key 'checks'"),
        ({"models": {"M": {"checks": {}}}}, "models.M.checks: expected a non-empty mapping"),
        ({"models": {}}, "models: expected a non-empty mapping"),
        ({"models": {"M": {"checks": {"bogus": valid_check()}}}}, "unknown check 'bogus'"),
        ({"models": {"M": {"checks": {"x": valid_check(check="bogus")}}}}, "unknown check 'bogus'"),
        ({"models": {"M": {"checks": {"color": valid_check(frame="blurry")}}}}, "models.M.checks.color.frame"),
        ({"models": {"M": {"checks": {"color": valid_check(frame=[])}}}}, "models.M.checks.color.frame"),
        ({"models": {"M": {"checks": {"color": valid_check(parms={})}}}}, "unknown key(s) ['parms']"),
        ({"models": {"M": {"checks": {"color": valid_check(params=[1])}}}}, "models.M.checks.color.params"),
        ({"models": {"M": {"checks": {"color": valid_check(enabled="yes")}}}}, "models.M.checks.color.enabled"),
        ({"models": {"M": {"checks": {"color": "x"}}}}, "models.M.checks.color: expected a mapping"),
        ({"models": {"M": {"checks": {"color": valid_check(enabled=False)}}}}, "at least one check must be enabled"),
        ({"models": {"M": {"checks": {"color": valid_check()}}}, "extra": 1}, "unknown key(s) ['extra']"),
        (["not", "a", "mapping"], "missing required top-level key 'models'"),
    ],
)
def test_parse_config_errors_are_specific(data: Any, message: str) -> None:
    with pytest.raises(ConfigError) as info:
        parse_config(data, source="test.yaml")
    assert message in str(info.value)
    assert str(info.value).startswith("test.yaml: ")


def test_check_name_defaults_to_key_and_frame_string_becomes_tuple() -> None:
    cfg = parse_config({"models": {"M": {"checks": {"pixels_hot": {"check": "pixels", "frame": "dark"}, "color": {"frame": "flat"}}}}})
    hot, col = cfg.models["M"].checks
    assert (hot.check, hot.frames, hot.enabled, dict(hot.params)) == ("pixels", ("dark",), True, {})
    assert (col.check, col.frames) == ("color", ("flat",))
