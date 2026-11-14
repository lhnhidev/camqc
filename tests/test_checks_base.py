"""Tests for CheckResult and make_result."""

from __future__ import annotations

import dataclasses

import pytest

from camqc.checks.base import CheckResult, make_result


@pytest.mark.parametrize(
    ("value", "low", "high", "expected"),
    [
        (5.0, 1.0, 10.0, True),
        (1.0, 1.0, 10.0, True),  # inclusive bounds
        (10.0, 1.0, 10.0, True),
        (0.9, 1.0, 10.0, False),
        (10.1, 1.0, 10.0, False),
        (-100.0, None, 10.0, True),
        (100.0, None, 10.0, False),
        (100.0, 1.0, None, True),
        (0.0, 1.0, None, False),
        (123.0, None, None, True),
    ],
)
def test_make_result_computes_passed(value: float, low: float | None, high: float | None, expected: bool) -> None:
    result = make_result("x", value, "u", low, high)
    assert result.passed is expected
    assert (result.name, result.unit, result.low, result.high) == ("x", "u", low, high)


def test_make_result_defaults_and_copies() -> None:
    details = {"a": 1}
    overlay = [{"type": "circle", "x": 1, "y": 2, "r": 3}]
    result = make_result("x", 1, "u", None, None, details, overlay)
    assert result.value == 1.0 and isinstance(result.value, float)
    details["a"] = 2
    overlay.clear()
    assert result.details == {"a": 1}
    assert result.overlay == [{"type": "circle", "x": 1, "y": 2, "r": 3}]
    empty = make_result("x", 1, "u", None, None)
    assert empty.details == {} and empty.overlay == []


def test_nan_fails_when_limit_set() -> None:
    assert make_result("x", float("nan"), "u", 0.0, None).passed is False


def test_check_result_is_frozen() -> None:
    result = make_result("x", 1, "u", None, None)
    assert isinstance(result, CheckResult)
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.passed = False  # type: ignore[misc]
