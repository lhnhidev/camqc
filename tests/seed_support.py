"""Seed data for the History/Statistics tests (pure Python; no Qt)."""

from __future__ import annotations

import sqlite3
from datetime import datetime

from camqc import db
from camqc.checks.base import make_result
from camqc.pipeline import ERROR, FAIL, INCOMPLETE, PASS, SKIPPED, CheckOutcome, RunResult


def outcome(name: str, status: str = PASS, value: float = 1.0) -> CheckOutcome:
    """A check outcome: PASS/FAIL carry a result, SKIPPED/ERROR do not."""
    if status in (PASS, FAIL):
        details = {"reason": f"{name} out of range"} if status == FAIL else {}
        res = make_result(name, value, "u", 0.0, 10.0, details)
        return CheckOutcome(name, status, res, None, "flat", 1.5)
    reason = "required frame missing" if status == SKIPPED else "ValueError: boom"
    return CheckOutcome(name, status, None, reason, None if status == SKIPPED else "flat", 0.1)


def run_result(verdict: str, *outcomes: CheckOutcome, duration_ms: float = 120.0) -> RunResult:
    """A pipeline result with the given outcomes."""
    failed = tuple(o.name for o in outcomes if o.status == FAIL)
    return RunResult(verdict, tuple(outcomes), failed, duration_ms)


PASSING = run_result(PASS, outcome("brightness"), outcome("sharpness"))
FAILING_SHARP = run_result(FAIL, outcome("brightness"), outcome("sharpness", FAIL, 99.0))
FAILING_BOTH = run_result(FAIL, outcome("sharpness", FAIL), outcome("blemish", FAIL))
INCOMPLETE_RESULT = run_result(INCOMPLETE, outcome("brightness"), outcome("pixels_dead", SKIPPED))
ERRORED = run_result(INCOMPLETE, outcome("brightness"), outcome("noise", ERROR))


def add_run(conn: sqlite3.Connection, serial: str, when: datetime, result: RunResult,
            model: str = "CAM_A", station: str = "ST1") -> int:
    """Save one run and return its id."""
    return db.save_run(conn, db.RunInput(serial, model, station, when), result)


def seed(conn: sqlite3.Connection) -> None:
    """Six runs over three days.

    Nov 14: A1 PASS, A2 FAIL (sharpness+blemish), A3 PASS  -> yield 66.7 %
    Nov 15: B1 FAIL (sharpness), B2 INCOMPLETE             -> yield 0 %
    Nov 16: C1 PASS                                        -> yield 100 %
    Overall 3/6 = 50 %; Pareto: sharpness 2, blemish 1.
    """
    add_run(conn, "A1", datetime(2026, 11, 14, 8), PASSING)
    add_run(conn, "A2", datetime(2026, 11, 14, 10), FAILING_BOTH)
    add_run(conn, "A3", datetime(2026, 11, 14, 23), PASSING)
    add_run(conn, "B1", datetime(2026, 11, 15, 9), FAILING_SHARP)
    add_run(conn, "B2", datetime(2026, 11, 15, 11), INCOMPLETE_RESULT)
    add_run(conn, "C1", datetime(2026, 11, 16, 9), PASSING)
