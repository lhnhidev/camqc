"""Tests for the SQLite persistence layer, using in-memory databases."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

from camqc import db
from camqc.checks.base import make_result
from camqc.db import RunInput
from camqc.pipeline import ERROR, FAIL, INCOMPLETE, PASS, SKIPPED, CheckOutcome, RunResult


@pytest.fixture
def conn() -> Iterator[sqlite3.Connection]:
    connection = db.init_db(":memory:")
    yield connection
    connection.close()


def outcome(name: str, status: str = PASS, value: float = 1.0, **details: object) -> CheckOutcome:
    if status in (PASS, FAIL):
        res = make_result(name, value, "u", 0.0, 10.0, dict(details))
        return CheckOutcome(name, status, res, None, "flat", 1.5)
    reason = "required frame missing" if status == SKIPPED else "ValueError: boom"
    return CheckOutcome(name, status, None, reason, None if status == SKIPPED else "flat", 0.1)


def run_result(verdict: str, *outcomes: CheckOutcome, duration_ms: float = 120.4) -> RunResult:
    failed = tuple(o.name for o in outcomes if o.status == FAIL)
    return RunResult(verdict, tuple(outcomes), failed, duration_ms)


def at(day: int, hour: int = 9, month: int = 11) -> datetime:
    return datetime(2026, month, day, hour, 30, 15)


def run_input(serial: str, when: datetime, model: str = "CAM_A", station: str = "ST1") -> RunInput:
    return RunInput(serial, model, station, when)


PASSING = run_result(PASS, outcome("brightness"), outcome("sharpness"))
FAILING_SHARP = run_result(FAIL, outcome("brightness"), outcome("sharpness", FAIL, 99.0))
FAILING_BOTH = run_result(FAIL, outcome("sharpness", FAIL), outcome("blemish", FAIL))


@pytest.fixture
def seeded(conn: sqlite3.Connection) -> sqlite3.Connection:
    """Day 14: PASS, FAIL(sharpness+blemish), PASS. Day 15: FAIL(sharpness), INCOMPLETE. Day 16: PASS."""
    db.save_run(conn, run_input("A1", at(14, 8)), PASSING)
    db.save_run(conn, run_input("A2", at(14, 10)), FAILING_BOTH)
    db.save_run(conn, run_input("A3", at(14, 23)), PASSING)
    db.save_run(conn, run_input("B1", at(15, 9)), FAILING_SHARP)
    db.save_run(conn, run_input("B2", at(15, 11)), run_result(
        INCOMPLETE, outcome("brightness"), outcome("pixels_dead", SKIPPED)))
    db.save_run(conn, run_input("C1", at(16, 9)), PASSING)
    return conn


# --- schema ----------------------------------------------------------------------------


def test_init_db_creates_tables_indexes_and_foreign_keys(conn: sqlite3.Connection) -> None:
    objects = {r["name"]: r["type"] for r in conn.execute("SELECT name, type FROM sqlite_master")}
    assert objects["runs"] == objects["results"] == "table"
    assert {"idx_runs_serial", "idx_runs_started_at", "idx_results_check_status"} <= set(objects)
    cols = [r["name"] for r in conn.execute("PRAGMA table_info(runs)")]
    assert cols == ["id", "serial", "model", "station", "started_at", "verdict", "duration_ms"]
    cols = [r["name"] for r in conn.execute("PRAGMA table_info(results)")]
    assert cols == ["id", "run_id", "check_name", "value", "unit", "low", "high", "status", "details_json"]
    assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute("INSERT INTO results (run_id, check_name, status, details_json) VALUES (999, 'x', 'PASS', '{}')")


def test_init_db_is_idempotent_and_persists_to_file(tmp_path: Path) -> None:
    path = tmp_path / "sub" / "camqc.db"
    first = db.init_db(path)
    run_id = db.save_run(first, run_input("F1", at(14)), PASSING)
    first.close()
    second = db.init_db(path)  # re-open: schema creation must not wipe data
    assert [r["id"] for r in db.find_by_serial(second, "F1")] == [run_id]
    second.close()


# --- save_run / get_run_detail ---------------------------------------------------------


def test_save_run_and_detail_roundtrip(conn: sqlite3.Connection) -> None:
    result = run_result(
        FAIL,
        outcome("sharpness", FAIL, 3.5, center=3.5, corners={"top_left": 1.0}),
        outcome("pixels_dead", SKIPPED),
        outcome("noise", ERROR),
        duration_ms=1234.6,
    )
    run_id = db.save_run(conn, run_input("SN001", at(14, 13), model="CAM_B", station="ST9"), result)
    detail = db.get_run_detail(conn, run_id)
    assert detail is not None
    assert (detail["serial"], detail["model"], detail["station"]) == ("SN001", "CAM_B", "ST9")
    assert detail["started_at"] == "2026-11-14T13:30:15"
    assert detail["verdict"] == FAIL and detail["duration_ms"] == 1235
    sharp, skipped, errored = detail["results"]
    assert (sharp["check_name"], sharp["status"], sharp["value"], sharp["unit"]) == ("sharpness", FAIL, 3.5, "u")
    assert (sharp["low"], sharp["high"]) == (0.0, 10.0)
    assert sharp["details"] == {"center": 3.5, "corners": {"top_left": 1.0}} and sharp["frame"] == "flat"
    assert skipped["status"] == SKIPPED and skipped["value"] is None and skipped["low"] is None
    assert skipped["reason"] == "required frame missing" and skipped["details"] == {}
    assert errored["status"] == ERROR and errored["reason"] == "ValueError: boom"
    assert "details_json" not in sharp


def test_get_run_detail_unknown_id_returns_none(conn: sqlite3.Connection) -> None:
    assert db.get_run_detail(conn, 12345) is None


def test_numpy_values_in_details_are_stored(conn: sqlite3.Connection) -> None:
    res = run_result(PASS, outcome("pixels", PASS, defects=[(1, 2)], arr=np.array([1.5, 2.5]), n=np.int64(3)))
    detail = db.get_run_detail(conn, db.save_run(conn, run_input("N1", at(14)), res))
    assert detail["results"][0]["details"] == {"defects": [[1, 2]], "arr": [1.5, 2.5], "n": 3}  # type: ignore[index]


def test_aware_datetime_is_stored_as_naive_local_text(conn: sqlite3.Connection) -> None:
    when = datetime(2026, 11, 14, 12, 0, 0, tzinfo=timezone(timedelta(hours=7)))
    run_id = db.save_run(conn, run_input("TZ1", when), PASSING)
    stored = db.get_run_detail(conn, run_id)["started_at"]  # type: ignore[index]
    assert stored == when.astimezone().replace(tzinfo=None).isoformat(timespec="seconds")


def test_serial_is_stripped_before_storing(conn: sqlite3.Connection) -> None:
    run_id = db.save_run(conn, run_input("  SN5 ", at(14)), PASSING)
    assert [r["id"] for r in db.find_by_serial(conn, "SN5")] == [run_id]


def test_empty_serial_is_rejected(conn: sqlite3.Connection) -> None:
    with pytest.raises(ValueError):
        db.save_run(conn, run_input("  ", at(14)), PASSING)
    assert conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0] == 0


def test_save_run_is_atomic(conn: sqlite3.Connection) -> None:
    bad = run_result(PASS, outcome("brightness"), outcome("sharpness", PASS, junk=object()))
    with pytest.raises(TypeError):
        db.save_run(conn, run_input("ATOM", at(14)), bad)
    assert conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM results").fetchone()[0] == 0
    assert db.save_run(conn, run_input("ATOM", at(14)), PASSING) > 0  # connection still usable


# --- find_by_serial / retest -----------------------------------------------------------


def test_retest_creates_two_runs_newest_first(conn: sqlite3.Connection) -> None:
    first = db.save_run(conn, run_input("SN777", at(14, 9)), FAILING_SHARP)
    second = db.save_run(conn, run_input("SN777", at(14, 11)), PASSING)
    db.save_run(conn, run_input("OTHER", at(14, 10)), PASSING)
    runs = db.find_by_serial(conn, "SN777")
    assert [r["id"] for r in runs] == [second, first]
    assert [r["verdict"] for r in runs] == [PASS, FAIL]
    assert db.find_by_serial(conn, "NOPE") == []
    # the two runs keep separate result sets
    assert len(db.get_run_detail(conn, first)["results"]) == 2  # type: ignore[index]
    assert conn.execute("SELECT COUNT(*) FROM results WHERE run_id = ?", (second,)).fetchone()[0] == 2


def test_same_timestamp_retest_orders_by_id(conn: sqlite3.Connection) -> None:
    a = db.save_run(conn, run_input("SN1", at(14)), FAILING_SHARP)
    b = db.save_run(conn, run_input("SN1", at(14)), PASSING)
    assert [r["id"] for r in db.find_by_serial(conn, "SN1")] == [b, a]


def test_serial_lookup_is_parameterized(conn: sqlite3.Connection) -> None:
    nasty = "x'; DROP TABLE runs; --"
    run_id = db.save_run(conn, run_input(nasty, at(14)), PASSING)
    assert [r["id"] for r in db.find_by_serial(conn, nasty)] == [run_id]
    assert db.find_by_serial(conn, "x' OR '1'='1") == []
    assert conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0] == 1


# --- list_runs -------------------------------------------------------------------------


def test_list_runs_by_range_is_inclusive_and_newest_first(seeded: sqlite3.Connection) -> None:
    runs = db.list_runs(seeded, date(2026, 11, 14), date(2026, 11, 15))
    assert [r["serial"] for r in runs] == ["B2", "B1", "A3", "A2", "A1"]
    assert [r["serial"] for r in db.list_runs(seeded, "2026-11-16", "2026-11-16")] == ["C1"]
    assert [r["serial"] for r in db.list_runs(seeded, datetime(2026, 11, 14, 23, 59), "2026-11-14")] == [
        "A3", "A2", "A1"]  # datetimes count as their calendar day; 23:xx run on day end is included


def test_list_runs_filters_by_verdict_and_open_bounds(seeded: sqlite3.Connection) -> None:
    fails = db.list_runs(seeded, "2026-11-14", "2026-11-16", verdict=FAIL)
    assert [r["serial"] for r in fails] == ["B1", "A2"]
    assert [r["serial"] for r in db.list_runs(seeded, None, None, INCOMPLETE)] == ["B2"]
    assert len(db.list_runs(seeded, None, None)) == 6
    assert [r["serial"] for r in db.list_runs(seeded, "2026-11-16", None)] == ["C1"]
    assert db.list_runs(seeded, "2026-12-01", "2026-12-31") == []
    assert db.list_runs(seeded, "2026-11-14", "2026-11-16", verdict="BOGUS") == []


# --- yield ------------------------------------------------------------------------------


def test_yield_by_day(seeded: sqlite3.Connection) -> None:
    rows = db.yield_by_day(seeded, "2026-11-14", "2026-11-16")
    assert [r["day"] for r in rows] == ["2026-11-14", "2026-11-15", "2026-11-16"]
    d14, d15, d16 = rows
    assert (d14["total"], d14["passed"], d14["failed"], d14["incomplete"]) == (3, 2, 1, 0)
    assert d14["yield_pct"] == pytest.approx(200 / 3)
    assert (d15["total"], d15["passed"], d15["failed"], d15["incomplete"]) == (2, 0, 1, 1)
    assert d15["yield_pct"] == 0.0
    assert d16["yield_pct"] == 100.0
    assert [r["day"] for r in db.yield_by_day(seeded, "2026-11-15", "2026-11-15")] == ["2026-11-15"]
    assert db.yield_by_day(seeded, "2027-01-01", "2027-01-02") == []


def test_retests_count_as_separate_runs_in_yield(conn: sqlite3.Connection) -> None:
    db.save_run(conn, run_input("SN9", at(14, 9)), FAILING_SHARP)
    db.save_run(conn, run_input("SN9", at(14, 10)), PASSING)
    (day,) = db.yield_by_day(conn, "2026-11-14", "2026-11-14")
    assert (day["total"], day["passed"], day["failed"]) == (2, 1, 1) and day["yield_pct"] == 50.0


# --- pareto -----------------------------------------------------------------------------


def test_pareto_failures(seeded: sqlite3.Connection) -> None:
    rows = db.pareto_failures(seeded, "2026-11-14", "2026-11-16")
    assert [(r["check_name"], r["fail_count"]) for r in rows] == [("sharpness", 2), ("blemish", 1)]
    assert rows[0]["pct"] == pytest.approx(200 / 3) and rows[1]["pct"] == pytest.approx(100 / 3)
    assert rows[0]["cumulative_pct"] == pytest.approx(200 / 3) and rows[1]["cumulative_pct"] == pytest.approx(100.0)


def test_pareto_respects_range_and_ignores_skipped_and_errors(seeded: sqlite3.Connection) -> None:
    only_15 = db.pareto_failures(seeded, "2026-11-15", "2026-11-15")
    assert [(r["check_name"], r["fail_count"]) for r in only_15] == [("sharpness", 1)]
    assert only_15[0]["pct"] == 100.0 and only_15[0]["cumulative_pct"] == 100.0
    assert db.pareto_failures(seeded, "2026-11-16", "2026-11-16") == []  # pass-only day
    assert db.pareto_failures(seeded, "2027-01-01", None) == []  # skipped pixels_dead never appears


def test_pareto_ties_are_ordered_by_name(conn: sqlite3.Connection) -> None:
    db.save_run(conn, run_input("T1", at(14)), run_result(FAIL, outcome("zeta", FAIL), outcome("alpha", FAIL)))
    assert [r["check_name"] for r in db.pareto_failures(conn, None, None)] == ["alpha", "zeta"]


# --- search_runs ------------------------------------------------------------------------


def serials(rows: list[dict]) -> list[str]:
    return [r["serial"] for r in rows]


def test_search_runs_substring_is_case_insensitive_and_newest_first(conn: sqlite3.Connection) -> None:
    db.save_run(conn, run_input("SN-001", at(14, 8)), PASSING)
    db.save_run(conn, run_input("sn-002", at(14, 9)), PASSING)
    db.save_run(conn, run_input("XSN-003", at(14, 10)), PASSING)
    db.save_run(conn, run_input("OTHER", at(14, 11)), PASSING)
    assert serials(db.search_runs(conn, "sn")) == ["XSN-003", "sn-002", "SN-001"]
    assert serials(db.search_runs(conn, "N-00")) == ["XSN-003", "sn-002", "SN-001"]
    assert serials(db.search_runs(conn, "SN-001")) == ["SN-001"]
    assert serials(db.search_runs(conn, "zzz")) == []
    assert len(db.search_runs(conn, "")) == len(db.search_runs(conn, None)) == 4  # empty = no filter


def test_search_runs_treats_wildcards_literally(conn: sqlite3.Connection) -> None:
    for i, serial in enumerate(["A%B", "A_B", "AXB", "50!off", r"back\slash"]):
        db.save_run(conn, run_input(serial, at(14, 8 + i)), PASSING)
    assert serials(db.search_runs(conn, "%")) == ["A%B"]
    assert serials(db.search_runs(conn, "_")) == ["A_B"]
    assert serials(db.search_runs(conn, "A_B")) == ["A_B"]  # does not also match AXB
    assert serials(db.search_runs(conn, "!")) == ["50!off"]
    assert serials(db.search_runs(conn, "\\")) == [r"back\slash"]


def test_search_runs_combines_filters_and_limit(seeded: sqlite3.Connection) -> None:
    assert serials(db.search_runs(seeded, "B", "2026-11-14", "2026-11-15", FAIL)) == ["B1"]
    assert serials(db.search_runs(seeded, None, "2026-11-15", "2026-11-15")) == ["B2", "B1"]
    assert serials(db.search_runs(seeded, None, None, None, limit=2)) == ["C1", "B2"]
    assert db.search_runs(seeded, "x'; DROP TABLE runs; --") == []
    assert len(db.search_runs(seeded)) == 6  # the table is still there
