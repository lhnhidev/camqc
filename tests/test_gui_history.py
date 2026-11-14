"""Tests for the History tab, the run-detail dialog and the shared date bar / lazy refresh."""

from __future__ import annotations

from collections.abc import Iterator
from datetime import date, datetime
from pathlib import Path

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QCoreApplication, QEvent, Qt  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from camqc import db  # noqa: E402
from camqc.gui import history_tab  # noqa: E402
from camqc.gui.common import DateRangeBar  # noqa: E402
from camqc.gui.history_tab import HistoryTab  # noqa: E402
from gui_support import click, wait_for  # noqa: E402
from seed_support import PASSING, add_run, seed  # noqa: E402


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    path = tmp_path / "t.db"
    conn = db.init_db(path)
    seed(conn)
    conn.close()
    return path


@pytest.fixture
def tab(qapp: QApplication, db_path: Path) -> Iterator[HistoryTab]:
    widget = HistoryTab(db_path)
    widget.resize(1000, 600)
    widget.show()  # the first show loads the data
    yield widget
    widget.close()


def displayed(widget: HistoryTab, column: int) -> list[str]:
    """The text of one column, in the order shown (after sorting)."""
    return [widget.proxy.index(r, column).data() for r in range(widget.proxy.rowCount())]


def serials(widget: HistoryTab) -> list[str]:
    return displayed(widget, 2)


# --- loading and filtering ---------------------------------------------------------------


def test_first_show_loads_runs_newest_first(tab: HistoryTab) -> None:
    assert serials(tab) == ["C1", "B2", "B1", "A3", "A2", "A1"]
    assert displayed(tab, 0) == ["6", "5", "4", "3", "2", "1"]
    assert tab.count_label.text() == "6 runs"
    assert tab.table.isVisibleTo(tab) and not tab.empty_label.isVisibleTo(tab)
    row = tab.proxy.index(0, 0)
    assert [tab.proxy.index(0, c).data() for c in range(1, 7)] == [
        "2026-11-16T09:30:15".replace(":30:15", ":00:00"), "C1", "CAM_A", "ST1", "PASS", "120"]
    assert row.data(Qt.ItemDataRole.TextAlignmentRole) & Qt.AlignmentFlag.AlignRight


def test_verdict_cells_are_colored(tab: HistoryTab) -> None:
    def bg(serial: str) -> str:
        row = serials(tab).index(serial)
        return tab.proxy.index(row, 5).data(Qt.ItemDataRole.BackgroundRole).color().name()

    assert (bg("A1"), bg("A2"), bg("B2")) == ("#c6efce", "#ffc7ce", "#ffeb9c")  # PASS, FAIL, INCOMPLETE
    assert tab.proxy.index(0, 2).data(Qt.ItemDataRole.BackgroundRole) is None  # other columns are plain


def test_serial_search_matches_substrings_ignoring_case(tab: HistoryTab) -> None:
    tab.serial_edit.setText("a")
    tab.refresh()
    assert sorted(serials(tab)) == ["A1", "A2", "A3"]
    tab.serial_edit.setText("b2")
    tab.refresh()
    assert serials(tab) == ["B2"] and tab.count_label.text() == "1 run"


def test_typing_in_the_search_box_searches_after_a_short_pause(tab: HistoryTab) -> None:
    QTest.keyClicks(tab.serial_edit, "B1")
    assert len(serials(tab)) == 6  # not yet: the search is debounced
    wait_for(lambda: serials(tab) == ["B1"])


def test_enter_and_search_button_search_immediately(tab: HistoryTab) -> None:
    tab.serial_edit.setText("C")
    QTest.keyClick(tab.serial_edit, Qt.Key.Key_Return)
    assert serials(tab) == ["C1"]
    tab.serial_edit.setText("A1")
    click(tab.search_button)
    assert serials(tab) == ["A1"]


def test_date_range_filter(tab: HistoryTab) -> None:
    tab.date_bar.set_range(date(2026, 11, 15), date(2026, 11, 15))  # triggers a reload by itself
    assert serials(tab) == ["B2", "B1"]
    tab.date_bar.set_range(date(2026, 11, 14), date(2026, 11, 15))
    assert len(serials(tab)) == 5
    tab.date_bar.check.setChecked(False)  # off: all dates again
    assert len(serials(tab)) == 6


def test_verdict_filter(tab: HistoryTab) -> None:
    tab.verdict_combo.setCurrentText("FAIL")
    assert serials(tab) == ["B1", "A2"]
    tab.verdict_combo.setCurrentText("INCOMPLETE")
    assert serials(tab) == ["B2"]
    tab.verdict_combo.setCurrentIndex(0)
    assert len(serials(tab)) == 6


def test_clear_resets_every_filter(tab: HistoryTab) -> None:
    tab.serial_edit.setText("A")
    tab.verdict_combo.setCurrentText("FAIL")
    tab.date_bar.set_range(date(2026, 11, 14), date(2026, 11, 14))
    assert serials(tab) == ["A2"]
    click(tab.clear_button)
    assert len(serials(tab)) == 6
    assert tab.serial_edit.text() == "" and tab.verdict_combo.currentIndex() == 0
    assert not tab.date_bar.check.isChecked() and tab.date_bar.date_range() == (None, None)


def test_sorting_by_run_id_is_numeric_not_alphabetical(qapp: QApplication, tmp_path: Path) -> None:
    conn = db.init_db(tmp_path / "many.db")
    for i in range(12):
        add_run(conn, f"S{i:02d}", datetime(2026, 11, 14, i), PASSING)
    conn.close()
    widget = HistoryTab(tmp_path / "many.db")
    widget.show()
    widget.table.sortByColumn(0, Qt.SortOrder.AscendingOrder)
    assert displayed(widget, 0) == [str(i) for i in range(1, 13)]  # text order would put "10" before "2"
    widget.table.sortByColumn(6, Qt.SortOrder.DescendingOrder)
    assert displayed(widget, 6) == ["120"] * 12
    widget.table.sortByColumn(2, Qt.SortOrder.DescendingOrder)
    assert serials(widget)[0] == "S11"
    widget.close()


# --- empty and error states ----------------------------------------------------------------


def test_empty_database_shows_a_friendly_message(qapp: QApplication, tmp_path: Path) -> None:
    widget = HistoryTab(tmp_path / "empty.db")
    widget.show()
    assert widget.empty_label.isVisibleTo(widget) and widget.empty_label.text() == "No runs found."
    assert not widget.table.isVisibleTo(widget) and widget.count_label.text() == "0 runs"
    assert widget.message_label.text() == ""
    widget.close()


def test_search_without_matches_shows_the_empty_state(tab: HistoryTab) -> None:
    tab.serial_edit.setText("nothing like this")
    tab.refresh()
    assert tab.empty_label.isVisibleTo(tab) and not tab.table.isVisibleTo(tab)
    assert tab.count_label.text() == "0 runs"
    tab.serial_edit.setText("")
    tab.refresh()
    assert tab.table.isVisibleTo(tab) and not tab.empty_label.isVisibleTo(tab)


def test_database_error_is_reported_not_raised(qapp: QApplication, tmp_path: Path) -> None:
    widget = HistoryTab(tmp_path)  # a directory cannot be opened as a database
    widget.show()
    assert "Cannot read the database" in widget.message_label.text()
    assert widget.empty_label.isVisibleTo(widget) and widget.proxy.rowCount() == 0
    widget.close()


def test_results_are_truncated_with_a_notice(tab: HistoryTab, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(history_tab, "MAX_ROWS", 4)
    tab.refresh()
    assert serials(tab) == ["C1", "B2", "B1", "A3"]
    assert "first 4 shown" in tab.count_label.text()
    monkeypatch.setattr(history_tab, "MAX_ROWS", 6)  # exactly as many as exist: no notice
    tab.refresh()
    assert tab.count_label.text() == "6 runs"


# --- details dialog ----------------------------------------------------------------------


def test_double_click_opens_the_runs_check_results(tab: HistoryTab) -> None:
    pos = tab.table.visualRect(tab.proxy.index(0, 2)).center()  # newest run: C1
    viewport = tab.table.viewport()
    QTest.mouseClick(viewport, Qt.MouseButton.LeftButton, pos=pos)  # a real double-click starts with a click
    QTest.mouseDClick(viewport, Qt.MouseButton.LeftButton, pos=pos)
    dialog = tab.detail_dialog
    assert dialog is not None and dialog.isVisible()
    assert dialog.windowTitle() == "Run #6 - C1"
    header = dialog.header.text()
    assert "C1" in header and "CAM_A" in header and "ST1" in header and "PASS" in header
    table = dialog.table
    assert [table.horizontalHeaderItem(c).text() for c in range(6)] == [
        "Check", "Value", "Unit", "Limits", "Status", "Note"]
    assert [[table.item(r, c).text() for c in range(5)] for r in range(table.rowCount())] == [
        ["brightness", "1", "u", "0 .. 10", "PASS"], ["sharpness", "1", "u", "0 .. 10", "PASS"]]
    assert table.item(0, 0).background().color().name() == "#c6efce"
    dialog.close()


def test_detail_shows_failures_skips_and_notes(tab: HistoryTab) -> None:
    fail = tab.open_run_detail(5 - 1)  # run #4 = B1 (sharpness failed)
    assert fail is not None
    rows = {fail.table.item(r, 0).text(): r for r in range(fail.table.rowCount())}
    r = rows["sharpness"]
    assert fail.table.item(r, 4).text() == "FAIL" and fail.table.item(r, 5).text() == "sharpness out of range"
    assert fail.table.item(r, 0).background().color().name() == "#ffc7ce"
    fail.close()
    skipped = tab.open_run_detail(5)  # run #5 = B2 (pixels_dead skipped)
    assert skipped is not None
    row = [skipped.table.item(1, c).text() for c in range(6)]
    assert row == ["pixels_dead", "-", "-", "-", "SKIPPED", "required frame missing"]
    assert skipped.table.item(1, 0).background().color().name() == "#ffeb9c"
    assert "INCOMPLETE" in skipped.header.text()
    skipped.close()


def test_unknown_run_id_is_reported(tab: HistoryTab) -> None:
    assert tab.open_run_detail(999) is None
    assert "Run #999 no longer exists" in tab.message_label.text()


def test_closed_dialog_is_forgotten(tab: HistoryTab) -> None:
    dialog = tab.open_run_detail(1)
    assert dialog is not None and tab.detail_dialog is dialog
    dialog.close()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete.value)  # WA_DeleteOnClose
    assert tab.detail_dialog is None


# --- lazy refresh ---------------------------------------------------------------------------


def test_hidden_tab_does_not_query_until_shown(qapp: QApplication, db_path: Path) -> None:
    widget = HistoryTab(db_path)
    assert widget.is_stale and widget.proxy.rowCount() == 0  # nothing loaded while hidden
    widget.request_refresh()
    assert widget.is_stale and widget.proxy.rowCount() == 0
    widget.show()
    assert not widget.is_stale and widget.proxy.rowCount() == 6
    widget.close()


def test_visible_tab_refreshes_after_new_data_with_one_coalesced_reload(
    tab: HistoryTab, db_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[int] = []
    real = tab.reload
    monkeypatch.setattr(tab, "reload", lambda: (calls.append(1), real())[1])
    conn = db.init_db(db_path)
    for i in range(3):
        add_run(conn, f"N{i}", datetime(2026, 11, 17, i), PASSING)
        tab.request_refresh()  # a burst of saves
    conn.close()
    assert len(serials(tab)) == 6  # not reloaded yet
    wait_for(lambda: len(serials(tab)) == 9)
    assert len(calls) == 1 and serials(tab)[0] == "N2"


# --- the date bar ---------------------------------------------------------------------------


def test_date_range_bar(qapp: QApplication) -> None:
    bar = DateRangeBar()
    changes: list[int] = []
    bar.changed.connect(lambda: changes.append(1))
    assert not bar.from_edit.isEnabled() and not bar.to_edit.isEnabled() and bar.date_range() == (None, None)
    bar.check.setChecked(True)
    assert bar.from_edit.isEnabled() and bar.to_edit.isEnabled() and len(changes) == 1
    first, last = bar.date_range()
    assert first is not None and last == date.today() and (last - first).days == 30
    bar.set_range(date(2026, 1, 2), date(2026, 3, 4))
    assert bar.date_range() == (date(2026, 1, 2), date(2026, 3, 4)) and changes
    count = len(changes)
    bar.to_edit.setDate(bar.to_edit.date().addDays(1))
    assert len(changes) == count + 1 and bar.date_range()[1] == date(2026, 3, 5)
    bar.check.setChecked(False)
    assert bar.date_range() == (None, None) and not bar.from_edit.isEnabled()


def test_dialog_header_shows_free_text_literally(qapp: QApplication, tmp_path: Path) -> None:
    conn = db.init_db(tmp_path / "x.db")
    run_id = add_run(conn, "<SN-1> & <b>co</b>", datetime(2026, 11, 14, 8), PASSING, model="M<1>", station="S&T")
    conn.close()
    widget = HistoryTab(tmp_path / "x.db")
    widget.show()
    dialog = widget.open_run_detail(run_id)
    assert dialog is not None
    assert "&lt;SN-1&gt; &amp; &lt;b&gt;co&lt;/b&gt;" in dialog.header.text()
    assert "M&lt;1&gt;" in dialog.header.text() and "S&amp;T" in dialog.header.text()
    assert "<SN-1>" not in dialog.header.text()
    dialog.close()
    widget.close()


def test_opening_a_run_clears_an_earlier_error(tab: HistoryTab) -> None:
    assert tab.open_run_detail(999) is None and tab.message_label.text()
    dialog = tab.open_run_detail(1)
    assert dialog is not None and tab.message_label.text() == ""
    dialog.close()


def test_reversed_dates_are_swapped_not_silently_empty(tab: HistoryTab) -> None:
    tab.date_bar.set_range(date(2026, 11, 15), date(2026, 11, 14))  # from after to
    assert tab.date_bar.date_range() == (date(2026, 11, 14), date(2026, 11, 15))
    assert len(serials(tab)) == 5


def test_reload_is_skipped_if_the_tab_was_hidden_during_the_delay(tab: HistoryTab, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[int] = []
    real = tab.reload
    monkeypatch.setattr(tab, "reload", lambda: (calls.append(1), real())[1])
    tab.request_refresh()  # starts the delay timer while visible
    tab.hide()  # ... and the tab disappears before it fires
    wait_for(lambda: not tab._timer.isActive())
    assert calls == [] and tab.is_stale
    tab.show()  # reloads on the next show
    assert calls == [1] and not tab.is_stale
