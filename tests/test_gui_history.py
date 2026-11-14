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
from camqc.gui.common import DateRangeBar  # noqa: E402
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
