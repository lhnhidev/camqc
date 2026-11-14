"""Pieces shared by the GUI tabs: status colors, a date-range bar and a lazy-refresh base class."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import date, timedelta

from PySide6.QtCore import QDate, QTimer, Signal
from PySide6.QtGui import QShowEvent
from PySide6.QtWidgets import QCheckBox, QDateEdit, QHBoxLayout, QLabel, QWidget

GREEN, RED, AMBER, GRAY = "#c6efce", "#ffc7ce", "#ffeb9c", "#e0e0e0"
STATUS_COLORS: Mapping[str, tuple[str, str]] = {  # status -> (background, text)
    "PASS": (GREEN, "#006100"),
    "FAIL": (RED, "#9c0006"),
    "INCOMPLETE": (AMBER, "#9c5700"),
    "SKIPPED": (AMBER, "#9c5700"),
    "ERROR": (RED, "#9c0006"),
}
NEUTRAL_COLORS = (GRAY, "#404040")
DATE_FORMAT = "yyyy-MM-dd"
DEFAULT_RANGE_DAYS = 30


def colors_for(status: str) -> tuple[str, str]:
    """Return ``(background, text)`` colors for a status or verdict word."""
    return STATUS_COLORS.get(status, NEUTRAL_COLORS)


def to_qdate(day: date) -> QDate:
    """Convert a ``date`` to a ``QDate``."""
    return QDate(day.year, day.month, day.day)


class DateRangeBar(QWidget):
    """A checkable from/to date filter.

    Unchecked, the range is open (all dates). Checked, :meth:`date_range` returns the
    inclusive ``(from, to)`` days from the two date editors.

    Signals:
        changed: the checkbox or either date was changed by the user.
    """

    changed = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.check = QCheckBox("Dates", objectName="date_filter_check")
        today = date.today()
        self.from_edit = QDateEdit(to_qdate(today - timedelta(days=DEFAULT_RANGE_DAYS)), objectName="date_from")
        self.to_edit = QDateEdit(to_qdate(today), objectName="date_to")
        for edit in (self.from_edit, self.to_edit):
            edit.setDisplayFormat(DATE_FORMAT)
            edit.setCalendarPopup(True)
            edit.setEnabled(False)
            edit.dateChanged.connect(lambda _d: self.changed.emit())
        self.check.toggled.connect(self._on_toggled)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        for widget in (self.check, self.from_edit, QLabel("to"), self.to_edit):
            layout.addWidget(widget)

    def _on_toggled(self, checked: bool) -> None:
        self.from_edit.setEnabled(checked)
        self.to_edit.setEnabled(checked)
        self.changed.emit()

    def date_range(self) -> tuple[date | None, date | None]:
        """Return the inclusive ``(from, to)`` days, or ``(None, None)`` when the filter is off.

        If "from" is after "to", the two are swapped instead of silently matching nothing.
        """
        if not self.check.isChecked():
            return None, None
        first, last = self.from_edit.date().toPython(), self.to_edit.date().toPython()
        return (first, last) if first <= last else (last, first)  # picked in the wrong order: swap

    def set_range(self, first: date, last: date) -> None:
        """Turn the filter on with the given inclusive range."""
        self.from_edit.setDate(to_qdate(first))
        self.to_edit.setDate(to_qdate(last))
        self.check.setChecked(True)


class LazyRefreshWidget(QWidget):
    """A tab that reloads its data on first show and when new data arrives.

    Subclasses implement :meth:`reload`. :meth:`request_refresh` is for "data changed":
    while the widget is visible the reload happens after a short delay (so a burst of
    saves causes one reload); while hidden it only marks the widget stale, and the reload
    happens the next time it is shown. :meth:`refresh` reloads immediately.
    """

    REFRESH_DELAY_MS = 200

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._stale = True
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(self.REFRESH_DELAY_MS)
        self._timer.timeout.connect(self._on_timer)

    @property
    def is_stale(self) -> bool:
        """True if the data changed (or was never loaded) since the last reload."""
        return self._stale

    def reload(self) -> None:
        """Load the data and update the widgets (implemented by subclasses)."""
        raise NotImplementedError

    def refresh(self) -> None:
        """Reload now."""
        self._timer.stop()
        self._stale = False
        self.reload()

    def _on_timer(self) -> None:
        """Reload if still visible; if it was hidden meanwhile, just mark it stale."""
        if self.isVisible():
            self.refresh()
        else:
            self._stale = True

    def request_refresh(self) -> None:
        """Note that the underlying data changed; reload soon if visible, else when shown."""
        if self.isVisible():
            self._timer.start()
        else:
            self._stale = True

    def showEvent(self, event: QShowEvent) -> None:  # noqa: N802 - Qt override
        """Reload when shown after the data changed."""
        super().showEvent(event)
        if self._stale:
            self.refresh()
