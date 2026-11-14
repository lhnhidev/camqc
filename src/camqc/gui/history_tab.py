"""History tab: search and browse stored results by serial number, date, and verdict.

A filter bar (serial substring, date range, verdict) drives a sortable table of runs;
double-clicking a run opens a dialog with its per-check results.
"""

from __future__ import annotations

import html
import logging
import sqlite3
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from PySide6.QtCore import (
    QAbstractTableModel,
    QModelIndex,
    QPersistentModelIndex,
    QSortFilterProxyModel,
    Qt,
    QTimer,
)
from PySide6.QtGui import QBrush, QColor
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QTableView,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from camqc import db
from camqc.cli import format_limits
from camqc.gui.common import DateRangeBar, LazyRefreshWidget, colors_for

logger = logging.getLogger(__name__)

MAX_ROWS = 1000
SEARCH_DELAY_MS = 250  # wait for typing to pause before searching
VERDICT_FILTERS = ("All verdicts", "PASS", "FAIL", "INCOMPLETE")
RUN_HEADERS = ("Run", "Started", "Serial", "Model", "Station", "Verdict", "Duration (ms)")
RUN_KEYS = ("id", "started_at", "serial", "model", "station", "verdict", "duration_ms")
NUMERIC_COLUMNS = {0, 6}
SERIAL_COLUMN = 2
VERDICT_COLUMN = 5
DETAIL_HEADERS = ("Check", "Value", "Unit", "Limits", "Status", "Note")


def detail_rows(detail: dict[str, Any]) -> list[tuple[list[str], str]]:
    """Return ``(cells, status)`` per check of a :func:`camqc.db.get_run_detail` result."""
    rows = []
    for r in detail["results"]:
        has_value = r["value"] is not None
        note = r.get("reason") or r["details"].get("reason", "")
        cells = [
            r["check_name"],
            f"{r['value']:.5g}" if has_value else "-",
            r["unit"] or "-",
            format_limits(r["low"], r["high"]) if has_value else "-",
            r["status"],
            note,
        ]
        rows.append((cells, r["status"]))
    return rows


class RunsTableModel(QAbstractTableModel):
    """Table model of run rows (dicts from :mod:`camqc.db`).

    ``Qt.UserRole`` returns the raw value so the proxy sorts numbers numerically.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._runs: list[dict[str, Any]] = []

    def set_runs(self, runs: Sequence[dict[str, Any]]) -> None:
        """Replace all rows."""
        self.beginResetModel()
        self._runs = list(runs)
        self.endResetModel()

    def run_id_at(self, row: int) -> int:
        """Return the ``runs.id`` shown in ``row``."""
        return int(self._runs[row]["id"])

    def rowCount(self, parent: QModelIndex | QPersistentModelIndex = QModelIndex()) -> int:  # noqa: B008, N802
        """Number of runs."""
        return 0 if parent.isValid() else len(self._runs)

    def columnCount(self, parent: QModelIndex | QPersistentModelIndex = QModelIndex()) -> int:  # noqa: B008, N802
        """Number of columns."""
        return 0 if parent.isValid() else len(RUN_HEADERS)

    def headerData(  # noqa: N802
        self, section: int, orientation: Qt.Orientation, role: int = Qt.ItemDataRole.DisplayRole
    ) -> Any:
        """Column titles."""
        if orientation == Qt.Orientation.Horizontal and role == Qt.ItemDataRole.DisplayRole:
            return RUN_HEADERS[section]
        return None

    def data(self, index: QModelIndex | QPersistentModelIndex, role: int = Qt.ItemDataRole.DisplayRole) -> Any:
        """Cell text, raw sort value, verdict colors and number alignment."""
        if not index.isValid():
            return None
        column = index.column()
        value = self._runs[index.row()][RUN_KEYS[column]]
        if role == Qt.ItemDataRole.DisplayRole:
            return str(value)
        if role == Qt.ItemDataRole.UserRole:
            return value
        if role == Qt.ItemDataRole.TextAlignmentRole and column in NUMERIC_COLUMNS:
            return int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        if column == VERDICT_COLUMN and role in (Qt.ItemDataRole.BackgroundRole, Qt.ItemDataRole.ForegroundRole):
            bg, fg = colors_for(str(value))
            return QBrush(QColor(bg if role == Qt.ItemDataRole.BackgroundRole else fg))
        return None


class RunDetailDialog(QDialog):
    """A dialog listing one run's header and its per-check results."""

    def __init__(self, detail: dict[str, Any], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"Run #{detail['id']} - {detail['serial']}")
        self.resize(820, 420)
        bg, fg = colors_for(detail["verdict"])
        esc = {k: html.escape(str(detail[k])) for k in ("serial", "model", "station", "started_at", "verdict")}
        self.header = QLabel(  # free-text fields are escaped: a serial like "<SN-1>" must show literally
            f"<b>Serial</b> {esc['serial']} &nbsp; <b>Model</b> {esc['model']} &nbsp; "
            f"<b>Station</b> {esc['station']} &nbsp; <b>Started</b> {esc['started_at']} &nbsp; "
            f"<b>Duration</b> {detail['duration_ms']} ms &nbsp; "
            f"<span style='background:{bg}; color:{fg}; padding:2px 8px;'><b>{esc['verdict']}</b></span>"
        )
        self.header.setObjectName("detail_header")
        self.header.setTextFormat(Qt.TextFormat.RichText)
        self.header.setWordWrap(True)

        rows = detail_rows(detail)
        self.table = QTableWidget(len(rows), len(DETAIL_HEADERS), objectName="detail_table")
        self.table.setHorizontalHeaderLabels(list(DETAIL_HEADERS))
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        for r, (cells, status) in enumerate(rows):
            row_bg, row_fg = colors_for(status)
            for c, text in enumerate(cells):
                item = QTableWidgetItem(text)
                item.setBackground(QBrush(QColor(row_bg)))
                item.setForeground(QBrush(QColor(row_fg)))
                self.table.setItem(r, c, item)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setStretchLastSection(True)

        layout = QVBoxLayout(self)
        layout.addWidget(self.header)
        layout.addWidget(self.table, 1)


class HistoryTab(LazyRefreshWidget):
    """Search box, filters and a table of stored runs; double-click opens the run's results."""

    def __init__(self, db_path: str | Path, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._db_path = db_path
        self.detail_dialog: RunDetailDialog | None = None
        self._search_timer = QTimer(self)
        self._search_timer.setSingleShot(True)
        self._search_timer.setInterval(SEARCH_DELAY_MS)
        self._search_timer.timeout.connect(self.refresh)
        self._build_ui()

    def _build_ui(self) -> None:
        """Create the filter bar, the table and the status labels."""
        self.serial_edit = QLineEdit(objectName="history_serial_edit")
        self.serial_edit.setPlaceholderText("Search serial (contains)...")
        self.serial_edit.setClearButtonEnabled(True)
        self.serial_edit.textChanged.connect(lambda _t: self._search_timer.start())
        self.serial_edit.returnPressed.connect(self.refresh)
        self.date_bar = DateRangeBar()
        self.date_bar.changed.connect(self.refresh)
        self.verdict_combo = QComboBox(objectName="history_verdict_combo")
        self.verdict_combo.addItems(list(VERDICT_FILTERS))
        self.verdict_combo.currentIndexChanged.connect(lambda _i: self.refresh())
        self.search_button = QPushButton("Search", objectName="history_search_button")
        self.search_button.clicked.connect(self.refresh)
        self.clear_button = QPushButton("Clear", objectName="history_clear_button")
        self.clear_button.clicked.connect(self.clear_filters)

        bar = QHBoxLayout()
        bar.addWidget(self.serial_edit, 2)
        bar.addWidget(self.date_bar)
        bar.addWidget(self.verdict_combo)
        bar.addWidget(self.search_button)
        bar.addWidget(self.clear_button)

        self.model = RunsTableModel(self)
        self.proxy = QSortFilterProxyModel(self)
        self.proxy.setSourceModel(self.model)
        self.proxy.setSortRole(Qt.ItemDataRole.UserRole)
        self.table = QTableView(objectName="history_table")
        self.table.setModel(self.proxy)
        self.table.setSortingEnabled(True)
        self.table.sortByColumn(0, Qt.SortOrder.DescendingOrder)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(SERIAL_COLUMN, QHeaderView.ResizeMode.Stretch)  # keeps numbers near their titles
        self.table.doubleClicked.connect(self._on_double_clicked)

        self.empty_label = QLabel("No runs found.", objectName="history_empty_label")
        self.empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.empty_label.setStyleSheet("color: #808080; font-size: 14pt; padding: 24px;")
        self.count_label = QLabel("", objectName="history_count_label")
        self.message_label = QLabel("", objectName="history_message_label", wordWrap=True)
        self.message_label.setStyleSheet("color: #c00000;")
        hint = QLabel("Double-click a run to see its check results.")
        hint.setStyleSheet("color: #808080;")
        footer = QHBoxLayout()
        footer.addWidget(self.count_label)
        footer.addStretch(1)
        footer.addWidget(hint)

        layout = QVBoxLayout(self)
        layout.addLayout(bar)
        layout.addWidget(self.table, 1)
        layout.addWidget(self.empty_label)
        layout.addWidget(self.message_label)
        layout.addLayout(footer)

    # -- data --

    def verdict_filter(self) -> str | None:
        """The selected verdict, or ``None`` for all."""
        return None if self.verdict_combo.currentIndex() == 0 else self.verdict_combo.currentText()

    def clear_filters(self) -> None:
        """Reset the search box, dates and verdict, then reload."""
        for widget in (self.serial_edit, self.date_bar, self.verdict_combo):
            widget.blockSignals(True)
        self.serial_edit.clear()
        self.date_bar.check.setChecked(False)
        self.verdict_combo.setCurrentIndex(0)
        for widget in (self.serial_edit, self.date_bar, self.verdict_combo):
            widget.blockSignals(False)
        self.refresh()

    def reload(self) -> None:
        """Query the database with the current filters and fill the table."""
        self._search_timer.stop()
        date_from, date_to = self.date_bar.date_range()
        try:
            conn = db.init_db(self._db_path)
            try:
                runs = db.search_runs(
                    conn, self.serial_edit.text().strip(), date_from, date_to,
                    self.verdict_filter(), limit=MAX_ROWS + 1,
                )
            finally:
                conn.close()
        except (sqlite3.Error, OSError) as exc:
            logger.exception("history query failed")
            self.model.set_runs([])
            self.message_label.setText(f"Cannot read the database: {exc}")
            self._show_counts(0, truncated=False)
            return
        self.message_label.setText("")
        truncated = len(runs) > MAX_ROWS
        self.model.set_runs(runs[:MAX_ROWS])
        self._show_counts(min(len(runs), MAX_ROWS), truncated)

    def _show_counts(self, shown: int, truncated: bool) -> None:
        """Update the count label and the empty-state label."""
        self.empty_label.setVisible(shown == 0)
        self.table.setVisible(shown > 0)
        text = f"{shown} run{'' if shown == 1 else 's'}"
        self.count_label.setText(f"{text} (first {MAX_ROWS} shown; narrow the filters)" if truncated else text)

    # -- details --

    def _on_double_clicked(self, index: QModelIndex) -> None:
        """Open the detail dialog of the double-clicked row."""
        source = self.proxy.mapToSource(index)
        if source.isValid():
            self.open_run_detail(self.model.run_id_at(source.row()))

    def open_run_detail(self, run_id: int) -> RunDetailDialog | None:
        """Show a dialog with the per-check results of ``run_id`` (``None`` if it vanished)."""
        try:
            conn = db.init_db(self._db_path)
            try:
                detail = db.get_run_detail(conn, run_id)
            finally:
                conn.close()
        except (sqlite3.Error, OSError) as exc:
            logger.exception("could not load run %s", run_id)
            self.message_label.setText(f"Cannot read the database: {exc}")
            return None
        if detail is None:
            self.message_label.setText(f"Run #{run_id} no longer exists.")
            return None
        self.message_label.setText("")
        dialog = RunDetailDialog(detail, self)
        dialog.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self.detail_dialog = dialog

        def forget(_obj: object = None, closed: RunDetailDialog = dialog) -> None:
            if self.detail_dialog is closed:  # a newer dialog may have replaced it
                self.detail_dialog = None

        dialog.destroyed.connect(forget)
        dialog.show()
        return dialog
