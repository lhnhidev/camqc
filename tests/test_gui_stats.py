"""Tests for the Statistics tab and its matplotlib charts."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("matplotlib")

from matplotlib.backends.backend_agg import FigureCanvasAgg  # noqa: E402
from matplotlib.colors import to_hex  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from camqc import db  # noqa: E402
from camqc.gui import stats_tab  # noqa: E402
from camqc.gui.stats_tab import StatsTab  # noqa: E402
from gui_support import wait_for  # noqa: E402
from seed_support import PASSING, add_run, seed  # noqa: E402


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    path = tmp_path / "t.db"
    conn = db.init_db(path)
    seed(conn)
    conn.close()
    return path


@pytest.fixture
def tab(qapp: QApplication, db_path: Path) -> Iterator[StatsTab]:
    widget = StatsTab(db_path)
    widget.resize(1000, 800)
    widget.show()  # the first show loads and draws
    yield widget
    widget.close()


def axes_of(widget: StatsTab) -> tuple[Any, Any]:
    top, bottom = widget.figure.axes
    return top, bottom


def bar_heights(ax: Any) -> list[float]:
    return [round(p.get_height(), 1) for p in ax.patches]


def texts(ax: Any) -> list[str]:
    return [t.get_text() for t in ax.texts]


def pixel_width(ax: Any, data_width: float) -> float:
    return float(ax.transData.transform((data_width, 0))[0] - ax.transData.transform((0, 0))[0])


def plain_figure() -> Figure:
    """A Figure with an Agg canvas, as used without Qt."""
    fig = Figure(figsize=(9, 7))
    FigureCanvasAgg(fig)
    return fig


def days(n: int, yields: Sequence[float] | None = None) -> list[dict[str, Any]]:
    values = yields or [50.0 + i for i in range(n)]
    return [{"day": f"2026-10-{i + 1:02d}", "total": 10, "passed": 5, "failed": 5, "incomplete": 0,
             "yield_pct": values[i]} for i in range(n)]


def pareto(n: int) -> list[dict[str, Any]]:
    return [{"check_name": f"check_{i}", "fail_count": n - i, "pct": 100.0 / n, "cumulative_pct": 100.0 * (i + 1) / n}
            for i in range(n)]


# --- the tab with data -------------------------------------------------------------------


def test_first_show_loads_and_draws_both_charts(tab: StatsTab) -> None:
    assert [d["day"] for d in tab.days] == ["2026-11-14", "2026-11-15", "2026-11-16"]
    assert [(p["check_name"], p["fail_count"]) for p in tab.pareto] == [("sharpness", 2), ("blemish", 1)]
    top, bottom = axes_of(tab)
    assert bar_heights(top) == [66.7, 0.0, 100.0]  # yield % per day
    assert bar_heights(bottom) == [66.7, 33.3]  # share of all check failures
    assert [round(y, 1) for y in bottom.lines[0].get_ydata()] == [66.7, 100.0]  # cumulative share
    assert top.get_title(loc="left") == "Yield by day" and bottom.get_title(loc="left") == "Pareto of failing checks"


def test_there_is_no_second_y_axis(tab: StatsTab) -> None:
    assert len(tab.figure.axes) == 2  # one axes per chart: no twinx
    for ax in axes_of(tab):
        assert ax.get_ylim() == (0, 112)  # both on the same 0-100 % scale (plus label headroom)
        assert [t.get_text() for t in ax.get_yticklabels()] == ["0%", "25%", "50%", "75%", "100%"]


def test_labels_are_selective_and_text_uses_ink_tokens(tab: StatsTab) -> None:
    top, bottom = axes_of(tab)
    assert texts(top) == ["67%", "0%", "100%"]
    assert texts(bottom) == ["2", "1"]  # failure counts at the bar tips
    assert {to_hex(t.get_color()) for t in top.texts + bottom.texts} == {stats_tab.INK_SECONDARY}
    assert [t.get_text() for t in top.get_xticklabels()] == [
        "2026-11-14\nn=3", "2026-11-15\nn=2", "2026-11-16\nn=1"]  # sample sizes ride on the tick labels


def test_series_colors_and_legend_rules(tab: StatsTab) -> None:
    top, bottom = axes_of(tab)
    assert {to_hex(p.get_facecolor()) for p in top.patches} == {stats_tab.SERIES_BLUE}
    assert top.get_legend() is None  # a single series needs no legend; the title names it
    assert to_hex(bottom.lines[0].get_color()) == stats_tab.SERIES_ORANGE and bottom.lines[0].get_linewidth() == 2
    legend = bottom.get_legend()  # two series: a legend is always present
    assert legend is not None and len(legend.get_texts()) == 2
    for ax in (top, bottom):
        assert ax.yaxis.get_gridlines()[0].get_linestyle() == "-" and not ax.spines["top"].get_visible()


def test_summary_and_data_tables(tab: StatsTab) -> None:
    assert tab.summary_label.text() == "6 runs - overall yield 50.0% - 3 failing check results"
    assert not tab.tables.isVisibleTo(tab)
    tab.show_table.setChecked(True)
    assert tab.tables.isVisibleTo(tab)
    rows = [[tab.yield_table.item(r, c).text() for c in range(6)] for r in range(tab.yield_table.rowCount())]
    assert rows == [["2026-11-14", "3", "2", "1", "0", "66.7"], ["2026-11-15", "2", "0", "1", "1", "0.0"],
                    ["2026-11-16", "1", "1", "0", "0", "100.0"]]
    prows = [[tab.pareto_table.item(r, c).text() for c in range(4)] for r in range(tab.pareto_table.rowCount())]
    assert prows == [["sharpness", "2", "66.7", "66.7"], ["blemish", "1", "33.3", "100.0"]]
    tab.show_table.setChecked(False)
    assert not tab.tables.isVisibleTo(tab)


def test_date_range_limits_both_charts(tab: StatsTab) -> None:
    tab.date_bar.set_range(date(2026, 11, 15), date(2026, 11, 15))  # reloads by itself
    top, bottom = axes_of(tab)
    assert bar_heights(top) == [0.0] and bar_heights(bottom) == [100.0]
    assert tab.summary_label.text() == "2 runs - overall yield 0.0% - 1 failing check results"
    tab.date_bar.check.setChecked(False)
    assert len(tab.days) == 3


# --- empty and error states ----------------------------------------------------------------


def test_empty_database_shows_placeholders_not_empty_axes(qapp: QApplication, tmp_path: Path) -> None:
    widget = StatsTab(tmp_path / "empty.db")
    widget.show()
    top, bottom = axes_of(widget)
    assert texts(top) == [stats_tab.NO_RUNS_TEXT] and texts(bottom) == [stats_tab.NO_FAILURES_TEXT]
    assert len(top.patches) == 0 and len(bottom.patches) == 0 and top.get_xticks().size == 0
    assert widget.summary_label.text() == "No runs" and widget.message_label.text() == ""
    assert widget.yield_table.rowCount() == 0 and widget.pareto_table.rowCount() == 0
    widget.close()


def test_runs_without_failures_show_yield_and_a_good_news_placeholder(qapp: QApplication, tmp_path: Path) -> None:
    conn = db.init_db(tmp_path / "ok.db")
    add_run(conn, "P1", datetime(2026, 11, 14, 8), PASSING)
    add_run(conn, "P2", datetime(2026, 11, 14, 9), PASSING)
    conn.close()
    widget = StatsTab(tmp_path / "ok.db")
    widget.show()
    top, bottom = axes_of(widget)
    assert bar_heights(top) == [100.0] and texts(top) == ["100%"]
    assert texts(bottom) == [stats_tab.NO_FAILURES_TEXT] and len(bottom.patches) == 0
    assert widget.summary_label.text() == "2 runs - overall yield 100.0% - 0 failing check results"
    widget.close()


def test_range_with_no_runs_shows_placeholders(tab: StatsTab) -> None:
    tab.date_bar.set_range(date(2027, 1, 1), date(2027, 1, 31))
    top, bottom = axes_of(tab)
    assert texts(top) == [stats_tab.NO_RUNS_TEXT] and texts(bottom) == [stats_tab.NO_FAILURES_TEXT]
    assert tab.summary_label.text() == "No runs"


def test_database_error_is_reported_and_charts_show_placeholders(qapp: QApplication, tmp_path: Path) -> None:
    widget = StatsTab(tmp_path)  # a directory cannot be opened as a database
    widget.show()
    assert "Cannot read the database" in widget.message_label.text()
    assert texts(axes_of(widget)[0]) == [stats_tab.NO_RUNS_TEXT]
    widget.close()


# --- refresh behaviour ---------------------------------------------------------------------


def test_hidden_tab_is_lazy_and_loads_when_shown(qapp: QApplication, db_path: Path) -> None:
    widget = StatsTab(db_path)
    assert widget.is_stale and widget.days == []
    widget.request_refresh()
    assert widget.is_stale and widget.days == []
    widget.show()
    assert not widget.is_stale and len(widget.days) == 3
    widget.close()


def test_new_runs_refresh_a_visible_tab_once(tab: StatsTab, db_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[int] = []
    real = tab.reload
    monkeypatch.setattr(tab, "reload", lambda: (calls.append(1), real())[1])
    conn = db.init_db(db_path)
    for i in range(3):
        add_run(conn, f"N{i}", datetime(2026, 11, 17, i), PASSING)
        tab.request_refresh()
    conn.close()
    assert len(tab.days) == 3  # still the old data: the reload is delayed and coalesced
    wait_for(lambda: len(tab.days) == 4)
    assert len(calls) == 1 and tab.days[-1]["day"] == "2026-11-17" and tab.days[-1]["total"] == 3


def test_resize_redraws_so_bar_thickness_stays_capped(tab: StatsTab, monkeypatch: pytest.MonkeyPatch) -> None:
    redraws: list[int] = []
    real = tab._redraw
    monkeypatch.setattr(tab, "_redraw", lambda: (redraws.append(1), real())[1])
    tab._resize_timer.timeout.disconnect()
    tab._resize_timer.timeout.connect(tab._redraw)
    tab.resize(700, 600)
    wait_for(lambda: bool(redraws))


# --- plotting functions without Qt ---------------------------------------------------------


def test_bar_width_caps_thickness_at_24px() -> None:
    fig = Figure(figsize=(9, 7), dpi=100)
    slot_px = fig.get_figwidth() * fig.dpi * 0.85 / stats_tab.MIN_SLOTS
    assert stats_tab.bar_width(fig, 1) * slot_px == pytest.approx(stats_tab.MAX_BAR_PX)  # few bars: thin
    assert stats_tab.bar_width(fig, 100) == 0.6  # many bars: capped by the slot, not the pixels
    wide = Figure(figsize=(20, 7), dpi=100)
    assert stats_tab.bar_width(wide, 1) < stats_tab.bar_width(fig, 1)  # wider chart -> thinner in slot units


@pytest.mark.parametrize("count", [1, 3, 30])
def test_rendered_bars_are_never_much_thicker_than_24px(count: int) -> None:
    fig = plain_figure()
    stats_tab.draw_charts(fig, days(count), pareto(min(count, 5)))
    fig.canvas.draw()
    top, bottom = fig.axes
    for ax in (top, bottom):
        for patch in ax.patches:
            assert pixel_width(ax, patch.get_width()) <= stats_tab.MAX_BAR_PX + 6  # estimate of axes width


def test_many_days_label_only_the_extremes_and_thin_out_ticks() -> None:
    fig = plain_figure()
    values = [50.0 + (i * 7) % 40 for i in range(30)]
    stats_tab.draw_charts(fig, days(30, values), [])
    top = fig.axes[0]
    assert sorted(texts(top)) == sorted([f"{min(values):.0f}%", f"{max(values):.0f}%"])  # not a number on every bar
    assert len(top.get_xticks()) <= 10


def test_pareto_is_capped_to_twelve_bars() -> None:
    fig = plain_figure()
    stats_tab.draw_charts(fig, [], pareto(20))
    bottom = fig.axes[1]
    assert len(bottom.patches) == stats_tab.MAX_PARETO_BARS and len(bottom.lines[0].get_xdata()) == 12


def test_plot_functions_work_on_plain_axes_and_handle_empty_input() -> None:
    fig = plain_figure()
    ax1, ax2 = fig.subplots(2, 1)
    stats_tab.plot_yield(ax1, [])
    stats_tab.plot_pareto(ax2, [])
    assert texts(ax1) == [stats_tab.NO_RUNS_TEXT] and texts(ax2) == [stats_tab.NO_FAILURES_TEXT]
    fig.canvas.draw()  # nothing raises


def test_charts_are_redrawn_from_scratch_each_time() -> None:
    fig = plain_figure()
    stats_tab.draw_charts(fig, days(3), pareto(3))
    stats_tab.draw_charts(fig, [], [])
    assert len(fig.axes) == 2 and all(len(ax.patches) == 0 for ax in fig.axes)
