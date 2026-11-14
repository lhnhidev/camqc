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
