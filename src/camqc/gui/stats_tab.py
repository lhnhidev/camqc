"""Statistics tab: yield and per-check failure distributions plotted with matplotlib.

Two stacked charts on one ``FigureCanvasQTAgg``: yield % per day (bars) and a Pareto of
failing checks (bars = share of all check failures, line = cumulative share). Both use the
same 0-100 % axis (no dual axis). The plotting functions work on plain matplotlib ``Axes``
so they can be used and tested without Qt.
"""

from __future__ import annotations

import logging
import math
import os
import sqlite3
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

os.environ.setdefault("QT_API", "pyside6")  # make matplotlib pick the binding we use

from matplotlib.axes import Axes  # noqa: E402
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from PySide6.QtCore import QTimer  # noqa: E402
from PySide6.QtGui import QResizeEvent  # noqa: E402
from PySide6.QtWidgets import (  # noqa: E402
    QAbstractItemView,
    QCheckBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from camqc import db  # noqa: E402
from camqc.gui.common import DateRangeBar, LazyRefreshWidget  # noqa: E402

logger = logging.getLogger(__name__)

# Chart colors from the validated reference palette (light surface).
SURFACE = "#fcfcfb"
INK_PRIMARY, INK_SECONDARY, INK_MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, BASELINE = "#e1e0d9", "#c3c2b7"
SERIES_BLUE, SERIES_ORANGE = "#2a78d6", "#eb6834"  # categorical slots 1 and 2

MAX_BAR_PX = 24  # bars are never thicker than this
MIN_SLOTS = 8  # so a single day does not fill the chart
PARETO_MIN_SLOTS = 4  # a Pareto usually has only a few checks
TIP_LABEL_OFFSET_PT = 3
PARETO_LABEL_OFFSET_PT = 7  # clear of the cumulative-line marker that sits on the first bar's tip
TIP_LABEL_LIMIT = 12  # label every bar tip up to this many bars; beyond, only the extremes
MAX_PARETO_BARS = 12
RESIZE_REDRAW_MS = 120

NO_RUNS_TEXT = "No runs in the selected range"
NO_FAILURES_TEXT = "No failing checks in this range"
YIELD_TITLE = "Yield by day"
PARETO_TITLE = "Pareto of failing checks"


# --- plotting (pure matplotlib) --------------------------------------------------------


def bar_width(fig: Figure, slots: int, min_slots: int = MIN_SLOTS) -> float:
    """Return the bar width (in slot units, <= 0.6) that keeps bars at most ``MAX_BAR_PX`` thick.

    Args:
        fig: The figure (its pixel width gives the slot size).
        slots: Number of bars.
        min_slots: Slots assumed at least, so a few bars do not fill the chart.
    """
    usable_px = fig.get_figwidth() * fig.dpi * 0.85  # axes width, roughly
    slot_px = usable_px / max(slots, min_slots)
    return min(0.6, MAX_BAR_PX / slot_px)


def style_axes(ax: Axes, title: str) -> None:
    """Apply the quiet chart chrome: hairline solid gridlines, a baseline, muted ticks."""
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", fontsize=11, fontweight="semibold", color=INK_PRIMARY)
    ax.grid(axis="y", color=GRID, linewidth=0.8, linestyle="-")
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE)
    ax.tick_params(axis="both", length=0, labelsize=8.5, labelcolor=INK_SECONDARY)


def draw_placeholder(ax: Axes, title: str, text: str) -> None:
    """Show a centered message instead of a chart."""
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", fontsize=11, fontweight="semibold", color=INK_PRIMARY)
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ax.spines.values():
        side.set_visible(False)
    ax.text(0.5, 0.5, text, ha="center", va="center", transform=ax.transAxes, fontsize=11, color=INK_MUTED)


def _tick_step(count: int, max_ticks: int = 10) -> int:
    return max(1, math.ceil(count / max_ticks))


def _label_indices(values: Sequence[float]) -> set[int]:
    """Indices whose tip gets a value label: all of them when few, else the extremes."""
    if len(values) <= TIP_LABEL_LIMIT:
        return set(range(len(values)))
    return {values.index(min(values)), values.index(max(values))}


def plot_yield(ax: Axes, days: Sequence[Mapping[str, Any]]) -> None:
    """Draw yield % per day as bars (one series, so no legend; the title names it).

    Args:
        ax: Target axes.
        days: Rows from :func:`camqc.db.yield_by_day`; empty shows a placeholder.
    """
    if not days:
        draw_placeholder(ax, YIELD_TITLE, NO_RUNS_TEXT)
        return
    values = [float(d["yield_pct"]) for d in days]
    xs = list(range(len(days)))
    ax.bar(xs, values, width=bar_width(ax.figure, len(days)), color=SERIES_BLUE, zorder=2)
    step = _tick_step(len(days))
    ticks = xs[::step]
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{days[i]['day']}\nn={days[i]['total']}" for i in ticks])
    pad = max(0, MIN_SLOTS - len(days)) / 2
    ax.set_xlim(-0.5 - pad, len(days) - 0.5 + pad)
    ax.set_ylim(0, 112)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
    style_axes(ax, YIELD_TITLE)
    for i in _label_indices(values):
        ax.annotate(f"{values[i]:.0f}%", (i, values[i]), xytext=(0, TIP_LABEL_OFFSET_PT), textcoords="offset points",
                    ha="center", va="bottom", fontsize=8.5, color=INK_SECONDARY)


def plot_pareto(ax: Axes, rows: Sequence[Mapping[str, Any]]) -> None:
    """Draw the Pareto of failing checks on a single % axis.

    Bars are each check's share of all check failures; the line is the cumulative share.
    Two series, so a legend is drawn. Counts are printed at the bar tips.

    Args:
        ax: Target axes.
        rows: Rows from :func:`camqc.db.pareto_failures`; empty shows a placeholder.
    """
    if not rows:
        draw_placeholder(ax, PARETO_TITLE, NO_FAILURES_TEXT)
        return
    rows = list(rows)[:MAX_PARETO_BARS]
    xs = list(range(len(rows)))
    pct = [float(r["pct"]) for r in rows]
    cumulative = [float(r["cumulative_pct"]) for r in rows]
    ax.bar(xs, pct, width=bar_width(ax.figure, len(rows), PARETO_MIN_SLOTS), color=SERIES_BLUE, zorder=2)
    ax.plot(xs, cumulative, color=SERIES_ORANGE, linewidth=2, solid_capstyle="round", solid_joinstyle="round",
            marker="o", markersize=6, markerfacecolor=SERIES_ORANGE, markeredgecolor=SURFACE,
            markeredgewidth=1.5, zorder=3)
    ax.set_xticks(xs)
    ax.set_xticklabels([r["check_name"] for r in rows], rotation=20 if len(rows) > 5 else 0,
                       ha="right" if len(rows) > 5 else "center")
    pad = max(0, PARETO_MIN_SLOTS - len(rows)) / 2
    ax.set_xlim(-0.5 - pad, len(rows) - 0.5 + pad)
    ax.set_ylim(0, 112)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.set_ylabel("% of all check failures", fontsize=8.5, color=INK_SECONDARY)
    style_axes(ax, PARETO_TITLE)
    for x, row, share in zip(xs, rows, pct):
        ax.annotate(f"{row['fail_count']}", (x, share), xytext=(0, PARETO_LABEL_OFFSET_PT), textcoords="offset points",
                    ha="center", va="bottom", fontsize=8.5, color=INK_SECONDARY)
    handles = [
        Line2D([0], [0], marker="s", linestyle="", markersize=8, markerfacecolor=SERIES_BLUE,
               markeredgecolor=SERIES_BLUE, label="Share of failures (bar; number = count)"),
        Line2D([0], [0], color=SERIES_ORANGE, linewidth=2, marker="o", markersize=6,
               markerfacecolor=SERIES_ORANGE, markeredgecolor=SURFACE, label="Cumulative share (line)"),
    ]
    ax.legend(handles=handles, loc="center right", frameon=False, fontsize=8.5, labelcolor=INK_SECONDARY)


def draw_charts(fig: Figure, days: Sequence[Mapping[str, Any]], pareto: Sequence[Mapping[str, Any]]) -> None:
    """Clear ``fig`` and draw the yield and Pareto charts on it (stacked)."""
    fig.clear()
    fig.set_facecolor(SURFACE)
    fig.set_layout_engine("constrained")
    top, bottom = fig.subplots(2, 1)
    plot_yield(top, days)
    plot_pareto(bottom, pareto)
