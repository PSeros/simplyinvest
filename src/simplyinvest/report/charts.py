"""Matplotlib views of what an appraisal, a simulation or a sweep came to.

Every chart draws on the axes it is given and returns it.  Given none, it makes
a figure through ``pyplot``, which the caller then owns.  None of them shows a
figure or changes the global style.

    from simplyinvest.report import breakdown_chart

    fig, ax = plt.subplots()
    breakdown_chart(result, ax=ax)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from simplyinvest._optional import require_extra
from simplyinvest.appraisal import Appraisal, ComparisonResult
from simplyinvest.cashflow import Component, Role

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.ticker import FuncFormatter

    from simplyinvest.financing import AmortisationSchedule
    from simplyinvest.timeline import Timeline
    from simplyinvest.uncertain import ScenarioTable, Simulation, Sweep, Switch, Tornado

__all__ = [
    "balance_chart",
    "breakdown_chart",
    "cumulative_chart",
    "differential_chart",
    "distribution_chart",
    "money_formatter",
    "scenario_chart",
    "schedule_chart",
    "sweep_chart",
    "tornado_chart",
]

DEFAULT_BINS = 70
"""How many bins a histogram uses unless another count is given."""

GROUP_WIDTH = 0.76
"""The share of one row or column a whole group of bars occupies."""

_HEADROOM = 0.18
"""Extra span left at the labelled end of a tornado's axis."""

_MARGIN = 0.05
"""Extra span left at the unlabelled ends of an axis."""

_PAIR = 2
"""How many alternatives a crossing is drawn between."""


def _pyplot() -> Any:
    """``matplotlib.pyplot``.

    Raises:
        ImportError: naming the extra to install, if matplotlib is missing.
    """
    require_extra("viz", "matplotlib")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    return plt


def _axes(ax: Axes | None) -> Axes:
    """``ax`` if given, otherwise a new axes on its own figure."""
    if ax is not None:
        return ax
    _, fresh = _pyplot().subplots()
    return fresh  # type: ignore[no-any-return]


def money_formatter(*, decimals: int = 0) -> FuncFormatter:
    """A tick formatter writing money with thousands separated.

    Raises:
        ImportError: naming the extra to install, if matplotlib is missing.
    """
    require_extra("viz", "matplotlib")
    from matplotlib.ticker import FuncFormatter  # noqa: PLC0415

    return FuncFormatter(lambda value, _: f"{value:,.{decimals}f}")


def _colours(count: int) -> list[str]:
    """``count`` colours taken in order from the active property cycle."""
    cycle = _pyplot().rcParams["axes.prop_cycle"].by_key().get("color") or ["C0"]
    return [cycle[index % len(cycle)] for index in range(count)]


def _appraisals(result: ComparisonResult | Appraisal) -> tuple[Appraisal, ...]:
    """``result`` as a tuple, whether it holds one appraisal or several."""
    if isinstance(result, ComparisonResult):
        return result.appraisals
    return (result,)


def _timeline(result: ComparisonResult | Appraisal) -> Timeline:
    """The grid every appraisal in ``result`` was resolved on."""
    return _appraisals(result)[0].ctx.timeline


def _period_axis(timeline: Timeline) -> tuple[Any, str]:
    """One horizontal value per period, and what to call the axis."""
    if timeline.anchored:
        return np.array(timeline.dates()), "date"
    return timeline.periods, "period"


def _offsets(count: int) -> Any:
    """Where each member of a group of ``count`` bars sits, relative to its tick."""
    if count == 1:
        return np.zeros(1)
    step = GROUP_WIDTH / count
    return (np.arange(count) - (count - 1) / 2.0) * step


# --------------------------------------------------------------- appraisals


def breakdown_chart(
    result: ComparisonResult | Appraisal,
    *,
    by: type[Component] | type[Role] = Component,
    ax: Axes | None = None,
) -> Axes:
    """Present value per component, one group of bars per alternative.

    Args:
        result: One appraisal or a whole comparison.
        by: ``Component`` for the full label, ``Role`` to roll details up.
        ax: Where to draw.  A new axes by default.
    """
    ax = _axes(ax)
    appraisals = _appraisals(result)
    parts = [
        {str(label): float(value) for label, value in appraisal.breakdown(by=by).items()}
        for appraisal in appraisals
    ]
    labels = sorted({label for part in parts for label in part})
    rows = np.arange(len(labels))
    height = GROUP_WIDTH / len(appraisals)

    for offset, appraisal, part, colour in zip(
        _offsets(len(appraisals)), appraisals, parts, _colours(len(appraisals)), strict=True
    ):
        ax.barh(
            rows + offset,
            [part.get(label, 0.0) for label in labels],
            height=height,
            label=appraisal.name,
            color=colour,
        )

    ax.set_yticks(rows, labels)
    ax.axvline(0.0, color="black", linewidth=0.8)
    ax.xaxis.set_major_formatter(money_formatter())
    ax.set_xlabel("present value")
    ax.invert_yaxis()
    if len(appraisals) > 1:
        ax.legend()
    return ax


def cumulative_chart(
    result: ComparisonResult | Appraisal, *, discounted: bool = True, ax: Axes | None = None
) -> Axes:
    """Money committed so far, period by period, one line per alternative.

    Args:
        result: One appraisal or a whole comparison.
        discounted: Whether to discount each period's amount before adding it.
        ax: Where to draw.  A new axes by default.
    """
    ax = _axes(ax)
    appraisals = _appraisals(result)
    timeline = _timeline(result)
    when, axis_label = _period_axis(timeline)
    weights = timeline.discount_factors() if discounted else 1.0

    running = {}
    for appraisal, colour in zip(appraisals, _colours(len(appraisals)), strict=True):
        totals = np.cumsum(np.asarray(appraisal.amounts, dtype=np.float64) * weights)
        running[appraisal.name] = totals
        ax.plot(when, totals, label=appraisal.name, color=colour, linewidth=2)

    if len(appraisals) == _PAIR:
        _mark_crossing(ax, when, *running.values())

    ax.yaxis.set_major_formatter(money_formatter())
    ax.set_xlabel(axis_label)
    ax.set_ylabel("cumulative present value" if discounted else "cumulative cash")
    if len(appraisals) > 1:
        ax.legend()
    return ax


def _mark_crossing(ax: Axes, when: Any, first: Any, second: Any) -> None:
    """A dashed rule at the first period where two running totals swap order."""
    gap = first - second
    crossings = np.flatnonzero(np.sign(gap[:-1]) != np.sign(gap[1:]))
    if crossings.size:
        ax.axvline(when[crossings[0] + 1], color="grey", linestyle="--", linewidth=1)


def schedule_chart(schedule: AmortisationSchedule, *, ax: Axes | None = None) -> Axes:
    """Each instalment split into interest and principal.

    Args:
        schedule: The amortisation to draw.
        ax: Where to draw.  A new axes by default.
    """
    ax = _axes(ax)
    interest, principal = _colours(2)
    ax.bar(schedule.period, schedule.interest, label="interest", color=interest)
    ax.bar(
        schedule.period,
        schedule.principal,
        bottom=schedule.interest,
        label="principal",
        color=principal,
    )
    ax.yaxis.set_major_formatter(money_formatter())
    ax.set_xlabel("period")
    ax.set_ylabel("per instalment")
    ax.legend()
    return ax


def balance_chart(schedule: AmortisationSchedule, *, ax: Axes | None = None) -> Axes:
    """What is still owed after each instalment.

    Args:
        schedule: The amortisation to draw.
        ax: Where to draw.  A new axes by default.
    """
    ax = _axes(ax)
    (colour,) = _colours(1)
    outstanding = np.concatenate(([schedule.opening[0]], schedule.closing))
    when = np.concatenate(([schedule.period[0] - 1], schedule.period))
    ax.plot(when, outstanding, color=colour, linewidth=2)
    ax.fill_between(when, outstanding, color=colour, alpha=0.12)
    ax.yaxis.set_major_formatter(money_formatter())
    ax.set_xlabel("period")
    ax.set_ylabel("outstanding")
    return ax


# -------------------------------------------------------------- uncertainty


def distribution_chart(
    simulation: Simulation, *, bins: int = DEFAULT_BINS, ax: Axes | None = None
) -> Axes:
    """Each alternative's spread of outcomes, on one set of bins.

    Args:
        simulation: The trials to draw.
        bins: How many bins to divide the whole range into.
        ax: Where to draw.  A new axes by default.
    """
    ax = _axes(ax)
    edges = np.histogram_bin_edges(simulation.npv, bins=bins).tolist()
    for name, colour in zip(simulation.names, _colours(len(simulation.names)), strict=True):
        values = simulation.values(name)
        ax.hist(values, bins=edges, alpha=0.55, label=name, color=colour)
        ax.axvline(float(values.mean()), color=colour, linestyle="--", linewidth=1.5)

    ax.xaxis.set_major_formatter(money_formatter())
    ax.set_xlabel("net present value")
    ax.set_ylabel("trials")
    ax.legend()
    return ax


def differential_chart(
    simulation: Simulation,
    better: str,
    worse: str,
    *,
    bins: int = DEFAULT_BINS,
    ax: Axes | None = None,
) -> Axes:
    """How much ``better`` beats ``worse`` by, trial for trial.

    Bars are coloured by which alternative that trial favoured.

    Args:
        simulation: The trials to draw.
        better: The alternative measured against the other.
        worse: The other one.
        bins: How many bins to divide the range into.
        ax: Where to draw.  A new axes by default.

    Raises:
        KeyError: if either name is not in the simulation.
    """
    ax = _axes(ax)
    difference = simulation.values(better) - simulation.values(worse)
    wins, loses = _colours(2)

    counts, edges = np.histogram(difference, bins=bins)
    centres = (edges[:-1] + edges[1:]) / 2.0
    ax.bar(
        centres,
        counts,
        width=np.diff(edges),
        color=[wins if centre >= 0.0 else loses for centre in centres],
    )
    ax.axvline(0.0, color="black", linewidth=1)
    ax.axvline(float(difference.mean()), color="grey", linestyle="--", linewidth=1)
    ax.xaxis.set_major_formatter(money_formatter())
    ax.set_xlabel(f"{better} less {worse}, in present value")
    ax.set_ylabel("trials")
    return ax


def tornado_chart(tornado: Tornado, *, ax: Axes | None = None) -> Axes:
    """Each parameter's swing around the base case, widest first.

    Args:
        tornado: The swings to draw.
        ax: Where to draw.  A new axes by default.
    """
    ax = _axes(ax)
    (colour,) = _colours(1)
    rows = np.arange(len(tornado.bars))

    for row, bar in zip(rows, tornado.bars, strict=True):
        left, right = sorted((bar.low, bar.high))
        ax.barh(row, right - left, left=left, height=0.55, color=colour)
        ax.text(right, float(row), f"  {bar.swing:,.0f}", va="center", fontsize=9)

    # Bar.low is the value at the parameter's low end, which may be the larger
    # number, so the limits come from every endpoint rather than from low/high.
    ends = [tornado.base, *(value for bar in tornado.bars for value in (bar.low, bar.high))]
    span = (max(ends) - min(ends)) or 1.0
    ax.set_xlim(min(ends) - _MARGIN * span, max(ends) + _HEADROOM * span)

    ax.axvline(tornado.base, color="black", linestyle="--", linewidth=1)
    ax.set_yticks(rows, [bar.label for bar in tornado.bars])
    ax.xaxis.set_major_formatter(money_formatter())
    ax.set_xlabel(f"net present value of {tornado.on}")
    ax.invert_yaxis()
    return ax


def sweep_chart(sweep: Sweep, *, switch: Switch | None = None, ax: Axes | None = None) -> Axes:
    """Present value against one parameter, one line per alternative.

    Args:
        sweep: The parameter walk to draw.
        switch: A value to rule off, where the ranking turns.
        ax: Where to draw.  A new axes by default.
    """
    ax = _axes(ax)
    for row, (name, colour) in enumerate(zip(sweep.names, _colours(len(sweep.names)), strict=True)):
        ax.plot(sweep.values, sweep.npv[row], label=name, color=colour, linewidth=2)

    if switch is not None:
        ax.axvline(switch.value, color="grey", linestyle="--", linewidth=1)

    ax.yaxis.set_major_formatter(money_formatter())
    ax.set_xlabel(sweep.label)
    ax.set_ylabel("net present value")
    ax.legend()
    return ax


def scenario_chart(table: ScenarioTable, *, ax: Axes | None = None) -> Axes:
    """Present value in every scenario, one group of bars per scenario.

    Args:
        table: The scenarios to draw.
        ax: Where to draw.  A new axes by default.
    """
    ax = _axes(ax)
    columns = np.arange(len(table.scenarios))
    width = GROUP_WIDTH / len(table.names)

    for index, (offset, name, colour) in enumerate(
        zip(_offsets(len(table.names)), table.names, _colours(len(table.names)), strict=True)
    ):
        ax.bar(columns + offset, table.npv[:, index], width=width, label=name, color=colour)

    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xticks(columns, table.scenarios)
    ax.yaxis.set_major_formatter(money_formatter())
    ax.set_ylabel("net present value")
    ax.legend()
    return ax
