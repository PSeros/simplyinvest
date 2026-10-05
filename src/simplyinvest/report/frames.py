"""Pandas views of what an appraisal, a simulation or a sweep came to.

Every function returns a frame and nothing else; none of them prints, styles or
writes a file.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

from simplyinvest._optional import require_extra
from simplyinvest.appraisal import Appraisal, ComparisonResult
from simplyinvest.cashflow import Component, Role
from simplyinvest.uncertain import DEFAULT_PERCENTILES

if TYPE_CHECKING:
    from pandas import DataFrame

    from simplyinvest.financing import AmortisationSchedule
    from simplyinvest.uncertain import ScenarioTable, Simulation, Sweep, Tornado

__all__ = [
    "breakdown_frame",
    "draws_frame",
    "flows_frame",
    "ranking_frame",
    "scenario_frame",
    "schedule_frame",
    "simulation_frame",
    "sweep_frame",
    "tornado_frame",
]


def _pandas() -> Any:
    """``pandas``.

    Raises:
        ImportError: naming the extra to install, if pandas is missing.
    """
    require_extra("frames", "pandas")
    import pandas  # noqa: PLC0415

    return pandas


def _appraisals(result: ComparisonResult | Appraisal) -> tuple[Appraisal, ...]:
    """``result`` as a tuple, whether it holds one appraisal or several."""
    if isinstance(result, ComparisonResult):
        return result.appraisals
    return (result,)


# --------------------------------------------------------------- appraisals


def flows_frame(appraisal: Appraisal, *, tolerance: float = 0.005) -> DataFrame:
    """One row per flow and period in which money moves.

    Columns are ``period``, ``date`` where the grid is anchored, ``role``,
    ``detail``, ``description``, ``amount`` and ``discounted``.

    Args:
        appraisal: The resolved alternative to unpack.
        tolerance: How small an amount counts as no movement.
    """
    timeline = appraisal.ctx.timeline
    factors = timeline.discount_factors()
    dates = timeline.dates() if timeline.anchored else None

    rows = []
    for flow in appraisal.series:
        amounts = np.asarray(flow.amounts(timeline), dtype=np.float64)
        for period in np.flatnonzero(np.abs(amounts) > tolerance):
            amount = float(amounts[period])
            row: dict[str, Any] = {
                "period": int(period),
                "role": flow.label.role.value,
                "detail": flow.label.detail,
                "description": flow.description,
                "amount": amount,
                "discounted": amount * float(factors[period]),
            }
            if dates is not None:
                row["date"] = dates[period]
            rows.append(row)

    order = ["period", "date", "role", "detail", "description", "amount", "discounted"]
    frame = _pandas().DataFrame(rows)
    if frame.empty:
        return frame
    return frame[[column for column in order if column in frame]].sort_values(
        ["period", "role", "detail"], ignore_index=True
    )


def breakdown_frame(
    result: ComparisonResult | Appraisal, *, by: type[Component] | type[Role] = Component
) -> DataFrame:
    """Present value per component, one column per alternative.

    Args:
        result: One appraisal or a whole comparison.
        by: ``Component`` for the full label, ``Role`` to roll details up.
    """
    columns = {
        appraisal.name: {
            str(label): float(value) for label, value in appraisal.breakdown(by=by).items()
        }
        for appraisal in _appraisals(result)
    }
    frame = _pandas().DataFrame(columns).fillna(0.0).sort_index()
    frame.index.name = "component"
    return frame


def ranking_frame(result: ComparisonResult | Appraisal) -> DataFrame:
    """One row per alternative, best net present value first.

    Args:
        result: One appraisal or a whole comparison.
    """
    appraisals = sorted(_appraisals(result), key=lambda a: float(a.npv), reverse=True)
    frame = _pandas().DataFrame(
        [
            {
                "alternative": appraisal.name,
                "npv": float(appraisal.npv),
                "pv_of_costs": float(appraisal.pv_of_costs),
                "eac": float(appraisal.eac),
                "irr": appraisal.irr.value,
                "payback_months": _months(appraisal.payback),
                "discounted_payback_months": _months(appraisal.discounted_payback),
                "profitability_index": float(appraisal.profitability_index),
            }
            for appraisal in appraisals
        ]
    )
    return frame.set_index("alternative")


def _months(term: Any) -> float | None:
    """``term`` in whole months, or ``None`` where there is no term."""
    return None if term is None else float(term.months)


def schedule_frame(schedule: AmortisationSchedule) -> DataFrame:
    """One row per instalment, with the balance either side of it.

    Args:
        schedule: The amortisation to tabulate.
    """
    frame = _pandas().DataFrame(
        {
            "period": schedule.period,
            "opening": schedule.opening,
            "interest": schedule.interest,
            "principal": schedule.principal,
            "payment": schedule.payments,
            "closing": schedule.closing,
        }
    )
    return frame.set_index("period")


# -------------------------------------------------------------- uncertainty


def simulation_frame(
    simulation: Simulation, *, levels: Sequence[float] = DEFAULT_PERCENTILES
) -> DataFrame:
    """One row per alternative, summarising its spread of outcomes.

    Args:
        simulation: The trials to summarise.
        levels: Which percentiles to report.
    """
    shares = simulation.win_share()
    rows = []
    for name in simulation.names:
        row: dict[str, Any] = {
            "alternative": name,
            "expected": simulation.expected_npv(name),
        }
        row |= {
            f"p{level:g}": value
            for level, value in simulation.npv_percentiles(name, levels).items()
        }
        row |= {
            "p_positive": simulation.probability_npv_positive(name),
            "value_at_risk": simulation.value_at_risk(name),
            "win_share": shares[name],
            "regret": simulation.regret(name),
        }
        rows.append(row)
    frame = _pandas().DataFrame(rows).set_index("alternative")
    return frame.sort_values("expected", ascending=False)


def draws_frame(simulation: Simulation) -> DataFrame:
    """One row per trial: what every parameter was drawn at, and what it came to.

    Args:
        simulation: The trials to unpack.
    """
    columns: dict[str, Any] = dict(simulation.draws)
    columns |= {name: simulation.values(name) for name in simulation.names}
    frame = _pandas().DataFrame(columns)
    frame.index.name = "trial"
    return frame


def sweep_frame(sweep: Sweep) -> DataFrame:
    """One row per swept value, with the best alternative at each.

    Args:
        sweep: The parameter walk to tabulate.
    """
    frame = _pandas().DataFrame(
        {name: sweep.npv[row] for row, name in enumerate(sweep.names)}, index=sweep.values
    )
    frame.index.name = sweep.label
    frame["best"] = list(sweep.winners())
    return frame


def tornado_frame(tornado: Tornado) -> DataFrame:
    """One row per parameter, widest swing first.

    Args:
        tornado: The swings to tabulate.
    """
    frame = _pandas().DataFrame(
        [
            {"parameter": bar.label, "low": bar.low, "high": bar.high, "swing": bar.swing}
            for bar in tornado.bars
        ]
    )
    return frame.set_index("parameter")


def scenario_frame(table: ScenarioTable) -> DataFrame:
    """One row per scenario, with the best alternative in each.

    Args:
        table: The scenarios to tabulate.
    """
    frame = _pandas().DataFrame(table.npv, index=list(table.scenarios), columns=list(table.names))
    frame.index.name = "scenario"
    frame["best"] = [table.winner_in(scenario) for scenario in table.scenarios]
    return frame
