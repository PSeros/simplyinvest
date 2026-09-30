"""Spreading a present value evenly over time."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from simplyinvest.timeline import Term, Timeline

__all__ = [
    "annuity_factor",
    "capital_recovery_factor",
    "equivalent_annual_cost",
]


def annuity_factor(rate: float, horizon: Term) -> float:
    """Present value of one unit received annually for ``horizon``."""
    years = horizon.years
    compounded = float((1.0 + rate) ** years)
    if compounded == 1.0:
        return years
    return (compounded - 1.0) / (rate * compounded)


def capital_recovery_factor(rate: float, horizon: Term) -> float:
    """The annual payment one unit of present value supports over ``horizon``."""
    years = horizon.years
    compounded = float((1.0 + rate) ** years)
    if compounded == 1.0:
        return 1.0 / years
    return rate * compounded / (compounded - 1.0)


def equivalent_annual_cost(
    pv_of_costs: Any, timeline: Timeline, horizon: Term | None = None
) -> Any:
    """The level annual charge with the same present value as ``pv_of_costs``.

    Args:
        pv_of_costs: Present value of the money going out, as a positive number.
        timeline: The grid, for its rate.
        horizon: The life to spread over.  Defaults to the timeline's horizon.
    """
    return pv_of_costs * capital_recovery_factor(timeline.rate, horizon or timeline.horizon)
