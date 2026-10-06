"""How much the system makes, however that figure was arrived at."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np

from simplyinvest.errors import AnchorlessResamplingWarning
from simplyinvest.money import Quantity
from simplyinvest.timeline import add_months

if TYPE_CHECKING:
    import numpy.typing as npt

    from simplyinvest.timeline import Timeline

__all__ = ["DEFAULT_DEGRADATION", "MONTHLY_SHARE", "Generation", "StatedYield"]

#: Annual loss of output as modules age, as a fraction of the year before.
DEFAULT_DEGRADATION = 0.005

#: Share of a year's yield falling in each month, for a tilted array in
#: Germany.  Normalised on use, so these need not sum to exactly one.
MONTHLY_SHARE = (
    0.025,  # January
    0.042,
    0.080,
    0.112,
    0.130,
    0.132,
    0.136,  # July
    0.120,
    0.093,
    0.063,
    0.032,
    0.021,  # December
)

_MONTHS_IN_YEAR = 12


@runtime_checkable
class Generation(Protocol):
    """Energy the system produces, resolved onto a period grid."""

    def per_period(self, timeline: Timeline) -> npt.NDArray[np.float64]:
        """Kilowatt-hours generated in each period, shape ``(..., n + 1)``."""
        ...

    def hourly(self, timeline: Timeline) -> npt.NDArray[np.float64] | None:
        """One representative year hour by hour, or ``None`` if unavailable."""
        ...

    @property
    def first_year_kwh(self) -> Quantity:
        """What the system makes in its first year, before it ages."""
        ...


def months_ending_at(timeline: Timeline, period: int) -> tuple[int, ...]:
    """The calendar months the interval ending at ``period`` covers.

    Raises:
        NotAnchoredError: if the grid has no start date.
    """
    if period <= 0:
        return ()
    opens = timeline.date_of(period - 1)
    return tuple(add_months(opens, step).month for step in range(timeline.months_per_period))


def seasonal_shares(timeline: Timeline, monthly: tuple[float, ...]) -> npt.NDArray[np.float64]:
    """What share of a year's output falls in each period of ``timeline``.

    Shares repeat every year and sum to one across each whole year.  Period
    zero gets nothing, because output accrues over the interval ending at a
    period rather than at the instant it opens.
    """
    weights = np.asarray(monthly, dtype=np.float64)
    weights = weights / weights.sum()
    out = np.zeros(timeline.n_periods + 1, dtype=np.float64)
    flat = timeline.months_per_period / _MONTHS_IN_YEAR

    if not timeline.anchored:
        if timeline.months_per_period < _MONTHS_IN_YEAR and not np.allclose(weights, weights[0]):
            warnings.warn(
                "This grid has no start date, so a seasonal yield cannot be placed in the "
                "right months and is spread evenly instead.  Give the timeline a "
                "start_date to resolve the seasons.",
                AnchorlessResamplingWarning,
                stacklevel=2,
            )
        out[1:] = flat
        return out

    for period in range(1, timeline.n_periods + 1):
        out[period] = sum(weights[month - 1] for month in months_ending_at(timeline, period))
    return out


def degradation_factors(timeline: Timeline, rate: float) -> npt.NDArray[np.float64]:
    """How much of the first year's output each period still delivers."""
    if rate == 0.0:
        return np.ones(timeline.n_periods + 1, dtype=np.float64)
    elapsed = np.maximum(timeline.periods - 1, 0) // timeline.periods_per_year
    return np.asarray((1.0 - rate) ** elapsed, dtype=np.float64)


@dataclass(frozen=True)
class StatedYield(Generation):
    """A yield the user states, spread over the year and declining with age.

    Args:
        annual_kwh: What the system makes in its first year.
        degradation: Annual loss of output as the modules age.
        seasonality: Share of a year's yield falling in each month.

    Raises:
        ValueError: on a negative yield, or a degradation outside ``[0, 1)``.
    """

    annual_kwh: Quantity = 0.0
    degradation: float = DEFAULT_DEGRADATION
    seasonality: tuple[float, ...] = MONTHLY_SHARE

    def __post_init__(self) -> None:
        if np.any(np.asarray(self.annual_kwh, dtype=np.float64) < 0.0):
            raise ValueError(f"an annual yield cannot be negative, got {self.annual_kwh!r}")
        if not 0.0 <= self.degradation < 1.0:
            raise ValueError(
                f"degradation is the fraction lost each year, so it runs from 0 to 1, "
                f"got {self.degradation}"
            )
        if len(self.seasonality) != _MONTHS_IN_YEAR:
            raise ValueError(f"seasonality needs one share per month, got {len(self.seasonality)}")
        if any(share < 0.0 for share in self.seasonality) or sum(self.seasonality) <= 0.0:
            raise ValueError("seasonal shares are non-negative and cannot all be zero")

    def per_period(self, timeline: Timeline) -> npt.NDArray[np.float64]:
        """Kilowatt-hours generated in each period, shape ``(..., n + 1)``."""
        shape = seasonal_shares(timeline, self.seasonality)
        ageing = degradation_factors(timeline, self.degradation)
        stated = np.asarray(self.annual_kwh, dtype=np.float64)
        return stated[..., np.newaxis] * shape * ageing

    def hourly(self, timeline: Timeline) -> npt.NDArray[np.float64] | None:
        """``None``; a stated annual figure carries no hour-by-hour detail."""
        return None

    @property
    def first_year_kwh(self) -> Quantity:
        """What the system makes in its first year, as stated."""
        return self.annual_kwh

    def __str__(self) -> str:
        return f"{float(np.max(np.asarray(self.annual_kwh))):,.0f} kWh a year, as stated"
