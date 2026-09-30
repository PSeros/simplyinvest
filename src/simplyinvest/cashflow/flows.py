"""The shapes a payment takes on the grid."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from simplyinvest.errors import CashFlowError
from simplyinvest.money import Amount

from .base import Component, Frequency, Role

if TYPE_CHECKING:
    import numpy.typing as npt

    from simplyinvest.timeline import Timeline

__all__ = [
    "Explicit",
    "OneOff",
    "Recurring",
    "Terminal",
]


def _canvas(timeline: Timeline, amount: Amount) -> npt.NDArray[np.float64]:
    """A zeroed amounts array shaped for this amount's trial axes, if any."""
    trial_shape = np.shape(amount.magnitude)
    return np.zeros((*trial_shape, timeline.n_periods + 1), dtype=np.float64)


def _check_period(timeline: Timeline, t: int, what: str) -> int:
    if not 0 <= t <= timeline.n_periods:
        raise CashFlowError(
            f"{what} falls at period {t}, outside the grid 0..{timeline.n_periods}.  "
            f"Either lengthen the horizon or place the flow inside it."
        )
    return t


@dataclass(frozen=True)
class OneOff:
    """A single payment at one period."""

    amount: Amount
    at: int = 0
    label: Component = field(default_factory=lambda: Component(Role.CAPITAL))
    description: str = ""

    def amounts(self, timeline: Timeline) -> npt.NDArray[np.float64]:
        """Signed amounts per period.

        Raises:
            CashFlowError: if the flow falls outside the grid.
        """
        out = _canvas(timeline, self.amount)
        out[..., _check_period(timeline, self.at, self.description or "a one-off flow")] = (
            self.amount.signed
        )
        return out


@dataclass(frozen=True)
class Recurring:
    """A repeating payment in arrears, optionally escalating.

    Args:
        amount: The payment per occurrence, at the grid's base price level.
        frequency: How often it falls.
        growth: Annual escalation, as a key into the timeline's escalation set
            or a rate given directly.  Applied on the accrual convention.
        start: Period the entitlement begins; the first payment is one stride on.
        end: Last period a payment may fall on.  Defaults to the horizon.
        label: What this flow is, for the breakdown.
        description: A human-readable line for reports.
    """

    amount: Amount
    frequency: Frequency = Frequency.PER_PERIOD
    growth: str | float = 0.0
    start: int = 0
    end: int | None = None
    label: Component = field(default_factory=lambda: Component(Role.OPERATING))
    description: str = ""

    def payment_periods(self, timeline: Timeline) -> npt.NDArray[np.int64]:
        """Every period this flow falls on."""
        stride = self.frequency.stride(timeline)
        last = timeline.n_periods if self.end is None else min(self.end, timeline.n_periods)
        first = self.start + stride
        if first > last:
            return np.empty(0, dtype=np.int64)
        return np.arange(first, last + 1, stride, dtype=np.int64)

    def amounts(self, timeline: Timeline) -> npt.NDArray[np.float64]:
        """Signed amounts per period, escalated."""
        out = _canvas(timeline, self.amount)
        when = self.payment_periods(timeline)
        if when.size == 0:
            return out
        index = timeline.accrual_index(self.growth)[when]
        out[..., when] = np.asarray(self.amount.signed)[..., np.newaxis] * index
        return out


@dataclass(frozen=True)
class Terminal:
    """A value realised when the asset leaves the model.

    Args:
        amount: What the asset is worth.
        at: The period it is realised in.
        label: What this flow is, for the breakdown.
        description: A human-readable line for reports.
        basis: How the figure was arrived at.
    """

    amount: Amount
    at: int
    label: Component = field(default_factory=lambda: Component(Role.TERMINAL))
    description: str = ""
    basis: str = ""

    def amounts(self, timeline: Timeline) -> npt.NDArray[np.float64]:
        """Signed amounts per period.

        Raises:
            CashFlowError: if the flow falls outside the grid.
        """
        out = _canvas(timeline, self.amount)
        out[..., _check_period(timeline, self.at, self.description or "a terminal value")] = (
            self.amount.signed
        )
        return out


@dataclass(frozen=True)
class Explicit:
    """A full signed vector, one value per period.

    Args:
        values: Signed amounts, shape ``(..., n_periods + 1)``.
        label: What this flow is, for the breakdown.
        description: A human-readable line for reports.
    """

    values: npt.NDArray[np.float64]
    label: Component = field(default_factory=lambda: Component(Role.OPERATING))
    description: str = ""

    def amounts(self, timeline: Timeline) -> npt.NDArray[np.float64]:
        """Signed amounts per period.

        Raises:
            CashFlowError: if the vector does not match the grid, or is not finite.
        """
        values = np.asarray(self.values, dtype=np.float64)
        if values.shape[-1] != timeline.n_periods + 1:
            raise CashFlowError(
                f"{self.description or 'an explicit flow'} carries {values.shape[-1]} "
                f"periods but the grid has {timeline.n_periods + 1} (0..{timeline.n_periods})."
            )
        if not np.all(np.isfinite(values)):
            raise CashFlowError(
                f"{self.description or 'an explicit flow'} contains a non-finite value"
            )
        return values.copy()
