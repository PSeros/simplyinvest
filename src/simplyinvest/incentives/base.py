"""The incentive contract."""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date
from typing import TYPE_CHECKING, Any

import numpy as np

from simplyinvest.cashflow import CashFlowSeries
from simplyinvest.errors import DoubleCountingWarning, NotAnchoredError

if TYPE_CHECKING:
    from simplyinvest.domain import Asset, Context
    from simplyinvest.money import Amount

__all__ = ["BoundIncentive", "Eligibility", "Incentive", "warn_on_double_counting"]


@dataclass(frozen=True, slots=True)
class Eligibility:
    """Whether a benefit applies, and why not when it does not.

    Args:
        applies: A flag, or an array of flags with one entry per trial.
        reason: Why the benefit does not apply.
    """

    applies: bool | Any
    reason: str = ""

    @classmethod
    def yes(cls) -> Eligibility:
        """A benefit that applies."""
        return cls(applies=True)

    @classmethod
    def no(cls, reason: str) -> Eligibility:
        """A benefit that does not apply, and why."""
        return cls(applies=False, reason=reason)

    def __bool__(self) -> bool:
        return bool(np.all(self.applies))

    @property
    def anywhere(self) -> bool:
        """Whether the benefit applies in at least one trial."""
        return bool(np.any(self.applies))

    @property
    def mask(self) -> Any:
        """The eligibility as a multiplier: 1 where it applies, 0 where it does not."""
        return np.asarray(self.applies, dtype=np.float64)


class Incentive(ABC):
    """A public benefit attaching to an asset or its use."""

    valid_from: date | None = None
    valid_until: date | None = None

    @property
    @abstractmethod
    def citation(self) -> str:
        """The statute or programme this comes from."""

    @abstractmethod
    def eligibility(self, asset: Asset, ctx: Context) -> Eligibility:
        """Whether this benefit applies to ``asset`` in these circumstances."""

    @abstractmethod
    def flows(self, asset: Asset, ctx: Context) -> CashFlowSeries:
        """The money this benefit moves, scaled by :meth:`eligibility`."""

    def adjust_capital_cost(self, asset: Asset, ctx: Context) -> Amount:
        """The price after this benefit, before financing sizes a loan."""
        return asset.capital_cost

    def constraints(self, asset: Asset, ctx: Context) -> tuple[str, ...]:
        """Non-cash obligations that come with taking the benefit."""
        return ()

    @property
    def capitalised_into_price(self) -> bool:
        """Whether this benefit is already priced into the quoted deal."""
        return False

    def check_period(self, ctx: Context) -> None:
        """Warn if the model starts outside this programme's validity."""
        if self.valid_from is None and self.valid_until is None:
            return
        try:
            opens = ctx.timeline.date_of(ctx.start)
        except NotAnchoredError:
            return
        if self.valid_from is not None and opens < self.valid_from:
            warnings.warn(
                f"{type(self).__name__} opens on {self.valid_from.isoformat()}, after this "
                f"model starts on {opens.isoformat()} ({self.citation}).",
                stacklevel=2,
            )
        if self.valid_until is not None and opens > self.valid_until:
            warnings.warn(
                f"{type(self).__name__} closed on {self.valid_until.isoformat()}, before "
                f"this model starts on {opens.isoformat()} ({self.citation}).",
                stacklevel=2,
            )

    def bind(self, asset: Asset) -> BoundIncentive:
        """This incentive with ``asset`` fixed, as a flow source."""
        return BoundIncentive(self, asset)

    def __str__(self) -> str:
        return f"{type(self).__name__} ({self.citation})"


@dataclass(frozen=True)
class BoundIncentive:
    """An incentive with its asset fixed."""

    incentive: Incentive
    asset: Asset

    def flows(self, ctx: Context) -> CashFlowSeries:
        """The benefit's flows, empty when it does not apply."""
        self.incentive.check_period(ctx)
        if not self.incentive.eligibility(self.asset, ctx).anywhere:
            return CashFlowSeries()
        return self.incentive.flows(self.asset, ctx)

    def constraints(self, ctx: Context) -> tuple[str, ...]:
        """The benefit's obligations, or why it does not apply."""
        verdict = self.incentive.eligibility(self.asset, ctx)
        if not verdict:
            return (f"{type(self.incentive).__name__} does not apply: {verdict.reason}",)
        return self.incentive.constraints(self.asset, ctx)

    @property
    def supports_batch(self) -> bool:
        """Whether the flow structure is independent of the input values."""
        return True


def warn_on_double_counting(*, capitalised: bool, incentives: object) -> None:
    """Warn when a benefit already in the price is also modelled explicitly."""
    if capitalised and incentives:
        warnings.warn(
            "This deal is quoted as already including a public benefit, and an explicit "
            "incentive has been supplied as well.  One of the two is counted twice.",
            DoubleCountingWarning,
            stacklevel=2,
        )
