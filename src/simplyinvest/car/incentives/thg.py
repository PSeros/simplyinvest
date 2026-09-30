"""The annual greenhouse-gas quota credit."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from simplyinvest.car.vehicle import Propulsion, Vehicle
from simplyinvest.cashflow import CashFlowSeries, Component, Frequency, Recurring, Role
from simplyinvest.incentives import Eligibility, Incentive
from simplyinvest.money import Amount, Quantity

if TYPE_CHECKING:
    from simplyinvest.domain import Asset, Context

__all__ = ["GhgQuota"]


@dataclass(frozen=True)
class GhgQuota(Incentive):
    """A yearly credit for the emissions a battery-electric vehicle displaces.

    Args:
        annual_amount: Credit per year at today's certificate price.
        growth: Escalation key or annual rate the credit grows at.
        eligible_propulsion: Drivetrains that may claim it.
        reference: What to cite the scheme as.
    """

    annual_amount: Quantity = 0.0
    growth: str | float = 0.0
    eligible_propulsion: tuple[Propulsion, ...] = (Propulsion.BEV,)
    reference: str = "Treibhausgasminderungsquote, 37a BImSchG"

    @property
    def citation(self) -> str:
        """The scheme this comes from."""
        return self.reference

    def eligibility(self, asset: Asset, ctx: Context) -> Eligibility:
        """Whether this vehicle may claim the quota."""
        if not isinstance(asset, Vehicle):
            return Eligibility.no("the quota applies to vehicles")
        if asset.propulsion not in self.eligible_propulsion:
            return Eligibility.no(f"{asset.propulsion.value} earns no quota credit")
        if not np.any(np.asarray(self.annual_amount, dtype=np.float64) != 0.0):
            return Eligibility.no("no credit was quoted")
        return Eligibility.yes()

    def flows(self, asset: Asset, ctx: Context) -> CashFlowSeries:
        """The credit, claimed once a year in arrears."""
        earned = Amount.received(self.annual_amount)
        return CashFlowSeries.of(
            Recurring(
                earned.scaled(self.eligibility(asset, ctx).mask),
                frequency=Frequency.ANNUAL,
                growth=self.growth,
                start=ctx.start,
                end=ctx.last,
                label=Component(Role.INCENTIVE, "ghg_quota"),
                description="Greenhouse-gas quota credit",
            )
        )
