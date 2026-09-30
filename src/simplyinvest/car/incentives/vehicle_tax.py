"""Exemption from circulation tax, for a capped span from first registration."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import TYPE_CHECKING

import numpy as np

from simplyinvest.car.operations import DEFAULT_ESCALATION
from simplyinvest.car.vehicle import Propulsion, Vehicle
from simplyinvest.cashflow import CashFlowSeries, Component, Recurring, Role
from simplyinvest.incentives import Eligibility, Incentive
from simplyinvest.money import Amount
from simplyinvest.timeline import Term, add_months, months_between

if TYPE_CHECKING:
    from simplyinvest.domain import Asset, Context

__all__ = ["CirculationTaxExemption"]


@dataclass(frozen=True)
class CirculationTaxExemption(Incentive):
    """A credit against the vehicle's circulation tax, keyed on first registration.

    A used vehicle inherits only the unexpired remainder.

    Args:
        max_term: Length of the exemption from first registration.
        expires: Last date the scheme grants an exemption.
        registered_from: First registration date the scheme covers.
        registered_by: Last registration date the scheme covers.
        eligible_propulsion: Drivetrains that qualify.
        escalation: Escalation key or annual rate the credit grows at.  Matches
            :class:`~simplyinvest.car.operations.CarOperating` so that the credit
            and the tax it offsets cancel.
        reference: What to cite the exemption as.
    """

    max_term: Term = field(default_factory=lambda: Term.of_years(10))
    expires: date | None = date(2035, 12, 31)
    registered_from: date | None = date(2011, 5, 18)
    registered_by: date | None = date(2030, 12, 31)
    eligible_propulsion: tuple[Propulsion, ...] = (Propulsion.BEV,)
    escalation: str | float = DEFAULT_ESCALATION
    reference: str = "Kfz-Steuerbefreiung fuer Elektrofahrzeuge, 3d KraftStG"

    @property
    def citation(self) -> str:
        """The statute this comes from."""
        return self.reference

    def remaining(self, vehicle: Vehicle) -> Term:
        """How much of the exemption is unexpired when the vehicle is acquired."""
        left = self.max_term - vehicle.age_at_acquisition
        if left.months <= 0:
            return Term.ZERO
        if self.expires is not None and vehicle.first_registration is not None:
            acquired = add_months(vehicle.first_registration, vehicle.age_at_acquisition.months)
            statutory = Term.of_months(max(months_between(acquired, self.expires), 0))
            left = min(left, statutory)
        return left

    def _outside_the_registration_window(self, registered: date | None) -> str | None:
        """Why ``registered`` falls outside the scheme, or ``None`` if it does not."""
        if registered is None:
            return None
        if self.registered_from is not None and registered < self.registered_from:
            return (
                f"first registered on {registered.isoformat()}, before the scheme opened "
                f"on {self.registered_from.isoformat()}"
            )
        if self.registered_by is not None and registered > self.registered_by:
            return (
                f"first registered on {registered.isoformat()}, after the scheme closed "
                f"to new vehicles on {self.registered_by.isoformat()}"
            )
        return None

    def eligibility(self, asset: Asset, ctx: Context) -> Eligibility:
        """Whether this vehicle still carries an unexpired exemption."""
        if not isinstance(asset, Vehicle):
            return Eligibility.no("the exemption applies to vehicles")
        if asset.propulsion not in self.eligible_propulsion:
            return Eligibility.no(f"{asset.propulsion.value} is not exempt")
        if not np.any(np.asarray(asset.circulation_tax, dtype=np.float64) != 0.0):
            return Eligibility.no("no circulation tax was quoted to exempt")
        outside = self._outside_the_registration_window(asset.first_registration)
        if outside is not None:
            return Eligibility.no(outside)
        if self.remaining(asset).is_zero:
            return Eligibility.no("the exemption had already run out when it was acquired")
        return Eligibility.yes()

    def flows(self, asset: Asset, ctx: Context) -> CashFlowSeries:
        """A credit matching the tax, for as long as the exemption runs."""
        verdict = self.eligibility(asset, ctx)
        if not isinstance(asset, Vehicle) or not verdict.anywhere:
            return CashFlowSeries()
        span = min(self.remaining(asset), ctx.held)
        credited = Amount.received(asset.circulation_tax / ctx.timeline.periods_per_year)
        return CashFlowSeries.of(
            Recurring(
                credited.scaled(verdict.mask),
                growth=self.escalation,
                start=ctx.start,
                end=min(ctx.timeline.offset(ctx.start, span), ctx.last),
                label=Component(Role.INCENTIVE, "tax_exemption"),
                description=f"Circulation tax exempt for {span}",
            )
        )
