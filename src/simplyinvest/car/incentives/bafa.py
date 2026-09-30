"""The means-tested purchase premium."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Final

import numpy as np

from simplyinvest.car.party import CarBuyer
from simplyinvest.car.vehicle import Propulsion, Vehicle, VehicleCategory
from simplyinvest.cashflow import CashFlowSeries, Component, OneOff, Role
from simplyinvest.domain import require
from simplyinvest.incentives import Eligibility, Incentive
from simplyinvest.money import Amount, Quantity
from simplyinvest.timeline import Term

if TYPE_CHECKING:
    import numpy.typing as npt

    from simplyinvest.domain import Asset, Context

__all__ = ["PREMIUM_2026_BEV", "PREMIUM_2026_PHEV", "PurchasePremium"]

#: Most dependants the schedule distinguishes.
_LAST_COLUMN = 2

#: Bands of the 2026 purchase premium, as (income floor, no child, one, two or more).
#: The floors carry a cent because the bands are published as "up to 45,000" and
#: "45,001 to 60,000", so a household on exactly 45,000 sits in the higher band.
#: The ceiling of 80,000 rises by 5,000 for each of the first two children, which
#: is what the last three rows encode.  Fuel-cell vehicles take these rates too.
PREMIUM_2026_BEV: Final[tuple[tuple[float, float, float, float], ...]] = (
    (0.0, 5000.0, 5500.0, 6000.0),
    (45_000.01, 4000.0, 4500.0, 5000.0),
    (60_000.01, 3000.0, 3500.0, 4000.0),
    (80_000.01, 0.0, 3500.0, 4000.0),
    (85_000.01, 0.0, 0.0, 4000.0),
    (90_000.01, 0.0, 0.0, 0.0),
)

#: The same schedule for plug-in hybrids and range extenders.
PREMIUM_2026_PHEV: Final[tuple[tuple[float, float, float, float], ...]] = (
    (0.0, 3500.0, 4000.0, 4500.0),
    (45_000.01, 2500.0, 3000.0, 3500.0),
    (60_000.01, 1500.0, 2000.0, 2500.0),
    (80_000.01, 0.0, 2000.0, 2500.0),
    (85_000.01, 0.0, 0.0, 2500.0),
    (90_000.01, 0.0, 0.0, 0.0),
)


@dataclass(frozen=True)
class PurchasePremium(Incentive):
    """A grant on a new vehicle, scaled by household income and dependants.

    Args:
        matrix: Income-by-dependants schedule to apply.
        eligible_propulsion: Drivetrains the schedule covers.
        disbursement_lag: Time from acquisition to payment.
        minimum_holding: Holding period the grant is conditional on.
        available: Whether the programme is still taking applications.
        reference: What to cite the schedule as.
    """

    matrix: tuple[tuple[float, float, float, float], ...] = PREMIUM_2026_BEV
    eligible_propulsion: tuple[Propulsion, ...] = (Propulsion.BEV, Propulsion.FCEV)
    disbursement_lag: Term = field(default_factory=lambda: Term.of_months(4))
    minimum_holding: Term = field(default_factory=lambda: Term.of_years(3))
    available: bool = True
    reference: str = "E-Auto-Kaufpraemie 2026 (BAFA), income and child schedule"

    @property
    def citation(self) -> str:
        """The programme this comes from."""
        return self.reference

    def grant(self, buyer: CarBuyer) -> Quantity:
        """What this buyer's income and dependants entitle them to."""
        table = np.asarray(self.matrix, dtype=np.float64)
        income = np.asarray(buyer.taxable_income, dtype=np.float64)
        row = np.searchsorted(table[:, 0], income, side="right") - 1
        column = min(buyer.children, _LAST_COLUMN) + 1
        entitlement: npt.NDArray[np.float64] = table[np.maximum(row, 0), column]
        return entitlement

    def eligibility(self, asset: Asset, ctx: Context) -> Eligibility:
        """Whether this vehicle and buyer qualify.

        Raises:
            PartyFactsMissingError: if the party carries no income or dependants.
        """
        if not isinstance(asset, Vehicle):
            return Eligibility.no("the premium applies to vehicles")
        if not self.available:
            return Eligibility.no("the programme is no longer taking applications")
        if asset.is_used:
            return Eligibility.no("the premium covers new vehicles only")
        if asset.category is not VehicleCategory.M1:
            return Eligibility.no(f"class {asset.category.value} is outside the premium's scope")
        if asset.propulsion not in self.eligible_propulsion:
            return Eligibility.no(f"{asset.propulsion.value} is not a covered drivetrain")
        buyer = require(ctx.party, CarBuyer)
        return Eligibility(
            applies=np.asarray(self.grant(buyer)) > 0.0,
            reason="the household's income is above the last band that pays",
        )

    def flows(self, asset: Asset, ctx: Context) -> CashFlowSeries:
        """The grant, paid once the application has been processed.

        Raises:
            TermNotRepresentableError: if the lag does not land on the grid.
        """
        if ctx.timeline.periods_in(self.disbursement_lag) > ctx.timeline.n_periods - ctx.start:
            return CashFlowSeries()
        at = ctx.timeline.offset(ctx.start, self.disbursement_lag)
        buyer = require(ctx.party, CarBuyer)
        entitled = Amount.received(self.grant(buyer))
        return CashFlowSeries.of(
            OneOff(
                entitled.scaled(self.eligibility(asset, ctx).mask),
                at=at,
                label=Component(Role.INCENTIVE, "purchase_premium"),
                description=f"Purchase premium, paid at period {at}",
            )
        )

    def constraints(self, asset: Asset, ctx: Context) -> tuple[str, ...]:
        """The holding period the grant is conditional on."""
        return (
            f"The purchase premium requires the vehicle to be held for "
            f"{self.minimum_holding} from first registration; disposing earlier "
            f"triggers repayment.",
        )
