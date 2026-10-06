"""Zero-rating of the supply and installation of a small residential system."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from simplyinvest.cashflow import CashFlowSeries
from simplyinvest.incentives import Eligibility, Incentive
from simplyinvest.pv.system import System
from simplyinvest.pv.tax import RESIDENTIAL_EXEMPTION_KW

if TYPE_CHECKING:
    from simplyinvest.domain import Asset, Context
    from simplyinvest.money import Amount

__all__ = ["STANDARD_VAT_RATE", "ZeroRatedSupply"]

_CITATION = "UStG §12(3)"

#: The standard German rate, which the relief takes the supply out of.
STANDARD_VAT_RATE = 0.19


@dataclass(frozen=True)
class ZeroRatedSupply(Incentive):
    """Value-added tax on a residential system: none, up to a size.

    The relief applies to the invoice rather than to a later receipt, so a
    quoted price for a qualifying system already carries no tax and there is
    nothing to add back.  Above the threshold the supply is standard-rated,
    which :meth:`adjust_capital_cost` prices.

    Args:
        threshold_kw: Nameplate power the relief runs up to.
        vat_rate: The rate the supply would otherwise bear.
        residential: Whether the system sits on or beside a dwelling.
        reference: The statute this rests on.
    """

    threshold_kw: float = RESIDENTIAL_EXEMPTION_KW
    vat_rate: float = STANDARD_VAT_RATE
    residential: bool = True
    reference: str = _CITATION

    @property
    def citation(self) -> str:
        """The statute this rests on."""
        return self.reference

    def eligibility(self, asset: Asset, ctx: Context) -> Eligibility:
        """Whether the supply of this system is taken out of the tax."""
        if not isinstance(asset, System):
            return Eligibility.no(f"{self.reference} zero-rates photovoltaic systems")
        if not self.residential:
            return Eligibility.no(
                f"{self.reference} zero-rates systems on or beside a dwelling; this one "
                f"is declared as sitting elsewhere."
            )
        size = np.asarray(asset.peak_kw, dtype=np.float64)
        return Eligibility(
            applies=size <= self.threshold_kw,
            reason=(
                f"the system is larger than the {self.threshold_kw:,.0f} kWp "
                f"{self.reference} zero-rates"
            ),
        )

    def flows(self, asset: Asset, ctx: Context) -> CashFlowSeries:
        """None; the relief changes the invoice rather than moving money later."""
        return CashFlowSeries()

    def adjust_capital_cost(self, asset: Asset, ctx: Context) -> Amount:
        """The price with tax added where the relief does not reach."""
        if self.eligibility(asset, ctx):
            return asset.capital_cost
        return asset.capital_cost.scaled(1.0 + self.vat_rate)

    def constraints(self, asset: Asset, ctx: Context) -> tuple[str, ...]:
        """What the quoted price is taken to mean."""
        return (
            f"The price is taken as what is actually invoiced.  Under {self.reference} a "
            f"system of at most {self.threshold_kw:,.0f} kWp on a dwelling is zero-rated, "
            f"so that invoice carries no value-added tax.  Above that size, add the tax "
            f"to the price yourself.",
        )

    def __str__(self) -> str:
        return f"ZeroRatedSupply ({self.reference})"
