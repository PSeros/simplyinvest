"""How tax touches a residential photovoltaic system: it does not."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from simplyinvest.tax import TaxTreatment

if TYPE_CHECKING:
    from simplyinvest.cashflow import CashFlowSeries
    from simplyinvest.domain import Context

__all__ = ["RESIDENTIAL_EXEMPTION_KW", "ResidentialExempt"]

#: Nameplate power up to which a system on a single dwelling is exempt,
#: under EStG §3 Nr. 72.  Multi-unit buildings get 15 kWp per unit, and a
#: taxpayer is capped at 100 kWp across all of their systems.
RESIDENTIAL_EXEMPTION_KW = 30.0

_CITATION = "EStG §3 Nr. 72"


@dataclass(frozen=True)
class ResidentialExempt(TaxTreatment):
    """Income tax on a system small enough to be exempt from it.

    Args:
        threshold_kw: Nameplate power the exemption runs up to.
        reference: The statute this rests on.
    """

    threshold_kw: float = RESIDENTIAL_EXEMPTION_KW
    reference: str = _CITATION

    def adjust(self, series: CashFlowSeries, ctx: Context) -> CashFlowSeries:
        """``series`` unchanged; the exemption adds nothing and takes nothing."""
        return series

    def constraints(self, ctx: Context) -> tuple[str, ...]:
        """What the exemption assumes, since the figures depend on it holding."""
        return (
            f"These figures assume {self.reference} applies: the system is at most "
            f"{self.threshold_kw:,.0f} kWp on a dwelling, and the operator stays under the "
            f"100 kWp cap across everything they own.  Above either limit the feed-in "
            f"revenue is taxable and this appraisal understates the cost.",
        )

    def __str__(self) -> str:
        return f"ResidentialExempt ({self.reference})"
