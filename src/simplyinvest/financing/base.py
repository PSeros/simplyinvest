"""How an asset is paid for."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

from simplyinvest.cashflow import CashFlowSeries

if TYPE_CHECKING:
    from simplyinvest.domain import Asset, Context
    from simplyinvest.money import Amount

__all__ = ["BoundFinancing", "Financing", "bind"]


class Financing(ABC):
    """A way of paying for an asset."""

    @abstractmethod
    def flows(self, asset: Asset, price: Amount, ctx: Context) -> CashFlowSeries:
        """The flows this plan produces for ``asset`` at ``price``."""

    def constraints(self, asset: Asset, ctx: Context) -> tuple[str, ...]:
        """Non-cash obligations this plan carries."""
        return ()

    @property
    def bears_residual_risk(self) -> bool:
        """Whether the investor owns the asset at the end."""
        return True

    @property
    def capitalises_subsidy(self) -> bool:
        """Whether a quoted rate already has a public subsidy priced into it."""
        return False

    @property
    def supports_batch(self) -> bool:
        """Whether the flow structure is independent of the input values."""
        return True

    def bind(self, asset: Asset, price: Amount | None = None) -> BoundFinancing:
        """This plan with ``asset`` and ``price`` fixed, as a flow source."""
        return BoundFinancing(self, asset, price if price is not None else asset.capital_cost)

    def __str__(self) -> str:
        return type(self).__name__


@dataclass(frozen=True)
class BoundFinancing:
    """A financing plan with its asset and price fixed."""

    plan: Financing
    asset: Asset
    price: Amount

    def flows(self, ctx: Context) -> CashFlowSeries:
        """The plan's flows for this asset."""
        return self.plan.flows(self.asset, self.price, ctx)

    def constraints(self, ctx: Context) -> tuple[str, ...]:
        """The plan's non-cash obligations for this asset."""
        return self.plan.constraints(self.asset, ctx)

    @property
    def supports_batch(self) -> bool:
        """Whether the flow structure is independent of the input values."""
        return self.plan.supports_batch


def bind(plan: Financing, asset: Asset, price: Amount | None = None) -> BoundFinancing:
    """``plan`` with ``asset`` and ``price`` fixed, as a flow source."""
    return plan.bind(asset, price)
