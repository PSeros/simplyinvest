"""Paying outright."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from simplyinvest.cashflow import CashFlow, CashFlowSeries, Component, OneOff, Role
from simplyinvest.domain import terminal_value

from .base import Financing

if TYPE_CHECKING:
    from simplyinvest.domain import Asset, Context
    from simplyinvest.money import Amount

__all__ = ["CashPurchase"]


@dataclass(frozen=True)
class CashPurchase(Financing):
    """The whole price at the start, and the asset is owned."""

    def flows(self, asset: Asset, price: Amount, ctx: Context) -> CashFlowSeries:
        """Outlay at the start, residual value at the end."""
        flows: list[CashFlow] = [
            OneOff(
                price,
                at=ctx.start,
                label=Component(Role.CAPITAL, "purchase"),
                description=f"Purchase of {asset.name}",
            )
        ]
        setup = asset.setup_cost
        if np.any(np.asarray(setup.magnitude) != 0.0):
            flows.append(
                OneOff(
                    setup,
                    at=ctx.start,
                    label=Component(Role.CAPITAL, "setup"),
                    description=f"Commissioning {asset.name}",
                )
            )
        flows.append(terminal_value(asset, held=ctx.held, at=ctx.last))
        return CashFlowSeries(tuple(flows))
