"""Tax treatments, constructed with the rates that apply."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from simplyinvest.cashflow import CashFlow, CashFlowSeries, Component, Explicit, Role

from .base import TaxTreatment

if TYPE_CHECKING:
    import numpy.typing as npt

    from simplyinvest.domain import Context
    from simplyinvest.timeline import Timeline

__all__ = ["FlatRateIncome", "VatRegistered"]


def _costs_of(flow: CashFlow, timeline: Timeline) -> npt.NDArray[np.float64]:
    """The outflow part of ``flow``, as positive amounts per period."""
    amounts = flow.amounts(timeline)
    return -amounts * (amounts < 0.0)


def _offset_costs(
    series: CashFlowSeries,
    timeline: Timeline,
    *,
    on: frozenset[Component],
    fraction: float,
    label: Component,
    describe: Callable[[Component], str],
) -> CashFlowSeries:
    """``series`` with ``fraction`` of its costs on ``on`` added back as ``label``."""
    offsets: list[CashFlow] = []
    for flow in series:
        if flow.label not in on:
            continue
        costs = _costs_of(flow, timeline)
        if not costs.any():
            continue
        offsets.append(Explicit(costs * fraction, label=label, description=describe(flow.label)))
    return series + CashFlowSeries(tuple(offsets))


@dataclass(frozen=True)
class FlatRateIncome(TaxTreatment):
    """Deductible costs relieved at a single marginal rate.

    Args:
        marginal_rate: The rate at which a deduction is worth something.
        deductible: Which components attract relief.
    """

    marginal_rate: float
    deductible: frozenset[Component] = field(default_factory=frozenset)

    def adjust(self, series: CashFlowSeries, ctx: Context) -> CashFlowSeries:
        """``series`` with the tax saved on its deductible costs added."""
        if not self.marginal_rate or not self.deductible:
            return series
        return _offset_costs(
            series,
            ctx.timeline,
            on=self.deductible,
            fraction=self.marginal_rate,
            label=Component(Role.TAX, "relief"),
            describe=lambda label: f"Tax relief at {self.marginal_rate:.1%} on {label}",
        )


@dataclass(frozen=True)
class VatRegistered(TaxTreatment):
    """Input VAT recovered on the components that bear it.

    Args:
        vat_rate: The rate included in the gross figures already modelled.
        bearing: Which components carry VAT.
        recoverable_share: How much of it is recoverable.
    """

    vat_rate: float
    bearing: frozenset[Component] = field(default_factory=frozenset)
    recoverable_share: float = 1.0

    @property
    def recoverable_fraction(self) -> float:
        """The share of a gross figure that is recoverable input VAT."""
        return self.vat_rate / (1.0 + self.vat_rate) * self.recoverable_share

    def vat_bearing(self) -> frozenset[Component]:
        """Which components carry value-added tax."""
        return self.bearing

    def adjust(self, series: CashFlowSeries, ctx: Context) -> CashFlowSeries:
        """``series`` with recoverable input VAT added."""
        if not self.vat_rate or not self.bearing:
            return series
        return _offset_costs(
            series,
            ctx.timeline,
            on=self.bearing,
            fraction=self.recoverable_fraction,
            label=Component(Role.TAX, "input_vat"),
            describe=lambda label: f"Input VAT recovered on {label}",
        )

    def constraints(self, ctx: Context) -> tuple[str, ...]:
        """Non-cash obligations registration implies."""
        return (
            "Recovering input VAT requires registration, and with it returns, records "
            "and output VAT on anything sold.",
        )
