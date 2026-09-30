"""The tax-treatment contract."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from simplyinvest.cashflow import CashFlowSeries, Component
    from simplyinvest.domain import Context

__all__ = ["TaxTreatment", "Untaxed"]


class TaxTreatment(ABC):
    """How tax touches an alternative's cash flows."""

    @abstractmethod
    def adjust(self, series: CashFlowSeries, ctx: Context) -> CashFlowSeries:
        """``series`` with the tax consequences of its flows added."""

    def vat_bearing(self) -> frozenset[Component]:
        """Which components carry value-added tax."""
        return frozenset()

    def constraints(self, ctx: Context) -> tuple[str, ...]:
        """Non-cash obligations this treatment implies."""
        return ()

    def __str__(self) -> str:
        return type(self).__name__


@dataclass(frozen=True)
class Untaxed(TaxTreatment):
    """No tax consequences."""

    def adjust(self, series: CashFlowSeries, ctx: Context) -> CashFlowSeries:
        """``series`` unchanged."""
        return series
