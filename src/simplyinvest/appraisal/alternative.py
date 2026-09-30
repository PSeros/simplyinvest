"""One course of action, assembled from its sources of cash flow."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from simplyinvest.cashflow import CashFlowSeries
from simplyinvest.tax import TaxTreatment, Untaxed

if TYPE_CHECKING:
    from simplyinvest.domain import Context
    from simplyinvest.domain.source import FlowSource
    from simplyinvest.timeline import Term

__all__ = ["Alternative", "Evaluable"]


@runtime_checkable
class Evaluable(Protocol):
    """Anything that can be appraised or ranked."""

    @property
    def name(self) -> str:
        """What to call this course of action."""
        ...

    @property
    def life(self) -> Term | None:
        """How long it lasts, where that differs from the horizon."""
        ...

    def flows(self, ctx: Context) -> CashFlowSeries:
        """The labelled cash flows this course of action produces."""
        ...

    def constraints(self, ctx: Context) -> tuple[str, ...]:
        """Non-cash obligations, as sentences for a report."""
        ...


@dataclass(frozen=True)
class Alternative:
    """A course of action: some sources of cash flow, under a tax treatment.

    Args:
        name: What to call this course of action.
        sources: Everything contributing flows, in report order.
        tax: How tax touches the finished stream.
        life: How long this alternative lasts, if shorter than the horizon.
    """

    name: str
    sources: tuple[FlowSource, ...] = ()
    tax: TaxTreatment = field(default_factory=Untaxed)
    life: Term | None = None

    def flows(self, ctx: Context) -> CashFlowSeries:
        """Every source's flows, with tax applied to the finished stream."""
        gross = CashFlowSeries.concat(source.flows(ctx) for source in self.sources)
        return self.tax.adjust(gross, ctx)

    def constraints(self, ctx: Context) -> tuple[str, ...]:
        """Every non-cash obligation this course of action carries, deduplicated."""
        seen: list[str] = []
        for note in (
            *(n for source in self.sources for n in source.constraints(ctx)),
            *self.tax.constraints(ctx),
        ):
            if note not in seen:
                seen.append(note)
        return tuple(seen)

    @property
    def supports_batch(self) -> bool:
        """Whether every source's flow structure is independent of its inputs."""
        return all(source.supports_batch for source in self.sources)
