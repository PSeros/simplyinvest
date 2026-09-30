"""The flow-source contract a domain package implements."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from simplyinvest.cashflow import CashFlowSeries

    from .context import Context

__all__ = ["FlowSource"]


@runtime_checkable
class FlowSource(Protocol):
    """Anything that contributes cash flows to an alternative."""

    def flows(self, ctx: Context) -> CashFlowSeries:
        """The labelled flows this source contributes over ``ctx``'s window."""
        ...

    def constraints(self, ctx: Context) -> tuple[str, ...]:
        """Non-cash obligations, as sentences for a report."""
        ...

    @property
    def supports_batch(self) -> bool:
        """Whether the flow structure is independent of the input values."""
        ...
