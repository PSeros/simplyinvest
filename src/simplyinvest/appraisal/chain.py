"""Repeating a shorter alternative until it covers the horizon."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from simplyinvest.cashflow import CashFlowSeries

if TYPE_CHECKING:
    from simplyinvest.domain import Context
    from simplyinvest.timeline import Term

    from .alternative import Alternative

__all__ = ["ReplacementChain"]


@dataclass(frozen=True)
class ReplacementChain:
    """An alternative repeated until the horizon is covered.

    Args:
        name: What to call the chain.
        first: The leg acquired at the start.
        leg_life: How long each leg lasts before it is replaced.
        successors: Legs for the later windows, in order.  The last one given is
            reused for any windows beyond it; with none given, ``first`` repeats.
    """

    name: str
    first: Alternative
    leg_life: Term
    successors: tuple[Alternative, ...] = field(default_factory=tuple)

    @property
    def life(self) -> Term | None:
        """``None``; a chain covers the whole horizon."""
        return None

    def leg(self, index: int) -> Alternative:
        """The alternative occupying window ``index``."""
        if index == 0 or not self.successors:
            return self.first
        return self.successors[min(index - 1, len(self.successors) - 1)]

    def windows(self, ctx: Context) -> tuple[tuple[int, int], ...]:
        """Each leg's ``(start, end)`` periods, the last truncated at the horizon.

        Raises:
            ValueError: if a leg spans less than one period.
        """
        span = ctx.timeline.periods_in(self.leg_life)
        if span <= 0:
            raise ValueError(f"a leg must span at least one period, got {self.leg_life}")
        out: list[tuple[int, int]] = []
        start = ctx.start
        while start < ctx.last:
            out.append((start, min(start + span, ctx.last)))
            start += span
        return tuple(out)

    def flows(self, ctx: Context) -> CashFlowSeries:
        """Every leg's flows, each built in its own window."""
        return CashFlowSeries.concat(
            self.leg(index).flows(ctx.window(start, end))
            for index, (start, end) in enumerate(self.windows(ctx))
        )

    def constraints(self, ctx: Context) -> tuple[str, ...]:
        """Every leg's obligations, plus the assumption that chaining makes."""
        seen: list[str] = []
        windows = self.windows(ctx)
        for index, (start, end) in enumerate(windows):
            for note in self.leg(index).constraints(ctx.window(start, end)):
                if note not in seen:
                    seen.append(note)
        if len(windows) > 1:
            seen.append(
                f"The horizon is covered by {len(windows)} successive acquisitions of "
                f"{self.leg_life} each; every one after the first is an assumption about "
                f"what will be available and at what price."
            )
        return tuple(seen)

    @property
    def supports_batch(self) -> bool:
        """Whether every leg's flow structure is independent of its inputs."""
        legs = (self.first, *self.successors)
        return all(leg.supports_batch for leg in legs)
