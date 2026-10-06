"""What the system saves, and what it costs to keep."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from simplyinvest.cashflow import (
    CashFlow,
    CashFlowSeries,
    Component,
    Explicit,
    OneOff,
    Recurring,
    Role,
)
from simplyinvest.domain import require
from simplyinvest.money import Amount
from simplyinvest.pv.party import PvOperator
from simplyinvest.pv.supply import SELF_CONSUMED

if TYPE_CHECKING:
    import numpy.typing as npt

    from simplyinvest.domain import Context
    from simplyinvest.pv.system import System
    from simplyinvest.timeline import Term

__all__ = ["DEFAULT_COST_ESCALATION", "PvOperations"]

#: Escalation key the recurring costs grow at unless told otherwise.
DEFAULT_COST_ESCALATION = "operating_cost"

#: The fixed annual items, as (system attribute, breakdown detail).
_FIXED = (
    ("service_cost", "service"),
    ("insurance", "insurance"),
    ("metering_cost", "metering"),
    ("other_annual_cost", "other"),
)

#: The parts replaced during the horizon, as (system attribute, breakdown detail).
_REPLACED = (
    ("inverter", "inverter_replacement"),
    ("battery", "battery_replacement"),
)


@dataclass(frozen=True)
class PvOperations:
    """Self-consumption valued as an avoided bill, less what the system costs.

    Args:
        system: The installation.
        cost_escalation: How the recurring costs grow.
    """

    system: System
    cost_escalation: str | float = DEFAULT_COST_ESCALATION

    def self_consumed(self, ctx: Context) -> npt.NDArray[np.float64]:
        """Kilowatt-hours used on site in each period of this window.

        Raises:
            UnknownQuantityError: if the usage profile carries no such stream.
        """
        return ctx.usage.per_period(SELF_CONSUMED, ctx.timeline) * ctx.mask

    def avoided_cost(self, ctx: Context) -> npt.NDArray[np.float64]:
        """What the self-consumed energy would have cost from the grid.

        Raises:
            PartyFactsMissingError: if the party carries no retail price.
        """
        operator = require(ctx.party, PvOperator)
        return self.self_consumed(ctx) * operator.retail_price.per_kwh(ctx.timeline)

    def flows(self, ctx: Context) -> CashFlowSeries:
        """The avoided bill, the costs of keeping the system, and replacements."""
        flows: list[CashFlow] = [
            Explicit(
                self.avoided_cost(ctx),
                label=Component(Role.REVENUE, "self_consumption"),
                description=f"Grid electricity {self.system.name} displaces",
            )
        ]
        per_year = ctx.timeline.periods_per_year
        for attribute, detail in _FIXED:
            annual = getattr(self.system, attribute)
            if not np.any(np.asarray(annual, dtype=np.float64) != 0.0):
                continue
            flows.append(
                Recurring(
                    Amount.paid(annual / per_year),
                    growth=self.cost_escalation,
                    start=ctx.start,
                    end=ctx.last,
                    label=Component(Role.OPERATING, detail),
                    description=f"{detail.capitalize()}, per period",
                )
            )
        flows.extend(self._replacements(ctx))
        return CashFlowSeries(tuple(flows))

    def _replacements(self, ctx: Context) -> list[CashFlow]:
        """Each part that wears out inside this window, priced when it does."""
        out: list[CashFlow] = []
        for attribute, detail in _REPLACED:
            part = getattr(self.system, attribute)
            if part is None or part.replaced_after is None:
                continue
            at = self._replacement_period(ctx, part.replaced_after)
            if at is None:
                continue
            cost = np.asarray(part.replacement_cost, dtype=np.float64)
            if not np.any(cost != 0.0):
                continue
            grown = cost * ctx.timeline.escalation_index(self.cost_escalation)[at]
            out.append(
                OneOff(
                    Amount.paid(grown),
                    at=at,
                    label=Component(Role.CAPITAL, detail),
                    description=f"{detail.replace('_', ' ').capitalize()} at period {at}",
                )
            )
        return out

    def _replacement_period(self, ctx: Context, after: Term) -> int | None:
        """When a part is replaced, or ``None`` if that falls outside the window."""
        spans = ctx.timeline.periods_in(after)
        if spans <= 0 or ctx.start + spans > ctx.last:
            return None
        return ctx.start + spans

    def constraints(self, ctx: Context) -> tuple[str, ...]:
        """What the figures lean on that they do not price."""
        notes = [
            "Self-consumption is valued at the working price alone.  The standing charge "
            "is owed whatever the roof does, so it is not avoided.",
            "The roof is assumed to carry the array for the whole horizon; stripping and "
            "refitting it for a re-roof is not priced.",
        ]
        used = float(np.sum(self.self_consumed(ctx)))
        years = max(ctx.timeline.term_of(ctx.last - ctx.start).years, 1e-9)
        notes.append(
            f"This implies about {used / years:,.0f} kWh a year used on site.  Check that "
            f"against a bill before trusting the result."
        )
        if not self.system.has_storage:
            notes.append(
                "No storage is fitted, so everything generated outside the hours it is "
                "used is exported."
            )
        return tuple(notes)

    @property
    def supports_batch(self) -> bool:
        """True; every figure is a vector over the same periods whatever the inputs."""
        return True

    def __str__(self) -> str:
        return f"Running {self.system.name}"
