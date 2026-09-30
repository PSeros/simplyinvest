"""Renting rather than owning."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from simplyinvest.cashflow import CashFlow, CashFlowSeries, Component, Explicit, OneOff, Role
from simplyinvest.money import Amount
from simplyinvest.timeline import Term

from .base import Financing

if TYPE_CHECKING:
    from simplyinvest.domain import Asset, Context

__all__ = ["Lease"]


@dataclass(frozen=True)
class Lease(Financing):
    """A chain of fixed-term rental contracts covering the horizon.

    Args:
        rent: Payment per period under the first contract.
        term: Length of one contract.
        initial_payment: Paid at the start of each contract.
        renewal_escalation: How much dearer each successive contract is.
        rate_includes_subsidy: Whether the quoted rent already includes a public
            subsidy.

    Raises:
        ValueError: on a negative rent or initial payment.
    """

    rent: float = 0.0
    term: Term = field(default_factory=lambda: Term.of_years(3))
    initial_payment: float = 0.0
    renewal_escalation: float = 0.0
    rate_includes_subsidy: bool = False

    def __post_init__(self) -> None:
        if self.rent < 0.0 or self.initial_payment < 0.0:
            raise ValueError("a rent is an amount, not a direction; it cannot be negative")

    @property
    def bears_residual_risk(self) -> bool:
        """False; the lessor keeps the asset."""
        return False

    @property
    def capitalises_subsidy(self) -> bool:
        """Whether the quoted rent already includes a subsidy."""
        return self.rate_includes_subsidy

    @property
    def supports_batch(self) -> bool:
        """False; each contract's rent is placed on the grid as a single number."""
        return False

    # ---------------------------------------------------------------- windows

    def contract_windows(self, ctx: Context) -> tuple[tuple[int, int], ...]:
        """Each contract's ``(start, end)`` periods, the last truncated at the horizon.

        Raises:
            ValueError: if the term spans less than one period.
        """
        span = ctx.timeline.periods_in(self.term)
        if span <= 0:
            raise ValueError(f"a lease term must span at least one period, got {self.term}")
        windows: list[tuple[int, int]] = []
        start = ctx.start
        while start < ctx.last:
            windows.append((start, min(start + span, ctx.last)))
            start += span
        return tuple(windows)

    def settlement(self, index: int, window: tuple[int, int], ctx: Context) -> CashFlow | None:
        """Anything owed when contract ``index`` ends, or ``None``."""
        return None

    # ------------------------------------------------------------------ flows

    def flows(self, asset: Asset, price: Amount, ctx: Context) -> CashFlowSeries:
        """Initial payments and rent for each contract in the chain."""
        flows: list[CashFlow] = []
        rents = ctx.timeline.zeros()

        for index, window in enumerate(self.contract_windows(ctx)):
            start, end = window
            uplift = (1.0 + self.renewal_escalation) ** index
            if self.initial_payment:
                flows.append(
                    OneOff(
                        Amount.paid(self.initial_payment * uplift),
                        at=start,
                        label=Component(Role.FINANCING, "lease_initial"),
                        description=f"Initial payment, contract {index + 1}",
                    )
                )
            if self.rent:
                due = np.arange(start + 1, end + 1, dtype=np.int64)
                rents[due] -= self.rent * uplift
            owed = self.settlement(index, window, ctx)
            if owed is not None:
                flows.append(owed)

        flows.append(
            Explicit(
                rents,
                label=Component(Role.FINANCING, "lease_rent"),
                description=f"Lease of {asset.name} at {self.rent:,.2f} per period",
            )
        )
        return CashFlowSeries(tuple(flows))

    def constraints(self, asset: Asset, ctx: Context) -> tuple[str, ...]:
        """Non-cash obligations this lease carries."""
        windows = self.contract_windows(ctx)
        notes = [
            f"The asset is never owned: {len(windows)} contract(s) of {self.term} cover the "
            f"horizon, and nothing is left at the end."
        ]
        if len(windows) > 1:
            notes.append(
                "Renewal terms beyond the first contract are assumed, not agreed"
                + (
                    f" (each {self.renewal_escalation:.1%} dearer than the last)."
                    if self.renewal_escalation
                    else " (at today's rate)."
                )
            )
        last_start, last_end = windows[-1]
        if last_end - last_start < ctx.timeline.periods_in(self.term):
            notes.append(
                "The final contract is truncated at the horizon; in practice it would run "
                "to its full term."
            )
        return tuple(notes)
