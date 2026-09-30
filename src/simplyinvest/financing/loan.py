"""Borrowing to buy."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from simplyinvest.cashflow import CashFlow, CashFlowSeries, Component, Explicit, OneOff, Role
from simplyinvest.domain import terminal_value
from simplyinvest.errors import CashFlowError
from simplyinvest.money import Amount
from simplyinvest.timeline import Periodisation, Term

from .base import Financing
from .schedule import AmortisationSchedule

if TYPE_CHECKING:
    import numpy.typing as npt

    from simplyinvest.domain import Asset, Context
    from simplyinvest.timeline import Timeline

__all__ = ["AnnuityLoan"]


@dataclass(frozen=True)
class AnnuityLoan(Financing):
    """A level-instalment loan, optionally with a deposit and a balloon.

    Args:
        rate: Annual borrowing rate.
        term: How long the loan runs.
        down_payment: Paid at the start from the investor's own money.
        balloon: Balance left outstanding, settled at the end.
        fees: Arrangement costs, paid at the start.
        periodisation: How the annual rate becomes a per-period rate.  Defaults
            to the timeline's.
        interest_deductible: Whether a tax treatment may deduct the interest.

    Raises:
        ValueError: on a negative amount or a rate at or below -100%.
    """

    rate: float = 0.0
    term: Term = field(default_factory=lambda: Term.of_years(4))
    down_payment: float = 0.0
    balloon: float = 0.0
    fees: float = 0.0
    periodisation: Periodisation | None = None
    interest_deductible: bool = False

    def __post_init__(self) -> None:
        for name in ("down_payment", "balloon", "fees"):
            if float(getattr(self, name)) < 0.0:
                raise ValueError(f"{name} is an amount, not a direction; it cannot be negative")
        if self.rate <= -1.0:
            raise ValueError(f"a borrowing rate must exceed -100%, got {self.rate!r}")

    @property
    def supports_batch(self) -> bool:
        """False; the instalment schedule is rebuilt from every input."""
        return False

    # ------------------------------------------------------------------ shape

    def principal(self, price: Amount) -> float:
        """What is borrowed, after the deposit.

        Raises:
            CashFlowError: if the deposit exceeds the price, or the balloon
                exceeds the amount borrowed.
        """
        borrowed = float(price.magnitude) - float(self.down_payment)
        if borrowed < 0.0:
            raise CashFlowError(
                f"the deposit of {self.down_payment:,.2f} exceeds the price of "
                f"{float(price.magnitude):,.2f}; there is nothing left to finance."
            )
        if self.balloon > borrowed:
            raise CashFlowError(
                f"the balloon of {self.balloon:,.2f} exceeds the {borrowed:,.2f} borrowed; "
                f"the loan would never amortise."
            )
        return borrowed

    def periodic_rate(self, timeline: Timeline) -> float:
        """The borrowing rate for one period of ``timeline``."""
        convention = self.periodisation or timeline.periodisation
        if convention is Periodisation.PROPORTIONAL:
            return self.rate / timeline.periods_per_year
        return float((1.0 + self.rate) ** (1.0 / timeline.periods_per_year)) - 1.0

    def payment_periods(self, ctx: Context) -> npt.NDArray[np.int64]:
        """The grid periods the instalments fall on, in arrears.

        Raises:
            CashFlowError: if the term runs past the horizon.
        """
        n = ctx.timeline.periods_in(self.term)
        last = ctx.start + n
        if last > ctx.timeline.n_periods:
            raise CashFlowError(
                f"a {self.term} loan starting at period {ctx.start} runs to period {last}, "
                f"past the horizon at {ctx.timeline.n_periods}.  Lengthen the horizon, or "
                f"shorten the term."
            )
        return np.arange(ctx.start + 1, last + 1, dtype=np.int64)

    def schedule(self, price: Amount, ctx: Context) -> AmortisationSchedule:
        """This loan's instalments, split into interest and principal."""
        return AmortisationSchedule.build(
            principal=self.principal(price),
            periodic_rate=self.periodic_rate(ctx.timeline),
            periods=self.payment_periods(ctx),
            balloon=float(self.balloon),
        )

    def instalment(self, price: Amount, ctx: Context) -> float:
        """The level payment, before any balloon settlement."""
        return self.schedule(price, ctx).instalment

    # ------------------------------------------------------------------ flows

    def _upfront_flows(self, asset: Asset, ctx: Context) -> list[CashFlow]:
        """Deposit, fees and commissioning cost, all at the start of the window."""
        flows: list[CashFlow] = []
        if self.down_payment:
            flows.append(
                OneOff(
                    Amount.paid(self.down_payment),
                    at=ctx.start,
                    label=Component(Role.CAPITAL, "down_payment"),
                    description=f"Deposit on {asset.name}",
                )
            )
        if self.fees:
            flows.append(
                OneOff(
                    Amount.paid(self.fees),
                    at=ctx.start,
                    label=Component(Role.FINANCING, "fees"),
                    description="Loan arrangement fees",
                )
            )
        setup = asset.setup_cost
        if float(setup.magnitude) != 0.0:
            flows.append(
                OneOff(
                    setup,
                    at=ctx.start,
                    label=Component(Role.CAPITAL, "setup"),
                    description=f"Commissioning {asset.name}",
                )
            )
        return flows

    def _instalment_flow(self, price: Amount, ctx: Context) -> Explicit:
        """Every instalment placed on the grid as one signed vector."""
        plan = self.schedule(price, ctx)
        amounts = ctx.timeline.zeros()
        amounts[plan.period] = -plan.payments
        return Explicit(
            amounts,
            label=Component(Role.FINANCING, "instalment"),
            description=f"Loan instalments at {self.rate:.2%} over {self.term}",
        )

    def flows(self, asset: Asset, price: Amount, ctx: Context) -> CashFlowSeries:
        """Deposit and fees at the start, instalments in arrears, residual at the end."""
        return CashFlowSeries(
            (
                *self._upfront_flows(asset, ctx),
                self._instalment_flow(price, ctx),
                terminal_value(asset, held=ctx.held, at=ctx.last),
            )
        )

    def constraints(self, asset: Asset, ctx: Context) -> tuple[str, ...]:
        """Non-cash obligations this loan carries."""
        notes = []
        if self.balloon:
            notes.append(
                f"A balloon of {self.balloon:,.2f} falls due after {self.term}; it must be "
                f"refinanced, settled from savings, or met by selling the asset."
            )
        if self.rate > ctx.timeline.rate:
            notes.append(
                f"Borrowing at {self.rate:.2%} against a discount rate of "
                f"{ctx.timeline.rate:.2%}: the loan destroys value relative to paying cash."
            )
        return tuple(notes)
