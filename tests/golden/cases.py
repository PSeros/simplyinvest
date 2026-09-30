"""Frozen cases whose numbers are pinned by the golden snapshots."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from simplyinvest import Amount, CashFlowSeries, Component, Context, Recurring, Role, Term, Timeline
from simplyinvest.appraisal import Alternative, Case
from simplyinvest.domain import ConstantAnnualUsage, GeometricDecline
from simplyinvest.financing import AnnuityLoan, CashPurchase, Lease
from simplyinvest.money import Quantity
from simplyinvest.uncertain import LogNormal, Normal, Scenario, uncertain


@dataclass(frozen=True)
class Asset:
    """A plain asset, carrying no domain behaviour."""

    name: str
    price: float
    decline: float = 0.18
    setup: float = 0.0
    age_at_acquisition: Term = Term.ZERO
    economic_life: Term | None = None

    @property
    def capital_cost(self) -> Amount:
        return Amount.paid(self.price)

    @property
    def setup_cost(self) -> Amount:
        return Amount.paid(self.setup)

    def residual_value(self, *, held: Term) -> Amount:
        return Amount.received(
            GeometricDecline(self.decline).value_after(
                self.price, held=held, age_at_acquisition=self.age_at_acquisition
            )
        )


@dataclass(frozen=True)
class Hire:
    """Paying a monthly rate for the use of a machine, owning nothing."""

    rate: Quantity = 0.0

    def flows(self, ctx: Context) -> CashFlowSeries:
        return CashFlowSeries.of(
            Recurring(
                Amount.paid(self.rate),
                label=Component(Role.OPERATING, "hire"),
                description="Machine hire",
            )
        )

    def constraints(self, ctx: Context) -> tuple[str, ...]:
        return ("The machine is never owned, so nothing is left at the end.",)

    @property
    def supports_batch(self) -> bool:
        return True


def loan_vs_cash() -> Case:
    """A five-year annuity loan against paying outright."""
    machine = Asset("machine", price=86_000.0, setup=2_400.0)
    timeline = Timeline(
        Term.of_years(8),
        periods_per_year=12,
        rate=0.05,
        start_date=date(2026, 4, 1),
    )
    return Case(
        alternatives=(
            Alternative("cash", sources=(CashPurchase().bind(machine),)),
            Alternative(
                "loan",
                sources=(
                    AnnuityLoan(
                        rate=0.065, term=Term.of_years(5), down_payment=15_000.0, fees=450.0
                    ).bind(machine),
                ),
            ),
        ),
        timeline=timeline,
        usage=ConstantAnnualUsage({"hours": 1_600}),
    )


def lease_chained_over_seven_years() -> Case:
    """A 36-month lease chained across a seven-year horizon, against a purchase."""
    machine = Asset("machine", price=48_000.0, decline=0.22)
    timeline = Timeline(Term.of_years(7), periods_per_year=12, rate=0.04)
    return Case(
        alternatives=(
            Alternative("buy", sources=(CashPurchase().bind(machine),)),
            Alternative(
                "lease",
                sources=(
                    Lease(
                        rent=980.0,
                        term=Term.of_months(36),
                        initial_payment=3_500.0,
                        renewal_escalation=0.06,
                    ).bind(machine),
                ),
            ),
        ),
        timeline=timeline,
    )


def balloon_loan_on_a_quarterly_grid() -> Case:
    """A balloon loan on a quarterly grid."""
    machine = Asset("machine", price=31_500.0)
    timeline = Timeline(Term.of_years(6), periods_per_year=4, rate=0.045)
    return Case(
        alternatives=(
            Alternative("cash", sources=(CashPurchase().bind(machine),)),
            Alternative(
                "balloon",
                sources=(
                    AnnuityLoan(rate=0.039, term=Term.of_years(3), balloon=12_000.0).bind(machine),
                ),
            ),
        ),
        timeline=timeline,
    )


def an_uncertain_purchase() -> Case:
    """A machine of uncertain price and upkeep, bought outright or hired in."""
    machine = Asset(
        "machine",
        price=uncertain(64_000.0, "price", LogNormal.from_mean_cv(64_000.0, 0.09)),
        setup=uncertain(3_000.0, "setup", Normal(3_000.0, 400.0)),
    )
    timeline = Timeline(Term.of_years(7), periods_per_year=12, rate=0.045)
    return Case(
        alternatives=(
            Alternative("buy", sources=(CashPurchase().bind(machine),)),
            Alternative(
                "hire", sources=(Hire(uncertain(1_150.0, "hire_rate", Normal(1_150.0, 90.0))),)
            ),
        ),
        timeline=timeline,
    )


CASES = {
    "loan_vs_cash": loan_vs_cash,
    "lease_chained_over_seven_years": lease_chained_over_seven_years,
    "balloon_loan_on_a_quarterly_grid": balloon_loan_on_a_quarterly_grid,
}

SCENARIOS = {
    "an_uncertain_purchase": (
        an_uncertain_purchase,
        (
            Scenario("central", {}),
            Scenario("dear machine", {"price": 82_000.0}),
            Scenario("cheap hire", {"hire_rate": 880.0}),
            Scenario("both", {"price": 82_000.0, "hire_rate": 880.0}),
        ),
    ),
}
