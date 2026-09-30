"""Frozen cases whose numbers are pinned by the golden snapshots."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from simplyinvest import Amount, CashFlowSeries, Component, Context, Recurring, Role, Term, Timeline
from simplyinvest.appraisal import Alternative, Case
from simplyinvest.car import (
    CarOperating,
    Electricity,
    Household,
    Mileage,
    MileageLease,
    Petrol,
    Propulsion,
    Vehicle,
)
from simplyinvest.car.incentives import CirculationTaxExemption, GhgQuota, PurchasePremium
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


def electric_vs_petrol() -> Case:
    """A subsidised battery car against a petrol one, bought outright."""
    electric = Vehicle(
        name="electric",
        price=42_000.0,
        propulsion=Propulsion.BEV,
        energy=Electricity(
            consumption=17.5, price=0.31, public_price=0.55, home_share=0.8, charging_loss=0.10
        ),
        residual=GeometricDecline(0.15),
        insurance=780.0,
        maintenance=350.0,
        circulation_tax=180.0,
        infrastructure_cost=1_400.0,
        first_registration=date(2026, 4, 1),
    )
    petrol = Vehicle(
        name="petrol",
        price=33_500.0,
        propulsion=Propulsion.ICE,
        energy=Petrol(consumption=6.4, price=1.79, real_world_factor=1.15),
        residual=GeometricDecline(0.13),
        insurance=640.0,
        maintenance=620.0,
        circulation_tax=190.0,
    )
    timeline = Timeline(
        Term.of_years(8),
        periods_per_year=12,
        rate=0.03,
        start_date=date(2026, 4, 1),
        escalations={"energy": 0.03, "running_cost": 0.02},
    )
    return Case(
        alternatives=(
            Alternative(
                "electric",
                sources=(
                    CashPurchase().bind(electric),
                    CarOperating(electric),
                    PurchasePremium().bind(electric),
                    GhgQuota(annual_amount=300.0).bind(electric),
                    CirculationTaxExemption().bind(electric),
                ),
            ),
            Alternative(
                "petrol",
                sources=(CashPurchase().bind(petrol), CarOperating(petrol)),
            ),
        ),
        timeline=timeline,
        party=Household(taxable_income=52_000.0, children=1),
        usage=Mileage(annual_km=15_000.0),
    )


def a_leased_car_over_six_years() -> Case:
    """Two 36-month leases with a distance settlement, against buying outright."""
    car = Vehicle(
        name="estate",
        price=38_000.0,
        propulsion=Propulsion.ICE,
        energy=Petrol(consumption=6.8, price=1.82, real_world_factor=1.12),
        residual=GeometricDecline(0.16),
        insurance=720.0,
        maintenance=540.0,
        circulation_tax=210.0,
    )
    timeline = Timeline(Term.of_years(6), periods_per_year=12, rate=0.04)
    return Case(
        alternatives=(
            Alternative(
                "buy",
                sources=(CashPurchase().bind(car), CarOperating(car)),
            ),
            Alternative(
                "lease",
                sources=(
                    MileageLease(
                        rent=429.0,
                        term=Term.of_years(3),
                        initial_payment=3_200.0,
                        renewal_escalation=0.06,
                        annual_included_km=15_000.0,
                        excess_rate=0.12,
                        refund_rate=0.06,
                        refund_cap_km=5_000.0,
                    ).bind(car),
                    CarOperating(car),
                ),
            ),
        ),
        timeline=timeline,
        usage=Mileage(annual_km=18_000.0),
    )


CASES = {
    "electric_vs_petrol": electric_vs_petrol,
    "a_leased_car_over_six_years": a_leased_car_over_six_years,
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
