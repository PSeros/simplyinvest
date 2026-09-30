"""Loans and leases."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from simplyinvest import Amount, Term, Timeline, appraise
from simplyinvest.appraisal import Alternative
from simplyinvest.cashflow import Component, OneOff, Role
from simplyinvest.domain import Context, GeometricDecline
from simplyinvest.errors import CashFlowError, TermNotRepresentableError
from simplyinvest.financing import (
    AmortisationSchedule,
    AnnuityLoan,
    CashPurchase,
    Lease,
    level_payment,
)
from simplyinvest.timeline import Periodisation


@dataclass(frozen=True)
class Thing:
    """A minimal asset, carrying no domain behaviour."""

    name: str = "thing"
    price: float = 24_000.0
    decline: float = 0.15
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


def context(years=6, per_year=12, rate=0.03, start=0):
    grid = Timeline(Term.of_years(years), periods_per_year=per_year, rate=rate)
    return Context(grid, start=start)


class TestLevelPayment:
    def test_against_a_spreadsheet(self):
        """Excel: PMT(5%, 5, -10000) = 2309.748..."""
        assert level_payment(10_000, 0.05, 5) == pytest.approx(2_309.748, abs=1e-3)

    def test_a_zero_rate_just_divides(self):
        assert level_payment(12_000, 0.0, 12) == pytest.approx(1_000.0)

    def test_a_balloon_lowers_the_instalment(self):
        assert level_payment(10_000, 0.05, 5, balloon=3_000) < level_payment(10_000, 0.05, 5)

    def test_a_loan_needs_an_instalment(self):
        with pytest.raises(CashFlowError, match="at least one instalment"):
            level_payment(10_000, 0.05, 0)


class TestAmortisationSchedule:
    @pytest.fixture
    def plan(self):
        return AmortisationSchedule.build(
            principal=10_000, periodic_rate=0.05, periods=np.arange(1, 6)
        )

    def test_it_closes_at_zero(self, plan):
        assert plan.closing[-1] == pytest.approx(0.0, abs=1e-9)

    def test_principal_repaid_equals_the_amount_borrowed(self, plan):
        assert plan.principal.sum() == pytest.approx(10_000.0, abs=1e-9)

    def test_interest_and_principal_make_up_the_instalment(self, plan):
        assert np.allclose(plan.payments, plan.instalment)

    def test_each_opening_is_the_previous_closing(self, plan):
        assert np.allclose(plan.opening[1:], plan.closing[:-1])

    def test_interest_falls_as_the_balance_does(self, plan):
        assert np.all(np.diff(plan.interest) < 0)

    def test_a_balloon_leaves_the_balance_then_settles_it(self):
        plan = AmortisationSchedule.build(
            principal=20_000, periodic_rate=0.004, periods=np.arange(1, 37), balloon=8_000
        )
        assert plan.closing[-1] == pytest.approx(0.0, abs=1e-9)
        assert plan.principal.sum() == pytest.approx(20_000.0, abs=1e-9)
        assert plan.payments[-1] == pytest.approx(plan.instalment + 8_000, abs=1e-6)

    def test_a_zero_rate_schedule_pays_no_interest(self):
        plan = AmortisationSchedule.build(
            principal=12_000, periodic_rate=0.0, periods=np.arange(1, 13)
        )
        assert plan.total_interest() == pytest.approx(0.0)

    def test_a_schedule_that_does_not_close_is_refused(self):
        with pytest.raises(CashFlowError, match="outstanding"):
            AmortisationSchedule(
                period=np.array([1]),
                opening=np.array([100.0]),
                interest=np.array([0.0]),
                principal=np.array([10.0]),
                closing=np.array([90.0]),
            )

    def test_markdown_shows_the_last_row(self, plan):
        assert plan.to_markdown(every=2).strip().endswith("| 0.00 |")


class TestAnnuityLoan:
    def test_a_loan_at_the_discount_rate_is_worth_paying_cash(self):
        """If money costs what it is worth, how you pay changes nothing."""
        ctx = context(rate=0.05)
        asset, price = Thing(), Thing().capital_cost
        cash = CashPurchase().flows(asset, price, ctx).pv(ctx.timeline)
        loan = AnnuityLoan(
            rate=0.05, term=Term.of_years(4), periodisation=Periodisation.CONFORMAL
        ).flows(asset, price, ctx)
        assert loan.pv(ctx.timeline) == pytest.approx(cash, abs=1e-6)

    def test_borrowing_cheaper_than_the_discount_rate_beats_cash(self):
        ctx = context(rate=0.06)
        asset, price = Thing(), Thing().capital_cost
        cash = CashPurchase().flows(asset, price, ctx).pv(ctx.timeline)
        loan = AnnuityLoan(rate=0.02, term=Term.of_years(4)).flows(asset, price, ctx)
        assert loan.pv(ctx.timeline) > cash

    def test_borrowing_dearer_than_the_discount_rate_loses_to_cash(self):
        ctx = context(rate=0.02)
        asset, price = Thing(), Thing().capital_cost
        cash = CashPurchase().flows(asset, price, ctx).pv(ctx.timeline)
        loan = AnnuityLoan(rate=0.08, term=Term.of_years(4)).flows(asset, price, ctx)
        assert loan.pv(ctx.timeline) < cash

    def test_a_deposit_reduces_what_is_borrowed(self):
        loan = AnnuityLoan(rate=0.04, term=Term.of_years(4), down_payment=6_000)
        assert loan.principal(Thing().capital_cost) == pytest.approx(18_000.0)

    def test_a_deposit_larger_than_the_price_is_refused(self):
        loan = AnnuityLoan(rate=0.04, term=Term.of_years(4), down_payment=30_000)
        with pytest.raises(CashFlowError, match="nothing left to finance"):
            loan.principal(Thing().capital_cost)

    def test_a_balloon_larger_than_the_loan_is_refused(self):
        loan = AnnuityLoan(rate=0.04, term=Term.of_years(4), balloon=30_000)
        with pytest.raises(CashFlowError, match="never amortise"):
            loan.principal(Thing().capital_cost)

    def test_fees_and_deposit_land_at_the_start(self):
        ctx = context()
        series = AnnuityLoan(rate=0.04, term=Term.of_years(4), down_payment=5_000, fees=300).flows(
            Thing(), Thing().capital_cost, ctx
        )
        at_start = series.amounts(ctx.timeline)[0]
        assert at_start == pytest.approx(-5_300.0)

    def test_a_term_past_the_horizon_is_refused(self):
        ctx = context(years=3)
        with pytest.raises(CashFlowError, match="past the horizon"):
            AnnuityLoan(rate=0.04, term=Term.of_years(5)).flows(Thing(), Thing().capital_cost, ctx)

    def test_a_negative_deposit_is_refused(self):
        with pytest.raises(ValueError, match="not a direction"):
            AnnuityLoan(down_payment=-100)

    def test_the_proportional_convention_costs_more_than_the_conformal_one(self):
        ctx = context()
        common = {"rate": 0.06, "term": Term.of_years(4)}
        conformal = AnnuityLoan(**common, periodisation=Periodisation.CONFORMAL)
        proportional = AnnuityLoan(**common, periodisation=Periodisation.PROPORTIONAL)
        price = Thing().capital_cost
        assert proportional.instalment(price, ctx) > conformal.instalment(price, ctx)

    def test_borrowing_dearer_than_the_discount_rate_is_flagged(self):
        ctx = context(rate=0.02)
        notes = AnnuityLoan(rate=0.08, term=Term.of_years(4)).constraints(Thing(), ctx)
        assert any("destroys value" in note for note in notes)

    def test_a_balloon_is_flagged(self):
        ctx = context()
        notes = AnnuityLoan(rate=0.04, term=Term.of_years(4), balloon=8_000).constraints(
            Thing(), ctx
        )
        assert any("refinanced" in note for note in notes)


class TestTheMonthsBugInFinancing:
    """A loan term means the same span whatever the grid resolution."""

    @pytest.mark.parametrize("per_year", [1, 2, 4, 12])
    def test_an_interest_free_loan_repays_exactly_the_principal_on_every_grid(self, per_year):
        """Grid resolution changes when money moves, never how much is owed."""
        ctx = context(years=6, per_year=per_year, rate=0.0)
        loan = AnnuityLoan(rate=0.0, term=Term.of_years(3))
        plan = loan.schedule(Thing().capital_cost, ctx)
        assert plan.payments.sum() == pytest.approx(24_000.0, abs=1e-6)
        assert len(plan.period) == 3 * per_year

    @pytest.mark.parametrize("per_year", [1, 2, 4, 12])
    def test_the_same_loan_costs_about_the_same_on_every_grid(self, per_year):
        """Close but not identical: a finer grid pays earlier within each year."""
        ctx = context(years=6, per_year=per_year, rate=0.03)
        loan = AnnuityLoan(rate=0.04, term=Term.of_years(3), periodisation=Periodisation.CONFORMAL)
        asset, price = Thing(), Thing().capital_cost
        value = loan.flows(asset, price, ctx).pv(ctx.timeline)
        monthly_ctx = context(years=6, per_year=12, rate=0.03)
        reference = loan.flows(asset, price, monthly_ctx).pv(monthly_ctx.timeline)
        assert value == pytest.approx(reference, rel=0.01)

    @pytest.mark.parametrize("per_year", [1, 2, 4, 12])
    def test_the_loan_runs_for_the_stated_term_on_every_grid(self, per_year):
        ctx = context(years=6, per_year=per_year)
        loan = AnnuityLoan(rate=0.04, term=Term.of_years(3))
        periods = loan.payment_periods(ctx)
        assert len(periods) == 3 * per_year
        assert periods[-1] == 3 * per_year

    def test_a_term_that_misses_the_grid_is_refused_not_rounded(self):
        ctx = context(years=6, per_year=4)
        loan = AnnuityLoan(rate=0.04, term=Term.of_months(5))
        with pytest.raises(TermNotRepresentableError):
            loan.payment_periods(ctx)


class TestCashPurchase:
    def test_price_now_residual_later(self):
        ctx = context(years=5, per_year=1, rate=0.0)
        series = CashPurchase().flows(Thing(), Thing().capital_cost, ctx)
        amounts = series.amounts(ctx.timeline)
        assert amounts[0] == pytest.approx(-24_000.0)
        assert amounts[-1] == pytest.approx(24_000 * 0.85**5)

    def test_setup_cost_is_separate_from_the_price(self):
        ctx = context(years=5, per_year=1)
        series = CashPurchase().flows(Thing(setup=1_500), Thing().capital_cost, ctx)
        parts = series.breakdown(ctx.timeline)
        assert Component(Role.CAPITAL, "setup") in parts

    def test_it_bears_the_residual_risk(self):
        assert CashPurchase().bears_residual_risk


class TestLease:
    def test_a_term_matching_the_horizon_is_one_contract(self):
        ctx = context(years=3)
        assert Lease(rent=250, term=Term.of_years(3)).contract_windows(ctx) == ((0, 36),)

    def test_a_shorter_term_chains(self):
        ctx = context(years=6)
        windows = Lease(rent=250, term=Term.of_years(3)).contract_windows(ctx)
        assert windows == ((0, 36), (36, 72))

    def test_the_final_contract_is_truncated_at_the_horizon(self):
        ctx = context(years=7)
        windows = Lease(rent=250, term=Term.of_years(3)).contract_windows(ctx)
        assert windows == ((0, 36), (36, 72), (72, 84))
        assert windows[-1][1] == ctx.last

    @pytest.mark.parametrize("per_year", [1, 2, 4, 12])
    def test_chaining_covers_the_same_span_on_every_grid(self, per_year):
        ctx = context(years=6, per_year=per_year)
        windows = Lease(rent=1, term=Term.of_years(3)).contract_windows(ctx)
        assert len(windows) == 2
        assert windows[-1][1] == ctx.last

    def test_rent_is_paid_in_arrears_for_every_period(self):
        ctx = context(years=2, per_year=1, rate=0.0)
        series = Lease(rent=1_000, term=Term.of_years(2)).flows(Thing(), Thing().capital_cost, ctx)
        assert series.amounts(ctx.timeline) == pytest.approx([0.0, -1_000.0, -1_000.0])

    def test_renewal_escalation_applies_from_the_second_contract(self):
        ctx = context(years=2, per_year=1, rate=0.0)
        series = Lease(rent=1_000, term=Term.of_years(1), renewal_escalation=0.10).flows(
            Thing(), Thing().capital_cost, ctx
        )
        assert series.amounts(ctx.timeline) == pytest.approx([0.0, -1_000.0, -1_100.0])

    def test_an_initial_payment_falls_at_each_contract_start(self):
        ctx = context(years=2, per_year=1, rate=0.0)
        series = Lease(rent=0, term=Term.of_years(1), initial_payment=500).flows(
            Thing(), Thing().capital_cost, ctx
        )
        assert series.amounts(ctx.timeline) == pytest.approx([-500.0, -500.0, 0.0])

    def test_it_books_no_residual_value(self):
        ctx = context(years=3, per_year=1)
        series = Lease(rent=250, term=Term.of_years(3)).flows(Thing(), Thing().capital_cost, ctx)
        assert Role.TERMINAL not in {flow.label.role for flow in series}
        assert not Lease().bears_residual_risk

    def test_a_domain_may_add_an_end_of_term_settlement(self):

        @dataclass(frozen=True)
        class MileageLease(Lease):
            excess_charge: float = 400.0

            def settlement(self, index, window, ctx):
                return OneOff(
                    Amount.paid(self.excess_charge),
                    at=window[1],
                    label=Component(Role.FINANCING, "excess_mileage"),
                    description=f"Excess mileage, contract {index + 1}",
                )

        ctx = context(years=2, per_year=1, rate=0.0)
        series = MileageLease(rent=0, term=Term.of_years(1)).flows(
            Thing(), Thing().capital_cost, ctx
        )
        assert series.amounts(ctx.timeline) == pytest.approx([0.0, -400.0, -400.0])

    def test_chaining_is_flagged_as_an_assumption(self):
        ctx = context(years=6)
        notes = Lease(rent=250, term=Term.of_years(3)).constraints(Thing(), ctx)
        assert any("not agreed" in note for note in notes)
        assert any("never owned" in note for note in notes)


class TestFinancingThroughTheEngine:
    def test_a_bound_plan_is_a_flow_source(self):
        grid = Timeline(Term.of_years(5), rate=0.03)
        result = appraise(Alternative("Buy", sources=(CashPurchase().bind(Thing()),)), grid)
        assert result.npv < 0
        assert sum(result.breakdown().values()) == pytest.approx(result.npv, abs=1e-9)

    def test_cash_and_credit_can_be_compared(self):
        grid = Timeline(Term.of_years(5), rate=0.03)
        cash = appraise(Alternative("Cash", sources=(CashPurchase().bind(Thing()),)), grid)
        credit = appraise(
            Alternative(
                "Credit",
                sources=(AnnuityLoan(rate=0.055, term=Term.of_years(4)).bind(Thing()),),
            ),
            grid,
        )
        assert credit.npv < cash.npv  # borrowing above the discount rate costs money

    def test_an_incentive_may_change_the_price_the_loan_is_sized_on(self):
        grid = Timeline(Term.of_years(5), rate=0.03)
        loan = AnnuityLoan(rate=0.04, term=Term.of_years(4))
        full = appraise(Alternative("Full", sources=(loan.bind(Thing()),)), grid)
        discounted = appraise(
            Alternative("Net", sources=(loan.bind(Thing(), Amount.paid(20_000)),)), grid
        )
        assert discounted.npv > full.npv
