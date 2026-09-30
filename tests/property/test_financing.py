"""Invariants a loan must satisfy whatever its terms."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from simplyinvest import Amount, Term, Timeline
from simplyinvest.domain import Context, GeometricDecline
from simplyinvest.financing import AmortisationSchedule, AnnuityLoan, CashPurchase, Lease
from simplyinvest.timeline import Periodisation

principals = st.floats(min_value=100.0, max_value=1e7)
loan_rates = st.floats(min_value=0.0, max_value=0.25)
terms = st.integers(min_value=1, max_value=10)
periods = st.sampled_from([1, 2, 4, 12])


@dataclass(frozen=True)
class Thing:
    name: str = "thing"
    price: float = 24_000.0
    age_at_acquisition: Term = Term.ZERO
    economic_life: Term | None = None

    @property
    def capital_cost(self) -> Amount:
        return Amount.paid(self.price)

    @property
    def setup_cost(self) -> Amount:
        return Amount.zero()

    def residual_value(self, *, held: Term) -> Amount:
        return Amount.received(
            GeometricDecline(0.15).value_after(self.price, held=held, age_at_acquisition=Term.ZERO)
        )


@given(principal=principals, rate=st.floats(min_value=0.0, max_value=0.02), n=terms)
@settings(max_examples=200)
def test_a_schedule_always_closes(principal: float, rate: float, n: int) -> None:
    """Construction asserts this, so reaching here at all is the test."""
    plan = AmortisationSchedule.build(
        principal=principal, periodic_rate=rate, periods=np.arange(1, n + 1)
    )
    assert abs(plan.closing[-1]) <= 1e-9 * max(1.0, principal)


@given(principal=principals, rate=st.floats(min_value=0.0, max_value=0.02), n=terms)
@settings(max_examples=200)
def test_principal_repaid_equals_the_amount_borrowed(principal: float, rate: float, n: int) -> None:
    plan = AmortisationSchedule.build(
        principal=principal, periodic_rate=rate, periods=np.arange(1, n + 1)
    )
    assert abs(plan.principal.sum() - principal) <= 1e-9 * max(1.0, principal)


@given(principal=principals, rate=st.floats(min_value=0.0, max_value=0.02), n=terms)
@settings(max_examples=200)
def test_every_opening_balance_is_the_previous_closing(
    principal: float, rate: float, n: int
) -> None:
    plan = AmortisationSchedule.build(
        principal=principal, periodic_rate=rate, periods=np.arange(1, n + 1)
    )
    assert np.allclose(plan.opening[1:], plan.closing[:-1], rtol=1e-9, atol=1e-9)


@given(rate=loan_rates, term=terms, per_year=periods)
@settings(max_examples=100)
def test_a_loan_priced_at_the_discount_rate_is_worth_paying_cash(
    rate: float, term: int, per_year: int
) -> None:
    """The strongest single check on the instalment arithmetic.

    If money costs exactly what it is worth, the choice of how to pay cannot
    change the present value.  An instalment that is even slightly wrong breaks
    this immediately.
    """
    grid = Timeline(
        Term.of_years(term + 2),
        periods_per_year=per_year,
        rate=rate,
        periodisation=Periodisation.CONFORMAL,
    )
    ctx = Context(grid)
    asset, price = Thing(), Thing().capital_cost
    cash = CashPurchase().flows(asset, price, ctx).pv(grid)
    loan = AnnuityLoan(
        rate=rate, term=Term.of_years(term), periodisation=Periodisation.CONFORMAL
    ).flows(asset, price, ctx)
    assert abs(loan.pv(grid) - cash) <= 1e-6 * max(1.0, abs(cash))


@given(term=terms, per_year=periods, horizon=st.integers(min_value=1, max_value=20))
@settings(max_examples=100)
def test_lease_contracts_always_cover_the_horizon_without_overrunning_it(
    term: int, per_year: int, horizon: int
) -> None:
    grid = Timeline(Term.of_years(horizon), periods_per_year=per_year)
    ctx = Context(grid)
    windows = Lease(rent=100, term=Term.of_years(term)).contract_windows(ctx)
    assert windows[0][0] == 0
    assert windows[-1][1] == ctx.last
    for (_, end), (next_start, _) in pairwise(windows):
        assert end == next_start


@given(rate=loan_rates, term=terms, deposit=st.floats(min_value=0.0, max_value=20_000.0))
@settings(max_examples=100)
def test_a_larger_deposit_always_means_a_smaller_instalment(
    rate: float, term: int, deposit: float
) -> None:
    grid = Timeline(Term.of_years(term + 2), rate=0.03)
    ctx = Context(grid)
    price = Thing().capital_cost
    smaller = AnnuityLoan(rate=rate, term=Term.of_years(term), down_payment=deposit)
    larger = AnnuityLoan(rate=rate, term=Term.of_years(term), down_payment=deposit + 1_000)
    assert larger.instalment(price, ctx) < smaller.instalment(price, ctx)
