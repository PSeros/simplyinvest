"""Invariants the period grid must hold for any inputs, not just the ones we chose."""

from __future__ import annotations

import numpy as np
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from simplyinvest.timeline import (
    ANCHORABLE_PERIODS,
    Periodisation,
    Term,
    Timeline,
    fisher_nominal,
    fisher_real,
)

rates = st.floats(min_value=-0.5, max_value=0.5, allow_nan=False, allow_infinity=False)
inflations = st.floats(min_value=-0.1, max_value=0.3, allow_nan=False, allow_infinity=False)
periods = st.sampled_from(sorted(ANCHORABLE_PERIODS))
years = st.integers(min_value=1, max_value=40)


@given(real=rates, inflation=inflations)
def test_fisher_round_trips(real: float, inflation: float) -> None:
    """Converting a real rate to nominal and back returns the original."""
    assert abs(fisher_real(fisher_nominal(real, inflation), inflation) - real) < 1e-12


@given(rate=rates, per_year=periods)
def test_conformal_periods_compound_to_the_annual_rate(rate: float, per_year: int) -> None:
    """The defining property of conformal periodisation."""
    grid = Timeline(Term.of_years(1), periods_per_year=per_year, rate=rate)
    compounded = (1.0 + grid.periodic_rate) ** per_year - 1.0
    assert abs(compounded - rate) < 1e-12


@given(rate=st.floats(min_value=0.0, max_value=0.5), per_year=periods, horizon=years)
def test_discount_factors_never_increase(rate: float, per_year: int, horizon: int) -> None:
    """Money later is never worth more than money sooner, at a non-negative rate."""
    grid = Timeline(Term.of_years(horizon), periods_per_year=per_year, rate=rate)
    factors = grid.discount_factors()
    assert np.all(np.diff(factors) <= 1e-15)
    assert factors[0] == 1.0


@given(rate=rates, per_year=periods, horizon=st.integers(min_value=1, max_value=10))
@settings(max_examples=50)
def test_present_value_is_linear(rate: float, per_year: int, horizon: int) -> None:
    """PV(a + b) == PV(a) + PV(b), which is what lets flows be composed at all."""
    grid = Timeline(Term.of_years(horizon), periods_per_year=per_year, rate=rate)
    rng = np.random.default_rng(0)
    a = rng.normal(size=grid.n_periods + 1) * 1_000
    b = rng.normal(size=grid.n_periods + 1) * 1_000
    combined = grid.pv(a + b)
    separate = grid.pv(a) + grid.pv(b)
    assert abs(combined - separate) <= 1e-6 * max(1.0, abs(separate))


@given(
    per_year=periods,
    horizon=years,
    span=st.integers(min_value=0, max_value=20),
)
def test_a_term_spans_the_same_time_whatever_the_grid(
    per_year: int, horizon: int, span: int
) -> None:
    """The months-vs-periods invariant, stated generally.

    A span of whole years lands on every anchorable grid, and lands at the same
    calendar distance on each.
    """
    assume(span <= horizon)
    grid = Timeline(Term.of_years(horizon), periods_per_year=per_year)
    assert grid.periods_in(Term.of_years(span)) == span * per_year


@given(rate=st.floats(min_value=0.0, max_value=0.3), per_year=periods, span=years)
def test_a_payment_a_fixed_term_out_discounts_identically_on_every_grid(
    rate: float, per_year: int, span: int
) -> None:
    """Grid resolution is a modelling choice; it must not change the answer."""
    grid = Timeline(Term.of_years(50), periods_per_year=per_year, rate=rate)
    at = grid.offset(0, Term.of_years(span))
    assert abs(grid.discount_factor(at) - (1.0 + rate) ** -span) < 1e-9


@given(rate=rates, per_year=periods)
def test_proportional_never_discounts_less_than_conformal(rate: float, per_year: int) -> None:
    """The banking convention compounds faster, so it discounts harder."""
    assume(rate > 0)
    common = {"periods_per_year": per_year, "rate": rate}
    conformal = Timeline(Term.of_years(10), periodisation=Periodisation.CONFORMAL, **common)
    proportional = Timeline(Term.of_years(10), periodisation=Periodisation.PROPORTIONAL, **common)
    assert (
        proportional.discount_factor(proportional.n_periods)
        <= conformal.discount_factor(conformal.n_periods) + 1e-12
    )


@given(growth=rates, per_year=periods, horizon=years)
def test_an_escalation_index_starts_at_one(growth: float, per_year: int, horizon: int) -> None:
    """Period 0 is today, so today's price is today's price."""
    grid = Timeline(
        Term.of_years(horizon), periods_per_year=per_year, escalations={"thing": growth}
    )
    assert grid.escalation_index("thing")[0] == 1.0
