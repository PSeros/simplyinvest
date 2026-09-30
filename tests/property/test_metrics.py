"""Relationships between the metrics that must hold for any stream."""

from __future__ import annotations

import numpy as np
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from simplyinvest import Term, Timeline
from simplyinvest.metrics import (
    annuity_factor,
    capital_recovery_factor,
    discounted_payback,
    irr,
    net_present_value,
    payback,
)

rates = st.floats(min_value=0.0, max_value=0.3, allow_nan=False, allow_infinity=False)
years = st.integers(min_value=2, max_value=30)
periods = st.sampled_from([1, 2, 4, 12])


@given(
    outlay=st.floats(min_value=1.0, max_value=1e6),
    receipt=st.floats(min_value=1.0, max_value=1e6),
    horizon=years,
)
@settings(max_examples=100)
def test_the_rate_found_zeroes_the_present_value(
    outlay: float, receipt: float, horizon: int
) -> None:
    """The defining property of an internal rate of return."""
    grid = Timeline(Term.of_years(horizon), periods_per_year=1)
    amounts = np.concatenate([[-outlay], np.full(horizon, receipt)])
    result = irr(amounts, grid)
    assume(bool(result))
    at_rate = Timeline(Term.of_years(horizon), periods_per_year=1, rate=result.value)
    assert abs(net_present_value(amounts, at_rate)) <= 1e-6 * max(outlay, receipt)


@given(
    outlay=st.floats(min_value=1.0, max_value=1e6),
    receipt=st.floats(min_value=1.0, max_value=1e6),
    horizon=years,
    rate=rates,
)
@settings(max_examples=100)
def test_discounting_never_shortens_payback(
    outlay: float, receipt: float, horizon: int, rate: float
) -> None:
    """Later money is worth less, so it can only take longer to add up."""
    grid = Timeline(Term.of_years(horizon), periods_per_year=1, rate=rate)
    amounts = np.concatenate([[-outlay], np.full(horizon, receipt)])
    plain, discounted = payback(amounts, grid), discounted_payback(amounts, grid)
    assume(discounted is not None)
    assert discounted >= plain


@given(rate=rates, horizon=years)
def test_the_annuity_factors_are_reciprocals(rate: float, horizon: int) -> None:
    span = Term.of_years(horizon)
    product = annuity_factor(rate, span) * capital_recovery_factor(rate, span)
    assert abs(product - 1.0) < 1e-12


@given(rate=st.floats(min_value=0.001, max_value=0.3), horizon=years)
def test_a_longer_life_carries_a_lower_annual_charge(rate: float, horizon: int) -> None:
    """The reason unequal lives are compared on annuity rather than present value."""
    shorter = capital_recovery_factor(rate, Term.of_years(horizon))
    longer = capital_recovery_factor(rate, Term.of_years(horizon + 1))
    assert longer < shorter


@given(rate=rates, per_year=periods, horizon=st.integers(min_value=1, max_value=10))
def test_a_positive_rate_never_raises_present_value(
    rate: float, per_year: int, horizon: int
) -> None:
    """For a stream of receipts, discounting harder can only reduce the total."""
    grid = Timeline(Term.of_years(horizon), periods_per_year=per_year, rate=rate)
    undiscounted = Timeline(Term.of_years(horizon), periods_per_year=per_year, rate=0.0)
    amounts = np.full(grid.n_periods + 1, 100.0)
    assert net_present_value(amounts, grid) <= net_present_value(amounts, undiscounted) + 1e-9
