"""Invariants the cash-flow layer must hold whatever the model looks like."""

from __future__ import annotations

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from simplyinvest import Amount, Term, Timeline
from simplyinvest.cashflow import (
    CashFlowSeries,
    Component,
    Frequency,
    OneOff,
    Recurring,
    Role,
)

magnitudes = st.floats(min_value=0.0, max_value=1e6, allow_nan=False, allow_infinity=False)
rates = st.floats(min_value=0.0, max_value=0.3, allow_nan=False, allow_infinity=False)
periods = st.sampled_from([1, 2, 4, 12])
years = st.integers(min_value=1, max_value=15)


@st.composite
def series_and_grid(draw):
    """A small random series together with a grid it resolves on."""
    horizon = draw(years)
    per_year = draw(periods)
    grid = Timeline(Term.of_years(horizon), periods_per_year=per_year, rate=draw(rates))
    flows: list[object] = []
    for i in range(draw(st.integers(min_value=0, max_value=6))):
        role = draw(st.sampled_from(list(Role)))
        label = Component(role, draw(st.sampled_from(["", "a", "b"])))
        if draw(st.booleans()):
            flows.append(
                OneOff(
                    Amount.paid(draw(magnitudes)),
                    at=draw(st.integers(min_value=0, max_value=grid.n_periods)),
                    label=label,
                    description=f"flow {i}",
                )
            )
        else:
            flows.append(
                Recurring(
                    Amount.received(draw(magnitudes)),
                    Frequency.PER_PERIOD,
                    label=label,
                    description=f"flow {i}",
                )
            )
    return CashFlowSeries(tuple(flows)), grid


@given(series_and_grid())
@settings(max_examples=100)
def test_a_breakdown_always_sums_to_the_total(case) -> None:
    """The guarantee that makes a breakdown worth reading."""
    series, grid = case
    parts = series.breakdown(grid)
    assert abs(sum(parts.values()) - series.pv(grid)) <= 1e-6 * max(1.0, abs(series.pv(grid)))


@given(series_and_grid())
@settings(max_examples=100)
def test_rolling_up_to_roles_preserves_the_total(case) -> None:
    series, grid = case
    by_component = sum(series.breakdown(grid).values())
    by_role = sum(series.breakdown(grid, by=Role).values())
    assert abs(by_component - by_role) <= 1e-6 * max(1.0, abs(by_component))


@given(series_and_grid())
@settings(max_examples=100)
def test_concatenating_series_adds_their_values(case) -> None:
    """What lets an alternative be assembled from independent sources."""
    series, grid = case
    doubled = series + series
    assert abs(doubled.pv(grid) - 2 * series.pv(grid)) <= 1e-6 * max(1.0, abs(series.pv(grid)))


@given(magnitudes, rates, periods, years)
def test_a_single_outflow_today_is_worth_its_face_value(
    magnitude: float, rate: float, per_year: int, horizon: int
) -> None:
    """Nothing is discounted at period 0, whatever the grid or the rate."""
    grid = Timeline(Term.of_years(horizon), periods_per_year=per_year, rate=rate)
    series = CashFlowSeries.of(OneOff(Amount.paid(magnitude)))
    assert abs(series.pv(grid) + magnitude) <= 1e-9 * max(1.0, magnitude)


@given(magnitudes, rates, years)
def test_batched_draws_agree_with_resolving_them_one_at_a_time(
    magnitude: float, rate: float, horizon: int
) -> None:
    """The equivalence the vectorised simulation rests on."""
    grid = Timeline(Term.of_years(horizon), periods_per_year=1, rate=rate)
    draws = np.array([magnitude, magnitude * 2, magnitude * 3])
    batched = CashFlowSeries.of(OneOff(Amount.paid(draws))).pv(grid)
    one_by_one = np.array(
        [CashFlowSeries.of(OneOff(Amount.paid(float(d)))).pv(grid) for d in draws]
    )
    assert np.allclose(batched, one_by_one, atol=1e-10)
