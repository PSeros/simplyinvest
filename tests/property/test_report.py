"""Invariants the charts and frames hold for any result they are handed."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from simplyinvest import Amount, Term, Timeline
from simplyinvest.appraisal import Alternative, Case
from simplyinvest.cashflow import Role
from simplyinvest.domain import GeometricDecline
from simplyinvest.financing import AnnuityLoan, CashPurchase
from simplyinvest.report import charts, frames
from simplyinvest.uncertain import Bar, Tornado

MONEY = st.floats(min_value=-1e7, max_value=1e7, allow_nan=False, allow_infinity=False)


@pytest.fixture(scope="session")
def grid_case():
    """A factory for a purchase against a loan, at a given rate and price."""

    @dataclass(frozen=True)
    class Machine:
        price: float

        @property
        def name(self) -> str:
            return "machine"

        @property
        def capital_cost(self) -> Amount:
            return Amount.paid(self.price)

        @property
        def setup_cost(self) -> Amount:
            return Amount.paid(0.0)

        @property
        def economic_life(self) -> Term | None:
            return None

        @property
        def age_at_acquisition(self) -> Term:
            return Term.ZERO

        def residual_value(self, *, held: Term) -> Amount:
            return Amount.received(
                GeometricDecline(0.2).value_after(
                    self.price, held=held, age_at_acquisition=Term.ZERO
                )
            )

    def build(*, rate: float, price: float) -> Case:
        machine = Machine(price)
        return Case(
            alternatives=(
                Alternative("cash", sources=(CashPurchase().bind(machine),)),
                Alternative(
                    "loan",
                    sources=(AnnuityLoan(rate=0.06, term=Term.of_years(4)).bind(machine),),
                ),
            ),
            timeline=Timeline(Term.of_years(6), periods_per_year=12, rate=rate),
        )

    return build


@contextmanager
def an_axes(matplotlib):
    """A fresh axes, closed when the example that used it is done.

    Hypothesis runs every example inside one test, so a chart left to make its
    own figure each time fills pyplot's registry.
    """
    figure, ax = matplotlib.pyplot.subplots()
    try:
        yield ax
    finally:
        matplotlib.pyplot.close(figure)


def bars() -> st.SearchStrategy[tuple[Bar, ...]]:
    """Between one and six parameter swings, widest first."""
    return st.lists(
        st.builds(Bar, label=st.text(min_size=1, max_size=8), low=MONEY, high=MONEY),
        min_size=1,
        max_size=6,
    ).map(lambda found: tuple(sorted(found, key=lambda bar: bar.swing, reverse=True)))


@given(base=MONEY, found=bars())
def test_every_bar_and_the_base_fit_inside_the_tornado(matplotlib, base, found):
    """`Bar.low` is the value at the parameter's low end, so it is often the
    larger number; limits read off low and high in order clip every bar."""
    with an_axes(matplotlib) as ax:
        charts.tornado_chart(Tornado(on="x", base=base, bars=found), ax=ax)
        left, right = ax.get_xlim()
        assert left <= base <= right
        for bar in found:
            assert left <= min(bar.low, bar.high)
            assert max(bar.low, bar.high) <= right


@given(
    rate=st.floats(min_value=0.0, max_value=0.2),
    price=st.floats(min_value=1_000.0, max_value=500_000.0),
)
def test_the_breakdown_frame_sums_to_the_present_value(pandas, grid_case, rate, price):
    """A column of components is the present value, taken apart."""
    result = grid_case(rate=rate, price=price).run()
    frame = frames.breakdown_frame(result)
    for name in result.names:
        assert frame[name].sum() == pytest.approx(float(result[name].npv), rel=1e-9, abs=1e-6)


@given(
    rate=st.floats(min_value=0.0, max_value=0.2),
    price=st.floats(min_value=1_000.0, max_value=500_000.0),
)
def test_rolling_up_by_role_keeps_every_cent(pandas, grid_case, rate, price):
    """Details roll into roles without losing anything."""
    result = grid_case(rate=rate, price=price).run()
    detailed = frames.breakdown_frame(result)
    rolled = frames.breakdown_frame(result, by=Role)
    for name in result.names:
        assert rolled[name].sum() == pytest.approx(detailed[name].sum(), rel=1e-9, abs=1e-6)


@given(
    rate=st.floats(min_value=0.0, max_value=0.2),
    price=st.floats(min_value=1_000.0, max_value=500_000.0),
)
def test_the_cumulative_line_ends_on_the_net_present_value(matplotlib, grid_case, rate, price):
    """The running total of discounted amounts is the present value by the end."""
    result = grid_case(rate=rate, price=price).run()
    with an_axes(matplotlib) as ax:
        charts.cumulative_chart(result, ax=ax)
        for line, name in zip(ax.get_lines(), result.names, strict=False):
            assert float(np.asarray(line.get_ydata())[-1]) == pytest.approx(
                float(result[name].npv), rel=1e-9, abs=1e-6
            )
