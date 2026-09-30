"""Cash flows resolve onto the grid, and their parts add up to their whole."""

from __future__ import annotations

import numpy as np
import pytest

from simplyinvest import Amount, Term, Timeline
from simplyinvest.cashflow import (
    CashFlowSeries,
    Component,
    Explicit,
    Frequency,
    OneOff,
    Recurring,
    Role,
    Terminal,
)
from simplyinvest.errors import CashFlowError, TermNotRepresentableError


@pytest.fixture
def annual():
    """A three-year annual grid at 10%, small enough to check by hand."""
    return Timeline(Term.of_years(3), periods_per_year=1, rate=0.10)


class TestComponent:
    def test_a_domain_may_invent_a_detail_without_touching_the_core(self):
        feed_in = Component(Role.REVENUE, "feed_in")
        assert feed_in.role is Role.REVENUE
        assert str(feed_in) == "revenue/feed_in"

    def test_components_sort_and_hash(self):
        labels = {Component(Role.CAPITAL), Component(Role.REVENUE, "x")}
        assert len(labels) == 2
        assert sorted(labels)[0].role is Role.CAPITAL


class TestFrequency:
    def test_stride_on_a_monthly_grid(self):
        grid = Timeline(Term.of_years(1), periods_per_year=12)
        assert Frequency.ANNUAL.stride(grid) == 12
        assert Frequency.QUARTERLY.stride(grid) == 3
        assert Frequency.PER_PERIOD.stride(grid) == 1

    def test_a_frequency_finer_than_the_grid_is_refused(self):
        grid = Timeline(Term.of_years(5), periods_per_year=1)
        with pytest.raises(TermNotRepresentableError):
            Frequency.MONTHLY.stride(grid)


class TestOneOff:
    def test_it_lands_where_it_is_put(self, annual):
        assert np.array_equal(
            OneOff(Amount.paid(1_000), at=2).amounts(annual),
            np.array([0.0, 0.0, -1_000.0, 0.0]),
        )

    def test_off_the_grid_is_refused(self, annual):
        with pytest.raises(CashFlowError, match="outside the grid"):
            OneOff(Amount.paid(1_000), at=9).amounts(annual)


class TestRecurring:
    def test_payments_fall_in_arrears(self, annual):
        amounts = Recurring(Amount.paid(100), Frequency.ANNUAL).amounts(annual)
        assert np.array_equal(amounts, np.array([0.0, -100.0, -100.0, -100.0]))

    def test_it_can_stop_early(self, annual):
        amounts = Recurring(Amount.paid(100), Frequency.ANNUAL, end=2).amounts(annual)
        assert np.array_equal(amounts, np.array([0.0, -100.0, -100.0, 0.0]))

    def test_it_can_start_late(self, annual):
        amounts = Recurring(Amount.paid(100), Frequency.ANNUAL, start=1).amounts(annual)
        assert np.array_equal(amounts, np.array([0.0, 0.0, -100.0, -100.0]))

    def test_a_window_that_admits_no_payment_is_empty(self, annual):
        amounts = Recurring(Amount.paid(100), Frequency.ANNUAL, start=3).amounts(annual)
        assert np.array_equal(amounts, np.zeros(4))

    def test_escalation_by_name(self):
        grid = Timeline(Term.of_years(3), periods_per_year=1, escalations={"energy": 0.10})
        amounts = Recurring(Amount.paid(100), Frequency.ANNUAL, growth="energy").amounts(grid)
        assert amounts == pytest.approx([0.0, -100.0, -110.0, -121.0])

    @pytest.mark.parametrize("per_year", [1, 2, 4, 12])
    def test_escalation_does_not_depend_on_the_grid(self, per_year):
        """A year of payments costs the same however finely the year is sliced."""
        grid = Timeline(Term.of_years(3), periods_per_year=per_year, escalations={"energy": 0.10})
        flow = Recurring(Amount.paid(1_200 / per_year), Frequency.PER_PERIOD, growth="energy")
        amounts = flow.amounts(grid)
        first_year = -amounts[1 : per_year + 1].sum()
        second_year = -amounts[per_year + 1 : 2 * per_year + 1].sum()
        assert first_year == pytest.approx(1_200.0)
        assert second_year == pytest.approx(1_320.0)

    def test_an_undeclared_escalation_does_not_grow(self, annual):
        amounts = Recurring(Amount.paid(100), Frequency.ANNUAL, growth="not_declared").amounts(
            annual
        )
        assert np.array_equal(amounts, np.array([0.0, -100.0, -100.0, -100.0]))


class TestTerminal:
    def test_it_is_realised_at_its_period(self, annual):
        flow = Terminal(Amount.received(5_000), at=3, basis="geometric decline")
        assert np.array_equal(flow.amounts(annual), np.array([0.0, 0.0, 0.0, 5_000.0]))
        assert flow.label.role is Role.TERMINAL


class TestExplicit:
    def test_it_takes_signed_values_directly(self, annual):
        values = np.array([-100.0, 50.0, -20.0, 300.0])
        assert np.array_equal(Explicit(values).amounts(annual), values)

    def test_the_wrong_length_is_refused(self, annual):
        with pytest.raises(CashFlowError, match="periods but the grid"):
            Explicit(np.zeros(9)).amounts(annual)

    def test_a_nan_is_refused(self, annual):
        with pytest.raises(CashFlowError, match="non-finite"):
            Explicit(np.array([1.0, np.nan, 1.0, 1.0])).amounts(annual)


class TestTheFirstRealNpv:
    """Pay 1,000 today, receive 500 at the end of each of three years, at 10%.

    PV = -1000 + 500/1.1 + 500/1.21 + 500/1.331 = 243.4259...
    """

    EXPECTED = -1_000 + 500 / 1.1 + 500 / 1.1**2 + 500 / 1.1**3

    def test_it_matches_the_hand_calculation(self, annual):
        series = CashFlowSeries.of(
            OneOff(Amount.paid(1_000)),
            Recurring(Amount.received(500), Frequency.ANNUAL),
        )
        assert series.pv(annual) == pytest.approx(self.EXPECTED, abs=1e-9)

    def test_the_arithmetic_is_what_we_think_it_is(self):
        assert pytest.approx(243.42599549, abs=1e-8) == self.EXPECTED


class TestSeries:
    def test_an_empty_series_is_worth_nothing(self, annual):
        assert CashFlowSeries().pv(annual) == 0.0
        assert not CashFlowSeries()

    def test_series_concatenate_and_add(self, annual):
        a = CashFlowSeries.of(OneOff(Amount.paid(100)))
        b = CashFlowSeries.of(OneOff(Amount.received(100)))
        assert len(a + b) == 2
        assert (a + b).pv(annual) == 0.0
        assert len(CashFlowSeries.concat([a, b, a])) == 3

    def test_undiscounted_ignores_the_rate(self, annual):
        series = CashFlowSeries.of(Recurring(Amount.paid(100), Frequency.ANNUAL))
        assert series.undiscounted(annual) == pytest.approx(-300.0)

    def test_labelled_filters(self, annual):
        series = CashFlowSeries.of(
            OneOff(Amount.paid(100), label=Component(Role.CAPITAL)),
            OneOff(Amount.paid(10), label=Component(Role.OPERATING, "fuel")),
        )
        assert len(series.labelled(role=Role.CAPITAL)) == 1
        assert len(series.labelled(detail="fuel")) == 1


class TestBreakdown:
    @pytest.fixture
    def mixed(self):
        return CashFlowSeries.of(
            OneOff(Amount.paid(1_000), label=Component(Role.CAPITAL)),
            Recurring(
                Amount.paid(50), Frequency.ANNUAL, label=Component(Role.OPERATING, "maintenance")
            ),
            Recurring(
                Amount.paid(30), Frequency.ANNUAL, label=Component(Role.OPERATING, "insurance")
            ),
            Terminal(Amount.received(400), at=3),
        )

    def test_the_parts_sum_to_the_whole(self, mixed, annual):
        parts = mixed.breakdown(annual)
        assert sum(parts.values()) == pytest.approx(mixed.pv(annual), abs=1e-9)

    def test_details_stay_separate_by_component(self, mixed, annual):
        parts = mixed.breakdown(annual)
        assert Component(Role.OPERATING, "maintenance") in parts
        assert Component(Role.OPERATING, "insurance") in parts

    def test_roles_roll_the_details_up(self, mixed, annual):
        parts = mixed.breakdown(annual, by=Role)
        assert set(parts) == {Role.CAPITAL, Role.OPERATING, Role.TERMINAL}
        assert sum(parts.values()) == pytest.approx(mixed.pv(annual), abs=1e-9)

    def test_detail_lists_every_flow(self, mixed, annual):
        assert len(mixed.detail(annual)) == 4


class TestBatch:
    def test_a_batched_amount_gives_one_row_per_trial(self, annual):
        prices = np.array([900.0, 1_000.0, 1_100.0])
        amounts = OneOff(Amount.paid(prices)).amounts(annual)
        assert amounts.shape == (3, 4)
        assert np.array_equal(amounts[:, 0], -prices)

    def test_scalar_and_batched_flows_combine(self, annual):
        series = CashFlowSeries.of(
            OneOff(Amount.paid(np.array([900.0, 1_100.0]))),
            Recurring(Amount.received(500), Frequency.ANNUAL),
        )
        values = series.pv(annual)
        assert values.shape == (2,)
        scalar = CashFlowSeries.of(
            OneOff(Amount.paid(900.0)), Recurring(Amount.received(500), Frequency.ANNUAL)
        ).pv(annual)
        assert values[0] == pytest.approx(scalar)

    def test_a_batch_costs_one_discounting_pass(self, annual):
        series = CashFlowSeries.of(OneOff(Amount.paid(np.full(5_000, 1_000.0))))
        assert series.pv(annual).shape == (5_000,)
