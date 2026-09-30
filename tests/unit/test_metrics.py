"""The appraisal metrics, against figures a spreadsheet would give."""

from __future__ import annotations

import numpy as np
import pytest

from simplyinvest import Term, Timeline
from simplyinvest.metrics import (
    annuity_factor,
    capital_recovery_factor,
    discounted_payback,
    equivalent_annual_cost,
    irr,
    mirr,
    net_present_value,
    payback,
    profitability_index,
    pv_of_inflows,
    pv_of_outflows,
)


@pytest.fixture
def annual():
    return Timeline(Term.of_years(3), periods_per_year=1, rate=0.10)


class TestPresentValue:
    def test_against_the_hand_calculation(self, annual):
        amounts = np.array([-1_000.0, 500.0, 500.0, 500.0])
        assert net_present_value(amounts, annual) == pytest.approx(243.42599549, abs=1e-8)

    def test_outflows_and_inflows_split_the_total(self, annual):
        amounts = np.array([-1_000.0, 500.0, 500.0, 500.0])
        assert pv_of_inflows(amounts, annual) - pv_of_outflows(amounts, annual) == pytest.approx(
            net_present_value(amounts, annual)
        )

    def test_outflows_are_reported_positive(self, annual):
        assert pv_of_outflows(np.array([-100.0, 0.0, 0.0, 0.0]), annual) == pytest.approx(100.0)

    def test_profitability_index(self, annual):
        amounts = np.array([-1_000.0, 500.0, 500.0, 500.0])
        assert profitability_index(amounts, annual) == pytest.approx(1.2434259955, abs=1e-8)

    def test_an_investment_with_no_outlay_has_infinite_index(self, annual):
        assert profitability_index(np.array([0.0, 1.0, 1.0, 1.0]), annual) == np.inf


class TestIrr:
    def test_a_textbook_rate(self):
        grid = Timeline(Term.of_years(2), periods_per_year=1)
        result = irr(np.array([-100.0, 0.0, 121.0]), grid)
        assert result
        assert result.value == pytest.approx(0.10, abs=1e-9)

    def test_the_rate_zeroes_the_present_value(self, annual):
        amounts = np.array([-1_000.0, 500.0, 500.0, 500.0])
        result = irr(amounts, annual)
        at_irr = Timeline(Term.of_years(3), periods_per_year=1, rate=result.value)
        assert at_irr.pv(amounts) == pytest.approx(0.0, abs=1e-9)

    def test_it_works_on_a_monthly_grid(self):
        grid = Timeline(Term.of_years(1), periods_per_year=12)
        amounts = np.zeros(13)
        amounts[0], amounts[12] = -1_000.0, 1_100.0
        assert irr(amounts, grid).value == pytest.approx(0.10, abs=1e-9)

    def test_a_long_horizon_does_not_break_it(self):
        grid = Timeline(Term.of_years(20), periods_per_year=12)
        amounts = np.zeros(241)
        amounts[0] = -14_000.0
        amounts[1:] = 110.0
        result = irr(amounts, grid)
        assert result
        at_irr = Timeline(Term.of_years(20), periods_per_year=12, rate=result.value)
        assert at_irr.pv(amounts) == pytest.approx(0.0, abs=1e-6)

    def test_an_all_one_way_stream_has_no_rate(self, annual):
        result = irr(np.array([-100.0, -100.0, -100.0, -100.0]), annual)
        assert not result
        assert "never changes direction" in result.reason

    def test_a_stream_that_turns_twice_is_refused(self, annual):
        result = irr(np.array([-100.0, 300.0, -250.0, 100.0]), annual)
        assert not result
        assert result.sign_changes == 3
        assert "net present value" in result.reason

    def test_a_negative_rate_is_found(self, annual):
        result = irr(np.array([-1_000.0, 100.0, 100.0, 100.0]), annual)
        assert result
        assert result.value < 0

    def test_a_batch_is_refused_rather_than_guessed_at(self, annual):
        with pytest.raises(ValueError, match="one stream at a time"):
            irr(np.zeros((5, 4)), annual)

    def test_str_is_readable(self, annual):
        assert str(irr(np.array([-100.0, 0.0, 0.0, 133.1]), annual)) == "10.00%"
        assert "no IRR" in str(irr(np.array([-1.0, -1.0, -1.0, -1.0]), annual))


class TestMirr:
    def test_it_sits_between_the_finance_and_reinvest_assumptions(self, annual):
        amounts = np.array([-1_000.0, 500.0, 500.0, 500.0])
        modified = mirr(amounts, annual, finance_rate=0.05, reinvest_rate=0.05)
        assert 0.0 < modified < irr(amounts, annual).value

    def test_it_needs_something_to_finance(self, annual):
        with pytest.raises(ValueError, match="at least one outflow"):
            mirr(np.array([0.0, 1.0, 1.0, 1.0]), annual, finance_rate=0.05, reinvest_rate=0.05)


class TestPayback:
    def test_a_simple_payback(self):
        grid = Timeline(Term.of_years(5), periods_per_year=1)
        amounts = np.array([-300.0, 100.0, 100.0, 100.0, 0.0, 0.0])
        assert payback(amounts, grid) == Term.of_years(3)

    def test_a_stream_that_never_recovers(self):
        grid = Timeline(Term.of_years(3), periods_per_year=1)
        assert payback(np.array([-300.0, 10.0, 10.0, 10.0]), grid) is None

    def test_discounting_never_shortens_it(self):
        grid = Timeline(Term.of_years(5), periods_per_year=1, rate=0.10)
        amounts = np.array([-300.0, 100.0, 100.0, 100.0, 100.0, 100.0])
        assert discounted_payback(amounts, grid) >= payback(amounts, grid)

    def test_a_stream_that_never_goes_negative_pays_back_at_once(self):
        grid = Timeline(Term.of_years(3), periods_per_year=1)
        assert payback(np.array([0.0, 100.0, 100.0, 100.0]), grid) == Term.ZERO

    def test_it_is_measured_from_the_first_outflow_not_from_period_zero(self):
        grid = Timeline(Term.of_years(4), periods_per_year=1)
        amounts = np.array([0.0, -300.0, 100.0, 100.0, 100.0])
        assert payback(amounts, grid) == Term.of_years(4)

    def test_it_reports_a_span_not_a_period_count(self):
        monthly = Timeline(Term.of_years(5), periods_per_year=12)
        quarterly = Timeline(Term.of_years(5), periods_per_year=4)
        monthly_amounts = np.concatenate([[-360.0], np.full(60, 10.0)])
        quarterly_amounts = np.concatenate([[-360.0], np.full(20, 30.0)])
        assert payback(monthly_amounts, monthly) == payback(quarterly_amounts, quarterly)


class TestAnnuity:
    def test_annuity_factor_against_the_tables(self):
        assert annuity_factor(0.05, Term.of_years(10)) == pytest.approx(7.721735, abs=1e-6)

    def test_the_two_factors_are_reciprocals(self):
        rate, horizon = 0.07, Term.of_years(12)
        assert annuity_factor(rate, horizon) * capital_recovery_factor(rate, horizon) == (
            pytest.approx(1.0, abs=1e-12)
        )

    def test_a_zero_rate_spreads_evenly(self):
        assert annuity_factor(0.0, Term.of_years(10)) == 10.0
        assert capital_recovery_factor(0.0, Term.of_years(10)) == pytest.approx(0.1)

    def test_equivalent_annual_cost_reverses_to_the_present_value(self):
        grid = Timeline(Term.of_years(10), rate=0.05)
        cost = 10_000.0
        annual = equivalent_annual_cost(cost, grid)
        assert annual * annuity_factor(grid.rate, grid.horizon) == pytest.approx(cost, abs=1e-9)

    def test_a_shorter_life_carries_a_higher_annual_charge(self):
        grid = Timeline(Term.of_years(10), rate=0.05)
        assert equivalent_annual_cost(10_000.0, grid, Term.of_years(3)) > equivalent_annual_cost(
            10_000.0, grid, Term.of_years(10)
        )
