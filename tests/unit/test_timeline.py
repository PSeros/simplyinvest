"""The period grid: durations, discounting, escalation and the calendar."""

from __future__ import annotations

from datetime import date

import numpy as np
import pytest

from simplyinvest.errors import (
    InconsistentRateBasisError,
    NoCalendarAnchorError,
    NotAnchoredError,
    TermNotRepresentableError,
    TimelineError,
)
from simplyinvest.timeline import (
    Escalation,
    EscalationSet,
    Periodisation,
    RateBasis,
    Term,
    Timeline,
    add_months,
    fisher_nominal,
    fisher_real,
    months_between,
)


class TestTerm:
    def test_years_and_months_agree(self):
        assert Term.of_years(3) == Term.of_months(36)

    def test_a_fractional_year_that_lands_on_a_month_is_fine(self):
        assert Term.of_years(1.5) == Term.of_months(18)

    def test_a_fractional_year_that_does_not_is_refused(self):
        with pytest.raises(ValueError, match="not a whole number"):
            Term.of_years(0.1)

    def test_a_fractional_month_is_refused(self):
        with pytest.raises(TypeError, match="whole number of months"):
            Term(1.5)

    def test_a_bool_is_not_a_month_count(self):
        with pytest.raises(TypeError):
            Term(True)

    def test_arithmetic(self):
        assert Term.of_years(1) + Term.of_months(6) == Term.of_months(18)
        assert Term.of_years(2) - Term.of_months(6) == Term.of_months(18)
        assert Term.of_months(4) * 3 == Term.of_years(1)
        assert 3 * Term.of_months(4) == Term.of_years(1)
        assert -Term.of_years(1) == Term.of_months(-12)

    def test_terms_order(self):
        assert Term.of_months(18) > Term.of_years(1)
        assert sorted([Term.of_years(2), Term.of_months(1)])[0] == Term.of_months(1)

    def test_a_term_will_not_add_to_a_bare_integer(self):
        with pytest.raises(TypeError):
            Term.of_months(36) + 12

    @pytest.mark.parametrize(
        ("term", "text"),
        [
            (Term.of_years(2), "2 years"),
            (Term.of_years(1), "1 year"),
            (Term.of_months(5), "5 months"),
        ],
    )
    def test_str(self, term, text):
        assert str(term) == text


class TestTermsOnAGrid:
    def test_a_term_that_lands_on_the_grid(self):
        grid = Timeline(Term.of_years(5), periods_per_year=4)
        assert grid.periods_in(Term.of_months(36)) == 12

    def test_a_term_that_does_not_is_refused(self):
        grid = Timeline(Term.of_years(5), periods_per_year=4)
        with pytest.raises(TermNotRepresentableError, match="not a whole number"):
            grid.periods_in(Term.of_months(5))

    def test_the_refusal_explains_the_grid(self):
        grid = Timeline(Term.of_years(5), periods_per_year=4)
        with pytest.raises(TermNotRepresentableError, match="3 months each"):
            grid.periods_in(Term.of_months(5))

    @pytest.mark.parametrize(("per_year", "expected"), [(12, 36), (4, 12), (2, 6), (1, 3)])
    def test_the_same_term_across_grids(self, per_year, expected):
        grid = Timeline(Term.of_years(5), periods_per_year=per_year)
        assert grid.periods_in(Term.of_years(3)) == expected

    def test_offset_advances_by_a_term(self):
        grid = Timeline(Term.of_years(5), periods_per_year=4)
        assert grid.offset(2, Term.of_years(1)) == 6

    def test_offset_off_the_end_is_refused(self):
        grid = Timeline(Term.of_years(2))
        with pytest.raises(TimelineError, match="outside the grid"):
            grid.offset(0, Term.of_years(3))


class TestShape:
    def test_periods_run_from_zero_inclusive(self):
        grid = Timeline(Term.of_years(5))
        assert grid.n_periods == 60
        assert grid.periods[0] == 0
        assert grid.periods[-1] == 60
        assert len(grid.periods) == 61

    def test_a_non_positive_horizon_is_refused(self):
        with pytest.raises(TimelineError, match="positive"):
            Timeline(Term.of_months(0))

    def test_a_horizon_that_misses_the_grid_is_refused(self):
        with pytest.raises(TermNotRepresentableError):
            Timeline(Term.of_months(5), periods_per_year=4)

    def test_zeros_match_the_grid(self):
        grid = Timeline(Term.of_years(1))
        assert grid.zeros().shape == (13,)
        assert grid.zeros(trials=100).shape == (100, 13)


class TestDiscounting:
    def test_a_zero_rate_discounts_nothing(self):
        grid = Timeline(Term.of_years(5), rate=0.0)
        assert grid.discount_factor(60) == 1.0

    def test_conformal_periods_compound_to_the_annual_rate(self):
        """Twelve conformal monthly periods are worth exactly one annual period."""
        grid = Timeline(Term.of_years(1), periods_per_year=12, rate=0.03)
        assert grid.discount_factor(12) == pytest.approx(1 / 1.03, abs=1e-12)

    def test_proportional_periods_do_not(self):
        grid = Timeline(Term.of_years(1), rate=0.03, periodisation=Periodisation.PROPORTIONAL)
        assert grid.periodic_rate == pytest.approx(0.03 / 12)
        assert grid.discount_factor(12) < 1 / 1.03

    def test_pv_of_a_single_unit(self):
        grid = Timeline(Term.of_years(1), periods_per_year=1, rate=0.10)
        amounts = np.array([0.0, 110.0])
        assert grid.pv(amounts) == pytest.approx(100.0)

    def test_pv_of_a_batch_discounts_every_trial_at_once(self):
        grid = Timeline(Term.of_years(1), periods_per_year=1, rate=0.10)
        batch = np.array([[0.0, 110.0], [0.0, 220.0], [-100.0, 110.0]])
        assert np.allclose(grid.pv(batch), [100.0, 200.0, 0.0])

    def test_pv_of_the_wrong_length_is_refused(self):
        grid = Timeline(Term.of_years(1), periods_per_year=1)
        with pytest.raises(TimelineError, match="periods"):
            grid.pv(np.zeros(5))


class TestRateBasis:
    def test_a_real_rate_must_state_its_inflation(self):
        with pytest.raises(InconsistentRateBasisError, match="inflation"):
            Timeline(Term.of_years(5), rate=0.03, basis=RateBasis.REAL)

    def test_nominal_discounting_of_unescalated_flows_is_refused(self):
        with pytest.raises(InconsistentRateBasisError, match="today's money"):
            Timeline(Term.of_years(5), rate=0.05, inflation=0.02)

    def test_nominal_with_escalation_is_fine(self):
        grid = Timeline(Term.of_years(5), rate=0.05, inflation=0.02, escalations={"energy": 0.02})
        assert grid.basis is RateBasis.NOMINAL

    def test_fisher_round_trips_exactly(self):
        assert fisher_real(fisher_nominal(0.03, 0.02), 0.02) == pytest.approx(0.03, abs=1e-15)

    def test_real_to_nominal_converts_every_escalation(self):
        grid = Timeline.real(
            Term.of_years(5), rate=0.03, inflation=0.02, escalations={"electricity": 0.01}
        )
        nominal = grid.to_nominal()
        assert nominal.basis is RateBasis.NOMINAL
        assert nominal.rate == pytest.approx(fisher_nominal(0.03, 0.02))
        assert nominal.escalations["electricity"] == pytest.approx(fisher_nominal(0.01, 0.02))

    def test_nominal_to_real_and_back(self):
        grid = Timeline(Term.of_years(5), rate=0.0506, escalations={"energy": 0.03})
        assert grid.to_real(0.02).to_nominal().rate == pytest.approx(0.0506, abs=1e-12)


class TestEscalation:
    def test_an_undeclared_rate_does_not_escalate(self):
        grid = Timeline(Term.of_years(2), periods_per_year=1)
        assert np.allclose(grid.escalation_index("nothing_declared"), 1.0)

    def test_annual_step_holds_flat_within_a_year(self):
        grid = Timeline(Term.of_years(2), periods_per_year=12, escalations={"energy": 0.10})
        index = grid.escalation_index("energy")
        assert np.allclose(index[:12], 1.0)
        assert index[12] == pytest.approx(1.10)
        assert index[24] == pytest.approx(1.21)

    def test_continuous_creeps_every_period(self):
        grid = Timeline(
            Term.of_years(1),
            periods_per_year=12,
            escalations={"energy": 0.10},
            escalation_mode=Escalation.CONTINUOUS,
        )
        index = grid.escalation_index("energy")
        assert index[1] > 1.0
        assert index[12] == pytest.approx(1.10)

    def test_the_accrual_index_lags_the_point_index_by_one_interval(self):
        grid = Timeline(Term.of_years(3), periods_per_year=1, escalations={"energy": 0.10})
        assert grid.escalation_index("energy") == pytest.approx([1.0, 1.1, 1.21, 1.331])
        assert grid.accrual_index("energy") == pytest.approx([1.0, 1.0, 1.1, 1.21])

    @pytest.mark.parametrize("per_year", [1, 2, 4, 12])
    def test_the_first_year_accrues_at_the_base_price_on_every_grid(self, per_year):
        grid = Timeline(Term.of_years(3), periods_per_year=per_year, escalations={"energy": 0.10})
        index = grid.accrual_index("energy")
        assert np.allclose(index[1 : per_year + 1], 1.0)
        assert np.allclose(index[per_year + 1 : 2 * per_year + 1], 1.1)

    def test_an_explicit_rate_may_be_passed_instead_of_a_name(self):
        grid = Timeline(Term.of_years(1), periods_per_year=1)
        assert grid.escalation_index(0.05)[1] == pytest.approx(1.05)

    def test_a_ruinous_rate_is_refused(self):
        with pytest.raises(ValueError, match="100%"):
            EscalationSet({"energy": -1.0})


class TestCalendar:
    def test_an_unanchored_grid_will_not_answer_calendar_questions(self):
        grid = Timeline(Term.of_years(5))
        assert not grid.anchored
        with pytest.raises(NotAnchoredError, match="start_date"):
            grid.date_of(12)

    def test_a_year_on_is_the_same_day_next_year(self):
        grid = Timeline(Term.of_years(2), start_date=date(2026, 5, 1))
        assert grid.date_of(12) == date(2027, 5, 1)

    def test_twenty_years_on_does_not_drift(self):
        grid = Timeline(Term.of_years(25), start_date=date(2026, 5, 1))
        assert grid.date_of(240) == date(2046, 5, 1)

    def test_a_quarterly_grid_steps_three_months(self):
        grid = Timeline(Term.of_years(2), periods_per_year=4, start_date=date(2026, 1, 1))
        assert grid.date_of(1) == date(2026, 4, 1)

    def test_period_of_is_the_inverse_of_date_of(self):
        grid = Timeline(Term.of_years(5), start_date=date(2026, 5, 1))
        for t in (0, 1, 7, 37, 60):
            assert grid.period_of(grid.date_of(t)) == t

    def test_a_date_before_the_start_is_a_negative_period(self):
        grid = Timeline(Term.of_years(5), start_date=date(2026, 5, 1))
        assert grid.period_of(date(2025, 5, 1)) == -12

    def test_period_bounds_use_real_month_lengths(self):
        grid = Timeline(Term.of_years(1), start_date=date(2026, 1, 1))
        feb_start, feb_end = grid.period_bounds(1)
        jun_start, jun_end = grid.period_bounds(5)
        assert (feb_end - feb_start).days == 28
        assert (jun_end - jun_start).days == 30

    def test_an_unmappable_grid_refuses_an_anchor(self):
        with pytest.raises(NoCalendarAnchorError, match="whole months"):
            Timeline(Term.of_years(5), periods_per_year=5, start_date=date(2026, 1, 1))

    def test_month_end_clamps_rather_than_overflowing(self):
        assert add_months(date(2026, 1, 31), 1) == date(2026, 2, 28)

    def test_months_between_counts_whole_months(self):
        assert months_between(date(2026, 5, 1), date(2026, 5, 31)) == 0
        assert months_between(date(2026, 5, 1), date(2026, 6, 1)) == 1


class TestTheMonthsBug:
    """A term means the same span whatever the grid resolution."""

    @pytest.mark.parametrize("per_year", [1, 2, 4, 12])
    def test_a_term_spans_the_same_calendar_time_on_every_grid(self, per_year):
        grid = Timeline(Term.of_years(10), periods_per_year=per_year, start_date=date(2026, 1, 1))
        end = grid.offset(0, Term.of_years(3))
        assert grid.date_of(end) == date(2029, 1, 1)

    @pytest.mark.parametrize("per_year", [1, 2, 4, 12])
    def test_a_term_discounts_to_the_same_present_value_on_every_grid(self, per_year):
        grid = Timeline(Term.of_years(10), periods_per_year=per_year, rate=0.04)
        at = grid.offset(0, Term.of_years(3))
        assert grid.discount_factor(at) == pytest.approx(1.04**-3, abs=1e-12)
