"""Charts and frames over real result objects.

Each chart is asserted on what it drew — artists, labels, limits — rather than
on an image, so a defect that a reader would see is a failing test.
"""

from __future__ import annotations

import sys

import numpy as np
import pytest
from cases import an_uncertain_purchase, loan_vs_cash

from simplyinvest import Amount, Term, Timeline
from simplyinvest.cashflow import Role
from simplyinvest.domain import Context
from simplyinvest.financing import AnnuityLoan
from simplyinvest.report import charts, frames
from simplyinvest.uncertain import Scenario, one_way, run_scenarios, simulate, switch_point, tornado

TRIALS = 400


@pytest.fixture(autouse=True)
def _close_every_figure():
    """A chart asked for its own axes makes a figure that pyplot then holds open."""
    yield
    pyplot = sys.modules.get("matplotlib.pyplot")
    if pyplot is not None:
        pyplot.close("all")


@pytest.fixture(scope="module")
def comparison():
    return loan_vs_cash().run()


@pytest.fixture(scope="module")
def marked():
    return an_uncertain_purchase()


@pytest.fixture(scope="module")
def trials(marked):
    return simulate(marked, n=TRIALS, seed=424_242)


@pytest.fixture(scope="module")
def swings(marked):
    return tornado(marked, on="buy")


@pytest.fixture(scope="module")
def walk(marked):
    return one_way(marked, "hire_rate", np.linspace(600.0, 2_000.0, 21))


@pytest.fixture(scope="module")
def table(marked):
    return run_scenarios(
        marked,
        (
            Scenario("as quoted", {}),
            Scenario("dear machine", {"price": 82_000.0}),
            Scenario("cheap hire", {"hire_rate": 780.0}),
        ),
    )


@pytest.fixture(scope="module")
def schedule():
    loan = AnnuityLoan(rate=0.055, term=Term.of_years(4))
    timeline = Timeline(Term.of_years(6), periods_per_year=12, rate=0.03)
    return loan.schedule(Amount.paid(40_000.0), Context(timeline=timeline))


class TestTheChartsNeedAnAxes:
    def test_one_is_made_when_none_is_given(self, matplotlib, comparison):
        assert charts.breakdown_chart(comparison) is not None

    def test_the_one_given_is_the_one_returned(self, matplotlib, comparison):
        figure, ax = matplotlib.pyplot.subplots()
        assert charts.breakdown_chart(comparison, ax=ax) is ax
        matplotlib.pyplot.close(figure)

    @pytest.mark.parametrize(
        "draw",
        [
            charts.breakdown_chart,
            charts.cumulative_chart,
        ],
    )
    def test_nothing_is_shown_and_no_style_is_set(self, matplotlib, comparison, draw, monkeypatch):
        """A library drawing on someone else's figure must not take the screen."""
        monkeypatch.setattr(
            matplotlib.pyplot, "show", lambda *a, **k: pytest.fail("the chart called plt.show()")
        )
        before = dict(matplotlib.rcParams)
        draw(comparison)
        assert dict(matplotlib.rcParams) == before


class TestTheBreakdownChart:
    def test_one_group_of_bars_per_alternative(self, matplotlib, comparison):
        ax = charts.breakdown_chart(comparison)
        assert [text.get_text() for text in ax.get_legend().get_texts()] == list(comparison.names)

    def test_every_component_gets_a_row(self, matplotlib, comparison):
        ax = charts.breakdown_chart(comparison)
        labels = {label.get_text() for label in ax.get_yticklabels()}
        expected = {str(key) for appraisal in comparison for key in appraisal.breakdown()}
        assert labels == expected

    def test_rolling_up_by_role_leaves_fewer_rows(self, matplotlib, comparison):
        detailed = charts.breakdown_chart(comparison)
        rolled = charts.breakdown_chart(comparison, by=Role)
        assert len(rolled.get_yticklabels()) < len(detailed.get_yticklabels())

    def test_a_single_appraisal_needs_no_legend(self, matplotlib, comparison):
        ax = charts.breakdown_chart(comparison["cash"])
        assert ax.get_legend() is None

    def test_the_bars_carry_the_present_values(self, matplotlib, comparison):
        ax = charts.breakdown_chart(comparison["cash"])
        drawn = sorted(round(patch.get_width(), 6) for patch in ax.patches)
        wanted = sorted(round(float(v), 6) for v in comparison["cash"].breakdown().values())
        assert drawn == wanted


class TestTheCumulativeChart:
    def test_one_line_per_alternative(self, matplotlib, comparison):
        ax = charts.cumulative_chart(comparison)
        assert [line.get_label() for line in ax.get_lines()[: len(comparison)]] == list(
            comparison.names
        )

    def test_the_last_point_is_the_net_present_value(self, matplotlib, comparison):
        ax = charts.cumulative_chart(comparison)
        for line, name in zip(ax.get_lines(), comparison.names, strict=False):
            assert line.get_ydata()[-1] == pytest.approx(float(comparison[name].npv))

    def test_undiscounted_ends_on_the_plain_total(self, matplotlib, comparison):
        ax = charts.cumulative_chart(comparison, discounted=False)
        total = float(comparison["cash"].amounts.sum())
        assert ax.get_lines()[0].get_ydata()[-1] == pytest.approx(total)

    def test_an_anchored_grid_is_drawn_against_dates(self, matplotlib, comparison):
        ax = charts.cumulative_chart(comparison)
        assert ax.get_xlabel() == "date"

    def test_an_unanchored_grid_is_drawn_against_periods(self, matplotlib):
        result = loan_vs_cash()
        bare = result.__class__(
            alternatives=result.alternatives,
            timeline=Timeline(Term.of_years(8), periods_per_year=12, rate=0.05),
            usage=result.usage,
        ).run()
        assert charts.cumulative_chart(bare).get_xlabel() == "period"


class TestTheScheduleCharts:
    def test_the_split_stacks_to_the_instalment(self, matplotlib, schedule):
        ax = charts.schedule_chart(schedule)
        tops = [patch.get_y() + patch.get_height() for patch in ax.patches]
        assert max(tops) == pytest.approx(schedule.instalment, rel=1e-9)

    def test_interest_and_principal_are_both_drawn(self, matplotlib, schedule):
        ax = charts.schedule_chart(schedule)
        assert [text.get_text() for text in ax.get_legend().get_texts()] == [
            "interest",
            "principal",
        ]

    def test_the_balance_starts_at_the_principal_and_closes(self, matplotlib, schedule):
        ax = charts.balance_chart(schedule)
        drawn = ax.get_lines()[0].get_ydata()
        assert drawn[0] == pytest.approx(float(schedule.opening[0]))
        assert drawn[-1] == pytest.approx(0.0, abs=1e-6)


class TestTheDistributionCharts:
    def test_every_alternative_is_in_the_legend(self, matplotlib, trials):
        ax = charts.distribution_chart(trials)
        assert [text.get_text() for text in ax.get_legend().get_texts()] == list(trials.names)

    def test_the_counts_add_up_to_the_trials(self, matplotlib, trials):
        ax = charts.distribution_chart(trials)
        drawn = sum(patch.get_height() for patch in ax.patches)
        assert drawn == pytest.approx(TRIALS * len(trials.names))

    def test_the_differential_counts_every_trial_once(self, matplotlib, trials):
        ax = charts.differential_chart(trials, "buy", "hire")
        assert sum(patch.get_height() for patch in ax.patches) == pytest.approx(TRIALS)

    def test_the_differential_names_both_sides(self, matplotlib, trials):
        ax = charts.differential_chart(trials, "buy", "hire")
        assert "buy" in ax.get_xlabel()
        assert "hire" in ax.get_xlabel()

    def test_an_unknown_alternative_is_named(self, matplotlib, trials):
        with pytest.raises(KeyError, match="nope"):
            charts.differential_chart(trials, "nope", "hire")


class TestTheTornadoChart:
    def test_a_row_per_parameter_widest_first(self, matplotlib, swings):
        ax = charts.tornado_chart(swings)
        assert [label.get_text() for label in ax.get_yticklabels()] == [
            bar.label for bar in swings.bars
        ]

    def test_every_bar_is_inside_the_limits(self, matplotlib, swings):
        """`Bar.low` is the value at the parameter's low end, which is often the
        larger number; limits taken from low and high alone clip every bar."""
        ax = charts.tornado_chart(swings)
        left, right = ax.get_xlim()
        for bar in swings.bars:
            assert left <= min(bar.low, bar.high)
            assert max(bar.low, bar.high) <= right

    def test_the_base_is_inside_the_limits(self, matplotlib, swings):
        ax = charts.tornado_chart(swings)
        left, right = ax.get_xlim()
        assert left <= swings.base <= right

    def test_each_bar_is_as_wide_as_its_swing(self, matplotlib, swings):
        ax = charts.tornado_chart(swings)
        drawn = sorted(round(patch.get_width(), 6) for patch in ax.patches)
        assert drawn == sorted(round(bar.swing, 6) for bar in swings.bars)


class TestTheSweepChart:
    def test_one_line_per_alternative_over_the_swept_values(self, matplotlib, walk):
        ax = charts.sweep_chart(walk)
        assert len(ax.get_lines()) == len(walk.names)
        assert ax.get_lines()[0].get_xdata() == pytest.approx(walk.values)

    def test_the_parameter_names_the_axis(self, matplotlib, walk):
        assert charts.sweep_chart(walk).get_xlabel() == "hire_rate"

    def test_a_switch_is_ruled_off(self, matplotlib, marked, walk):
        turn = switch_point(
            marked, "hire_rate", better="hire", worse="buy", bracket=(200.0, 4_000.0)
        )
        assert turn is not None
        ax = charts.sweep_chart(walk, switch=turn)
        ruled = [line.get_xdata()[0] for line in ax.get_lines()[len(walk.names) :]]
        assert ruled == pytest.approx([turn.value])


class TestTheScenarioChart:
    def test_a_tick_per_scenario(self, matplotlib, table):
        ax = charts.scenario_chart(table)
        assert [label.get_text() for label in ax.get_xticklabels()] == list(table.scenarios)

    def test_a_bar_per_scenario_and_alternative(self, matplotlib, table):
        ax = charts.scenario_chart(table)
        assert len(ax.patches) == len(table.scenarios) * len(table.names)


class TestTheFrames:
    def test_the_ranking_is_ordered_by_present_value(self, pandas, comparison):
        frame = frames.ranking_frame(comparison)
        assert list(frame.index) == [a.name for a in comparison.ranking()]
        assert frame["npv"].is_monotonic_decreasing

    def test_the_breakdown_sums_to_the_present_value(self, pandas, comparison):
        frame = frames.breakdown_frame(comparison)
        for name in comparison.names:
            assert frame[name].sum() == pytest.approx(float(comparison[name].npv))

    def test_rolling_up_by_role_keeps_the_total(self, pandas, comparison):
        rolled = frames.breakdown_frame(comparison, by=Role)
        for name in comparison.names:
            assert rolled[name].sum() == pytest.approx(float(comparison[name].npv))

    def test_the_flows_discount_to_the_present_value(self, pandas, comparison):
        frame = frames.flows_frame(comparison["loan"])
        assert frame["discounted"].sum() == pytest.approx(float(comparison["loan"].npv), abs=0.5)

    def test_the_flows_carry_a_date_on_an_anchored_grid(self, pandas, comparison):
        assert "date" in frames.flows_frame(comparison["cash"]).columns

    def test_only_periods_where_money_moves_are_listed(self, pandas, comparison):
        frame = frames.flows_frame(comparison["cash"])
        assert (frame["amount"].abs() > 0.005).all()

    def test_the_schedule_closes_at_zero(self, pandas, schedule):
        frame = frames.schedule_frame(schedule)
        assert frame["closing"].iloc[-1] == pytest.approx(0.0, abs=1e-6)
        assert frame["payment"].sum() == pytest.approx(
            frame["interest"].sum() + frame["principal"].sum()
        )

    def test_the_simulation_reports_every_alternative(self, pandas, trials):
        frame = frames.simulation_frame(trials)
        assert set(frame.index) == set(trials.names)
        assert frame["win_share"].sum() == pytest.approx(1.0)

    def test_the_percentiles_asked_for_are_the_columns_given(self, pandas, trials):
        frame = frames.simulation_frame(trials, levels=(10.0, 90.0))
        assert [c for c in frame.columns if c.startswith("p") and c[1:].isdigit()] == ["p10", "p90"]

    def test_the_draws_hold_one_row_per_trial(self, pandas, trials):
        frame = frames.draws_frame(trials)
        assert len(frame) == TRIALS
        assert set(trials.draws) <= set(frame.columns)
        assert set(trials.names) <= set(frame.columns)

    def test_the_sweep_names_its_own_winner(self, pandas, walk):
        frame = frames.sweep_frame(walk)
        assert list(frame["best"]) == list(walk.winners())

    def test_the_tornado_is_ordered_widest_first(self, pandas, swings):
        frame = frames.tornado_frame(swings)
        assert frame["swing"].is_monotonic_decreasing

    def test_the_scenarios_name_their_own_winner(self, pandas, table):
        frame = frames.scenario_frame(table)
        assert list(frame["best"]) == [table.winner_in(s) for s in table.scenarios]
        assert list(frame.index) == list(table.scenarios)
