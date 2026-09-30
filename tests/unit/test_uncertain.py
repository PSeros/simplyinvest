"""The uncertainty layer: marks, addresses, draws, simulation and sweeps."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pytest

from simplyinvest import (
    Alternative,
    Amount,
    Case,
    CashFlowSeries,
    Component,
    Context,
    Recurring,
    Role,
    Term,
    Timeline,
)
from simplyinvest.errors import NotSweepableError, ParameterError
from simplyinvest.financing import AnnuityLoan, CashPurchase, bind
from simplyinvest.money import Quantity
from simplyinvest.uncertain import (
    Constant,
    Empirical,
    LogNormal,
    Normal,
    ParameterIndex,
    Scenario,
    Triangular,
    Uncertain,
    Uniform,
    correlation_matrix,
    lens,
    one_way,
    ref,
    run_scenarios,
    simulate,
    switch_point,
    tornado,
    uncertain,
    uniforms,
)
from simplyinvest.uncertain.gaussian import norm_cdf, norm_pdf, norm_ppf, unit_interval

# --------------------------------------------------------------- the fixtures


@dataclass(frozen=True)
class Machine:
    """An asset whose price may be uncertain."""

    name: str
    capital_cost: Amount
    setup_cost: Amount = field(default_factory=Amount.zero)
    economic_life: Term | None = None
    age_at_acquisition: Term = Term.ZERO

    def residual_value(self, *, held: Term) -> Amount:
        """Two fifths of what it cost."""
        return self.capital_cost.scaled(0.4).reversed()


@dataclass(frozen=True)
class Running:
    """A recurring cost of any size."""

    cost: Quantity = 0.0
    label: Component = field(default_factory=lambda: Component(Role.OPERATING, "running"))

    def flows(self, ctx: Context) -> CashFlowSeries:
        """One escalating recurring payment."""
        return CashFlowSeries.of(
            Recurring(Amount.paid(self.cost), label=self.label, description="Running cost")
        )

    def constraints(self, ctx: Context) -> tuple[str, ...]:
        """None."""
        return ()

    @property
    def supports_batch(self) -> bool:
        """True; the payment dates never move."""
        return True


def a_timeline() -> Timeline:
    """A six-year monthly grid at 5%."""
    return Timeline(horizon=Term.of_years(6), rate=0.05)


def a_case() -> Case:
    """Buying a machine against renting a bureau, with three marked parameters."""
    price = uncertain(120_000.0, "price", LogNormal.from_mean_cv(120_000.0, 0.08))
    machine = Machine("Mill", Amount.paid(price))
    return Case(
        alternatives=(
            Alternative(
                "Buy",
                sources=(
                    bind(CashPurchase(), machine),
                    Running(uncertain(900.0, "running", Normal(900.0, 120.0))),
                ),
            ),
            Alternative(
                "Rent", sources=(Running(uncertain(2_400.0, "fee", Uniform(2_100.0, 2_900.0))),)
            ),
        ),
        timeline=a_timeline(),
    )


# ------------------------------------------------------------- the normal law


class TestTheStandardNormal:
    @pytest.mark.parametrize(
        ("probability", "expected"),
        [
            (0.5, 0.0),
            (0.9, 1.2815515655446004),
            (0.975, 1.959963984540054),
            (0.99, 2.3263478740408408),
            (0.025, -1.959963984540054),
        ],
    )
    def test_it_matches_published_quantiles(self, probability: float, expected: float) -> None:
        assert float(norm_ppf(np.asarray(probability))) == pytest.approx(expected, abs=1e-12)

    def test_it_inverts_its_own_distribution_function_into_the_far_tail(self) -> None:
        """The quantile function is refined until it undoes the distribution function."""
        probabilities = np.array([1e-15, 1e-10, 1e-4, 0.3, 0.5, 0.7, 1 - 1e-9])
        assert norm_cdf(norm_ppf(probabilities)) == pytest.approx(probabilities, rel=1e-13)

    def test_the_density_integrates_to_one(self) -> None:
        grid = np.linspace(-12.0, 12.0, 400_001)
        assert float(np.trapezoid(norm_pdf(grid), grid)) == pytest.approx(1.0, abs=1e-12)

    def test_a_certainty_is_not_a_probability(self) -> None:
        with pytest.raises(ValueError, match="strictly between 0 and 1"):
            norm_ppf(np.asarray(1.0))

    def test_a_draw_of_exactly_zero_is_pulled_inside_the_interval(self) -> None:
        assert np.isfinite(norm_ppf(unit_interval(np.array([0.0, 1.0])))).all()


# ------------------------------------------------------------ distributions


class TestDistributions:
    def test_a_constant_is_the_same_whatever_is_drawn(self) -> None:
        assert Constant(7.0).ppf(np.array([0.01, 0.5, 0.99])) == pytest.approx(7.0)

    def test_a_uniform_spreads_evenly(self) -> None:
        assert Uniform(2.0, 6.0).quantile(0.25) == pytest.approx(3.0)
        assert Uniform(2.0, 6.0).expectation == pytest.approx(4.0)

    def test_a_triangular_peaks_at_its_mode(self) -> None:
        shape = Triangular(0.0, 1.0, 4.0)
        drawn = shape.ppf(np.linspace(1e-9, 1 - 1e-9, 200_001))
        assert drawn.min() >= 0.0
        assert drawn.max() <= 4.0
        assert float(drawn.mean()) == pytest.approx(shape.expectation, rel=1e-3)

    def test_a_triangular_must_be_ordered(self) -> None:
        with pytest.raises(ValueError, match="low <= mode <= high"):
            Triangular(1.0, 0.0, 4.0)

    def test_a_normal_can_be_elicited_from_two_quantiles(self) -> None:
        shape = Normal.from_quantiles(low=80.0, high=120.0)
        assert shape.quantile(0.1) == pytest.approx(80.0)
        assert shape.quantile(0.9) == pytest.approx(120.0)

    def test_an_empirical_returns_its_own_sample_quantiles(self) -> None:
        sample = Empirical(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
        assert sample.quantile(0.5) == pytest.approx(3.0)
        assert sample.expectation == pytest.approx(3.0)


class TestLogNormalCannotBeHeldWrong:
    def test_positional_parameters_are_refused(self) -> None:
        with pytest.raises(TypeError, match="exp"):
            LogNormal(12_000, 1_000)  # type: ignore[call-arg]

    def test_the_message_names_every_way_to_build_one(self) -> None:
        with pytest.raises(TypeError, match=r"from_mean_sd.*from_mean_cv.*from_median_gsd"):
            LogNormal()  # type: ignore[call-arg]

    def test_from_mean_sd_recovers_the_mean_it_was_given(self) -> None:
        assert LogNormal.from_mean_sd(38_900.0, 2_400.0).expectation == pytest.approx(38_900.0)

    def test_from_mean_cv_recovers_the_spread_it_was_given(self) -> None:
        shape = LogNormal.from_mean_cv(38_900.0, 0.06)
        drawn = shape.ppf(np.linspace(1e-6, 1 - 1e-6, 400_001))
        assert float(drawn.std() / drawn.mean()) == pytest.approx(0.06, rel=1e-3)

    def test_from_median_gsd_puts_half_the_mass_below_the_median(self) -> None:
        assert LogNormal.from_median_gsd(100.0, 1.5).quantile(0.5) == pytest.approx(100.0)

    def test_from_quantiles_hits_both_of_them(self) -> None:
        shape = LogNormal.from_quantiles(low=90.0, high=150.0)
        assert shape.quantile(0.1) == pytest.approx(90.0)
        assert shape.quantile(0.9) == pytest.approx(150.0)

    def test_it_is_strictly_positive(self) -> None:
        drawn = LogNormal.from_mean_cv(10.0, 0.5).ppf(np.linspace(1e-12, 1 - 1e-12, 10_001))
        assert drawn.min() > 0.0

    def test_a_negative_mean_is_refused(self) -> None:
        with pytest.raises(ValueError, match="strictly positive"):
            LogNormal.from_mean_sd(-1.0, 1.0)


# -------------------------------------------------------------------- marking


class TestMarking:
    def test_a_marked_number_is_still_a_number(self) -> None:
        price = uncertain(38_900.0, "price")
        assert price == 38_900.0
        assert price * 2 == 77_800.0

    def test_it_carries_its_label_and_prior(self) -> None:
        prior = Normal(1.0, 0.1)
        marked = uncertain(1.0, "growth", prior)
        assert isinstance(marked, Uncertain)
        assert (marked.label, marked.prior) == ("growth", prior)

    def test_it_passes_through_an_amount(self) -> None:
        assert Amount.paid(uncertain(500.0, "cost")).signed == pytest.approx(-500.0)

    def test_a_parameter_needs_a_label(self) -> None:
        with pytest.raises(ParameterError, match="needs a label"):
            uncertain(1.0, "  ")


# ------------------------------------------------------------------ addressing


class TestLenses:
    def test_a_ref_path_becomes_an_address(self) -> None:
        case = a_case()
        address = lens(ref(case).timeline.rate, "discount_rate")
        assert address.get(case) == pytest.approx(0.05)
        assert address.path == "timeline.rate"

    def test_setting_leaves_the_original_untouched(self) -> None:
        case = a_case()
        moved = lens(ref(case).timeline.rate).set(case, 0.09)
        assert (moved.timeline.rate, case.timeline.rate) == (0.09, 0.05)

    def test_it_walks_through_a_tuple(self) -> None:
        case = a_case()
        address = lens(ref(case).alternatives[1].sources[0].cost)
        assert address.get(case) == pytest.approx(2_400.0)
        assert address.path == "alternatives[1].sources[0].cost"

    def test_a_mistyped_field_fails_where_it_was_typed(self) -> None:
        case = a_case()
        with pytest.raises(ParameterError, match=r"no field 'discount_rate'.*horizon"):
            ref(case).timeline.discount_rate  # noqa: B018

    def test_a_parameter_that_is_not_a_number_is_refused(self) -> None:
        case = a_case()
        with pytest.raises(NotSweepableError, match="cannot hold a draw"):
            lens(ref(case).alternatives[0].name)

    def test_the_root_itself_is_not_an_address(self) -> None:
        with pytest.raises(ParameterError, match="into the tree"):
            lens(ref(a_case()))

    def test_a_bare_object_is_not_a_path(self) -> None:
        with pytest.raises(ParameterError, match="takes a path through ref"):
            lens(a_case())


class TestTheParameterIndex:
    def test_it_finds_every_mark_in_the_tree(self) -> None:
        assert ParameterIndex.of(a_case()).labels == ("fee", "price", "running")

    def test_it_ties_together_every_place_one_parameter_lives(self) -> None:
        """Binding a plan to an asset copies its price, and both must move together."""
        case = a_case()
        addressed = ParameterIndex.of(case)["price"]
        assert len(addressed.paths) > 1

        moved = addressed.set(case, 200_000.0)
        purchase = moved.alternatives[0].sources[0]
        assert float(purchase.price.magnitude) == pytest.approx(200_000.0)
        assert float(purchase.asset.capital_cost.magnitude) == pytest.approx(200_000.0)

    def test_two_different_numbers_may_not_share_a_label(self) -> None:
        machine = Machine("Mill", Amount.paid(uncertain(120_000.0, "cost")))
        case = Case(
            alternatives=(
                Alternative("Buy", sources=(bind(CashPurchase(), machine),)),
                Alternative("Rent", sources=(Running(uncertain(2_400.0, "cost")),)),
            ),
            timeline=a_timeline(),
        )
        with pytest.raises(ParameterError, match="two different numbers"):
            ParameterIndex.of(case)

    def test_an_unknown_label_names_the_ones_that_exist(self) -> None:
        with pytest.raises(ParameterError, match=r"offers.*'fee', 'price', 'running'"):
            ParameterIndex.of(a_case())["diesel"]

    def test_a_specification_may_not_name_an_unknown_parameter(self) -> None:
        with pytest.raises(ParameterError, match="does not have"):
            ParameterIndex.of(a_case()).distributions({"diesel": Constant(1.0)})

    def test_a_parameter_without_a_distribution_is_named(self) -> None:
        machine = Machine("Mill", Amount.paid(uncertain(120_000.0, "price")))
        case = Case(
            alternatives=(
                Alternative("Buy", sources=(bind(CashPurchase(), machine),)),
                Alternative("Rent", sources=(Running(2_400.0),)),
            ),
            timeline=a_timeline(),
        )
        with pytest.raises(ParameterError, match=r"no distribution.*'price'"):
            ParameterIndex.of(case).distributions()

    def test_a_specification_overrides_a_declared_prior(self) -> None:
        override = Constant(1.0)
        found = ParameterIndex.of(a_case()).distributions({"price": override})
        assert found["price"] is override


# ------------------------------------------------------------------- sampling


class TestSampling:
    def test_uniforms_stay_inside_the_unit_interval(self) -> None:
        drawn = uniforms(np.random.default_rng(0), 5_000, ("a", "b"))
        for values in drawn.values():
            assert values.min() > 0.0
            assert values.max() < 1.0

    def test_a_seed_repeats_a_run_exactly(self) -> None:
        first = uniforms(np.random.default_rng(3), 100, ("a",))["a"]
        second = uniforms(np.random.default_rng(3), 100, ("a",))["a"]
        assert first == pytest.approx(second)

    def test_a_copula_imposes_the_correlation_asked_for(self) -> None:
        drawn = uniforms(np.random.default_rng(1), 40_000, ("a", "b"), {("a", "b"): -0.6})
        observed = float(np.corrcoef(drawn["a"], drawn["b"])[0, 1])
        assert observed == pytest.approx(-0.6, abs=0.02)

    def test_a_copula_leaves_each_marginal_uniform(self) -> None:
        """Correlation is imposed on the ranks, so every distribution keeps its shape."""
        drawn = uniforms(np.random.default_rng(2), 40_000, ("a", "b"), {("a", "b"): 0.9})
        assert float(drawn["a"].mean()) == pytest.approx(0.5, abs=0.01)
        assert float(np.percentile(drawn["b"], 25.0)) == pytest.approx(0.25, abs=0.01)

    def test_impossible_correlations_are_refused(self) -> None:
        with pytest.raises(ParameterError, match="not positive definite"):
            uniforms(
                np.random.default_rng(0),
                10,
                ("a", "b", "c"),
                {("a", "b"): 0.99, ("a", "c"): 0.99, ("b", "c"): -0.99},
            )

    def test_a_correlation_with_itself_is_refused(self) -> None:
        with pytest.raises(ParameterError, match="with itself"):
            correlation_matrix(("a",), {("a", "a"): 0.5})

    def test_a_coefficient_outside_the_unit_range_is_refused(self) -> None:
        with pytest.raises(ValueError, match="between -1 and 1"):
            correlation_matrix(("a", "b"), {("a", "b"): 1.4})

    def test_the_matrix_is_symmetric_with_a_unit_diagonal(self) -> None:
        matrix = correlation_matrix(("a", "b", "c"), {("a", "c"): 0.3})
        assert matrix == pytest.approx(matrix.T)
        assert np.diag(matrix) == pytest.approx(1.0)


# ----------------------------------------------------------------- simulation


class TestSimulation:
    def test_one_pass_and_trial_by_trial_agree(self) -> None:
        """The two paths must be the same calculation, or neither can be trusted."""
        case = a_case()
        batched = simulate(case, n=300, seed=11, mode="batch")
        looped = simulate(case, n=300, seed=11, mode="loop")
        assert batched.npv == pytest.approx(looped.npv, rel=1e-12)

    def test_a_seed_repeats_a_simulation_exactly(self) -> None:
        assert simulate(a_case(), n=200, seed=5).npv == pytest.approx(
            simulate(a_case(), n=200, seed=5).npv
        )

    def test_it_reports_one_value_per_alternative_and_trial(self) -> None:
        run = simulate(a_case(), n=250, seed=1)
        assert run.npv.shape == (2, 250)
        assert run.trials == 250
        assert run.names == ("Buy", "Rent")

    def test_constant_parameters_reproduce_the_deterministic_answer(self) -> None:
        """A simulation with no spread is the plain appraisal, run many times."""
        case = a_case()
        spec = {
            label: Constant(float(lens_.get(case)))
            for label, lens_ in ParameterIndex.of(case).items()
        }
        run = simulate(case, n=20, seed=0, spec=spec)
        deterministic = case.run()
        for name in run.names:
            assert run.expected_npv(name) == pytest.approx(float(deterministic[name].npv))

    def test_the_draws_follow_the_distributions_asked_for(self) -> None:
        run = simulate(a_case(), n=40_000, seed=4)
        assert float(run.draws["running"].mean()) == pytest.approx(900.0, rel=0.01)
        assert float(run.draws["fee"].min()) >= 2_100.0
        assert float(run.draws["fee"].max()) <= 2_900.0

    def test_the_shares_of_the_ranking_sum_to_one(self) -> None:
        assert sum(simulate(a_case(), n=500, seed=2).win_share().values()) == pytest.approx(1.0)

    def test_the_winner_has_no_regret_when_it_always_wins(self) -> None:
        run = simulate(a_case(), n=500, seed=2)
        if run.win_share()["Buy"] == 1.0:
            assert run.regret("Buy") == pytest.approx(0.0)

    def test_regret_is_never_negative(self) -> None:
        run = simulate(a_case(), n=400, seed=6)
        assert all(run.regret(name) >= 0.0 for name in run.names)

    def test_the_two_probabilities_of_a_pair_account_for_every_trial(self) -> None:
        run = simulate(a_case(), n=400, seed=6)
        assert run.probability("Buy", "Rent") + run.probability("Rent", "Buy") == pytest.approx(1.0)

    def test_the_standard_error_shrinks_with_more_trials(self) -> None:
        few = simulate(a_case(), n=200, seed=8).standard_error("Buy", "Rent")
        many = simulate(a_case(), n=5_000, seed=8).standard_error("Buy", "Rent")
        assert many < few

    def test_percentiles_come_out_in_order(self) -> None:
        spread = simulate(a_case(), n=1_000, seed=9).npv_percentiles("Buy")
        assert list(spread.values()) == sorted(spread.values())

    def test_value_at_risk_is_the_lower_tail(self) -> None:
        run = simulate(a_case(), n=2_000, seed=9)
        assert run.value_at_risk("Buy") <= run.npv_percentiles("Buy", (50.0,))[50.0]

    def test_an_unknown_alternative_names_the_ones_that_ran(self) -> None:
        with pytest.raises(KeyError, match="Buy"):
            simulate(a_case(), n=10, seed=0).values("Lease")

    def test_a_case_with_nothing_uncertain_is_refused(self) -> None:
        machine = Machine("Mill", Amount.paid(120_000.0))
        case = Case(
            alternatives=(
                Alternative("Buy", sources=(bind(CashPurchase(), machine),)),
                Alternative("Rent", sources=(Running(2_400.0),)),
            ),
            timeline=a_timeline(),
        )
        with pytest.raises(ParameterError, match="nothing in this case is marked uncertain"):
            simulate(case, n=10)

    def test_no_trials_is_refused(self) -> None:
        with pytest.raises(ValueError, match="at least one trial"):
            simulate(a_case(), n=0)

    def test_the_report_names_the_winner_and_the_spread(self) -> None:
        text = simulate(a_case(), n=500, seed=3).to_markdown()
        assert "| Alternative | Expected |" in text
        assert "Buy" in text
        assert "trials" in text


class TestWhatCannotBeBatched:
    def test_a_loan_declines_to_be_resolved_in_one_pass(self) -> None:
        """Its instalment schedule is rebuilt from the rate, so a draw changes its shape."""
        assert AnnuityLoan(rate=0.05).supports_batch is False

    def test_auto_falls_back_to_trial_by_trial(self) -> None:
        run = simulate(self._borrowing_case(), n=50, seed=1)
        assert run.batched is False

    def test_batch_refuses_and_names_the_offender(self) -> None:
        with pytest.raises(ValueError, match=r"cannot be resolved in one pass.*'Buy'"):
            simulate(self._borrowing_case(), n=50, seed=1, mode="batch")

    @staticmethod
    def _borrowing_case() -> Case:
        machine = Machine(
            "Mill", Amount.paid(uncertain(120_000.0, "price", Normal(120_000.0, 6_000.0)))
        )
        return Case(
            alternatives=(
                Alternative(
                    "Buy", sources=(bind(AnnuityLoan(rate=0.06, term=Term.of_years(5)), machine),)
                ),
                Alternative(
                    "Rent", sources=(Running(uncertain(2_400.0, "fee", Uniform(2_100.0, 2_900.0))),)
                ),
            ),
            timeline=a_timeline(),
        )


# --------------------------------------------------------------------- sweeps


class TestSweeps:
    def test_one_way_resolves_the_case_once_per_value(self) -> None:
        swept = one_way(a_case(), "price", [80_000.0, 120_000.0, 160_000.0])
        assert swept.npv.shape == (2, 3)
        assert swept.names == ("Buy", "Rent")

    def test_a_dearer_machine_is_worth_less(self) -> None:
        swept = one_way(a_case(), "price", [80_000.0, 120_000.0, 160_000.0])
        buying = swept.npv[swept.names.index("Buy")]
        assert list(buying) == sorted(buying, reverse=True)

    def test_a_parameter_of_one_alternative_leaves_the_other_alone(self) -> None:
        swept = one_way(a_case(), "price", [80_000.0, 160_000.0])
        renting = swept.npv[swept.names.index("Rent")]
        assert renting[0] == pytest.approx(renting[1])

    def test_it_reports_where_the_winner_changes(self) -> None:
        swept = one_way(a_case(), "price", [40_000.0, 400_000.0])
        assert not swept.is_decisive()
        assert swept.winners() == ("Buy", "Rent")

    def test_a_sweep_needs_values(self) -> None:
        with pytest.raises(ValueError, match="at least one value"):
            one_way(a_case(), "price", [])

    def test_an_unknown_parameter_is_refused(self) -> None:
        with pytest.raises(ParameterError, match="no parameter is labelled 'diesel'"):
            one_way(a_case(), "diesel", [1.0])


class TestTornado:
    def test_the_bars_come_widest_first(self) -> None:
        bars = tornado(a_case(), on="Buy").bars
        assert [bar.swing for bar in bars] == sorted((bar.swing for bar in bars), reverse=True)

    def test_a_parameter_of_another_alternative_does_not_move_this_one(self) -> None:
        found = {bar.label: bar.swing for bar in tornado(a_case(), on="Buy").bars}
        assert found["fee"] == pytest.approx(0.0)

    def test_the_base_is_the_deterministic_answer(self) -> None:
        case = a_case()
        assert tornado(case, on="Buy").base == pytest.approx(float(case.run()["Buy"].npv))

    def test_it_measures_the_first_alternative_by_default(self) -> None:
        assert tornado(a_case()).on == "Buy"

    def test_the_report_names_the_parameters(self) -> None:
        text = tornado(a_case(), on="Buy").to_markdown()
        assert "| Parameter | Low | High | Swing |" in text
        assert "price" in text


class TestSwitchPoint:
    def test_it_finds_the_price_at_which_the_ranking_turns(self) -> None:
        case = a_case()
        turn = switch_point(
            case, "price", better="Buy", worse="Rent", bracket=(10_000.0, 400_000.0)
        )
        assert turn is not None

        index = ParameterIndex.of(case)
        below = index.apply(case, {"price": turn.value - 1_000.0}).run()
        above = index.apply(case, {"price": turn.value + 1_000.0}).run()
        assert float(below["Buy"].npv) > float(below["Rent"].npv)
        assert float(above["Buy"].npv) < float(above["Rent"].npv)

    def test_it_says_which_alternative_leads_on_each_side(self) -> None:
        """Reading the direction off the number alone is how it gets read backwards."""
        turn = switch_point(
            a_case(), "price", better="Buy", worse="Rent", bracket=(10_000.0, 400_000.0)
        )
        assert turn is not None
        assert (turn.below, turn.above) == ("Buy", "Rent")

    def test_the_direction_does_not_depend_on_which_way_round_the_pair_is_given(self) -> None:
        case = a_case()
        one = switch_point(case, "price", better="Buy", worse="Rent", bracket=(1e4, 4e5))
        other = switch_point(case, "price", better="Rent", worse="Buy", bracket=(1e4, 4e5))
        assert one is not None
        assert other is not None
        assert (one.below, one.above) == (other.below, other.above)
        assert one.value == pytest.approx(other.value, rel=1e-9)

    def test_the_two_are_worth_the_same_at_the_switch(self) -> None:
        case = a_case()
        turn = switch_point(
            case, "price", better="Buy", worse="Rent", bracket=(10_000.0, 400_000.0)
        )
        assert turn is not None
        result = ParameterIndex.of(case).apply(case, {"price": turn.value}).run()
        assert float(result["Buy"].npv) == pytest.approx(float(result["Rent"].npv), abs=1e-6)

    def test_a_range_with_no_crossing_returns_nothing(self) -> None:
        assert (
            switch_point(
                a_case(), "price", better="Buy", worse="Rent", bracket=(10_000.0, 20_000.0)
            )
            is None
        )

    def test_an_empty_bracket_is_refused(self) -> None:
        with pytest.raises(ValueError, match="runs from low to high"):
            switch_point(a_case(), "price", better="Buy", worse="Rent", bracket=(5.0, 5.0))


# ------------------------------------------------------------------ scenarios


class TestScenarios:
    def test_each_scenario_is_resolved_on_its_own(self) -> None:
        table = run_scenarios(
            a_case(),
            [Scenario("Cheap", {"price": 90_000.0}), Scenario("Dear", {"price": 190_000.0})],
        )
        assert table.npv.shape == (2, 2)
        assert table.scenarios == ("Cheap", "Dear")

    def test_it_reports_which_alternative_wins_where(self) -> None:
        table = run_scenarios(
            a_case(),
            [Scenario("Cheap", {"price": 90_000.0}), Scenario("Dear", {"price": 400_000.0})],
        )
        assert table.winners() == {"Cheap": "Buy", "Dear": "Rent"}
        assert not table.is_robust()

    def test_a_choice_that_survives_every_scenario_is_robust(self) -> None:
        table = run_scenarios(
            a_case(),
            [Scenario("Cheap", {"price": 80_000.0}), Scenario("Dearer", {"price": 100_000.0})],
        )
        assert table.is_robust()
        assert "wins in all 2 scenarios" in table.verdict()

    def test_an_unknown_scenario_is_refused(self) -> None:
        table = run_scenarios(a_case(), [Scenario("Cheap", {"price": 90_000.0})])
        with pytest.raises(KeyError, match="Cheap"):
            table.winner_in("Dear")

    def test_a_scenario_naming_an_unknown_parameter_is_refused(self) -> None:
        with pytest.raises(ParameterError, match="no parameter is labelled 'diesel'"):
            run_scenarios(a_case(), [Scenario("Odd", {"diesel": 1.0})])

    def test_no_scenarios_is_refused(self) -> None:
        with pytest.raises(ValueError, match="at least one scenario"):
            run_scenarios(a_case(), [])

    def test_the_report_names_the_best_in_each_row(self) -> None:
        text = run_scenarios(a_case(), [Scenario("Cheap", {"price": 90_000.0})]).to_markdown()
        assert "| Scenario | Buy | Rent | Best |" in text
