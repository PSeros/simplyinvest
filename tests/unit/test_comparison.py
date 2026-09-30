"""Ranking alternatives, chaining them, and the tax and incentive contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import pytest

from simplyinvest import Amount, Term, Timeline, compare
from simplyinvest.appraisal import Alternative, Case, ReplacementChain
from simplyinvest.cashflow import (
    CashFlowSeries,
    Component,
    Frequency,
    OneOff,
    Recurring,
    Role,
)
from simplyinvest.domain import Context, GeometricDecline
from simplyinvest.errors import UnequalLivesError
from simplyinvest.financing import AnnuityLoan, CashPurchase, Lease
from simplyinvest.incentives import Eligibility, Incentive
from simplyinvest.tax import (
    FlatRateIncome,
    Untaxed,
    VatRegistered,
    declining_balance,
    straight_line,
    tabulated,
)


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
            GeometricDecline(0.15).value_after(
                self.price, held=held, age_at_acquisition=self.age_at_acquisition
            )
        )


@dataclass(frozen=True)
class Fee:
    """A trivial flow source."""

    amount: float
    label: Component = field(default_factory=lambda: Component(Role.OPERATING, "fee"))
    supports_batch: bool = True

    def flows(self, ctx: Context) -> CashFlowSeries:
        return CashFlowSeries.of(
            Recurring(
                Amount.paid(self.amount),
                Frequency.ANNUAL,
                start=ctx.start,
                end=ctx.last,
                label=self.label,
            )
        )

    def constraints(self, ctx: Context) -> tuple[str, ...]:
        return ()


@pytest.fixture
def grid():
    return Timeline(Term.of_years(6), periods_per_year=1, rate=0.05)


class TestCase:
    def test_a_comparison_needs_two_alternatives(self, grid):
        with pytest.raises(ValueError, match="at least two"):
            Case((Alternative("One"),), grid)

    def test_names_must_be_distinct(self, grid):
        with pytest.raises(ValueError, match="distinct names"):
            Case((Alternative("Same"), Alternative("Same")), grid)

    def test_unequal_lives_are_refused_by_default(self, grid):
        with pytest.raises(UnequalLivesError, match="flatters the shortest"):
            Case(
                (
                    Alternative("Short", life=Term.of_years(3)),
                    Alternative("Long", life=Term.of_years(6)),
                ),
                grid,
            )

    def test_the_refusal_names_the_ways_out(self, grid):
        with pytest.raises(UnequalLivesError, match="ReplacementChain"):
            Case(
                (
                    Alternative("Short", life=Term.of_years(3)),
                    Alternative("Long", life=Term.of_years(6)),
                ),
                grid,
            )

    def test_unequal_lives_may_be_allowed_deliberately(self, grid):
        result = Case(
            (
                Alternative("Short", life=Term.of_years(3)),
                Alternative("Long", life=Term.of_years(6)),
            ),
            grid,
            allow_unequal_lives=True,
        ).run()
        assert len(result) == 2

    def test_alternatives_without_a_stated_life_do_not_clash(self, grid):
        assert len(Case((Alternative("A"), Alternative("B")), grid).run()) == 2


class TestRanking:
    @pytest.fixture
    def result(self, grid):
        return compare(
            [
                Alternative("Cheap", sources=(Fee(100),)),
                Alternative("Dear", sources=(Fee(500),)),
                Alternative("Middling", sources=(Fee(300),)),
            ],
            grid,
        )

    def test_the_least_costly_wins(self, result):
        assert result.best().name == "Cheap"

    def test_ranking_is_ordered_by_value_not_by_magnitude(self, result):
        assert [a.name for a in result.ranking()] == ["Cheap", "Middling", "Dear"]

    def test_lookup_by_name(self, result):
        assert result["Dear"].npv < result["Cheap"].npv
        with pytest.raises(KeyError, match="no alternative named"):
            result["Nonexistent"]

    def test_the_margin_is_the_gap_to_the_runner_up(self, result):
        ranked = result.ranking()
        assert result.margin() == pytest.approx(float(ranked[0].npv) - float(ranked[1].npv))

    def test_a_clear_win_is_material(self, result):
        assert result.is_material()
        assert "better than" in result.verdict()

    def test_a_near_tie_is_reported_as_one(self, grid):
        result = compare(
            [
                Alternative("A", sources=(Fee(300),)),
                Alternative("B", sources=(Fee(300.5),)),
            ],
            grid,
        )
        assert not result.is_material()
        assert "margin of error" in result.verdict()

    def test_markdown_carries_the_verdict(self, result):
        text = result.to_markdown()
        assert "Net present value" in text
        assert "better than" in text


class TestIncremental:
    def test_the_difference_is_what_the_switch_costs(self, grid):
        result = compare(
            [
                Alternative("Cheap", sources=(Fee(100),)),
                Alternative("Dear", sources=(Fee(500),)),
            ],
            grid,
        )
        step = result.incremental("Cheap", "Dear")
        assert step.npv == pytest.approx(result["Cheap"].npv - result["Dear"].npv)
        assert "Cheap over Dear" in str(step)

    def test_a_switch_that_costs_now_and_pays_later_has_a_rate(self, grid):
        result = compare(
            [
                Alternative(
                    "Efficient",
                    sources=(
                        Fee(100),
                        _Upfront(4_000),
                    ),
                ),
                Alternative("Basic", sources=(Fee(1_500),)),
            ],
            grid,
        )
        step = result.incremental("Efficient", "Basic")
        assert step.irr
        assert step.payback is not None


@dataclass(frozen=True)
class _Upfront:
    amount: float
    supports_batch: bool = True

    def flows(self, ctx: Context) -> CashFlowSeries:
        return CashFlowSeries.of(
            OneOff(Amount.paid(self.amount), at=ctx.start, label=Component(Role.CAPITAL))
        )

    def constraints(self, ctx: Context) -> tuple[str, ...]:
        return ()


class TestReplacementChain:
    def test_a_chain_covers_the_horizon(self, grid):
        chain = ReplacementChain(
            "Repeat", Alternative("Leg", sources=(Fee(100),)), Term.of_years(2)
        )
        assert len(chain.windows(Context(grid))) == 3

    def test_a_chain_has_no_life_of_its_own(self, grid):
        chain = ReplacementChain("Repeat", Alternative("Leg"), Term.of_years(2))
        assert chain.life is None

    def test_each_leg_is_built_in_its_own_window(self, grid):
        chain = ReplacementChain(
            "Repeat",
            Alternative("Leg", sources=(_Upfront(1_000),)),
            Term.of_years(2),
        )
        amounts = chain.flows(Context(grid)).amounts(grid)
        assert amounts[0] == pytest.approx(-1_000)
        assert amounts[2] == pytest.approx(-1_000)
        assert amounts[4] == pytest.approx(-1_000)

    def test_successors_may_differ_from_the_first_leg(self, grid):
        chain = ReplacementChain(
            "Repeat",
            Alternative("First", sources=(_Upfront(1_000),)),
            Term.of_years(2),
            successors=(Alternative("Later", sources=(_Upfront(1_500),)),),
        )
        amounts = chain.flows(Context(grid)).amounts(grid)
        assert amounts[0] == pytest.approx(-1_000)
        assert amounts[2] == pytest.approx(-1_500)
        assert amounts[4] == pytest.approx(-1_500)

    def test_chaining_is_flagged_as_an_assumption(self, grid):
        chain = ReplacementChain("Repeat", Alternative("Leg"), Term.of_years(2))
        assert any("assumption" in note for note in chain.constraints(Context(grid)))

    def test_a_chain_makes_a_short_alternative_comparable_with_a_long_one(self, grid):
        short = ReplacementChain(
            "Lease, renewed",
            Alternative("Lease", sources=(Lease(rent=2_000, term=Term.of_years(3)).bind(Thing()),)),
            Term.of_years(3),
        )
        long = Alternative("Buy", sources=(CashPurchase().bind(Thing()),))
        result = compare([short, long], grid)
        assert len(result) == 2
        assert result.best().name in {"Lease, renewed", "Buy"}

    def test_a_chained_successor_is_valued_at_its_own_age(self, grid):
        chain = ReplacementChain(
            "Repeat",
            Alternative("Leg", sources=(CashPurchase().bind(Thing()),)),
            Term.of_years(3),
        )
        series = chain.flows(Context(grid))
        residuals = [flow for flow in series if flow.label == Component(Role.TERMINAL, "residual")]
        assert len(residuals) == 2
        assert residuals[0].amount.magnitude == pytest.approx(residuals[1].amount.magnitude)


class TestDepreciationSchedules:
    def test_straight_line_writes_off_everything(self):
        assert sum(straight_line(Term.of_years(5))) == pytest.approx(1.0)

    def test_declining_balance_writes_off_everything_when_it_switches(self):
        assert sum(declining_balance(0.3, Term.of_years(6))) == pytest.approx(1.0)

    def test_declining_balance_front_loads_the_deduction(self):
        schedule = declining_balance(0.3, Term.of_years(6))
        assert schedule[0] > schedule[-1]

    def test_without_the_switch_it_never_finishes(self):
        schedule = declining_balance(0.3, Term.of_years(6), switch_to_straight_line=False)
        assert sum(schedule) < 1.0

    def test_a_table_is_checked_for_sanity(self):
        assert tabulated([0.5, 0.3, 0.2]) == (0.5, 0.3, 0.2)
        with pytest.raises(ValueError, match="more than was paid"):
            tabulated([0.8, 0.8])
        with pytest.raises(ValueError, match="cannot be negative"):
            tabulated([0.8, -0.1])


class TestTaxTreatments:
    def test_untaxed_changes_nothing(self, grid):
        series = CashFlowSeries.of(Recurring(Amount.paid(1_000), Frequency.ANNUAL))
        assert Untaxed().adjust(series, Context(grid)).pv(grid) == series.pv(grid)

    def test_relief_makes_a_deductible_cost_cheaper(self, grid):
        label = Component(Role.OPERATING, "fee")
        series = CashFlowSeries.of(Recurring(Amount.paid(1_000), Frequency.ANNUAL, label=label))
        relieved = FlatRateIncome(0.42, frozenset({label})).adjust(series, Context(grid))
        assert relieved.pv(grid) > series.pv(grid)
        assert relieved.pv(grid) == pytest.approx(series.pv(grid) * 0.58)

    def test_relief_only_touches_what_was_declared_deductible(self, grid):
        label = Component(Role.OPERATING, "fee")
        other = Component(Role.OPERATING, "other")
        series = CashFlowSeries.of(Recurring(Amount.paid(1_000), Frequency.ANNUAL, label=other))
        assert FlatRateIncome(0.42, frozenset({label})).adjust(series, Context(grid)).pv(
            grid
        ) == series.pv(grid)

    def test_a_treatment_only_adds_flows(self, grid):
        label = Component(Role.OPERATING, "fee")
        series = CashFlowSeries.of(Recurring(Amount.paid(1_000), Frequency.ANNUAL, label=label))
        adjusted = FlatRateIncome(0.42, frozenset({label})).adjust(series, Context(grid))
        parts = adjusted.breakdown(grid)
        assert label in parts
        assert Component(Role.TAX, "relief") in parts

    def test_input_vat_is_recovered_from_a_gross_figure(self, grid):
        label = Component(Role.CAPITAL, "purchase")
        series = CashFlowSeries.of(OneOff(Amount.paid(11_900), label=label))
        recovered = VatRegistered(0.19, frozenset({label})).adjust(series, Context(grid))
        assert recovered.pv(grid) == pytest.approx(-10_000.0, abs=1e-6)

    def test_a_partial_business_share_recovers_part(self, grid):
        label = Component(Role.CAPITAL, "purchase")
        series = CashFlowSeries.of(OneOff(Amount.paid(11_900), label=label))
        recovered = VatRegistered(0.19, frozenset({label}), recoverable_share=0.5).adjust(
            series, Context(grid)
        )
        assert recovered.pv(grid) == pytest.approx(-10_950.0, abs=1e-6)

    def test_registering_carries_an_obligation(self, grid):
        notes = VatRegistered(0.19, frozenset()).constraints(Context(grid))
        assert any("registration" in note for note in notes)


class TestIncentiveContract:
    """A domain-supplied incentive, against the core contract."""

    @dataclass(frozen=True)
    class Grant(Incentive):
        amount: float = 3_000.0
        minimum_price: float = 10_000.0
        valid_until: date | None = date(2027, 12, 31)

        @property
        def citation(self) -> str:
            return "An imaginary programme, s.1"

        def eligibility(self, asset, ctx) -> Eligibility:
            if float(asset.capital_cost.magnitude) < self.minimum_price:
                return Eligibility(False, f"the price is below {self.minimum_price:,.0f}")
            return Eligibility(True)

        def flows(self, asset, ctx) -> CashFlowSeries:
            return CashFlowSeries.of(
                OneOff(
                    Amount.received(self.amount),
                    at=ctx.timeline.offset(ctx.start, Term.of_months(4)),
                    label=Component(Role.INCENTIVE, "grant"),
                    description="Purchase grant",
                )
            )

        def constraints(self, asset, ctx) -> tuple[str, ...]:
            return ("The asset must be kept for three years or the grant is repayable.",)

    def test_an_eligible_grant_is_received(self):
        grid = Timeline(Term.of_years(5), rate=0.03)
        result = compare(
            [
                Alternative("Bare", sources=(CashPurchase().bind(Thing()),)),
                Alternative(
                    "Granted",
                    sources=(CashPurchase().bind(Thing()), self.Grant().bind(Thing())),
                ),
            ],
            grid,
        )
        assert result.best().name == "Granted"

    def test_an_ineligible_grant_pays_nothing_and_says_why(self):
        grid = Timeline(Term.of_years(5), rate=0.03)
        cheap = Thing(price=5_000)
        bound = self.Grant().bind(cheap)
        ctx = Context(grid)
        assert bound.flows(ctx).pv(grid) == 0.0
        assert "below" in bound.constraints(ctx)[0]

    def test_obligations_reach_the_result(self):
        grid = Timeline(Term.of_years(5), rate=0.03)
        result = compare(
            [
                Alternative("Bare", sources=(CashPurchase().bind(Thing()),)),
                Alternative(
                    "Granted",
                    sources=(CashPurchase().bind(Thing()), self.Grant().bind(Thing())),
                ),
            ],
            grid,
        )
        assert any("repayable" in note for note in result.constraints)

    def test_a_closed_programme_warns(self):
        grid = Timeline(Term.of_years(5), rate=0.03, start_date=date(2030, 1, 1))
        with pytest.warns(UserWarning, match="closed on"):
            self.Grant().bind(Thing()).flows(Context(grid))

    def test_an_unanchored_timeline_cannot_be_asked_about_dates(self):
        grid = Timeline(Term.of_years(5), rate=0.03)
        self.Grant().bind(Thing()).flows(Context(grid))

    def test_an_incentive_may_change_the_price_before_financing(self):

        @dataclass(frozen=True)
        class ZeroRated(Incentive):
            rate: float = 0.19

            @property
            def citation(self) -> str:
                return "An imaginary zero rating"

            def eligibility(self, asset, ctx) -> Eligibility:
                return Eligibility(True)

            def flows(self, asset, ctx) -> CashFlowSeries:
                return CashFlowSeries()

            def adjust_capital_cost(self, asset, ctx) -> Amount:
                return Amount.paid(float(asset.capital_cost.magnitude) / (1 + self.rate))

        grid = Timeline(Term.of_years(5), rate=0.03)
        ctx = Context(grid)
        net = ZeroRated().adjust_capital_cost(Thing(), ctx)
        assert float(net.magnitude) == pytest.approx(24_000 / 1.19)

        loan = AnnuityLoan(rate=0.04, term=Term.of_years(4))
        assert loan.principal(net) < loan.principal(Thing().capital_cost)
