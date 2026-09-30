"""The extension contract, exercised by two invented domains."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
import pytest

from simplyinvest import Amount, Term, Timeline, appraise
from simplyinvest.appraisal import Alternative
from simplyinvest.cashflow import (
    CashFlowSeries,
    Component,
    Explicit,
    Frequency,
    OneOff,
    Recurring,
    Role,
)
from simplyinvest.domain import (
    Anonymous,
    Asset,
    ConstantAnnualUsage,
    Context,
    FirstYearDropThenGeometric,
    GeometricDecline,
    Party,
    ProfileUsage,
    TabulatedResiduals,
    require,
    terminal_value,
)
from simplyinvest.errors import PartyFactsMissingError, UnknownQuantityError

# --------------------------------------------------------------- a made-up domain


@dataclass(frozen=True)
class Van:
    """A domain asset."""

    name: str
    price: float
    running_cost_per_km: float = 0.25
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
            GeometricDecline(0.20).value_after(
                self.price, held=held, age_at_acquisition=self.age_at_acquisition
            )
        )


@dataclass(frozen=True)
class VanOperating:
    """A flow source measured in kilometres."""

    van: Van
    supports_batch: bool = True

    def flows(self, ctx: Context) -> CashFlowSeries:
        km = ctx.usage.per_period("km", ctx.timeline)
        return CashFlowSeries.of(
            OneOff(self.van.capital_cost, at=ctx.start, label=Component(Role.CAPITAL)),
            Explicit(
                -km * self.van.running_cost_per_km,
                label=Component(Role.OPERATING, "running"),
                description="Distance-based running cost",
            ),
            terminal_value(self.van, held=ctx.held, at=ctx.last),
        )

    def constraints(self, ctx: Context) -> tuple[str, ...]:
        return ()


@dataclass(frozen=True)
class SolarArray:
    """A second domain asset, measured in kilowatt-hours."""

    name: str
    price: float
    tariff: float = 0.08
    age_at_acquisition: Term = Term.ZERO
    economic_life: Term | None = None

    @property
    def capital_cost(self) -> Amount:
        return Amount.paid(self.price)

    @property
    def setup_cost(self) -> Amount:
        return Amount.zero()

    def residual_value(self, *, held: Term) -> Amount:
        return Amount.zero()


@dataclass(frozen=True)
class SolarOperating:
    """A flow source measured in kilowatt-hours."""

    array: SolarArray
    supports_batch: bool = True

    def flows(self, ctx: Context) -> CashFlowSeries:
        kwh = ctx.usage.per_period("kwh_exported", ctx.timeline)
        return CashFlowSeries.of(
            OneOff(self.array.capital_cost, at=ctx.start, label=Component(Role.CAPITAL)),
            Explicit(
                kwh * self.array.tariff,
                label=Component(Role.REVENUE, "feed_in"),
                description="Statutory feed-in payments",
            ),
        )

    def constraints(self, ctx: Context) -> tuple[str, ...]:
        return ("The tariff is fixed for twenty years from commissioning.",)


# --------------------------------------------------------------------- the tests


class TestTheContractAdmitsAnyDomain:
    def test_a_domain_asset_satisfies_the_protocol_structurally(self):
        assert isinstance(Van("Kangoo", 24_000), Asset)
        assert isinstance(SolarArray("Roof", 14_000), Asset)

    def test_two_unrelated_domains_both_reach_a_present_value(self):
        grid = Timeline(Term.of_years(5), periods_per_year=1, rate=0.04)

        van = appraise(
            Alternative("Van", sources=(VanOperating(Van("Kangoo", 24_000)),)),
            grid,
            usage=ConstantAnnualUsage({"km": 20_000}),
        )
        solar = appraise(
            Alternative("Array", sources=(SolarOperating(SolarArray("Roof", 14_000)),)),
            grid,
            usage=ConstantAnnualUsage({"kwh_exported": 9_000}),
        )

        assert van.npv < 0  # a van is a cost
        assert solar.npv > -14_000  # the array earns some of itself back

    def test_the_engine_never_learns_the_unit(self):
        grid = Timeline(Term.of_years(3), periods_per_year=1, rate=0.04)
        result = appraise(
            Alternative("Van", sources=(VanOperating(Van("Kangoo", 24_000)),)),
            grid,
            usage=ConstantAnnualUsage({"km": 10_000}),
        )
        assert result.cost_per_unit("km") > 0

    def test_asking_for_a_quantity_the_user_never_supplied_is_an_error(self):
        grid = Timeline(Term.of_years(3), periods_per_year=1)
        with pytest.raises(UnknownQuantityError, match="km"):
            appraise(
                Alternative("Van", sources=(VanOperating(Van("Kangoo", 24_000)),)),
                grid,
                usage=ConstantAnnualUsage({"miles": 10_000}),
            )

    def test_constraints_reach_the_result(self):
        grid = Timeline(Term.of_years(3), periods_per_year=1)
        result = appraise(
            Alternative("Array", sources=(SolarOperating(SolarArray("Roof", 14_000)),)),
            grid,
            usage=ConstantAnnualUsage({"kwh_exported": 9_000}),
        )
        assert "twenty years" in result.constraints[0]


class TestUsageProfiles:
    def test_usage_accrues_in_arrears(self):
        grid = Timeline(Term.of_years(2), periods_per_year=1)
        assert ConstantAnnualUsage({"km": 12_000}).per_period("km", grid)[0] == 0.0

    def test_a_year_of_usage_lands_in_a_year_of_periods(self):
        grid = Timeline(Term.of_years(1), periods_per_year=12)
        assert ConstantAnnualUsage({"km": 12_000}).per_period("km", grid).sum() == pytest.approx(
            12_000
        )

    def test_usage_may_grow(self):
        grid = Timeline(Term.of_years(2), periods_per_year=1)
        profile = ConstantAnnualUsage({"km": 10_000}, growth={"km": 0.10})
        per_period = profile.per_period("km", grid)
        assert per_period[1] == pytest.approx(10_000)
        assert per_period[2] == pytest.approx(11_000)

    def test_a_resolved_profile_is_taken_as_given(self):
        grid = Timeline(Term.of_years(2), periods_per_year=1)
        vectors = {"kwh": np.array([0.0, 900.0, 880.0])}
        profile = ProfileUsage(vectors)
        assert np.array_equal(profile.per_period("kwh", grid), vectors["kwh"])
        assert profile.annual("kwh") == pytest.approx(1_780.0)

    def test_a_profile_of_the_wrong_length_is_refused(self):
        grid = Timeline(Term.of_years(5), periods_per_year=1)
        with pytest.raises(ValueError, match="resample it"):
            ProfileUsage({"kwh": np.zeros(3)}).per_period("kwh", grid)

    def test_the_quantities_are_discoverable(self):
        assert ConstantAnnualUsage({"km": 1, "hours": 2}).quantities == frozenset({"km", "hours"})


class TestParty:
    def test_the_default_investor_knows_nothing_in_particular(self):
        assert Anonymous().name == "investor"
        assert isinstance(Anonymous(), Party)

    def test_a_domain_may_require_its_own_facts(self):
        @runtime_checkable
        class Taxpayer(Party, Protocol):
            @property
            def taxable_income(self) -> float: ...

        @dataclass(frozen=True)
        class Household(Taxpayer):
            name: str = "household"
            taxable_income: float = 0.0

        assert require(Household(taxable_income=55_000), Taxpayer).taxable_income == 55_000

    def test_missing_facts_fail_legibly_rather_than_as_an_attribute_error(self):
        @runtime_checkable
        class Taxpayer(Party, Protocol):
            @property
            def taxable_income(self) -> float: ...

        with pytest.raises(PartyFactsMissingError, match="taxable_income"):
            require(Anonymous(), Taxpayer)


class TestResidualValue:
    def test_geometric_decline(self):
        assert GeometricDecline(0.15).retained(Term.of_years(3)) == pytest.approx(0.85**3)

    def test_an_invalid_rate_is_refused(self):
        with pytest.raises(ValueError, match="fraction lost per year"):
            GeometricDecline(1.5)

    def test_a_first_year_drop_then_decline(self):
        model = FirstYearDropThenGeometric(drop=0.20, rate=0.10)
        assert model.retained(Term.ZERO) == 1.0
        assert model.retained(Term.of_years(1)) == pytest.approx(0.80)
        assert model.retained(Term.of_years(2)) == pytest.approx(0.80 * 0.90)

    def test_a_table_interpolates_between_its_rows(self):
        model = TabulatedResiduals(((0.0, 1.0), (1.0, 0.80), (3.0, 0.60)))
        assert model.retained(Term.of_years(2)) == pytest.approx(0.70)

    def test_a_table_needs_rows_to_interpolate_between(self):
        with pytest.raises(ValueError, match="at least two rows"):
            TabulatedResiduals(((0.0, 1.0),))

    def test_the_curve_is_keyed_on_age_not_on_the_horizon(self):
        """A three-year-old asset held two years is worth what a new one is after five."""
        model = GeometricDecline(0.15)
        used = model.value_after(
            model.retained(Term.of_years(3)) * 30_000,
            held=Term.of_years(2),
            age_at_acquisition=Term.of_years(3),
        )
        new = model.value_after(30_000, held=Term.of_years(5), age_at_acquisition=Term.ZERO)
        assert used == pytest.approx(new)


class TestContext:
    def test_a_window_narrows_without_changing_anything_else(self):
        grid = Timeline(Term.of_years(10), periods_per_year=1)
        ctx = Context(grid, usage=ConstantAnnualUsage({"km": 1}))
        leg = ctx.window(3, 6)
        assert leg.start == 3
        assert leg.last == 6
        assert leg.held == Term.of_years(3)
        assert leg.usage is ctx.usage

    def test_the_window_cannot_run_past_the_grid(self):
        grid = Timeline(Term.of_years(5), periods_per_year=1)
        assert Context(grid, end=99).last == 5


class TestAppraisalSurface:
    @pytest.fixture
    def result(self):
        grid = Timeline(Term.of_years(5), periods_per_year=1, rate=0.04)
        return appraise(
            Alternative(
                "Van",
                sources=(VanOperating(Van("Kangoo", 24_000)),),
                life=Term.of_years(5),
            ),
            grid,
            usage=ConstantAnnualUsage({"km": 20_000}),
        )

    def test_the_breakdown_sums_to_the_total(self, result):
        assert sum(result.breakdown().values()) == pytest.approx(result.npv, abs=1e-9)

    def test_it_reports_every_metric(self, result):
        assert result.pv_of_costs > 0
        assert result.eac > 0
        assert result.profitability_index >= 0
        assert result.irr.reason

    def test_markdown_is_readable_and_totals(self, result):
        text = result.to_markdown()
        assert "Net present value" in text
        assert "**Total**" in text
        assert "Equivalent annual cost" in text

    def test_str_is_a_one_liner(self, result):
        assert result.name in str(result)


class TestAlternativeIsDomainFree:
    def test_it_has_no_field_named_after_any_domain_object(self):
        assert {f.name for f in Alternative.__dataclass_fields__.values()} == {
            "name",
            "sources",
            "tax",
            "life",
        }

    def test_an_alternative_with_no_sources_is_worth_nothing(self):
        grid = Timeline(Term.of_years(3), periods_per_year=1)
        assert appraise(Alternative("Do nothing"), grid).npv == 0.0

    def test_sources_compose(self):
        grid = Timeline(Term.of_years(3), periods_per_year=1, rate=0.05)

        @dataclass(frozen=True)
        class Fee:
            amount: float
            supports_batch: bool = True

            def flows(self, ctx: Context) -> CashFlowSeries:
                return CashFlowSeries.of(Recurring(Amount.paid(self.amount), Frequency.ANNUAL))

            def constraints(self, ctx: Context) -> tuple[str, ...]:
                return ()

        one = appraise(Alternative("One", sources=(Fee(100),)), grid).npv
        two = appraise(Alternative("Two", sources=(Fee(100), Fee(100))), grid).npv
        assert two == pytest.approx(2 * one)
