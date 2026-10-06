"""The photovoltaic domain: the system, what it makes, and what it earns."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date

import numpy as np
import pytest

from simplyinvest import Amount, Term, Timeline, appraise
from simplyinvest.appraisal import Alternative
from simplyinvest.domain import Asset, Context, FlowSource, GeometricDecline, UsageProfile
from simplyinvest.errors import (
    AnchorlessResamplingWarning,
    PartyFactsMissingError,
    UnknownQuantityError,
)
from simplyinvest.financing import CashPurchase
from simplyinvest.pv import (
    CACHE_VARIABLE,
    EAST,
    EXPORTED,
    GENERATED,
    HOURS_IN_YEAR,
    SELF_CONSUMED,
    SOUTH,
    WEST,
    Array,
    Battery,
    ClearSkyYear,
    DeclaredShare,
    GivenWeather,
    Householder,
    Inverter,
    PvgisTmy,
    PvOperations,
    PvOperator,
    PvSupply,
    ResidentialExempt,
    SimulatedYield,
    Site,
    StatedYield,
    StaticPrice,
    System,
    cache_root,
    cached_frame,
)
from simplyinvest.pv.incentives import (
    EEG_2026_AUGUST,
    EEG_2027_DRAFT,
    FeedInTariff,
    FixedTariff,
    ZeroRatedSupply,
    blended_rate,
)

LIVE = date(2026, 9, 1)


@dataclass(frozen=True)
class _NotASystem:
    """An asset from some other domain, offered to a photovoltaic rule."""

    name: str = "something else"
    age_at_acquisition: Term = Term.ZERO
    economic_life: Term | None = None

    @property
    def capital_cost(self) -> Amount:
        return Amount.paid(1_000.0)

    @property
    def setup_cost(self) -> Amount:
        return Amount.zero()

    def residual_value(self, *, held: Term) -> Amount:
        return Amount.zero()


def a_system(**kwargs) -> System:
    defaults = {
        "name": "roof",
        "price": 14_500.0,
        "peak_kw": 10.0,
        "residual": GeometricDecline(0.06),
        "inverter": Inverter(rated_kw=9.0, replacement_cost=1_600.0),
        "commissioning": LIVE,
        "connection_cost": 500.0,
        "service_cost": 120.0,
        "insurance": 90.0,
        "metering_cost": 60.0,
    }
    return System(**(defaults | kwargs))


def a_timeline(**kwargs) -> Timeline:
    defaults = {
        "horizon": Term.of_years(25),
        "periods_per_year": 12,
        "rate": 0.03,
        "start_date": LIVE,
        "escalations": {"electricity": 0.03, "operating_cost": 0.02},
    }
    return Timeline(**defaults | kwargs)


def a_supply(annual=9_500.0, share=0.30, **kwargs) -> PvSupply:
    return PvSupply(StatedYield(annual, **kwargs), DeclaredShare(share))


def an_owner(**kwargs) -> Householder:
    defaults = {"retail_price": StaticPrice(0.34), "annual_consumption_kwh": 4_200.0}
    return Householder(**(defaults | kwargs))


def a_context(**kwargs) -> Context:
    defaults = {"timeline": a_timeline(), "party": an_owner(), "usage": a_supply()}
    return Context(**(defaults | kwargs))


class TestTheDomainContracts:
    def test_a_system_is_an_asset(self):
        assert isinstance(a_system(), Asset)

    def test_a_supply_is_a_usage_profile(self):
        assert isinstance(a_supply(), UsageProfile)

    def test_running_the_system_is_a_flow_source(self):
        assert isinstance(PvOperations(a_system()), FlowSource)

    def test_a_householder_carries_the_operator_facts(self):
        assert isinstance(an_owner(), PvOperator)

    def test_a_rule_needing_a_price_says_so_when_it_is_missing(self):
        ctx = Context(a_timeline(), usage=a_supply())
        with pytest.raises(PartyFactsMissingError, match="retail_price"):
            PvOperations(a_system()).avoided_cost(ctx)


class TestTheSystem:
    def test_the_peak_is_what_the_statute_keys_on(self):
        """Geometry lives on the generation side; the asset carries nameplate power."""
        assert float(a_system(peak_kw=10.5).peak_kw) == pytest.approx(10.5)
        assert not hasattr(a_system(), "arrays")

    @pytest.mark.parametrize("field", ["price", "service_cost", "insurance", "metering_cost"])
    def test_a_magnitude_cannot_be_negative(self, field):
        with pytest.raises(ValueError, match="cannot be negative"):
            a_system(**{field: -1.0})

    def test_storage_is_reported_only_when_it_holds_something(self):
        assert not a_system().has_storage
        assert not a_system(battery=Battery(usable_kwh=0.0)).has_storage
        assert a_system(battery=Battery(usable_kwh=8.0)).has_storage

    def test_the_residual_falls_with_age(self):
        system = a_system()
        early = float(system.residual_value(held=Term.of_years(5)).magnitude)
        late = float(system.residual_value(held=Term.of_years(20)).magnitude)
        assert late < early < float(system.price)

    def test_a_tilt_off_the_compass_is_refused(self):
        with pytest.raises(ValueError, match="tilt runs from 0 to 90"):
            Array(peak_kw=5.0, tilt=120.0)

    def test_a_site_off_the_globe_is_refused(self):
        with pytest.raises(ValueError, match="latitude runs"):
            Site(latitude=120.0, longitude=6.0)

    def test_a_battery_efficiency_above_one_is_refused(self):
        with pytest.raises(ValueError, match="fraction above zero"):
            Battery(usable_kwh=8.0, round_trip_efficiency=1.4)

    def test_one_way_efficiency_squares_to_the_round_trip(self):
        battery = Battery(usable_kwh=8.0, round_trip_efficiency=0.9)
        assert battery.one_way_efficiency**2 == pytest.approx(0.9)


class TestAStatedYield:
    def test_the_first_year_is_what_was_stated(self):
        timeline = a_timeline()
        made = StatedYield(9_500.0, degradation=0.0).per_period(timeline)
        assert made[1:13].sum() == pytest.approx(9_500.0)

    def test_nothing_is_made_before_the_first_period_closes(self):
        assert StatedYield(9_500.0).per_period(a_timeline())[0] == 0.0

    def test_summer_makes_several_times_what_winter_does(self):
        made = StatedYield(9_500.0, degradation=0.0).per_period(a_timeline())
        june, december = made[10], made[4]  # anchored to September
        assert june > 4.0 * december

    def test_output_falls_as_the_modules_age(self):
        made = StatedYield(9_500.0, degradation=0.005).per_period(a_timeline())
        assert made[-12:].sum() == pytest.approx(9_500.0 * 0.995**24, rel=1e-9)

    def test_a_flat_year_needs_no_calendar(self):
        bare = Timeline(Term.of_years(5), periods_per_year=12, rate=0.03)
        flat = (1.0,) * 12
        made = StatedYield(1_200.0, degradation=0.0, seasonality=flat).per_period(bare)
        assert made[1:13].sum() == pytest.approx(1_200.0)

    def test_a_seasonal_year_on_an_unanchored_grid_says_it_cannot_place_the_months(self):
        bare = Timeline(Term.of_years(5), periods_per_year=12, rate=0.03)
        with pytest.warns(AnchorlessResamplingWarning, match="start_date"):
            StatedYield(9_500.0).per_period(bare)

    def test_an_annual_grid_needs_no_seasons(self):
        yearly = Timeline(Term.of_years(5), periods_per_year=1, rate=0.03)
        made = StatedYield(9_500.0, degradation=0.0).per_period(yearly)
        assert made[1] == pytest.approx(9_500.0)

    def test_a_negative_yield_is_refused(self):
        with pytest.raises(ValueError, match="cannot be negative"):
            StatedYield(-1.0)

    def test_degradation_outside_a_fraction_is_refused(self):
        with pytest.raises(ValueError, match="runs from 0 to 1"):
            StatedYield(9_500.0, degradation=1.5)

    def test_a_stated_yield_has_no_hourly_detail(self):
        assert StatedYield(9_500.0).hourly(a_timeline()) is None


class TestADeclaredShare:
    def test_the_split_adds_back_to_what_was_generated(self):
        split = DeclaredShare(0.3).split(StatedYield(9_500.0), a_timeline())
        assert split.self_consumed + split.exported == pytest.approx(split.generated)

    def test_the_share_is_what_was_declared(self):
        split = DeclaredShare(0.42).split(StatedYield(9_500.0), a_timeline())
        assert float(split.share) == pytest.approx(0.42)

    def test_a_share_outside_a_fraction_is_refused(self):
        with pytest.raises(ValueError, match="between 0 and 1"):
            DeclaredShare(1.3)

    def test_a_swept_share_keeps_one_answer_per_draw(self):
        drawn = np.linspace(0.1, 0.6, 7)
        split = DeclaredShare(drawn).split(StatedYield(9_500.0), a_timeline())
        assert split.self_consumed.shape == (7, a_timeline().n_periods + 1)
        assert split.share == pytest.approx(drawn)


class TestTheSupply:
    def test_it_carries_the_three_streams(self):
        assert a_supply().quantities == {GENERATED, SELF_CONSUMED, EXPORTED}

    def test_an_unknown_stream_names_what_it_does_carry(self):
        with pytest.raises(UnknownQuantityError, match="kwh_exported"):
            a_supply().per_period("kwh_hoped_for", a_timeline())

    def test_the_annual_figures_divide_in_the_declared_share(self):
        supply = a_supply(annual=9_500.0, share=0.3)
        assert supply.annual(GENERATED) == pytest.approx(9_500.0)
        assert supply.annual(SELF_CONSUMED) == pytest.approx(2_850.0)
        assert supply.annual(EXPORTED) == pytest.approx(6_650.0)

    def test_a_batch_has_no_single_annual_figure(self):
        supply = PvSupply(StatedYield(np.array([9_000.0, 9_500.0])), DeclaredShare(0.3))
        with pytest.raises(ValueError, match="no single annual"):
            supply.annual(GENERATED)

    def test_the_streams_add_up_period_by_period(self):
        timeline, supply = a_timeline(), a_supply()
        made = supply.per_period(GENERATED, timeline)
        used = supply.per_period(SELF_CONSUMED, timeline)
        sold = supply.per_period(EXPORTED, timeline)
        assert used + sold == pytest.approx(made)


class TestRunningTheSystem:
    def test_the_avoided_bill_is_the_energy_at_the_working_price(self):
        ctx = a_context()
        avoided = PvOperations(a_system()).avoided_cost(ctx)
        used = ctx.usage.per_period(SELF_CONSUMED, ctx.timeline) * ctx.mask
        assert avoided[1] == pytest.approx(used[1] * 0.34)

    def test_every_fixed_cost_gets_its_own_line(self):
        parts = {str(flow.label) for flow in PvOperations(a_system()).flows(a_context())}
        assert {"operating/service", "operating/insurance", "operating/metering"} <= parts

    def test_a_cost_that_is_zero_is_not_drawn(self):
        system = a_system(insurance=0.0)
        parts = {str(flow.label) for flow in PvOperations(system).flows(a_context())}
        assert "operating/insurance" not in parts

    def test_the_inverter_is_replaced_inside_the_horizon(self):
        parts = {str(flow.label) for flow in PvOperations(a_system()).flows(a_context())}
        assert "capital/inverter_replacement" in parts

    def test_a_part_outlasting_the_horizon_is_never_replaced(self):
        system = a_system(inverter=Inverter(replacement_cost=1_600.0, replaced_after=None))
        parts = {str(flow.label) for flow in PvOperations(system).flows(a_context())}
        assert "capital/inverter_replacement" not in parts

    def test_the_replacement_costs_more_the_later_it_falls(self):
        ctx = a_context()
        flows = PvOperations(a_system()).flows(ctx)
        swap = next(f for f in flows if f.label.detail == "inverter_replacement")
        assert float(np.asarray(swap.amount.magnitude)) > 1_600.0

    def test_it_says_the_standing_charge_is_not_avoided(self):
        notes = " ".join(PvOperations(a_system()).constraints(a_context()))
        assert "standing charge" in notes

    def test_it_reports_the_self_consumption_it_implies(self):
        """The sentence has to carry the figure the flows were actually built on."""
        ctx = a_context()
        running = PvOperations(a_system())
        years = ctx.timeline.term_of(ctx.last - ctx.start).years
        averaged = float(np.sum(running.self_consumed(ctx))) / years
        notes = " ".join(running.constraints(ctx))
        assert f"{averaged:,.0f} kWh a year" in notes

    def test_batching_is_declared(self):
        assert PvOperations(a_system()).supports_batch


class TestTheFeedInBands:
    @pytest.mark.parametrize(
        ("peak", "expected"),
        [(5.0, 0.0770), (10.0, 0.0770), (15.0, (10 * 0.0770 + 5 * 0.0666) / 15), (40.0, 0.0692)],
    )
    def test_the_rate_is_blended_over_the_tranches(self, peak, expected):
        assert float(blended_rate(EEG_2026_AUGUST.surplus, peak)) == pytest.approx(
            expected, rel=1e-3
        )

    def test_a_bigger_system_never_earns_a_better_rate(self):
        sizes = np.linspace(1.0, 40.0, 40)
        rates = blended_rate(EEG_2026_AUGUST.surplus, sizes)
        assert np.all(np.diff(rates) <= 1e-12)

    def test_full_feed_in_pays_better_than_surplus(self):
        full = float(blended_rate(EEG_2026_AUGUST.full, 10.0))
        surplus = float(blended_rate(EEG_2026_AUGUST.surplus, 10.0))
        assert full > surplus


class TestTheFeedInTariff:
    def test_the_term_runs_to_the_end_of_the_twentieth_full_year(self):
        """Commissioned September 2026, the last paid month is December 2046.

        Energy is carried on the period it closes, so the last paid index is the
        one whose interval opens on 1 December 2046.
        """
        ctx = a_context()
        closes = FixedTariff().runs_until(ctx, LIVE)
        assert ctx.timeline.date_of(closes - 1) == date(LIVE.year + 20, 12, 1)

    @pytest.mark.parametrize("month", [8, 9, 11, 12])
    def test_the_last_paid_month_is_december_whatever_month_it_went_live(self, month):
        """The term runs from the end of the commissioning year, not from the day."""
        ctx = a_context()
        closes = FixedTariff().runs_until(ctx, date(2026, month, 1))
        assert ctx.timeline.date_of(closes - 1) == date(2046, 12, 1)

    def test_a_date_outside_every_regime_earns_nothing(self):
        """Taking a term from a regime that does not cover the date invents an
        entitlement; the seam refuses instead."""
        ctx = a_context()
        assert FixedTariff().runs_until(ctx, date(2026, 1, 1)) == ctx.start
        assert not FixedTariff().covers(date(2026, 1, 1))

    def test_a_system_with_no_commissioning_date_says_so(self):
        system = a_system(commissioning=None)
        verdict = FeedInTariff().eligibility(system, a_context())
        assert not verdict
        assert "commissioning date" in verdict.reason

    def test_a_date_no_regime_covers_says_the_table_may_need_extending(self):
        system = a_system(commissioning=date(2019, 1, 1))
        verdict = FeedInTariff().eligibility(system, a_context())
        assert not verdict
        assert "degress" in verdict.reason

    def test_an_unanchored_grid_cannot_place_the_term(self):
        bare = Timeline(Term.of_years(25), periods_per_year=1, rate=0.03)
        ctx = Context(bare, party=an_owner(), usage=a_supply(seasonality=(1.0,) * 12))
        verdict = FeedInTariff().eligibility(a_system(), ctx)
        assert not verdict
        assert "start_date" in verdict.reason

    def test_nothing_is_paid_after_the_term_ends(self):
        ctx = a_context()
        earned = FeedInTariff().flows(a_system(), ctx).amounts(ctx.timeline)
        closes = FixedTariff().runs_until(ctx, LIVE)
        assert earned[closes + 1 :] == pytest.approx(0.0)

    def test_the_later_regime_wins_where_two_overlap(self):
        value = FixedTariff(regimes=(EEG_2026_AUGUST, EEG_2027_DRAFT))
        assert value.regime_for(date(2027, 1, 15)) is EEG_2027_DRAFT

    def test_the_draft_says_it_is_not_law(self):
        tariff = FeedInTariff(value=FixedTariff(regimes=(EEG_2027_DRAFT,)))
        system = a_system(commissioning=date(2027, 3, 1))
        assert any("draft" in note for note in tariff.constraints(system, a_context()))

    def test_the_draft_term_is_far_shorter_than_twenty_years(self):
        ctx = a_context()
        value = FixedTariff(regimes=(EEG_2026_AUGUST, EEG_2027_DRAFT))
        under_draft = value.runs_until(ctx, date(2027, 3, 1))
        under_law = value.runs_until(ctx, LIVE)
        assert under_draft < under_law


class TestTheNegativePriceRule:
    def test_losing_hours_lowers_what_is_earned(self):
        ctx = a_context()
        paid = FeedInTariff().flows(a_system(), ctx).breakdown(ctx.timeline)
        docked = FeedInTariff(negative_price_share=0.1).flows(a_system(), ctx)
        parts = docked.breakdown(ctx.timeline)
        feed_in = next(k for k in paid if k.detail == "feed_in")
        assert float(parts[feed_in]) < float(paid[feed_in])

    def test_the_extension_makes_the_energy_up_but_late(self):
        ctx = a_context()
        whole = FeedInTariff().flows(a_system(), ctx).pv(ctx.timeline)
        docked = FeedInTariff(negative_price_share=0.1).flows(a_system(), ctx)
        assert float(docked.pv(ctx.timeline)) < float(whole)

    def test_without_the_extension_the_loss_is_larger(self):
        ctx = a_context()
        kept = FeedInTariff(negative_price_share=0.1).flows(a_system(), ctx)
        lost = FeedInTariff(negative_price_share=0.1, makes_up_lost_hours=False).flows(
            a_system(), ctx
        )
        assert float(lost.pv(ctx.timeline)) < float(kept.pv(ctx.timeline))

    def test_the_makeup_lands_when_the_term_ends(self):
        ctx = a_context()
        flows = FeedInTariff(negative_price_share=0.1).flows(a_system(), ctx)
        makeup = next(f for f in flows if f.label.detail == "negative_price_makeup")
        assert makeup.at == FixedTariff().runs_until(ctx, LIVE)

    def test_it_warns_the_reader_that_the_makeup_is_twenty_years_out(self):
        notes = FeedInTariff(negative_price_share=0.1).constraints(a_system(), a_context())
        assert any("twenty years later" in note for note in notes)


class TestARuleMeetingTheWrongAsset:
    """Reaching for a field with getattr and a default would read zero instead."""

    def test_the_tariff_refuses_anything_that_is_not_a_system(self):
        verdict = FeedInTariff().eligibility(_NotASystem(), a_context())
        assert not verdict
        assert "photovoltaic systems" in verdict.reason

    def test_the_tariff_pays_nothing_on_it(self):
        assert not FeedInTariff().flows(_NotASystem(), a_context())

    def test_the_relief_refuses_it_too(self):
        verdict = ZeroRatedSupply().eligibility(_NotASystem(), a_context())
        assert not verdict
        assert "photovoltaic systems" in verdict.reason


class TestTheValueAddedTaxRelief:
    def test_a_small_residential_system_is_zero_rated(self):
        assert ZeroRatedSupply().eligibility(a_system(), a_context())

    def test_a_system_over_the_threshold_is_not(self):
        big = a_system(peak_kw=32.0)
        verdict = ZeroRatedSupply().eligibility(big, a_context())
        assert not verdict
        assert "larger than" in verdict.reason

    def test_a_system_away_from_a_dwelling_is_not(self):
        verdict = ZeroRatedSupply(residential=False).eligibility(a_system(), a_context())
        assert not verdict

    def test_the_relief_leaves_a_qualifying_price_alone(self):
        system = a_system()
        priced = ZeroRatedSupply().adjust_capital_cost(system, a_context())
        assert float(np.asarray(priced.magnitude)) == pytest.approx(float(system.price))

    def test_above_the_threshold_the_price_carries_the_tax(self):
        big = a_system(peak_kw=32.0, price=40_000.0)
        priced = ZeroRatedSupply().adjust_capital_cost(big, a_context())
        assert float(np.asarray(priced.magnitude)) == pytest.approx(40_000.0 * 1.19)

    def test_it_moves_no_money_of_its_own(self):
        assert not ZeroRatedSupply().flows(a_system(), a_context())


class TestTheIncomeTaxExemption:
    def test_it_changes_nothing(self):
        ctx = a_context()
        gross = PvOperations(a_system()).flows(ctx)
        assert ResidentialExempt().adjust(gross, ctx) is gross

    def test_it_states_what_it_is_leaning_on(self):
        notes = " ".join(ResidentialExempt().constraints(a_context()))
        assert "§3 Nr. 72" in notes
        assert "30 kWp" in notes


class TestTheWholeAppraisal:
    def test_a_stated_system_resolves_end_to_end(self):
        plan = Alternative(
            "roof",
            sources=(
                CashPurchase().bind(a_system()),
                PvOperations(a_system()),
                FeedInTariff().bind(a_system()),
            ),
            tax=ResidentialExempt(),
        )
        result = appraise(plan, a_timeline(), party=an_owner(), usage=a_supply())
        assert float(result.npv) > 0.0
        assert set(result.breakdown()) >= {
            flow.label for flow in PvOperations(a_system()).flows(result.ctx)
        }

    def test_the_cost_of_a_generated_kilowatt_hour_is_plausible(self):
        plan = Alternative("roof", sources=(CashPurchase().bind(a_system()),))
        result = appraise(plan, a_timeline(), party=an_owner(), usage=a_supply())
        assert 0.03 < result.cost_per_unit(GENERATED) < 0.25

    def test_the_breakdown_sums_to_the_value(self):
        plan = Alternative(
            "roof",
            sources=(
                CashPurchase().bind(a_system()),
                PvOperations(a_system()),
                FeedInTariff().bind(a_system()),
            ),
        )
        result = appraise(plan, a_timeline(), party=an_owner(), usage=a_supply())
        assert sum(float(v) for v in result.breakdown().values()) == pytest.approx(
            float(result.npv), abs=1e-6
        )


class TestTheGridDoesNotChangeTheAnswer:
    @pytest.mark.parametrize("per_year", [1, 2, 4, 12])
    def test_a_coarser_grid_gives_the_same_generation(self, per_year):
        timeline = a_timeline(periods_per_year=per_year)
        made = StatedYield(9_500.0, degradation=0.0).per_period(timeline)
        assert made[1 : per_year + 1].sum() == pytest.approx(9_500.0)

    @pytest.mark.parametrize("per_year", [1, 2, 4, 12])
    def test_the_whole_horizon_makes_the_same_energy_whatever_the_grid(self, per_year):
        timeline = a_timeline(periods_per_year=per_year)
        made = StatedYield(9_500.0, degradation=0.005).per_period(timeline)
        assert made.sum() == pytest.approx(
            9_500.0 * sum(0.995**year for year in range(25)), rel=1e-9
        )


AACHEN = Site(50.7753, 6.0839, altitude=180, name="Aachen")


@pytest.fixture(scope="module")
def clear_sky(pvlib):
    """A cloudless year, which needs no network and never varies."""
    return ClearSkyYear(AACHEN, year=2026)


def a_simulated(weather, arrays=None, **kwargs) -> SimulatedYield:
    return SimulatedYield(
        AACHEN, arrays or (Array(peak_kw=10.0, tilt=30.0, azimuth=SOUTH),), weather, **kwargs
    )


class TestWhereWeatherComesFrom:
    def test_a_given_frame_is_handed_back(self, pvlib, clear_sky):
        frame = clear_sky.hourly()
        assert GivenWeather(frame).hourly() is frame

    def test_a_frame_missing_a_column_says_which(self, pvlib, clear_sky):
        short = clear_sky.hourly().drop(columns=["dni"])
        with pytest.raises(ValueError, match=r"missing \['dni'\]"):
            GivenWeather(short).hourly()

    def test_a_cloudless_year_is_one_row_an_hour(self, pvlib, clear_sky):
        assert len(clear_sky.hourly()) == HOURS_IN_YEAR

    def test_every_source_says_where_it_came_from(self, pvlib, clear_sky):
        assert "Aachen" in clear_sky.provenance
        assert "cloudless" in clear_sky.provenance

    def test_the_cache_root_follows_the_environment(self, monkeypatch, tmp_path):
        monkeypatch.setenv(CACHE_VARIABLE, str(tmp_path / "somewhere"))
        assert cache_root() == tmp_path / "somewhere"

    def test_a_download_is_kept_with_a_note_of_where_it_came_from(self, pvlib, tmp_path):
        """A cached figure nobody can attribute is a figure nobody can defend."""
        calls = []

        def fetch():
            calls.append(1)
            return ClearSkyYear(AACHEN).hourly(), {"attribution": "a test"}

        first = cached_frame({"service": "test"}, fetch, root=tmp_path)
        again = cached_frame({"service": "test"}, fetch, root=tmp_path)
        assert len(calls) == 1
        assert len(first) == len(again) == HOURS_IN_YEAR

        note = next((tmp_path / "weather").glob("*.json"))
        recorded = json.loads(note.read_text())
        assert recorded["rows"] == HOURS_IN_YEAR
        assert recorded["source"]["attribution"] == "a test"
        assert recorded["request"] == {"service": "test"}


class TestASimulatedYield:
    def test_a_german_roof_lands_in_the_published_band(self, pvlib):
        """A cloudless year is a ceiling, so it sits above a real one."""
        made = float(a_simulated(ClearSkyYear(AACHEN)).first_year_kwh)
        assert 1_200.0 < made / 10.0 < 1_800.0

    def test_facing_south_beats_facing_east_and_west(self, pvlib, clear_sky):
        south = a_simulated(clear_sky).first_year_kwh
        split = a_simulated(
            clear_sky,
            (Array(5.0, tilt=30.0, azimuth=EAST), Array(5.0, tilt=30.0, azimuth=WEST)),
        ).first_year_kwh
        assert float(split) < float(south)

    def test_the_monthly_shares_are_shares(self, pvlib, clear_sky):
        shares = a_simulated(clear_sky).monthly_share
        assert len(shares) == 12
        assert sum(shares) == pytest.approx(1.0)

    def test_summer_makes_several_times_what_winter_does(self, pvlib, clear_sky):
        shares = a_simulated(clear_sky).monthly_share
        assert shares[5] > 3.0 * shares[11]

    def test_the_first_year_totals_what_the_run_said(self, pvlib, clear_sky):
        made = a_simulated(clear_sky, degradation=0.0)
        assert made.per_period(a_timeline())[1:13].sum() == pytest.approx(
            float(made.first_year_kwh)
        )

    def test_output_falls_as_the_modules_age(self, pvlib, clear_sky):
        made = a_simulated(clear_sky, degradation=0.005)
        periods = made.per_period(a_timeline())
        assert periods[-12:].sum() == pytest.approx(
            float(made.first_year_kwh) * 0.995**24, rel=1e-9
        )

    def test_extra_losses_lower_the_yield_proportionally(self, pvlib, clear_sky):
        whole = float(a_simulated(clear_sky).first_year_kwh)
        lossy = float(a_simulated(clear_sky, system_losses=0.10).first_year_kwh)
        assert lossy == pytest.approx(whole * 0.90, rel=1e-9)

    def test_it_keeps_the_hours_for_whatever_needs_them(self, pvlib, clear_sky):
        assert len(a_simulated(clear_sky).hourly(a_timeline())) == HOURS_IN_YEAR

    def test_a_yield_needs_an_array(self, pvlib, clear_sky):
        with pytest.raises(ValueError, match="at least one array"):
            SimulatedYield(AACHEN, (), clear_sky)

    def test_it_runs_the_chain_only_once(self, pvlib, clear_sky):
        made = a_simulated(clear_sky)
        assert made.first_year_kwh == made.first_year_kwh
        assert made.monthly_share is not None

    def test_it_drops_into_a_supply_like_a_stated_one(self, pvlib, clear_sky):
        """Both sides of the seam produce the same object downstream."""
        supply = PvSupply(a_simulated(clear_sky), DeclaredShare(0.30))
        assert supply.quantities == {GENERATED, SELF_CONSUMED, EXPORTED}
        assert supply.annual(SELF_CONSUMED) == pytest.approx(supply.annual(GENERATED) * 0.30)


@pytest.mark.network
class TestTheDownloadedYear:
    def test_a_typical_year_is_one_row_an_hour(self, pvlib, tmp_path):
        assert len(PvgisTmy(AACHEN, root=tmp_path).hourly()) == HOURS_IN_YEAR

    def test_a_real_year_sits_below_a_cloudless_one(self, pvlib, tmp_path):
        real = float(a_simulated(PvgisTmy(AACHEN, root=tmp_path)).first_year_kwh)
        assert 800.0 < real / 10.0 < 1_150.0
