"""The vehicle domain: energy, use, running costs, leases and incentives."""

from __future__ import annotations

import warnings
from datetime import date

import numpy as np
import pytest

from simplyinvest import (
    Alternative,
    ConstantAnnualUsage,
    GeometricDecline,
    Term,
    Timeline,
    compare,
)
from simplyinvest.appraisal import Case
from simplyinvest.car import (
    DIESEL,
    ELECTRICITY,
    HYDROGEN,
    LPG,
    LPG_CARRIER,
    PETROL,
    Bivalent,
    CarOperating,
    ChargingTariff,
    Diesel,
    Electricity,
    EnergyPrice,
    Household,
    Hydrogen,
    Mileage,
    MileageLease,
    Petrol,
    Propulsion,
    PumpPrice,
    Vehicle,
    VehicleCategory,
)
from simplyinvest.car.incentives import (
    PREMIUM_2026_PHEV,
    CirculationTaxExemption,
    GhgQuota,
    PurchasePremium,
)
from simplyinvest.domain import Asset, Context, FlowSource, UsageProfile
from simplyinvest.errors import (
    ImplausibleRateWarning,
    PartyFactsMissingError,
    TermNotRepresentableError,
    UnknownQuantityError,
)
from simplyinvest.financing import CashPurchase
from simplyinvest.uncertain import Normal, simulate, uncertain


def prices(**quoted: EnergyPrice) -> tuple[EnergyPrice, ...]:
    """What the buyer pays for each carrier, overridable per test.

    Each quote carries its own carrier, so there is no key to disagree with it.
    """
    standing = {
        ELECTRICITY: Electricity.price(0.30),
        PETROL: Petrol.price(1.80),
        DIESEL: Diesel.price(1.70),
        LPG_CARRIER: LPG.price(1.00),
        HYDROGEN: Hydrogen.price(12.00),
    }
    return tuple((standing | quoted).values())


def a_timeline(**kwargs) -> Timeline:
    defaults = {"horizon": Term.of_years(4), "rate": 0.0}
    return Timeline(**(defaults | kwargs))


def a_car(**kwargs) -> Vehicle:
    defaults = {
        "name": "a car",
        "price": 30_000.0,
        "propulsion": Propulsion.ICE,
        "energy": Petrol(consumption=6.0),
        "residual": GeometricDecline(0.15),
    }
    return Vehicle(**(defaults | kwargs))


def a_buyer(**kwargs) -> Household:
    """A buyer who has quoted a price for every carrier."""
    return Household(**({"taxable_income": 40_000.0, "energy_prices": prices()} | kwargs))


def a_context(vehicle_usage: Mileage | None = None, **kwargs) -> Context:
    return Context(
        timeline=a_timeline(**kwargs),
        party=a_buyer(),
        usage=vehicle_usage or Mileage(annual_km=12_000.0),
    )


class TestTheDomainContracts:
    def test_a_vehicle_is_an_asset(self):
        assert isinstance(a_car(), Asset)

    def test_mileage_is_a_usage_profile(self):
        assert isinstance(Mileage(annual_km=10_000.0), UsageProfile)

    def test_running_a_car_is_a_flow_source(self):
        assert isinstance(CarOperating(a_car()), FlowSource)

    def test_a_vehicle_prices_its_own_acquisition(self):
        car = a_car(price=28_500.0, infrastructure_cost=900.0)
        assert float(car.capital_cost.signed) == -28_500.0
        assert float(car.setup_cost.signed) == -900.0

    def test_a_used_vehicle_says_so(self):
        assert a_car(age_at_acquisition=Term.of_years(3)).is_used
        assert not a_car().is_used

    def test_a_negative_price_is_refused(self):
        with pytest.raises(ValueError, match="price is an amount"):
            a_car(price=-1.0)

    def test_a_vehicle_cannot_predate_itself(self):
        with pytest.raises(ValueError, match="before it exists"):
            a_car(age_at_acquisition=Term.of_months(-1))

    def test_residual_value_is_keyed_on_age_not_on_the_horizon(self):
        """A three-year-old car held three years is worth a six-year-old car."""
        new = a_car()
        used = a_car(age_at_acquisition=Term.of_years(3))
        held = Term.of_years(3)
        assert float(used.residual_value(held=held).signed) == pytest.approx(
            float(new.residual_value(held=Term.of_years(6)).signed)
            / new.residual.retained(Term.of_years(3))
        )


class TestEnergy:
    def test_consumption_is_quoted_per_hundred_kilometres(self):
        petrol = Petrol(consumption=6.0)
        assert petrol.cost_per_km(prices(), a_timeline())[1] == pytest.approx(0.108)

    def test_a_real_world_factor_raises_consumption(self):
        thirsty = Petrol(consumption=6.0, real_world_factor=1.2)
        assert thirsty.effective_consumption == pytest.approx(7.2)

    def test_charging_losses_are_billed_at_the_meter(self):
        """Consumption is measured at the battery; the bill is at the meter."""
        battery = Electricity(consumption=18.0)
        metered = Electricity(consumption=18.0, charging_loss=0.10)
        assert metered.effective_consumption == pytest.approx(20.0)
        assert metered.effective_consumption > battery.effective_consumption

    def test_the_home_share_blends_the_two_prices(self):
        """Where the car is plugged in is a habit of the driver, so it is priced there."""
        mixed = Electricity.price(0.30, public=0.60, home_share=0.75)
        assert mixed.blended == pytest.approx(0.375)

    def test_charging_only_in_public_is_the_whole_public_price(self):
        public = Electricity.price(0.30, public=0.60, home_share=0.0)
        assert public.blended == pytest.approx(0.60)

    def test_lpg_carries_its_volumetric_penalty(self):
        gas = LPG(consumption=7.0, volumetric_penalty=1.20)
        assert gas.effective_consumption == pytest.approx(8.4)

    def test_a_bivalent_source_weights_its_two_carriers(self):
        electric = Electricity(consumption=20.0)
        petrol = Petrol(consumption=6.0)
        hybrid = Bivalent(electric, petrol, primary_share=0.6)
        timeline = a_timeline()
        quoted = prices()
        expected = 0.6 * electric.cost_per_km(quoted, timeline) + 0.4 * petrol.cost_per_km(
            quoted, timeline
        )
        assert hybrid.cost_per_km(prices(), timeline) == pytest.approx(expected)

    def test_a_share_of_one_is_just_the_primary(self):
        electric = Electricity(consumption=20.0)
        only = Bivalent(electric, Petrol(consumption=6.0), primary_share=1.0)
        timeline, quoted = a_timeline(), prices()
        assert only.cost_per_km(quoted, timeline) == pytest.approx(
            electric.cost_per_km(quoted, timeline)
        )

    def test_a_bivalent_source_needs_both_carriers_priced(self):
        hybrid = Bivalent(Electricity(consumption=20.0), Petrol(consumption=6.0), 0.6)
        assert hybrid.carriers == {ELECTRICITY, PETROL}
        with pytest.raises(UnknownQuantityError, match="petrol"):
            hybrid.cost_per_km((Electricity.price(0.30),), a_timeline())

    def test_prices_escalate_on_their_own_key(self):
        timeline = a_timeline(escalations={"energy": 0.10})
        quoted = Petrol.price(1.80).per_unit(timeline)
        assert quoted[12] == pytest.approx(1.80 * 1.10)
        assert quoted[24] == pytest.approx(1.80 * 1.10**2)

    def test_a_carrier_may_take_its_own_rate_instead_of_a_key(self):
        quoted = Petrol.price(1.80, escalation=0.05).per_unit(a_timeline())
        assert quoted[12] == pytest.approx(1.80 * 1.05)

    def test_cost_per_period_is_the_distance_times_the_rate(self):
        timeline = a_timeline()
        diesel = Diesel(consumption=5.5)
        distance = np.full(timeline.n_periods + 1, 1_000.0)
        assert diesel.cost_per_period(distance, prices(), timeline) == pytest.approx(
            distance * diesel.cost_per_km(prices(), timeline)
        )

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"consumption": -1.0}, "consumption cannot be negative"),
            ({"consumption": 6.0, "real_world_factor": 0.0}, "must be positive"),
        ],
    )
    def test_nonsense_is_refused(self, kwargs, message):
        with pytest.raises(ValueError, match=message):
            Petrol(**kwargs)

    def test_a_negative_price_is_refused_where_prices_now_live(self):
        with pytest.raises(ValueError, match="price cannot be negative"):
            Petrol.price(-1.0)

    def test_a_growth_factor_given_as_a_rate_is_flagged(self):
        """Five per cent a year is 0.05; 1.05 would be 105% and compound away.

        It is a warning rather than a refusal, because a very large rate is
        unusual rather than impossible.
        """
        with pytest.warns(ImplausibleRateWarning, match="0.05 rather than"):
            Petrol.price(1.79, escalation=1.05)

    def test_an_ordinary_rate_passes_quietly(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert Petrol.price(1.79, escalation=0.05)

    def test_a_rate_that_wipes_out_the_price_is_still_refused(self):
        with pytest.raises(ValueError, match="fall of 100%"):
            Petrol.price(1.79, escalation=-1.0)

    def test_a_price_has_to_say_what_it_prices(self):
        with pytest.raises(ValueError, match="which carrier"):
            PumpPrice(1.79)

    def test_quoting_a_carrier_twice_is_refused(self):
        with pytest.raises(ValueError, match="more than once"):
            Household(energy_prices=(Petrol.price(1.79), Petrol.price(1.82)))

    def test_each_carrier_builds_the_price_it_is_sold_at(self):
        """The class holding the consumption knows the shape its price takes."""
        assert isinstance(Petrol.price(1.79), PumpPrice)
        assert isinstance(Electricity.price(0.31), ChargingTariff)

    def test_the_factory_builds_what_the_constructor_would(self):
        built = Electricity.price(0.31, public=0.55, home_share=0.8)
        assert built == ChargingTariff(home=0.31, public=0.55, home_share=0.8, carrier=ELECTRICITY)
        assert Petrol.price(1.79) == PumpPrice(1.79, carrier=PETROL)
        assert built.carrier == ELECTRICITY

    def test_charging_only_at_home_needs_no_public_price(self):
        assert float(Electricity.price(0.31).blended) == pytest.approx(0.31)

    def test_a_charging_loss_of_one_would_divide_by_zero(self):
        with pytest.raises(ValueError, match="fraction below one"):
            Electricity(consumption=18.0, charging_loss=1.0)

    def test_a_share_outside_the_unit_interval_is_refused(self):
        with pytest.raises(ValueError, match="fraction of the distance"):
            Bivalent(Petrol(consumption=6.0), Diesel(consumption=5.0), primary_share=1.5)


class TestMileage:
    def test_it_supplies_kilometres_and_nothing_else(self):
        assert Mileage(annual_km=10_000.0).quantities == frozenset({"km"})

    def test_asking_for_another_quantity_says_what_it_has(self):
        with pytest.raises(UnknownQuantityError, match=r"\['km'\]"):
            Mileage(annual_km=10_000.0).per_period("kwh", a_timeline())

    def test_the_year_adds_up(self):
        timeline = a_timeline()
        driven = Mileage(annual_km=12_000.0).per_period("km", timeline)
        assert driven[1:13].sum() == pytest.approx(12_000.0)

    def test_nothing_is_driven_before_the_clock_starts(self):
        assert Mileage(annual_km=12_000.0).per_period("km", a_timeline())[0] == 0.0

    def test_distance_grows_where_told_to(self):
        timeline = a_timeline()
        driven = Mileage(annual_km=12_000.0, growth=0.10).per_period("km", timeline)
        assert driven[13:25].sum() == pytest.approx(12_000.0 * 1.10)

    def test_a_grid_of_draws_keeps_its_trial_axis(self):
        timeline = a_timeline()
        driven = Mileage(annual_km=np.array([9_000.0, 15_000.0])).per_period("km", timeline)
        assert driven.shape == (2, timeline.n_periods + 1)
        assert driven[:, 1:13].sum(axis=1) == pytest.approx([9_000.0, 15_000.0])

    def test_a_headline_figure_is_refused_for_a_vector_of_draws(self):
        """One number cannot summarise a distribution without choosing which."""
        with pytest.raises(ValueError, match="no single annual distance"):
            Mileage(annual_km=np.array([9_000.0, 15_000.0])).annual("km")

    def test_a_negative_distance_is_refused(self):
        with pytest.raises(ValueError, match="distance cannot be negative"):
            Mileage(annual_km=-1.0)


class TestRunningCosts:
    def test_energy_costs_what_the_distance_and_the_rate_say(self):
        car = a_car(energy=Petrol(consumption=6.0))
        ctx = a_context(Mileage(annual_km=12_000.0))
        yearly = CarOperating(car).energy_cost(ctx)[1:13].sum()
        assert yearly == pytest.approx(12_000.0 * 0.06 * 1.80)

    def test_each_fixed_cost_keeps_its_own_line(self):
        car = a_car(
            insurance=600.0, maintenance=400.0, circulation_tax=180.0, other_annual_cost=120.0
        )
        parts = CarOperating(car).flows(a_context()).breakdown(a_timeline())
        details = {str(label) for label in parts}
        assert details == {
            "operating/energy",
            "operating/insurance",
            "operating/maintenance",
            "operating/circulation_tax",
            "operating/other",
        }

    def test_a_cost_that_is_zero_earns_no_line(self):
        parts = CarOperating(a_car()).flows(a_context()).breakdown(a_timeline())
        assert {str(label) for label in parts} == {"operating/energy"}

    def test_fixed_costs_total_what_was_quoted(self):
        car = a_car(insurance=600.0, energy=Petrol(consumption=0.0))
        ctx = a_context()
        amounts = CarOperating(car).flows(ctx).amounts(ctx.timeline)
        assert amounts[1:13].sum() == pytest.approx(-600.0)

    def test_fixed_costs_escalate_on_their_own_key(self):
        car = a_car(insurance=600.0, energy=Petrol(consumption=0.0))
        ctx = a_context(escalations={"running_cost": 0.05})
        amounts = CarOperating(car).flows(ctx).amounts(ctx.timeline)
        assert amounts[13:25].sum() == pytest.approx(-600.0 * 1.05)

    def test_nothing_is_charged_outside_the_window(self):
        car = a_car(insurance=600.0)
        ctx = a_context().window(12, 24)
        amounts = CarOperating(car).flows(ctx).amounts(ctx.timeline)
        assert amounts[:13] == pytest.approx(0.0)
        assert amounts[25:] == pytest.approx(0.0)

    def test_it_says_so_when_the_usage_carries_no_distance(self):
        ctx = Context(
            timeline=a_timeline(), party=a_buyer(), usage=ConstantAnnualUsage({"kwh": 1.0})
        )
        with pytest.raises(UnknownQuantityError, match="km"):
            CarOperating(a_car()).flows(ctx)

    def test_it_batches(self):
        assert CarOperating(a_car()).supports_batch


class TestAMileageLease:
    def a_lease(self, **kwargs):
        defaults = {
            "rent": 300.0,
            "term": Term.of_years(2),
            "annual_included_km": 10_000.0,
            "excess_rate": 0.10,
        }
        return MileageLease(**(defaults | kwargs))

    def test_driving_over_the_allowance_is_charged(self):
        ctx = a_context(Mileage(annual_km=15_000.0))
        settled = self.a_lease().settlement(0, (0, 24), ctx)
        assert float(settled.amount.signed) == pytest.approx(-10_000.0 * 0.10)

    def test_driving_under_the_allowance_costs_nothing_by_default(self):
        ctx = a_context(Mileage(annual_km=5_000.0))
        assert self.a_lease().settlement(0, (0, 24), ctx) is None

    def test_a_refund_is_paid_where_one_was_agreed(self):
        ctx = a_context(Mileage(annual_km=8_000.0))
        lease = self.a_lease(refund_rate=0.05, refund_cap_km=5_000.0)
        settled = lease.settlement(0, (0, 24), ctx)
        assert float(settled.amount.signed) == pytest.approx(4_000.0 * 0.05)

    def test_a_refund_stops_at_its_cap(self):
        ctx = a_context(Mileage(annual_km=2_000.0))
        lease = self.a_lease(refund_rate=0.05, refund_cap_km=5_000.0)
        settled = lease.settlement(0, (0, 24), ctx)
        assert float(settled.amount.signed) == pytest.approx(5_000.0 * 0.05)

    def test_the_settlement_falls_when_the_contract_ends(self):
        ctx = a_context(Mileage(annual_km=15_000.0))
        assert self.a_lease().settlement(0, (0, 24), ctx).at == 24

    def test_a_lease_with_no_rates_settles_nothing(self):
        ctx = a_context(Mileage(annual_km=15_000.0))
        lease = MileageLease(rent=300.0, term=Term.of_years(2), annual_included_km=10_000.0)
        assert lease.settlement(0, (0, 24), ctx) is None

    def test_the_allowance_is_prorated_over_a_truncated_contract(self):
        ctx = a_context(Mileage(annual_km=10_000.0))
        assert self.a_lease().allowance((0, 6), ctx) == pytest.approx(5_000.0)

    def test_the_excess_charge_reaches_the_stream(self):
        car = a_car()
        ctx = a_context(Mileage(annual_km=15_000.0))
        parts = self.a_lease().flows(car, car.capital_cost, ctx).breakdown(ctx.timeline)
        assert any(str(label) == "financing/mileage_settlement" for label in parts)

    def test_it_says_what_the_allowance_assumes(self):
        car = a_car()
        notes = " ".join(self.a_lease().constraints(car, a_context()))
        assert "10,000 km a year" in notes

    def test_a_negative_rate_is_refused(self):
        with pytest.raises(ValueError, match="excess_rate is a magnitude"):
            self.a_lease(excess_rate=-0.10)


class TestThePurchasePremium:
    def a_new_bev(self, **kwargs):
        return a_car(propulsion=Propulsion.BEV, **kwargs)

    @pytest.mark.parametrize(
        ("income", "children", "expected"),
        [
            (30_000.0, 0, 5000.0),
            (45_000.0, 0, 5000.0),
            (45_000.01, 0, 4000.0),
            (52_000.0, 1, 4500.0),
            (70_000.0, 2, 4000.0),
            (82_000.0, 0, 0.0),
            (82_000.0, 3, 4000.0),
            (95_000.0, 2, 0.0),
        ],
    )
    def test_the_matrix_is_read_at_its_bands(self, income, children, expected):
        premium = PurchasePremium()
        assert float(premium.grant(Household(income, children))) == expected

    def test_a_household_on_a_boundary_sits_in_the_lower_band(self):
        """The bands are published as "up to 45,000", so 45,000 exactly still pays the most."""
        premium = PurchasePremium()
        assert float(premium.grant(Household(45_000.0, 0))) == 5000.0
        assert float(premium.grant(Household(45_000.02, 0))) == 4000.0

    def test_a_plug_in_reads_its_own_schedule(self):
        premium = PurchasePremium(matrix=PREMIUM_2026_PHEV, eligible_propulsion=(Propulsion.PHEV,))
        assert float(premium.grant(Household(30_000.0, 0))) == 3500.0

    def test_a_used_vehicle_is_refused(self):
        verdict = PurchasePremium().eligibility(
            self.a_new_bev(age_at_acquisition=Term.of_years(2)),
            Context(timeline=a_timeline(), party=Household(30_000.0, 0)),
        )
        assert not verdict
        assert "new vehicles only" in verdict.reason

    def test_a_quadricycle_is_refused(self):
        verdict = PurchasePremium().eligibility(
            self.a_new_bev(category=VehicleCategory.L7E),
            Context(timeline=a_timeline(), party=Household(30_000.0, 0)),
        )
        assert "L7e" in verdict.reason

    def test_a_petrol_car_is_refused(self):
        verdict = PurchasePremium().eligibility(
            a_car(), Context(timeline=a_timeline(), party=Household(30_000.0, 0))
        )
        assert "ice is not a covered drivetrain" in verdict.reason

    def test_an_exhausted_programme_pays_nobody(self):
        verdict = PurchasePremium(available=False).eligibility(
            self.a_new_bev(), Context(timeline=a_timeline(), party=Household(30_000.0, 0))
        )
        assert "no longer taking applications" in verdict.reason

    def test_a_party_without_an_income_is_named_not_guessed(self):
        with pytest.raises(PartyFactsMissingError, match="taxable_income"):
            PurchasePremium().eligibility(self.a_new_bev(), Context(timeline=a_timeline()))

    def test_the_grant_lands_after_the_processing_lag(self):
        ctx = Context(timeline=a_timeline(), party=Household(30_000.0, 0))
        series = PurchasePremium(disbursement_lag=Term.of_months(4)).flows(self.a_new_bev(), ctx)
        amounts = series.amounts(ctx.timeline)
        assert amounts[4] == pytest.approx(5000.0)
        assert amounts[0] == 0.0

    def test_a_grant_paid_past_the_horizon_never_arrives(self):
        ctx = Context(timeline=a_timeline(horizon=Term.of_months(2)), party=Household(30_000.0, 0))
        assert not PurchasePremium(disbursement_lag=Term.of_months(9)).flows(self.a_new_bev(), ctx)

    def test_it_states_the_holding_period(self):
        ctx = Context(timeline=a_timeline(), party=Household(30_000.0, 0))
        notes = " ".join(PurchasePremium().constraints(self.a_new_bev(), ctx))
        assert "3 years" in notes


class TestAMeansTestOverUncertainIncome:
    """A grant that some draws qualify for must reach exactly those draws."""

    def a_case(self, incomes):
        car = a_car(propulsion=Propulsion.BEV)
        ctx = Context(timeline=a_timeline(), party=Household(np.asarray(incomes), 0))
        return car, ctx

    def test_each_draw_gets_its_own_band(self):
        _, ctx = self.a_case([30_000.0, 70_000.0, 95_000.0])
        assert PurchasePremium().grant(ctx.party) == pytest.approx([5000.0, 3000.0, 0.0])

    def test_a_mixed_verdict_is_a_mask_not_a_refusal(self):
        car, ctx = self.a_case([30_000.0, 95_000.0])
        verdict = PurchasePremium().eligibility(car, ctx)
        assert verdict.anywhere
        assert not verdict
        assert verdict.mask == pytest.approx([1.0, 0.0])

    def test_the_grant_reaches_the_qualifying_draws_only(self):
        car, ctx = self.a_case([30_000.0, 95_000.0])
        amounts = PurchasePremium().bind(car).flows(ctx).amounts(ctx.timeline)
        assert amounts[:, 4] == pytest.approx([5000.0, 0.0])

    def test_nothing_is_paid_when_no_draw_qualifies(self):
        car, ctx = self.a_case([95_000.0, 99_000.0])
        assert not PurchasePremium().bind(car).flows(ctx)


class TestTheQuotaCredit:
    def test_a_battery_vehicle_earns_it_every_year(self):
        car = a_car(propulsion=Propulsion.BEV)
        ctx = a_context()
        amounts = GhgQuota(annual_amount=300.0).flows(car, ctx).amounts(ctx.timeline)
        assert amounts[12] == pytest.approx(300.0)
        assert amounts.sum() == pytest.approx(1200.0)

    def test_a_petrol_car_earns_nothing(self):
        verdict = GhgQuota(annual_amount=300.0).eligibility(a_car(), a_context())
        assert "earns no quota credit" in verdict.reason

    def test_an_unquoted_credit_is_not_an_entitlement(self):
        car = a_car(propulsion=Propulsion.BEV)
        assert "no credit was quoted" in GhgQuota().eligibility(car, a_context()).reason

    def test_the_credit_grows_where_told_to(self):
        car = a_car(propulsion=Propulsion.BEV)
        ctx = a_context()
        amounts = GhgQuota(annual_amount=300.0, growth=0.10).flows(car, ctx).amounts(ctx.timeline)
        assert amounts[24] == pytest.approx(300.0 * 1.10)


class TestTheCirculationTaxExemption:
    def a_bev(self, **kwargs):
        return a_car(**({"propulsion": Propulsion.BEV, "circulation_tax": 180.0} | kwargs))

    def test_a_new_vehicle_gets_the_whole_exemption(self):
        assert CirculationTaxExemption(expires=None).remaining(self.a_bev()) == Term.of_years(10)

    def test_a_used_vehicle_inherits_only_the_remainder(self):
        """The clock runs from first registration, whoever was driving."""
        car = self.a_bev(age_at_acquisition=Term.of_years(3))
        assert CirculationTaxExemption(expires=None).remaining(car) == Term.of_years(7)

    def test_an_exemption_already_run_out_is_worth_nothing(self):
        car = self.a_bev(age_at_acquisition=Term.of_years(12))
        assert CirculationTaxExemption(expires=None).remaining(car).is_zero

    def test_the_scheme_end_date_caps_the_remainder(self):
        car = self.a_bev(age_at_acquisition=Term.of_years(1), first_registration=date(2030, 1, 1))
        exemption = CirculationTaxExemption(expires=date(2034, 1, 1))
        assert exemption.remaining(car) == Term.of_years(3)

    def test_the_cap_is_counted_in_whole_months_not_in_average_days(self):
        """Four calendar years is 48 months, not 1461 days over 365.25."""
        car = self.a_bev(first_registration=date(2026, 3, 1))
        exemption = CirculationTaxExemption(expires=date(2030, 3, 1))
        assert exemption.remaining(car) == Term.of_months(48)

    def test_the_credit_matches_the_tax_it_offsets(self):
        ctx = a_context()
        amounts = CirculationTaxExemption().flows(self.a_bev(), ctx).amounts(ctx.timeline)
        assert amounts[1:13].sum() == pytest.approx(180.0)

    def test_the_credit_stops_when_the_exemption_does(self):
        ctx = a_context(horizon=Term.of_years(6))
        car = self.a_bev(age_at_acquisition=Term.of_years(8))
        amounts = CirculationTaxExemption(expires=None).flows(car, ctx).amounts(ctx.timeline)
        assert amounts[1:25].sum() == pytest.approx(2 * 180.0)
        assert amounts[25:] == pytest.approx(0.0)

    def test_the_cost_and_the_credit_both_stay_visible(self):
        """Netting them would hide what is being given up if the rule changed."""
        car = self.a_bev()
        ctx = a_context()
        parts = (
            CarOperating(car).flows(ctx) + CirculationTaxExemption().bind(car).flows(ctx)
        ).breakdown(ctx.timeline)
        labels = {str(label) for label in parts}
        assert "operating/circulation_tax" in labels
        assert "incentive/tax_exemption" in labels

    def test_a_petrol_car_is_refused(self):
        verdict = CirculationTaxExemption().eligibility(a_car(circulation_tax=180.0), a_context())
        assert "ice is not exempt" in verdict.reason

    def test_there_is_nothing_to_exempt_without_a_tax(self):
        untaxed = self.a_bev(circulation_tax=0.0)
        verdict = CirculationTaxExemption().eligibility(untaxed, a_context())
        assert "no circulation tax was quoted" in verdict.reason


class TestTheGridDoesNotChangeTheAnswer:
    """The months-versus-periods regression, on the domain that first hit it."""

    def build(self, periods_per_year):
        car = a_car(
            propulsion=Propulsion.BEV,
            circulation_tax=180.0,
            insurance=600.0,
            energy=Electricity(consumption=18.0),
            first_registration=date(2026, 1, 1),
        )
        timeline = Timeline(
            horizon=Term.of_years(6),
            periods_per_year=periods_per_year,
            rate=0.04,
            start_date=date(2026, 1, 1),
        )
        return compare(
            [
                Alternative(
                    "electric",
                    (
                        CashPurchase().bind(car),
                        CarOperating(car),
                        CirculationTaxExemption().bind(car),
                        PurchasePremium(disbursement_lag=Term.of_months(6)).bind(car),
                    ),
                ),
                Alternative("nothing", (CashPurchase().bind(a_car(price=0.0)),)),
            ],
            timeline,
            party=a_buyer(taxable_income=40_000.0, children=1),
            usage=Mileage(annual_km=12_000.0),
        )

    def test_monthly_and_quarterly_agree(self):
        monthly = float(self.build(12).ranking()[0].npv)
        quarterly = float(self.build(4).ranking()[0].npv)
        assert monthly == pytest.approx(quarterly, rel=2e-3)

    def test_a_term_that_does_not_fit_the_grid_is_refused(self):
        timeline = Timeline(horizon=Term.of_years(4), periods_per_year=4)
        with pytest.raises(TermNotRepresentableError):
            timeline.periods_in(Term.of_months(5))

    def test_a_four_month_lag_will_not_be_guessed_onto_a_quarterly_grid(self):
        """Rounding it to one quarter or two is how a plausible wrong answer starts."""
        ctx = Context(
            timeline=Timeline(horizon=Term.of_years(4), periods_per_year=4),
            party=a_buyer(taxable_income=30_000.0, children=0),
        )
        with pytest.raises(TermNotRepresentableError, match=r"4 months is 1\.33"):
            PurchasePremium(disbursement_lag=Term.of_months(4)).flows(
                a_car(propulsion=Propulsion.BEV), ctx
            )


class TestTheExemptionCancelsTheTax:
    """A full exemption must leave nothing behind, or the car is taxed after all."""

    def a_bev(self, **kwargs):
        return a_car(**({"propulsion": Propulsion.BEV, "circulation_tax": 180.0} | kwargs))

    @pytest.mark.parametrize("escalation", [0.0, 0.02, 0.05])
    def test_they_net_to_nothing_over_a_covered_horizon(self, escalation):
        car = self.a_bev(energy=Petrol(consumption=0.0))
        ctx = a_context(escalations={"running_cost": escalation})
        series = CarOperating(car).flows(ctx) + CirculationTaxExemption().bind(car).flows(ctx)
        assert float(series.pv(ctx.timeline)) == pytest.approx(0.0, abs=1e-9)

    def test_a_partial_exemption_leaves_the_uncovered_years_taxed(self):
        car = self.a_bev(age_at_acquisition=Term.of_years(8), energy=Petrol(consumption=0.0))
        ctx = a_context(horizon=Term.of_years(4))
        series = CarOperating(car).flows(ctx) + CirculationTaxExemption(expires=None).bind(
            car
        ).flows(ctx)
        assert float(series.pv(ctx.timeline)) == pytest.approx(-2 * 180.0)


class TestBatchingAgreesWithLooping:
    """One pass with vectors must give what rebuilding the tree per trial gives."""

    def a_marked_case(self):
        car = Vehicle(
            name="electric",
            price=uncertain(40_000.0, "price", Normal(40_000.0, 2_000.0)),
            propulsion=Propulsion.BEV,
            energy=Electricity(
                consumption=uncertain(17.5, "consumption", Normal(17.5, 1.0)),
                charging_loss=0.10,
            ),
            residual=GeometricDecline(0.15),
            insurance=780.0,
            circulation_tax=180.0,
            first_registration=date(2026, 1, 1),
        )
        other = a_car(price=33_000.0)
        return Case(
            alternatives=(
                Alternative(
                    "electric",
                    (
                        CashPurchase().bind(car),
                        CarOperating(car),
                        PurchasePremium(disbursement_lag=Term.of_months(3)).bind(car),
                        CirculationTaxExemption().bind(car),
                    ),
                ),
                Alternative("petrol", (CashPurchase().bind(other), CarOperating(other))),
            ),
            timeline=a_timeline(horizon=Term.of_years(6), rate=0.03),
            party=a_buyer(taxable_income=52_000.0, children=1),
            usage=Mileage(annual_km=uncertain(14_000.0, "driven", Normal(14_000.0, 2_000.0))),
        )

    def test_the_two_paths_give_the_same_answer(self):
        case = self.a_marked_case()
        batched = simulate(case, n=200, seed=7, mode="batch")
        looped = simulate(case, n=200, seed=7, mode="loop")
        assert batched.batched
        assert not looped.batched
        assert batched.npv == pytest.approx(looped.npv, rel=1e-9)

    def test_energy_pricing_keeps_the_trial_axis_in_front(self):
        """A swept price must widen the trial axis, not collide with the periods."""
        timeline = a_timeline()
        drawn = prices(electricity=Electricity.price(np.array([0.28, 0.31, 0.34])))
        source = Electricity(consumption=18.0)
        assert source.cost_per_km(drawn, timeline).shape == (3, timeline.n_periods + 1)

    def test_a_swept_price_is_read_trial_by_trial(self):
        timeline = a_timeline()
        drawn = np.array([1.70, 1.80, 1.90])
        source = Petrol(consumption=6.0)
        together = source.cost_per_km(prices(petrol=Petrol.price(drawn)), timeline)
        apart = [source.cost_per_km(prices(petrol=Petrol.price(one)), timeline) for one in drawn]
        assert together == pytest.approx(np.stack(apart))


#: The 2026 schedule as published, as (income, children) -> euros.
#: Rows: up to 45,000 | 45,001-60,000 | 60,001-80,000 | 80,001-85,000 | 85,001-90,000.
#: The ceiling of 80,000 rises by 5,000 for each of the first two children.
PUBLISHED_BEV = {
    (20_000.0, 0): 5000.0,
    (20_000.0, 1): 5500.0,
    (20_000.0, 2): 6000.0,
    (45_000.0, 0): 5000.0,
    (45_000.0, 1): 5500.0,
    (45_000.0, 2): 6000.0,
    (45_001.0, 0): 4000.0,
    (45_001.0, 1): 4500.0,
    (45_001.0, 2): 5000.0,
    (60_000.0, 0): 4000.0,
    (60_000.0, 1): 4500.0,
    (60_000.0, 2): 5000.0,
    (60_001.0, 0): 3000.0,
    (60_001.0, 1): 3500.0,
    (60_001.0, 2): 4000.0,
    (80_000.0, 0): 3000.0,
    (80_000.0, 1): 3500.0,
    (80_000.0, 2): 4000.0,
    (80_001.0, 0): 0.0,
    (80_001.0, 1): 3500.0,
    (80_001.0, 2): 4000.0,
    (85_000.0, 0): 0.0,
    (85_000.0, 1): 3500.0,
    (85_000.0, 2): 4000.0,
    (85_001.0, 0): 0.0,
    (85_001.0, 1): 0.0,
    (85_001.0, 2): 4000.0,
    (90_000.0, 0): 0.0,
    (90_000.0, 1): 0.0,
    (90_000.0, 2): 4000.0,
    (90_001.0, 0): 0.0,
    (90_001.0, 1): 0.0,
    (90_001.0, 2): 0.0,
}

#: The same schedule for plug-in hybrids and range extenders.
PUBLISHED_PHEV = {
    (20_000.0, 0): 3500.0,
    (20_000.0, 1): 4000.0,
    (20_000.0, 2): 4500.0,
    (45_001.0, 0): 2500.0,
    (45_001.0, 1): 3000.0,
    (45_001.0, 2): 3500.0,
    (60_001.0, 0): 1500.0,
    (60_001.0, 1): 2000.0,
    (60_001.0, 2): 2500.0,
    (80_001.0, 0): 0.0,
    (80_001.0, 1): 2000.0,
    (80_001.0, 2): 2500.0,
    (85_001.0, 0): 0.0,
    (85_001.0, 1): 0.0,
    (85_001.0, 2): 2500.0,
    (90_001.0, 0): 0.0,
    (90_001.0, 1): 0.0,
    (90_001.0, 2): 0.0,
}


class TestTheScheduleMatchesWhatIsPublished:
    """Every cell of the published table, including both sides of every boundary."""

    @pytest.mark.parametrize(("key", "expected"), sorted(PUBLISHED_BEV.items()))
    def test_the_battery_schedule(self, key, expected):
        income, children = key
        grant = PurchasePremium().grant(Household(income, children))
        assert float(grant) == expected

    @pytest.mark.parametrize(("key", "expected"), sorted(PUBLISHED_PHEV.items()))
    def test_the_plug_in_schedule(self, key, expected):
        income, children = key
        premium = PurchasePremium(matrix=PREMIUM_2026_PHEV, eligible_propulsion=(Propulsion.PHEV,))
        assert float(premium.grant(Household(income, children))) == expected

    def test_a_household_on_exactly_a_boundary_takes_the_higher_band(self):
        """The bands are published as "up to 45,000", so 45,000 exactly still pays 5,000."""
        assert float(PurchasePremium().grant(Household(45_000.0, 0))) == 5000.0
        assert float(PurchasePremium().grant(Household(45_000.01, 0))) == 4000.0

    def test_only_the_first_two_children_count(self):
        premium = PurchasePremium()
        for children in (2, 3, 6):
            assert float(premium.grant(Household(30_000.0, children))) == 6000.0

    def test_a_fuel_cell_vehicle_takes_the_battery_rates(self):
        car = a_car(propulsion=Propulsion.FCEV)
        ctx = Context(timeline=a_timeline(), party=Household(30_000.0, 0))
        assert PurchasePremium().eligibility(car, ctx)


class TestTheRegistrationWindow:
    """The exemption covers registrations in a fixed window, not any vehicle at all."""

    def a_bev(self, **kwargs):
        return a_car(**({"propulsion": Propulsion.BEV, "circulation_tax": 180.0} | kwargs))

    def test_a_vehicle_registered_after_the_window_is_refused(self):
        car = self.a_bev(first_registration=date(2031, 6, 1))
        verdict = CirculationTaxExemption().eligibility(car, a_context())
        assert "after the scheme closed" in verdict.reason

    def test_a_vehicle_registered_before_the_window_is_refused(self):
        car = self.a_bev(first_registration=date(2010, 1, 1))
        verdict = CirculationTaxExemption().eligibility(car, a_context())
        assert "before the scheme opened" in verdict.reason

    @pytest.mark.parametrize("when", [date(2011, 5, 18), date(2026, 4, 1), date(2030, 12, 31)])
    def test_a_vehicle_inside_the_window_qualifies(self, when):
        car = self.a_bev(first_registration=when, age_at_acquisition=Term.ZERO)
        assert CirculationTaxExemption().eligibility(car, a_context())

    def test_an_unknown_registration_date_is_not_held_against_it(self):
        assert CirculationTaxExemption().eligibility(self.a_bev(), a_context())

    def test_nothing_is_paid_to_a_vehicle_outside_the_window(self):
        car = self.a_bev(first_registration=date(2031, 6, 1))
        assert not CirculationTaxExemption().bind(car).flows(a_context())
