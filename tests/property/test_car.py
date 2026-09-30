"""Invariants the vehicle domain must hold for any inputs."""

from __future__ import annotations

from datetime import date

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from simplyinvest import GeometricDecline, Term, Timeline
from simplyinvest.car import (
    Bivalent,
    CarOperating,
    Electricity,
    Household,
    Mileage,
    Petrol,
    Propulsion,
    Vehicle,
)
from simplyinvest.car.incentives import CirculationTaxExemption, PurchasePremium
from simplyinvest.domain import Context

money = st.floats(min_value=0.0, max_value=1e5, allow_nan=False, allow_infinity=False)
rate = st.floats(min_value=0.0, max_value=0.2, allow_nan=False, allow_infinity=False)
fraction = st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)
distance = st.floats(min_value=0.0, max_value=1e6, allow_nan=False, allow_infinity=False)
incomes = st.floats(min_value=0.0, max_value=2e5, allow_nan=False, allow_infinity=False)
whole_years = st.integers(min_value=0, max_value=25)


def a_timeline(**kwargs) -> Timeline:
    return Timeline(**({"horizon": Term.of_years(4), "rate": 0.03} | kwargs))


def a_bev(**kwargs) -> Vehicle:
    return Vehicle(
        **(
            {
                "name": "car",
                "price": 40_000.0,
                "propulsion": Propulsion.BEV,
                "energy": Petrol(consumption=0.0),
                "residual": GeometricDecline(0.15),
            }
            | kwargs
        )
    )


class TestEnergyCosts:
    @given(consumption=money, price=money, driven=distance)
    def test_cost_is_linear_in_distance(self, consumption, price, driven):
        petrol = Petrol(consumption=consumption, price=price)
        timeline = a_timeline()
        one = petrol.cost_per_period(np.full(timeline.n_periods + 1, driven), timeline)
        two = petrol.cost_per_period(np.full(timeline.n_periods + 1, 2 * driven), timeline)
        assert two == pytest.approx(2 * one)

    @given(loss=st.floats(min_value=0.0, max_value=0.95), consumption=money)
    def test_charging_losses_never_lower_the_bill(self, loss, consumption):
        battery = Electricity(consumption=consumption, price=0.30)
        metered = Electricity(consumption=consumption, price=0.30, charging_loss=loss)
        assert metered.effective_consumption >= battery.effective_consumption

    @given(share=fraction)
    def test_a_blend_lies_between_its_two_prices(self, share):
        home, public = 0.28, 0.62
        mixed = Electricity(consumption=18.0, price=home, public_price=public, home_share=share)
        assert home - 1e-12 <= mixed.blended_price <= public + 1e-12

    @given(share=fraction)
    def test_a_bivalent_lies_between_its_two_carriers(self, share):
        first = Electricity(consumption=19.0, price=0.32)
        second = Petrol(consumption=6.5, price=1.85)
        timeline = a_timeline()
        pair = Bivalent(first, second, primary_share=share).cost_per_km(timeline)
        low = np.minimum(first.cost_per_km(timeline), second.cost_per_km(timeline))
        high = np.maximum(first.cost_per_km(timeline), second.cost_per_km(timeline))
        assert np.all(pair >= low - 1e-12)
        assert np.all(pair <= high + 1e-12)

    @given(escalation=rate)
    def test_a_price_never_falls_where_escalation_is_positive(self, escalation):
        prices = Petrol(consumption=6.0, price=1.80, escalation=escalation).unit_price(a_timeline())
        assert np.all(np.diff(prices) >= -1e-12)


class TestDistanceDriven:
    @given(annual=distance, periods_per_year=st.sampled_from([1, 2, 4, 12]))
    def test_the_first_year_totals_the_annual_figure(self, annual, periods_per_year):
        timeline = a_timeline(periods_per_year=periods_per_year)
        driven = Mileage(annual_km=annual).per_period("km", timeline)
        assert driven[1 : periods_per_year + 1].sum() == pytest.approx(annual)

    @given(annual=distance, growth=rate)
    def test_distance_never_falls_where_growth_is_positive(self, annual, growth):
        driven = Mileage(annual_km=annual, growth=growth).per_period("km", a_timeline())
        assert np.all(np.diff(driven[1:]) >= -1e-9)


class TestTheResidualCurve:
    @given(held=whole_years, extra=st.integers(min_value=0, max_value=10))
    def test_holding_longer_is_never_worth_more(self, held, extra):
        car = a_bev()
        sooner = float(car.residual_value(held=Term.of_years(held)).signed)
        later = float(car.residual_value(held=Term.of_years(held + extra)).signed)
        assert later <= sooner + 1e-9


class TestTheMeansTest:
    @given(income=incomes, more=st.floats(min_value=0.0, max_value=1e5))
    def test_a_larger_income_never_earns_a_larger_grant(self, income, more):
        premium = PurchasePremium()
        poorer = float(premium.grant(Household(income, 1)))
        richer = float(premium.grant(Household(income + more, 1)))
        assert richer <= poorer

    @given(income=incomes, children=st.integers(min_value=0, max_value=6))
    def test_another_child_never_earns_a_smaller_grant(self, income, children):
        premium = PurchasePremium()
        fewer = float(premium.grant(Household(income, children)))
        more = float(premium.grant(Household(income, children + 1)))
        assert more >= fewer

    @given(draws=st.lists(incomes, min_size=1, max_size=40))
    def test_a_vector_of_incomes_agrees_with_reading_them_one_at_a_time(self, draws):
        """The batched lookup and the scalar lookup are the same schedule."""
        premium = PurchasePremium()
        together = np.asarray(premium.grant(Household(np.asarray(draws), 1)))
        apart = [float(premium.grant(Household(income, 1))) for income in draws]
        assert together == pytest.approx(apart)

    @given(draws=st.lists(incomes, min_size=1, max_size=20))
    def test_the_paid_draws_are_exactly_the_eligible_ones(self, draws):
        car = a_bev()
        ctx = Context(timeline=a_timeline(), party=Household(np.asarray(draws), 1))
        premium = PurchasePremium()
        entitled = np.asarray(premium.grant(ctx.party))
        paid = premium.bind(car).flows(ctx).amounts(ctx.timeline)
        if entitled.max() == 0.0:
            assert paid.size == 0 or np.all(paid == 0.0)
        else:
            assert paid[..., 4] == pytest.approx(entitled)


class TestTheExemption:
    @given(age=whole_years, extra=st.integers(min_value=0, max_value=15))
    def test_an_older_vehicle_never_carries_more_exemption(self, age, extra):
        exemption = CirculationTaxExemption(expires=None)
        younger = exemption.remaining(a_bev(age_at_acquisition=Term.of_years(age)))
        older = exemption.remaining(a_bev(age_at_acquisition=Term.of_years(age + extra)))
        assert older <= younger

    @given(tax=money, escalation=rate)
    @settings(max_examples=40)
    def test_a_covered_horizon_is_taxed_nothing_on_balance(self, tax, escalation):
        car = a_bev(circulation_tax=tax, first_registration=date(2026, 1, 1))
        ctx = Context(
            timeline=a_timeline(escalations={"running_cost": escalation}),
            usage=Mileage(annual_km=0.0),
        )
        series = CarOperating(car).flows(ctx) + CirculationTaxExemption().bind(car).flows(ctx)
        assert float(series.pv(ctx.timeline)) == pytest.approx(0.0, abs=1e-6)

    @given(months=st.integers(min_value=0, max_value=240))
    def test_the_remainder_never_exceeds_the_statutory_span(self, months):
        exemption = CirculationTaxExemption(max_term=Term.of_months(months), expires=None)
        assert exemption.remaining(a_bev()) <= Term.of_months(months)
