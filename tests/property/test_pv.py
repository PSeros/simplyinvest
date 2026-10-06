"""Invariants the photovoltaic model holds for any system it is given."""

from __future__ import annotations

from datetime import date

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from simplyinvest import Term, Timeline
from simplyinvest.domain import Context, GeometricDecline
from simplyinvest.pv import (
    EXPORTED,
    GENERATED,
    SELF_CONSUMED,
    DeclaredShare,
    Householder,
    PvOperations,
    PvSupply,
    StatedYield,
    StaticPrice,
    System,
)
from simplyinvest.pv.incentives import EEG_2026_AUGUST, FeedInTariff, blended_rate

yields = st.floats(min_value=1.0, max_value=50_000.0)
shares = st.floats(min_value=0.0, max_value=1.0)
decay = st.floats(min_value=0.0, max_value=0.05)
sizes = st.floats(min_value=0.1, max_value=40.0)
grids = st.sampled_from([1, 2, 4, 12])
LIVE = date(2026, 9, 1)


def a_timeline(per_year: int = 12, years: int = 25) -> Timeline:
    return Timeline(
        Term.of_years(years),
        periods_per_year=per_year,
        rate=0.03,
        start_date=LIVE,
        escalations={"electricity": 0.03, "operating_cost": 0.02},
    )


def a_system(peak: float = 10.0) -> System:
    return System(
        name="roof",
        price=14_500.0,
        peak_kw=peak,
        residual=GeometricDecline(0.06),
        commissioning=LIVE,
    )


@given(annual=yields, rate=decay, per_year=grids)
def test_the_first_year_makes_what_was_stated(annual, rate, per_year):
    """However the grid is cut, year one totals the stated yield."""
    timeline = a_timeline(per_year)
    made = StatedYield(annual, degradation=rate).per_period(timeline)
    assert made[1 : per_year + 1].sum() == pytest.approx(annual, rel=1e-9)


@given(annual=yields, rate=decay, per_year=grids)
def test_output_never_rises_from_one_year_to_the_next(annual, rate, per_year):
    """Modules age in one direction."""
    timeline = a_timeline(per_year)
    made = StatedYield(annual, degradation=rate).per_period(timeline)
    yearly = made[1:].reshape(-1, per_year).sum(axis=1)
    assert np.all(np.diff(yearly) <= 1e-6)


@given(annual=yields, share=shares, per_year=grids)
def test_the_split_loses_nothing(annual, share, per_year):
    """Every kilowatt-hour is either used on site or exported."""
    timeline = a_timeline(per_year)
    divided = DeclaredShare(share).split(StatedYield(annual), timeline)
    assert divided.self_consumed + divided.exported == pytest.approx(divided.generated)
    assert np.all(divided.self_consumed >= -1e-12)
    assert np.all(divided.exported >= -1e-12)


@given(annual=yields, share=shares)
def test_a_larger_share_never_exports_more(annual, share):
    """Using more on site leaves less for the grid."""
    timeline = a_timeline()
    supply = PvSupply(StatedYield(annual), DeclaredShare(share))
    greedier = PvSupply(StatedYield(annual), DeclaredShare(min(share + 0.1, 1.0)))
    assert greedier.annual(EXPORTED) <= supply.annual(EXPORTED) + 1e-9
    assert greedier.annual(SELF_CONSUMED) >= supply.annual(SELF_CONSUMED) - 1e-9
    assert supply.per_period(GENERATED, timeline) == pytest.approx(
        greedier.per_period(GENERATED, timeline)
    )


@given(peak=sizes)
def test_the_blended_rate_stays_inside_its_bands(peak):
    """A blend of the band rates cannot escape them."""
    rates = [band.rate for band in EEG_2026_AUGUST.surplus]
    blended = float(blended_rate(EEG_2026_AUGUST.surplus, peak))
    assert min(rates) - 1e-12 <= blended <= max(rates) + 1e-12


@given(small=sizes, extra=st.floats(min_value=0.0, max_value=20.0))
def test_a_bigger_system_never_earns_a_better_rate(small, extra):
    """The tranches only ever add cheaper capacity."""
    assert (
        float(blended_rate(EEG_2026_AUGUST.surplus, small + extra))
        <= float(blended_rate(EEG_2026_AUGUST.surplus, small)) + 1e-12
    )


@given(annual=yields, share=shares, lost=st.floats(min_value=0.0, max_value=0.5))
@settings(max_examples=40)
def test_losing_hours_to_negative_prices_never_helps(annual, share, lost):
    """Being paid for less of the same energy cannot be worth more."""
    timeline = a_timeline()
    ctx = Context(
        timeline,
        party=Householder(StaticPrice(0.34), 4_200.0),
        usage=PvSupply(StatedYield(annual), DeclaredShare(share)),
    )
    system = a_system()
    whole = FeedInTariff().flows(system, ctx).pv(timeline)
    docked = FeedInTariff(negative_price_share=lost).flows(system, ctx).pv(timeline)
    assert float(docked) <= float(whole) + 1e-9


@given(annual=yields, share=shares, lost=st.floats(min_value=0.01, max_value=0.5))
@settings(max_examples=40)
def test_the_extension_recovers_some_of_the_loss_but_never_all_of_it(annual, share, lost):
    """§51a returns the energy twenty years later, so discounting takes a cut."""
    timeline = a_timeline()
    ctx = Context(
        timeline,
        party=Householder(StaticPrice(0.34), 4_200.0),
        usage=PvSupply(StatedYield(annual), DeclaredShare(share)),
    )
    system = a_system()
    whole = float(FeedInTariff().flows(system, ctx).pv(timeline))
    kept = float(FeedInTariff(negative_price_share=lost).flows(system, ctx).pv(timeline))
    bare = float(
        FeedInTariff(negative_price_share=lost, makes_up_lost_hours=False)
        .flows(system, ctx)
        .pv(timeline)
    )
    assert bare <= kept <= whole + 1e-9


@given(annual=yields, share=shares, price=st.floats(min_value=0.05, max_value=1.0))
def test_the_avoided_bill_scales_with_the_price(annual, share, price):
    """Twice the tariff avoided is twice the benefit."""
    timeline = a_timeline()
    supply = PvSupply(StatedYield(annual), DeclaredShare(share))
    running = PvOperations(a_system())
    cheap = Context(timeline, Householder(StaticPrice(price), 4_200.0), supply)
    dear = Context(timeline, Householder(StaticPrice(2 * price), 4_200.0), supply)
    once = running.avoided_cost(cheap)
    twice = running.avoided_cost(dear)
    assert twice == pytest.approx(2.0 * once, rel=1e-9)


@given(annual=yields, shares_drawn=st.lists(shares, min_size=2, max_size=6))
def test_a_batch_of_shares_agrees_with_resolving_them_one_at_a_time(annual, shares_drawn):
    """Trial axes must not change the answer for any single trial."""
    timeline = a_timeline()
    drawn = np.asarray(shares_drawn, dtype=np.float64)
    batched = DeclaredShare(drawn).split(StatedYield(annual), timeline)
    for index, one in enumerate(shares_drawn):
        alone = DeclaredShare(one).split(StatedYield(annual), timeline)
        assert batched.self_consumed[index] == pytest.approx(alone.self_consumed)
        assert batched.exported[index] == pytest.approx(alone.exported)
