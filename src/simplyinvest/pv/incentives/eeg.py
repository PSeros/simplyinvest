"""The feed-in tariff, keyed on the date the system went live."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import TYPE_CHECKING, Final, Protocol, runtime_checkable

import numpy as np

from simplyinvest.cashflow import CashFlow, CashFlowSeries, Component, Explicit, OneOff, Role
from simplyinvest.incentives import Eligibility, Incentive
from simplyinvest.money import Amount, Quantity
from simplyinvest.pv.supply import EXPORTED
from simplyinvest.pv.system import System
from simplyinvest.timeline import Term

if TYPE_CHECKING:
    import numpy.typing as npt

    from simplyinvest.domain import Asset, Context

__all__ = [
    "EEG_2026_AUGUST",
    "EEG_2027_DRAFT",
    "ENACTED_REGIMES",
    "Band",
    "FeedInTariff",
    "FeedInValue",
    "FixedTariff",
    "Regime",
]

_CITATION = "EEG 2023 §48, §51, §51a"

#: Last month of the year, which the entitlement always runs to the end of.
_DECEMBER = 12


@dataclass(frozen=True, slots=True)
class Band:
    """A tranche of nameplate power and what each of its kilowatt-hours earns.

    Bands are tranched, not stepped: a 15 kWp system earns the first band's
    rate on its first 10 kWp and the second band's rate on the remaining 5.

    Args:
        up_to_kw: Nameplate power this tranche runs up to.
        rate: What one kilowatt-hour from this tranche earns.
    """

    up_to_kw: float
    rate: float


@dataclass(frozen=True)
class Regime:
    """What a system commissioned in one window is entitled to.

    Args:
        name: What to call this state of the law.
        opens: First commissioning date it covers.
        closes: Last commissioning date it covers, or ``None`` if open-ended.
        surplus: Bands for a system that also consumes its own output.
        full: Bands for a system that exports everything.
        term: How long the entitlement runs, on top of the commissioning year.
        enacted: Whether this is law, as against a published draft.
    """

    name: str
    opens: date
    closes: date | None = None
    surplus: tuple[Band, ...] = ()
    full: tuple[Band, ...] = ()
    term: Term = field(default_factory=lambda: Term.of_years(20))
    enacted: bool = True

    def covers(self, when: date) -> bool:
        """Whether a system commissioned on ``when`` falls under this regime."""
        return self.opens <= when and (self.closes is None or when <= self.closes)

    def bands(self, *, full_feed_in: bool) -> tuple[Band, ...]:
        """The bands that apply to this way of selling the output."""
        return self.full if full_feed_in else self.surplus

    def __str__(self) -> str:
        return self.name if self.enacted else f"{self.name} (draft, not enacted)"


def blended_rate(bands: tuple[Band, ...], peak_kw: Quantity) -> npt.NDArray[np.float64]:
    """What one exported kilowatt-hour earns, averaged over the tranches.

    Capacity above the last band earns nothing, which is deliberate: beyond the
    bands a system sells its output rather than drawing a tariff.
    """
    limits = np.asarray([band.up_to_kw for band in bands], dtype=np.float64)
    rates = np.asarray([band.rate for band in bands], dtype=np.float64)
    floors = np.concatenate(([0.0], limits[:-1]))
    size = np.asarray(peak_kw, dtype=np.float64)[..., np.newaxis]
    within = np.clip(size, floors, limits) - floors
    earned = (within * rates).sum(axis=-1)
    total = np.asarray(peak_kw, dtype=np.float64)
    return np.asarray(
        np.divide(earned, total, out=np.zeros_like(earned), where=total > 0.0),
        dtype=np.float64,
    )


#: Commissioning between these dates, as published by the Bundesnetzagentur.
#: Surplus feed-in earns 7.70 ct/kWh on the first 10 kWp and 6.66 ct/kWh from
#: there to 40 kWp; full feed-in earns 12.22 ct/kWh on the first 10 kWp.  Rates
#: fall 1% every six months, so this regime expires and a new one replaces it.
EEG_2026_AUGUST: Final = Regime(
    name="EEG 2023, commissioned August 2026 to January 2027",
    opens=date(2026, 8, 1),
    closes=date(2027, 1, 31),
    surplus=(Band(10.0, 0.0770), Band(40.0, 0.0666)),
    full=(Band(10.0, 0.1222),),
)

#: The government draft replacing the fixed tariff for systems commissioned
#: from January 2027 with a transitional payment of 5.20 ct/kWh for 36 months,
#: after which output must be sold on the market.  Not enacted: supply it
#: deliberately, as a scenario, rather than relying on it.
EEG_2027_DRAFT: Final = Regime(
    name="EEG amendment, commissioned from January 2027",
    opens=date(2027, 1, 1),
    surplus=(Band(100.0, 0.0520),),
    full=(Band(100.0, 0.0520),),
    term=Term.of_months(36),
    enacted=False,
)

#: The regimes that are law.  A draft is passed in explicitly.
ENACTED_REGIMES: Final = (EEG_2026_AUGUST,)


@runtime_checkable
class FeedInValue(Protocol):
    """What an exported kilowatt-hour earns, and for how long.

    A statutory tariff fixes both at commissioning.  Selling on the market
    fixes neither, which is why this is a seam rather than a rate.
    """

    def per_kwh(self, system: System, live: date) -> npt.NDArray[np.float64]:
        """What one exported kilowatt-hour earns."""
        ...

    def runs_until(self, ctx: Context, live: date) -> int:
        """The last period it is paid on, clamped to the grid."""
        ...

    def covers(self, live: date) -> bool:
        """Whether this source of value applies to that commissioning date."""
        ...

    @property
    def description(self) -> str:
        """What to call this way of being paid, in a report."""
        ...


@dataclass(frozen=True)
class FixedTariff(FeedInValue):
    """The statutory tariff, fixed by the date the system went live.

    Args:
        regimes: The states of the law to look the commissioning date up in.
        full_feed_in: Whether everything is exported rather than consumed.
    """

    regimes: tuple[Regime, ...] = ENACTED_REGIMES
    full_feed_in: bool = False

    def regime_for(self, when: date) -> Regime | None:
        """The latest regime covering a system commissioned on ``when``."""
        covering = [regime for regime in self.regimes if regime.covers(when)]
        return max(covering, key=lambda regime: regime.opens) if covering else None

    def covers(self, live: date) -> bool:
        """Whether any regime on file covers that commissioning date."""
        return self.regime_for(live) is not None

    def per_kwh(self, system: System, live: date) -> npt.NDArray[np.float64]:
        """The blended band rate for this system's nameplate power."""
        regime = self.regime_for(live)
        if regime is None:
            return np.zeros(np.shape(system.peak_kw), dtype=np.float64)
        return blended_rate(regime.bands(full_feed_in=self.full_feed_in), system.peak_kw)

    def runs_until(self, ctx: Context, live: date) -> int:
        """The last period the tariff is paid on, clamped to the grid.

        The term runs from the end of the commissioning year, so a system that
        goes live in July 2026 is paid to the end of 2046.

        ``period_of`` indexes a period by the date it opens, while energy is
        carried on the period it closes, so the December index is one further on.
        """
        regime = self.regime_for(live)
        if regime is None:
            return ctx.start
        year_end = date(live.year, _DECEMBER, 31)
        december = ctx.timeline.period_of(year_end) + 1
        return min(december + ctx.timeline.periods_in(regime.term), ctx.last)

    @property
    def enacted(self) -> bool:
        """Whether every regime on file is law."""
        return all(regime.enacted for regime in self.regimes)

    @property
    def description(self) -> str:
        """What to call this way of being paid."""
        return "the statutory feed-in tariff"

    def __str__(self) -> str:
        return self.description


@dataclass(frozen=True)
class FeedInTariff(Incentive):
    """Payment for exported energy, however that payment is arrived at.

    The statutory tariff is nominal and does not escalate, so in a real-rate
    model it declines at the rate of inflation for twenty years.  That, against
    a retail price that rises, is most of the economics of a domestic system.

    Args:
        value: Where the payment comes from.  The statutory tariff by default.
        commissioned: When the system went live, if not carried by the asset.
        negative_price_share: Share of exported energy falling in quarter-hours
            of negative spot price, which earns nothing under §51.  An estimate;
            a value model built on real prices computes it instead.
        makes_up_lost_hours: Whether §51a's extension of the entitlement is
            modelled, restoring the lost energy at the end of the term.
        reference: The statute this rests on.
    """

    value: FeedInValue = field(default_factory=FixedTariff)
    commissioned: date | None = None
    negative_price_share: Quantity = 0.0
    makes_up_lost_hours: bool = True
    reference: str = _CITATION

    @property
    def citation(self) -> str:
        """The statute this rests on."""
        return self.reference

    def went_live(self, asset: Asset) -> date | None:
        """When the system was commissioned, from this rule or from the asset."""
        if self.commissioned is not None:
            return self.commissioned
        return asset.commissioning if isinstance(asset, System) else None

    def eligibility(self, asset: Asset, ctx: Context) -> Eligibility:
        """Whether a payment is due on this system at all."""
        if not isinstance(asset, System):
            return Eligibility.no("the feed-in tariff is paid on photovoltaic systems")
        live = self.went_live(asset)
        if live is None:
            return Eligibility.no(
                "the system has no commissioning date, and the tariff is fixed by it.  "
                "Give the System a `commissioning` date, or the tariff a `commissioned` one."
            )
        if not self.value.covers(live):
            return Eligibility.no(
                f"no tariff regime on file covers a system commissioned on "
                f"{live.isoformat()}; the rates degress every six months, so the table "
                f"may need extending."
            )
        if not ctx.timeline.anchored:
            return Eligibility.no(
                "the entitlement runs twenty years from commissioning, which a grid with "
                "no start_date cannot place.  Anchor the timeline to a calendar."
            )
        return Eligibility.yes()

    def flows(self, asset: Asset, ctx: Context) -> CashFlowSeries:
        """Payment for every exported kilowatt-hour inside the entitlement."""
        if not isinstance(asset, System) or not self.eligibility(asset, ctx).anywhere:
            return CashFlowSeries()
        live = self.went_live(asset)
        if live is None:
            return CashFlowSeries()

        closes = self.value.runs_until(ctx, live)
        window = np.zeros(ctx.timeline.n_periods + 1, dtype=np.float64)
        window[ctx.start + 1 : closes + 1] = 1.0
        exported = ctx.usage.per_period(EXPORTED, ctx.timeline) * window
        earned = exported * self.value.per_kwh(asset, live)[..., np.newaxis]
        unpaid = np.asarray(self.negative_price_share, dtype=np.float64)[..., np.newaxis]

        flows: list[CashFlow] = [
            Explicit(
                earned * (1.0 - unpaid),
                label=Component(Role.INCENTIVE, "feed_in"),
                description=f"Export paid at {self.value.description}",
            )
        ]
        makeup = self._made_up(earned * unpaid, closes, ctx)
        if makeup is not None:
            flows.append(makeup)
        return CashFlowSeries(tuple(flows))

    def _made_up(self, lost: npt.NDArray[np.float64], closes: int, ctx: Context) -> CashFlow | None:
        """§51a's extension, as the lost revenue restored when the term ends.

        The entitlement is extended by the unpaid quarter-hours, so the energy
        is not forfeited.  It arrives twenty years out, which is the point.
        """
        if not self.makes_up_lost_hours or closes >= ctx.last:
            return None
        total = lost.sum(axis=-1)
        if not np.any(np.asarray(total) > 0.0):
            return None
        return OneOff(
            Amount.received(total),
            at=closes,
            label=Component(Role.INCENTIVE, "negative_price_makeup"),
            description="Entitlement extended for quarter-hours of negative price",
        )

    def constraints(self, asset: Asset, ctx: Context) -> tuple[str, ...]:
        """What taking the payment commits the operator to, and what it assumes."""
        notes = [
            "The tariff is fixed in money of the day for the whole term, so it buys less "
            "every year.  The retail price it is compared against does not.",
        ]
        if isinstance(self.value, FixedTariff) and not self.value.enacted:
            notes.append(
                "A regime on file is a published draft, not law.  Treat anything resting "
                "on it as a scenario."
            )
        if np.any(np.asarray(self.negative_price_share, dtype=np.float64) > 0.0):
            notes.append(
                "Export earns nothing in quarter-hours of negative spot price.  The "
                "entitlement is extended to make the energy up, but twenty years later, "
                "which is worth a fraction of losing it today."
            )
        return tuple(notes)

    def __str__(self) -> str:
        return f"FeedInTariff ({self.reference})"
