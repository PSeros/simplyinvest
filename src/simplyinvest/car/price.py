"""What one unit of a carrier costs the driver.

A price is a contract, not a property of the car.  Two vehicles burning the
same fuel are filled at the same pump, so the figure is quoted once, on the
buyer, rather than once per vehicle.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np

from simplyinvest.errors import UnknownQuantityError
from simplyinvest.money import Quantity
from simplyinvest.timeline import warn_if_a_factor

if TYPE_CHECKING:
    import numpy.typing as npt

    from simplyinvest.timeline import Timeline

__all__ = [
    "DEFAULT_ESCALATION",
    "ChargingTariff",
    "EnergyPrice",
    "PumpPrice",
    "duplicated",
    "price_of",
]

#: Escalation key a carrier's price grows at unless told otherwise.
DEFAULT_ESCALATION = "energy"


def _check_escalation(escalation: str | float) -> None:
    """Check a rate given directly, rather than as a key into the grid.

    Raises:
        ValueError: if the rate is a fall of 100% or more per year.
    """
    if isinstance(escalation, str):
        return
    if escalation <= -1.0:
        raise ValueError(f"escalation is {escalation}, which is a fall of 100% or more per year")
    warn_if_a_factor("this price", escalation)


@runtime_checkable
class EnergyPrice(Protocol):
    """What one unit of a carrier costs over a period grid."""

    @property
    def carrier(self) -> str:
        """Which carrier this prices, so a quote cannot be filed under another."""
        ...

    def per_unit(self, timeline: Timeline) -> npt.NDArray[np.float64]:
        """The price of one unit in each period, shape ``(..., n + 1)``."""
        ...


@dataclass(frozen=True)
class PumpPrice(EnergyPrice):
    """One price per unit, escalating with the grid.

    Args:
        price: What one unit costs before escalation.
        escalation: Escalation key or annual rate the price grows at.
        carrier: Which carrier this prices.  Build it from the source instead,
            with ``Petrol.price(...)``, and this is filled in for you.

    Raises:
        ValueError: on a negative price, an implausible escalation, or no carrier.
    """

    price: Quantity = 0.0
    escalation: str | float = DEFAULT_ESCALATION
    carrier: str = ""

    def __post_init__(self) -> None:
        if np.any(np.asarray(self.price, dtype=np.float64) < 0.0):
            raise ValueError(f"a price cannot be negative, got {self.price!r}")
        _check_escalation(self.escalation)
        _require_carrier(self.carrier)

    def per_unit(self, timeline: Timeline) -> npt.NDArray[np.float64]:
        """The price of one unit in each period."""
        price = np.asarray(self.price, dtype=np.float64)[..., np.newaxis]
        return np.asarray(price * timeline.escalation_index(self.escalation), dtype=np.float64)

    def __str__(self) -> str:
        return f"{float(np.max(np.asarray(self.price))):.4f} per unit"


@dataclass(frozen=True)
class ChargingTariff(EnergyPrice):
    """Electricity drawn partly at home and partly in public.

    Where the car is plugged in is a habit of the driver, not a fact about the
    car, so the mix is quoted here.

    Args:
        home: Price per kWh charging at home.
        public: Price per kWh charging in public.
        home_share: Fraction of energy drawn at ``home``.
        escalation: Escalation key or annual rate the prices grow at.
        carrier: Which carrier this prices.  Build it from the source instead,
            with ``Electricity.price(...)``, and this is filled in for you.

    Raises:
        ValueError: on a negative price, a share outside ``[0, 1]``, an
            implausible escalation, or no carrier.
    """

    home: Quantity = 0.0
    public: Quantity = 0.0
    home_share: float = 1.0
    escalation: str | float = DEFAULT_ESCALATION
    carrier: str = ""

    def __post_init__(self) -> None:
        for name in ("home", "public"):
            if np.any(np.asarray(getattr(self, name), dtype=np.float64) < 0.0):
                raise ValueError(f"a price cannot be negative, got {getattr(self, name)!r}")
        if not 0.0 <= self.home_share <= 1.0:
            raise ValueError(f"home_share is a fraction of energy, got {self.home_share!r}")
        _check_escalation(self.escalation)
        _require_carrier(self.carrier)

    @property
    def blended(self) -> Quantity:
        """The home and public mix, per kWh, before escalation."""
        return self.home * self.home_share + self.public * (1.0 - self.home_share)

    def per_unit(self, timeline: Timeline) -> npt.NDArray[np.float64]:
        """The blended price of one kWh in each period."""
        price = np.asarray(self.blended, dtype=np.float64)[..., np.newaxis]
        return np.asarray(price * timeline.escalation_index(self.escalation), dtype=np.float64)

    def __str__(self) -> str:
        return f"{float(np.max(np.asarray(self.blended))):.4f} per kWh blended"


def _require_carrier(carrier: str) -> None:
    """Refuse a price that does not say what it prices.

    Raises:
        ValueError: if no carrier is named.
    """
    if not carrier:
        raise ValueError(
            "a price has to say which carrier it is for.  Build it from the source "
            "that burns it - Petrol.price(1.79), Electricity.price(0.31) - or pass "
            "carrier= yourself."
        )


def price_of(prices: Sequence[EnergyPrice], carrier: str) -> EnergyPrice:
    """The price quoted for ``carrier``.

    Raises:
        UnknownQuantityError: if nothing is quoted for it, naming what is.
    """
    for quoted in prices:
        if quoted.carrier == carrier:
            return quoted
    named = sorted(quoted.carrier for quoted in prices)
    raise UnknownQuantityError(
        f"no price is quoted for {carrier!r}; the buyer has prices for "
        f"{named or 'nothing at all'}.  Quote one with {carrier.capitalize()}.price(...)."
    )


def duplicated(prices: Sequence[EnergyPrice]) -> tuple[str, ...]:
    """Any carrier quoted more than once, which would make the answer arbitrary."""
    seen: list[str] = [quoted.carrier for quoted in prices]
    return tuple(sorted({name for name in seen if seen.count(name) > 1}))
