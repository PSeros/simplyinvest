"""The asset: one photovoltaic system, and the facts its benefits key on."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import TYPE_CHECKING

import numpy as np

from simplyinvest.money import Amount, Quantity
from simplyinvest.timeline import Term

if TYPE_CHECKING:
    from simplyinvest.domain import ResidualValueModel

__all__ = ["Battery", "Inverter", "System"]

#: Fields of :class:`System` that are magnitudes, and so may not be negative.
_MAGNITUDES = (
    "price",
    "peak_kw",
    "connection_cost",
    "service_cost",
    "insurance",
    "metering_cost",
    "other_annual_cost",
)


@dataclass(frozen=True)
class Inverter:
    """What turns direct current into alternating current.

    Args:
        rated_kw: Alternating-current power it can deliver.
        efficiency: Share of direct-current energy that reaches the meter.
        replacement_cost: What a new one costs when this one fails.
        replaced_after: How long it lasts before replacement, if within the horizon.

    Raises:
        ValueError: if the efficiency is not a fraction above zero.
    """

    rated_kw: Quantity = 0.0
    efficiency: float = 0.96
    replacement_cost: Quantity = 0.0
    replaced_after: Term | None = field(default_factory=lambda: Term.of_years(13))

    def __post_init__(self) -> None:
        if not 0.0 < self.efficiency <= 1.0:
            raise ValueError(f"efficiency is a fraction above zero, got {self.efficiency}")


@dataclass(frozen=True)
class Battery:
    """Storage, sized by what can actually be taken out of it.

    Args:
        usable_kwh: Capacity available between the charge limits, not nameplate.
        round_trip_efficiency: Share of stored energy that comes back out.
        charge_kw: Fastest it can take energy in.
        discharge_kw: Fastest it can give energy out.
        standby_w: Continuous draw of the electronics, whatever the state.
        replacement_cost: What a new one costs when this one is spent.
        replaced_after: How long it lasts before replacement, if within the horizon.

    Raises:
        ValueError: on a negative size, or an efficiency outside ``(0, 1]``.
    """

    usable_kwh: Quantity = 0.0
    round_trip_efficiency: float = 0.90
    charge_kw: Quantity = 0.0
    discharge_kw: Quantity = 0.0
    standby_w: float = 0.0
    replacement_cost: Quantity = 0.0
    replaced_after: Term | None = None

    def __post_init__(self) -> None:
        if np.any(np.asarray(self.usable_kwh, dtype=np.float64) < 0.0):
            raise ValueError(f"usable capacity cannot be negative, got {self.usable_kwh!r}")
        if not 0.0 < self.round_trip_efficiency <= 1.0:
            raise ValueError(
                f"round-trip efficiency is a fraction above zero, got {self.round_trip_efficiency}"
            )
        if self.standby_w < 0.0:
            raise ValueError(f"standby draw cannot be negative, got {self.standby_w}")

    @property
    def one_way_efficiency(self) -> float:
        """Share of energy surviving a charge, or a discharge, taken alone."""
        return float(np.sqrt(self.round_trip_efficiency))

    def __str__(self) -> str:
        return f"{float(np.max(self.usable_kwh)):.1f} kWh usable"


@dataclass(frozen=True)
class System:
    """One photovoltaic installation, priced as it is bought.

    Args:
        name: What to call it.
        price: What the installation costs, all in.
        peak_kw: Nameplate power of the modules, which the tariff bands and the
            zero-rating threshold key on.
        residual: What it is worth as it ages.
        inverter: What converts its output.
        battery: Storage, if any.
        commissioning: The date it goes live, which keys the feed-in tariff.
        connection_cost: Grid connection and metering work at commissioning.
        service_cost: Servicing and monitoring each year.
        insurance: Cover each year.
        metering_cost: The metering-point fee each year.
        other_annual_cost: Anything else recurring.
        age_at_acquisition: How old it already is when acquired.
        economic_life: How long it remains useful.

    Raises:
        ValueError: on a negative magnitude.
    """

    name: str
    price: Quantity
    peak_kw: Quantity
    residual: ResidualValueModel
    inverter: Inverter | None = None
    battery: Battery | None = None
    commissioning: date | None = None
    connection_cost: Quantity = 0.0
    service_cost: Quantity = 0.0
    insurance: Quantity = 0.0
    metering_cost: Quantity = 0.0
    other_annual_cost: Quantity = 0.0
    age_at_acquisition: Term = field(default_factory=lambda: Term.ZERO)
    economic_life: Term | None = None

    def __post_init__(self) -> None:
        for name in _MAGNITUDES:
            if np.any(np.asarray(getattr(self, name), dtype=np.float64) < 0.0):
                raise ValueError(f"{name} is an amount, not a direction; it cannot be negative")

    @property
    def capital_cost(self) -> Amount:
        """What the installation costs."""
        return Amount.paid(self.price)

    @property
    def setup_cost(self) -> Amount:
        """Grid connection and metering work at commissioning."""
        return Amount.paid(self.connection_cost)

    @property
    def fixed_annual_cost(self) -> Quantity:
        """Everything owed each year regardless of what the sun does."""
        return self.service_cost + self.insurance + self.metering_cost + self.other_annual_cost

    @property
    def has_storage(self) -> bool:
        """Whether a battery is fitted."""
        if self.battery is None:
            return False
        return bool(np.any(np.asarray(self.battery.usable_kwh, dtype=np.float64) > 0.0))

    def residual_value(self, *, held: Term) -> Amount:
        """What the installation is worth after being held for ``held``."""
        fraction = self.residual.value_after(
            1.0, held=held, age_at_acquisition=self.age_at_acquisition
        )
        return Amount.received(self.price * fraction)

    def __str__(self) -> str:
        return f"{self.name} ({float(np.max(self.peak_kw)):.2f} kWp)"
