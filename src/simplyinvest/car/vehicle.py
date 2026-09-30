"""The asset: one vehicle, and the facts incentives key on."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from enum import StrEnum
from typing import TYPE_CHECKING

import numpy as np

from simplyinvest.money import Amount, Quantity
from simplyinvest.timeline import Term

if TYPE_CHECKING:
    from simplyinvest.domain import ResidualValueModel

    from .energy import EnergySource

__all__ = ["Propulsion", "Vehicle", "VehicleCategory"]

#: Fields that are magnitudes, and so may not be negative.
_MAGNITUDES = (
    "price",
    "insurance",
    "maintenance",
    "circulation_tax",
    "other_annual_cost",
    "infrastructure_cost",
)


class VehicleCategory(StrEnum):
    """EU vehicle classes, to the extent eligibility depends on them."""

    M1 = "M1"
    N1 = "N1"
    L6E = "L6e"
    L7E = "L7e"
    OTHER = "other"


class Propulsion(StrEnum):
    """How the vehicle is driven, to the extent eligibility depends on it."""

    BEV = "bev"
    PHEV = "phev"
    FCEV = "fcev"
    HYBRID = "hybrid"
    ICE = "ice"


@dataclass(frozen=True)
class Vehicle:
    """One vehicle, priced at the age it is acquired.

    Args:
        name: What to call it.
        price: Market price paid at ``age_at_acquisition``.
        energy: How it is fuelled and what that costs.
        residual: The curve its terminal value comes from.
        propulsion: How it is driven.
        insurance: Annual premium.
        maintenance: Annual maintenance and wear.
        circulation_tax: Annual circulation tax before any exemption.
        other_annual_cost: Any further recurring cost.
        infrastructure_cost: One-off equipment needed to operate it, paid at
            acquisition.
        age_at_acquisition: How old it is when acquired.
        economic_life: How long it remains useful.
        first_registration: Date it was first registered.
        category: EU vehicle class.

    Raises:
        ValueError: on a negative magnitude or a negative age.
    """

    name: str
    price: Quantity
    energy: EnergySource
    residual: ResidualValueModel
    propulsion: Propulsion
    insurance: Quantity = 0.0
    maintenance: Quantity = 0.0
    circulation_tax: Quantity = 0.0
    other_annual_cost: Quantity = 0.0
    infrastructure_cost: Quantity = 0.0
    age_at_acquisition: Term = field(default_factory=lambda: Term.ZERO)
    economic_life: Term | None = None
    first_registration: date | None = None
    category: VehicleCategory = VehicleCategory.M1

    def __post_init__(self) -> None:
        for name in _MAGNITUDES:
            if np.any(np.asarray(getattr(self, name), dtype=np.float64) < 0.0):
                raise ValueError(f"{name} is an amount, not a direction; it cannot be negative")
        if self.age_at_acquisition.months < 0:
            raise ValueError(
                f"a vehicle cannot be acquired before it exists, got an age of "
                f"{self.age_at_acquisition}"
            )

    @property
    def capital_cost(self) -> Amount:
        """The acquisition outlay."""
        return Amount.paid(self.price)

    @property
    def setup_cost(self) -> Amount:
        """The charging or fuelling equipment bought with it."""
        return Amount.paid(self.infrastructure_cost)

    @property
    def is_used(self) -> bool:
        """Whether it was already in service when acquired."""
        return self.age_at_acquisition.months > 0

    @property
    def fixed_annual_cost(self) -> Quantity:
        """Insurance, maintenance, circulation tax and any other recurring cost."""
        return self.insurance + self.maintenance + self.circulation_tax + self.other_annual_cost

    def residual_value(self, *, held: Term) -> Amount:
        """What it is worth after being held for ``held``."""
        retained = self.residual.value_after(
            1.0, held=held, age_at_acquisition=self.age_at_acquisition
        )
        return Amount.received(self.price * retained)

    def __str__(self) -> str:
        return self.name
