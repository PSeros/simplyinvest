"""How much a vehicle burns, and what that comes to per kilometre.

A source knows the carrier it needs and how much of it the vehicle uses.  What
the carrier costs is quoted by the buyer, not here.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

import numpy as np

from simplyinvest.car.price import DEFAULT_ESCALATION, EnergyPrice, PumpPrice, price_of
from simplyinvest.money import Quantity

if TYPE_CHECKING:
    import numpy.typing as npt

    from simplyinvest.timeline import Timeline

__all__ = ["DISTANCE_UNIT", "EnergySource", "MeteredSource"]

#: Consumption is quoted per this many kilometres.
DISTANCE_UNIT = 100.0


class EnergySource(ABC):
    """A way of fuelling a vehicle."""

    @property
    @abstractmethod
    def carriers(self) -> frozenset[str]:
        """Every carrier this source draws on."""

    @abstractmethod
    def cost_per_km(
        self, prices: Sequence[EnergyPrice], timeline: Timeline
    ) -> npt.NDArray[np.float64]:
        """The cost of one kilometre in each period."""

    def cost_per_period(
        self,
        distance: npt.NDArray[np.float64],
        prices: Sequence[EnergyPrice],
        timeline: Timeline,
    ) -> npt.NDArray[np.float64]:
        """What the kilometres in each period cost."""
        return distance * self.cost_per_km(prices, timeline)

    def __str__(self) -> str:
        return type(self).__name__


@dataclass(frozen=True)
class MeteredSource(EnergySource):
    """A single carrier sold by the unit.

    Args:
        consumption: Units per 100 km, as type-approved.
        real_world_factor: Uplift of real consumption over type approval.

    Raises:
        ValueError: on a negative consumption, or a non-positive real-world factor.
    """

    consumption: Quantity = 0.0
    real_world_factor: float = 1.0

    #: What the carrier is sold in.
    unit: ClassVar[str] = "unit"

    #: What the buyer must have quoted a price for.
    carrier: ClassVar[str] = "energy"

    def __post_init__(self) -> None:
        if np.any(np.asarray(self.consumption, dtype=np.float64) < 0.0):
            raise ValueError(f"consumption cannot be negative, got {self.consumption!r}")
        if self.real_world_factor <= 0.0:
            raise ValueError(
                f"real_world_factor scales type-approved consumption, so it must be "
                f"positive, got {self.real_world_factor!r}"
            )

    @classmethod
    def price(
        cls, price: Quantity = 0.0, *, escalation: str | float = DEFAULT_ESCALATION
    ) -> EnergyPrice:
        """A price quoted in the form this carrier is sold in.

        Args:
            price: What one unit costs before escalation.
            escalation: Escalation key or annual rate the price grows at.
        """
        return PumpPrice(price, escalation, carrier=cls.carrier)

    @property
    def carriers(self) -> frozenset[str]:
        """The one carrier this source draws on."""
        return frozenset({self.carrier})

    @property
    def effective_consumption(self) -> Quantity:
        """Units per 100 km after the real-world uplift."""
        return self.consumption * self.real_world_factor

    def units_per_km(self) -> npt.NDArray[np.float64]:
        """Units drawn for one kilometre."""
        used = np.asarray(self.effective_consumption, dtype=np.float64)
        return np.asarray(used[..., np.newaxis] / DISTANCE_UNIT, dtype=np.float64)

    def cost_per_km(
        self, prices: Sequence[EnergyPrice], timeline: Timeline
    ) -> npt.NDArray[np.float64]:
        """The cost of one kilometre in each period, shape ``(..., n + 1)``.

        Raises:
            UnknownQuantityError: if the buyer quotes no price for this carrier.
        """
        return self.units_per_km() * price_of(prices, self.carrier).per_unit(timeline)
