"""What one kilometre costs, whatever the carrier."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

import numpy as np

from simplyinvest.money import Quantity

if TYPE_CHECKING:
    import numpy.typing as npt

    from simplyinvest.timeline import Timeline

__all__ = ["DISTANCE_UNIT", "EnergySource", "MeteredSource"]

#: Consumption is quoted per this many kilometres.
DISTANCE_UNIT = 100.0

#: Escalation key a carrier's price grows at unless told otherwise.
DEFAULT_ESCALATION = "energy"


class EnergySource(ABC):
    """A way of fuelling a vehicle, priced over a period grid."""

    @abstractmethod
    def cost_per_period(
        self, distance: npt.NDArray[np.float64], timeline: Timeline
    ) -> npt.NDArray[np.float64]:
        """What the kilometres in each period cost."""

    @abstractmethod
    def cost_per_km(self, timeline: Timeline) -> npt.NDArray[np.float64]:
        """The cost of one kilometre in each period."""

    def __str__(self) -> str:
        return type(self).__name__


@dataclass(frozen=True)
class MeteredSource(EnergySource):
    """A single carrier sold by the unit.

    Args:
        consumption: Units per 100 km, as type-approved.
        price: Price of one unit before escalation.
        real_world_factor: Uplift of real consumption over type approval.
        escalation: Escalation key or annual rate the price grows at.

    Raises:
        ValueError: on a negative consumption or price, or a non-positive
            real-world factor.
    """

    consumption: Quantity
    price: Quantity = 0.0
    real_world_factor: float = 1.0
    escalation: str | float = DEFAULT_ESCALATION

    #: What the carrier is sold in.
    unit: ClassVar[str] = "unit"

    def __post_init__(self) -> None:
        if np.any(np.asarray(self.consumption, dtype=np.float64) < 0.0):
            raise ValueError(f"consumption cannot be negative, got {self.consumption!r}")
        if np.any(np.asarray(self.price, dtype=np.float64) < 0.0):
            raise ValueError(f"a price cannot be negative, got {self.price!r}")
        if self.real_world_factor <= 0.0:
            raise ValueError(
                f"real_world_factor scales type-approved consumption, so it must be "
                f"positive, got {self.real_world_factor!r}"
            )

    @property
    def effective_consumption(self) -> Quantity:
        """Units per 100 km after the real-world uplift."""
        return self.consumption * self.real_world_factor

    def unit_price(self, timeline: Timeline) -> npt.NDArray[np.float64]:
        """The price of one unit in each period, shape ``(..., n + 1)``."""
        price = np.asarray(self.price, dtype=np.float64)
        return price[..., np.newaxis] * timeline.escalation_index(self.escalation)

    def cost_per_km(self, timeline: Timeline) -> npt.NDArray[np.float64]:
        """The cost of one kilometre in each period, shape ``(..., n + 1)``."""
        consumption = np.asarray(self.effective_consumption, dtype=np.float64)
        return consumption[..., np.newaxis] / DISTANCE_UNIT * self.unit_price(timeline)

    def cost_per_period(
        self, distance: npt.NDArray[np.float64], timeline: Timeline
    ) -> npt.NDArray[np.float64]:
        """What the kilometres in each period cost."""
        return distance * self.cost_per_km(timeline)
