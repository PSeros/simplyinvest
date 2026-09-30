"""Two carriers sharing one vehicle's mileage."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from .base import EnergySource

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt

    from simplyinvest.timeline import Timeline

__all__ = ["Bivalent"]


@dataclass(frozen=True)
class Bivalent(EnergySource):
    """Two sources splitting the distance driven.

    Covers a plug-in hybrid's electric share and a bivalent gas conversion
    alike.  The carriers are priced in different units, so only the cost per
    kilometre is defined for the pair.

    Args:
        primary: The source carrying ``primary_share`` of the distance.
        secondary: The source carrying the rest.
        primary_share: Fraction of the distance run on ``primary``.

    Raises:
        ValueError: if the share is outside ``[0, 1]``.
    """

    primary: EnergySource
    secondary: EnergySource
    primary_share: float = 1.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.primary_share <= 1.0:
            raise ValueError(
                f"primary_share is a fraction of the distance, got {self.primary_share!r}"
            )

    def cost_per_km(self, timeline: Timeline) -> npt.NDArray[np.float64]:
        """The distance-weighted cost of one kilometre in each period."""
        return self.primary.cost_per_km(timeline) * self.primary_share + self.secondary.cost_per_km(
            timeline
        ) * (1.0 - self.primary_share)

    def cost_per_period(
        self, distance: npt.NDArray[np.float64], timeline: Timeline
    ) -> npt.NDArray[np.float64]:
        """What the kilometres in each period cost across both carriers."""
        return distance * self.cost_per_km(timeline)

    def __str__(self) -> str:
        return f"{self.primary_share:.0%} {self.primary}, rest {self.secondary}"
