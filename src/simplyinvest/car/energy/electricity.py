"""Grid electricity, with a home/public price split and charging losses."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

import numpy as np

from simplyinvest.money import Quantity

from .base import MeteredSource

if TYPE_CHECKING:
    import numpy.typing as npt

    from simplyinvest.timeline import Timeline

__all__ = ["Electricity"]


@dataclass(frozen=True)
class Electricity(MeteredSource):
    """Grid electricity drawn partly at home and partly in public.

    ``consumption`` is measured at the battery and ``charging_loss`` is taken
    between meter and battery, so the billed energy is the larger figure.

    Args:
        public_price: Price per kWh charging in public.
        home_share: Fraction of energy drawn at ``price``.
        charging_loss: Fraction of energy lost between meter and battery.

    Raises:
        ValueError: if the home share is outside ``[0, 1]`` or the charging
            loss is outside ``[0, 1)``.
    """

    public_price: Quantity = 0.0
    home_share: float = 1.0
    charging_loss: float = 0.0

    unit: ClassVar[str] = "kWh"

    def __post_init__(self) -> None:
        super().__post_init__()
        if np.any(np.asarray(self.public_price, dtype=np.float64) < 0.0):
            raise ValueError(f"a price cannot be negative, got {self.public_price!r}")
        if not 0.0 <= self.home_share <= 1.0:
            raise ValueError(f"home_share is a fraction of energy, got {self.home_share!r}")
        if not 0.0 <= self.charging_loss < 1.0:
            raise ValueError(f"charging_loss is a fraction below one, got {self.charging_loss!r}")

    @property
    def blended_price(self) -> Quantity:
        """The home and public mix, per kWh, before escalation."""
        return self.price * self.home_share + self.public_price * (1.0 - self.home_share)

    @property
    def effective_consumption(self) -> Quantity:
        """Energy drawn at the meter, per 100 km."""
        return self.consumption * self.real_world_factor / (1.0 - self.charging_loss)

    def unit_price(self, timeline: Timeline) -> npt.NDArray[np.float64]:
        """The blended price of one kWh in each period, shape ``(..., n + 1)``."""
        price = np.asarray(self.blended_price, dtype=np.float64)
        return price[..., np.newaxis] * timeline.escalation_index(self.escalation)
