"""How far the vehicle is driven."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from simplyinvest.errors import UnknownQuantityError
from simplyinvest.money import Quantity

if TYPE_CHECKING:
    import numpy.typing as npt

    from simplyinvest.timeline import Timeline

__all__ = ["KM", "Mileage"]

#: The quantity this domain measures use in.
KM = "km"


@dataclass(frozen=True)
class Mileage:
    """Distance driven per year, spread evenly and accruing in arrears.

    Args:
        annual_km: Kilometres driven in the first year.
        growth: Annual growth of the distance driven.

    Raises:
        ValueError: on a negative distance.
    """

    annual_km: Quantity = 0.0
    growth: float = 0.0

    def __post_init__(self) -> None:
        if np.any(np.asarray(self.annual_km, dtype=np.float64) < 0.0):
            raise ValueError(f"a distance cannot be negative, got {self.annual_km!r}")

    @property
    def quantities(self) -> frozenset[str]:
        """The named quantities this profile can supply."""
        return frozenset({KM})

    def annual(self, quantity: str) -> float:
        """The headline annual distance.

        Raises:
            UnknownQuantityError: if ``quantity`` is not kilometres.
            ValueError: if the distance is a vector of draws rather than one number.
        """
        self._check(quantity)
        driven = np.asarray(self.annual_km, dtype=np.float64)
        if driven.size != 1:
            raise ValueError(
                f"annual_km holds {driven.size} draws, so there is no single annual "
                f"distance.  Read the distribution from a Simulation instead."
            )
        return float(driven.reshape(()))

    def per_period(self, quantity: str, timeline: Timeline) -> npt.NDArray[np.float64]:
        """How far the vehicle travels in each period, shape ``(..., n + 1)``.

        Raises:
            UnknownQuantityError: if ``quantity`` is not kilometres.
        """
        self._check(quantity)
        per = np.asarray(self.annual_km, dtype=np.float64) / timeline.periods_per_year
        spread = np.ones(timeline.n_periods + 1, dtype=np.float64)
        spread[0] = 0.0
        return per[..., None] * spread * timeline.accrual_index(self.growth)

    def _check(self, quantity: str) -> None:
        if quantity != KM:
            raise UnknownQuantityError(
                f"a Mileage carries no quantity {quantity!r}; it has ['{KM}']"
            )

    def __str__(self) -> str:
        return f"{self.annual_km:,.0f} km a year"
