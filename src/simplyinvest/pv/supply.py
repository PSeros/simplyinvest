"""What the roof supplies, as named quantities on the period grid."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from simplyinvest.domain import UsageProfile
from simplyinvest.errors import UnknownQuantityError

if TYPE_CHECKING:
    import numpy.typing as npt

    from simplyinvest.pv.generation import Generation
    from simplyinvest.pv.selfconsumption import SelfConsumption
    from simplyinvest.timeline import Timeline

__all__ = ["EXPORTED", "GENERATED", "SELF_CONSUMED", "PvSupply"]

#: Kilowatt-hours the modules produce.
GENERATED = "kwh_generated"

#: Kilowatt-hours used on site instead of bought from the grid.
SELF_CONSUMED = "kwh_self_consumed"

#: Kilowatt-hours fed into the grid.
EXPORTED = "kwh_exported"


@dataclass(frozen=True)
class PvSupply(UsageProfile):
    """Generation split between the site and the grid, resolved on demand.

    The split is computed each time it is asked for rather than stored, so a
    swept yield or share reaches it through the same parameter index as any
    other number in the model.

    Args:
        generation: How much the system makes.
        self_consumption: How much of that is used on site.
    """

    generation: Generation
    self_consumption: SelfConsumption

    @property
    def quantities(self) -> frozenset[str]:
        """The three energy streams this profile carries."""
        return frozenset({GENERATED, SELF_CONSUMED, EXPORTED})

    def per_period(self, quantity: str, timeline: Timeline) -> npt.NDArray[np.float64]:
        """How much of ``quantity`` falls in each period, shape ``(..., n + 1)``.

        Raises:
            UnknownQuantityError: if this profile does not carry it.
        """
        if quantity not in self.quantities:
            raise UnknownQuantityError(
                f"a photovoltaic supply carries no quantity {quantity!r}; it has "
                f"{sorted(self.quantities)}"
            )
        divided = self.self_consumption.split(self.generation, timeline)
        if quantity == GENERATED:
            return np.asarray(divided.generated, dtype=np.float64)
        if quantity == SELF_CONSUMED:
            return np.asarray(divided.self_consumed, dtype=np.float64)
        return np.asarray(divided.exported, dtype=np.float64)

    def annual(self, quantity: str) -> float:
        """The first full year of ``quantity``, before the modules age.

        Raises:
            UnknownQuantityError: if this profile does not carry it.
            ValueError: if the figure is a batch of draws.
        """
        if quantity not in self.quantities:
            raise UnknownQuantityError(
                f"a photovoltaic supply carries no quantity {quantity!r}; it has "
                f"{sorted(self.quantities)}"
            )
        produced = np.asarray(self.generation.first_year_kwh, dtype=np.float64)
        used = np.asarray(self.self_consumption.annual_share(self.generation), dtype=np.float64)
        totals = {GENERATED: produced, SELF_CONSUMED: produced * used}
        value = totals.get(quantity, produced * (1.0 - used))
        if value.ndim > 0:
            raise ValueError(
                f"this supply carries a batch of draws, so there is no single annual "
                f"{quantity}.  Take the figure from one trial, or from the base case."
            )
        return float(value)

    def __str__(self) -> str:
        return f"{self.generation}, {self.self_consumption}"
