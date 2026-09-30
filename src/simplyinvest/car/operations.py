"""What it costs to run the vehicle."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from simplyinvest.cashflow import CashFlow, CashFlowSeries, Component, Explicit, Recurring, Role
from simplyinvest.money import Amount

from .usage import KM

if TYPE_CHECKING:
    import numpy.typing as npt

    from simplyinvest.domain import Context

    from .vehicle import Vehicle

__all__ = ["CarOperating"]

#: Escalation key the fixed annual costs grow at unless told otherwise.
DEFAULT_ESCALATION = "running_cost"

#: The fixed annual items, as (vehicle attribute, breakdown detail).
_FIXED = (
    ("insurance", "insurance"),
    ("maintenance", "maintenance"),
    ("circulation_tax", "circulation_tax"),
    ("other_annual_cost", "other"),
)


def window_mask(ctx: Context) -> npt.NDArray[np.float64]:
    """One in each period this context covers, zero elsewhere."""
    mask = np.zeros(ctx.timeline.n_periods + 1, dtype=np.float64)
    mask[ctx.start + 1 : ctx.last + 1] = 1.0
    return mask


@dataclass(frozen=True)
class CarOperating:
    """Energy and the fixed annual costs of keeping a vehicle on the road.

    Args:
        vehicle: The vehicle being run.
        escalation: Escalation key or annual rate the fixed costs grow at.
    """

    vehicle: Vehicle
    escalation: str | float = DEFAULT_ESCALATION

    def distance(self, ctx: Context) -> npt.NDArray[np.float64]:
        """Kilometres driven in each period of this window.

        Raises:
            UnknownQuantityError: if the usage profile carries no distance.
        """
        return ctx.usage.per_period(KM, ctx.timeline) * window_mask(ctx)

    def energy_cost(self, ctx: Context) -> npt.NDArray[np.float64]:
        """What the energy for those kilometres costs in each period."""
        return self.vehicle.energy.cost_per_period(self.distance(ctx), ctx.timeline)

    def flows(self, ctx: Context) -> CashFlowSeries:
        """Energy per period, and each fixed annual cost spread over the year."""
        flows: list[CashFlow] = [
            Explicit(
                -self.energy_cost(ctx),
                label=Component(Role.OPERATING, "energy"),
                description=f"{self.vehicle.energy} for {self.vehicle.name}",
            )
        ]
        per_year = ctx.timeline.periods_per_year
        for attribute, detail in _FIXED:
            annual = getattr(self.vehicle, attribute)
            if not np.any(np.asarray(annual, dtype=np.float64) != 0.0):
                continue
            flows.append(
                Recurring(
                    Amount.paid(annual / per_year),
                    growth=self.escalation,
                    start=ctx.start,
                    end=ctx.last,
                    label=Component(Role.OPERATING, detail),
                    description=f"{detail.replace('_', ' ').capitalize()}, per period",
                )
            )
        return CashFlowSeries(tuple(flows))

    def constraints(self, ctx: Context) -> tuple[str, ...]:
        """Non-cash obligations running the vehicle carries."""
        return ()

    @property
    def supports_batch(self) -> bool:
        """True; every cost is a vector over the same periods whatever the inputs."""
        return True

    def __str__(self) -> str:
        return f"Running {self.vehicle.name}"
