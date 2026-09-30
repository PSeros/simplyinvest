"""A lease that settles on the distance actually driven."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from simplyinvest.cashflow import CashFlow, Component, OneOff, Role
from simplyinvest.financing import Lease
from simplyinvest.money import Amount

from .usage import KM

if TYPE_CHECKING:
    from simplyinvest.domain import Asset, Context

__all__ = ["MileageLease"]


@dataclass(frozen=True)
class MileageLease(Lease):
    """A lease with a distance allowance, settled when each contract ends.

    Args:
        annual_included_km: Distance the rent covers each year.
        excess_rate: Charged per kilometre driven over the allowance.
        refund_rate: Credited per kilometre left unused.
        refund_cap_km: Most kilometres a contract will refund.

    Raises:
        ValueError: on a negative allowance, rate or cap.
    """

    annual_included_km: float = 0.0
    excess_rate: float = 0.0
    refund_rate: float = 0.0
    refund_cap_km: float = 0.0

    def __post_init__(self) -> None:
        super().__post_init__()
        for name in ("annual_included_km", "excess_rate", "refund_rate", "refund_cap_km"):
            if getattr(self, name) < 0.0:
                raise ValueError(f"{name} is a magnitude; it cannot be negative")

    def allowance(self, window: tuple[int, int], ctx: Context) -> float:
        """Kilometres the rent covers over ``window``."""
        start, end = window
        return self.annual_included_km * (end - start) / ctx.timeline.periods_per_year

    def driven(self, window: tuple[int, int], ctx: Context) -> Any:
        """Kilometres actually driven over ``window``.

        Raises:
            UnknownQuantityError: if the usage profile carries no distance.
        """
        start, end = window
        per_period = ctx.usage.per_period(KM, ctx.timeline)
        return np.sum(per_period[..., start + 1 : end + 1], axis=-1)

    def settlement(self, index: int, window: tuple[int, int], ctx: Context) -> CashFlow | None:
        """What the distance driven under contract ``index`` costs or refunds."""
        if not self.excess_rate and not self.refund_rate:
            return None
        over = np.maximum(self.driven(window, ctx) - self.allowance(window, ctx), 0.0)
        under = np.minimum(
            np.maximum(self.allowance(window, ctx) - self.driven(window, ctx), 0.0),
            self.refund_cap_km,
        )
        net = under * self.refund_rate - over * self.excess_rate
        if not np.any(net != 0.0):
            return None
        return OneOff(
            Amount.net(net),
            at=window[1],
            label=Component(Role.FINANCING, "mileage_settlement"),
            description=f"Distance settlement, contract {index + 1}",
        )

    def constraints(self, asset: Asset, ctx: Context) -> tuple[str, ...]:
        """The lease's obligations, plus the distance the allowance assumes."""
        notes = list(super().constraints(asset, ctx))
        if self.excess_rate:
            notes.append(
                f"Distance over {self.annual_included_km:,.0f} km a year is charged at "
                f"{self.excess_rate:,.2f} per km at the end of each contract."
            )
        return tuple(notes)
