"""What a kilowatt-hour from the grid costs the site."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np

from simplyinvest.money import Quantity

if TYPE_CHECKING:
    import numpy.typing as npt

    from simplyinvest.timeline import Timeline

__all__ = ["DEFAULT_PRICE_ESCALATION", "RetailPrice", "StaticPrice"]

#: Escalation key the retail price grows at unless told otherwise.
DEFAULT_PRICE_ESCALATION = "electricity"


@runtime_checkable
class RetailPrice(Protocol):
    """A supply contract, as a price per kilowatt-hour over time."""

    def per_kwh(self, timeline: Timeline) -> npt.NDArray[np.float64]:
        """What one kilowatt-hour costs in each period, shape ``(..., n + 1)``."""
        ...

    @property
    def is_time_varying(self) -> bool:
        """Whether the price moves within a period as well as between them."""
        ...


@dataclass(frozen=True)
class StaticPrice(RetailPrice):
    """One working price, escalating with the grid's electricity rate.

    The standing charge belongs nowhere near this: it is owed whatever the roof
    does, so it is never avoided.

    Args:
        price: Working price of one kilowatt-hour.
        escalation: How it grows, as an escalation key or a rate.

    Raises:
        ValueError: on a negative price.
    """

    price: Quantity = 0.0
    escalation: str | float = DEFAULT_PRICE_ESCALATION

    def __post_init__(self) -> None:
        if np.any(np.asarray(self.price, dtype=np.float64) < 0.0):
            raise ValueError(f"a retail price cannot be negative, got {self.price!r}")

    def per_kwh(self, timeline: Timeline) -> npt.NDArray[np.float64]:
        """The working price, grown onto each period."""
        base = np.asarray(self.price, dtype=np.float64)[..., np.newaxis]
        return np.asarray(base * timeline.accrual_index(self.escalation), dtype=np.float64)

    @property
    def is_time_varying(self) -> bool:
        """``False``; one price holds across every hour of a period."""
        return False

    def __str__(self) -> str:
        return f"{float(np.max(np.asarray(self.price))):.4f} per kWh"
