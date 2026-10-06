"""The operator, and the facts a photovoltaic appraisal needs from them."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

from simplyinvest.money import Quantity
from simplyinvest.pv.price import RetailPrice, StaticPrice

__all__ = ["Householder", "PvOperator"]

#: A contract quoting nothing, so nothing is avoided by using your own power.
_UNPRICED = StaticPrice()


@runtime_checkable
class PvOperator(Protocol):
    """Whoever runs the system, and buys the electricity it does not supply."""

    @property
    def name(self) -> str:
        """What to call them in a report."""
        ...

    @property
    def retail_price(self) -> RetailPrice:
        """The supply contract a self-consumed kilowatt-hour displaces."""
        ...

    @property
    def annual_consumption_kwh(self) -> Quantity:
        """How much electricity the site uses in a year."""
        ...


@dataclass(frozen=True)
class Householder(PvOperator):
    """A household that consumes its own generation and buys the rest.

    The price is a contract, not a number, so a flat tariff and a spot-linked
    one are the same kind of thing to everything downstream.

    Args:
        retail_price: The supply contract a self-consumed kilowatt-hour
            displaces, as a :class:`~simplyinvest.pv.price.RetailPrice`.
        annual_consumption_kwh: How much electricity the site uses in a year.
        name: What to call them in a report.

    Raises:
        ValueError: on a negative consumption.
    """

    retail_price: RetailPrice = _UNPRICED
    annual_consumption_kwh: Quantity = 0.0
    name: str = "householder"

    def __post_init__(self) -> None:
        if np.any(np.asarray(self.annual_consumption_kwh, dtype=np.float64) < 0.0):
            raise ValueError(
                f"annual consumption cannot be negative, got {self.annual_consumption_kwh!r}"
            )

    def __str__(self) -> str:
        return self.name
