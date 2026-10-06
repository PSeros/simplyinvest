"""The buyer, and the facts a means-tested benefit needs from them."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

from simplyinvest.car.price import EnergyPrice, duplicated
from simplyinvest.money import Quantity

__all__ = ["CarBuyer", "EnergyBuyer", "Household"]

#: A buyer who has quoted no prices at all.
_UNPRICED: Sequence[EnergyPrice] = ()


@runtime_checkable
class EnergyBuyer(Protocol):
    """A driver who has quoted what each carrier costs them."""

    @property
    def name(self) -> str:
        """What to call them in a report."""
        ...

    @property
    def energy_prices(self) -> Sequence[EnergyPrice]:
        """What each carrier costs them, one quote per carrier."""
        ...


@runtime_checkable
class CarBuyer(Protocol):
    """A buyer whose income and dependants a benefit may be keyed on."""

    @property
    def name(self) -> str:
        """What to call them in a report."""
        ...

    @property
    def taxable_income(self) -> Quantity:
        """Annual taxable income."""
        ...

    @property
    def children(self) -> int:
        """Dependent children."""
        ...


@dataclass(frozen=True)
class Household(CarBuyer, EnergyBuyer):
    """A buyer with an income, dependants, and a price for every carrier.

    The prices sit here rather than on each vehicle, so comparing two cars that
    burn the same fuel cannot quote it twice and disagree with itself.

    Args:
        taxable_income: Annual taxable income.
        children: Dependent children.
        energy_prices: One quote per carrier, each built from the source that
            burns it, as ``Petrol.price(1.79)``.
        name: What to call them in a report.

    Raises:
        ValueError: on a negative income, a negative number of children, or a
            carrier quoted twice.
    """

    taxable_income: Quantity = 0.0
    children: int = 0
    energy_prices: Sequence[EnergyPrice] = _UNPRICED
    name: str = "household"

    def __post_init__(self) -> None:
        if np.any(np.asarray(self.taxable_income, dtype=np.float64) < 0.0):
            raise ValueError(f"taxable income cannot be negative, got {self.taxable_income!r}")
        if self.children < 0:
            raise ValueError(f"children cannot be negative, got {self.children!r}")
        twice = duplicated(self.energy_prices)
        if twice:
            raise ValueError(
                f"{list(twice)} is quoted more than once, so which price applies would "
                f"be arbitrary.  Give each carrier exactly one quote."
            )

    def __str__(self) -> str:
        return self.name
