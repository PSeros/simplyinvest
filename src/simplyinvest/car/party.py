"""The buyer, and the facts a means-tested benefit needs from them."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

from simplyinvest.money import Quantity

__all__ = ["CarBuyer", "Household"]


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
class Household(CarBuyer):
    """A buyer with an income and a number of children.

    Args:
        taxable_income: Annual taxable income.
        children: Dependent children.
        name: What to call them in a report.

    Raises:
        ValueError: on a negative income or a negative number of children.
    """

    taxable_income: Quantity = 0.0
    children: int = 0
    name: str = "household"

    def __post_init__(self) -> None:
        if np.any(np.asarray(self.taxable_income, dtype=np.float64) < 0.0):
            raise ValueError(f"taxable income cannot be negative, got {self.taxable_income!r}")
        if self.children < 0:
            raise ValueError(f"children cannot be negative, got {self.children!r}")

    def __str__(self) -> str:
        return self.name
