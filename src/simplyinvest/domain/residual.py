"""Age-keyed curves from an asset's price to what it is later worth."""

from __future__ import annotations

from abc import ABC, abstractmethod
from bisect import bisect_right
from dataclasses import dataclass
from typing import TYPE_CHECKING

from simplyinvest.cashflow import Component, Role, Terminal

if TYPE_CHECKING:
    from simplyinvest.timeline import Term

    from .asset import Asset

__all__ = [
    "FirstYearDropThenGeometric",
    "GeometricDecline",
    "ResidualValueModel",
    "TabulatedResiduals",
    "terminal_value",
]

_MIN_TABLE_ROWS = 2


class ResidualValueModel(ABC):
    """A curve from an asset's age to the fraction of its price it retains."""

    @abstractmethod
    def retained(self, age: Term) -> float:
        """The fraction of the original price still held at ``age``."""

    def value_after(self, price_paid: float, *, held: Term, age_at_acquisition: Term) -> float:
        """What an asset bought at ``age_at_acquisition`` is worth after ``held``."""
        base = self.retained(age_at_acquisition)
        if base <= 0.0:
            return 0.0
        return price_paid * self.retained(age_at_acquisition + held) / base

    def __str__(self) -> str:
        return type(self).__name__


@dataclass(frozen=True)
class GeometricDecline(ResidualValueModel):
    """A constant proportion lost every year.

    Args:
        rate: Fraction of the remaining value lost per year.

    Raises:
        ValueError: if ``rate`` is not a fraction below one.
    """

    rate: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.rate < 1.0:
            raise ValueError(f"a decline rate is a fraction lost per year, got {self.rate!r}")

    def retained(self, age: Term) -> float:
        """The fraction of the original price still held at ``age``."""
        return float((1.0 - self.rate) ** age.years)


@dataclass(frozen=True)
class FirstYearDropThenGeometric(ResidualValueModel):
    """A discrete loss over the first year, then a constant proportion each year.

    Args:
        drop: Fraction lost across the first year.
        rate: Fraction of the remaining value lost per year thereafter.

    Raises:
        ValueError: if either argument is not a fraction below one.
    """

    drop: float
    rate: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.drop < 1.0:
            raise ValueError(f"the first-year drop is a fraction, got {self.drop!r}")
        if not 0.0 <= self.rate < 1.0:
            raise ValueError(f"a decline rate is a fraction lost per year, got {self.rate!r}")

    def retained(self, age: Term) -> float:
        """The fraction of the original price still held at ``age``."""
        if age.years <= 0.0:
            return 1.0
        after_drop = 1.0 - self.drop
        if age.years <= 1.0:
            return float(1.0 - self.drop * age.years)
        return float(after_drop * (1.0 - self.rate) ** (age.years - 1.0))


@dataclass(frozen=True)
class TabulatedResiduals(ResidualValueModel):
    """A table of age to retained fraction, interpolated linearly between rows.

    Beyond the last row the final year-on-year ratio is carried forward.

    Args:
        points: ``(age in years, retained fraction)`` pairs, ordered by age.

    Raises:
        ValueError: if there are fewer than two rows, or they are unordered.
    """

    points: tuple[tuple[float, float], ...]

    def __post_init__(self) -> None:
        if len(self.points) < _MIN_TABLE_ROWS:
            raise ValueError("a residual table needs at least two rows to interpolate between")
        ages = [age for age, _ in self.points]
        if ages != sorted(ages):
            raise ValueError("a residual table must be ordered by age")

    def retained(self, age: Term) -> float:
        """The fraction of the original price still held at ``age``."""
        years = age.years
        ages = [a for a, _ in self.points]
        values = [v for _, v in self.points]
        if years <= ages[0]:
            return values[0]
        if years >= ages[-1]:
            span = ages[-1] - ages[-2]
            ratio = (values[-1] / values[-2]) ** (1.0 / span) if values[-2] > 0 else 0.0
            return float(values[-1] * ratio ** (years - ages[-1]))
        i = bisect_right(ages, years) - 1
        width = ages[i + 1] - ages[i]
        weight = (years - ages[i]) / width
        return float(values[i] + weight * (values[i + 1] - values[i]))


def terminal_value(
    asset: Asset,
    *,
    held: Term,
    at: int,
    label: Component | None = None,
    description: str = "",
) -> Terminal:
    """The asset's residual value as a flow realised at period ``at``."""
    return Terminal(
        amount=asset.residual_value(held=held),
        at=at,
        label=label or Component(Role.TERMINAL, "residual"),
        description=description or f"Residual value of {asset.name}",
        basis=f"{type(asset).__name__} value curve after {held}",
    )
