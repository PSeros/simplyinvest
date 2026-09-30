"""Depreciation schedules as fractions written off per year."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from simplyinvest.timeline import Term

__all__ = ["declining_balance", "straight_line", "tabulated"]

_TOTAL_TOLERANCE = 1e-9


def straight_line(life: Term) -> tuple[float, ...]:
    """An equal fraction written off every year.

    Raises:
        ValueError: if ``life`` is under a year.
    """
    years = round(life.years)
    if years < 1:
        raise ValueError(f"a depreciable life is at least a year, got {life}")
    return tuple([1.0 / years] * years)


def declining_balance(
    rate: float, life: Term, *, switch_to_straight_line: bool = True
) -> tuple[float, ...]:
    """A constant proportion of the remaining value each year.

    Args:
        rate: Fraction of the written-down value taken each year.
        life: How long the asset is written off over.
        switch_to_straight_line: Whether to switch once straight-line over the
            remaining years gives the larger deduction.

    Raises:
        ValueError: if ``life`` is under a year, or ``rate`` is not a fraction.
    """
    years = round(life.years)
    if years < 1:
        raise ValueError(f"a depreciable life is at least a year, got {life}")
    if not 0.0 < rate <= 1.0:
        raise ValueError(f"a declining-balance rate is a fraction, got {rate!r}")

    fractions: list[float] = []
    remaining = 1.0
    for year in range(years):
        left = years - year
        declining = remaining * rate
        straight = remaining / left
        taken = max(declining, straight) if switch_to_straight_line else declining
        taken = min(taken, remaining)
        fractions.append(taken)
        remaining -= taken
    return tuple(fractions)


def tabulated(fractions: Sequence[float]) -> tuple[float, ...]:
    """A schedule given outright.

    Raises:
        ValueError: if it is empty, holds a negative fraction, or totals more
            than the asset.
    """
    values = tuple(float(f) for f in fractions)
    if not values:
        raise ValueError("a depreciation schedule needs at least one year")
    if any(f < 0.0 for f in values):
        raise ValueError(f"a depreciation fraction cannot be negative, got {values}")
    total = sum(values)
    if total > 1.0 + _TOTAL_TOLERANCE:
        raise ValueError(
            f"this schedule writes off {total:.1%} of the asset, which is more than was paid"
        )
    return values
