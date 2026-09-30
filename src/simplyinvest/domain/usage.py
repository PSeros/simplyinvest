"""Physical intensity of use, in whatever units a domain measures."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np

from simplyinvest.errors import UnknownQuantityError

if TYPE_CHECKING:
    import numpy.typing as npt

    from simplyinvest.timeline import Timeline

__all__ = [
    "ConstantAnnualUsage",
    "ProfileUsage",
    "UsageProfile",
]


@runtime_checkable
class UsageProfile(Protocol):
    """Named physical quantities, resolvable onto a period grid."""

    @property
    def quantities(self) -> frozenset[str]:
        """The named quantities this profile can supply."""
        ...

    def per_period(self, quantity: str, timeline: Timeline) -> npt.NDArray[np.float64]:
        """How much of ``quantity`` falls in each period, shape ``(n + 1,)``."""
        ...

    def annual(self, quantity: str) -> float:
        """The headline annual figure for ``quantity``."""
        ...


def _unknown(quantity: str, available: frozenset[str]) -> UnknownQuantityError:
    return UnknownQuantityError(
        f"this usage profile carries no quantity {quantity!r}; it has "
        f"{sorted(available) or 'none at all'}"
    )


@dataclass(frozen=True)
class ConstantAnnualUsage(UsageProfile):
    """Flat annual intensities, spread evenly and accruing in arrears.

    Args:
        amounts: Annual figure per quantity.
        growth: Annual growth rate per quantity.
    """

    amounts: Mapping[str, float] = field(default_factory=dict)
    growth: Mapping[str, float] = field(default_factory=dict)

    @property
    def quantities(self) -> frozenset[str]:
        """The named quantities this profile can supply."""
        return frozenset(self.amounts)

    def annual(self, quantity: str) -> float:
        """The headline annual figure for ``quantity``.

        Raises:
            UnknownQuantityError: if this profile does not carry it.
        """
        if quantity not in self.amounts:
            raise _unknown(quantity, self.quantities)
        return float(self.amounts[quantity])

    def per_period(self, quantity: str, timeline: Timeline) -> npt.NDArray[np.float64]:
        """How much of ``quantity`` falls in each period.

        Raises:
            UnknownQuantityError: if this profile does not carry it.
        """
        per = self.annual(quantity) / timeline.periods_per_year
        out = np.full(timeline.n_periods + 1, per, dtype=np.float64)
        out[0] = 0.0
        return out * timeline.accrual_index(self.growth.get(quantity, 0.0))


@dataclass(frozen=True)
class ProfileUsage(UsageProfile):
    """Per-period vectors resolved elsewhere.

    Args:
        vectors: Per-period values per quantity.
        annual_totals: Headline annual figures, inferred from the vectors when
            not given.
    """

    vectors: Mapping[str, npt.NDArray[np.float64]]
    annual_totals: Mapping[str, float] = field(default_factory=dict)

    @property
    def quantities(self) -> frozenset[str]:
        """The named quantities this profile can supply."""
        return frozenset(self.vectors)

    def annual(self, quantity: str) -> float:
        """The headline annual figure for ``quantity``.

        Raises:
            UnknownQuantityError: if this profile does not carry it.
        """
        if quantity in self.annual_totals:
            return float(self.annual_totals[quantity])
        if quantity not in self.vectors:
            raise _unknown(quantity, self.quantities)
        return float(np.sum(self.vectors[quantity]))

    def per_period(self, quantity: str, timeline: Timeline) -> npt.NDArray[np.float64]:
        """How much of ``quantity`` falls in each period.

        Raises:
            UnknownQuantityError: if this profile does not carry it.
            ValueError: if the stored vector does not match the grid's length.
        """
        if quantity not in self.vectors:
            raise _unknown(quantity, self.quantities)
        values = np.asarray(self.vectors[quantity], dtype=np.float64)
        if values.shape[-1] != timeline.n_periods + 1:
            raise ValueError(
                f"the {quantity!r} profile has {values.shape[-1]} periods but the grid "
                f"has {timeline.n_periods + 1}; resample it against this timeline."
            )
        return values
