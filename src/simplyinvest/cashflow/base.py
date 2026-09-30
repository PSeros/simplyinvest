"""What a cash flow is, and the labels a present value decomposes into."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from simplyinvest.timeline import Term

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt

    from simplyinvest.timeline import Timeline

__all__ = [
    "CashFlow",
    "Component",
    "Frequency",
    "Role",
]

_MONTHS_BETWEEN_PAYMENTS = {
    "monthly": 1,
    "quarterly": 3,
    "semiannual": 6,
    "annual": 12,
}


class Role(StrEnum):
    """The economic role of a flow."""

    CAPITAL = "capital"
    FINANCING = "financing"
    OPERATING = "operating"
    REVENUE = "revenue"
    TAX = "tax"
    INCENTIVE = "incentive"
    TERMINAL = "terminal"


@dataclass(frozen=True, slots=True, order=True)
class Component:
    """A :class:`Role` with an optional domain-owned sub-label."""

    role: Role
    detail: str = ""

    def __str__(self) -> str:
        return f"{self.role.value}/{self.detail}" if self.detail else self.role.value


class Frequency(StrEnum):
    """How often a recurring flow is paid."""

    PER_PERIOD = "per_period"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    SEMIANNUAL = "semiannual"
    ANNUAL = "annual"

    @property
    def term(self) -> Term | None:
        """The span between payments, or ``None`` for one per grid period."""
        if self is Frequency.PER_PERIOD:
            return None
        return Term.of_months(_MONTHS_BETWEEN_PAYMENTS[self.value])

    def stride(self, timeline: Timeline) -> int:
        """Periods between two consecutive payments on ``timeline``.

        Raises:
            TermNotRepresentableError: if the frequency does not fit the grid.
        """
        span = self.term
        return 1 if span is None else timeline.periods_in(span)


@runtime_checkable
class CashFlow(Protocol):
    """Anything that resolves to signed amounts on a period grid."""

    @property
    def label(self) -> Component:
        """What this flow is, for the present-value breakdown."""
        ...

    @property
    def description(self) -> str:
        """A human-readable line for reports."""
        ...

    def amounts(self, timeline: Timeline) -> npt.NDArray[np.float64]:
        """Signed amounts per period, shape ``(..., n_periods + 1)``."""
        ...
