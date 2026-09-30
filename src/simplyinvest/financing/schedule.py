"""Loan instalments, split into interest and principal."""

from __future__ import annotations

from dataclasses import dataclass
from math import expm1, log1p
from typing import TYPE_CHECKING

import numpy as np

from simplyinvest.errors import CashFlowError

if TYPE_CHECKING:
    import numpy.typing as npt

__all__ = ["AmortisationSchedule", "level_payment"]

_CLOSING_TOLERANCE = 1e-9

_DISPLAY_ZERO = 5e-3


def level_payment(principal: float, periodic_rate: float, n: int, balloon: float = 0.0) -> float:
    """The constant instalment that repays ``principal`` down to ``balloon``.

    Args:
        principal: Amount borrowed.
        periodic_rate: Interest per period.
        n: Number of instalments.
        balloon: Balance left outstanding at the end.

    Raises:
        CashFlowError: if ``n`` is not positive.
    """
    if n <= 0:
        raise CashFlowError(f"a loan needs at least one instalment, got {n}")
    if periodic_rate == 0.0:
        return (principal - balloon) / n

    growth = log1p(periodic_rate)
    discount = expm1(-n * growth)
    annuity = -discount / periodic_rate
    if annuity == 0.0:
        return (principal - balloon) / n
    amortised = principal - balloon * (1.0 + discount)
    return float(amortised / annuity)


@dataclass(frozen=True)
class AmortisationSchedule:
    """An instalment-by-instalment account of a loan.

    Arrays run over the instalments; ``period[k]`` is the grid period
    instalment ``k`` falls on.

    Raises:
        CashFlowError: if the final balance is not zero.
    """

    period: npt.NDArray[np.int64]
    opening: npt.NDArray[np.float64]
    interest: npt.NDArray[np.float64]
    principal: npt.NDArray[np.float64]
    closing: npt.NDArray[np.float64]

    def __post_init__(self) -> None:
        self.check_closes()

    @classmethod
    def build(
        cls,
        *,
        principal: float,
        periodic_rate: float,
        periods: npt.NDArray[np.int64],
        balloon: float = 0.0,
    ) -> AmortisationSchedule:
        """Amortise ``principal`` over ``periods``, settling ``balloon`` at the end."""
        n = len(periods)
        payment = level_payment(principal, periodic_rate, n, balloon)

        opening = np.zeros(n, dtype=np.float64)
        interest = np.zeros(n, dtype=np.float64)
        repaid = np.zeros(n, dtype=np.float64)
        closing = np.zeros(n, dtype=np.float64)

        balance = principal
        for k in range(n):
            opening[k] = balance
            interest[k] = balance * periodic_rate
            repaid[k] = payment - interest[k]
            balance -= repaid[k]
            closing[k] = balance

        if balloon:
            repaid[-1] += closing[-1]
            closing[-1] = 0.0

        return cls(np.asarray(periods, dtype=np.int64), opening, interest, repaid, closing)

    @property
    def instalment(self) -> float:
        """The level payment, before any balloon settlement."""
        return float(self.interest[0] + self.principal[0])

    @property
    def payments(self) -> npt.NDArray[np.float64]:
        """What is handed over at each instalment, balloon included."""
        return self.interest + self.principal

    def total_interest(self) -> float:
        """The undiscounted cost of borrowing."""
        return float(self.interest.sum())

    def check_closes(self, tolerance: float = _CLOSING_TOLERANCE) -> None:
        """Assert the loan is fully repaid, within ``tolerance`` of the principal.

        Raises:
            CashFlowError: if a balance remains after the last instalment.
        """
        left = abs(float(self.closing[-1]))
        allowed = max(tolerance, tolerance * abs(float(self.opening[0])))
        if left > allowed:
            raise CashFlowError(
                f"the schedule leaves {left:,.6g} outstanding after the last instalment "
                f"on {float(self.opening[0]):,.2f} borrowed; the instalment and the "
                f"principal do not agree."
            )

    @staticmethod
    def _cell(value: float) -> str:
        """Two decimals, with anything below half a cent shown as zero."""
        return f"{0.0 if abs(value) < _DISPLAY_ZERO else value:,.2f}"

    def to_markdown(self, *, every: int = 12) -> str:
        """A table with one row every ``every`` instalments, plus the last."""
        rows = ["| Period | Opening | Interest | Principal | Closing |", "| ---: |" + " ---: |" * 4]
        shown = set(range(0, len(self.period), max(every, 1))) | {len(self.period) - 1}
        for k in sorted(shown):
            rows.append(
                f"| {self.period[k]} | {self._cell(self.opening[k])} "
                f"| {self._cell(self.interest[k])} | {self._cell(self.principal[k])} "
                f"| {self._cell(self.closing[k])} |"
            )
        return "\n".join(rows)
