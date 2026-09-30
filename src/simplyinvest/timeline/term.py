"""A span of calendar time, measured in whole months."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from .conventions import MONTHS_PER_YEAR

__all__ = ["Term"]

_WHOLE_MONTH_TOLERANCE = 1e-9


@dataclass(frozen=True, slots=True, order=True)
class Term:
    """A whole number of months, which may be negative when used as an offset.

    Args:
        months: Length of the span.

    Raises:
        TypeError: if ``months`` is not an integer.
    """

    months: int

    ZERO: ClassVar[Term]

    def __post_init__(self) -> None:
        if not isinstance(self.months, int) or isinstance(self.months, bool):
            raise TypeError(
                f"a Term is a whole number of months, got {self.months!r}.  Use "
                f"Term.of_years(...) for a fractional year such as 1.5."
            )

    @classmethod
    def of_months(cls, months: int) -> Term:
        """A span of ``months`` months."""
        return cls(int(months))

    @classmethod
    def of_years(cls, years: float) -> Term:
        """A span of ``years`` years.

        Raises:
            ValueError: if ``years`` is not a whole number of months.
        """
        months = years * MONTHS_PER_YEAR
        if abs(months - round(months)) > _WHOLE_MONTH_TOLERANCE:
            raise ValueError(
                f"{years} years is {months} months, which is not a whole number.  "
                f"Express the span in months with Term.of_months(...) if you meant "
                f"something between."
            )
        return cls(round(months))

    @property
    def years(self) -> float:
        """This span in years, which may be fractional."""
        return self.months / MONTHS_PER_YEAR

    @property
    def is_zero(self) -> bool:
        """Whether this span is no time at all."""
        return self.months == 0

    def __add__(self, other: Term) -> Term:
        if not isinstance(other, Term):
            return NotImplemented
        return Term(self.months + other.months)

    def __sub__(self, other: Term) -> Term:
        if not isinstance(other, Term):
            return NotImplemented
        return Term(self.months - other.months)

    def __mul__(self, factor: int) -> Term:
        if not isinstance(factor, int) or isinstance(factor, bool):
            return NotImplemented
        return Term(self.months * factor)

    __rmul__ = __mul__

    def __neg__(self) -> Term:
        return Term(-self.months)

    def __str__(self) -> str:
        if self.months % MONTHS_PER_YEAR == 0 and self.months:
            years = self.months // MONTHS_PER_YEAR
            return f"{years} year{'s' if abs(years) != 1 else ''}"
        return f"{self.months} month{'s' if abs(self.months) != 1 else ''}"


Term.ZERO = Term(0)
