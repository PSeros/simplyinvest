"""The discrete period grid every cash flow is resolved against."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import date
from typing import TYPE_CHECKING, Any, Self

import numpy as np

from simplyinvest.errors import (
    InconsistentRateBasisError,
    NoCalendarAnchorError,
    NotAnchoredError,
    TermNotRepresentableError,
    TimelineError,
)

from .anchor import add_months, months_between
from .conventions import (
    ANCHORABLE_PERIODS,
    MONTHS_PER_YEAR,
    Escalation,
    Periodisation,
    RateBasis,
    fisher_nominal,
    fisher_real,
)
from .escalation import EscalationSet
from .term import Term

if TYPE_CHECKING:
    import numpy.typing as npt

__all__ = ["Timeline"]

_WHOLE_PERIOD_TOLERANCE = 1e-9


@dataclass(frozen=True)
class Timeline:
    """An evenly spaced period grid with discounting, escalation and a calendar.

    Args:
        horizon: Length of the appraisal.
        periods_per_year: Sub-periods per year.
        rate: Discount rate per annum, quoted on ``basis``.
        basis: Whether ``rate`` and the cash flows include inflation.
        inflation: General inflation, required on a real basis.
        escalations: Named annual growth rates.
        start_date: Calendar date of period 0.
        periodisation: How the annual rate becomes a per-period rate.
        escalation_mode: How growth is spread within a year.

    Raises:
        TimelineError: on a non-positive horizon or period count.
        InconsistentRateBasisError: if real and nominal quantities are mixed.
        NoCalendarAnchorError: if a start date is given for a grid that does not
            divide the year into whole months.
    """

    horizon: Term
    periods_per_year: int = MONTHS_PER_YEAR
    rate: float = 0.0
    basis: RateBasis = RateBasis.NOMINAL
    inflation: float | None = None
    escalations: EscalationSet | Mapping[str, float] = field(default_factory=EscalationSet)
    start_date: date | None = None
    periodisation: Periodisation = Periodisation.CONFORMAL
    escalation_mode: Escalation = Escalation.ANNUAL_STEP

    _factors: npt.NDArray[np.float64] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.horizon.months <= 0:
            raise TimelineError(f"the horizon must be positive, got {self.horizon}")
        if self.periods_per_year <= 0:
            raise TimelineError(f"periods_per_year must be positive, got {self.periods_per_year!r}")
        if self.rate <= -1.0:
            raise TimelineError(f"the discount rate must exceed -100%, got {self.rate!r}")
        object.__setattr__(self, "escalations", EscalationSet(self.escalations))
        self._validate_rate_basis()
        self._validate_anchor()
        _ = self.n_periods
        steps = np.arange(self.n_periods + 1, dtype=np.float64)
        object.__setattr__(self, "_factors", (1.0 + self.periodic_rate) ** (-steps))

    def _validate_rate_basis(self) -> None:
        if self.basis is RateBasis.REAL:
            if self.inflation is None:
                raise InconsistentRateBasisError(
                    "a real discount rate needs an explicit `inflation`, so that the "
                    "price level the cash flows are stated in is on the record.  Use "
                    "Timeline.real(...); the escalations are then read as real growth."
                )
            return
        if self.inflation is not None and not EscalationSet(self.escalations).any_nonzero:
            raise InconsistentRateBasisError(
                "nominal discounting with an inflation assumption but nothing growing: "
                "the cash flows are in today's money while the rate is not, which "
                "understates every later flow.  Either escalate the flows, or switch to "
                "Timeline.real(rate=fisher_real(nominal, inflation), inflation=...)."
            )

    def _validate_anchor(self) -> None:
        if self.start_date is None:
            return
        if self.periods_per_year not in ANCHORABLE_PERIODS:
            raise NoCalendarAnchorError(
                f"a start_date needs a grid that divides the year into whole months, "
                f"but periods_per_year is {self.periods_per_year}.  Use one of "
                f"{sorted(ANCHORABLE_PERIODS)}, or drop the anchor."
            )

    @classmethod
    def real(cls, horizon: Term, *, rate: float, inflation: float, **kwargs: Any) -> Self:
        """A real-basis grid, whose escalations are read as real growth."""
        return cls(horizon, rate=rate, basis=RateBasis.REAL, inflation=inflation, **kwargs)

    @classmethod
    def nominal(cls, horizon: Term, *, rate: float, **kwargs: Any) -> Self:
        """A nominal-basis grid."""
        return cls(horizon, rate=rate, basis=RateBasis.NOMINAL, **kwargs)

    def to_nominal(self) -> Timeline:
        """This grid restated on a nominal basis, escalations included."""
        if self.basis is RateBasis.NOMINAL:
            return self
        assert self.inflation is not None
        return Timeline(
            horizon=self.horizon,
            periods_per_year=self.periods_per_year,
            rate=fisher_nominal(self.rate, self.inflation),
            basis=RateBasis.NOMINAL,
            inflation=None,
            escalations=EscalationSet(self.escalations).to_nominal(self.inflation),
            start_date=self.start_date,
            periodisation=self.periodisation,
            escalation_mode=self.escalation_mode,
        )

    def to_real(self, inflation: float) -> Timeline:
        """This grid restated on a real basis at ``inflation``."""
        if self.basis is RateBasis.REAL:
            return self
        return Timeline(
            horizon=self.horizon,
            periods_per_year=self.periods_per_year,
            rate=fisher_real(self.rate, inflation),
            basis=RateBasis.REAL,
            inflation=inflation,
            escalations=EscalationSet(
                {k: fisher_real(v, inflation) for k, v in self.escalations.items()}
            ),
            start_date=self.start_date,
            periodisation=self.periodisation,
            escalation_mode=self.escalation_mode,
        )

    @property
    def n_periods(self) -> int:
        """Number of periods after ``t = 0``.  The grid runs ``0 .. n_periods``."""
        return self.periods_in(self.horizon)

    @property
    def periods(self) -> npt.NDArray[np.int64]:
        """Every period index on the grid."""
        return np.arange(self.n_periods + 1, dtype=np.int64)

    @property
    def last(self) -> int:
        """The final period index."""
        return self.n_periods

    def zeros(self, trials: int | None = None) -> npt.NDArray[np.float64]:
        """An empty amounts vector, or a batch of ``trials`` of them."""
        shape = (self.n_periods + 1,) if trials is None else (trials, self.n_periods + 1)
        return np.zeros(shape, dtype=np.float64)

    def periods_in(self, term: Term) -> int:
        """How many periods ``term`` spans on this grid.

        Raises:
            TermNotRepresentableError: if ``term`` is not a whole number of periods.
        """
        exact = term.months * self.periods_per_year / MONTHS_PER_YEAR
        if abs(exact - round(exact)) > _WHOLE_PERIOD_TOLERANCE:
            months_each = MONTHS_PER_YEAR / self.periods_per_year
            raise TermNotRepresentableError(
                f"{term} is {exact} periods on a grid of {self.periods_per_year} per year "
                f"({months_each:g} months each), which is not a whole number.  Either "
                f"choose a term that lands on the grid, or build the grid at a resolution "
                f"that fits the term."
            )
        return round(exact)

    def term_of(self, periods: int) -> Term:
        """The calendar span ``periods`` grid periods cover.

        Raises:
            TermNotRepresentableError: if the grid's periods are not whole months.
        """
        exact = periods * MONTHS_PER_YEAR / self.periods_per_year
        if abs(exact - round(exact)) > _WHOLE_PERIOD_TOLERANCE:
            raise TermNotRepresentableError(
                f"{periods} periods on a grid of {self.periods_per_year} per year is "
                f"{exact} months, which is not a whole number.  Use a grid whose periods "
                f"are whole months to report spans in calendar terms."
            )
        return Term.of_months(round(exact))

    def offset(self, t: int, term: Term) -> int:
        """Period ``t`` advanced by ``term``.

        Raises:
            TimelineError: if the result falls outside the grid.
        """
        result = t + self.periods_in(term)
        if not 0 <= result <= self.n_periods:
            raise TimelineError(
                f"period {t} advanced by {term} lands on {result}, outside the grid "
                f"0..{self.n_periods}.  Lengthen the horizon, or use clamped()."
            )
        return result

    def clamped(self, t: int) -> int:
        """Period ``t`` brought inside the grid."""
        return max(0, min(int(t), self.n_periods))

    @property
    def periodic_rate(self) -> float:
        """The discount rate for one period, on this grid's periodisation."""
        if self.periodisation is Periodisation.PROPORTIONAL:
            return self.rate / self.periods_per_year
        return float((1.0 + self.rate) ** (1.0 / self.periods_per_year)) - 1.0

    def discount_factor(self, t: int) -> float:
        """Present value of one unit received at period ``t``."""
        return float(self._factors[t])

    def discount_factors(self) -> npt.NDArray[np.float64]:
        """Present value of one unit at every period."""
        return self._factors.copy()

    def pv(self, amounts: npt.NDArray[np.float64]) -> Any:
        """Present value of an amounts vector, or of a batch of them.

        Args:
            amounts: Signed amounts per period, shape ``(..., n_periods + 1)``,
                where leading axes are trial axes.

        Returns:
            A float for a single vector, otherwise an array over the trial axes.

        Raises:
            TimelineError: if ``amounts`` does not match the grid's length.
        """
        values = np.asarray(amounts, dtype=np.float64)
        if values.shape[-1] != self.n_periods + 1:
            raise TimelineError(
                f"amounts has {values.shape[-1]} periods but this grid has "
                f"{self.n_periods + 1} (0..{self.n_periods})."
            )
        result = values @ self._factors
        return float(result) if result.ndim == 0 else result

    def year_index(self, t: int | npt.NDArray[np.int64]) -> Any:
        """Which whole year of the horizon period ``t`` falls in."""
        return np.asarray(t) // self.periods_per_year

    def _rate_for(self, growth: str | float) -> float:
        return self.escalations[growth] if isinstance(growth, str) else float(growth)

    def _point_exponent(self) -> npt.NDArray[np.float64]:
        if self.escalation_mode is Escalation.CONTINUOUS:
            return np.asarray(self.periods / self.periods_per_year, dtype=np.float64)
        return np.asarray(self.year_index(self.periods), dtype=np.float64)

    def _interval_exponent(self) -> npt.NDArray[np.float64]:
        steps = self.periods
        if self.escalation_mode is Escalation.CONTINUOUS:
            midpoints = np.maximum(steps - 0.5, 0.0) / self.periods_per_year
            return np.asarray(midpoints, dtype=np.float64)
        interval_starts = np.maximum(steps - 1, 0) // self.periods_per_year
        return np.asarray(interval_starts, dtype=np.float64)

    def escalation_index(self, growth: str | float) -> npt.NDArray[np.float64]:
        """The price level at each period, for a named or explicit growth rate."""
        rate = self._rate_for(growth)
        if rate == 0.0:
            return np.ones(self.n_periods + 1, dtype=np.float64)
        return np.asarray((1.0 + rate) ** self._point_exponent(), dtype=np.float64)

    def accrual_index(self, growth: str | float) -> npt.NDArray[np.float64]:
        """The price level over the interval ending at each period."""
        rate = self._rate_for(growth)
        if rate == 0.0:
            return np.ones(self.n_periods + 1, dtype=np.float64)
        return np.asarray((1.0 + rate) ** self._interval_exponent(), dtype=np.float64)

    @property
    def anchored(self) -> bool:
        """Whether this grid has a calendar date for its periods."""
        return self.start_date is not None

    @property
    def months_per_period(self) -> int:
        """Whole months in one period."""
        return MONTHS_PER_YEAR // self.periods_per_year

    def _require_anchor(self, what: str) -> date:
        if self.start_date is None:
            raise NotAnchoredError(
                f"{what} needs to know what date this grid starts on, but no "
                f"`start_date` was given.  Build the timeline with "
                f"start_date=date(YYYY, M, D)."
            )
        return self.start_date

    def date_of(self, t: int) -> date:
        """The calendar date period ``t`` begins on.

        Raises:
            NotAnchoredError: if this grid has no start date.
        """
        start = self._require_anchor("date_of")
        return add_months(start, int(t) * self.months_per_period)

    def period_of(self, when: date) -> int:
        """The period containing ``when``, which may fall outside the grid.

        Raises:
            NotAnchoredError: if this grid has no start date.
        """
        start = self._require_anchor("period_of")
        return months_between(start, when) // self.months_per_period

    def period_bounds(self, t: int) -> tuple[date, date]:
        """The half-open date range ``[start, end)`` that period ``t`` covers."""
        return self.date_of(t), self.date_of(t + 1)

    def dates(self) -> tuple[date, ...]:
        """The calendar date of every period on the grid."""
        return tuple(self.date_of(t) for t in range(self.n_periods + 1))

    def __str__(self) -> str:
        parts = [
            f"{self.horizon} at {self.periods_per_year}/year",
            f"{self.rate:.2%} {self.basis.value}",
        ]
        if self.start_date is not None:
            parts.append(f"from {self.start_date.isoformat()}")
        if EscalationSet(self.escalations).any_nonzero:
            parts.append(str(EscalationSet(self.escalations)))
        return " | ".join(parts)
