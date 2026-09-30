"""Conventions under which a discount rate is quoted."""

from __future__ import annotations

from enum import StrEnum

__all__ = [
    "ANCHORABLE_PERIODS",
    "MONTHS_PER_YEAR",
    "Escalation",
    "Periodisation",
    "RateBasis",
    "fisher_nominal",
    "fisher_real",
]

MONTHS_PER_YEAR = 12

ANCHORABLE_PERIODS = frozenset({1, 2, 4, 12})
"""Period counts that divide a year into whole months."""


class Periodisation(StrEnum):
    """How an annual rate becomes a per-period rate."""

    CONFORMAL = "conformal"
    """``(1 + i) ** (1 / m) - 1``, which compounds to exactly the annual rate."""

    PROPORTIONAL = "proportional"
    """``i / m``, which compounds to slightly more than the annual rate."""


class RateBasis(StrEnum):
    """Whether a rate and the cash flows beside it include inflation."""

    NOMINAL = "nominal"
    """Money of the day."""

    REAL = "real"
    """Today's money, net of a stated inflation."""


class Escalation(StrEnum):
    """How a growth rate is spread across the periods within a year."""

    ANNUAL_STEP = "annual_step"
    """Flat through a year, stepping on the anniversary."""

    CONTINUOUS = "continuous"
    """Growing every period."""


def fisher_nominal(real_rate: float, inflation: float) -> float:
    """The nominal rate equivalent to ``real_rate`` under ``inflation``."""
    return (1.0 + real_rate) * (1.0 + inflation) - 1.0


def fisher_real(nominal_rate: float, inflation: float) -> float:
    """The real rate equivalent to ``nominal_rate`` under ``inflation``."""
    return (1.0 + nominal_rate) / (1.0 + inflation) - 1.0
