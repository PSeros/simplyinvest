"""Tax treatments: what they add to a stream, never what they rewrite."""

from __future__ import annotations

from .base import TaxTreatment, Untaxed
from .depreciation import declining_balance, straight_line, tabulated
from .treatments import FlatRateIncome, VatRegistered

__all__ = [
    "FlatRateIncome",
    "TaxTreatment",
    "Untaxed",
    "VatRegistered",
    "declining_balance",
    "straight_line",
    "tabulated",
]
