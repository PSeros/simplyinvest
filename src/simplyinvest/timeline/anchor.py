"""Calendar arithmetic in whole months."""

from __future__ import annotations

import calendar
from datetime import date

__all__ = [
    "add_months",
    "months_between",
]


def add_months(start: date, months: int) -> date:
    """``start`` advanced by ``months``, clamped to the target month's last day."""
    total = start.month - 1 + months
    year = start.year + total // 12
    month = total % 12 + 1
    return date(year, month, min(start.day, calendar.monthrange(year, month)[1]))


def months_between(start: date, end: date) -> int:
    """Whole months from ``start`` to ``end``, rounded towards minus infinity."""
    months = (end.year - start.year) * 12 + (end.month - start.month)
    if end.day < start.day:
        months -= 1
    return months
