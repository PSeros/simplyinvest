"""How long an investment takes to recover its outlay."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import numpy.typing as npt

    from simplyinvest.timeline import Term, Timeline

__all__ = ["discounted_payback", "payback"]


def _recovery(cumulative: npt.NDArray[np.float64], timeline: Timeline) -> Term | None:
    """When the running total first returns to zero after going below it."""
    under = np.flatnonzero(cumulative < 0.0)
    if under.size == 0:
        return timeline.term_of(0)
    start = int(under[0])
    reached = np.flatnonzero(cumulative[start:] >= 0.0)
    return None if reached.size == 0 else timeline.term_of(start + int(reached[0]))


def payback(amounts: npt.NDArray[np.float64], timeline: Timeline) -> Term | None:
    """When the undiscounted running total recovers, or ``None`` within the horizon."""
    values = np.asarray(amounts, dtype=np.float64)
    return _recovery(np.cumsum(values), timeline)


def discounted_payback(amounts: npt.NDArray[np.float64], timeline: Timeline) -> Term | None:
    """When the discounted running total recovers, or ``None`` within the horizon."""
    values = np.asarray(amounts, dtype=np.float64)
    return _recovery(np.cumsum(values * timeline.discount_factors()), timeline)
