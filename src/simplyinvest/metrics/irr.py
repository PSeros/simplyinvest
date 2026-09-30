"""Internal rate of return."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from itertools import pairwise
from typing import TYPE_CHECKING

import numpy as np

from simplyinvest.timeline import Periodisation

if TYPE_CHECKING:
    import numpy.typing as npt

    from simplyinvest.timeline import Timeline

__all__ = ["IRR", "irr", "mirr"]

_BRACKET_SCAN = (-0.9999, -0.9, -0.5, -0.25, -0.1, 0.0, 0.1, 0.25, 0.5, 1.0, 3.0, 10.0, 100.0)

_MIN_FLOWS_FOR_A_SIGN_CHANGE = 2

_BRENT_ABSOLUTE_TOLERANCE = 2e-12
_BRENT_RELATIVE_TOLERANCE = 8.881784197001252e-16


@dataclass(frozen=True, slots=True)
class IRR:
    """A rate, or a statement of why there is none.  Truthy when a rate was found."""

    value: float | None
    reason: str
    sign_changes: int = 0

    def __bool__(self) -> bool:
        return self.value is not None

    def __str__(self) -> str:
        return f"{self.value:.2%}" if self.value is not None else f"no IRR: {self.reason}"


def _pv_at(rate: float, amounts: npt.NDArray[np.float64], timeline: Timeline) -> float:
    """Present value of ``amounts`` at an annual ``rate``, on this grid's convention."""
    if timeline.periodisation is Periodisation.PROPORTIONAL:
        periodic = rate / timeline.periods_per_year
    else:
        periodic = float((1.0 + rate) ** (1.0 / timeline.periods_per_year)) - 1.0
    steps = np.arange(amounts.shape[-1], dtype=np.float64)
    return float(amounts @ (1.0 + periodic) ** (-steps))


def _brent(f: Callable[[float], float], lo: float, hi: float, *, max_iter: int = 100) -> float:
    """A root of ``f`` bracketed by ``lo`` and ``hi``, by Brent's method."""
    xpre, xcur = lo, hi
    fpre, fcur = f(xpre), f(xcur)
    if fpre == 0.0:
        return xpre
    if fcur == 0.0:
        return xcur
    xblk = fblk = spre = scur = 0.0

    for _ in range(max_iter):
        if fpre * fcur < 0.0:
            xblk, fblk = xpre, fpre
            spre = scur = xcur - xpre
        if abs(fblk) < abs(fcur):
            xpre, xcur, xblk = xcur, xblk, xcur
            fpre, fcur, fblk = fcur, fblk, fcur

        delta = (_BRENT_ABSOLUTE_TOLERANCE + _BRENT_RELATIVE_TOLERANCE * abs(xcur)) / 2.0
        sbis = (xblk - xcur) / 2.0
        if fcur == 0.0 or abs(sbis) < delta:
            return xcur

        if abs(spre) > delta and abs(fcur) < abs(fpre):
            if xpre == xblk:
                stry = -fcur * (xcur - xpre) / (fcur - fpre)
            else:
                dpre = (fpre - fcur) / (xpre - xcur)
                dblk = (fblk - fcur) / (xblk - xcur)
                stry = -fcur * (fblk * dblk - fpre * dpre) / (dblk * dpre * (fblk - fpre))
            if 2.0 * abs(stry) < min(abs(spre), 3.0 * abs(sbis) - delta):
                spre, scur = scur, stry
            else:
                spre, scur = sbis, sbis
        else:
            spre, scur = sbis, sbis

        xpre, fpre = xcur, fcur
        xcur += scur if abs(scur) > delta else (delta if sbis > 0 else -delta)
        fcur = f(xcur)

    return xcur


def _sign_changes(amounts: npt.NDArray[np.float64]) -> int:
    """How many times the stream changes direction, ignoring zeros."""
    nonzero = amounts[amounts != 0.0]
    if nonzero.size < _MIN_FLOWS_FOR_A_SIGN_CHANGE:
        return 0
    return int(np.count_nonzero(np.diff(np.sign(nonzero))))


def _bracket_and_refine(pv_at: Callable[[float], float], changes: int) -> IRR:
    """The root of ``pv_at`` within the scanned range, or why there is none."""
    scanned = [(rate, pv_at(rate)) for rate in _BRACKET_SCAN]
    for (lo, flo), (hi, fhi) in pairwise(scanned):
        if flo == 0.0:
            return IRR(lo, "exact", changes)
        if flo * fhi < 0.0:
            return IRR(_brent(pv_at, lo, hi), "unique root, bracketed and refined", changes)
    return IRR(
        None,
        f"no break-even rate between {_BRACKET_SCAN[0]:.0%} and {_BRACKET_SCAN[-1]:.0%}",
        changes,
    )


def irr(amounts: npt.NDArray[np.float64], timeline: Timeline) -> IRR:
    """The annual rate at which ``amounts`` has zero present value.

    Returns:
        An :class:`IRR` holding the rate when the stream changes direction
        exactly once, otherwise ``None`` with the reason.

    Raises:
        ValueError: if given more than one stream.
    """
    values = np.asarray(amounts, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError(
            f"irr takes one stream at a time, got shape {values.shape}.  Loop over the "
            f"trials, or read a distribution of rates off the simulation instead."
        )

    changes = _sign_changes(values)
    if changes == 0:
        return IRR(
            None,
            "the stream never changes direction, so there is no rate at which it breaks even",
            changes,
        )
    if changes > 1:
        return IRR(
            None,
            f"the stream changes direction {changes} times, so it may have several "
            f"break-even rates; rank on net present value instead",
            changes,
        )

    def pv_at(rate: float) -> float:
        return _pv_at(rate, values, timeline)

    return _bracket_and_refine(pv_at, changes)


def mirr(
    amounts: npt.NDArray[np.float64],
    timeline: Timeline,
    *,
    finance_rate: float,
    reinvest_rate: float,
) -> float:
    """Modified internal rate of return.

    Outflows are discounted at ``finance_rate`` and inflows compounded to the
    horizon at ``reinvest_rate``.

    Raises:
        ValueError: if the stream has no outflow.
    """
    values = np.asarray(amounts, dtype=np.float64)
    n = values.shape[-1] - 1
    years = n / timeline.periods_per_year
    steps = np.arange(values.shape[-1], dtype=np.float64) / timeline.periods_per_year

    outflows = np.minimum(values, 0.0) @ (1.0 + finance_rate) ** (-steps)
    inflows = np.maximum(values, 0.0) @ (1.0 + reinvest_rate) ** (years - steps)
    if outflows == 0.0:
        raise ValueError("a modified IRR needs at least one outflow to finance")
    return float((inflows / -outflows) ** (1.0 / years) - 1.0)
