"""Present value and the ratios read off it."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import numpy.typing as npt

    from simplyinvest.timeline import Timeline

__all__ = [
    "net_present_value",
    "profitability_index",
    "pv_of_inflows",
    "pv_of_outflows",
]


def net_present_value(amounts: npt.NDArray[np.float64], timeline: Timeline) -> Any:
    """Present value of a signed amounts vector, or a batch of them."""
    return timeline.pv(amounts)


def pv_of_outflows(amounts: npt.NDArray[np.float64], timeline: Timeline) -> Any:
    """Present value of the money going out, as a positive number."""
    return -timeline.pv(np.minimum(np.asarray(amounts, dtype=np.float64), 0.0))


def pv_of_inflows(amounts: npt.NDArray[np.float64], timeline: Timeline) -> Any:
    """Present value of the money coming in, as a positive number."""
    return timeline.pv(np.maximum(np.asarray(amounts, dtype=np.float64), 0.0))


def profitability_index(amounts: npt.NDArray[np.float64], timeline: Timeline) -> Any:
    """Present value of inflows per unit of outflow, or infinity if none go out."""
    out = np.asarray(pv_of_outflows(amounts, timeline), dtype=np.float64)
    coming_in = np.asarray(pv_of_inflows(amounts, timeline), dtype=np.float64)
    nothing_out = out == 0.0
    safe_denominator = np.where(nothing_out, 1.0, out)
    ratio = np.where(nothing_out, np.inf, coming_in / safe_denominator)
    return float(ratio) if ratio.ndim == 0 else ratio
