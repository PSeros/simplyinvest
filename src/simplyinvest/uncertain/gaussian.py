"""The standard normal distribution, in pure numpy."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import numpy.typing as npt

__all__ = ["norm_cdf", "norm_pdf", "norm_ppf", "unit_interval"]

_INV_SQRT_TWO = 1.0 / math.sqrt(2.0)

_INV_SQRT_TWO_PI = 1.0 / math.sqrt(2.0 * math.pi)

# Abramowitz & Stegun 26.2.23, accurate to about 4.5e-4 before refinement.
_START_NUMERATOR = (2.515517, 0.802853, 0.010328)
_START_DENOMINATOR = (1.432788, 0.189269, 0.001308)

_REFINEMENTS = 3

_EDGE = 1e-15

_HALF = 0.5

_erfc = np.frompyfunc(math.erfc, 1, 1)


def unit_interval(u: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """``u`` pulled just inside ``(0, 1)`` so an inverse CDF stays finite."""
    return np.clip(np.asarray(u, dtype=np.float64), _EDGE, 1.0 - _EDGE)


def norm_cdf(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Probability a standard normal falls at or below ``x``."""
    values = np.asarray(x, dtype=np.float64)
    return np.asarray(_erfc(-values * _INV_SQRT_TWO), dtype=np.float64) * _HALF


def norm_pdf(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Density of a standard normal at ``x``."""
    values = np.asarray(x, dtype=np.float64)
    return _INV_SQRT_TWO_PI * np.exp(-_HALF * values * values)


def norm_ppf(u: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """The value a standard normal falls below with probability ``u``.

    Raises:
        ValueError: if any probability lies outside ``(0, 1)``.
    """
    probabilities = np.asarray(u, dtype=np.float64)
    if np.any((probabilities <= 0.0) | (probabilities >= 1.0)):
        raise ValueError(
            "a probability must lie strictly between 0 and 1; pass it through "
            "unit_interval() if it comes from a random draw."
        )
    quantile = _rational_start(probabilities)
    for _ in range(_REFINEMENTS):
        quantile = _halley_step(quantile, probabilities)
    return quantile


def _rational_start(u: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """A first guess at the inverse CDF, good to about four decimals."""
    lower = u < _HALF
    tail = np.where(lower, u, 1.0 - u)
    t = np.sqrt(-2.0 * np.log(tail))
    a, b, c = _START_NUMERATOR
    d, e, f = _START_DENOMINATOR
    guess = t - (a + t * (b + t * c)) / (1.0 + t * (d + t * (e + t * f)))
    return np.where(lower, -guess, guess)


def _halley_step(x: npt.NDArray[np.float64], u: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """One Halley iteration towards ``norm_cdf(x) == u``."""
    error = norm_cdf(x) - u
    density = norm_pdf(x)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        step = error / density / (1.0 + x * error / (2.0 * density))
    return np.where(np.isfinite(step), x - step, x)
