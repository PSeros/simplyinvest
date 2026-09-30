"""How an uncertain number is distributed."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from .gaussian import norm_ppf, unit_interval

if TYPE_CHECKING:
    import numpy.typing as npt

__all__ = [
    "Constant",
    "Distribution",
    "Empirical",
    "LogNormal",
    "Normal",
    "Triangular",
    "Uniform",
]

LOW_QUANTILE = 0.1
"""The lower probability an interval is quoted at unless another is given."""

HIGH_QUANTILE = 0.9
"""The upper probability an interval is quoted at unless another is given."""

_Z_HIGH = float(norm_ppf(np.asarray(HIGH_QUANTILE)))


class Distribution(ABC):
    """A univariate distribution, sampled through its inverse CDF."""

    __slots__ = ()

    @abstractmethod
    def ppf(self, u: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """The value this falls below with probability ``u``."""

    @property
    @abstractmethod
    def expectation(self) -> float:
        """The expected value."""

    def sample(self, rng: np.random.Generator, n: int) -> npt.NDArray[np.float64]:
        """``n`` independent draws."""
        return self.ppf(unit_interval(rng.random(n)))

    def quantile(self, p: float) -> float:
        """The value this falls below with probability ``p``."""
        return float(self.ppf(unit_interval(np.asarray(p, dtype=np.float64))))

    def interval(
        self, *, low: float = LOW_QUANTILE, high: float = HIGH_QUANTILE
    ) -> tuple[float, float]:
        """The values bounding the span between two probabilities."""
        return self.quantile(low), self.quantile(high)

    def __str__(self) -> str:
        return repr(self)


@dataclass(frozen=True, slots=True)
class Constant(Distribution):
    """A number that turned out not to be uncertain after all.

    Args:
        value: The one value it takes.
    """

    value: float

    def ppf(self, u: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """``value``, shaped like ``u``."""
        return np.full(np.shape(u), float(self.value), dtype=np.float64)

    @property
    def expectation(self) -> float:
        """The value itself."""
        return float(self.value)


@dataclass(frozen=True, slots=True)
class Uniform(Distribution):
    """Every value between two bounds equally likely.

    Args:
        low: The smallest value.
        high: The largest value.

    Raises:
        ValueError: if ``high`` does not exceed ``low``.
    """

    low: float
    high: float

    def __post_init__(self) -> None:
        if self.high <= self.low:
            raise ValueError(f"a uniform needs high > low, got low={self.low} high={self.high}")

    def ppf(self, u: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """The inverse CDF at ``u``."""
        return self.low + (self.high - self.low) * np.asarray(u, dtype=np.float64)

    @property
    def expectation(self) -> float:
        """The midpoint."""
        return (self.low + self.high) / 2.0


@dataclass(frozen=True, slots=True)
class Triangular(Distribution):
    """A smallest, a likeliest and a largest value.

    Args:
        low: The smallest value.
        mode: The likeliest value.
        high: The largest value.

    Raises:
        ValueError: if the three are not in order, or the span is empty.
    """

    low: float
    mode: float
    high: float

    def __post_init__(self) -> None:
        if not self.low <= self.mode <= self.high:
            raise ValueError(
                f"a triangular needs low <= mode <= high, got low={self.low} "
                f"mode={self.mode} high={self.high}"
            )
        if self.high <= self.low:
            raise ValueError(f"a triangular needs high > low, got low={self.low} high={self.high}")

    def ppf(self, u: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """The inverse CDF at ``u``."""
        probabilities = np.asarray(u, dtype=np.float64)
        span = self.high - self.low
        turn = (self.mode - self.low) / span
        rising = self.low + np.sqrt(probabilities * span * (self.mode - self.low))
        falling = self.high - np.sqrt((1.0 - probabilities) * span * (self.high - self.mode))
        return np.where(probabilities < turn, rising, falling)

    @property
    def expectation(self) -> float:
        """The average of the three corners."""
        return (self.low + self.mode + self.high) / 3.0


@dataclass(frozen=True, slots=True)
class Normal(Distribution):
    """A symmetric spread about a central value.

    Args:
        mean: The central value.
        sd: The standard deviation.

    Raises:
        ValueError: if ``sd`` is not positive.
    """

    mean: float
    sd: float

    def __post_init__(self) -> None:
        if self.sd <= 0.0:
            raise ValueError(f"a standard deviation must be positive, got {self.sd!r}")

    @classmethod
    def from_quantiles(cls, *, low: float, high: float) -> Normal:
        """A normal whose 10th and 90th percentiles are ``low`` and ``high``."""
        if high <= low:
            raise ValueError(f"the upper quantile must exceed the lower, got {low} and {high}")
        return cls(mean=(low + high) / 2.0, sd=(high - low) / (2.0 * _Z_HIGH))

    def ppf(self, u: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """The inverse CDF at ``u``."""
        return self.mean + self.sd * norm_ppf(unit_interval(u))

    @property
    def expectation(self) -> float:
        """The central value."""
        return self.mean


class LogNormal(Distribution):
    """A strictly positive quantity, built from what is actually known about it.

    Its natural parameters describe the underlying normal rather than the
    distribution itself, so they are not reachable positionally.  Use
    :meth:`from_mean_sd`, :meth:`from_mean_cv`, :meth:`from_median_gsd`,
    :meth:`from_quantiles` or :meth:`from_log_params`.
    """

    __slots__ = ("log_mean", "log_sd")

    log_mean: float
    log_sd: float

    def __init__(self, *args: object, **kwargs: object) -> None:
        """Always raises; a log-normal is built by one of its named constructors.

        Raises:
            TypeError: always.
        """
        raise TypeError(
            "LogNormal's raw parameters describe the underlying normal, not the "
            "distribution, so LogNormal(12_000, 1_000) would centre on exp(12_000).  "
            "Build it by what you know: from_mean_sd, from_mean_cv, from_median_gsd, "
            "from_quantiles, or from_log_params if you really do have log-space parameters."
        )

    @classmethod
    def from_log_params(cls, *, log_mean: float, log_sd: float) -> LogNormal:
        """A log-normal whose underlying normal has these parameters.

        Raises:
            ValueError: if ``log_sd`` is not positive.
        """
        if log_sd <= 0.0:
            raise ValueError(f"a standard deviation must be positive, got {log_sd!r}")
        built = object.__new__(cls)
        built.log_mean = log_mean
        built.log_sd = log_sd
        return built

    @classmethod
    def from_mean_sd(cls, mean: float, sd: float) -> LogNormal:
        """A log-normal with this arithmetic mean and standard deviation.

        Raises:
            ValueError: if ``mean`` is not positive.
        """
        if mean <= 0.0:
            raise ValueError(f"a log-normal is strictly positive, got a mean of {mean!r}")
        variance = math.log1p((sd / mean) ** 2)
        return cls.from_log_params(
            log_mean=math.log(mean) - variance / 2.0, log_sd=math.sqrt(variance)
        )

    @classmethod
    def from_mean_cv(cls, mean: float, cv: float) -> LogNormal:
        """A log-normal with this mean and coefficient of variation."""
        return cls.from_mean_sd(mean, mean * cv)

    @classmethod
    def from_median_gsd(cls, median: float, gsd: float) -> LogNormal:
        """A log-normal with this median and geometric standard deviation.

        Raises:
            ValueError: if ``median`` is not positive or ``gsd`` is not above one.
        """
        if median <= 0.0:
            raise ValueError(f"a log-normal is strictly positive, got a median of {median!r}")
        if gsd <= 1.0:
            raise ValueError(f"a geometric standard deviation must exceed 1, got {gsd!r}")
        return cls.from_log_params(log_mean=math.log(median), log_sd=math.log(gsd))

    @classmethod
    def from_quantiles(cls, *, low: float, high: float) -> LogNormal:
        """A log-normal whose 10th and 90th percentiles are ``low`` and ``high``.

        Raises:
            ValueError: if either bound is not positive, or they are not ordered.
        """
        if low <= 0.0 or high <= 0.0:
            raise ValueError(f"a log-normal is strictly positive, got {low} and {high}")
        if high <= low:
            raise ValueError(f"the upper quantile must exceed the lower, got {low} and {high}")
        return cls.from_log_params(
            log_mean=(math.log(low) + math.log(high)) / 2.0,
            log_sd=(math.log(high) - math.log(low)) / (2.0 * _Z_HIGH),
        )

    def ppf(self, u: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """The inverse CDF at ``u``."""
        return np.exp(self.log_mean + self.log_sd * norm_ppf(unit_interval(u)))

    @property
    def expectation(self) -> float:
        """The arithmetic mean."""
        return math.exp(self.log_mean + self.log_sd**2 / 2.0)

    @property
    def median(self) -> float:
        """The value half the draws fall below."""
        return math.exp(self.log_mean)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, LogNormal):
            return NotImplemented
        return (self.log_mean, self.log_sd) == (other.log_mean, other.log_sd)

    def __hash__(self) -> int:
        return hash((type(self), self.log_mean, self.log_sd))

    def __repr__(self) -> str:
        return f"LogNormal.from_log_params(log_mean={self.log_mean!r}, log_sd={self.log_sd!r})"


@dataclass(frozen=True, slots=True, eq=False)
class Empirical(Distribution):
    """A sample standing in for the distribution it came from.

    Args:
        values: Observed values, in any order.

    Raises:
        ValueError: if the sample is empty or holds a non-finite value.
    """

    values: npt.NDArray[np.float64]

    def __post_init__(self) -> None:
        observed = np.asarray(self.values, dtype=np.float64)
        if observed.size == 0:
            raise ValueError("an empirical distribution needs at least one observation")
        if not np.all(np.isfinite(observed)):
            raise ValueError("an empirical distribution cannot hold a non-finite observation")

    def ppf(self, u: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """The sample quantile at ``u``, interpolated between observations."""
        return np.quantile(
            np.asarray(self.values, dtype=np.float64), np.asarray(u, dtype=np.float64)
        )

    @property
    def expectation(self) -> float:
        """The sample mean."""
        return float(np.mean(np.asarray(self.values, dtype=np.float64)))

    def __repr__(self) -> str:
        return f"Empirical(<{np.size(self.values)} observations>)"
