"""Monetary amounts and the direction they move in."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

import numpy as np

from simplyinvest.errors import SignConventionError

if TYPE_CHECKING:
    import numpy.typing as npt

__all__ = [
    "Amount",
    "Direction",
    "Quantity",
]


type Quantity = float | np.float64 | npt.NDArray[np.float64]
"""A scalar, or an array of trial draws with the trials on the leading axes."""


class Direction(StrEnum):
    """Which way money moves, from the investor's point of view."""

    OUT = "out"
    IN = "in"

    @property
    def sign(self) -> float:
        """``-1.0`` for an outflow, ``+1.0`` for an inflow."""
        return -1.0 if self is Direction.OUT else 1.0


@dataclass(frozen=True, slots=True)
class Amount:
    """A non-negative magnitude together with a direction.

    Args:
        magnitude: A non-negative size, scalar or an array of trial draws.
        direction: Whether the money leaves or arrives.

    Raises:
        SignConventionError: if ``magnitude`` is negative or non-finite.
    """

    magnitude: Quantity
    direction: Direction

    def __post_init__(self) -> None:
        values = np.asarray(self.magnitude, dtype=np.float64)
        if not np.all(np.isfinite(values)):
            raise SignConventionError(f"an Amount must be finite, got {self.magnitude!r}")
        if np.any(values < 0.0):
            worst = float(values.min())
            raise SignConventionError(
                f"an Amount carries a magnitude and a direction, but the magnitude "
                f"{worst} is negative.  Use Amount.received({abs(worst)}) for money "
                f"coming in, or Amount.net(...) when the direction is itself data."
            )

    @classmethod
    def paid(cls, magnitude: Quantity) -> Amount:
        """An outflow of ``magnitude``."""
        return cls(magnitude, Direction.OUT)

    @classmethod
    def received(cls, magnitude: Quantity) -> Amount:
        """An inflow of ``magnitude``."""
        return cls(magnitude, Direction.IN)

    @classmethod
    def net(cls, signed: Quantity) -> Amount:
        """An amount whose direction is taken from the sign of ``signed``.

        Raises:
            SignConventionError: if a batch of draws spans zero.
        """
        values = np.asarray(signed, dtype=np.float64)
        if values.ndim == 0:
            return (
                cls(float(-values), Direction.OUT)
                if values < 0.0
                else cls(float(values), Direction.IN)
            )
        if np.all(values <= 0.0):
            return cls(-values, Direction.OUT)
        if np.all(values >= 0.0):
            return cls(values, Direction.IN)
        raise SignConventionError(
            "Amount.net was given a batch that spans zero, so no single direction "
            "describes it.  Split it into an inflow and an outflow, or express it "
            "as a signed per-period vector."
        )

    @classmethod
    def zero(cls) -> Amount:
        """An amount of nothing."""
        return cls(0.0, Direction.IN)

    @property
    def signed(self) -> Quantity:
        """The magnitude with its sign applied: negative out, positive in."""
        if isinstance(self.magnitude, np.ndarray):
            return self.direction.sign * self.magnitude
        return self.direction.sign * float(self.magnitude)

    @property
    def is_scalar(self) -> bool:
        """Whether this is a single number rather than a batch of trial draws."""
        return not isinstance(self.magnitude, np.ndarray)

    def scaled(self, factor: Quantity) -> Amount:
        """This amount multiplied by a non-negative ``factor``, direction kept."""
        return Amount(np.asarray(self.magnitude) * np.asarray(factor), self.direction)

    def reversed(self) -> Amount:
        """The same magnitude moving the other way."""
        other = Direction.IN if self.direction is Direction.OUT else Direction.OUT
        return Amount(self.magnitude, other)

    def __str__(self) -> str:
        arrow = "-" if self.direction is Direction.OUT else "+"
        if self.is_scalar:
            return f"{arrow}{float(self.magnitude):,.2f}"
        return f"{arrow}<{np.asarray(self.magnitude).shape} draws>"
