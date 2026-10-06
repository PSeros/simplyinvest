"""The carriers sold by volume or by mass."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from simplyinvest.money import Quantity

from .base import MeteredSource

__all__ = ["DIESEL", "HYDROGEN", "LPG", "LPG_CARRIER", "PETROL", "Diesel", "Hydrogen", "Petrol"]

#: The carriers sold at a pump.
PETROL = "petrol"
DIESEL = "diesel"
HYDROGEN = "hydrogen"
LPG_CARRIER = "lpg"


@dataclass(frozen=True)
class Petrol(MeteredSource):
    """Petrol, litres per 100 km."""

    unit: ClassVar[str] = "l"
    carrier: ClassVar[str] = PETROL


@dataclass(frozen=True)
class Diesel(MeteredSource):
    """Diesel, litres per 100 km."""

    unit: ClassVar[str] = "l"
    carrier: ClassVar[str] = DIESEL


@dataclass(frozen=True)
class Hydrogen(MeteredSource):
    """Hydrogen, kilograms per 100 km."""

    unit: ClassVar[str] = "kg"
    carrier: ClassVar[str] = HYDROGEN


@dataclass(frozen=True)
class LPG(MeteredSource):
    """Autogas, litres per 100 km.

    Args:
        volumetric_penalty: Extra litres burned per litre of the equivalent
            petrol figure.  Leave at one when ``consumption`` is already
            stated in litres of LPG.

    Raises:
        ValueError: if the penalty is not positive.
    """

    volumetric_penalty: float = 1.0

    unit: ClassVar[str] = "l"
    carrier: ClassVar[str] = LPG_CARRIER

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.volumetric_penalty <= 0.0:
            raise ValueError(
                f"volumetric_penalty scales consumption, so it must be positive, "
                f"got {self.volumetric_penalty!r}"
            )

    @property
    def effective_consumption(self) -> Quantity:
        """Litres per 100 km after the real-world uplift and the volumetric penalty."""
        return self.consumption * self.real_world_factor * self.volumetric_penalty
