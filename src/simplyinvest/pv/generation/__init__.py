"""How much the system makes, and the roof it makes it from."""

from __future__ import annotations

from .base import DEFAULT_DEGRADATION, MONTHLY_SHARE, Generation, StatedYield
from .geometry import EAST, SOUTH, WEST, Array, Site
from .simulated import (
    DEFAULT_SYSTEM_LOSSES,
    DEFAULT_TEMPERATURE_COEFFICIENT,
    SimulatedYield,
)

__all__ = [
    "DEFAULT_DEGRADATION",
    "DEFAULT_SYSTEM_LOSSES",
    "DEFAULT_TEMPERATURE_COEFFICIENT",
    "EAST",
    "MONTHLY_SHARE",
    "SOUTH",
    "WEST",
    "Array",
    "Generation",
    "SimulatedYield",
    "Site",
    "StatedYield",
]
