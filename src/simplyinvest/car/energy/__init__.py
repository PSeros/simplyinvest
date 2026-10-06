"""How a vehicle is fuelled, and what that costs."""

from __future__ import annotations

from .base import DISTANCE_UNIT, EnergySource, MeteredSource
from .bivalent import Bivalent
from .electricity import ELECTRICITY, Electricity
from .fuels import DIESEL, HYDROGEN, LPG, LPG_CARRIER, PETROL, Diesel, Hydrogen, Petrol

__all__ = [
    "DIESEL",
    "DISTANCE_UNIT",
    "ELECTRICITY",
    "HYDROGEN",
    "LPG",
    "LPG_CARRIER",
    "PETROL",
    "Bivalent",
    "Diesel",
    "Electricity",
    "EnergySource",
    "Hydrogen",
    "MeteredSource",
    "Petrol",
]
