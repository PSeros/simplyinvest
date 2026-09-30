"""Vehicles: what they cost to buy, to run, and what the state pays towards them.

Supplies the distance quantity ``"km"`` and the flow sources that turn it into
cash::

    uv add "simplyinvest[car]"
"""

from __future__ import annotations

from .energy import (
    LPG,
    Bivalent,
    Diesel,
    Electricity,
    EnergySource,
    Hydrogen,
    MeteredSource,
    Petrol,
)
from .incentives import (
    PREMIUM_2026_BEV,
    PREMIUM_2026_PHEV,
    CirculationTaxExemption,
    GhgQuota,
    PurchasePremium,
)
from .lease import MileageLease
from .operations import CarOperating
from .party import CarBuyer, Household
from .usage import KM, Mileage
from .vehicle import Propulsion, Vehicle, VehicleCategory

__all__ = [
    "KM",
    "LPG",
    "PREMIUM_2026_BEV",
    "PREMIUM_2026_PHEV",
    "Bivalent",
    "CarBuyer",
    "CarOperating",
    "CirculationTaxExemption",
    "Diesel",
    "Electricity",
    "EnergySource",
    "GhgQuota",
    "Household",
    "Hydrogen",
    "MeteredSource",
    "Mileage",
    "MileageLease",
    "Petrol",
    "Propulsion",
    "PurchasePremium",
    "Vehicle",
    "VehicleCategory",
]
