"""Photovoltaics: what a roof makes, what it saves, and what the state pays for it.

Two facts drive the appraisal, and each can be stated or simulated::

    uv add simplyinvest             # StatedYield + DeclaredShare
    uv add "simplyinvest[pv]"       # SimulatedYield + DispatchedLoad

The stated pair needs nothing beyond the core: a yield in kilowatt-hours and a
self-consumption share.  The simulated pair reads weather and a load profile.
"""

from __future__ import annotations

from .generation import (
    DEFAULT_DEGRADATION,
    DEFAULT_SYSTEM_LOSSES,
    DEFAULT_TEMPERATURE_COEFFICIENT,
    EAST,
    MONTHLY_SHARE,
    SOUTH,
    WEST,
    Array,
    Generation,
    SimulatedYield,
    Site,
    StatedYield,
)
from .operations import DEFAULT_COST_ESCALATION, PvOperations
from .party import Householder, PvOperator
from .price import DEFAULT_PRICE_ESCALATION, RetailPrice, StaticPrice
from .selfconsumption import DeclaredShare, SelfConsumption, Split
from .supply import EXPORTED, GENERATED, SELF_CONSUMED, PvSupply
from .system import Battery, Inverter, System
from .tax import RESIDENTIAL_EXEMPTION_KW, ResidentialExempt
from .weather import (
    CACHE_VARIABLE,
    HOURS_IN_YEAR,
    PVGIS_ATTRIBUTION,
    ClearSkyYear,
    GivenWeather,
    PvgisTmy,
    WeatherSource,
    cache_root,
    cached_frame,
)

__all__ = [
    "CACHE_VARIABLE",
    "DEFAULT_COST_ESCALATION",
    "DEFAULT_DEGRADATION",
    "DEFAULT_PRICE_ESCALATION",
    "DEFAULT_SYSTEM_LOSSES",
    "DEFAULT_TEMPERATURE_COEFFICIENT",
    "EAST",
    "EXPORTED",
    "GENERATED",
    "HOURS_IN_YEAR",
    "MONTHLY_SHARE",
    "PVGIS_ATTRIBUTION",
    "RESIDENTIAL_EXEMPTION_KW",
    "SELF_CONSUMED",
    "SOUTH",
    "WEST",
    "Array",
    "Battery",
    "ClearSkyYear",
    "DeclaredShare",
    "Generation",
    "GivenWeather",
    "Householder",
    "Inverter",
    "PvOperations",
    "PvOperator",
    "PvSupply",
    "PvgisTmy",
    "ResidentialExempt",
    "RetailPrice",
    "SelfConsumption",
    "SimulatedYield",
    "Site",
    "Split",
    "StatedYield",
    "StaticPrice",
    "System",
    "WeatherSource",
    "cache_root",
    "cached_frame",
]
