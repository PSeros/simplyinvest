"""Where a year of weather comes from."""

from __future__ import annotations

from .cache import CACHE_VARIABLE, cache_root, cached_frame
from .pvgis import PVGIS_ATTRIBUTION, PvgisTmy
from .source import HOURS_IN_YEAR, ClearSkyYear, GivenWeather, WeatherSource

__all__ = [
    "CACHE_VARIABLE",
    "HOURS_IN_YEAR",
    "PVGIS_ATTRIBUTION",
    "ClearSkyYear",
    "GivenWeather",
    "PvgisTmy",
    "WeatherSource",
    "cache_root",
    "cached_frame",
]
