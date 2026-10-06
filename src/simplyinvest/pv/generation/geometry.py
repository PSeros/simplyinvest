"""Where the modules stand and which way they face.

Nothing here reaches the appraisal: the money and the statute key on nameplate
power alone.  These describe the roof for a simulated yield.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from simplyinvest.money import Quantity

__all__ = ["EAST", "SOUTH", "WEST", "Array", "Site"]

#: Compass bearing of due south, on the convention pvlib uses.
SOUTH = 180.0

#: Compass bearing of due east.
EAST = 90.0

#: Compass bearing of due west.
WEST = 270.0

#: Limits the geometry is checked against.
_MAX_LATITUDE = 90.0
_MAX_LONGITUDE = 180.0
_MAX_TILT = 90.0
_FULL_CIRCLE = 360.0


@dataclass(frozen=True)
class Site:
    """Where the system stands.

    Args:
        latitude: Degrees north of the equator.
        longitude: Degrees east of Greenwich.
        altitude: Metres above sea level.
        timezone: An IANA zone name.
        name: What to call the place.

    Raises:
        ValueError: if the coordinates are off the globe.
    """

    latitude: float
    longitude: float
    altitude: float = 0.0
    timezone: str = "Europe/Berlin"
    name: str = "site"

    def __post_init__(self) -> None:
        if not -_MAX_LATITUDE <= self.latitude <= _MAX_LATITUDE:
            raise ValueError(f"latitude runs from -90 to 90, got {self.latitude}")
        if not -_MAX_LONGITUDE <= self.longitude <= _MAX_LONGITUDE:
            raise ValueError(f"longitude runs from -180 to 180, got {self.longitude}")

    def __str__(self) -> str:
        return self.name


@dataclass(frozen=True)
class Array:
    """One plane of modules, all at the same tilt and bearing.

    Args:
        peak_kw: Nameplate direct-current power, in kilowatts.
        tilt: Degrees from horizontal.
        azimuth: Compass bearing the modules face, north being zero.
        name: What to call this plane.

    Raises:
        ValueError: on a negative power, or a tilt or bearing off the compass.
    """

    peak_kw: Quantity
    tilt: float = 30.0
    azimuth: float = SOUTH
    name: str = "array"

    def __post_init__(self) -> None:
        if np.any(np.asarray(self.peak_kw, dtype=np.float64) < 0.0):
            raise ValueError(f"peak power cannot be negative, got {self.peak_kw!r}")
        if not 0.0 <= self.tilt <= _MAX_TILT:
            raise ValueError(f"tilt runs from 0 to 90 degrees, got {self.tilt}")
        if not 0.0 <= self.azimuth < _FULL_CIRCLE:
            raise ValueError(f"azimuth runs from 0 to 360 degrees, got {self.azimuth}")

    def __str__(self) -> str:
        return f"{self.name} ({float(np.max(self.peak_kw)):.2f} kWp at {self.azimuth:.0f}deg)"
