"""Where a year of weather comes from."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from simplyinvest._optional import require_extra

if TYPE_CHECKING:
    from simplyinvest.pv.generation.geometry import Site

__all__ = ["HOURS_IN_YEAR", "ClearSkyYear", "GivenWeather", "WeatherSource"]

#: Rows a representative year carries, one per hour.
HOURS_IN_YEAR = 8760

#: Columns a yield simulation reads.
REQUIRED_COLUMNS = ("ghi", "dni", "dhi", "temp_air", "wind_speed")


@runtime_checkable
class WeatherSource(Protocol):
    """One representative year of weather, hour by hour."""

    def hourly(self) -> Any:
        """A frame of :data:`REQUIRED_COLUMNS`, indexed by hour."""
        ...

    @property
    def provenance(self) -> str:
        """Where the figures came from, carried into every report."""
        ...


def check_columns(frame: Any, provenance: str) -> Any:
    """``frame`` if it carries what a simulation needs.

    Raises:
        ValueError: naming the columns that are missing.
    """
    missing = [name for name in REQUIRED_COLUMNS if name not in frame.columns]
    if missing:
        raise ValueError(
            f"the weather from {provenance} is missing {missing}; a yield simulation "
            f"needs {list(REQUIRED_COLUMNS)}."
        )
    return frame


@dataclass(frozen=True)
class GivenWeather(WeatherSource):
    """A year someone else measured, modelled or downloaded.

    Args:
        frame: Hourly weather carrying :data:`REQUIRED_COLUMNS`.
        provenance: Where it came from.
    """

    frame: Any
    provenance: str = "a frame supplied by hand"

    def hourly(self) -> Any:
        """The frame as given.

        Raises:
            ValueError: if it does not carry what a simulation needs.
        """
        return check_columns(self.frame, self.provenance)

    def __str__(self) -> str:
        return self.provenance


@dataclass(frozen=True)
class ClearSkyYear(WeatherSource):
    """A year with no clouds at all, computed from the sun's position.

    This is a ceiling, not a forecast: a real German year yields roughly
    two-thirds of it.  It needs no network, which makes it the fallback when a
    download is unavailable and the right input for a test that must not vary.

    Args:
        site: Where the sun is being tracked from.
        year: The calendar year to lay the hours on.
        temperature: Air temperature held constant, in degrees Celsius.
        wind_speed: Wind speed held constant, in metres per second.
    """

    site: Site
    year: int = 2026
    temperature: float = 12.0
    wind_speed: float = 2.5

    def hourly(self) -> Any:
        """One cloudless year.

        Raises:
            ImportError: naming the extra to install, if pvlib is missing.
        """
        require_extra("pv", "pvlib", "pandas")
        import pandas as pd  # noqa: PLC0415
        from pvlib.location import Location  # noqa: PLC0415

        where = Location(
            self.site.latitude,
            self.site.longitude,
            tz=self.site.timezone,
            altitude=self.site.altitude,
        )
        when = pd.date_range(
            f"{self.year}-01-01", periods=HOURS_IN_YEAR, freq="h", tz=self.site.timezone
        )
        frame = where.get_clearsky(when)
        frame["temp_air"] = self.temperature
        frame["wind_speed"] = self.wind_speed
        return check_columns(frame, self.provenance)

    @property
    def provenance(self) -> str:
        """Where the figures came from."""
        return f"a cloudless {self.year} over {self.site.name}, computed from the sun"

    def __str__(self) -> str:
        return self.provenance
