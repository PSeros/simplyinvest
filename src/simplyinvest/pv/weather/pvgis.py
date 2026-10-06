"""A typical year from PVGIS, the Commission's solar radiation service."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from simplyinvest._optional import require_extra
from simplyinvest.pv.weather.cache import cached_frame
from simplyinvest.pv.weather.source import WeatherSource, check_columns

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from simplyinvest.pv.generation.geometry import Site

__all__ = ["PVGIS_ATTRIBUTION", "PvgisTmy"]

#: What PVGIS asks to be said of figures taken from it.
PVGIS_ATTRIBUTION = "PVGIS, European Commission Joint Research Centre"

_DEFAULT_TIMEOUT = 60


@dataclass(frozen=True)
class PvgisTmy(WeatherSource):
    """The typical meteorological year PVGIS builds for a location.

    A typical year is stitched together from the most representative month of
    each calendar month across many observed years.  It describes the shape of
    a year at this site, not any year that happened.

    Args:
        site: Where to ask about.
        root: Where to keep the download.  The user cache directory by default.
        timeout: Seconds to wait on the service.

    Raises:
        ImportError: naming the extra to install, if pvlib is missing.
    """

    site: Site
    root: Path | None = None
    timeout: int = _DEFAULT_TIMEOUT

    def hourly(self) -> Any:
        """One typical year, downloaded once and kept.

        Raises:
            ValueError: if the service returns something a simulation cannot use.
        """
        require_extra("pv", "pvlib", "pandas")
        request = {
            "service": "pvgis-tmy",
            "latitude": round(self.site.latitude, 5),
            "longitude": round(self.site.longitude, 5),
        }
        frame = cached_frame(request, self._fetch, root=self.root)
        return check_columns(frame, self.provenance)

    def _fetch(self) -> tuple[Any, Mapping[str, Any]]:
        """Ask PVGIS, and keep what it says about the answer."""
        import pvlib  # noqa: PLC0415

        data, meta = pvlib.iotools.get_pvgis_tmy(
            self.site.latitude,
            self.site.longitude,
            map_variables=True,
            timeout=self.timeout,
        )
        inputs = meta.get("inputs", {}) if isinstance(meta, dict) else {}
        return data, {"attribution": PVGIS_ATTRIBUTION, "inputs": inputs}

    @property
    def provenance(self) -> str:
        """Where the figures came from."""
        return f"a typical year for {self.site.name} from {PVGIS_ATTRIBUTION}"

    def __str__(self) -> str:
        return self.provenance
