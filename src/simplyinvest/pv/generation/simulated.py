"""A yield worked out from the sun, the roof and a year of weather."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from simplyinvest._optional import require_extra
from simplyinvest.money import Quantity
from simplyinvest.pv.generation.base import (
    DEFAULT_DEGRADATION,
    Generation,
    degradation_factors,
    seasonal_shares,
)

if TYPE_CHECKING:
    import numpy.typing as npt

    from simplyinvest.pv.generation.geometry import Array, Site
    from simplyinvest.pv.weather.source import WeatherSource
    from simplyinvest.timeline import Timeline

__all__ = ["DEFAULT_SYSTEM_LOSSES", "DEFAULT_TEMPERATURE_COEFFICIENT", "SimulatedYield"]

#: Losses on top of what the chain already models.  Zero by default, because
#: pvlib's PVWatts path already takes the inverter's conversion curve and the
#: cells' temperature.  What is left for this field is soiling, snow, mismatch,
#: wiring and downtime -- real, site-specific, and nobody else's to assume.
#:
#: Left at zero, a 10 kWp array facing south at 30 degrees over Aachen comes to
#: about 950 kWh/kWp on a typical year.  PVGIS's own model puts a flat array
#: there at 941 and a tilted one at 1,126 for 2020, which was exceptionally
#: sunny, so that is the band to expect.
DEFAULT_SYSTEM_LOSSES = 0.0

#: How much output is lost per degree the cells run above 25 C.
DEFAULT_TEMPERATURE_COEFFICIENT = -0.004

#: An open rack of glass-on-glass modules, as pvlib parameterises it.
_THERMAL = {"a": -3.56, "b": -0.075, "deltaT": 3}

_WATTS_PER_KW = 1000.0
_MONTHS_IN_YEAR = 12


@dataclass(frozen=True)
class SimulatedYield(Generation):
    """Output computed from the roof's geometry and a representative year.

    The hourly run is reduced to an annual total and twelve monthly shares,
    which is all a cash-flow grid can carry: a period is a month at finest.
    The full year is kept for anything that needs the hours themselves.

    Args:
        site: Where the roof is.
        arrays: The planes of modules, at least one.
        weather: The year to run them against.
        degradation: Annual loss of output as the modules age.
        system_losses: Share lost between the modules and the meter.
        temperature_coefficient: Output lost per degree above 25 C.

    Raises:
        ValueError: on no arrays, or a loss or degradation outside ``[0, 1)``.
    """

    site: Site
    arrays: tuple[Array, ...]
    weather: WeatherSource
    degradation: float = DEFAULT_DEGRADATION
    system_losses: float = DEFAULT_SYSTEM_LOSSES
    temperature_coefficient: float = DEFAULT_TEMPERATURE_COEFFICIENT
    _run: dict[str, Any] = field(default_factory=dict, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not self.arrays:
            raise ValueError("a simulated yield needs at least one array of modules")
        for name in ("degradation", "system_losses"):
            value = getattr(self, name)
            if not 0.0 <= value < 1.0:
                raise ValueError(f"{name} is a fraction below one, got {value}")

    @property
    def peak_kw(self) -> float:
        """Nameplate power of every array together."""
        return float(sum(float(np.max(np.asarray(a.peak_kw))) for a in self.arrays))

    def hourly(self, timeline: Timeline) -> npt.NDArray[np.float64]:
        """Kilowatt-hours delivered in each hour of the representative year."""
        return np.asarray(self._simulate()["hourly"], dtype=np.float64)

    @property
    def first_year_kwh(self) -> Quantity:
        """What the roof makes in its first year, before it ages."""
        return float(self._simulate()["annual"])

    @property
    def monthly_share(self) -> tuple[float, ...]:
        """The share of a year's output falling in each month, summing to one."""
        monthly = np.asarray(self._simulate()["monthly"], dtype=np.float64)
        total = monthly.sum()
        if total <= 0.0:
            return (1.0 / _MONTHS_IN_YEAR,) * _MONTHS_IN_YEAR
        return tuple(float(share) for share in monthly / total)

    def per_period(self, timeline: Timeline) -> npt.NDArray[np.float64]:
        """Kilowatt-hours generated in each period, shape ``(n + 1,)``."""
        shape = seasonal_shares(timeline, self.monthly_share)
        ageing = degradation_factors(timeline, self.degradation)
        return np.asarray(self.first_year_kwh, dtype=np.float64) * shape * ageing

    def _simulate(self) -> dict[str, Any]:
        """Run the chain once, and keep what the grid can use.

        Raises:
            ImportError: naming the extra to install, if pvlib is missing.
        """
        if self._run:
            return self._run
        require_extra("pv", "pvlib", "pandas")
        delivered = self._alternating_current()
        monthly = delivered.groupby(delivered.index.month).sum()
        total = float(delivered.sum())
        self._run.update(
            hourly=delivered.to_numpy(),
            annual=total,
            monthly=tuple(
                float(monthly.get(month, 0.0)) for month in range(1, _MONTHS_IN_YEAR + 1)
            ),
        )
        return self._run

    def _alternating_current(self) -> Any:
        """Hourly kilowatt-hours at the meter, across every array."""
        from pvlib.location import Location  # noqa: PLC0415
        from pvlib.modelchain import ModelChain  # noqa: PLC0415
        from pvlib.pvsystem import Array as PvArray  # noqa: PLC0415
        from pvlib.pvsystem import FixedMount, PVSystem  # noqa: PLC0415

        where = Location(
            self.site.latitude,
            self.site.longitude,
            tz=self.site.timezone,
            altitude=self.site.altitude,
            name=self.site.name,
        )
        weather = self.weather.hourly()
        planes = [
            PvArray(
                FixedMount(surface_tilt=plane.tilt, surface_azimuth=plane.azimuth),
                module_parameters={
                    "pdc0": float(np.max(np.asarray(plane.peak_kw))) * _WATTS_PER_KW,
                    "gamma_pdc": self.temperature_coefficient,
                },
                temperature_model_parameters=_THERMAL,
                name=plane.name,
            )
            for plane in self.arrays
        ]
        system = PVSystem(
            arrays=planes,
            inverter_parameters={"pdc0": self.peak_kw * _WATTS_PER_KW},
        )
        chain = ModelChain.with_pvwatts(
            system, where, aoi_model="physical", spectral_model="no_loss"
        )
        chain.run_model(weather)
        kept = 1.0 - self.system_losses
        return chain.results.ac / _WATTS_PER_KW * kept

    def __str__(self) -> str:
        return f"{self.peak_kw:.1f} kWp simulated against {self.weather}"
