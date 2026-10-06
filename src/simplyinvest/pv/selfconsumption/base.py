"""Where the generated energy goes: used on site, or exported."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np

from simplyinvest.money import Quantity

if TYPE_CHECKING:
    import numpy.typing as npt

    from simplyinvest.pv.generation import Generation
    from simplyinvest.timeline import Timeline

__all__ = ["DeclaredShare", "SelfConsumption", "Split"]


@dataclass(frozen=True)
class Split:
    """Generated energy divided between the site and the grid.

    Args:
        generated: Kilowatt-hours produced in each period.
        self_consumed: Kilowatt-hours used on site in each period.
        exported: Kilowatt-hours fed into the grid in each period.
    """

    generated: npt.NDArray[np.float64]
    self_consumed: npt.NDArray[np.float64]
    exported: npt.NDArray[np.float64]

    @property
    def share(self) -> npt.NDArray[np.float64]:
        """The fraction used on site over the whole grid, per trial."""
        produced = self.generated.sum(axis=-1)
        used = self.self_consumed.sum(axis=-1)
        return np.asarray(np.divide(used, produced, out=np.zeros_like(used), where=produced > 0.0))


@runtime_checkable
class SelfConsumption(Protocol):
    """How much of what is generated is used on site rather than exported."""

    def split(self, generation: Generation, timeline: Timeline) -> Split:
        """Divide ``generation`` between the site and the grid, period by period."""
        ...

    def annual_share(self, generation: Generation) -> Quantity:
        """The fraction of a year's output used on site."""
        ...


@dataclass(frozen=True)
class DeclaredShare(SelfConsumption):
    """A self-consumption share the user states rather than simulates.

    The share is an annual average over the whole grid, not a physical
    constant: it falls as the system grows against the same household, and
    rises with a battery.  ``switch_point`` over it answers what share the
    system needs to break even, which is the question this form is for.

    Args:
        share: Fraction of generated energy used on site.

    Raises:
        ValueError: if the share is not a fraction.
    """

    share: Quantity = 0.3

    def __post_init__(self) -> None:
        values = np.asarray(self.share, dtype=np.float64)
        if np.any(values < 0.0) or np.any(values > 1.0):
            raise ValueError(
                f"a self-consumption share is a fraction between 0 and 1, got {self.share!r}"
            )

    def split(self, generation: Generation, timeline: Timeline) -> Split:
        """Divide every period's output in the stated proportion."""
        produced = generation.per_period(timeline)
        used = np.asarray(self.share, dtype=np.float64)[..., np.newaxis]
        return Split(
            generated=np.broadcast_to(produced, np.broadcast_shapes(produced.shape, used.shape)),
            self_consumed=produced * used,
            exported=produced * (1.0 - used),
        )

    def annual_share(self, generation: Generation) -> Quantity:
        """The stated fraction, which does not depend on the generation."""
        return self.share

    def __str__(self) -> str:
        return f"{float(np.max(np.asarray(self.share))):.0%} used on site, as stated"
