"""The circumstances a flow source builds against."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from .party import Anonymous, Party
from .usage import ConstantAnnualUsage, UsageProfile

if TYPE_CHECKING:
    import numpy.typing as npt

    from simplyinvest.timeline import Term, Timeline

__all__ = ["Context"]


@dataclass(frozen=True)
class Context:
    """The grid, the investor, the usage profile, and the window in play.

    Args:
        timeline: The period grid.
        party: Who is investing.
        usage: How hard the asset is used.
        start: First period of this window.
        end: Last period of this window.  Defaults to the horizon.
    """

    timeline: Timeline
    party: Party = field(default_factory=Anonymous)
    usage: UsageProfile = field(default_factory=ConstantAnnualUsage)
    start: int = 0
    end: int | None = None

    @property
    def last(self) -> int:
        """The final period of this window."""
        return self.timeline.n_periods if self.end is None else min(self.end, self.timeline.last)

    @property
    def mask(self) -> npt.NDArray[np.float64]:
        """One in each period this window covers, zero elsewhere.

        Periods are covered in arrears, so the window opening at ``start``
        first carries value at ``start + 1``.
        """
        covered = np.zeros(self.timeline.n_periods + 1, dtype=np.float64)
        covered[self.start + 1 : self.last + 1] = 1.0
        return covered

    @property
    def held(self) -> Term:
        """How long this window runs."""
        return self.timeline.term_of(self.last - self.start)

    def window(self, start: int, end: int | None = None) -> Context:
        """The same context over a different stretch of the grid."""
        return Context(
            timeline=self.timeline,
            party=self.party,
            usage=self.usage,
            start=start,
            end=end,
        )
