"""A bundle of labelled flows and the present value they come to."""

from __future__ import annotations

import math
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from .base import CashFlow, Component, Role

if TYPE_CHECKING:
    import numpy.typing as npt

    from simplyinvest.timeline import Timeline

__all__ = ["CashFlowSeries"]

_BREAKDOWN_TOLERANCE = 1e-9


@dataclass(frozen=True)
class CashFlowSeries:
    """Labelled flows that together describe one course of action."""

    flows: tuple[CashFlow, ...] = ()

    @classmethod
    def of(cls, *flows: CashFlow) -> CashFlowSeries:
        """A series of the flows given."""
        return cls(tuple(flows))

    @classmethod
    def concat(cls, series: Iterable[CashFlowSeries]) -> CashFlowSeries:
        """One series holding every flow of several."""
        return cls(tuple(flow for part in series for flow in part.flows))

    def __add__(self, other: CashFlowSeries) -> CashFlowSeries:
        if not isinstance(other, CashFlowSeries):
            return NotImplemented
        return CashFlowSeries(self.flows + other.flows)

    def __iter__(self) -> Iterator[CashFlow]:
        return iter(self.flows)

    def __len__(self) -> int:
        return len(self.flows)

    def __bool__(self) -> bool:
        return bool(self.flows)

    def labelled(self, *, role: Role | None = None, detail: str | None = None) -> CashFlowSeries:
        """The flows matching a role, a detail, or both."""
        return CashFlowSeries(
            tuple(
                flow
                for flow in self.flows
                if (role is None or flow.label.role is role)
                and (detail is None or flow.label.detail == detail)
            )
        )

    def amounts(self, timeline: Timeline) -> npt.NDArray[np.float64]:
        """Signed amounts per period, summed over every flow."""
        total: npt.NDArray[np.float64] | None = None
        for flow in self.flows:
            values = flow.amounts(timeline)
            total = values if total is None else total + values
        return timeline.zeros() if total is None else total

    def pv(self, timeline: Timeline) -> Any:
        """Present value of the whole series."""
        return timeline.pv(self.amounts(timeline))

    def undiscounted(self, timeline: Timeline) -> Any:
        """The plain sum of every amount."""
        return self.amounts(timeline).sum(axis=-1)

    def breakdown(
        self, timeline: Timeline, *, by: type[Component] | type[Role] = Component
    ) -> dict[Any, Any]:
        """Present value per label, which sums to :meth:`pv`.

        Args:
            timeline: The grid to resolve against.
            by: ``Component`` for the full label, ``Role`` to roll details up.

        Raises:
            AssertionError: if the parts do not sum to the total.
        """
        parts: dict[Any, Any] = {}
        for flow in self.flows:
            key = flow.label if by is Component else flow.label.role
            value = timeline.pv(flow.amounts(timeline))
            parts[key] = parts.get(key, 0.0) + value

        total = self.pv(timeline)
        summed = sum(parts.values())
        if np.ndim(total) == 0:
            assert math.isclose(
                float(summed), float(total), rel_tol=_BREAKDOWN_TOLERANCE, abs_tol=1e-6
            ), f"breakdown sums to {summed!r} but the series is worth {total!r}"
        return dict(sorted(parts.items(), key=lambda item: str(item[0])))

    def detail(self, timeline: Timeline) -> tuple[tuple[str, Component, float], ...]:
        """Every flow's description, label and present value."""
        return tuple(
            (flow.description, flow.label, timeline.pv(flow.amounts(timeline)))
            for flow in self.flows
        )

    def __str__(self) -> str:
        return f"{len(self.flows)} flow{'s' if len(self.flows) != 1 else ''}"
