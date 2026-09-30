"""Moving one parameter at a time and watching the answer."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal
from typing import TYPE_CHECKING

import numpy as np

from simplyinvest.errors import ParameterError

from .distribution import HIGH_QUANTILE, LOW_QUANTILE
from .index import ParameterIndex
from .lens import Lens

if TYPE_CHECKING:
    import numpy.typing as npt

    from simplyinvest.appraisal import Case

    from .distribution import Distribution

__all__ = ["Bar", "Sweep", "Switch", "Tornado", "one_way", "switch_point", "tornado"]

MAX_BISECTIONS = 80
"""How many times a switch point is bracketed before the search gives up."""

SWITCH_TOLERANCE = 1e-9
"""How close to zero a differential present value counts as a switch."""


def _money(value: float) -> str:
    """``value`` rounded to two decimals for display."""
    return f"{Decimal(str(value)).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP):,}"


def _npv_at(case: Case, address: Lens, value: float) -> dict[str, float]:
    """Each alternative's present value with one parameter held at ``value``."""
    result = address.set(case, value).run()
    return {name: float(result[name].npv) for name in result.names}


@dataclass(frozen=True)
class Sweep:
    """One parameter walked across a range.

    Args:
        label: The parameter that moved.
        values: What it was set to, in order.
        names: The alternatives, in the order the case gave them.
        npv: Net present value, shape ``(alternatives, points)``.
    """

    label: str
    values: npt.NDArray[np.float64]
    names: tuple[str, ...]
    npv: npt.NDArray[np.float64]

    def winner_at(self, position: int) -> str:
        """The best alternative at one point of the sweep."""
        return self.names[int(np.argmax(self.npv[:, position]))]

    def winners(self) -> tuple[str, ...]:
        """The best alternative at each point, in order."""
        return tuple(self.winner_at(position) for position in range(len(self.values)))

    def is_decisive(self) -> bool:
        """Whether one alternative wins across the whole range."""
        return len(set(self.winners())) == 1

    def to_markdown(self) -> str:
        """A table of present value against the parameter."""
        header = f"| {self.label} | " + " | ".join(self.names) + " |"
        rule = "| --- |" + " ---: |" * len(self.names)
        rows = [
            f"| {value:,.6g} | "
            + " | ".join(_money(float(self.npv[row, position])) for row in range(len(self.names)))
            + " |"
            for position, value in enumerate(self.values)
        ]
        return "\n".join([header, rule, *rows])

    def __str__(self) -> str:
        if self.is_decisive():
            return f"{self.winner_at(0)} wins across every value of {self.label}"
        return f"the winner changes as {self.label} moves"


def one_way(case: Case, label: str, values: Sequence[float]) -> Sweep:
    """Resolve ``case`` once per value of one parameter.

    Raises:
        ParameterError: if the model has no such parameter.
        ValueError: if no values are given.
    """
    if len(values) == 0:
        raise ValueError(f"a sweep of {label!r} needs at least one value")
    address = ParameterIndex.of(case)[label]
    points = np.asarray(values, dtype=np.float64)
    columns = [_npv_at(case, address, float(value)) for value in points]
    names = tuple(columns[0])
    npv = np.array([[column[name] for column in columns] for name in names], dtype=np.float64)
    return Sweep(label=label, values=points, names=names, npv=npv)


@dataclass(frozen=True, slots=True, order=True)
class Bar:
    """How far one parameter moves the answer.

    Args:
        label: The parameter.
        low: Present value at its low end.
        high: Present value at its high end.
    """

    label: str
    low: float
    high: float

    @property
    def swing(self) -> float:
        """How much present value separates the two ends."""
        return abs(self.high - self.low)


@dataclass(frozen=True)
class Tornado:
    """Every parameter's swing, widest first.

    Args:
        on: The alternative whose present value was measured.
        base: Its present value with every parameter at its declared value.
        bars: One bar per parameter, widest swing first.
    """

    on: str
    base: float
    bars: tuple[Bar, ...]

    def widest(self) -> Bar:
        """The parameter the answer depends on most."""
        return self.bars[0]

    def to_markdown(self) -> str:
        """A table of each parameter's low, high and swing."""
        lines = [
            f"Net present value of **{self.on}** is {_money(self.base)} at the declared values.",
            "",
            "| Parameter | Low | High | Swing |",
            "| --- | ---: | ---: | ---: |",
        ]
        lines += [
            f"| {bar.label} | {_money(bar.low)} | {_money(bar.high)} | {_money(bar.swing)} |"
            for bar in self.bars
        ]
        return "\n".join(lines)

    def __str__(self) -> str:
        return f"{self.on} turns most on {self.widest().label}"


def tornado(
    case: Case,
    *,
    on: str | None = None,
    spec: Mapping[str, Distribution] | None = None,
    low: float = LOW_QUANTILE,
    high: float = HIGH_QUANTILE,
) -> Tornado:
    """How far each parameter alone moves one alternative's present value.

    Args:
        case: The alternatives to weigh, carrying marked parameters.
        on: Which alternative to measure.  Defaults to the first.
        spec: Distributions by label, overriding any declared prior.
        low: The probability the low end of each parameter is taken at.
        high: The probability the high end is taken at.

    Raises:
        ParameterError: if nothing is marked uncertain, or a parameter has no
            distribution.
    """
    index = ParameterIndex.of(case)
    if not index:
        raise ParameterError(
            "nothing in this case is marked uncertain, so no parameter can be swung.  "
            "Wrap a number in uncertain(value, label, prior)."
        )
    distributions = index.distributions(spec)
    measured = on if on is not None else case.alternatives[0].name
    base = float(case.run()[measured].npv)

    bars = [
        Bar(
            label=label,
            low=_npv_at(case, index[label], distributions[label].quantile(low))[measured],
            high=_npv_at(case, index[label], distributions[label].quantile(high))[measured],
        )
        for label in index.labels
    ]
    bars.sort(key=lambda bar: bar.swing, reverse=True)
    return Tornado(on=measured, base=base, bars=tuple(bars))


@dataclass(frozen=True, slots=True)
class Switch:
    """The value of one parameter at which the ranking of two alternatives turns.

    Args:
        label: The parameter that was moved.
        value: The value at which the two are worth the same.
        below: Which alternative leads below it.
        above: Which alternative leads above it.
    """

    label: str
    value: float
    below: str
    above: str

    def __str__(self) -> str:
        return (
            f"{self.below} leads while {self.label} is below {self.value:,.2f}, "
            f"{self.above} above it"
        )


def switch_point(
    case: Case, label: str, *, better: str, worse: str, bracket: tuple[float, float]
) -> Switch | None:
    """Where moving one parameter turns the ranking of two alternatives.

    Args:
        case: The alternatives to weigh.
        label: The parameter to move.
        better: One of the two alternatives.
        worse: The other.
        bracket: The range to search, as ``(low, high)``.

    Returns:
        A :class:`Switch` naming which alternative leads on each side, or
        ``None`` when one leads across the whole range.

    Raises:
        ParameterError: if the model has no such parameter.
        ValueError: if the bracket is empty.
    """
    start, end = bracket
    if end <= start:
        raise ValueError(f"a bracket runs from low to high, got {bracket!r}")
    address = ParameterIndex.of(case)[label]

    def margin(value: float) -> float:
        found = _npv_at(case, address, value)
        return found[better] - found[worse]

    low_margin, high_margin = margin(start), margin(end)
    if low_margin * high_margin > 0.0:
        return None

    leads_low, leads_high = (better, worse) if low_margin > 0.0 else (worse, better)
    found = _bisect(margin, start, end, low_margin, high_margin)
    return Switch(label=label, value=found, below=leads_low, above=leads_high)


def _bisect(
    margin: Callable[[float], float], start: float, end: float, at_start: float, at_end: float
) -> float:
    """The value between ``start`` and ``end`` where ``margin`` crosses zero."""
    if at_start == 0.0:
        return start
    if at_end == 0.0:
        return end
    for _ in range(MAX_BISECTIONS):
        middle = (start + end) / 2.0
        middle_margin = margin(middle)
        if abs(middle_margin) < SWITCH_TOLERANCE or end - start < SWITCH_TOLERANCE:
            return middle
        if at_start * middle_margin < 0.0:
            end = middle
        else:
            start, at_start = middle, middle_margin
    return (start + end) / 2.0
