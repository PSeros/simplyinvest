"""Named sets of parameter values, run side by side."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal
from typing import TYPE_CHECKING

import numpy as np

from .index import ParameterIndex

if TYPE_CHECKING:
    import numpy.typing as npt

    from simplyinvest.appraisal import Case

__all__ = ["Scenario", "ScenarioTable", "run_scenarios"]


def _money(value: float) -> str:
    """``value`` rounded to two decimals for display."""
    return f"{Decimal(str(value)).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP):,}"


@dataclass(frozen=True)
class Scenario:
    """A named set of values for some of a model's parameters.

    Args:
        name: What to call this state of the world.
        values: The value each named parameter takes in it.
    """

    name: str
    values: Mapping[str, float]

    def __str__(self) -> str:
        return self.name


@dataclass(frozen=True)
class ScenarioTable:
    """Every alternative's present value in every scenario.

    Args:
        scenarios: The states of the world, in the order given.
        names: The alternatives, in the order the case gave them.
        npv: Net present value, shape ``(scenarios, alternatives)``.
    """

    scenarios: tuple[str, ...]
    names: tuple[str, ...]
    npv: npt.NDArray[np.float64]

    def winner_in(self, scenario: str) -> str:
        """The best alternative in one scenario.

        Raises:
            KeyError: if no scenario goes by that name.
        """
        if scenario not in self.scenarios:
            raise KeyError(f"no scenario named {scenario!r}; this table holds {self.scenarios}")
        return self.names[int(np.argmax(self.npv[self.scenarios.index(scenario)]))]

    def winners(self) -> dict[str, str]:
        """The best alternative in each scenario."""
        return {scenario: self.winner_in(scenario) for scenario in self.scenarios}

    def is_robust(self) -> bool:
        """Whether one alternative wins in every scenario."""
        return len(set(self.winners().values())) == 1

    def verdict(self) -> str:
        """One sentence stating whether the choice survives every scenario."""
        won = self.winners()
        if self.is_robust():
            return f"{next(iter(won.values()))} wins in all {len(self.scenarios)} scenarios."
        split = ", ".join(f"{scenario}: {name}" for scenario, name in won.items())
        return f"The choice depends on which scenario holds — {split}."

    def to_markdown(self) -> str:
        """A table of present value by scenario, and the verdict."""
        lines = [
            "| Scenario | " + " | ".join(self.names) + " | Best |",
            "| --- |" + " ---: |" * len(self.names) + " --- |",
        ]
        for row, scenario in enumerate(self.scenarios):
            values = " | ".join(_money(float(value)) for value in self.npv[row])
            lines.append(f"| {scenario} | {values} | {self.winner_in(scenario)} |")
        lines += ["", self.verdict()]
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.verdict()


def run_scenarios(case: Case, scenarios: Sequence[Scenario]) -> ScenarioTable:
    """Resolve ``case`` once per scenario.

    Raises:
        ParameterError: if a scenario names a parameter the model does not have.
        ValueError: if no scenarios are given.
    """
    if len(scenarios) == 0:
        raise ValueError("there is nothing to compare without at least one scenario")
    index = ParameterIndex.of(case)
    rows = []
    for scenario in scenarios:
        result = index.apply(case, scenario.values).run()
        rows.append([float(result[name].npv) for name in result.names])
    names = tuple(case.run().names)
    return ScenarioTable(
        scenarios=tuple(scenario.name for scenario in scenarios),
        names=names,
        npv=np.asarray(rows, dtype=np.float64),
    )
