"""What an appraisal and a comparison come to."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from decimal import ROUND_HALF_UP, Decimal
from functools import cached_property
from typing import TYPE_CHECKING, Any

import numpy as np

from simplyinvest.cashflow import Component, Role
from simplyinvest.errors import UnknownQuantityError
from simplyinvest.metrics import (
    IRR,
    discounted_payback,
    equivalent_annual_cost,
    irr,
    payback,
    profitability_index,
    pv_of_outflows,
)

if TYPE_CHECKING:
    import numpy.typing as npt

    from simplyinvest.cashflow import CashFlowSeries
    from simplyinvest.domain import Context
    from simplyinvest.timeline import Term

__all__ = ["Appraisal", "ComparisonResult", "Incremental"]

_DEFAULT_MATERIALITY = 0.01


def _money(value: float) -> str:
    """``value`` rounded to two decimals for display."""
    return f"{Decimal(str(value)).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP):,}"


@dataclass(frozen=True)
class Appraisal:
    """One course of action, resolved against one context.

    Args:
        name: What to call this course of action.
        series: Its cash flows.
        ctx: The circumstances it was resolved against.
        life: How long it lasts, where that differs from the horizon.
        constraints: Non-cash obligations it carries.
    """

    name: str
    series: CashFlowSeries
    ctx: Context
    life: Term | None = None
    constraints: tuple[str, ...] = ()
    _cache: dict[str, Any] = field(default_factory=dict, repr=False, compare=False)

    @cached_property
    def amounts(self) -> npt.NDArray[np.float64]:
        """Signed amounts per period."""
        return self.series.amounts(self.ctx.timeline)

    @cached_property
    def npv(self) -> Any:
        """Net present value."""
        return self.series.pv(self.ctx.timeline)

    @cached_property
    def pv_of_costs(self) -> Any:
        """Present value of the money going out, as a positive number."""
        return pv_of_outflows(self.amounts, self.ctx.timeline)

    @cached_property
    def eac(self) -> Any:
        """Equivalent annual cost, over this alternative's own life."""
        return equivalent_annual_cost(self.pv_of_costs, self.ctx.timeline, self.life)

    @cached_property
    def irr(self) -> IRR:
        """The rate at which this breaks even, where one exists."""
        if np.ndim(self.amounts) > 1:
            return IRR(None, "a batch of trials has a distribution of rates, not one rate")
        return irr(self.amounts, self.ctx.timeline)

    @cached_property
    def payback(self) -> Term | None:
        """How long until the undiscounted total recovers."""
        return payback(self.amounts, self.ctx.timeline)

    @cached_property
    def discounted_payback(self) -> Term | None:
        """How long until the discounted total recovers."""
        return discounted_payback(self.amounts, self.ctx.timeline)

    @cached_property
    def profitability_index(self) -> Any:
        """Present value in per unit of present value out."""
        return profitability_index(self.amounts, self.ctx.timeline)

    def cost_per_unit(self, quantity: str) -> float:
        """Equivalent annual cost per unit of ``quantity`` used each year.

        Raises:
            UnknownQuantityError: if the usage profile does not carry it, or
                carries it as zero.
        """
        annual = self.ctx.usage.annual(quantity)
        if annual == 0.0:
            raise UnknownQuantityError(
                f"cost per {quantity} needs a non-zero annual {quantity}, got {annual}"
            )
        return float(self.eac) / annual

    def breakdown(self, *, by: type[Component] | type[Role] = Component) -> dict[Any, Any]:
        """Present value per label, which sums to :attr:`npv`."""
        return self.series.breakdown(self.ctx.timeline, by=by)

    def to_markdown(self) -> str:
        """A table of this appraisal's measures and its component breakdown."""
        lines = [
            f"### {self.name}",
            "",
            "| Measure | Value |",
            "| --- | ---: |",
            f"| Net present value | {_money(float(self.npv))} |",
            f"| Present value of costs | {_money(float(self.pv_of_costs))} |",
            f"| Equivalent annual cost | {_money(float(self.eac))} |",
            f"| Internal rate of return | {self.irr} |",
            f"| Payback | {self.payback or 'never within the horizon'} |",
            f"| Discounted payback | {self.discounted_payback or 'never within the horizon'} |",
            "",
            "| Component | Present value |",
            "| --- | ---: |",
        ]
        parts = self.breakdown()
        lines += [f"| {label} | {_money(float(value))} |" for label, value in parts.items()]
        lines.append(f"| **Total** | **{_money(float(sum(parts.values())))}** |")
        if self.constraints:
            lines += ["", "**Obligations this does not price:**", ""]
            lines += [f"- {note}" for note in self.constraints]
        return "\n".join(lines)

    def __str__(self) -> str:
        return f"{self.name}: NPV {_money(float(self.npv))}"


@dataclass(frozen=True)
class Incremental:
    """The difference between two alternatives.

    Args:
        better: Name of the alternative being switched to.
        worse: Name of the alternative being switched from.
        amounts: The differential signed amounts per period.
        ctx: The circumstances both were resolved against.
    """

    better: str
    worse: str
    amounts: npt.NDArray[np.float64]
    ctx: Context

    @cached_property
    def npv(self) -> float:
        """Present value of switching from ``worse`` to ``better``."""
        return float(self.ctx.timeline.pv(self.amounts))

    @cached_property
    def irr(self) -> IRR:
        """The rate the switch earns, where one exists."""
        return irr(self.amounts, self.ctx.timeline)

    @cached_property
    def payback(self) -> Term | None:
        """How long the switch takes to repay itself."""
        return payback(self.amounts, self.ctx.timeline)

    def __str__(self) -> str:
        return f"{self.better} over {self.worse}: {_money(self.npv)}"


@dataclass(frozen=True)
class ComparisonResult:
    """Several alternatives, ranked by net present value.

    Args:
        appraisals: The resolved alternatives, in the order given.
        materiality: How large the margin must be, as a share of the winner's
            value, for the ranking to mean anything.
    """

    appraisals: tuple[Appraisal, ...]
    materiality: float = _DEFAULT_MATERIALITY

    def __iter__(self) -> Iterator[Appraisal]:
        return iter(self.appraisals)

    def __len__(self) -> int:
        return len(self.appraisals)

    def __getitem__(self, name: str) -> Appraisal:
        for appraisal in self.appraisals:
            if appraisal.name == name:
                return appraisal
        raise KeyError(f"no alternative named {name!r}; this case has {self.names}")

    @property
    def names(self) -> tuple[str, ...]:
        """Every alternative's name, in the order given."""
        return tuple(appraisal.name for appraisal in self.appraisals)

    def ranking(self) -> tuple[Appraisal, ...]:
        """Every alternative, highest net present value first."""
        return tuple(sorted(self.appraisals, key=lambda a: float(a.npv), reverse=True))

    def best(self) -> Appraisal:
        """The alternative with the highest net present value."""
        return self.ranking()[0]

    def margin(self) -> float:
        """How far the winner leads the runner-up, in present value."""
        ranked = self.ranking()
        return float(ranked[0].npv) - float(ranked[1].npv)

    def is_material(self) -> bool:
        """Whether the margin exceeds :attr:`materiality` of the winner's value."""
        best = abs(float(self.best().npv))
        return self.margin() > self.materiality * max(best, 1.0)

    def incremental(self, better: str, worse: str) -> Incremental:
        """The differential stream between two named alternatives."""
        return Incremental(
            better=better,
            worse=worse,
            amounts=self[better].amounts - self[worse].amounts,
            ctx=self[better].ctx,
        )

    @property
    def constraints(self) -> tuple[str, ...]:
        """Every obligation any alternative carries, with its owner named."""
        return tuple(
            f"{appraisal.name}: {note}"
            for appraisal in self.appraisals
            for note in appraisal.constraints
        )

    def verdict(self) -> str:
        """One sentence stating the result, or that it is a tie."""
        ranked = self.ranking()
        if not self.is_material():
            return (
                f"{ranked[0].name} and {ranked[1].name} are within "
                f"{_money(self.margin())} of each other, which is inside the margin of "
                f"error on inputs like these.  Choose on the obligations, not the number."
            )
        return (
            f"{ranked[0].name} is better than {ranked[1].name} by "
            f"{_money(self.margin())} in present value."
        )

    def to_markdown(self) -> str:
        """A ranked table, the verdict, and every obligation."""
        lines = [
            "| Alternative | Net present value | Equivalent annual cost | Payback |",
            "| --- | ---: | ---: | ---: |",
        ]
        for appraisal in self.ranking():
            lines.append(
                f"| {appraisal.name} | {_money(float(appraisal.npv))} "
                f"| {_money(float(appraisal.eac))} | {appraisal.payback or '—'} |"
            )
        lines += ["", self.verdict()]
        if self.constraints:
            lines += ["", "**Obligations these figures do not price:**", ""]
            lines += [f"- {note}" for note in self.constraints]
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.verdict()
