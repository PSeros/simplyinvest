"""Several courses of action, weighed against each other."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from simplyinvest.domain import Anonymous, ConstantAnnualUsage, Context
from simplyinvest.errors import UnequalLivesError

from .result import Appraisal, ComparisonResult

if TYPE_CHECKING:
    from simplyinvest.domain import Party, UsageProfile
    from simplyinvest.timeline import Timeline

    from .alternative import Evaluable

__all__ = ["Case", "compare"]

_DEFAULT_MATERIALITY = 0.01

_MIN_ALTERNATIVES = 2


@dataclass(frozen=True)
class Case:
    """A set of mutually exclusive alternatives and the circumstances they share.

    Args:
        alternatives: The courses of action, at least two and distinctly named.
        timeline: The grid, shared by all of them.
        party: Who is investing.
        usage: How hard the asset is used.
        materiality: How large the margin must be, as a share of the winner's
            value, for the ranking to mean anything.
        allow_unequal_lives: Whether to permit alternatives of different lives
            to be ranked on present value.

    Raises:
        ValueError: on fewer than two alternatives, or repeated names.
        UnequalLivesError: if lives differ and ``allow_unequal_lives`` is false.
    """

    alternatives: tuple[Evaluable, ...]
    timeline: Timeline
    party: Party = field(default_factory=Anonymous)
    usage: UsageProfile = field(default_factory=ConstantAnnualUsage)
    materiality: float = _DEFAULT_MATERIALITY
    allow_unequal_lives: bool = False

    def __post_init__(self) -> None:
        if len(self.alternatives) < _MIN_ALTERNATIVES:
            raise ValueError(
                f"a comparison needs at least two alternatives, got "
                f"{len(self.alternatives)}.  Use appraise() for a single course of action."
            )
        names = [alternative.name for alternative in self.alternatives]
        if len(set(names)) != len(names):
            raise ValueError(f"alternatives must have distinct names, got {names}")
        if not self.allow_unequal_lives:
            self._check_lives()

    def _check_lives(self) -> None:
        lives = {
            alternative.name: alternative.life
            for alternative in self.alternatives
            if alternative.life is not None
        }
        if len(set(lives.values())) > 1:
            stated = ", ".join(f"{name} {life}" for name, life in lives.items())
            raise UnequalLivesError(
                f"these alternatives do not last the same time ({stated}), so ranking them "
                f"on present value flatters the shortest.  Either equalise them with a "
                f"ReplacementChain, rank on equivalent annual cost, or pass "
                f"allow_unequal_lives=True having decided the difference does not matter."
            )

    @property
    def context(self) -> Context:
        """The circumstances every alternative is resolved against."""
        return Context(timeline=self.timeline, party=self.party, usage=self.usage)

    def run(self) -> ComparisonResult:
        """Resolve every alternative and rank them."""
        ctx = self.context
        appraisals = tuple(
            Appraisal(
                name=alternative.name,
                series=alternative.flows(ctx),
                ctx=ctx,
                life=alternative.life,
                constraints=alternative.constraints(ctx),
            )
            for alternative in self.alternatives
        )
        return ComparisonResult(appraisals=appraisals, materiality=self.materiality)


def compare(
    alternatives: Sequence[Evaluable],
    timeline: Timeline,
    *,
    party: Party | None = None,
    usage: UsageProfile | None = None,
    materiality: float = _DEFAULT_MATERIALITY,
    allow_unequal_lives: bool = False,
) -> ComparisonResult:
    """Rank mutually exclusive alternatives on one timeline.

    Args:
        alternatives: The courses of action, at least two and distinctly named.
        timeline: The grid, shared by all of them.
        party: Who is investing.
        usage: How hard the asset is used.
        materiality: How large the margin must be, as a share of the winner's
            value, for the ranking to mean anything.
        allow_unequal_lives: Whether to permit alternatives of different lives
            to be ranked on present value.
    """
    return Case(
        alternatives=tuple(alternatives),
        timeline=timeline,
        party=party or Anonymous(),
        usage=usage or ConstantAnnualUsage(),
        materiality=materiality,
        allow_unequal_lives=allow_unequal_lives,
    ).run()
