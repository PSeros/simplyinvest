"""Resolving one course of action against one set of circumstances."""

from __future__ import annotations

from typing import TYPE_CHECKING

from simplyinvest.domain import Anonymous, ConstantAnnualUsage, Context

from .result import Appraisal

if TYPE_CHECKING:
    from simplyinvest.domain import Party, UsageProfile
    from simplyinvest.timeline import Timeline

    from .alternative import Evaluable

__all__ = ["appraise"]


def appraise(
    alternative: Evaluable,
    timeline: Timeline,
    *,
    party: Party | None = None,
    usage: UsageProfile | None = None,
) -> Appraisal:
    """Work out what ``alternative`` comes to on ``timeline``.

    Args:
        alternative: The course of action.
        timeline: The period grid, carrying the discount rate and conventions.
        party: Who is investing.  Defaults to an anonymous investor.
        usage: How hard the asset is used.  Defaults to no usage.
    """
    ctx = Context(
        timeline=timeline,
        party=party or Anonymous(),
        usage=usage or ConstantAnnualUsage(),
    )
    return Appraisal(
        name=alternative.name,
        series=alternative.flows(ctx),
        ctx=ctx,
        life=alternative.life,
        constraints=alternative.constraints(ctx),
    )
