"""Exceptions and warnings raised by the package."""

from __future__ import annotations

__all__ = [
    "AnchorlessResamplingWarning",
    "CashFlowError",
    "DoubleCountingWarning",
    "EligibilityError",
    "InconsistentRateBasisError",
    "NoCalendarAnchorError",
    "NotAnchoredError",
    "NotSweepableError",
    "ParameterError",
    "PartyFactsMissingError",
    "SignConventionError",
    "SimplyInvestError",
    "SimplyInvestWarning",
    "TermNotRepresentableError",
    "TimelineError",
    "UnequalLivesError",
    "UnknownQuantityError",
]


class SimplyInvestError(Exception):
    """Base class for every error this package raises."""


class SimplyInvestWarning(UserWarning):
    """Base class for every warning this package issues."""


class SignConventionError(SimplyInvestError, ValueError):
    """An amount was given a sign instead of a direction."""


class TimelineError(SimplyInvestError, ValueError):
    """The period grid cannot represent what was asked of it."""


class InconsistentRateBasisError(TimelineError):
    """Real and nominal quantities were mixed."""


class TermNotRepresentableError(TimelineError):
    """A duration does not span a whole number of periods."""


class NotAnchoredError(TimelineError):
    """A calendar question was asked of a timeline with no start date."""


class NoCalendarAnchorError(TimelineError):
    """A start date was given for a grid that does not divide into whole months."""


class CashFlowError(SimplyInvestError, ValueError):
    """A cash flow cannot be resolved against the given grid."""


class UnknownQuantityError(SimplyInvestError, KeyError):
    """A usage profile does not carry the requested quantity."""


class PartyFactsMissingError(SimplyInvestError, TypeError):
    """The investor lacks facts a domain rule requires."""


class UnequalLivesError(SimplyInvestError, ValueError):
    """Alternatives of differing useful lives were ranked on present value."""


class EligibilityError(SimplyInvestError, ValueError):
    """An incentive was applied to something it cannot apply to."""


class ParameterError(SimplyInvestError, ValueError):
    """A parameter cannot be addressed in the model tree."""


class NotSweepableError(ParameterError):
    """A parameter cannot hold a draw."""


class DoubleCountingWarning(SimplyInvestWarning):
    """A benefit appears to be counted twice."""


class AnchorlessResamplingWarning(SimplyInvestWarning):
    """A seasonal profile was resampled onto a grid with no calendar."""
