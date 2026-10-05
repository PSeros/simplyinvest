"""Domain-agnostic investment appraisal.

Models an investment as a stream of timed cash flows, discounts it, and ranks
alternatives on the result.  Domain packages supply the flows and install as
extras::

    uv add simplyinvest                # generic investments
    uv add "simplyinvest[car]"         # vehicles
    uv add "simplyinvest[pv]"          # photovoltaics
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

from .appraisal import (
    Alternative,
    Appraisal,
    Case,
    ComparisonResult,
    ReplacementChain,
    appraise,
    compare,
)
from .cashflow import (
    CashFlow,
    CashFlowSeries,
    Component,
    Explicit,
    Frequency,
    OneOff,
    Recurring,
    Role,
    Terminal,
)
from .domain import (
    Anonymous,
    Asset,
    ConstantAnnualUsage,
    Context,
    FlowSource,
    GeometricDecline,
    Party,
    ProfileUsage,
    ResidualValueModel,
    UsageProfile,
)
from .errors import SimplyInvestError, SimplyInvestWarning
from .money import Amount, Direction, Quantity
from .tax import TaxTreatment, Untaxed
from .timeline import (
    Escalation,
    EscalationSet,
    Periodisation,
    RateBasis,
    Term,
    Timeline,
    fisher_nominal,
    fisher_real,
)

if TYPE_CHECKING:
    from types import ModuleType

__version__ = "0.1.0"

_LAZY = frozenset({"car", "report"})
"""Scopes whose dependencies are optional, imported on first use."""


def __getattr__(name: str) -> ModuleType:
    """Import a scope with optional dependencies the first time it is reached.

    Raises:
        AttributeError: if ``name`` is not such a scope.
    """
    if name in _LAZY:
        return importlib.import_module(f"{__name__}.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "Alternative",
    "Amount",
    "Anonymous",
    "Appraisal",
    "Asset",
    "Case",
    "CashFlow",
    "CashFlowSeries",
    "ComparisonResult",
    "Component",
    "ConstantAnnualUsage",
    "Context",
    "Direction",
    "Escalation",
    "EscalationSet",
    "Explicit",
    "FlowSource",
    "Frequency",
    "GeometricDecline",
    "OneOff",
    "Party",
    "Periodisation",
    "ProfileUsage",
    "Quantity",
    "RateBasis",
    "Recurring",
    "ReplacementChain",
    "ResidualValueModel",
    "Role",
    "SimplyInvestError",
    "SimplyInvestWarning",
    "TaxTreatment",
    "Term",
    "Terminal",
    "Timeline",
    "Untaxed",
    "UsageProfile",
    "__version__",
    "appraise",
    "compare",
    "fisher_nominal",
    "fisher_real",
]
