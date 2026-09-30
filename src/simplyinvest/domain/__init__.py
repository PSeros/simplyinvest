"""The contract a domain package implements to plug into the engine."""

from __future__ import annotations

from .asset import Asset
from .context import Context
from .party import Anonymous, Party, require
from .residual import (
    FirstYearDropThenGeometric,
    GeometricDecline,
    ResidualValueModel,
    TabulatedResiduals,
    terminal_value,
)
from .source import FlowSource
from .usage import ConstantAnnualUsage, ProfileUsage, UsageProfile

__all__ = [
    "Anonymous",
    "Asset",
    "ConstantAnnualUsage",
    "Context",
    "FirstYearDropThenGeometric",
    "FlowSource",
    "GeometricDecline",
    "Party",
    "ProfileUsage",
    "ResidualValueModel",
    "TabulatedResiduals",
    "UsageProfile",
    "require",
    "terminal_value",
]
