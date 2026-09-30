"""Resolving courses of action into numbers, and reading the numbers back."""

from __future__ import annotations

from .alternative import Alternative, Evaluable
from .appraise import appraise
from .case import Case, compare
from .chain import ReplacementChain
from .result import Appraisal, ComparisonResult, Incremental

__all__ = [
    "Alternative",
    "Appraisal",
    "Case",
    "ComparisonResult",
    "Evaluable",
    "Incremental",
    "ReplacementChain",
    "appraise",
    "compare",
]
