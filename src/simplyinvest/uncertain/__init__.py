"""Uncertainty: declaring it, drawing from it, and reading the spread back.

A number is marked where it is written::

    price = uncertain(38_900, "price", LogNormal.from_mean_cv(38_900, 0.06))

and a simulation finds every mark, draws it, and resolves the case once per
trial.  Anything not marked is reached instead with a lens over a ``ref`` path.
"""

from __future__ import annotations

from .distribution import (
    Constant,
    Distribution,
    Empirical,
    LogNormal,
    Normal,
    Triangular,
    Uniform,
)
from .index import ParameterIndex
from .lens import Attribute, Item, Lens, Step, lens, ref
from .mark import Uncertain, uncertain
from .sampler import correlation_matrix, uniforms
from .scenario import Scenario, ScenarioTable, run_scenarios
from .simulate import Mode, Simulation, simulate
from .sweep import Bar, Sweep, Switch, Tornado, one_way, switch_point, tornado

__all__ = [
    "Attribute",
    "Bar",
    "Constant",
    "Distribution",
    "Empirical",
    "Item",
    "Lens",
    "LogNormal",
    "Mode",
    "Normal",
    "ParameterIndex",
    "Scenario",
    "ScenarioTable",
    "Simulation",
    "Step",
    "Sweep",
    "Switch",
    "Tornado",
    "Triangular",
    "Uncertain",
    "Uniform",
    "correlation_matrix",
    "lens",
    "one_way",
    "ref",
    "run_scenarios",
    "simulate",
    "switch_point",
    "tornado",
    "uncertain",
    "uniforms",
]
