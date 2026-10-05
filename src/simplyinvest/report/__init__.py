"""Charts and frames over the objects an appraisal returns.

Neither module is imported with the package: ``matplotlib`` and ``pandas``
arrive with the ``viz`` and ``frames`` extras.

    from simplyinvest.report import breakdown_chart, ranking_frame
"""

from __future__ import annotations

from .charts import (
    balance_chart,
    breakdown_chart,
    cumulative_chart,
    differential_chart,
    distribution_chart,
    money_formatter,
    scenario_chart,
    schedule_chart,
    sweep_chart,
    tornado_chart,
)
from .frames import (
    breakdown_frame,
    draws_frame,
    flows_frame,
    ranking_frame,
    scenario_frame,
    schedule_frame,
    simulation_frame,
    sweep_frame,
    tornado_frame,
)

__all__ = [
    "balance_chart",
    "breakdown_chart",
    "breakdown_frame",
    "cumulative_chart",
    "differential_chart",
    "distribution_chart",
    "draws_frame",
    "flows_frame",
    "money_formatter",
    "ranking_frame",
    "scenario_chart",
    "scenario_frame",
    "schedule_chart",
    "schedule_frame",
    "simulation_frame",
    "sweep_chart",
    "sweep_frame",
    "tornado_chart",
    "tornado_frame",
]
