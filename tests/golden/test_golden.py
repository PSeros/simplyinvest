"""Numeric regressions, frozen to six decimal places.

Regenerate deliberately and read the diff::

    uv run pytest tests/golden --update-golden

A separate test refuses that flag under CI, so a snapshot can never be rewritten
by a run nobody was watching.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from cases import CASES, SCENARIOS

from simplyinvest.uncertain import run_scenarios, simulate, tornado

EXPECTED = Path(__file__).parent / "expected"
DECIMALS = 6


def pytest_addoption(parser):  # pragma: no cover - registered in conftest
    """Declared in tests/conftest.py; kept here for discoverability."""


def snapshot(name: str) -> dict:
    """Every number a case produces, rounded to a comparable precision."""
    result = CASES[name]().run()
    return {
        "verdict": result.verdict(),
        "material": result.is_material(),
        "margin": round(result.margin(), DECIMALS),
        "alternatives": {
            appraisal.name: {
                "npv": round(float(appraisal.npv), DECIMALS),
                "pv_of_costs": round(float(appraisal.pv_of_costs), DECIMALS),
                "eac": round(float(appraisal.eac), DECIMALS),
                "payback": str(appraisal.payback),
                "discounted_payback": str(appraisal.discounted_payback),
                "irr": None
                if appraisal.irr.value is None
                else round(appraisal.irr.value, DECIMALS),
                "breakdown": {
                    str(label): round(float(value), DECIMALS)
                    for label, value in appraisal.breakdown().items()
                },
            }
            for appraisal in result.ranking()
        },
        "constraints": list(result.constraints),
    }


@pytest.mark.golden
@pytest.mark.parametrize("name", sorted(CASES))
def test_the_numbers_have_not_moved(name: str, request: pytest.FixtureRequest) -> None:
    produced = snapshot(name)
    path = EXPECTED / f"{name}.json"

    if request.config.getoption("--update-golden"):
        path.write_text(json.dumps(produced, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        pytest.skip(f"rewrote {path.name}")

    assert path.exists(), (
        f"no snapshot for {name!r}.  Create it with: uv run pytest tests/golden --update-golden"
    )
    stored = json.loads(path.read_text(encoding="utf-8"))
    assert produced == stored, (
        f"{name} has changed.  If the change is intended, review the diff and rerun with "
        f"--update-golden."
    )


@pytest.mark.golden
@pytest.mark.parametrize("name", sorted(CASES))
def test_each_breakdown_still_sums_to_its_total(name: str) -> None:
    """Pinning the parts is worth nothing if they stop making up the whole."""
    for alternative in snapshot(name)["alternatives"].values():
        assert sum(alternative["breakdown"].values()) == pytest.approx(alternative["npv"], abs=1e-5)


def test_snapshots_cannot_be_rewritten_unattended() -> None:
    """A golden test that quietly updates itself is not a golden test."""
    if os.environ.get("CI"):
        assert "--update-golden" not in " ".join(os.sys.argv), (
            "--update-golden must never run in CI; a snapshot would be overwritten "
            "without anyone reading the diff."
        )


def uncertainty_snapshot(name: str) -> dict:
    """Every number a scenario table and a tornado produce for a marked case."""
    build, scenarios = SCENARIOS[name]
    case = build()
    table = run_scenarios(case, scenarios)
    return {
        "scenarios": {
            scenario: {
                alternative: round(float(table.npv[row, column]), DECIMALS)
                for column, alternative in enumerate(table.names)
            }
            for row, scenario in enumerate(table.scenarios)
        },
        "winners": table.winners(),
        "robust": table.is_robust(),
        "tornado": {
            alternative: [
                {
                    "label": bar.label,
                    "low": round(bar.low, DECIMALS),
                    "high": round(bar.high, DECIMALS),
                }
                for bar in tornado(case, on=alternative).bars
            ]
            for alternative in table.names
        },
    }


@pytest.mark.golden
@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_an_uncertain_case_has_not_moved(name: str, request: pytest.FixtureRequest) -> None:
    """Scenarios and tornados draw nothing, so they pin the chain without a seed."""
    produced = uncertainty_snapshot(name)
    path = EXPECTED / f"{name}.json"

    if request.config.getoption("--update-golden"):
        path.write_text(json.dumps(produced, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        pytest.skip(f"rewrote {path.name}")

    assert path.exists(), (
        f"no snapshot for {name!r}.  Create it with: uv run pytest tests/golden --update-golden"
    )
    assert produced == json.loads(path.read_text(encoding="utf-8")), (
        f"{name} has changed.  If the change is intended, review the diff and rerun with "
        f"--update-golden."
    )


@pytest.mark.golden
@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_a_seeded_simulation_repeats_itself(name: str) -> None:
    """A seed has to mean the same run twice, or nothing above can be reproduced."""
    build, _ = SCENARIOS[name]
    first = simulate(build(), n=500, seed=20260930)
    second = simulate(build(), n=500, seed=20260930)
    assert first.npv == pytest.approx(second.npv)
