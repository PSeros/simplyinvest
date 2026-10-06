"""The names the package offers, and the promise that they are documented."""

from __future__ import annotations

import importlib

import pytest

import simplyinvest

#: Every scope that publishes a surface of its own.
SCOPES = (
    "appraisal",
    "car",
    "cashflow",
    "domain",
    "errors",
    "financing",
    "incentives",
    "metrics",
    "money",
    "pv",
    "report",
    "tax",
    "timeline",
    "uncertain",
)


def test_every_exported_name_exists() -> None:
    missing = [name for name in simplyinvest.__all__ if not hasattr(simplyinvest, name)]
    assert not missing, f"__all__ promises names the package does not have: {missing}"


def test_all_is_sorted() -> None:
    """So a reader can find a name, and a diff shows a real change."""
    assert list(simplyinvest.__all__) == sorted(simplyinvest.__all__)


@pytest.mark.parametrize("scope", SCOPES)
def test_every_scope_export_exists(scope: str) -> None:
    module = importlib.import_module(f"simplyinvest.{scope}")
    missing = [name for name in module.__all__ if not hasattr(module, name)]
    assert not missing, f"simplyinvest.{scope}.__all__ promises {missing}"


@pytest.mark.parametrize("scope", SCOPES)
def test_every_scope_export_is_documented(scope: str) -> None:
    module = importlib.import_module(f"simplyinvest.{scope}")
    undocumented = [
        name
        for name in module.__all__
        if (obj := getattr(module, name)) is not None
        and callable(obj)
        and not getattr(obj, "__doc__", None)
    ]
    assert not undocumented, f"simplyinvest.{scope} exports undocumented names: {undocumented}"


def test_the_package_itself_is_documented() -> None:
    assert simplyinvest.__doc__
    assert simplyinvest.__version__
