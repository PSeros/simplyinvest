"""Shared fixtures and options."""

from __future__ import annotations

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register the golden-snapshot rewrite flag."""
    parser.addoption(
        "--update-golden",
        action="store_true",
        default=False,
        help="rewrite the frozen numeric snapshots under tests/golden/expected",
    )


@pytest.fixture(scope="session")
def pandas():
    """pandas, or a skip when the `frames` extra is not installed."""
    return pytest.importorskip("pandas")


@pytest.fixture(scope="session")
def matplotlib():
    """matplotlib on a headless backend, or a skip when `viz` is not installed."""
    module = pytest.importorskip("matplotlib")
    module.use("Agg")
    pytest.importorskip("matplotlib.pyplot")
    return module


@pytest.fixture(scope="session")
def pvlib():
    """pvlib, or a skip when the `pv` extra is not installed."""
    return pytest.importorskip("pvlib")
