"""Executes every example notebook."""

from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]

NOTEBOOKS = sorted((ROOT / "examples").glob("*.ipynb"))

TIMEOUT = 300


def test_the_examples_are_notebooks() -> None:
    """A renamed or moved example must not leave the check passing vacuously."""
    assert NOTEBOOKS, f"no notebooks found under {ROOT / 'examples'}"


@pytest.mark.notebook
@pytest.mark.parametrize("path", NOTEBOOKS, ids=[path.name for path in NOTEBOOKS])
def test_an_example_notebook_runs(path: Path) -> None:
    """Every cell executes against the current package."""
    nbformat = pytest.importorskip("nbformat")
    nbclient = pytest.importorskip("nbclient")
    pytest.importorskip("matplotlib")

    notebook = nbformat.read(path, as_version=4)
    client = nbclient.NotebookClient(
        notebook,
        timeout=TIMEOUT,
        kernel_name="python3",
        resources={"metadata": {"path": str(ROOT)}},
    )
    try:
        client.execute()
    except nbclient.exceptions.CellExecutionError as error:
        pytest.fail(f"{path.name} — a cell raised:\n{error}")
