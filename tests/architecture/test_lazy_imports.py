"""Importing the core stays cheap, and says so when an extra is missing."""

from __future__ import annotations

import subprocess
import sys

import pytest

from simplyinvest._optional import require_extra

#: Packages an extra may pull in, which importing the core must not.
HEAVY = ("pandas", "matplotlib", "scipy", "pvlib", "openpyxl")


def test_importing_the_core_pulls_in_nothing_heavy() -> None:
    """A user who only wants an NPV should not pay for a plotting library."""
    code = (
        "import sys, simplyinvest;"
        f"loaded=[m for m in {HEAVY!r} if m in sys.modules];"
        "print(','.join(loaded))"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "", f"importing simplyinvest loaded {out.stdout.strip()}"


def test_the_core_needs_only_numpy() -> None:
    """numpy is the one runtime dependency, and it earns it: every present value
    in the package is a dot product."""
    code = "import sys, simplyinvest; print('numpy' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "True"


class TestMissingExtra:
    def test_an_installed_extra_passes_quietly(self) -> None:
        require_extra("core", "numpy")

    def test_a_missing_one_names_the_install_command(self) -> None:
        with pytest.raises(ImportError, match=r'uv add "simplyinvest\[pv\]"'):
            require_extra("pv", "a_module_that_does_not_exist")

    def test_it_says_which_dependency_is_missing(self) -> None:
        with pytest.raises(ImportError, match="a_module_that_does_not_exist"):
            require_extra("pv", "a_module_that_does_not_exist")
