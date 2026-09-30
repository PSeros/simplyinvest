"""The core holds no jurisdiction's rules, rates or dates."""

from __future__ import annotations

import ast

from _scan import modules

# Jurisdiction belongs to the domain packages.
DOMAIN_SCOPES = {"car", "pv"}


def _core() -> list:
    return [m for m in modules() if m.scope not in DOMAIN_SCOPES]


def test_no_core_module_builds_a_calendar_date() -> None:
    """No core module writes a date literal, though it may compute one."""
    violations = [
        f"{m.relative}:{node.lineno} — {ast.unparse(node)}"
        for m in _core()
        for node in ast.walk(m.tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "date"
        and node.args
        # Only a date built entirely from literals is a statute.  Calendar
        # arithmetic computes dates from variables, which is this package's job.
        and all(isinstance(arg, ast.Constant) for arg in node.args)
    ]
    assert not violations, "statutory dates in the core:\n  " + "\n  ".join(violations)


def test_no_core_module_names_a_jurisdiction_or_a_programme() -> None:
    """No identifier in the core belongs to one country's tax code."""
    banned = (
        "vat_rate_de",
        "bafa",
        "eeg",
        "kfw",
        "ustg",
        "estg",
        "kfzsteuer",
        "thg_quote",
        "degressive_2026",
    )
    violations = [
        f"{m.relative}:{node.lineno} — {node.id if isinstance(node, ast.Name) else node.attr}"
        for m in _core()
        for node in ast.walk(m.tree)
        if isinstance(node, ast.Name | ast.Attribute)
        and (node.id if isinstance(node, ast.Name) else node.attr).lower() in banned
    ]
    assert not violations, "jurisdiction-specific names in the core:\n  " + "\n  ".join(violations)


def test_tax_treatments_take_their_rates_rather_than_knowing_them() -> None:
    """No tax rate or share is defaulted in the core."""
    violations: list[str] = []
    for module in _core():
        if module.scope != "tax":
            continue
        for node in ast.walk(module.tree):
            if not isinstance(node, ast.AnnAssign) or not isinstance(node.target, ast.Name):
                continue
            name = node.target.id
            if "rate" not in name and "share" not in name:
                continue
            if node.value is None:
                continue
            default = ast.unparse(node.value)
            if default not in {"1.0", "0.0", "None"} and not default.startswith("field("):
                violations.append(f"{module.relative}:{node.lineno} — {name} = {default}")
    assert not violations, "tax rates defaulted in the core:\n  " + "\n  ".join(violations)
