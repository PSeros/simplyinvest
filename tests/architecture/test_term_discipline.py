"""Durations are `Term`s, and only `timeline/` converts them into periods."""

from __future__ import annotations

import ast
import re

from _scan import PACKAGE, modules

# Identifiers whose name claims a calendar unit.
UNIT_SUFFIXED = re.compile(r".*_(months|years|days)$")

# The one scope allowed to convert calendar units into period indices.
CONVERSION_SCOPE = "timeline"


def _identifier(node: ast.expr) -> str | None:
    """The name an operand is known by, if it has one."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def test_only_the_timeline_scope_does_month_arithmetic() -> None:
    """No scope but `timeline` adds or subtracts a unit-suffixed number."""
    violations: list[str] = []
    for module in modules():
        if module.scope == CONVERSION_SCOPE:
            continue
        for node in ast.walk(module.tree):
            if not isinstance(node, ast.BinOp) or not isinstance(node.op, ast.Add | ast.Sub):
                continue
            for operand in (node.left, node.right):
                name = _identifier(operand)
                if name and UNIT_SUFFIXED.match(name):
                    violations.append(
                        f"{module.relative}:{node.lineno} — {name!r} in arithmetic outside "
                        f"{CONVERSION_SCOPE}/; use a Term and Timeline.periods_in/offset"
                    )
    assert not violations, "raw calendar-unit arithmetic:\n  " + "\n  ".join(violations)


def test_durations_are_terms_not_integers() -> None:
    """No dataclass field named for a calendar unit is typed as a bare number."""
    violations: list[str] = []
    for module in modules():
        if module.scope == CONVERSION_SCOPE:
            continue
        for node in ast.walk(module.tree):
            if not isinstance(node, ast.AnnAssign) or not isinstance(node.target, ast.Name):
                continue
            name = node.target.id
            annotation = ast.unparse(node.annotation)
            if UNIT_SUFFIXED.match(name) and annotation.lstrip("\"'").startswith(
                ("int", "float", "Quantity")
            ):
                violations.append(
                    f"{module.relative}:{node.lineno} — {name}: {annotation}; a duration is a Term"
                )
    assert not violations, "durations typed as bare numbers:\n  " + "\n  ".join(violations)


def test_the_scan_actually_sees_the_package() -> None:
    """The scan reaches real code, so the tests above cannot pass vacuously."""
    found = modules()
    assert len(found) >= 5, f"only {len(found)} modules found under {PACKAGE}"
    binops = sum(isinstance(node, ast.BinOp) for m in found for node in ast.walk(m.tree))
    assert binops >= 20, (
        f"only {binops} binary operations seen across the package — the arithmetic "
        f"scan is not reaching real code"
    )
