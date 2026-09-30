"""Money enters the engine through `Amount`, never as a bare signed number."""

from __future__ import annotations

import ast

from _scan import PACKAGE, modules

# Flow types that take a signed per-period vector instead of an Amount.
SIGNED_VECTOR_FLOWS = {"Explicit"}


def _flow_classes() -> list[tuple[str, str, ast.ClassDef]]:
    out = []
    for module in modules():
        if module.scope != "cashflow":
            continue
        for node in module.tree.body:
            if isinstance(node, ast.ClassDef):
                out.append((module.relative, node.name, node))
    return out


def test_every_flow_takes_its_money_as_an_amount() -> None:
    found = 0
    violations: list[str] = []
    for path, name, klass in _flow_classes():
        if name in SIGNED_VECTOR_FLOWS:
            continue
        for node in klass.body:
            if not isinstance(node, ast.AnnAssign) or not isinstance(node.target, ast.Name):
                continue
            if node.target.id != "amount":
                continue
            found += 1
            annotation = ast.unparse(node.annotation).strip("\"'")
            if annotation != "Amount":
                violations.append(f"{path}:{node.lineno} — {name}.amount: {annotation}")
    assert not violations, "money entering as a bare number:\n  " + "\n  ".join(violations)
    assert found >= 3, f"only {found} flow amount fields seen; the scan is not finding them"


def test_only_the_money_module_decides_a_sign() -> None:
    """``Direction`` is chosen only inside money.py."""
    users = {
        m.scope
        for m in modules()
        for node in ast.walk(m.tree)
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "Direction"
    }
    assert users <= {"money"}, f"Direction is chosen outside money.py, in {sorted(users)}"


def test_the_money_module_exists_where_the_scan_expects_it() -> None:
    assert (PACKAGE / "money.py").exists()
