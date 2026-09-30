"""A scope may depend only on scopes below it, and the graph must be acyclic."""

from __future__ import annotations

import ast

import pytest
from _scan import PACKAGE, imported_modules, modules, scope_of

# A scope may import from any scope with a strictly lower number.
LAYERS: dict[str, int] = {
    "errors": 0,
    "money": 1,
    "timeline": 2,
    "cashflow": 3,
    "metrics": 4,
    "domain": 5,
    "financing": 6,
    "tax": 6,
    "incentives": 6,
    "appraisal": 7,
    "uncertain": 8,
    "report": 9,
}

# Domain packages may import any core scope, never each other.
DOMAINS: dict[str, int] = {"car": 10, "pv": 10}

# Infrastructure any scope may reach for.
EXEMPT = {"_optional", ""}

ALL_LAYERS = LAYERS | DOMAINS


def test_layer_table_covers_every_scope() -> None:
    """Every directory and top-level module in the package has a declared layer."""
    scopes = {m.scope for m in modules()} - EXEMPT
    undeclared = sorted(scopes - ALL_LAYERS.keys())
    assert not undeclared, (
        f"scopes with no entry in the layer table: {undeclared}.  Add them to LAYERS "
        f"(or DOMAINS) with a deliberate layer rather than leaving them unchecked."
    )


def test_no_module_imports_from_a_higher_layer() -> None:
    """A scope depends only on scopes strictly below it."""
    violations: list[str] = []
    for module in modules():
        own = module.scope
        if own in EXEMPT or own not in ALL_LAYERS:
            continue
        for dotted, line in imported_modules(module):
            other = scope_of(dotted)
            if other in EXEMPT or other == own or other not in ALL_LAYERS:
                continue
            if ALL_LAYERS[other] >= ALL_LAYERS[own]:
                violations.append(
                    f"{module.relative}:{line} — {own} (layer {ALL_LAYERS[own]}) imports "
                    f"{dotted} (layer {ALL_LAYERS[other]})"
                )
    assert not violations, "imports against the dependency direction:\n  " + "\n  ".join(violations)


def test_module_graph_is_acyclic() -> None:
    """No import cycles, including within a single scope."""
    graph = {m.dotted: {d for d, _ in imported_modules(m)} for m in modules()}
    known = set(graph)
    state: dict[str, int] = {}
    cycle: list[str] = []

    def visit(node: str, stack: list[str]) -> bool:
        if state.get(node) == 2:
            return False
        if state.get(node) == 1:
            cycle.extend([*stack[stack.index(node) :], node])
            return True
        state[node] = 1
        for nxt in sorted(graph.get(node, set()) & known):
            if visit(nxt, [*stack, nxt]):
                return True
        state[node] = 2
        return False

    for node in sorted(graph):
        if visit(node, [node]):
            break
    assert not cycle, "import cycle: " + " -> ".join(cycle)


def test_core_never_imports_a_domain() -> None:
    """No core scope reaches into `car` or `pv`, not even under TYPE_CHECKING."""
    violations = [
        f"{m.relative}:{line} — core scope {m.scope!r} imports {dotted}"
        for m in modules()
        if m.scope not in DOMAINS
        for dotted, line in imported_modules(m)
        if scope_of(dotted) in DOMAINS
    ]
    assert not violations, "the core must not know its domains:\n  " + "\n  ".join(violations)


def test_domains_never_import_each_other() -> None:
    """`car` and `pv` are siblings; neither may depend on the other."""
    violations = [
        f"{m.relative}:{line} — {m.scope} imports {dotted}"
        for m in modules()
        if m.scope in DOMAINS
        for dotted, line in imported_modules(m)
        if scope_of(dotted) in DOMAINS and scope_of(dotted) != m.scope
    ]
    assert not violations, "domains must stay independent:\n  " + "\n  ".join(violations)


def test_scope_inits_only_re_export() -> None:
    """A scope's ``__init__.py`` holds only a docstring, imports and ``__all__``."""
    allowed = (ast.Import, ast.ImportFrom, ast.Expr, ast.Assign, ast.AnnAssign, ast.If)
    violations = [
        f"{m.relative}:{node.lineno} — {type(node).__name__} in a scope __init__"
        for m in modules()
        if m.is_scope_init
        for node in m.tree.body
        if not isinstance(node, allowed)
    ]
    assert not violations, "scope __init__ files re-export and nothing else:\n  " + "\n  ".join(
        violations
    )


def test_incentives_scope_holds_only_the_contract() -> None:
    """``simplyinvest/incentives/`` contains the contract and nothing else."""
    scope = PACKAGE / "incentives"
    if not scope.exists():
        pytest.skip("incentives scope not built yet")
    present = {p.name for p in scope.glob("*.py")}
    assert present == {"__init__.py", "base.py"}, (
        f"simplyinvest/incentives/ must contain only the contract, found: {sorted(present)}"
    )
