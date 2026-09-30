"""Shared AST scanning helpers for the architecture tests.

These tests read the source rather than importing it, so a module that would
fail to import for an unrelated reason still gets its structure checked, and so
a violation is reported at a file and line rather than as an ImportError.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "src"
PACKAGE = SRC / "simplyinvest"


@dataclass(frozen=True)
class Module:
    """One source file, parsed once."""

    path: Path
    tree: ast.Module

    @property
    def relative(self) -> str:
        """Path relative to the src directory, for error messages."""
        return str(self.path.relative_to(SRC))

    @property
    def dotted(self) -> str:
        """Full dotted module name, e.g. ``simplyinvest.timeline.grid``."""
        parts = self.path.relative_to(SRC).with_suffix("").parts
        if parts[-1] == "__init__":
            parts = parts[:-1]
        return ".".join(parts)

    @property
    def scope(self) -> str:
        """The scope this module belongs to.

        A package directory under ``simplyinvest`` is its own scope; a top-level
        module is a scope of its own name.  ``simplyinvest/__init__.py`` has no
        scope and is reported as the empty string.
        """
        parts = self.path.relative_to(PACKAGE).parts
        if len(parts) == 1:
            return "" if parts[0] == "__init__.py" else parts[0].removesuffix(".py")
        return parts[0]

    @property
    def is_scope_init(self) -> bool:
        """Whether this is a scope's ``__init__.py`` (not the package's own)."""
        return self.path.name == "__init__.py" and self.path.parent != PACKAGE


def modules() -> list[Module]:
    """Every source module in the package, parsed."""
    found = [
        Module(path, ast.parse(path.read_text(encoding="utf-8"), filename=str(path)))
        for path in sorted(PACKAGE.rglob("*.py"))
    ]
    assert found, (
        f"no source modules found under {PACKAGE} — the scan is not looking where it thinks"
    )
    return found


def imported_modules(module: Module) -> list[tuple[str, int]]:
    """Every ``simplyinvest`` module this one imports, as (dotted name, line).

    Resolves relative imports against the importing module's package, so
    ``from ..timeline import Timeline`` inside ``simplyinvest/cashflow/base.py``
    is reported as ``simplyinvest.timeline``.  Includes imports guarded by
    ``TYPE_CHECKING``: a dependency that exists only for typing is still a
    dependency of the design.
    """
    package_parts = module.dotted.split(".")
    if module.path.name != "__init__.py":
        package_parts = package_parts[:-1]

    out: list[tuple[str, int]] = []
    for node in ast.walk(module.tree):
        if isinstance(node, ast.ImportFrom):
            if node.level:
                base = package_parts[: len(package_parts) - node.level + 1]
                target = ".".join([*base, node.module] if node.module else base)
            elif node.module and node.module.split(".")[0] == "simplyinvest":
                target = node.module
            else:
                continue
            out.append((target, node.lineno))
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] == "simplyinvest":
                    out.append((alias.name, node.lineno))
    return out


def scope_of(dotted: str) -> str:
    """The scope a dotted ``simplyinvest`` module name belongs to."""
    parts = dotted.split(".")
    return parts[1] if len(parts) > 1 else ""
