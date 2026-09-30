"""Turning a missing optional dependency into a sentence that helps.

Every domain ships in the wheel; only its dependencies are optional.  So the
failure a user meets is not "this package is missing" but "pvlib is missing",
which does not tell them the install they wanted was
``uv add "simplyinvest[pv]"``.
"""

from __future__ import annotations

import importlib

__all__ = ["require_extra"]


def require_extra(extra: str, *modules: str) -> None:
    """Check that ``extra``'s dependencies are installed, or explain what to do.

    Args:
        extra: The optional-dependency group, as it appears in the install.
        modules: The importable names that group provides.

    Raises:
        ImportError: naming the extra and the exact install command.
    """
    missing = []
    for name in modules:
        try:
            importlib.import_module(name)
        except ImportError:
            missing.append(name)
    if missing:
        raise ImportError(
            f"simplyinvest[{extra}] needs {', '.join(missing)}, which "
            f"{'is' if len(missing) == 1 else 'are'} not installed.  Install it with:\n\n"
            f'    uv add "simplyinvest[{extra}]"\n'
        )
