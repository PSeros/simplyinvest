"""Keeping a downloaded year on disk, with a note of where it came from.

A cached figure nobody can attribute is a figure nobody can defend, so every
entry is written beside a sidecar recording the request that produced it.
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

__all__ = ["CACHE_VARIABLE", "cache_root", "cached_frame"]

#: Environment variable overriding where downloads are kept.
CACHE_VARIABLE = "SIMPLYINVEST_CACHE"

_FOLDER = "simplyinvest"


def cache_root() -> Path:
    """Where downloads are kept, outside the project by default."""
    named = os.environ.get(CACHE_VARIABLE)
    if named:
        return Path(named)
    xdg = os.environ.get("XDG_CACHE_HOME")
    if xdg:
        return Path(xdg) / _FOLDER
    local = os.environ.get("LOCALAPPDATA")
    if local:
        return Path(local) / _FOLDER / "cache"
    return Path.home() / ".cache" / _FOLDER


def _key(request: Mapping[str, Any]) -> str:
    """A stable name for one request."""
    stated = json.dumps(request, sort_keys=True, default=str)
    return hashlib.sha256(stated.encode()).hexdigest()[:16]


def cached_frame(
    request: Mapping[str, Any],
    fetch: Callable[[], tuple[Any, Mapping[str, Any]]],
    *,
    root: Path | None = None,
) -> Any:
    """``fetch``'s frame, from disk if it has been fetched before.

    Args:
        request: What identifies this download, hashed into the file name.
        fetch: Called on a miss; returns the frame and what to record about it.
        root: Where to keep it.  The user cache directory by default.

    Returns:
        The hourly frame, with its timestamps parsed back.
    """
    import pandas as pd  # noqa: PLC0415

    folder = (root or cache_root()) / "weather"
    name = _key(request)
    frame_path, note_path = folder / f"{name}.csv", folder / f"{name}.json"

    if frame_path.exists():
        return pd.read_csv(frame_path, index_col=0, parse_dates=True)

    frame, meta = fetch()
    folder.mkdir(parents=True, exist_ok=True)
    frame.to_csv(frame_path)
    note_path.write_text(
        json.dumps(
            {
                "request": dict(request),
                "fetched": datetime.now(UTC).isoformat(),
                "rows": len(frame),
                "source": dict(meta),
            },
            indent=2,
            sort_keys=True,
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )
    return frame
