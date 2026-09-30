"""Executes every python block in the documentation.

A block asserts its own output by following a ``print`` with a ``#>`` comment
holding the expected line.
"""

from __future__ import annotations

import re
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]

# ```python ... ``` blocks, non-greedy, keeping the body.
BLOCK = re.compile(r"^```python\n(.*?)^```", re.MULTILINE | re.DOTALL)

# A `#>` line states the output of the statement above it.  `ruff format`
# rewrites it to `# >`, so both spellings count.
EXPECTED = re.compile(r"^(\s*)#\s*>[ ]?(.*)$", re.MULTILINE)


def documents() -> list[Path]:
    found = [ROOT / "README.md", *sorted((ROOT / "docs").rglob("*.md"))]
    return [path for path in found if path.exists()]


def blocks() -> list[tuple[str, int, str]]:
    out = []
    for path in documents():
        text = path.read_text(encoding="utf-8")
        for match in BLOCK.finditer(text):
            line = text[: match.start()].count("\n") + 1
            out.append((path.relative_to(ROOT).as_posix(), line, match.group(1)))
    return out


def _as_script(source: str) -> tuple[str, list[str]]:
    """Turn ``#>`` comments into assertions on captured output."""
    expected = [m.group(2) for m in EXPECTED.finditer(source)]
    return EXPECTED.sub("", source), expected


@pytest.mark.parametrize(
    ("path", "line", "source"),
    blocks(),
    ids=[f"{path}:{line}" for path, line, _ in blocks()],
)
def test_a_documented_block_runs_and_says_what_it_claims(
    path: str, line: int, source: str, capsys: pytest.CaptureFixture[str]
) -> None:
    script, expected = _as_script(source)

    # A real module entry, because @dataclass resolves annotations through
    # sys.modules[cls.__module__], and a bare exec namespace has no module.
    name = f"_doc_{path.replace('/', '_').replace('.', '_')}_{line}"
    module = types.ModuleType(name)
    sys.modules[name] = module
    try:
        exec(compile(script, f"{path}:{line}", "exec"), module.__dict__)
    except Exception as error:  # pragma: no cover - the message is the point
        pytest.fail(f"{path}:{line} — the documented example raised {error!r}")
    finally:
        sys.modules.pop(name, None)

    if not expected:
        return
    printed = [ln for ln in capsys.readouterr().out.splitlines() if ln.strip()]
    assert printed == expected, (
        f"{path}:{line} — the example printed {printed!r} but the documentation claims {expected!r}"
    )


def test_there_are_blocks_to_check() -> None:
    """The scan finds blocks, so the test above cannot pass vacuously."""
    assert len(blocks()) >= 3, f"only {len(blocks())} python blocks found under {ROOT}"


def test_the_documentation_asserts_its_own_output() -> None:
    """A claim the harness stops recognising is a claim that stops being checked."""
    claims = sum(len(EXPECTED.findall(source)) for _, _, source in blocks())
    assert claims >= 8, (
        f"only {claims} documented outputs are being checked.  A `#>` comment that the "
        f"harness no longer matches leaves its figure unverified."
    )
