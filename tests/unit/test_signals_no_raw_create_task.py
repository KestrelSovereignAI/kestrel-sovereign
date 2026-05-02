"""Lint test: no raw asyncio.create_task(dispatch_signal/enqueue_signal).

Per SIGNAL_DISPATCHER.md §"The dispatcher contract" and §Concern P2 from
the v3 review: callers that want fire-and-forget must use
`dispatcher.enqueue_signal(...)` (which goes through the agent's
background task tracker), NOT raw `asyncio.create_task(dispatch_signal(...))`.
The latter leaks tasks, swallows exceptions, and survives shutdown.

This is the "lint rule" from acceptance criterion 4 of #890. When the
project gains ruff or pre-commit, port this to a real check.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
# Search the source tree but exclude this test file (the patterns appear
# in its docstring).
SEARCH_DIRS = [
    REPO_ROOT / "kestrel_sovereign",
    REPO_ROOT / "kestrel_sdk",
    REPO_ROOT / "endpoints",
]

FORBIDDEN = re.compile(
    r"asyncio\.create_task\s*\(\s*"
    r"(?:[\w.]*\.)?(?:dispatch_signal|enqueue_signal)\s*\("
)


def _python_files() -> list[Path]:
    files: list[Path] = []
    for root in SEARCH_DIRS:
        if not root.exists():
            continue
        files.extend(root.rglob("*.py"))
    return files


def test_no_raw_create_task_around_signal_dispatch():
    offenders: list[tuple[Path, int, str]] = []
    for path in _python_files():
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        for lineno, line in enumerate(text.splitlines(), start=1):
            if FORBIDDEN.search(line):
                offenders.append((path, lineno, line.strip()))

    if offenders:
        msg = "Found raw asyncio.create_task wrapping a signal dispatch call.\n"
        msg += "Use dispatcher.enqueue_signal(...) instead — it goes through\n"
        msg += "the agent's background task tracker (supervised lifetime).\n\n"
        for path, lineno, line in offenders:
            rel = path.relative_to(REPO_ROOT)
            msg += f"  {rel}:{lineno}: {line}\n"
        pytest.fail(msg)
