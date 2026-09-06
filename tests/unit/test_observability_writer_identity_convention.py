"""What in-repo recorders put in `a2a_observability.agent_name` (#3215).

The reads in `endpoints/observability.py` scope by the agent's DID. That
is only correct while the writers agree, and nothing asserted it: every
test hand-writes a DID-shaped literal, so the tests and the endpoint
agreed with each other and could both be wrong together. They were —
the first fix for #3215 scoped by `agent.agent_name`, passed its whole
suite, and would have returned zero rows in production.

So this reads the writers themselves. It is a source-level gate rather
than a runtime one because the alternative is booting a real agent per
recorder; what it buys is that adding a display-name writer fails here
instead of silently emptying a panel.

Scope: recorders of *this agent's own* events. The column also carries
other conventions on purpose — `kestrel_feature_talon` writes and reads
`talon_job` rows by display name — which is exactly why the reads are
documented as "my own events" rather than "everything an agent owns".
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest


CORE = Path(__file__).resolve().parents[2] / "kestrel_sovereign"

# The store methods that write the column.
_RECORDERS = {"log_metric", "log_tool_call", "log_error", "log_event"}

# `a2a/task_manager.py` takes `agent_name` as its own parameter and
# forwards it; its callers resolve a did-first value. Listed by path so a
# NEW pass-through has to be justified here rather than inherited.
_FORWARDS_A_PARAMETER = {"a2a/task_manager.py", "a2a/task_worker.py"}


def _agent_name_expressions() -> list[tuple[str, int, str]]:
    """Every `agent_name=` argument passed to a recorder, as source text."""
    found: list[tuple[str, int, str]] = []
    for path in CORE.rglob("*.py"):
        try:
            tree = ast.parse(path.read_text(errors="replace"))
        except SyntaxError:  # pragma: no cover - not expected in-tree
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name = getattr(func, "attr", None) or getattr(func, "id", None)
            if name not in _RECORDERS:
                continue
            for kw in node.keywords:
                if kw.arg == "agent_name":
                    rel = str(path.relative_to(CORE))
                    found.append((rel, node.lineno, ast.unparse(kw.value)))
    return found


def test_the_scan_finds_the_recorder_calls():
    """Positive control.

    A renamed store method or a changed keyword would make the assertion
    below vacuous by finding nothing, which is the failure mode a
    source-level gate is most prone to.
    """
    calls = _agent_name_expressions()
    assert len(calls) >= 8, calls
    assert any(expr == "self.did" for _, _, expr in calls), calls


def test_every_recorder_of_our_own_events_writes_the_did():
    """The convention the reads depend on.

    If this fails, either a writer started using the display name — in
    which case `endpoints/observability.py` now under-reads — or a new
    pass-through was added and belongs in `_FORWARDS_A_PARAMETER` with a
    note on how its callers resolve the value.
    """
    offenders = [
        f"{path}:{line} passes agent_name={expr}"
        for path, line, expr in _agent_name_expressions()
        if expr != "self.did" and path not in _FORWARDS_A_PARAMETER
    ]

    assert offenders == [], (
        "observability reads scope by the agent's DID, so every recorder of "
        "this agent's own events must write it: " + "; ".join(offenders)
    )
