"""What in-repo recorders put in `a2a_observability.agent_name` (#3215).

The reads in `endpoints/observability.py` scope by the agent's DID. That
is only correct while the writers agree, and nothing asserted it: every
other test hand-writes a DID-shaped literal, so the tests and the
endpoint agreed with each other and could both be wrong together. They
were — the first fix for #3215 scoped by `agent.agent_name`, passed its
whole suite, and would have returned zero rows in production.

So this reads the writers themselves, and derives everything it can
rather than restating it. The first version of this file hard-coded four
method names: one of them (`log_event`) does not exist, and it missed
`log_agent_response`, which does — a gate whose own inputs were wrong.
The recorder set now comes from the store: every method whose body
inserts into the table.

Scope: recorders of *this agent's own* events. The column carries other
conventions on purpose — `kestrel_feature_talon` writes and reads
`talon_job` rows by display name — which is why the reads are documented
as "my own events" rather than "everything an agent owns".
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest


CORE = Path(__file__).resolve().parents[2] / "kestrel_sovereign"
STORE = CORE / "a2a" / "stores" / "unified" / "observability_store.py"
TABLE = "INSERT INTO a2a_observability"

# Modules that take `agent_name` as their OWN parameter and forward it,
# so the value is chosen by their callers rather than here. Each must
# still match a real forwarding call — `test_the_allowlist_is_tight`
# fails on an entry that has stopped applying, so this cannot decay into
# a place to put inconvenient results.
_FORWARDS_A_PARAMETER = {"a2a/task_manager.py"}


def _recorder_arg_positions() -> dict[str, int]:
    """`{method name: index of agent_name}` for every writer of the table.

    Derived from the store rather than listed, so a new recorder is
    covered the day it is added and a renamed one fails loudly instead
    of silently dropping out of the scan.
    """
    source = STORE.read_text()
    tree = ast.parse(source)
    recorders: dict[str, int] = {}
    for node in ast.walk(tree):
        if not isinstance(node, (ast.AsyncFunctionDef, ast.FunctionDef)):
            continue
        body = ast.get_source_segment(source, node) or ""
        if TABLE not in body:
            continue
        names = [a.arg for a in node.args.args]
        if "agent_name" in names:
            # Minus one: `self` is not passed at the call site.
            recorders[node.name] = names.index("agent_name") - 1
    return recorders


def _agent_name_expressions() -> list[tuple[str, int, str]]:
    """Every identity passed to a recorder, keyword OR positional."""
    recorders = _recorder_arg_positions()
    found: list[tuple[str, int, str]] = []
    for path in CORE.rglob("*.py"):
        try:
            tree = ast.parse(path.read_text(errors="replace"))
        except SyntaxError:  # pragma: no cover - not expected in-tree
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = getattr(node.func, "attr", None) or getattr(node.func, "id", None)
            if name not in recorders:
                continue
            rel = str(path.relative_to(CORE))

            keyword = {kw.arg: kw.value for kw in node.keywords}
            if "agent_name" in keyword:
                found.append((rel, node.lineno, ast.unparse(keyword["agent_name"])))
                continue

            # A `**kwargs` splat could carry any identity; it is opaque
            # here, so it counts as one that is not `self.did` rather
            # than as nothing at all.
            if any(kw.arg is None for kw in node.keywords):
                found.append((rel, node.lineno, "**splat (opaque)"))
                continue

            index = recorders[name]
            if len(node.args) > index:
                found.append((rel, node.lineno, ast.unparse(node.args[index])))
            elif not node.args:
                # Neither keyword nor positional: the call is incomplete
                # or bound elsewhere. Surface it rather than skip it.
                found.append((rel, node.lineno, "<no identity argument>"))
    return found


def test_the_scan_covers_every_recorder_the_store_defines():
    """Positive control, and a closed one.

    A count threshold would tolerate losing a whole recorder. This
    requires the derived set to be non-empty, to include the known
    writers, and — the part that matters — that scanning finds a real
    `self.did` call, so a renamed method or changed keyword cannot make
    the assertion below vacuous by matching nothing.
    """
    recorders = _recorder_arg_positions()
    assert set(recorders) >= {
        "log_metric",
        "log_tool_call",
        "log_error",
        "log_agent_response",
    }, recorders
    assert all(index >= 0 for index in recorders.values()), recorders

    calls = _agent_name_expressions()
    assert any(expr == "self.did" for _, _, expr in calls), calls


def test_the_allowlist_is_tight():
    """An exemption that no longer applies must be removed, not inherited.

    The first version of this file also exempted `a2a/task_worker.py`,
    which has no production constructor at all — an entry that hid
    nothing and would have hidden anything added there later.
    """
    scanned = {path for path, _, _ in _agent_name_expressions()}
    stale = _FORWARDS_A_PARAMETER - scanned
    assert stale == set(), (
        f"these modules are exempted but no longer call a recorder: {stale}"
    )


def test_every_recorder_of_our_own_events_writes_the_did():
    """The convention the reads depend on.

    If this fails, either a writer started using the display name — in
    which case `endpoints/observability.py` now under-reads — or a new
    pass-through was added and belongs in `_FORWARDS_A_PARAMETER` with a
    note on how its callers resolve the value.
    """
    offenders = [
        f"{path}:{line} passes {expr}"
        for path, line, expr in _agent_name_expressions()
        if expr != "self.did" and path not in _FORWARDS_A_PARAMETER
    ]

    assert offenders == [], (
        "observability reads scope by the agent's DID, so every recorder of "
        "this agent's own events must write it: " + "; ".join(offenders)
    )
