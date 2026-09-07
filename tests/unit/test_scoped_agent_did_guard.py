"""One guard for "the DID this agent's self-scoped rows are bound to".

Four sites scoped a shared table to the calling agent and each carried
its own copy of the check (observability #3215, the A2A task list, the
consent log and audit anchors #3229/#3230); they had drifted — one gated
on truthiness alone and would bind a non-string value as a query
parameter. `resolve_scoped_agent_did` is the single copy. #3246 routed
the remaining self-scoped reads of the same kind through it: the A2A
task routes (reads, subscribe, cancel), the task feature's own durable
identity, the task wait provider, the pre-turn sections and the restart
status-events route. Other tables still resolve inline; the guard's
docstring names them.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from kestrel_sovereign.features.storage_access import (
    AgentIdentityUnavailable,
    resolve_scoped_agent_did,
)


def test_returns_the_agents_did():
    assert resolve_scoped_agent_did(SimpleNamespace(did="did:test:x")) == "did:test:x"
    mock = MagicMock()
    mock.did = "did:test:explicit"
    assert resolve_scoped_agent_did(mock) == "did:test:explicit"


@pytest.mark.parametrize(
    "agent",
    [
        pytest.param(SimpleNamespace(did=None), id="none"),
        pytest.param(SimpleNamespace(did=""), id="empty"),
        pytest.param(SimpleNamespace(did=123), id="non-string"),
        pytest.param(SimpleNamespace(agent_name="only-a-display-name"), id="absent"),
        pytest.param(MagicMock(), id="magicmock-fabrication"),
        pytest.param(None, id="no-agent"),
    ],
)
def test_refuses_anything_that_is_not_a_did(agent):
    """Not "unscoped": a store gating on ``if agent_id:`` reads every row
    for an empty string, and a MagicMock's fabricated attribute is truthy."""
    with pytest.raises(AgentIdentityUnavailable, match="identity"):
        resolve_scoped_agent_did(agent)
    assert issubclass(AgentIdentityUnavailable, RuntimeError)


# ---------------------------------------------------------------------------
# The `!tasks` command routes through the guard (it had no coverage before)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_tasks_command_scopes_by_the_guarded_did():
    from unittest.mock import AsyncMock

    from kestrel_sovereign.command_handler import CommandHandler

    agent = MagicMock()
    agent.did = "did:test:recipient"
    task_manager = MagicMock()
    task_manager.task_store.list_tasks = AsyncMock(return_value=[])
    handler = CommandHandler(agent, task_manager=task_manager)

    assert await handler._cmd_tasks("!tasks") == "📋 No tasks found"
    task_manager.task_store.list_tasks.assert_awaited_once_with(
        recipient_agent_id="did:test:recipient", limit=10
    )


@pytest.mark.asyncio
async def test_tasks_command_refuses_without_a_did():
    from unittest.mock import AsyncMock

    from kestrel_sovereign.command_handler import CommandHandler

    agent = MagicMock()  # a fabricated `did`, not an identity
    task_manager = MagicMock()
    task_manager.task_store.list_tasks = AsyncMock(return_value=[])
    handler = CommandHandler(agent, task_manager=task_manager)

    assert await handler._cmd_tasks("!tasks") == "❌ Task recipient identity unavailable"
    task_manager.task_store.list_tasks.assert_not_awaited()


# ---------------------------------------------------------------------------
# #3246: the five sites that still resolved the DID inline route through the
# guard. Each case below pins the mutation the guard's own test pins — the
# site reads a different field, or accepts an empty / fabricated DID — and
# that the store is never reached on a refusal.
# ---------------------------------------------------------------------------

ME = "did:test:me"
OTHER = "did:test:other"


def _identity_cases():
    """Agents the guard must refuse, one per shape the inline copies let in."""
    return [
        pytest.param(MagicMock(), id="magicmock-fabrication"),
        pytest.param(SimpleNamespace(did=""), id="empty"),
        pytest.param(SimpleNamespace(did=123), id="non-string"),
        pytest.param(SimpleNamespace(_did=ME), id="private-did-only"),
        pytest.param(SimpleNamespace(agent_id=ME), id="agent-id-only"),
    ]


# -- endpoints/agent.py: the A2A task routes' recipient --------------------


def test_task_recipient_is_the_did_not_agent_id():
    from kestrel_sovereign.endpoints.agent import _task_recipient_principal

    agent = SimpleNamespace(agent_id=OTHER, did=ME)
    assert _task_recipient_principal(agent) == ME


@pytest.mark.parametrize("agent", _identity_cases())
def test_task_recipient_refuses_with_503(agent):
    from fastapi import HTTPException

    from kestrel_sovereign.endpoints.agent import _task_recipient_principal

    with pytest.raises(HTTPException) as excinfo:
        _task_recipient_principal(agent)
    assert excinfo.value.status_code == 503
    assert "durable recipient identity" in excinfo.value.detail


@pytest.mark.asyncio
async def test_get_tasks_and_the_tasks_command_scope_one_table_the_same_way():
    """The ticket's concrete divergence: `GET /tasks` read `agent_id` first,
    `!tasks` read `did`. One agent object carrying both, distinct, must list
    the same rows on both surfaces."""
    from unittest.mock import AsyncMock

    from kestrel_sovereign.command_handler import CommandHandler
    from kestrel_sovereign.endpoints.agent import list_tasks

    task_manager = MagicMock()
    task_manager.task_store.list_tasks = AsyncMock(return_value=[])
    agent = SimpleNamespace(agent_id=OTHER, did=ME, task_manager=task_manager)

    request = SimpleNamespace(state=SimpleNamespace(agent=agent))
    body = await list_tasks(request, status=None, limit=50)
    assert body["total"] == 0
    task_manager.task_store.list_tasks.assert_awaited_once_with(
        recipient_agent_id=ME, status=None, limit=50
    )

    task_manager.task_store.list_tasks.reset_mock()
    handler = CommandHandler(agent, task_manager=task_manager)
    assert await handler._cmd_tasks("!tasks") == "📋 No tasks found"
    task_manager.task_store.list_tasks.assert_awaited_once_with(
        recipient_agent_id=ME, limit=10
    )


@pytest.mark.asyncio
async def test_tasks_command_ignores_a_distinct_agent_id():
    from unittest.mock import AsyncMock

    from kestrel_sovereign.command_handler import CommandHandler

    agent = MagicMock()
    agent.did = ME
    agent.agent_id = OTHER
    task_manager = MagicMock()
    task_manager.task_store.list_tasks = AsyncMock(return_value=[])
    handler = CommandHandler(agent, task_manager=task_manager)

    await handler._cmd_tasks("!tasks")
    task_manager.task_store.list_tasks.assert_awaited_once_with(
        recipient_agent_id=ME, limit=10
    )


# -- features/tasks/wait_provider.py: ownership at watch registration ------


def _task_waitable(agent):
    from unittest.mock import AsyncMock

    from kestrel_sovereign.features.tasks.wait_provider import TaskWaitable

    manager = MagicMock()
    manager.get_task_for_recipient = AsyncMock(return_value=object())
    feature = SimpleNamespace(task_manager=manager, agent=agent)
    return TaskWaitable(feature), manager


@pytest.mark.asyncio
async def test_wait_provider_owns_by_the_did_not_agent_id():
    provider, manager = _task_waitable(SimpleNamespace(agent_id=OTHER, did=ME))
    assert await provider.owns_handle("local-1") is True
    manager.get_task_for_recipient.assert_awaited_once_with("local-1", ME)


@pytest.mark.asyncio
@pytest.mark.parametrize("agent", _identity_cases())
async def test_wait_provider_owns_nothing_without_a_did(agent):
    """False, not None: an agent that cannot be scoped owns no task. None
    would fail open at registration."""
    provider, manager = _task_waitable(agent)
    assert await provider.owns_handle("local-1") is False
    manager.get_task_for_recipient.assert_not_awaited()


@pytest.mark.asyncio
async def test_wait_provider_with_no_agent_owns_nothing():
    from unittest.mock import AsyncMock

    from kestrel_sovereign.features.tasks.wait_provider import TaskWaitable

    manager = MagicMock()
    manager.get_task_for_recipient = AsyncMock(return_value=object())
    provider = TaskWaitable(SimpleNamespace(task_manager=manager))
    assert await provider.owns_handle("local-1") is False
    manager.get_task_for_recipient.assert_not_awaited()


# -- agent/preturn_state.py: the three DID-scoped sections -----------------


def _inbox_agent(**identity):
    from unittest.mock import AsyncMock

    task_manager = MagicMock()
    task_manager.task_store.list_tasks = AsyncMock(return_value=[])
    return SimpleNamespace(task_manager=task_manager, **identity), task_manager


@pytest.mark.asyncio
async def test_inbox_section_scopes_by_the_did_not_agent_id():
    from kestrel_sovereign.agent.preturn_state import _a2a_inbox_section

    agent, task_manager = _inbox_agent(agent_id=OTHER, did=ME)
    assert await _a2a_inbox_section(agent) == "A2A inbox: no pending tasks"
    task_manager.task_store.list_tasks.assert_awaited_once_with(
        recipient_agent_id=ME, limit=50
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("agent", _identity_cases())
async def test_inbox_section_is_silent_without_a_did(agent):
    from unittest.mock import AsyncMock

    from kestrel_sovereign.agent.preturn_state import _a2a_inbox_section

    task_manager = MagicMock()
    task_manager.task_store.list_tasks = AsyncMock(return_value=[])
    agent.task_manager = task_manager
    assert await _a2a_inbox_section(agent) is None
    task_manager.task_store.list_tasks.assert_not_awaited()


def _restart_agent(monkeypatch, **identity):
    """An agent whose RestartCoordinatorFeature has a db; the store read is
    replaced by a recorder so the section's scoping is observable."""
    from unittest.mock import AsyncMock

    from kestrel_sovereign.features.restart_coordinator import event_store

    reads = AsyncMock(return_value=[])
    monkeypatch.setattr(event_store, "list_recent_events_for_agent_context", reads)
    feat = SimpleNamespace(_db=object())
    agent = SimpleNamespace(get_feature=lambda name: feat, **identity)
    return agent, reads


@pytest.mark.asyncio
async def test_restart_section_scopes_by_the_did_not_private_did(monkeypatch):
    from kestrel_sovereign.agent.preturn_state import _restart_status_section

    agent, reads = _restart_agent(monkeypatch, _did=OTHER, did=ME)
    assert await _restart_status_section(agent) is None  # no rows, no line
    reads.assert_awaited_once()
    assert reads.await_args.kwargs["agent_id"] == ME


@pytest.mark.asyncio
@pytest.mark.parametrize("agent", _identity_cases())
async def test_restart_section_is_silent_without_a_did(monkeypatch, agent):
    from unittest.mock import AsyncMock

    from kestrel_sovereign.agent.preturn_state import _restart_status_section
    from kestrel_sovereign.features.restart_coordinator import event_store

    reads = AsyncMock(return_value=[])
    monkeypatch.setattr(event_store, "list_recent_events_for_agent_context", reads)
    agent.get_feature = lambda name: SimpleNamespace(_db=object())
    assert await _restart_status_section(agent) is None
    reads.assert_not_awaited()


def _decline_agent(monkeypatch, **identity):
    from unittest.mock import AsyncMock

    from kestrel_sovereign.llm import codex_decline_events

    reads = AsyncMock(return_value=[])
    monkeypatch.setattr(codex_decline_events, "list_recent_declines_for_agent", reads)
    agent = SimpleNamespace(_raw_storage=SimpleNamespace(db=object()), **identity)
    return agent, reads


@pytest.mark.asyncio
async def test_decline_section_scopes_by_the_did_not_private_did(monkeypatch):
    from kestrel_sovereign.agent.preturn_state import _codex_decline_section

    agent, reads = _decline_agent(monkeypatch, _did=OTHER, did=ME)
    assert await _codex_decline_section(agent) is None  # no rows, no line
    reads.assert_awaited_once()
    assert reads.await_args.kwargs["agent_id"] == ME


@pytest.mark.asyncio
@pytest.mark.parametrize("agent", _identity_cases())
async def test_decline_section_is_silent_without_a_did(monkeypatch, agent):
    from unittest.mock import AsyncMock

    from kestrel_sovereign.agent.preturn_state import _codex_decline_section
    from kestrel_sovereign.llm import codex_decline_events

    reads = AsyncMock(return_value=[])
    monkeypatch.setattr(codex_decline_events, "list_recent_declines_for_agent", reads)
    agent._raw_storage = SimpleNamespace(db=object())
    assert await _codex_decline_section(agent) is None
    reads.assert_not_awaited()


# -- endpoints/restart_events.py: the status-events route -------------------


def _events_request(monkeypatch, agent):
    from unittest.mock import AsyncMock

    from kestrel_sovereign.features.restart_coordinator import event_store

    reads = AsyncMock(return_value=[])
    monkeypatch.setattr(event_store, "list_recent_events_for_history", reads)
    agent._raw_storage = SimpleNamespace(db=object())
    return SimpleNamespace(state=SimpleNamespace(agent=agent)), reads


@pytest.mark.asyncio
async def test_status_events_scope_by_the_did_not_agent_id(monkeypatch):
    from kestrel_sovereign.endpoints.restart_events import get_restart_status_events

    request, reads = _events_request(monkeypatch, SimpleNamespace(agent_id=OTHER, did=ME))
    body = await get_restart_status_events(request, session="", limit=200)
    assert body == {"events": [], "count": 0}
    reads.assert_awaited_once()
    assert reads.await_args.kwargs["agent_id"] == ME


@pytest.mark.asyncio
@pytest.mark.parametrize("agent", _identity_cases())
async def test_status_events_are_empty_without_a_did(monkeypatch, agent):
    from kestrel_sovereign.endpoints.restart_events import get_restart_status_events

    request, reads = _events_request(monkeypatch, agent)
    assert await get_restart_status_events(request, session="", limit=200) == {
        "events": [],
        "count": 0,
    }
    reads.assert_not_awaited()


# -- the two sites the round-1 review found over the same a2a_tasks table --


def test_task_feature_durable_identity_is_the_did_not_agent_id():
    from kestrel_sovereign.features.tasks.feature import TaskFeature

    feature = TaskFeature(SimpleNamespace(agent_id=OTHER, did=ME))
    assert feature._durable_agent_id() == ME
    assert feature._recipient_agent_id() == ME


@pytest.mark.parametrize("agent", _identity_cases() + [pytest.param(None, id="no-agent")])
def test_task_feature_has_no_durable_identity_without_a_did(agent):
    from kestrel_sovereign.features.tasks.feature import TaskFeature

    feature = TaskFeature(agent)
    assert feature._durable_agent_id() is None
    with pytest.raises(ValueError, match="identity unavailable"):
        feature._recipient_agent_id()


def _post_cancel(monkeypatch, agent):
    """POST the cancel route on a real app: the route is rate-limited and
    needs a real Starlette request."""
    from fastapi.testclient import TestClient

    from tests.unit.test_a2a_principal_reads import _principal_endpoint_app

    app = _principal_endpoint_app(monkeypatch, agent)
    body = {
        "reason": "stop",
        "sessionId": "s-1",
        "metadata": {"a2a_verb": "cancel_task"},
    }
    with TestClient(app) as client:
        return client.post("/api/agent/tasks/task-1/cancel", json=body)


def test_cancel_route_refuses_when_only_the_task_manager_carries_an_id(monkeypatch):
    """The route used to take ``task_manager.host_agent_id`` before the
    agent's own DID. A manager that carries one while the agent has none
    is now a 503, not a cancellation on the manager's copy."""
    agent = SimpleNamespace(did="", task_manager=SimpleNamespace(host_agent_id=OTHER))
    response = _post_cancel(monkeypatch, agent)
    assert response.status_code == 503
    assert "cancellation requires a durable recipient identity" in response.json()["detail"]


def test_cancel_route_resolves_its_recipient_through_the_shared_helper(monkeypatch):
    """Wiring: the route asks the one helper, with the agent (not the task
    manager) and its own verb. A sentinel refusal proves the call site."""
    from fastapi import HTTPException

    from kestrel_sovereign.endpoints import agent as agent_endpoint

    asked = []

    def sentinel(agent, *, verb="reads require"):
        asked.append((agent, verb))
        raise HTTPException(status_code=503, detail="sentinel")

    monkeypatch.setattr(agent_endpoint, "_task_recipient_principal", sentinel)
    agent = SimpleNamespace(did=ME, task_manager=SimpleNamespace(host_agent_id=OTHER))
    response = _post_cancel(monkeypatch, agent)
    assert response.status_code == 503
    assert response.json()["detail"] == "sentinel"
    assert asked == [(agent, "cancellation requires")]


# -- the write side of the same table (review r2) ---------------------------


def _send_params():
    from kestrel_sovereign.a2a.types import Message, TaskSendParams, TextPart

    return TaskSendParams(
        id="task-1",
        sessionId="sess-1",
        message=Message(role="user", parts=[TextPart(text="do it")]),
        metadata={"sender": OTHER},
    )


@pytest.mark.asyncio
async def test_task_creation_refuses_a_recipient_without_a_did_before_verifying():
    """The creation path used to file the row under ``did``, else the display
    name, else ``"unknown"`` — a fourth resolution order over the same table
    the reads now guard. No identity: 503, and nothing is verified or
    created."""
    from fastapi import HTTPException
    from unittest.mock import AsyncMock

    from kestrel_sovereign.endpoints import agent as agent_endpoint

    task_manager = MagicMock()
    task_manager.create_task = AsyncMock()
    agent = SimpleNamespace(did="", _agent_name="recipient", task_manager=task_manager)
    params = _send_params()
    with pytest.raises(HTTPException) as excinfo:
        await agent_endpoint._create_a2a_task_under_lifecycle_lease(
            agent, params, params.message.parts, [], [], manager=None
        )
    assert excinfo.value.status_code == 503
    assert "creation requires a durable recipient identity" in excinfo.value.detail
    task_manager.create_task.assert_not_awaited()


@pytest.mark.asyncio
async def test_task_creation_resolves_its_recipient_through_the_shared_helper(monkeypatch):
    """Wiring: creation asks the one helper with the agent and its own verb,
    and an action carrying a ``commit`` does not (its route already did)."""
    from fastapi import HTTPException

    from kestrel_sovereign.endpoints import agent as agent_endpoint

    asked = []

    def sentinel(agent, *, verb="reads require"):
        asked.append((agent, verb))
        raise HTTPException(status_code=503, detail="sentinel")

    monkeypatch.setattr(agent_endpoint, "_task_recipient_principal", sentinel)
    agent = SimpleNamespace(did=ME, _agent_name=OTHER, task_manager=MagicMock())
    params = _send_params()
    with pytest.raises(HTTPException, match="sentinel"):
        await agent_endpoint._create_a2a_task_under_lifecycle_lease(
            agent, params, params.message.parts, [], [], manager=None
        )
    assert asked == [(agent, "creation requires")]

    # An action carrying a commit resolved its recipient at its route: the
    # creation guard is not consulted. The sentinel would raise first if it
    # were; instead the unsigned envelope is what refuses, later.
    asked.clear()
    from unittest.mock import AsyncMock

    commit = AsyncMock(return_value="committed")
    with pytest.raises(HTTPException) as excinfo:
        await agent_endpoint._create_a2a_task_under_lifecycle_lease(
            agent, params, params.message.parts, [], [], manager=None, commit=commit
        )
    assert excinfo.value.detail != "sentinel"
    assert asked == []
    commit.assert_not_awaited()


@pytest.mark.asyncio
async def test_task_creation_refuses_before_any_verification_work(monkeypatch):
    """Review r3: the ordering is load-bearing. Verification reserves the
    replay nonce with no rollback, so a refusal AFTER it would burn a
    correctly signed envelope that a repaired recipient could otherwise
    accept on retry. With no identity, neither the verifier nor the replay
    store is touched."""
    from fastapi import HTTPException
    from unittest.mock import AsyncMock

    from kestrel_sovereign.a2a import envelope_signing
    from kestrel_sovereign.endpoints import agent as agent_endpoint

    verifier = AsyncMock(side_effect=AssertionError("verified before the guard"))
    monkeypatch.setattr(envelope_signing, "verify_inbound_envelope", verifier)
    replay = MagicMock(side_effect=AssertionError("replay store touched"))
    monkeypatch.setattr(agent_endpoint, "_a2a_replay_store", replay)

    task_manager = MagicMock()
    task_manager.create_task = AsyncMock()
    agent = SimpleNamespace(did="", _agent_name="recipient", task_manager=task_manager)
    params = _send_params()
    with pytest.raises(HTTPException) as excinfo:
        await agent_endpoint._create_a2a_task_under_lifecycle_lease(
            agent, params, params.message.parts, [], [], manager=None
        )
    assert excinfo.value.status_code == 503
    verifier.assert_not_awaited()
    replay.assert_not_called()
    task_manager.create_task.assert_not_awaited()


# -- a2a/local_submission: the host-attested local writer ------------------


def test_local_submission_recipient_is_the_did_not_agent_id():
    from kestrel_sovereign.a2a.local_submission import _stable_agent_id

    assert _stable_agent_id(SimpleNamespace(agent_id=OTHER, did=ME)) == ME


@pytest.mark.parametrize("agent", _identity_cases() + [pytest.param(None, id="no-agent")])
def test_local_submission_has_no_recipient_without_a_did(agent):
    from kestrel_sovereign.a2a.local_submission import _stable_agent_id

    assert _stable_agent_id(agent) is None


# ---------------------------------------------------------------------------
# The census (review r4): every inline resolution left in the package is a
# known exception with a reason, and this test is the register — not a
# docstring. It fails when a copy appears or disappears.
# ---------------------------------------------------------------------------

import re
from pathlib import Path

_PACKAGE = Path(__file__).resolve().parents[2] / "kestrel_sovereign"


def _normalize(text: str) -> str:
    """One line, one space, no space inside parentheses: the scan sees the
    resolution, not the formatter's wrapping (review r5)."""
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\(\s+", "(", text)
    text = re.sub(r"\s+\)", ")", text)
    text = re.sub(r",\s*\)", ")", text)
    return text


# A loop that tries the two identity attributes in either order.
_TUPLE_LOOP = re.compile(r'for \w+ in \("(?:did|agent_id)", "(?:did|agent_id)"\)')
# Both identity attributes read from the SAME object within one bounded
# window, in either order and any nesting: an `or` chain, a nested default,
# a candidate tuple, or two statements.
_PAIR = re.compile(
    r'getattr\((?P<obj>[\w.]+), "(?P<a>did|agent_id)"[^\n]{0,240}?'
    r'getattr\((?P=obj), "(?P<b>did|agent_id)"'
)


def _count_inline(text: str) -> int:
    norm = _normalize(text)
    hits = len(_TUPLE_LOOP.findall(norm))
    pos = 0
    while True:
        match = _PAIR.search(norm, pos)
        if match is None:
            break
        if match.group("a") != match.group("b"):
            hits += 1
        # Non-overlapping: one resolution counts once. An overlapping scan
        # paired one chain's `did` with the next chain's `agent_id` inside
        # the window and counted a third resolution that does not exist.
        pos = match.end()
    return hits


# path (relative to the package) -> (count, why it is not routed through the guard)
KNOWN_INLINE_RESOLUTIONS = {
    # #3251: other shared tables, filed separately from the a2a_tasks work.
    "endpoints/agent.py": (
        2,
        "the reflection-status route scopes task_execution_log (#3251); the "
        "local-witness loop resolves the SENDER's stable identity from a "
        "witnessed peer object, not this agent's own scope",
    ),
    "features/memory/reflection_hook.py": (1, "memory table scope (#3251)"),
    "features/health/checks.py": (1, "scheduler status scope (#3251)"),
    "services/key_resolution.py": (1, "service-key storage scope (#3251)"),
    "features/todo/feature.py": (1, "todo table scope, falls back to 'default' (#3251)"),
    "waits/reconciler.py": (
        2,
        "the wait-signal store scope ('' is the documented solo-agent legacy "
        "scope) and a signal's target agent (#3251)",
    ),
    "features/skills/feature.py": (1, "reflection_insights scope, '' fallback (#3251)"),
    "features/bootstrap/feature.py": (
        1,
        "soul-seed created_by attribution, agent_id first (#3251)",
    ),
    "features/contribution_runtime.py": (
        1,
        "agent-scoped service registration key, agent_id first (#3251)",
    ),
    "features/scheduler/feature.py": (
        1,
        "scheduler watcher owner identity from a candidate tuple; raises when absent (#3251)",
    ),
    # The agent reading its own identity on `self`: on a real agent
    # `agent_id` is a property returning `did`, so nothing is resolved. Two
    # chains in one policy-context call.
    "agent/backup.py": (2, "the agent reads its own identity on self"),
    # Not table scopes: a filesystem namespace owner (raises on a missing
    # DID) and the hosted scheduler's identity match against a claim.
    "features/isolated_runtime.py": (1, "isolated-runtime namespace owner; raises when absent"),
    "features/scheduler/runner.py": (1, "hosted scheduler matches a resolved agent against a claimed id"),
    # The manager's routing/lineage identity loader — also the read end of
    # the host-attested local task route whose write end is routed. Its
    # suite builds agents carrying agent_id alone; routing it is a
    # suite-wide double change tracked in #3251.
    "multi_agent/agent_manager.py": (1, "manager identity loader; did then agent_id (#3251)"),
}


def _inline_resolutions() -> dict[str, int]:
    found: dict[str, int] = {}
    for path in sorted(_PACKAGE.rglob("*.py")):
        n = _count_inline(path.read_text())
        if n:
            found[str(path.relative_to(_PACKAGE))] = n
    return found


def test_every_inline_did_resolution_is_a_known_exception():
    # Positive controls: each shape counts exactly once on its own fixture,
    # so an empty census could never be a weakened scan passing vacuously.
    assert _count_inline('for attribute in ("did", "agent_id"):\n    pass') == 1
    assert _count_inline('for attribute in ("agent_id", "did"):\n    pass') == 1
    assert _count_inline('x = getattr(agent, "did", None) or getattr(agent, "agent_id", "")') == 1
    # Wrapped by a formatter (review r5): still one resolution.
    assert _count_inline('x = getattr(a.agent, "agent_id", None) or getattr(\n    a.agent, "did", None\n)') == 1
    # Nested default and a candidate tuple: still one each.
    assert _count_inline('getattr(agent, "agent_id", getattr(agent, "did", ""))') == 1
    assert _count_inline('(getattr(self.agent, "did", None), getattr(self.agent, "agent_id", None))') == 1
    # Two statements on one object: one resolution.
    assert _count_inline('did = getattr(self.agent, "did", None)\nreturn did or getattr(self.agent, "agent_id", "")') == 1
    # Different objects, or the same attribute twice, are not a resolution.
    assert _count_inline('getattr(storage, "agent_id", None) or getattr(self, "agent_id", "")') == 0
    assert _count_inline('getattr(agent, "did", None) or getattr(agent, "did", "")') == 0

    found = _inline_resolutions()
    expected = {path: count for path, (count, _why) in KNOWN_INLINE_RESOLUTIONS.items()}
    unexpected = {p: n for p, n in found.items() if p not in expected}
    assert not unexpected, (
        "inline DID resolution added — route it through resolve_scoped_agent_did, "
        f"or register it here with a reason: {unexpected}"
    )
    assert found == expected, (
        "the census drifted (a routed site restored, or a registered site gone): "
        f"found={found} expected={expected}"
    )


# -- a2a/inbound_authorization: the inbound-scope gate's recipient identity --


def test_inbound_authorizer_recipient_is_the_did_not_agent_id():
    from kestrel_sovereign.a2a.inbound_authorization import RecipientA2ASenderAuthorizer

    resolve = RecipientA2ASenderAuthorizer._stable_agent_id
    assert resolve(SimpleNamespace(agent_id=OTHER, did=ME)) == ME


@pytest.mark.parametrize("agent", _identity_cases() + [pytest.param(None, id="no-agent")])
def test_inbound_authorizer_has_no_recipient_without_a_did(agent):
    from kestrel_sovereign.a2a.inbound_authorization import RecipientA2ASenderAuthorizer

    assert RecipientA2ASenderAuthorizer._stable_agent_id(agent) is None


# The other half of the guard's docstring — the ROUTED sites — is a hand-
# written list too (review r7 found it one module short). This is its
# register: the modules that call the guard, asserted exactly.
ROUTED_MODULES = {
    "agent/preturn_state.py",
    "a2a/inbound_authorization.py",
    "a2a/local_submission.py",
    "command_handler.py",
    "endpoints/agent.py",
    "endpoints/observability.py",
    "endpoints/restart_events.py",
    "features/audit_anchor/feature.py",
    "features/consent/feature.py",
    "features/storage_access.py",  # the definition itself
    "features/tasks/feature.py",
    "features/tasks/wait_provider.py",
}


def test_every_routed_site_is_named_here():
    callers = {
        str(path.relative_to(_PACKAGE))
        for path in _PACKAGE.rglob("*.py")
        if "resolve_scoped_agent_did(" in path.read_text()
    }
    assert callers == ROUTED_MODULES, {
        "new callers (add them here and to the guard's docstring)": sorted(callers - ROUTED_MODULES),
        "callers gone (a site un-routed?)": sorted(ROUTED_MODULES - callers),
    }
