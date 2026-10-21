"""Provider/handle ownership validation + restart re-arm for wait watches (#2729).

Two guarantees this suite pins:

1. **Ownership validation at registration.** ``register_wait_watch`` rejects a
   handle the named provider does not own — synchronously, before any durable
   watch row or misleading terminal ``wait.complete`` failure is produced. The
   canonical bug: ``task:<outbound-a2a-id>`` was accepted, then the local task
   provider read "not found" and the reconciler emitted a false terminal
   failure. Now it fails up front and points at the ``a2a:`` provider.

2. **Valid A2A/CI waits re-arm across restart and complete once.** A durable
   ``mode="signal"`` watch registered before a restart is picked up by a fresh
   reconciler (new in-memory state, same DB) and emits exactly one signal when
   the handle settles.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from uuid import uuid4

import pytest
from a2a.helpers.proto_helpers import new_text_message
from a2a.types import (
    AgentInterface,
    Role,
    SendMessageRequest,
    Task,
    TaskState,
    TaskStatus,
)
from kestrel_sdk.signals import Status
from kestrel_sdk.tools.result import ToolResult

from kestrel_sovereign.a2a.lifecycle import task_state_from_label
from kestrel_sovereign.a2a.scope import NativeRecipientScope
from kestrel_sovereign.a2a.stores.unified.outbound_store import OutboundStore
from kestrel_sovereign.features.peers.wait_provider import A2AWaitable
from kestrel_sovereign.features.scheduler.ci_wait_provider import CIWaitable
from kestrel_sovereign.features.tasks.wait_provider import TaskWaitable
from kestrel_sovereign.signals.sources.github_pr_watch import CheckRollup
from kestrel_sovereign.storage.async_wait_signal_store import WaitSignalStore
from kestrel_sovereign.waits.engine import WaitRegistry
from kestrel_sovereign.waits.reconciler import (
    register_wait_watch,
    run_wait_reconcile,
)

AGENT_ID = "did:test:sender"
OUTBOUND_ID = "c3f404fb77df4b79b0508a68ea46bbb7"


class _CapturingDispatcher:
    """Collects enqueued signals; task resolves OK (harvested next tick)."""

    def __init__(self):
        self.signals = []

    async def enqueue_signal(self, signal):
        from kestrel_sdk.signals.models import SignalHandle, SignalResult

        self.signals.append(signal)

        async def _coro():
            return SignalResult(
                signal_id=signal.id, status=Status.OK, mode=signal.mode,
                duration_ms=1, error=None,
            )

        task = asyncio.create_task(_coro())
        await task
        return SignalHandle(signal_id=signal.id, task=task)


class _StubTM:
    def __init__(self, tasks):
        self._tasks = tasks

    async def get_task(self, task_id):
        return self._tasks.get(task_id)

    async def get_task_for_recipient(self, task_id, _recipient_agent_id):
        return self._tasks.get(task_id)


class _StubTaskFeature:
    def __init__(self, tasks):
        self.task_manager = _StubTM(tasks)
        # Principal-bound task reads require the same recipient DID that the
        # production TaskFeature carries on its owning agent.
        self.agent = SimpleNamespace(did=AGENT_ID)


class _NativeLedgerPeers:
    def __init__(self, db, agent, states):
        self.agent = agent
        self._states = states
        self._outbound = OutboundStore(db.backend, sender_scope=NativeRecipientScope(agent.did))

    def _native_outbound_store(self):
        return self._outbound

    async def get_peer_task_result(self, dispatch_id):
        state = self._states.get(dispatch_id)
        if state is None:
            return ToolResult.failed("unreachable", data={"dispatch_id": dispatch_id})
        current = await self._outbound.get(dispatch_id)
        await self._outbound.acknowledge(
            current,
            Task(id=current.remote_task_id, context_id=current.remote_context_id,
                 status=TaskStatus(state=task_state_from_label(state))),
            operation_id=uuid4().hex,
        )
        return ToolResult.ok("fetched", data={"dispatch_id": dispatch_id, "state": state})


async def _reserve_dispatch(db, *, dispatch_id, verb):
    store = OutboundStore(db.backend, sender_scope=NativeRecipientScope(AGENT_ID))
    request = SendMessageRequest(message=new_text_message(
        "Native tracked work", role=Role.ROLE_USER, context_id="peer-context",
    ))
    dispatch = await store.reserve(
        dispatch_id, request, recipient=NativeRecipientScope("did:peer:nellie"),
        signing_did="did:web:peer.test", dispatch_tool="send_a2a_" + verb, verb=verb,
        interface=AgentInterface(url="https://peer.test/a2a", protocol_binding="JSONRPC", protocol_version="1.0"),
    )
    submitting = await store.begin_submission(dispatch, operation_id=uuid4().hex)
    await store.acknowledge(
        submitting,
        Task(id="remote-" + dispatch_id, context_id="peer-context", status=TaskStatus(state=TaskState.TASK_STATE_WORKING)),
        operation_id=uuid4().hex,
    )


@pytest.fixture
async def db(tmp_path, sqlite_database_factory):
    database = await sqlite_database_factory(tmp_path / "agent.db")
    await OutboundStore(database.backend, sender_scope=NativeRecipientScope(AGENT_ID)).initialize()
    return database


def _make_agent(db, dispatcher=None):
    return SimpleNamespace(
        did=AGENT_ID,
        agent_id=AGENT_ID,
        _raw_storage=SimpleNamespace(db=db),
        wait_registry=WaitRegistry(),
        dispatcher=dispatcher or _CapturingDispatcher(),
    )


# ---------------------------------------------------------------------------
# Negative: task:<outbound-a2a-id> is rejected at registration
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_task_watch_on_outbound_a2a_id_is_rejected(db):
    """The exact #2729 failure: registering an outbound A2A task id as
    ``task:<id>`` must fail synchronously — no durable watch, no signal."""
    agent = _make_agent(db)
    # The id is an OUTBOUND A2A dispatch (in the outbound audit), NOT a local
    # background task.
    await _reserve_dispatch(db, dispatch_id=OUTBOUND_ID, verb='task')
    # Local task provider: the id is NOT in the local task store.
    agent.wait_registry.register(TaskWaitable(_StubTaskFeature(tasks={})))
    # The a2a provider DOES own it → used for the cross-provider hint.
    agent.wait_registry.register(A2AWaitable(_NativeLedgerPeers(db, agent, states={})))

    with pytest.raises(ValueError) as exc:
        await register_wait_watch(agent, f"task:{OUTBOUND_ID}")

    msg = str(exc.value)
    assert "not a valid 'task' wait handle" in msg
    # Cross-provider hint points at the correct provider.
    assert f"did you mean a2a:{OUTBOUND_ID}" in msg

    # No durable watch was created, and no reconciler/signal was produced.
    assert getattr(agent, "_wait_reconciler", None) is None
    store = WaitSignalStore(db, AGENT_ID)
    assert await store.list_watched() == []


@pytest.mark.asyncio
async def test_valid_local_task_watch_is_accepted(db):
    """Control: a real local task id is accepted (owns_handle True)."""
    agent = _make_agent(db)
    agent.wait_registry.register(
        TaskWaitable(_StubTaskFeature(tasks={"local-1": object()}))
    )
    await register_wait_watch(agent, "task:local-1")
    store = WaitSignalStore(db, AGENT_ID)
    assert {(w.kind, w.handle) for w in await store.list_watched()} == {
        ("task", "local-1")
    }


@pytest.mark.asyncio
async def test_a2a_watch_on_owned_outbound_id_is_accepted(db):
    """The same id, registered with the CORRECT provider, is accepted."""
    agent = _make_agent(db)
    await _reserve_dispatch(db, dispatch_id=OUTBOUND_ID, verb='task')
    agent.wait_registry.register(A2AWaitable(_NativeLedgerPeers(db, agent, states={})))
    await register_wait_watch(agent, f"a2a:{OUTBOUND_ID}")
    store = WaitSignalStore(db, AGENT_ID)
    assert {(w.kind, w.handle) for w in await store.list_watched()} == {
        ("a2a", OUTBOUND_ID)
    }


@pytest.mark.asyncio
async def test_a2a_watch_on_outbound_question_is_rejected(db):
    """#2729 P2: an outbound QUESTION id must be rejected as an ``a2a:`` watch
    — the question already resumes via a2a.question_answered, so a second
    wait.complete rail would double-wake. No durable watch is created."""
    agent = _make_agent(db)
    question_id = "question-abc123"
    await _reserve_dispatch(db, dispatch_id=question_id, verb='question')
    agent.wait_registry.register(A2AWaitable(_NativeLedgerPeers(db, agent, states={})))

    with pytest.raises(ValueError) as exc:
        await register_wait_watch(agent, f"a2a:{question_id}")
    assert "not a valid 'a2a' wait handle" in str(exc.value)

    # No durable watch row was created for the rejected question.
    store = WaitSignalStore(db, AGENT_ID)
    assert await store.list_watched() == []


# ---------------------------------------------------------------------------
# Restart re-arm + complete-once
# ---------------------------------------------------------------------------


async def _drain_pair(agent):
    """One enqueue tick + one harvest tick."""
    await run_wait_reconcile(agent)
    await run_wait_reconcile(agent)


@pytest.mark.asyncio
async def test_a2a_watch_rearms_across_restart_and_completes_once(db):
    dispatcher = _CapturingDispatcher()
    agent = _make_agent(db, dispatcher)
    await _reserve_dispatch(db, dispatch_id=OUTBOUND_ID, verb='task')
    states = {OUTBOUND_ID: "submitted"}  # still in flight
    provider = A2AWaitable(_NativeLedgerPeers(db, agent, states=states))
    agent.wait_registry.register(provider)

    await register_wait_watch(agent, f"a2a:{OUTBOUND_ID}")

    # Pre-restart tick while the peer task is still submitted → no signal.
    await run_wait_reconcile(agent)
    assert dispatcher.signals == []

    # --- Restart: drop the in-memory reconciler (durable watch row stays). ---
    agent._wait_reconciler = None
    store = WaitSignalStore(db, AGENT_ID)
    assert {(w.kind, w.handle) for w in await store.list_watched()} == {
        ("a2a", OUTBOUND_ID)
    }, "durable watch must survive restart"

    # Peer completes after restart.
    states[OUTBOUND_ID] = "completed"
    await _drain_pair(agent)
    assert len(dispatcher.signals) == 1
    sig = dispatcher.signals[0]
    assert sig.payload["kind"] == "a2a"
    assert sig.payload["handle"] == OUTBOUND_ID
    assert sig.payload["outcome"] == "done"

    # Complete once: further ticks must NOT re-emit.
    await _drain_pair(agent)
    assert len(dispatcher.signals) == 1


@pytest.mark.asyncio
async def test_ci_watch_rearms_across_restart_and_completes_once(db, monkeypatch):
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_test")
    dispatcher = _CapturingDispatcher()
    agent = _make_agent(db, dispatcher)

    provider = CIWaitable(feature=None)
    # Start open with checks in progress → non-terminal.
    fetch_result = {
        "value": (
            {"state": "open", "merged": False},
            CheckRollup(
                check_runs={"check_runs": [{"name": "ci", "status": "in_progress"}]},
                combined_status={"state": "pending"},
            ),
        ),
    }

    async def fake_fetch(repo, number, token):
        return fetch_result["value"]

    provider._fetch = fake_fetch
    agent.wait_registry.register(provider)

    await register_wait_watch(agent, "ci:owner/repo#7")

    await run_wait_reconcile(agent)
    assert dispatcher.signals == []

    # --- Restart ---
    agent._wait_reconciler = None
    store = WaitSignalStore(db, AGENT_ID)
    assert {(w.kind, w.handle) for w in await store.list_watched()} == {
        ("ci", "owner/repo#7")
    }

    # PR merges after restart → terminal DONE.
    fetch_result["value"] = ({"state": "closed", "merged": True}, CheckRollup())
    await _drain_pair(agent)
    assert len(dispatcher.signals) == 1
    sig = dispatcher.signals[0]
    assert sig.payload["kind"] == "ci"
    assert sig.payload["handle"] == "owner/repo#7"
    assert sig.payload["outcome"] == "done"

    await _drain_pair(agent)
    assert len(dispatcher.signals) == 1
