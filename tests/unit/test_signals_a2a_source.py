"""Phase 5 of #889: a2a.task_complete source registration + causation
chain propagation + the cycle detection mechanism's first real-world
exercise.

The headline test is `test_AB_A_ping_pong_rejected_at_depth_2` —
demonstrates that an A→B→A loop is caught by the dispatcher's cycle
check using the chain plumbed through `the declared native causation extension`.
TTL exhaustion is the catch-all when chains get long enough to wrap
around without same-source repetition.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
from kestrel_sdk.native_a2a.helpers.proto_helpers import new_text_message
from kestrel_sdk.native_a2a.types import AgentSkill, Role, SendMessageRequest, Task, TaskStatus
from google.protobuf.json_format import MessageToDict
from kestrel_sdk.signals import (
    CausationFrame,
    SignalMode,
    Status,
)

from kestrel_sovereign.a2a.lifecycle import task_state_from_label
from kestrel_sovereign.a2a.scope import NativeRecipientScope
from kestrel_sovereign.a2a.stores.unified.context_store import ContextOwnership
from kestrel_sovereign.a2a.stores.unified.task_store import NativeTaskSnapshot
from kestrel_sovereign.signals import (
    OrderedLockManager,
    SignalDispatcher,
    SignalLogStore,
    SourceRegistry,
)
from kestrel_sovereign.signals.sources.a2a import (
    METADATA_KEY,
    SOURCE_NAME,
    build_a2a_task_complete_registration,
    build_signal_for_completed_task,
    serialize_chain_for_metadata,
)
from kestrel_sovereign.storage.db import SQLiteBackend
from tests.unit.test_a2a_native_task_manager import CompletingProvider
from tests.utils.native_a2a import manager_for

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _task_snapshot(task_id="task-001", state="completed", summary_text="all done", metadata=None, *, recipient="did:test:A"):
    task = Task(id=task_id, context_id="context-" + task_id,
                status=TaskStatus(state=task_state_from_label(state)), metadata=metadata or {})
    if summary_text:
        task.status.message.CopyFrom(new_text_message(summary_text, role=Role.ROLE_AGENT))
    return NativeTaskSnapshot(task=task, lifecycle_revision=0, ownership=ContextOwnership(
        context_id=task.context_id, tenant="", creator_agent_id="did:test:sender", recipient_agent_id=recipient,
    ))


from kestrel_sovereign.agent.turn_lifecycle import TurnLifecycleMixin


class _FakeAgent(TurnLifecycleMixin):
    """Inherits TurnLifecycleMixin so the dispatcher's COGNITION route
    can call `_set_current_chain` / `_clear_current_chain` (#905 review
    P1 plumbing). Production `KestrelAgent` has the same inheritance."""

    def __init__(self, did: str = "did:test:A"):
        self._did = did
        self.background_tasks: list[asyncio.Task] = []
        self.process_input_calls: list[str] = []
        self.process_input_return: str = "ack"

    @property
    def did(self) -> str:
        return self._did

    async def process_input(self, prompt: str):
        self.process_input_calls.append(prompt)
        return self.process_input_return

    def _track_background_task(self, coro, *, name: str):
        task = asyncio.create_task(coro, name=name)
        self.background_tasks.append(task)
        return task


@pytest.fixture
async def components(tmp_path):
    backend = SQLiteBackend(str(tmp_path / "a2a_e2e.db"))
    await backend.connect()
    store = SignalLogStore(backend)
    await store.initialize()
    registry = SourceRegistry()
    locks = OrderedLockManager()
    agent = _FakeAgent()
    dispatcher = SignalDispatcher(
        agent=agent, registry=registry, lock_manager=locks, store=store,
    )
    registry.register(build_a2a_task_complete_registration())
    yield SimpleNamespace(
        agent=agent, registry=registry, dispatcher=dispatcher, backend=backend,
    )
    pending = [t for t in agent.background_tasks if not t.done()]
    if pending:
        await asyncio.gather(*pending, return_exceptions=True)
    await backend.close()


# ---------------------------------------------------------------------------
# Source registration shape
# ---------------------------------------------------------------------------


def test_registration_is_cognition_only_and_trusted():
    reg = build_a2a_task_complete_registration()
    assert reg.name == SOURCE_NAME
    assert reg.allowed_modes == frozenset({SignalMode.COGNITION})
    assert reg.prompt_template.exists()
    # A2A peers are TRUSTED in v1; sanitizer not required because
    # schema validates the structured fields. Untrusted peers would
    # need both trust=UNTRUSTED and a sanitizer.
    from kestrel_sdk.signals import Trust
    assert reg.trust == Trust.TRUSTED
    # Self-loops disabled — that IS the cycle detection mechanism for
    # this source.
    assert reg.allow_self_loops is False


def test_registration_schema_rejects_missing_required_fields():
    reg = build_a2a_task_complete_registration()
    with pytest.raises(ValueError, match="missing required key"):
        reg.schema({"task_id": "x", "task_state": "completed"})


def test_registration_redaction_caps_long_summaries():
    reg = build_a2a_task_complete_registration()
    long_text = "x" * 500
    redacted = reg.log_redaction.summarize({
        "task_id": "abc",
        "task_state": "completed",
        "result_summary": long_text,
    })
    assert "task_id=abc" in redacted
    assert "...(truncated)" in redacted
    assert len(redacted) < 500  # capped


# ---------------------------------------------------------------------------
# Signal builder
# ---------------------------------------------------------------------------


def test_build_signal_extracts_text_from_status_message():
    task = _task_snapshot(
        task_id="t-1", state="completed", summary_text="briefing ready"
    )
    sig = build_signal_for_completed_task(task, target_agent="did:test:A")
    assert sig.source == SOURCE_NAME
    assert sig.target_agent == "did:test:A"
    assert sig.payload["task_id"] == "t-1"
    assert sig.payload["task_state"] == "completed"
    assert sig.payload["result_summary"] == "briefing ready"


def test_build_signal_falls_back_when_no_status_message():
    task = _task_snapshot(
        task_id="t-2", state="failed", summary_text=""
    )
    sig = build_signal_for_completed_task(task, target_agent="did:test:A")
    assert "no message" in sig.payload["result_summary"]
    assert "failed" in sig.payload["result_summary"]


def test_build_signal_with_no_metadata_yields_empty_chain():
    task = _task_snapshot(metadata={})
    sig = build_signal_for_completed_task(task, target_agent="did:test:A")
    assert sig.causation_chain == []


def test_build_signal_rehydrates_chain_from_metadata():
    earlier_frame = CausationFrame(
        agent_id="did:test:A",
        source="heartbeat",
        signal_id="sig-old",
        turn_id="turn-1",
        depth=1,
        emitted_at=datetime(2026, 5, 1, 9, 0, tzinfo=timezone.utc),
    )
    task = _task_snapshot(metadata={
        METADATA_KEY: {"version": 1, "chain": serialize_chain_for_metadata([earlier_frame])},
    })
    sig = build_signal_for_completed_task(task, target_agent="did:test:A")
    assert len(sig.causation_chain) == 1
    rehydrated = sig.causation_chain[0]
    assert rehydrated.agent_id == "did:test:A"
    assert rehydrated.source == "heartbeat"
    assert rehydrated.depth == 1
    # CausationFrame is frozen — equality holds across serialize/deserialize.
    assert rehydrated == earlier_frame


def test_build_signal_refuses_malformed_frames_in_declared_extension():
    snapshot = _task_snapshot(metadata={METADATA_KEY: {"version":1, "chain":[{"not_a_frame":True}]}})
    with pytest.raises(ValueError, match="causation frame has invalid fields"):
        build_signal_for_completed_task(snapshot, target_agent="did:test:A")


# ---------------------------------------------------------------------------
# End-to-end through the real dispatcher
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_completed_task_wakes_the_bird(components):
    """Happy path: task completes locally → signal builder produces
    envelope → dispatcher routes COGNITION → agent.process_input
    receives the rendered prompt."""
    c = components
    task = _task_snapshot(task_id="t-happy", summary_text="briefing X done")
    sig = build_signal_for_completed_task(task, target_agent=c.agent.did)

    result = await c.dispatcher.dispatch_signal(sig)
    assert result.status == Status.OK
    assert "briefing X done" in c.agent.process_input_calls[0]
    assert "task_id" in c.agent.process_input_calls[0]


@pytest.mark.asyncio
async def test_AB_A_ping_pong_rejected_at_depth_2(components):
    """The headline cycle-detection test for #894.

    Scenario: agent A spawns task to peer B. B completes it. The
    completion fires locally on A; A's dispatcher receives a Signal
    whose chain carries the previous (A, a2a.task_complete) frame from
    a prior round trip. The dispatcher's cycle check spots A appearing
    twice with the same source and rejects.

    Without this guard, A and B could ping-pong indefinitely once
    task-completion becomes COGNITION (each completion triggering a
    new outbound task during the woken turn).
    """
    c = components

    # Simulate the prior frame: A previously received an
    # a2a.task_complete signal at depth 1.
    prior_frame = CausationFrame(
        agent_id=c.agent.did,
        source=SOURCE_NAME,
        signal_id="sig-prior",
        turn_id="turn-prior",
        depth=1,
        emitted_at=datetime.now(timezone.utc),
    )
    task = _task_snapshot(metadata={
        METADATA_KEY: {"version": 1, "chain": serialize_chain_for_metadata([prior_frame])},
    })
    sig = build_signal_for_completed_task(task, target_agent=c.agent.did)

    result = await c.dispatcher.dispatch_signal(sig)
    assert result.status == Status.DROPPED_CYCLE
    assert "Cycle detected" in (result.error or "")
    assert c.agent.did in (result.error or "")
    assert SOURCE_NAME in (result.error or "")
    # No LLM call must have happened.
    assert c.agent.process_input_calls == []


@pytest.mark.asyncio
async def test_AB_A_first_round_trip_passes(components):
    """Counterpoint to the cycle test: the FIRST A→B→A round trip
    must succeed (no prior A frame in the chain). This proves cycle
    detection is targeted at loops, not at all repeat work."""
    c = components

    # First round trip: A's outbound task carried no prior frames
    # (or only frames from non-A agents). Empty chain is the most
    # common case for v1 since outbound chain plumbing is a follow-up.
    task = _task_snapshot(metadata={})
    sig = build_signal_for_completed_task(task, target_agent=c.agent.did)

    result = await c.dispatcher.dispatch_signal(sig)
    assert result.status == Status.OK
    assert len(c.agent.process_input_calls) == 1


@pytest.mark.asyncio
async def test_ttl_exhaustion_rejected(components):
    """Long chain (depth > 5 default TTL) is rejected even when no
    same-(agent, source) repeat exists. Catches the case where a
    long C→D→E→F→A chain wraps around without explicit cycles."""
    c = components

    # Build a 5-frame chain with distinct agents/sources — no repeats.
    chain = [
        CausationFrame(
            agent_id=f"did:test:peer-{i}",
            source=f"peer.source.{i}",
            signal_id=f"sig-{i}",
            turn_id=f"turn-{i}",
            depth=i,
            emitted_at=datetime.now(timezone.utc),
        )
        for i in range(1, 6)
    ]
    task = _task_snapshot(metadata={
        METADATA_KEY: {"version": 1, "chain": serialize_chain_for_metadata(chain)},
    })
    sig = build_signal_for_completed_task(task, target_agent=c.agent.did)
    # Adding one more frame puts depth at 6 > TTL (5).

    result = await c.dispatcher.dispatch_signal(sig)
    assert result.status == Status.DROPPED_CYCLE
    assert "exceeds TTL" in (result.error or "")


# ---------------------------------------------------------------------------
# SSE notification path preserved
# ---------------------------------------------------------------------------


async def test_event_manager_callback_keeps_ui_notification_and_native_cognition(tmp_path):
    from kestrel_sovereign.agent.event_manager import EventManagerMixin
    backend = SQLiteBackend(str(tmp_path / "native-ui.db"))
    await backend.connect()
    manager = manager_for(backend, NativeRecipientScope("did:test:sse"), owns_backend=False)
    try:
        await manager.initialize()
        store = SignalLogStore(backend)
        await store.initialize()
        registry = SourceRegistry()
        registry.register(build_a2a_task_complete_registration())
        class NativeAgent(EventManagerMixin):
            did = "did:test:sse"
            def __init__(self):
                self.task_manager = manager
                self._event_listeners = []
                self._pending_task_notifications = []
                self._background_tasks = []
                self.dispatcher = SignalDispatcher(agent=self, registry=registry, lock_manager=OrderedLockManager(), store=store)
            async def process_input(self, prompt):
                return "ack"
            def _track_background_task(self, coro, *, name):
                task = asyncio.create_task(coro, name=name)
                self._background_tasks.append(task)
                return task
        agent = NativeAgent()
        accepted = await manager.create_task(
            SendMessageRequest(message=new_text_message("Work", role=Role.ROLE_USER), metadata={"skill":"do_thing"}),
            creator_agent_id="did:test:sender",
        )
        await manager.complete_task(accepted.id, "Done", recipient_agent_id=agent.did)
        candidate = await manager.get_task_snapshot(accepted.id, recipient_agent_id=agent.did)
        agent._on_background_task_complete(candidate)
        assert len(agent._pending_task_notifications) == 1
        assert "Background task completed" in agent._pending_task_notifications[0]
        assert "did:test:sender/do_thing" in agent._pending_task_notifications[0]
        assert agent._background_tasks
        while pending := [t for t in agent._background_tasks if not t.done()]:
            await asyncio.gather(*pending, return_exceptions=True)
        rows = await backend.fetch_all("SELECT source, status FROM signal_log WHERE source=?", (SOURCE_NAME,))
        assert rows == [(SOURCE_NAME, Status.OK.value)]
    finally:
        await manager.close()
        await backend.close()


@pytest.mark.asyncio
async def test_duplicate_terminal_completions_coalesced(components):
    """#905 review P2: build_signal_for_completed_task sets
    dedupe_key=`<task_id>:<terminal_state>` so retry storms or
    double-fired terminal callbacks for the same task collapse within
    the registration's coalescing_window (5s default). Without this
    the dispatcher coalescing pipeline can't fire — dedupe_key=None
    skips coalescing entirely."""
    c = components
    task = _task_snapshot(task_id="dup-1", state="completed", summary_text="done")

    sig1 = build_signal_for_completed_task(task, target_agent=c.agent.did)
    sig2 = build_signal_for_completed_task(task, target_agent=c.agent.did)

    # Sanity: both signals carry the same dedupe_key.
    assert sig1.dedupe_key == sig2.dedupe_key == "dup-1:completed"

    r1 = await c.dispatcher.dispatch_signal(sig1)
    r2 = await c.dispatcher.dispatch_signal(sig2)

    assert r1.status == Status.OK
    assert r2.status == Status.COALESCED, (
        f"second completion of task dup-1 must collapse; got {r2.status}"
    )
    # Only one LLM call.
    assert len(c.agent.process_input_calls) == 1


# ---------------------------------------------------------------------------
# Outbound chain plumbing (#905 review P1) — full loop through real
# TaskManager.create_task
# ---------------------------------------------------------------------------


async def test_native_local_task_attaches_the_current_causation_extension(tmp_path):
    frame = CausationFrame(agent_id="did:test:prior", source=SOURCE_NAME, signal_id="prior", turn_id="prior-turn", depth=1, emitted_at=datetime.now(timezone.utc))
    chain = serialize_chain_for_metadata([frame])
    manager = await _manager_with_chain(tmp_path, lambda: chain)
    try:
        accepted = await manager.execute_skill("feature", "do_thing", {}, sync=False)
        await manager.drain_execution_tasks()
        candidate = await manager.get_task_snapshot(accepted.id, recipient_agent_id="did:test:A")
        metadata = MessageToDict(candidate.task.metadata)
        assert metadata["provider_id"] == "feature" and metadata["skill"] == "do_thing"
        assert metadata[METADATA_KEY] == {"version":1, "chain":chain}
        assert build_signal_for_completed_task(candidate, target_agent="did:test:A").causation_chain == [frame]
    finally:
        await manager.close()


async def test_native_local_task_omits_causation_when_no_turn_exists(tmp_path):
    manager = await _manager_with_chain(tmp_path, lambda: None)
    try:
        accepted = await manager.execute_skill("feature", "do_thing", {}, sync=False)
        await manager.drain_execution_tasks()
        candidate = await manager.get_task_snapshot(accepted.id, recipient_agent_id="did:test:A")
        assert METADATA_KEY not in MessageToDict(candidate.task.metadata)
    finally:
        await manager.close()


async def test_completion_cycle_via_native_local_taskmanager_is_rejected(components, tmp_path):
    c = components
    def current_chain():
        frames = c.agent._get_current_chain()
        return serialize_chain_for_metadata(frames) if frames else None
    manager = await _manager_with_chain(tmp_path, current_chain)
    accepted = []
    async def turn_emits_native_work(prompt):
        accepted.append(await manager.execute_skill("feature", "do_thing", {}, sync=False))
        return "ack"
    c.agent.process_input = turn_emits_native_work
    try:
        first = build_signal_for_completed_task(_task_snapshot("first"), target_agent=c.agent.did)
        assert (await c.dispatcher.dispatch_signal(first)).status is Status.OK
        assert len(accepted) == 1
        await manager.drain_execution_tasks()
        candidate = await manager.get_task_snapshot(accepted[0].id, recipient_agent_id=c.agent.did)
        restored = build_signal_for_completed_task(candidate, target_agent=c.agent.did)
        assert len(restored.causation_chain) == 1
        assert restored.causation_chain[0].agent_id == c.agent.did
        assert restored.causation_chain[0].source == SOURCE_NAME
        assert (await c.dispatcher.dispatch_signal(restored)).status is Status.DROPPED_CYCLE
        assert len(accepted) == 1
    finally:
        await manager.close()


async def test_event_manager_callback_works_without_dispatcher(tmp_path):
    """A partially initialized agent still queues the task UI notification."""
    from kestrel_sovereign.agent.event_manager import EventManagerMixin


    class _AgentLikeNoDispatcher(EventManagerMixin):
        did = "did:test:no-dispatcher"
        def __init__(self):
            self._event_listeners = []
            self._pending_task_notifications = []
            # NOTE: no self.dispatcher

    agent = _AgentLikeNoDispatcher()

    real_task = _task_snapshot(
        task_id="task-ui", recipient=agent.did, metadata={"skill":"do_thing"},
    )

    # Must not raise even though dispatcher is absent.
    agent._on_background_task_complete(real_task)
    assert len(agent._pending_task_notifications) == 1


@pytest.mark.asyncio
async def test_durable_consumer_guard_reads_registration_not_drainer_state(tmp_path):
    """The guard is a registration fact, not "is a drainer running".

    ``_started_durable_cognition_consumers`` is empty in the boot window
    between ``register_durable_consumer`` and
    ``start_durable_cognition_consumer``, is discarded by
    ``deactivate_consumer``, and is cleared wholesale at shutdown. Gating the
    a2a wake on it drops the wake back to a non-durable, non-retried
    ``enqueue_signal`` during exactly the windows #3163 exists to close.
    """
    from kestrel_sovereign.signals.durable import DurableConsumerRegistration
    from kestrel_sovereign.signals.sources.a2a_task_submitted import (
        DURABLE_COGNITION_CONSUMER_ID,
    )

    backend = SQLiteBackend(str(tmp_path / "guard.db"))
    await backend.connect()
    try:
        store = SignalLogStore(backend)
        await store.initialize()
        registry = SourceRegistry()
        registry.register(build_a2a_task_complete_registration())

        class _Agent:
            did = "did:test:guard"

            async def process_input(self, prompt):
                return "ack"

            def _track_background_task(self, coro, *, name):
                return asyncio.create_task(coro, name=name)

        agent = _Agent()
        dispatcher = SignalDispatcher(
            agent=agent,
            registry=registry,
            lock_manager=OrderedLockManager(),
            store=store,
        )

        assert await dispatcher.has_durable_consumer(
            DURABLE_COGNITION_CONSUMER_ID
        ) is False

        await dispatcher.register_durable_consumer(
            DurableConsumerRegistration(
                consumer_id=DURABLE_COGNITION_CONSUMER_ID,
                source="a2a.task_complete",
                agent_id=agent.did,
            )
        )

        # The boot window: registered, but no drainer has started yet.
        assert dispatcher._started_durable_cognition_consumers == set()
        assert await dispatcher.has_durable_consumer(
            DURABLE_COGNITION_CONSUMER_ID
        ) is True, "a registered consumer must count before its drainer starts"
    finally:
        await backend.close()


@pytest.mark.asyncio
async def test_an_ask_tool_in_a_woken_turn_is_denied_not_queued(
    components, tmp_path
):
    """#3439 wiring. The dispatcher must bind the causation chain around the
    turn it routes, so the security hook running inside that turn sees a
    signal-driven turn and denies an ASK tool at once instead of queuing an
    approval nobody in the conversation can answer. Outside the turn the
    chain is unbound again."""
    from kestrel_sdk.hooks.base import HookInput, PermissionDecision

    from kestrel_sovereign.agent.turn_lifecycle import (
        current_turn_is_signal_driven,
    )
    from kestrel_sovereign.features.security.approval_queue import (
        ApprovalQueue,
    )
    from kestrel_sovereign.features.security.hooks import SecurityHook
    from kestrel_sovereign.features.security.permissions import (
        PermissionLevel,
        PermissionStore,
    )

    c = components
    store = PermissionStore(str(tmp_path / "perms.db"))
    await store.initialize()
    await store.register_tool("MemoryFeature", "memory_status", PermissionLevel.ASK)
    queue = ApprovalQueue(permission_store=store)
    hook = SecurityHook(store, queue)
    outputs = []

    async def turn_calls_ask_tool(prompt):
        outputs.append(
            await asyncio.wait_for(
                hook.execute(
                    HookInput(
                        session_id="a2a-session",
                        hook_event_name="PreToolUse",
                        tool_name="memory_status",
                        feature_name="MemoryFeature",
                        tool_input={},
                    )
                ),
                timeout=1.0,
            )
        )
        return "needs operator approval"

    c.agent.process_input = turn_calls_ask_tool
    sig = build_signal_for_completed_task(
        _task_snapshot(task_id="t-ask"), target_agent=c.agent.did
    )
    result = await c.dispatcher.dispatch_signal(sig)

    assert result.status == Status.OK
    assert len(outputs) == 1
    assert outputs[0].permission_decision == PermissionDecision.DENY
    assert "MemoryFeature.memory_status" in outputs[0].permission_reason
    assert queue.pending_requests == []
    assert current_turn_is_signal_driven() is False





async def _manager_with_chain(tmp_path, provider):
    backend = SQLiteBackend(str(tmp_path / "native-causation.db"))
    await backend.connect()
    manager = manager_for(backend, NativeRecipientScope("did:test:A"), causation_chain_provider=provider)
    await manager.initialize()
    manager.register_provider("feature", [AgentSkill(
        id="do_thing", name="Do thing", description="Native causation", tags=["test"],
    )], CompletingProvider())
    return manager
