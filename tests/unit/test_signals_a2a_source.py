"""Phase 5 of #889: a2a.task_complete source registration + causation
chain propagation + the cycle detection mechanism's first real-world
exercise.

The headline test is `test_AB_A_ping_pong_rejected_at_depth_2` —
demonstrates that an A→B→A loop is caught by the dispatcher's cycle
check using the chain plumbed through `task.metadata["causation_chain"]`.
TTL exhaustion is the catch-all when chains get long enough to wrap
around without same-source repetition.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from kestrel_sdk.signals import (
    CausationFrame,
    SignalMode,
    Status,
    Visibility,
)
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


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _fake_task(
    task_id: str = "task-001",
    state: str = "completed",
    summary_text: str = "all done",
    metadata: dict | None = None,
):
    """Duck-typed Task object — avoids constructing a real pydantic
    Task in unit tests. The signal builder reads .id, .status.state,
    .status.message.parts[*].text, .metadata."""

    class _State:
        value = state

    part = SimpleNamespace(text=summary_text) if summary_text else None
    message = SimpleNamespace(parts=[part]) if part else None
    status = SimpleNamespace(state=_State(), message=message)
    return SimpleNamespace(id=task_id, status=status, metadata=metadata or {})


from kestrel_sovereign.agent.turn_lifecycle import TurnLifecycleMixin


class _FakeAgent(TurnLifecycleMixin):
    """Inherits TurnLifecycleMixin so the dispatcher's COGNITION route
    can call `_set_current_chain` / `_clear_current_chain` (#905 review
    P1 plumbing). Production `KestrelAgent` has the same inheritance."""

    def __init__(self, did: str = "did:test:agent-A"):
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
    task = _fake_task(
        task_id="t-1", state="completed", summary_text="briefing ready"
    )
    sig = build_signal_for_completed_task(task, target_agent="did:test:A")
    assert sig.source == SOURCE_NAME
    assert sig.target_agent == "did:test:A"
    assert sig.payload["task_id"] == "t-1"
    assert sig.payload["task_state"] == "completed"
    assert sig.payload["result_summary"] == "briefing ready"


def test_build_signal_falls_back_when_no_status_message():
    task = _fake_task(
        task_id="t-2", state="failed", summary_text=""
    )
    sig = build_signal_for_completed_task(task, target_agent="did:test:A")
    assert "no message" in sig.payload["result_summary"]
    assert "failed" in sig.payload["result_summary"]


def test_build_signal_with_no_metadata_yields_empty_chain():
    task = _fake_task(metadata={})
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
    task = _fake_task(metadata={
        METADATA_KEY: serialize_chain_for_metadata([earlier_frame]),
    })
    sig = build_signal_for_completed_task(task, target_agent="did:test:A")
    assert len(sig.causation_chain) == 1
    rehydrated = sig.causation_chain[0]
    assert rehydrated.agent_id == "did:test:A"
    assert rehydrated.source == "heartbeat"
    assert rehydrated.depth == 1
    # CausationFrame is frozen — equality holds across serialize/deserialize.
    assert rehydrated == earlier_frame


def test_build_signal_drops_malformed_frames_in_metadata():
    """Defensive: A2A messages cross the wire; metadata could be
    corrupted. Drop bad frames silently rather than crashing the signal
    builder (the dispatcher's logs will show the partial chain)."""
    task = _fake_task(metadata={
        METADATA_KEY: [
            {"not_a_frame": True},
            {
                "agent_id": "did:test:B",
                "source": "a2a.task_complete",
                "signal_id": "sig-good",
                "turn_id": None,
                "depth": 2,
                "emitted_at": "2026-05-01T09:00:00+00:00",
            },
        ],
    })
    sig = build_signal_for_completed_task(task, target_agent="did:test:A")
    assert len(sig.causation_chain) == 1
    assert sig.causation_chain[0].source == "a2a.task_complete"


# ---------------------------------------------------------------------------
# End-to-end through the real dispatcher
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_completed_task_wakes_the_bird(components):
    """Happy path: task completes locally → signal builder produces
    envelope → dispatcher routes COGNITION → agent.process_input
    receives the rendered prompt."""
    c = components
    task = _fake_task(task_id="t-happy", summary_text="briefing X done")
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
    task = _fake_task(metadata={
        METADATA_KEY: serialize_chain_for_metadata([prior_frame]),
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
    task = _fake_task(metadata={})
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
    task = _fake_task(metadata={
        METADATA_KEY: serialize_chain_for_metadata(chain),
    })
    sig = build_signal_for_completed_task(task, target_agent=c.agent.did)
    # Adding one more frame puts depth at 6 > TTL (5).

    result = await c.dispatcher.dispatch_signal(sig)
    assert result.status == Status.DROPPED_CYCLE
    assert "exceeds TTL" in (result.error or "")


# ---------------------------------------------------------------------------
# SSE notification path preserved
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_event_manager_callback_keeps_sse_notification_when_dispatcher_present(
    tmp_path,
):
    """Phase 5 requirement: SSE notification path must survive the
    cognition signal addition. Existing browser/UI consumers don't
    care about signals — they read pending_task_notifications."""
    from kestrel_sovereign.agent.event_manager import EventManagerMixin
    from kestrel_sovereign.a2a.types import (
        Message, Task, TaskState, TaskStatus, TextPart,
    )

    backend = SQLiteBackend(str(tmp_path / "sse.db"))
    await backend.connect()
    store = SignalLogStore(backend)
    await store.initialize()
    registry = SourceRegistry()
    registry.register(build_a2a_task_complete_registration())

    class _AgentLikeWithDispatcher(EventManagerMixin):
        did = "did:test:sse"
        def __init__(self):
            self._event_listeners = []
            self._pending_task_notifications = []
            self._background_tasks = []
            self.dispatcher = SignalDispatcher(
                agent=self, registry=registry,
                lock_manager=OrderedLockManager(), store=store,
            )

        async def process_input(self, prompt):
            return "ack"

        def _track_background_task(self, coro, *, name):
            task = asyncio.create_task(coro, name=name)
            self._background_tasks.append(task)
            return task

    agent = _AgentLikeWithDispatcher()

    real_task = Task(
        id="task-sse-001",
        sessionId="session-1",
        status=TaskStatus(
            state=TaskState.COMPLETED,
            message=Message(role="agent", parts=[TextPart(text="done")]),
        ),
        metadata={"agent_id": "PeerAgent", "skill": "do_thing"},
    )

    agent._on_background_task_complete(real_task)

    # SSE notification appended (legacy path).
    assert len(agent._pending_task_notifications) == 1
    assert "Background task completed" in agent._pending_task_notifications[0]
    assert "PeerAgent/do_thing" in agent._pending_task_notifications[0]

    # And the cognition signal was enqueued (new path).
    assert len(agent._background_tasks) >= 1

    # Drain the background dispatch.
    pending = [t for t in agent._background_tasks if not t.done()]
    if pending:
        await asyncio.gather(*pending, return_exceptions=True)

    # Confirm the signal landed in signal_log.
    rows = await backend.fetch_all(
        "SELECT source, status FROM signal_log WHERE source=?",
        (SOURCE_NAME,),
    )
    assert len(rows) == 1
    assert rows[0][1] == Status.OK.value

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
    task = _fake_task(task_id="dup-1", state="completed", summary_text="done")

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


@pytest.mark.asyncio
async def test_taskmanager_create_task_attaches_chain_via_provider(tmp_path):
    """`TaskManager.create_task` reads the agent's in-flight causation
    chain via the registered provider and attaches it to outbound task
    metadata. The receive side rehydrates from this metadata.

    Verifies the wiring without a full agent — uses a stub provider
    returning a known chain and asserts it lands in task.metadata.
    """
    from kestrel_sovereign.a2a.stores import (
        SQLiteSessionService, SQLiteTaskStore, SQLiteObservabilityStore,
    )
    from kestrel_sovereign.a2a.task_manager import TaskManager
    from kestrel_sovereign.a2a.types import (
        Message, TaskSendParams, TextPart,
    )

    db_path = str(tmp_path / "tm.db")
    task_store = SQLiteTaskStore(db_path)
    session_service = SQLiteSessionService(db_path)
    observability_store = SQLiteObservabilityStore(db_path)

    # Provider returns an already-serialized chain — same shape
    # KestrelAgent._provide_causation_chain emits.
    serialized_chain = [
        {
            "agent_id": "did:test:A",
            "source": SOURCE_NAME,
            "signal_id": "sig-prior",
            "turn_id": "turn-prior",
            "depth": 1,
            "emitted_at": datetime(2026, 5, 1, 9, 0, tzinfo=timezone.utc).isoformat(),
        }
    ]
    tm = TaskManager(
        task_store=task_store,
        session_service=session_service,
        observability_store=observability_store,
        causation_chain_provider=lambda: serialized_chain,
    )
    await tm.initialize()

    params = TaskSendParams(
        id="outbound-1",
        sessionId="session-1",
        message=Message(role="user", parts=[TextPart(text="do thing")]),
        metadata={"agent_id": "PeerB", "skill": "do_thing"},
    )
    task = await tm.create_task(params, agent_name="PeerB")

    # Original metadata preserved + chain appended.
    assert task.metadata["agent_id"] == "PeerB"
    assert task.metadata["skill"] == "do_thing"
    assert task.metadata["causation_chain"] == serialized_chain

    await tm.close()


@pytest.mark.asyncio
async def test_taskmanager_create_task_omits_chain_when_provider_returns_none(
    tmp_path,
):
    """No active turn → provider returns None → no causation_chain in
    metadata. Avoids bloating the task row with empty chain entries
    for direct HTTP user input or tests that don't drive cognition."""
    from kestrel_sovereign.a2a.stores import (
        SQLiteSessionService, SQLiteTaskStore, SQLiteObservabilityStore,
    )
    from kestrel_sovereign.a2a.task_manager import TaskManager
    from kestrel_sovereign.a2a.types import (
        Message, TaskSendParams, TextPart,
    )

    db_path = str(tmp_path / "tm.db")
    tm = TaskManager(
        task_store=SQLiteTaskStore(db_path),
        session_service=SQLiteSessionService(db_path),
        observability_store=SQLiteObservabilityStore(db_path),
        causation_chain_provider=lambda: None,
    )
    await tm.initialize()

    params = TaskSendParams(
        id="outbound-2",
        sessionId="session-1",
        message=Message(role="user", parts=[TextPart(text="x")]),
        metadata={"agent_id": "PeerB"},
    )
    task = await tm.create_task(params, agent_name="PeerB")
    assert "causation_chain" not in task.metadata

    await tm.close()


@pytest.mark.asyncio
async def test_full_AB_A_loop_via_real_taskmanager_rejected_at_depth_2(
    components, tmp_path,
):
    """End-to-end: simulate the agent dispatching an A2A completion
    cognition, the resulting turn spawning an outbound A2A task that
    PICKS UP the chain via the agent's _current_chain (set by the
    dispatcher), the receiving side reading the chain back from
    metadata, and the SECOND completion signal being rejected at
    depth 2 by cycle detection.

    This is the test that Phase 5's headline test (with manual
    metadata setup) didn't actually validate end-to-end."""
    from kestrel_sovereign.a2a.stores import (
        SQLiteSessionService, SQLiteTaskStore, SQLiteObservabilityStore,
    )
    from kestrel_sovereign.a2a.task_manager import TaskManager
    from kestrel_sovereign.a2a.types import (
        Message, TaskSendParams, TextPart,
    )

    c = components

    # Wire the agent's chain provider — same as KestrelAgent.initialize().
    def _provide_chain():
        chain = c.agent._current_chain if hasattr(c.agent, "_current_chain") else None
        if not chain:
            return None
        return serialize_chain_for_metadata(chain)

    db_path = str(tmp_path / "loop.db")
    tm = TaskManager(
        task_store=SQLiteTaskStore(db_path),
        session_service=SQLiteSessionService(db_path),
        observability_store=SQLiteObservabilityStore(db_path),
        causation_chain_provider=_provide_chain,
    )
    await tm.initialize()

    # ---- First completion: dispatcher sets chain on agent, turn runs.
    initial_task = _fake_task(task_id="t-loop-1", metadata={})
    sig1 = build_signal_for_completed_task(initial_task, target_agent=c.agent.did)

    # Stub process_input to simulate the in-turn outbound task creation
    # — like the real agent would when a feature decides to spawn work
    # on a peer in response to receiving the completion notification.
    outbound_tasks_seen = []

    async def turn_emits_outbound(prompt):
        params = TaskSendParams(
            id=f"outbound-{len(outbound_tasks_seen) + 1}",
            sessionId="session-loop",
            message=Message(role="user", parts=[TextPart(text="follow-up")]),
            metadata={"agent_id": "PeerB", "skill": "do_thing"},
        )
        new_task = await tm.create_task(params, agent_name="PeerB")
        outbound_tasks_seen.append(new_task)
        return "ack"

    c.agent.process_input = turn_emits_outbound  # patch for this test

    r1 = await c.dispatcher.dispatch_signal(sig1)
    assert r1.status == Status.OK
    assert len(outbound_tasks_seen) == 1

    # ---- Outbound task carries the chain (depth 1: agent A frame).
    outbound_task = outbound_tasks_seen[0]
    assert "causation_chain" in outbound_task.metadata
    chain_in_metadata = outbound_task.metadata["causation_chain"]
    assert len(chain_in_metadata) == 1
    assert chain_in_metadata[0]["agent_id"] == c.agent.did
    assert chain_in_metadata[0]["source"] == SOURCE_NAME

    # ---- Second completion: peer finishes the outbound task and the
    # callback fires locally. The receive-side signal builder rehydrates
    # the chain from metadata → the new signal carries A's prior frame.
    completed_outbound = _fake_task(
        task_id=outbound_task.id,
        state="completed",
        metadata=outbound_task.metadata,
    )
    sig2 = build_signal_for_completed_task(
        completed_outbound, target_agent=c.agent.did
    )
    assert len(sig2.causation_chain) == 1, (
        "rehydrated chain must carry the prior A frame"
    )

    r2 = await c.dispatcher.dispatch_signal(sig2)
    assert r2.status == Status.DROPPED_CYCLE, (
        f"second completion must be rejected as cycle (A appears "
        f"in chain with same source); got {r2.status}"
    )
    # No second outbound task was created — the cycle was caught.
    assert len(outbound_tasks_seen) == 1

    await tm.close()


@pytest.mark.asyncio
async def test_event_manager_callback_works_without_dispatcher(tmp_path):
    """Backward compat: agents without a dispatcher (pre-Phase-5
    fixtures, partially-initialized agents) keep getting the SSE
    notification — they just don't get the cognition signal."""
    from kestrel_sovereign.agent.event_manager import EventManagerMixin
    from kestrel_sovereign.a2a.types import (
        Message, Task, TaskState, TaskStatus, TextPart,
    )

    class _AgentLikeNoDispatcher(EventManagerMixin):
        did = "did:test:no-dispatcher"
        def __init__(self):
            self._event_listeners = []
            self._pending_task_notifications = []
            # NOTE: no self.dispatcher

    agent = _AgentLikeNoDispatcher()

    real_task = Task(
        id="task-x",
        sessionId="session-1",
        status=TaskStatus(state=TaskState.COMPLETED),
        metadata={},
    )

    # Must not raise even though dispatcher is absent.
    agent._on_background_task_complete(real_task)
    assert len(agent._pending_task_notifications) == 1
