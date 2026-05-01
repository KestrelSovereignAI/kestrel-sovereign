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


class _FakeAgent:
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
