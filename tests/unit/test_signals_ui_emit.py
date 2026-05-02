"""Phase 7 of #889: UI side-channel SSE emit for non-INTERNAL signals.

The `notifications_sse` endpoint already forwards `agent.emit_event`
events to the browser as named SSE events. Phase 7 wires the
SignalDispatcher to call `agent.emit_event("signal_completed", ...)`
after each non-INTERNAL signal logs to signal_log. The actual UI
rendering (side channel vs inline vs hidden) lives in the consumer
(Open WebUI or any client built on the SSE endpoint) and routes
based on the payload's `visibility` and `session_id`.

Three rendering tiers per the design:
- INTERNAL              → log only, no UI emit (default for all
                          existing sources: heartbeat, cron, A2A,
                          Stripe; sources opt in by constructing
                          signals with explicit visibility)
- USER_VISIBLE          → side channel (when session_id is None)
                          OR inline with attribution (when
                          session_id matches an active chat)
- ADMIN_VISIBLE         → admin-tools side channel only
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any

import pytest

from kestrel_sdk.signals import (
    RedactionPolicy,
    Signal,
    SignalMode,
    SourceRegistration,
    Status,
    Urgency,
    Visibility,
)
from kestrel_sovereign.signals import (
    OrderedLockManager,
    SignalDispatcher,
    SignalLogStore,
    SourceRegistry,
)
from kestrel_sovereign.storage.db import SQLiteBackend


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _redaction() -> RedactionPolicy:
    return RedactionPolicy(summarize=lambda p: "<redacted>")


class _FakeAgent:
    did = "did:test:phase7"

    def __init__(self):
        self.background_tasks: list[asyncio.Task] = []
        self.emitted: list[tuple[str, dict]] = []

    async def emit_event(self, event_type: str, data: dict) -> None:
        self.emitted.append((event_type, data))

    async def process_input(self, prompt: str):  # not used in these tests
        return "ok"

    def _track_background_task(self, coro, *, name: str):
        task = asyncio.create_task(coro, name=name)
        self.background_tasks.append(task)
        return task


@pytest.fixture
async def components(tmp_path):
    backend = SQLiteBackend(str(tmp_path / "ui_emit.db"))
    await backend.connect()
    store = SignalLogStore(backend)
    await store.initialize()
    registry = SourceRegistry()
    locks = OrderedLockManager()
    agent = _FakeAgent()
    dispatcher = SignalDispatcher(
        agent=agent, registry=registry, lock_manager=locks, store=store,
    )

    async def _action_handler(payload):
        return {"ran": payload}

    registry.register(
        SourceRegistration(
            name="ui_test.action",
            schema=lambda p: p,
            default_mode=SignalMode.ACTION,
            allowed_modes=frozenset({SignalMode.ACTION}),
            handler=_action_handler,
            log_redaction=_redaction(),
        )
    )
    yield SimpleNamespace(
        agent=agent, registry=registry, dispatcher=dispatcher, backend=backend,
    )
    pending = [t for t in agent.background_tasks if not t.done()]
    if pending:
        await asyncio.gather(*pending, return_exceptions=True)
    await backend.close()


def _make_signal(
    *, visibility: Visibility, session_id: str | None = None,
    target: str = "did:test:phase7", caller: str | None = None,
) -> Signal:
    return Signal(
        source="ui_test.action",
        kind="run",
        mode=SignalMode.ACTION,
        payload={"x": 1},
        target_agent=target,
        visibility=visibility,
        session_id=session_id,
        caller=caller,
    )


async def _drain(agent: _FakeAgent) -> None:
    pending = [t for t in agent.background_tasks if not t.done()]
    if pending:
        await asyncio.gather(*pending, return_exceptions=True)


# ---------------------------------------------------------------------------
# INTERNAL signals never emit
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_internal_signal_does_not_emit_ui_event(components):
    """The default-deny posture for the side channel: a signal with
    visibility=INTERNAL writes to signal_log but never reaches the
    SSE event bus. Existing sources (heartbeat, cron, A2A, Stripe)
    all default to INTERNAL — they can't surprise-emit to the UI
    without explicit visibility upgrade."""
    c = components
    sig = _make_signal(visibility=Visibility.INTERNAL)
    result = await c.dispatcher.dispatch_signal(sig)
    assert result.status == Status.OK
    await _drain(c.agent)
    assert c.agent.emitted == []


# ---------------------------------------------------------------------------
# USER_VISIBLE / ADMIN_VISIBLE signals emit signal_completed
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_user_visible_signal_emits_signal_completed(components):
    c = components
    sig = _make_signal(visibility=Visibility.USER_VISIBLE)
    result = await c.dispatcher.dispatch_signal(sig)
    assert result.status == Status.OK
    await _drain(c.agent)
    assert len(c.agent.emitted) == 1
    event_type, payload = c.agent.emitted[0]
    assert event_type == "signal_completed"
    assert payload["visibility"] == "user_visible"
    assert payload["status"] == "ok"
    assert payload["source"] == "ui_test.action"


@pytest.mark.asyncio
async def test_admin_visible_signal_also_emits(components):
    c = components
    sig = _make_signal(visibility=Visibility.ADMIN_VISIBLE)
    result = await c.dispatcher.dispatch_signal(sig)
    assert result.status == Status.OK
    await _drain(c.agent)
    assert len(c.agent.emitted) == 1
    assert c.agent.emitted[0][1]["visibility"] == "admin_visible"


# ---------------------------------------------------------------------------
# Payload shape — routing fields preserved; body NOT included
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_payload_carries_routing_fields(components):
    """Consumers route based on session_id, visibility, target_agent,
    source, caller. All five must be in the payload."""
    c = components
    sig = _make_signal(
        visibility=Visibility.USER_VISIBLE,
        session_id="chat-123",
        caller="user:alice",
    )
    await c.dispatcher.dispatch_signal(sig)
    await _drain(c.agent)
    payload = c.agent.emitted[0][1]
    assert payload["session_id"] == "chat-123"
    assert payload["target_agent"] == "did:test:phase7"
    assert payload["caller"] == "user:alice"
    assert payload["visibility"] == "user_visible"
    assert payload["source"] == "ui_test.action"


@pytest.mark.asyncio
async def test_payload_excludes_artifact_and_action_result(components):
    """Phase 7 design: SSE payload carries metadata only; the body
    (artifact text, action_result dict) is fetched from signal_log
    if needed. Keeps SSE bandwidth bounded and centralizes redaction
    at the store. Verifies that no `artifact` or `action_result` key
    leaks into the SSE event."""
    c = components
    sig = _make_signal(visibility=Visibility.USER_VISIBLE)
    await c.dispatcher.dispatch_signal(sig)
    await _drain(c.agent)
    payload = c.agent.emitted[0][1]
    assert "artifact" not in payload
    assert "action_result" not in payload


# ---------------------------------------------------------------------------
# Failure / drop signals also emit (so consumers see errors)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_failed_signal_with_visibility_still_emits(components):
    """A handler raising → Status.FAILED → still logs and still
    emits to the UI side channel (so failure surfaces in the
    operator dashboard or user notifications)."""
    c = components

    async def boom(payload):
        raise RuntimeError("kaboom")

    c.registry.register(
        SourceRegistration(
            name="ui_test.failing",
            schema=lambda p: p,
            default_mode=SignalMode.ACTION,
            allowed_modes=frozenset({SignalMode.ACTION}),
            handler=boom,
            log_redaction=_redaction(),
        )
    )
    sig = Signal(
        source="ui_test.failing",
        kind="run",
        mode=SignalMode.ACTION,
        payload={},
        target_agent="did:test:phase7",
        visibility=Visibility.USER_VISIBLE,
    )
    result = await c.dispatcher.dispatch_signal(sig)
    assert result.status == Status.FAILED
    await _drain(c.agent)
    assert len(c.agent.emitted) == 1
    payload = c.agent.emitted[0][1]
    assert payload["status"] == "failed"
    assert "kaboom" in (payload["error"] or "")


# ---------------------------------------------------------------------------
# Backward-compat — agents without emit_event don't crash
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_agent_without_emit_event_is_safe(tmp_path):
    """Test fixtures or partially-initialized agents may not have
    emit_event. The dispatcher must not crash — it just skips the
    UI emit (signal_log still writes; existing behavior preserved)."""

    class _NoEmitAgent:
        did = "did:test:no-emit"

        def __init__(self):
            self.background_tasks = []

        async def process_input(self, prompt):
            return "ok"

        def _track_background_task(self, coro, *, name):
            task = asyncio.create_task(coro, name=name)
            self.background_tasks.append(task)
            return task

    backend = SQLiteBackend(str(tmp_path / "no_emit.db"))
    await backend.connect()
    store = SignalLogStore(backend)
    await store.initialize()
    registry = SourceRegistry()
    agent = _NoEmitAgent()
    dispatcher = SignalDispatcher(
        agent=agent, registry=registry,
        lock_manager=OrderedLockManager(), store=store,
    )

    async def _h(payload):
        return None

    registry.register(
        SourceRegistration(
            name="no_emit.action",
            schema=lambda p: p,
            default_mode=SignalMode.ACTION,
            allowed_modes=frozenset({SignalMode.ACTION}),
            handler=_h,
            log_redaction=_redaction(),
        )
    )
    sig = Signal(
        source="no_emit.action",
        kind="run",
        mode=SignalMode.ACTION,
        payload={},
        target_agent=agent.did,
        visibility=Visibility.USER_VISIBLE,
    )
    result = await dispatcher.dispatch_signal(sig)
    assert result.status == Status.OK
    pending = [t for t in agent.background_tasks if not t.done()]
    if pending:
        await asyncio.gather(*pending, return_exceptions=True)
    # No assertion that emit happened — just that we didn't crash.

    await backend.close()
