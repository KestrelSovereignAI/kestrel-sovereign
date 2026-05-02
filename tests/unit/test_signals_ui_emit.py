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


def _result_summary(body):
    """Test helper: stringify the body. Real sources implement
    domain-specific bounded summaries (see a2a/heartbeat/wallet
    sources). The dispatcher's behavior is symmetric — whatever the
    callback returns, capped at MAX_RESULT_SUMMARY_BYTES."""
    return str(body) if body is not None else ""


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
            result_summary=_result_summary,
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
async def test_payload_excludes_raw_artifact_and_action_result(components):
    """The raw artifact / action_result is NEVER on the wire — the
    per-source `result_summary` callback decides what bounded text
    becomes user-visible. Verifies the raw fields don't leak into
    the SSE event regardless of the summary."""
    c = components
    sig = _make_signal(visibility=Visibility.USER_VISIBLE)
    await c.dispatcher.dispatch_signal(sig)
    await _drain(c.agent)
    payload = c.agent.emitted[0][1]
    assert "artifact" not in payload
    assert "action_result" not in payload


@pytest.mark.asyncio
async def test_payload_includes_result_summary_when_source_sets_callback(components):
    """#907 review P1 fix: sources opt in via
    `SourceRegistration.result_summary`; when set, the bounded body
    appears in the SSE payload AND in signal_log.result_summary.
    Without this, USER_VISIBLE/ADMIN_VISIBLE signals were a
    metadata-only toast — the side channel had nothing to render."""
    c = components
    sig = _make_signal(visibility=Visibility.USER_VISIBLE)
    await c.dispatcher.dispatch_signal(sig)
    await _drain(c.agent)
    payload = c.agent.emitted[0][1]
    # The action handler returned {"ran": payload}; _result_summary
    # stringifies it. The presence of the field is the contract;
    # the exact text is the source's choice.
    assert payload["result_summary"] is not None
    assert "ran" in payload["result_summary"]


@pytest.mark.asyncio
async def test_payload_result_summary_is_none_when_source_omits_callback(
    tmp_path,
):
    """A source registered without `result_summary` gets None in the
    SSE payload — UI consumers see metadata-only and must fetch the
    body from the source-specific surface (chat history, etc.).
    This is the legitimate metadata-only case the contract supports."""
    backend = SQLiteBackend(str(tmp_path / "no_summary.db"))
    await backend.connect()
    store = SignalLogStore(backend)
    await store.initialize()
    registry = SourceRegistry()
    agent = _FakeAgent()
    dispatcher = SignalDispatcher(
        agent=agent, registry=registry,
        lock_manager=OrderedLockManager(), store=store,
    )

    async def _h(payload):
        return {"unused": True}

    registry.register(
        SourceRegistration(
            name="ui_test.no_summary",
            schema=lambda p: p,
            default_mode=SignalMode.ACTION,
            allowed_modes=frozenset({SignalMode.ACTION}),
            handler=_h,
            log_redaction=_redaction(),
            # NO result_summary — opt-out.
        )
    )

    sig = Signal(
        source="ui_test.no_summary",
        kind="run",
        mode=SignalMode.ACTION,
        payload={},
        target_agent="did:test:phase7",
        visibility=Visibility.USER_VISIBLE,
    )
    await dispatcher.dispatch_signal(sig)
    pending = [t for t in agent.background_tasks if not t.done()]
    if pending:
        await asyncio.gather(*pending, return_exceptions=True)

    payload = agent.emitted[0][1]
    assert payload["result_summary"] is None
    await backend.close()


@pytest.mark.asyncio
async def test_result_summary_truncated_at_byte_cap(tmp_path):
    """Defense in depth: even if a misconfigured callback returns
    megabyte text, the store hard-caps at MAX_RESULT_SUMMARY_BYTES
    so SSE bandwidth and signal_log row size stay bounded.

    The cap is BYTES (suffix included), not Python characters — the
    suffix is budgeted INSIDE the cap so the returned string's UTF-8
    encoding is guaranteed <= MAX_RESULT_SUMMARY_BYTES."""
    from kestrel_sdk.signals import MAX_RESULT_SUMMARY_BYTES

    backend = SQLiteBackend(str(tmp_path / "huge.db"))
    await backend.connect()
    store = SignalLogStore(backend)
    await store.initialize()
    registry = SourceRegistry()
    agent = _FakeAgent()
    dispatcher = SignalDispatcher(
        agent=agent, registry=registry,
        lock_manager=OrderedLockManager(), store=store,
    )

    async def _h(payload):
        return "x" * 100_000  # huge return body

    registry.register(
        SourceRegistration(
            name="ui_test.huge",
            schema=lambda p: p,
            default_mode=SignalMode.ACTION,
            allowed_modes=frozenset({SignalMode.ACTION}),
            handler=_h,
            log_redaction=_redaction(),
            # Misconfigured callback: returns the full body unbounded.
            result_summary=lambda body: body,
        )
    )

    sig = Signal(
        source="ui_test.huge",
        kind="run",
        mode=SignalMode.ACTION,
        payload={},
        target_agent="did:test:phase7",
        visibility=Visibility.USER_VISIBLE,
    )
    await dispatcher.dispatch_signal(sig)
    pending = [t for t in agent.background_tasks if not t.done()]
    if pending:
        await asyncio.gather(*pending, return_exceptions=True)

    payload = agent.emitted[0][1]
    summary = payload["result_summary"]
    assert summary.endswith("...(truncated)")
    # Byte invariant: encoded length (suffix included) <= cap.
    assert len(summary.encode("utf-8")) <= MAX_RESULT_SUMMARY_BYTES

    rows = await backend.fetch_all(
        "SELECT result_summary FROM signal_log WHERE source=?",
        ("ui_test.huge",),
    )
    assert rows[0][0] == summary  # store has the same capped text

    await backend.close()


@pytest.mark.asyncio
async def test_result_summary_truncates_non_ascii_by_bytes(tmp_path):
    """#907 review P2 regression: prior truncation used `len()` which
    counts Python characters. Multi-byte codepoints (CJK, emoji)
    could exceed the documented byte cap — a 2048-char Japanese
    summary is ~6KB on the wire. Verify the byte invariant holds for
    non-ASCII text and that the truncation point is UTF-8 valid (no
    UnicodeDecodeError, no half-codepoint at the cut)."""
    from kestrel_sdk.signals import MAX_RESULT_SUMMARY_BYTES

    backend = SQLiteBackend(str(tmp_path / "cjk.db"))
    await backend.connect()
    store = SignalLogStore(backend)
    await store.initialize()
    registry = SourceRegistry()
    agent = _FakeAgent()
    dispatcher = SignalDispatcher(
        agent=agent, registry=registry,
        lock_manager=OrderedLockManager(), store=store,
    )

    # Each "あ" is 3 bytes in UTF-8. 1500 chars = 4500 bytes, well
    # over the 2048-byte cap. Under the buggy len()-based truncation
    # this would have passed through (1500 < 2048 chars) producing a
    # 4500-byte result.
    big_cjk = "あ" * 1500
    assert len(big_cjk) < MAX_RESULT_SUMMARY_BYTES  # would have passed under old code
    assert len(big_cjk.encode("utf-8")) > MAX_RESULT_SUMMARY_BYTES

    async def _h(payload):
        return big_cjk

    registry.register(
        SourceRegistration(
            name="ui_test.cjk",
            schema=lambda p: p,
            default_mode=SignalMode.ACTION,
            allowed_modes=frozenset({SignalMode.ACTION}),
            handler=_h,
            log_redaction=_redaction(),
            result_summary=lambda body: body,
        )
    )

    sig = Signal(
        source="ui_test.cjk",
        kind="run",
        mode=SignalMode.ACTION,
        payload={},
        target_agent="did:test:phase7",
        visibility=Visibility.USER_VISIBLE,
    )
    await dispatcher.dispatch_signal(sig)
    pending = [t for t in agent.background_tasks if not t.done()]
    if pending:
        await asyncio.gather(*pending, return_exceptions=True)

    summary = agent.emitted[0][1]["result_summary"]
    encoded = summary.encode("utf-8")
    assert len(encoded) <= MAX_RESULT_SUMMARY_BYTES, (
        f"byte cap violated: {len(encoded)} > {MAX_RESULT_SUMMARY_BYTES}"
    )
    assert summary.endswith("...(truncated)")
    # No UnicodeDecodeError implicit in reaching this point — the
    # truncation didn't strand a partial codepoint.
    # Body before suffix is valid UTF-8 (encode/decode roundtrip OK).
    body = summary[: -len("...(truncated)")]
    assert body.encode("utf-8").decode("utf-8") == body

    await backend.close()


@pytest.mark.asyncio
async def test_result_summary_callback_failure_records_placeholder(tmp_path):
    """A misconfigured callback that raises shouldn't break the
    dispatch — it gets a placeholder so the operator can debug, and
    the SSE event still fires."""
    backend = SQLiteBackend(str(tmp_path / "fail.db"))
    await backend.connect()
    store = SignalLogStore(backend)
    await store.initialize()
    registry = SourceRegistry()
    agent = _FakeAgent()
    dispatcher = SignalDispatcher(
        agent=agent, registry=registry,
        lock_manager=OrderedLockManager(), store=store,
    )

    async def _h(payload):
        return {"ok": True}

    def bad_summary(body):
        raise ValueError("summary bug")

    registry.register(
        SourceRegistration(
            name="ui_test.broken",
            schema=lambda p: p,
            default_mode=SignalMode.ACTION,
            allowed_modes=frozenset({SignalMode.ACTION}),
            handler=_h,
            log_redaction=_redaction(),
            result_summary=bad_summary,
        )
    )

    sig = Signal(
        source="ui_test.broken",
        kind="run",
        mode=SignalMode.ACTION,
        payload={},
        target_agent="did:test:phase7",
        visibility=Visibility.USER_VISIBLE,
    )
    result = await dispatcher.dispatch_signal(sig)
    # Dispatch itself succeeds; the callback failure is contained.
    assert result.status == Status.OK
    pending = [t for t in agent.background_tasks if not t.done()]
    if pending:
        await asyncio.gather(*pending, return_exceptions=True)

    payload = agent.emitted[0][1]
    assert payload["result_summary"] is not None
    assert "result_summary failed" in payload["result_summary"]
    assert "ValueError" in payload["result_summary"]
    await backend.close()


# ---------------------------------------------------------------------------
# P2 fix: failed log write does NOT emit
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_log_write_failure_suppresses_ui_emit(components, monkeypatch):
    """#907 review P2: contract is "SSE event fires AFTER the log
    write." If the write fails, the audit trail is broken — emitting
    a signal_completed with a signal_id that can't be looked up in
    signal_log would mislead consumers. Verify the dispatcher
    suppresses the emit on log failure."""
    c = components

    async def fail_append(*args, **kwargs):
        raise RuntimeError("disk full")

    monkeypatch.setattr(c.dispatcher._store, "append", fail_append)

    sig = _make_signal(visibility=Visibility.USER_VISIBLE)
    result = await c.dispatcher.dispatch_signal(sig)
    # Dispatch reports OK because the handler ran successfully;
    # the log write happens in the background.
    assert result.status == Status.OK
    await _drain(c.agent)
    # No SSE emit because the log write raised.
    assert c.agent.emitted == []


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
