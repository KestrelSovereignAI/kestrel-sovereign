"""Unit tests for the SignalDispatcher pipeline.

Covers each pipeline step:
- validation (unknown source, mode mismatch, sanitizer)
- cycle detection (TTL, same (agent, source) earlier in chain, allow_self_loops)
- quiet-hours (modes_governed, urgency_override)
- coalescing (dedupe_key within window)
- rate-limit (per_minute, per_hour, burst)
- routing (ACTION, ARTIFACT, COGNITION)
- enqueue_signal (tracked task)
- failure encoding (Status.FAILED, error string set)
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import tempfile
from datetime import datetime, time, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from kestrel_sdk.signals import (
    AttentionPolicy,
    CausationFrame,
    RateLimit,
    RedactionPolicy,
    ResourceLock,
    Signal,
    SignalMode,
    SourceRegistration,
    Status,
    Trust,
    Urgency,
    Visibility,
)
from kestrel_sovereign.signals import (
    OrderedLockManager,
    SignalDispatcher,
    SignalLogStore,
    SignalWithPromptTemplateOverride,
    SourceRegistry,
    SourceRegistrationWithPromptOverride,
)
from kestrel_sovereign.storage.db import SQLiteBackend


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _redaction() -> RedactionPolicy:
    return RedactionPolicy(summarize=lambda p: "<redacted>")


@pytest.fixture
def db_path(tmp_path):
    return str(tmp_path / "signal_log.db")


class _FakeAgent:
    """Stand-in for KestrelAgent. Satisfies DispatcherAgent protocol."""

    def __init__(self, did: str = "agent-test"):
        self._did = did
        self.background_tasks: list[asyncio.Task] = []
        self.process_input_calls: list[str] = []
        self.process_input_return: object = "ok"

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
async def dispatcher_components(db_path):
    backend = SQLiteBackend(db_path)
    await backend.connect()
    store = SignalLogStore(backend)
    await store.initialize()
    registry = SourceRegistry()
    locks = OrderedLockManager()
    agent = _FakeAgent()
    dispatcher = SignalDispatcher(
        agent=agent,
        registry=registry,
        lock_manager=locks,
        store=store,
    )
    yield SimpleNamespace(
        dispatcher=dispatcher,
        agent=agent,
        registry=registry,
        store=store,
        backend=backend,
        locks=locks,
    )
    # Drain background log writes before closing the backend.
    pending = [t for t in agent.background_tasks if not t.done()]
    if pending:
        await asyncio.gather(*pending, return_exceptions=True)
    await backend.close()


def _action_reg(name: str = "action_src", handler=None) -> SourceRegistration:
    async def _default(payload):
        return {"received": payload}

    return SourceRegistration(
        name=name,
        schema=dict,
        default_mode=SignalMode.ACTION,
        allowed_modes=frozenset({SignalMode.ACTION}),
        handler=handler or _default,
        log_redaction=_redaction(),
    )


def _artifact_reg(name: str = "artifact_src", handler=None) -> SourceRegistration:
    async def _default(signal):
        return f"artifact:{signal.kind}"

    return SourceRegistration(
        name=name,
        schema=dict,
        default_mode=SignalMode.ARTIFACT,
        allowed_modes=frozenset({SignalMode.ARTIFACT}),
        artifact_handler=handler or _default,
        log_redaction=_redaction(),
    )


def _cognition_reg(
    template_path: Path, name: str = "cog_src", **overrides
) -> SourceRegistration:
    base = dict(
        name=name,
        schema=dict,
        default_mode=SignalMode.COGNITION,
        allowed_modes=frozenset({SignalMode.COGNITION}),
        prompt_template=template_path,
        log_redaction=_redaction(),
    )
    base.update(overrides)
    return SourceRegistration(**base)


def _signal(
    source: str,
    *,
    mode: SignalMode = SignalMode.ACTION,
    target: str = "agent-test",
    payload=None,
    urgency: Urgency = Urgency.NORMAL,
    chain: list[CausationFrame] | None = None,
    **kwargs,
) -> Signal:
    return Signal(
        source=source,
        kind=kwargs.pop("kind", "tick"),
        mode=mode,
        payload=payload if payload is not None else {},
        target_agent=target,
        urgency=urgency,
        causation_chain=chain or [],
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_unknown_source_drops_validation(dispatcher_components):
    c = dispatcher_components
    result = await c.dispatcher.dispatch_signal(_signal("nonexistent"))
    assert result.status == Status.DROPPED_VALIDATION
    assert "Unknown source" in (result.error or "")


@pytest.mark.asyncio
async def test_mode_not_in_allowed_drops_validation(dispatcher_components):
    c = dispatcher_components
    c.registry.register(_action_reg("only_action"))
    result = await c.dispatcher.dispatch_signal(
        _signal("only_action", mode=SignalMode.COGNITION)
    )
    assert result.status == Status.DROPPED_VALIDATION
    assert "not in allowed modes" in (result.error or "")


@pytest.mark.asyncio
async def test_sanitizer_runs_on_untrusted_non_action(dispatcher_components, tmp_path):
    """UNTRUSTED + COGNITION → sanitizer runs and replaces payload."""
    c = dispatcher_components

    def sanitize(payload: dict) -> dict:
        return {k: "<scrubbed>" for k in payload}

    template = tmp_path / "tpl.md"
    template.write_text("payload: {payload}")
    c.registry.register(
        _cognition_reg(
            template,
            name="webhook_test",
            trust=Trust.UNTRUSTED,
            sanitizer=sanitize,
        )
    )
    result = await c.dispatcher.dispatch_signal(
        _signal("webhook_test", mode=SignalMode.COGNITION, payload={"secret": "x"})
    )
    assert result.status == Status.OK
    assert "scrubbed" in c.agent.process_input_calls[0]


@pytest.mark.asyncio
async def test_schema_failure_drops_validation(dispatcher_components):
    """Registered schema must reject malformed payloads BEFORE they reach
    handlers, artifact handlers, cognition templates, or signal logs.
    Regression for an earlier draft where SourceRegistration.schema was
    declared but never invoked by the pipeline."""
    c = dispatcher_components

    def strict_schema(payload: dict) -> dict:
        if "user_id" not in payload:
            raise ValueError("missing required field 'user_id'")
        return payload

    captured: list[dict] = []

    async def handler(payload: dict) -> dict:
        captured.append(payload)
        return {"ok": True}

    c.registry.register(
        SourceRegistration(
            name="strict",
            schema=strict_schema,
            default_mode=SignalMode.ACTION,
            allowed_modes=frozenset({SignalMode.ACTION}),
            handler=handler,
            log_redaction=_redaction(),
        )
    )

    bad = await c.dispatcher.dispatch_signal(
        _signal("strict", payload={"wrong": "shape"})
    )
    assert bad.status == Status.DROPPED_VALIDATION
    assert "Schema rejected" in (bad.error or "")
    assert "user_id" in (bad.error or "")
    assert captured == [], "handler must not run when schema rejects"

    good = await c.dispatcher.dispatch_signal(
        _signal("strict", payload={"user_id": "abc"})
    )
    assert good.status == Status.OK
    assert captured == [{"user_id": "abc"}]


@pytest.mark.asyncio
async def test_schema_normalization_replaces_payload(dispatcher_components):
    """Schema is allowed to normalize (defaults, type coercion). Downstream
    handlers see the normalized form, not the original."""
    c = dispatcher_components

    def normalize(payload: dict) -> dict:
        return {**payload, "added_by_schema": True}

    captured: list[dict] = []

    async def handler(payload: dict) -> dict:
        captured.append(payload)
        return None

    c.registry.register(
        SourceRegistration(
            name="normalizing",
            schema=normalize,
            default_mode=SignalMode.ACTION,
            allowed_modes=frozenset({SignalMode.ACTION}),
            handler=handler,
            log_redaction=_redaction(),
        )
    )
    result = await c.dispatcher.dispatch_signal(
        _signal("normalizing", payload={"orig": 1})
    )
    assert result.status == Status.OK
    assert captured == [{"orig": 1, "added_by_schema": True}]


@pytest.mark.asyncio
async def test_schema_runs_after_sanitizer(dispatcher_components, tmp_path):
    """For UNTRUSTED sources, sanitizer scrubs first, then schema validates
    the scrubbed form. The schema sees the canonical payload that will
    actually reach the handler."""
    c = dispatcher_components
    template = tmp_path / "tpl.md"
    template.write_text("payload: {payload}")

    sanitizer_seen: list[dict] = []
    schema_seen: list[dict] = []

    def sanitize(p: dict) -> dict:
        sanitizer_seen.append(p)
        return {"safe_field": p.get("safe_field", "")}

    def schema(p: dict) -> dict:
        schema_seen.append(p)
        if "safe_field" not in p:
            raise ValueError("schema requires safe_field")
        return p

    c.registry.register(
        SourceRegistration(
            name="ordered",
            schema=schema,
            default_mode=SignalMode.COGNITION,
            allowed_modes=frozenset({SignalMode.COGNITION}),
            prompt_template=template,
            trust=Trust.UNTRUSTED,
            sanitizer=sanitize,
            log_redaction=_redaction(),
        )
    )
    result = await c.dispatcher.dispatch_signal(
        _signal(
            "ordered",
            mode=SignalMode.COGNITION,
            payload={"safe_field": "hi", "evil": "<script>"},
        )
    )
    assert result.status == Status.OK
    assert sanitizer_seen == [{"safe_field": "hi", "evil": "<script>"}]
    # Schema saw the post-sanitizer payload — no "evil" key.
    assert schema_seen == [{"safe_field": "hi"}]


@pytest.mark.asyncio
async def test_sanitizer_failure_drops_validation(dispatcher_components, tmp_path):
    c = dispatcher_components

    def sanitize(payload: dict) -> dict:
        raise ValueError("bad payload")

    template = tmp_path / "tpl.md"
    template.write_text("x")
    c.registry.register(
        _cognition_reg(
            template,
            name="webhook_bad",
            trust=Trust.UNTRUSTED,
            sanitizer=sanitize,
        )
    )
    result = await c.dispatcher.dispatch_signal(
        _signal("webhook_bad", mode=SignalMode.COGNITION)
    )
    assert result.status == Status.DROPPED_VALIDATION
    assert "Sanitizer raised" in (result.error or "")


# ---------------------------------------------------------------------------
# Cycle detection
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_first_hop_passes_cycle_check(dispatcher_components):
    c = dispatcher_components
    c.registry.register(_action_reg())
    result = await c.dispatcher.dispatch_signal(_signal("action_src"))
    assert result.status == Status.OK


@pytest.mark.asyncio
async def test_self_emit_same_source_is_cycle(dispatcher_components):
    """Heartbeat-driven turn that re-emits a heartbeat signal during the
    turn → cycle. Worked example from SIGNAL_DISPATCHER.md §6."""
    c = dispatcher_components
    c.registry.register(_action_reg("heartbeat_like"))

    chain = [
        CausationFrame(
            agent_id="agent-test",
            source="heartbeat_like",
            signal_id="prior",
            turn_id="turn-1",
            depth=1,
            emitted_at=datetime.now(timezone.utc),
        )
    ]
    result = await c.dispatcher.dispatch_signal(
        _signal("heartbeat_like", chain=chain)
    )
    assert result.status == Status.DROPPED_CYCLE
    assert "Cycle detected" in (result.error or "")


@pytest.mark.asyncio
async def test_self_emit_with_allow_self_loops(dispatcher_components):
    """Source that opts in to self-loops only enforces TTL."""
    c = dispatcher_components
    reg = _action_reg("looper")
    reg = SourceRegistration(
        name=reg.name,
        schema=reg.schema,
        default_mode=reg.default_mode,
        allowed_modes=reg.allowed_modes,
        handler=reg.handler,
        log_redaction=reg.log_redaction,
        allow_self_loops=True,
    )
    c.registry.register(reg)
    chain = [
        CausationFrame(
            agent_id="agent-test",
            source="looper",
            signal_id="prior",
            turn_id="t1",
            depth=1,
            emitted_at=datetime.now(timezone.utc),
        )
    ]
    result = await c.dispatcher.dispatch_signal(_signal("looper", chain=chain))
    assert result.status == Status.OK


@pytest.mark.asyncio
async def test_ttl_exhaustion_drops_cycle(dispatcher_components):
    c = dispatcher_components
    c.registry.register(_action_reg())
    chain = [
        CausationFrame(
            agent_id=f"agent-{i}",
            source=f"src-{i}",
            signal_id=f"sig-{i}",
            turn_id=None,
            depth=i,
            emitted_at=datetime.now(timezone.utc),
        )
        for i in range(1, 6)  # depths 1..5; new hop would be 6 > TTL
    ]
    result = await c.dispatcher.dispatch_signal(_signal("action_src", chain=chain))
    assert result.status == Status.DROPPED_CYCLE
    assert "exceeds TTL" in (result.error or "")


@pytest.mark.asyncio
async def test_different_source_same_agent_is_not_cycle(dispatcher_components):
    """A receiving heartbeat then later an a2a_complete in same chain →
    legitimate; same agent, different source."""
    c = dispatcher_components
    c.registry.register(_action_reg("a2a_complete"))
    chain = [
        CausationFrame(
            agent_id="agent-test",
            source="heartbeat",  # different source
            signal_id="prior",
            turn_id="t1",
            depth=1,
            emitted_at=datetime.now(timezone.utc),
        )
    ]
    result = await c.dispatcher.dispatch_signal(
        _signal("a2a_complete", chain=chain)
    )
    assert result.status == Status.OK


# ---------------------------------------------------------------------------
# Quiet hours
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_quiet_hours_drops_cognition_below_override(
    dispatcher_components, tmp_path
):
    c = dispatcher_components
    template = tmp_path / "tpl.md"
    template.write_text("x")

    # Pin the clock to 12:00 UTC to ensure we're always inside the quiet window.
    # The test isn't testing wall-clock behavior; it's testing the quiet-hours
    # decision logic. Using a pinned clock makes the test deterministic.
    pinned_time = datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
    c.dispatcher._clock = lambda: pinned_time

    # Quiet 00:00–23:59 in UTC — effectively always quiet for this test.
    policy = AttentionPolicy(
        quiet_hours=(time(0, 0), time(23, 59)),
        tz="UTC",
        modes_governed=frozenset({SignalMode.COGNITION}),
        urgency_override=Urgency.HIGH,
    )
    c.registry.register(
        _cognition_reg(template, name="quiet_cog", attention_policy=policy)
    )
    result = await c.dispatcher.dispatch_signal(
        _signal("quiet_cog", mode=SignalMode.COGNITION, urgency=Urgency.NORMAL)
    )
    assert result.status == Status.DROPPED_QUIET_HOURS


@pytest.mark.asyncio
async def test_quiet_hours_bypassed_by_high_urgency(
    dispatcher_components, tmp_path
):
    c = dispatcher_components
    template = tmp_path / "tpl.md"
    template.write_text("x")
    policy = AttentionPolicy(
        quiet_hours=(time(0, 0), time(23, 59)),
        tz="UTC",
        modes_governed=frozenset({SignalMode.COGNITION}),
        urgency_override=Urgency.HIGH,
    )
    c.registry.register(
        _cognition_reg(template, name="urgent_cog", attention_policy=policy)
    )
    result = await c.dispatcher.dispatch_signal(
        _signal("urgent_cog", mode=SignalMode.COGNITION, urgency=Urgency.HIGH)
    )
    assert result.status == Status.OK


@pytest.mark.asyncio
async def test_quiet_hours_does_not_apply_to_action(dispatcher_components):
    """ACTION is not in default modes_governed; backups should run at 3am."""
    c = dispatcher_components
    policy = AttentionPolicy(
        quiet_hours=(time(0, 0), time(23, 59)),
        tz="UTC",
        # default modes_governed excludes ACTION
    )
    reg = _action_reg("backup")
    reg = SourceRegistration(
        name=reg.name,
        schema=reg.schema,
        default_mode=reg.default_mode,
        allowed_modes=reg.allowed_modes,
        handler=reg.handler,
        log_redaction=reg.log_redaction,
        attention_policy=policy,
    )
    c.registry.register(reg)
    result = await c.dispatcher.dispatch_signal(_signal("backup"))
    assert result.status == Status.OK


# ---------------------------------------------------------------------------
# Coalescing
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_coalescing_drops_duplicate_within_window(dispatcher_components):
    c = dispatcher_components
    reg = _action_reg("coalesce_src")
    reg = SourceRegistration(
        name=reg.name,
        schema=reg.schema,
        default_mode=reg.default_mode,
        allowed_modes=reg.allowed_modes,
        handler=reg.handler,
        log_redaction=reg.log_redaction,
        coalescing_window=timedelta(seconds=10),
    )
    c.registry.register(reg)

    r1 = await c.dispatcher.dispatch_signal(
        _signal("coalesce_src", dedupe_key="user-123")
    )
    r2 = await c.dispatcher.dispatch_signal(
        _signal("coalesce_src", dedupe_key="user-123")
    )
    assert r1.status == Status.OK
    assert r2.status == Status.COALESCED


@pytest.mark.asyncio
async def test_no_coalescing_without_dedupe_key(dispatcher_components):
    c = dispatcher_components
    c.registry.register(_action_reg("no_dedupe"))
    r1 = await c.dispatcher.dispatch_signal(_signal("no_dedupe"))
    r2 = await c.dispatcher.dispatch_signal(_signal("no_dedupe"))
    assert r1.status == Status.OK
    assert r2.status == Status.OK


# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_rate_limit_blocks_excess(dispatcher_components):
    c = dispatcher_components
    reg = _action_reg("limited")
    reg = SourceRegistration(
        name=reg.name,
        schema=reg.schema,
        default_mode=reg.default_mode,
        allowed_modes=reg.allowed_modes,
        handler=reg.handler,
        log_redaction=reg.log_redaction,
        rate_limit=RateLimit(per_minute=2),
    )
    c.registry.register(reg)
    r1 = await c.dispatcher.dispatch_signal(_signal("limited"))
    r2 = await c.dispatcher.dispatch_signal(_signal("limited"))
    r3 = await c.dispatcher.dispatch_signal(_signal("limited"))
    assert r1.status == Status.OK
    assert r2.status == Status.OK
    assert r3.status == Status.DROPPED_RATE_LIMIT


# ---------------------------------------------------------------------------
# Routing — three modes
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_action_routes_to_handler(dispatcher_components):
    c = dispatcher_components
    captured: list[dict] = []

    async def handler(payload):
        captured.append(payload)
        return {"ok": True}

    c.registry.register(_action_reg("act", handler=handler))
    result = await c.dispatcher.dispatch_signal(
        _signal("act", payload={"hello": "world"})
    )
    assert result.status == Status.OK
    assert result.action_result == {"ok": True}
    assert captured == [{"hello": "world"}]


@pytest.mark.asyncio
async def test_artifact_routes_to_artifact_handler(dispatcher_components):
    c = dispatcher_components

    async def handler(signal):
        return f"made:{signal.kind}"

    c.registry.register(_artifact_reg("art", handler=handler))
    result = await c.dispatcher.dispatch_signal(
        _signal("art", mode=SignalMode.ARTIFACT, kind="briefing")
    )
    assert result.status == Status.OK
    assert result.artifact == "made:briefing"


@pytest.mark.asyncio
async def test_cognition_routes_through_agent_process_input(
    dispatcher_components, tmp_path
):
    c = dispatcher_components
    template = tmp_path / "tpl.md"
    template.write_text("source={source} payload={payload}")
    c.registry.register(_cognition_reg(template, name="cog"))
    c.agent.process_input_return = "agent-said"

    result = await c.dispatcher.dispatch_signal(
        _signal("cog", mode=SignalMode.COGNITION, payload={"k": "v"})
    )
    assert result.status == Status.OK
    assert result.artifact == "agent-said"
    assert "source=cog" in c.agent.process_input_calls[0]
    assert "payload={'k': 'v'}" in c.agent.process_input_calls[0]


@pytest.mark.asyncio
async def test_cognition_uses_allowed_signal_prompt_template_override(
    dispatcher_components, tmp_path
):
    c = dispatcher_components
    base = tmp_path / "base.md"
    override = tmp_path / "override.md"
    base.write_text("base payload={payload}")
    override.write_text("override payload={payload}")
    reg = SourceRegistrationWithPromptOverride(
        **_cognition_reg(base, name="override_cog").__dict__,
        allow_prompt_override=True,
    )
    c.registry.register(reg)
    c.agent.process_input_return = "agent-said"

    signal = SignalWithPromptTemplateOverride(
        source="override_cog",
        kind="tick",
        mode=SignalMode.COGNITION,
        payload={"k": "v"},
        target_agent="agent-test",
        urgency=Urgency.NORMAL,
        prompt_template_override=override,
    )
    result = await c.dispatcher.dispatch_signal(signal)
    assert result.status == Status.OK
    assert c.agent.process_input_calls == ["override payload={'k': 'v'}"]

    pending = [t for t in c.agent.background_tasks if not t.done()]
    if pending:
        await asyncio.gather(*pending)
    rows = await c.backend.fetch_all(
        "SELECT prompt_template_hash FROM signal_log WHERE source='override_cog'"
    )
    expected = hashlib.sha256(override.read_bytes()).hexdigest()
    assert rows == [(expected,)]


@pytest.mark.asyncio
async def test_cognition_ignores_prompt_template_override_without_opt_in(
    dispatcher_components, tmp_path
):
    c = dispatcher_components
    base = tmp_path / "base.md"
    override = tmp_path / "override.md"
    base.write_text("base payload={payload}")
    override.write_text("override payload={payload}")
    c.registry.register(_cognition_reg(base, name="no_override_cog"))
    c.agent.process_input_return = "agent-said"

    signal = SignalWithPromptTemplateOverride(
        source="no_override_cog",
        kind="tick",
        mode=SignalMode.COGNITION,
        payload={"k": "v"},
        target_agent="agent-test",
        urgency=Urgency.NORMAL,
        prompt_template_override=override,
    )
    result = await c.dispatcher.dispatch_signal(signal)
    assert result.status == Status.OK
    assert c.agent.process_input_calls == ["base payload={'k': 'v'}"]

    pending = [t for t in c.agent.background_tasks if not t.done()]
    if pending:
        await asyncio.gather(*pending)
    rows = await c.backend.fetch_all(
        "SELECT prompt_template_hash FROM signal_log WHERE source='no_override_cog'"
    )
    expected = hashlib.sha256(base.read_bytes()).hexdigest()
    assert rows == [(expected,)]


@pytest.mark.asyncio
async def test_handler_exception_is_failed_not_raised(dispatcher_components):
    c = dispatcher_components

    async def boom(payload):
        raise RuntimeError("kaboom")

    c.registry.register(_action_reg("explode", handler=boom))
    result = await c.dispatcher.dispatch_signal(_signal("explode"))
    assert result.status == Status.FAILED
    assert "RuntimeError" in (result.error or "")
    assert "kaboom" in (result.error or "")


# ---------------------------------------------------------------------------
# enqueue_signal
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_enqueue_signal_returns_handle_immediately(dispatcher_components):
    c = dispatcher_components
    started = asyncio.Event()
    can_finish = asyncio.Event()

    async def slow_handler(payload):
        started.set()
        await can_finish.wait()
        return "done"

    c.registry.register(_action_reg("slow", handler=slow_handler))
    handle = await c.dispatcher.enqueue_signal(_signal("slow"))

    # Handle returns before the handler completes.
    assert not handle.done
    await started.wait()
    assert not handle.done

    can_finish.set()
    result = await handle.wait()
    assert result.status == Status.OK
    assert result.action_result == "done"
    assert handle.done


@pytest.mark.asyncio
async def test_enqueue_signal_uses_agent_background_tracker(dispatcher_components):
    c = dispatcher_components
    c.registry.register(_action_reg("trk"))
    handle = await c.dispatcher.enqueue_signal(_signal("trk"))
    # The agent's background-task list should have grown.
    assert len(c.agent.background_tasks) >= 1
    # Wait so the test doesn't leak the task.
    await handle.wait()


# ---------------------------------------------------------------------------
# Resource lock acquisition
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_action_acquires_declared_resources(dispatcher_components):
    """Two concurrent ACTION dispatches against sources that both declare
    MEMORY must serialize."""
    c = dispatcher_components
    order: list[str] = []
    order_lock = asyncio.Lock()

    async def make_handler(name: str):
        async def h(payload):
            async with order_lock:
                order.append(f"start:{name}")
            await asyncio.sleep(0.05)
            async with order_lock:
                order.append(f"end:{name}")
            return name

        return h

    h_a = await make_handler("A")
    h_b = await make_handler("B")
    reg_a = SourceRegistration(
        name="src_a",
        schema=dict,
        default_mode=SignalMode.ACTION,
        allowed_modes=frozenset({SignalMode.ACTION}),
        handler=h_a,
        log_redaction=_redaction(),
        resources=frozenset({ResourceLock.MEMORY}),
    )
    reg_b = SourceRegistration(
        name="src_b",
        schema=dict,
        default_mode=SignalMode.ACTION,
        allowed_modes=frozenset({SignalMode.ACTION}),
        handler=h_b,
        log_redaction=_redaction(),
        resources=frozenset({ResourceLock.MEMORY}),
    )
    c.registry.register(reg_a)
    c.registry.register(reg_b)

    await asyncio.gather(
        c.dispatcher.dispatch_signal(_signal("src_a")),
        c.dispatcher.dispatch_signal(_signal("src_b")),
    )

    # Whichever started first must end before the other starts.
    assert order in (
        ["start:A", "end:A", "start:B", "end:B"],
        ["start:B", "end:B", "start:A", "end:A"],
    ), order


# ---------------------------------------------------------------------------
# signal_log integration
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_dispatch_writes_signal_log_entry(dispatcher_components):
    c = dispatcher_components
    c.registry.register(_action_reg("logged"))
    result = await c.dispatcher.dispatch_signal(_signal("logged"))
    assert result.status == Status.OK

    # Drain background log writes.
    pending = [t for t in c.agent.background_tasks if not t.done()]
    if pending:
        await asyncio.gather(*pending)

    rows = await c.backend.fetch_all(
        "SELECT id, source, status, payload_redacted FROM signal_log"
    )
    assert len(rows) == 1
    assert rows[0][1] == "logged"
    assert rows[0][2] == "ok"
    assert rows[0][3] == "<redacted>"


@pytest.mark.asyncio
async def test_signal_log_does_not_store_raw_untrusted(
    dispatcher_components, tmp_path
):
    """UNTRUSTED + store_raw_trusted=False (default) → payload_raw is NULL."""
    c = dispatcher_components
    template = tmp_path / "tpl.md"
    template.write_text("x")
    c.registry.register(
        _cognition_reg(
            template,
            name="untrusted_cog",
            trust=Trust.UNTRUSTED,
            sanitizer=lambda p: p,
        )
    )
    await c.dispatcher.dispatch_signal(
        _signal(
            "untrusted_cog",
            mode=SignalMode.COGNITION,
            payload={"secret": "do not store"},
        )
    )
    pending = [t for t in c.agent.background_tasks if not t.done()]
    if pending:
        await asyncio.gather(*pending)

    rows = await c.backend.fetch_all(
        "SELECT payload_raw FROM signal_log WHERE source='untrusted_cog'"
    )
    assert len(rows) == 1
    assert rows[0][0] is None
