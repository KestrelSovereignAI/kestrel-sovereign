"""Phase 3 of #889: heartbeat source registration + end-to-end through
the real dispatcher.

The heartbeat-runner unit tests in test_heartbeat.py use a mock
dispatcher to keep the heartbeat-side surface tested in isolation. These
tests exercise the full dispatch path with the real SignalDispatcher,
SourceRegistry, OrderedLockManager, and SignalLogStore — so registration
correctness, prompt rendering, signal_log writes, and quiet-hours
enforcement are all verified end-to-end.
"""

from __future__ import annotations

import pytest

from kestrel_sdk.signals import (
    Signal,
    SignalMode,
    Status,
    Trust,
    Urgency,
    Visibility,
)
from kestrel_sovereign.signals import (
    OrderedLockManager,
    SignalDispatcher,
    SignalLogStore,
    SourceRegistry,
)
from kestrel_sovereign.signals.sources.heartbeat import (
    PROMPT_TEMPLATE,
    SOURCE_NAME,
    build_heartbeat_registration,
)
from kestrel_sovereign.storage.db import SQLiteBackend


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class _FakeAgent:
    """Satisfies DispatcherAgent — see test_signals_dispatcher.py."""

    def __init__(self, did: str = "did:test:heartbeat-e2e"):
        self._did = did
        self.background_tasks = []
        self.process_input_calls = []
        self.process_input_return = "HEARTBEAT_OK"

    @property
    def did(self) -> str:
        return self._did

    async def process_input(self, prompt):
        self.process_input_calls.append(prompt)
        return self.process_input_return

    def _track_background_task(self, coro, *, name):
        import asyncio

        task = asyncio.create_task(coro, name=name)
        self.background_tasks.append(task)
        return task


@pytest.fixture
async def components(tmp_path):
    backend = SQLiteBackend(str(tmp_path / "heartbeat_e2e.db"))
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
    yield (agent, registry, dispatcher, backend)
    import asyncio

    pending = [t for t in agent.background_tasks if not t.done()]
    if pending:
        await asyncio.gather(*pending, return_exceptions=True)
    await backend.close()


# ---------------------------------------------------------------------------
# Source registration
# ---------------------------------------------------------------------------


def test_heartbeat_registration_shape():
    reg = build_heartbeat_registration(
        interval_seconds=1800,
        active_hours_start="09:00",
        active_hours_end="22:00",
    )
    assert reg.name == SOURCE_NAME
    assert reg.allowed_modes == frozenset({SignalMode.COGNITION})
    assert reg.trust == Trust.TRUSTED
    assert reg.prompt_template == PROMPT_TEMPLATE
    # CONVERSATION must NOT be declared — turn lifecycle owns it.
    from kestrel_sdk.signals import ResourceLock
    assert ResourceLock.CONVERSATION not in reg.resources


def test_heartbeat_registration_quiet_hours_inverts_active_window():
    """Active 09:00-22:00 → quiet 22:01-09:00. The +1 minute shift
    preserves the legacy inclusive-end behavior: a tick at exactly
    active_end (22:00) stays active. See `_build_quiet_hours` docstring
    in heartbeat.py for the boundary rationale."""
    from datetime import time as dtime

    reg = build_heartbeat_registration(
        interval_seconds=1800,
        active_hours_start="09:00",
        active_hours_end="22:00",
    )
    assert reg.attention_policy.quiet_hours == (dtime(22, 1), dtime(9, 0))


def test_heartbeat_active_end_boundary_stays_active():
    """Regression for #903 P1: at exactly active_end (e.g. 22:00:00) the
    legacy `_is_within_active_hours` returned True (inclusive end).
    A naive inversion to quiet=(22:00, 09:00) would have flipped this to
    quiet. The +1 minute shift in `_build_quiet_hours` keeps 22:00:00
    active; quiet starts at 22:01:00."""
    from datetime import datetime, time as dtime, timezone
    from unittest.mock import patch

    from kestrel_sdk.signals import (
        AttentionPolicy,
        Signal,
        SignalMode,
        Status,
        Urgency,
    )
    from kestrel_sovereign.signals import (
        OrderedLockManager,
        SignalDispatcher,
        SignalLogStore,
        SourceRegistry,
    )

    reg = build_heartbeat_registration(
        interval_seconds=1800,
        active_hours_start="09:00",
        active_hours_end="22:00",
    )
    assert reg.attention_policy.quiet_hours == (dtime(22, 1), dtime(9, 0))

    # Drive the dispatcher's internal clock to exactly 22:00:00 UTC and
    # confirm the heartbeat source is NOT in its quiet window.
    fixed_now = datetime(2026, 5, 1, 22, 0, 0, tzinfo=timezone.utc)
    locks = OrderedLockManager()
    registry = SourceRegistry()
    registry.register(reg)

    class _Stub:
        did = "did:test:boundary"
        def __init__(self):
            self.calls = []
        async def process_input(self, prompt):
            self.calls.append(prompt)
            return "HEARTBEAT_OK"
        def _track_background_task(self, coro, *, name):
            import asyncio
            return asyncio.create_task(coro, name=name)

    agent = _Stub()

    import asyncio

    async def _run():
        # signal_log store needs a backend; use a fresh in-memory sqlite.
        from kestrel_sovereign.storage.db import SQLiteBackend
        backend = SQLiteBackend(":memory:")
        await backend.connect()
        store = SignalLogStore(backend)
        await store.initialize()
        dispatcher = SignalDispatcher(
            agent=agent,
            registry=registry,
            lock_manager=locks,
            store=store,
            clock=lambda: fixed_now,
        )
        signal = Signal(
            source="heartbeat",
            kind="tick",
            mode=SignalMode.COGNITION,
            payload={"heartbeat_md": ""},
            target_agent=agent.did,
            urgency=Urgency.NORMAL,
        )
        result = await dispatcher.dispatch_signal(signal)
        await backend.close()
        return result

    result = asyncio.run(_run())
    assert result.status == Status.OK, (
        f"22:00:00 (exact active_end) must stay active under the +1 minute "
        f"inversion; got {result.status} ({result.error})"
    )


def test_heartbeat_active_end_plus_one_minute_is_quiet():
    """Boundary on the other side: 22:01:00 falls inside quiet window."""
    from datetime import datetime, timezone

    from kestrel_sdk.signals import (
        Signal,
        SignalMode,
        Status,
        Urgency,
    )
    from kestrel_sovereign.signals import (
        OrderedLockManager,
        SignalDispatcher,
        SignalLogStore,
        SourceRegistry,
    )

    reg = build_heartbeat_registration(
        interval_seconds=1800,
        active_hours_start="09:00",
        active_hours_end="22:00",
    )
    fixed_now = datetime(2026, 5, 1, 22, 1, 0, tzinfo=timezone.utc)

    class _Stub:
        did = "did:test:boundary2"
        async def process_input(self, prompt):
            raise AssertionError("LLM must not be called when quiet")
        def _track_background_task(self, coro, *, name):
            import asyncio
            return asyncio.create_task(coro, name=name)

    agent = _Stub()
    registry = SourceRegistry()
    registry.register(reg)

    import asyncio

    async def _run():
        from kestrel_sovereign.storage.db import SQLiteBackend
        backend = SQLiteBackend(":memory:")
        await backend.connect()
        store = SignalLogStore(backend)
        await store.initialize()
        dispatcher = SignalDispatcher(
            agent=agent,
            registry=registry,
            lock_manager=OrderedLockManager(),
            store=store,
            clock=lambda: fixed_now,
        )
        signal = Signal(
            source="heartbeat",
            kind="tick",
            mode=SignalMode.COGNITION,
            payload={"heartbeat_md": ""},
            target_agent=agent.did,
            urgency=Urgency.NORMAL,
        )
        result = await dispatcher.dispatch_signal(signal)
        await backend.close()
        return result

    result = asyncio.run(_run())
    assert result.status == Status.DROPPED_QUIET_HOURS


def test_heartbeat_quiet_hours_handles_midnight_active_end():
    """active_hours_end = 23:59 → quiet starts at 00:00 (wraps).
    Verifies the _add_minute helper handles the day boundary."""
    from datetime import time as dtime

    reg = build_heartbeat_registration(
        interval_seconds=1800,
        active_hours_start="06:00",
        active_hours_end="23:59",
    )
    assert reg.attention_policy.quiet_hours == (dtime(0, 0), dtime(6, 0))


def test_heartbeat_registration_no_active_hours_means_no_quiet_window():
    reg = build_heartbeat_registration(
        interval_seconds=1800,
        active_hours_start=None,
        active_hours_end=None,
    )
    assert reg.attention_policy.quiet_hours is None


def test_heartbeat_schema_rejects_unknown_keys():
    reg = build_heartbeat_registration(
        interval_seconds=1800,
        active_hours_start=None,
        active_hours_end=None,
    )
    with pytest.raises(ValueError, match="unexpected keys"):
        reg.schema({"heartbeat_md": "ok", "evil_extra": "bad"})


def test_heartbeat_schema_accepts_empty_payload():
    reg = build_heartbeat_registration(
        interval_seconds=1800,
        active_hours_start=None,
        active_hours_end=None,
    )
    assert reg.schema({}) == {"heartbeat_md": ""}
    assert reg.schema({"heartbeat_md": "checklist"}) == {"heartbeat_md": "checklist"}


# ---------------------------------------------------------------------------
# End-to-end through the real dispatcher
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_heartbeat_dispatch_calls_process_input_with_rendered_prompt(components):
    agent, registry, dispatcher, _backend = components
    registry.register(
        build_heartbeat_registration(
            interval_seconds=1800,
            active_hours_start=None,
            active_hours_end=None,
        )
    )

    signal = Signal(
        source=SOURCE_NAME,
        kind="tick",
        mode=SignalMode.COGNITION,
        payload={"heartbeat_md": "Check pending tasks"},
        target_agent=agent.did,
        urgency=Urgency.NORMAL,
        visibility=Visibility.INTERNAL,
    )
    result = await dispatcher.dispatch_signal(signal)

    assert result.status == Status.OK
    assert result.artifact == "HEARTBEAT_OK"
    # Prompt template was rendered and reached the agent.
    assert len(agent.process_input_calls) == 1
    rendered = agent.process_input_calls[0]
    assert "[HEARTBEAT]" in rendered
    assert "Check pending tasks" in rendered


@pytest.mark.asyncio
async def test_heartbeat_dispatch_writes_signal_log_with_redaction(components):
    agent, registry, dispatcher, backend = components
    registry.register(
        build_heartbeat_registration(
            interval_seconds=1800,
            active_hours_start=None,
            active_hours_end=None,
        )
    )

    signal = Signal(
        source=SOURCE_NAME,
        kind="tick",
        mode=SignalMode.COGNITION,
        payload={"heartbeat_md": "Personal private reminder content"},
        target_agent=agent.did,
    )
    await dispatcher.dispatch_signal(signal)

    # Drain the background log write.
    import asyncio
    pending = [t for t in agent.background_tasks if not t.done()]
    if pending:
        await asyncio.gather(*pending)

    rows = await backend.fetch_all(
        "SELECT source, status, payload_redacted, payload_raw FROM signal_log "
        "WHERE source=?",
        (SOURCE_NAME,),
    )
    assert len(rows) == 1
    source, status, redacted, raw = rows[0]
    assert source == SOURCE_NAME
    assert status == Status.OK.value
    # Redaction policy stores length + content digest only — never the
    # text itself. Verifies the #903 P2 fix where the prior policy
    # stored the first 80 characters of HEARTBEAT.md.
    assert "len=" in redacted
    assert "sha256_12=" in redacted
    assert "Personal" not in redacted, (
        "redacted summary must not contain HEARTBEAT.md content; "
        "see #903 review P2"
    )
    assert "private" not in redacted
    assert "reminder" not in redacted
    # store_raw_trusted=False on heartbeat → payload_raw is null even for
    # TRUSTED sources, so HEARTBEAT.md content never lands in the log.
    assert raw is None


@pytest.mark.asyncio
async def test_heartbeat_dispatch_drops_quiet_hours_below_override(components):
    """End-to-end quiet-hours enforcement: register a heartbeat with a
    quiet window that includes "now", dispatch a NORMAL-urgency signal,
    and confirm the dispatcher returns DROPPED_QUIET_HOURS."""
    from datetime import datetime, time as dtime, timezone
    from kestrel_sdk.signals import AttentionPolicy

    agent, registry, dispatcher, _ = components

    # Build a registration whose quiet window is 24h (effectively always
    # quiet for this test).
    reg = build_heartbeat_registration(
        interval_seconds=1800,
        active_hours_start=None,
        active_hours_end=None,
    )
    # Override attention_policy to be always-quiet for the test.
    from dataclasses import replace
    reg = replace(
        reg,
        attention_policy=AttentionPolicy(
            quiet_hours=(dtime(0, 0), dtime(23, 59)),
            tz="UTC",
            modes_governed=frozenset({SignalMode.COGNITION}),
            urgency_override=Urgency.HIGH,
        ),
    )
    registry.register(reg)

    signal = Signal(
        source=SOURCE_NAME,
        kind="tick",
        mode=SignalMode.COGNITION,
        payload={"heartbeat_md": ""},
        target_agent=agent.did,
        urgency=Urgency.NORMAL,
    )
    result = await dispatcher.dispatch_signal(signal)
    assert result.status == Status.DROPPED_QUIET_HOURS
    assert agent.process_input_calls == [], "LLM must not be called when quiet"


@pytest.mark.asyncio
async def test_heartbeat_dispatch_high_urgency_overrides_quiet_hours(components):
    from datetime import time as dtime
    from kestrel_sdk.signals import AttentionPolicy
    from dataclasses import replace

    agent, registry, dispatcher, _ = components

    reg = build_heartbeat_registration(
        interval_seconds=1800,
        active_hours_start=None,
        active_hours_end=None,
    )
    reg = replace(
        reg,
        attention_policy=AttentionPolicy(
            quiet_hours=(dtime(0, 0), dtime(23, 59)),
            tz="UTC",
            modes_governed=frozenset({SignalMode.COGNITION}),
            urgency_override=Urgency.HIGH,
        ),
    )
    registry.register(reg)

    signal = Signal(
        source=SOURCE_NAME,
        kind="tick",
        mode=SignalMode.COGNITION,
        payload={"heartbeat_md": ""},
        target_agent=agent.did,
        urgency=Urgency.HIGH,
    )
    result = await dispatcher.dispatch_signal(signal)
    assert result.status == Status.OK
