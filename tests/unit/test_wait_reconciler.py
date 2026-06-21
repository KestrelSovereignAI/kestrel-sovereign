"""Tests for the generic wait reconciler (Wave 2 of #1860).

The reconciler is the generic successor to the talon-specific
``talon_monitor`` cron: it enumerates every MonitorableWaitable provider in
``agent.wait_registry``, polls each provider's in-flight handles, and emits
one COGNITION signal per terminal-state transition — reproducing
talon_monitor's proven TWO-PHASE delivery semantics generically.

These tests port the INTENT of the retired talon_monitor suite (formerly in
``test_talon_env_and_health.py``) to the generic reconciler, plus cover the
generic signal-construction the bespoke ``build_signal_for_completed_job``
builder used to own:

  - dedup once per transition (#1510)
  - soft-fail leaves the transition re-tryable, fresh attempt suffix (#1528)
  - hard-fail locks signaled to stop loops (#1528)
  - retry cap synthesizes max_attempts_exceeded (#1528 codex P2)
  - restart with a lost in-memory handle → lost_at_restart, re-detect (#1528)
  - no dispatcher → skipped, NOT marked signaled (#1510)
  - in-flight handle is not re-enqueued (#1528)
  - provider signal name routes the source; payload spreads WaitStatus.data
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from kestrel_sdk.signals import Status
from kestrel_sdk.tools import Outcome, WaitStatus

from kestrel_sovereign.storage.async_database import AsyncDatabase
from kestrel_sovereign.waits.engine import WaitRegistry
from kestrel_sovereign.waits.reconciler import (
    MAX_DELIVERY_ATTEMPTS,
    WaitReconciler,
    run_wait_reconcile,
)


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class _CapturingDispatcher:
    """Minimal SignalDispatcher stand-in — collects enqueued signals and
    returns SignalHandles whose task resolves to a configurable
    SignalResult. Mirrors the proven talon_monitor test double, but generic.

    The reconciler calls ``enqueue_signal`` (fire-and-forget) and harvests
    delivery outcomes via ``handle.task.result()`` on the NEXT tick. Default
    outcome is ``Status.OK``; ``pending=True`` keeps the task in flight.
    """

    def __init__(self, status_override=None, error_override=None, pending=False):
        self.signals = []
        self._status = status_override or Status.OK
        self._error = error_override
        self._pending = pending
        # A gate that keeps a ``pending=True`` task genuinely in flight until
        # the test releases it — otherwise a trivially-returning coro would
        # complete on the next loop yield and defeat the in-flight assertion.
        self._gate = asyncio.Event()

    def release(self):
        self._gate.set()

    async def enqueue_signal(self, signal):
        from kestrel_sdk.signals.models import SignalHandle, SignalResult

        self.signals.append(signal)

        async def _coro():
            if self._pending:
                await self._gate.wait()
            return SignalResult(
                signal_id=signal.id,
                status=self._status,
                mode=signal.mode,
                duration_ms=1,
                error=self._error,
            )

        task = asyncio.create_task(_coro())
        if not self._pending:
            await task
        return SignalHandle(signal_id=signal.id, task=task)


class _FakeProvider:
    """A MonitorableWaitable test double over an in-memory handle table.

    ``states`` maps handle -> Outcome; ``active`` is the list active_handles
    returns. Declares a ``signal`` name so we can assert source routing.
    """

    kind = "fake"
    signal = "wait.complete"

    def __init__(self, states=None, active=None, signal="wait.complete"):
        self._states = states or {}
        self._active = active if active is not None else list(self._states)
        self.signal = signal

    def set(self, handle, outcome, *, summary="done", data=None):
        self._states[handle] = (outcome, summary, data or {})
        if handle not in self._active:
            self._active.append(handle)

    async def active_handles(self):
        return list(self._active)

    async def poll(self, handle):
        entry = self._states.get(handle)
        if entry is None:
            return WaitStatus(Outcome.FAILED, f"unknown {handle}", data={})
        outcome, summary, data = entry
        return WaitStatus(outcome, summary, data=dict(data))


async def _make_agent(tmp_path, provider, dispatcher):
    db = await AsyncDatabase.sqlite(str(tmp_path / "agent.db"))
    registry = WaitRegistry()
    registry.register(provider)
    return SimpleNamespace(
        did="did:test:agent",
        agent_id="did:test:agent",
        _raw_storage=SimpleNamespace(db=db),
        wait_registry=registry,
        dispatcher=dispatcher,
    )


# ---------------------------------------------------------------------------
# Two-phase delivery + dedup
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_emits_one_signal_per_transition(tmp_path):
    provider = _FakeProvider()
    provider.set("h1", Outcome.DONE, summary="all good", data={"x": 1})
    dispatcher = _CapturingDispatcher()
    agent = await _make_agent(tmp_path, provider, dispatcher)
    rec = WaitReconciler(agent)

    # Tick 1 enqueues, tick 2 harvests the (already-complete) task.
    t1 = await rec.reconcile()
    assert t1.data["signals_enqueued"] == 1
    assert t1.data["signals_emitted"] == 0
    assert len(dispatcher.signals) == 1
    sig = dispatcher.signals[0]
    assert sig.payload["kind"] == "fake"
    assert sig.payload["handle"] == "h1"
    assert sig.payload["outcome"] == "done"
    assert sig.payload["summary"] == "all good"
    # WaitStatus.data is spread into the payload.
    assert sig.payload["x"] == 1

    t2 = await rec.reconcile()
    assert t2.data["signals_emitted"] == 1
    assert t2.data["signals_hard_failed"] == 0
    assert t2.data["signals_soft_failed"] == 0

    # Tick 3 — no further change, must NOT re-emit.
    t3 = await rec.reconcile()
    assert t3.data["signals_emitted"] == 0
    assert t3.data["signals_enqueued"] == 0
    assert len(dispatcher.signals) == 1


@pytest.mark.asyncio
async def test_no_signal_for_pending_handles(tmp_path):
    provider = _FakeProvider()
    provider.set("h1", Outcome.PENDING, summary="working")
    dispatcher = _CapturingDispatcher()
    agent = await _make_agent(tmp_path, provider, dispatcher)
    rec = WaitReconciler(agent)

    t = await rec.reconcile()
    assert t.data["signals_enqueued"] == 0
    assert t.data["signals_emitted"] == 0
    assert dispatcher.signals == []


@pytest.mark.asyncio
async def test_records_ok_as_delivered_and_locks_outcome(tmp_path):
    provider = _FakeProvider()
    provider.set("h1", Outcome.DONE)
    dispatcher = _CapturingDispatcher()
    agent = await _make_agent(tmp_path, provider, dispatcher)
    rec = WaitReconciler(agent)

    t1 = await rec.reconcile()
    assert t1.data["signals_enqueued"] == 1
    # Pending row recorded with the enqueued signal id.
    row = await rec._store.get("fake", "h1")
    assert row.pending_signal_id == dispatcher.signals[0].id
    assert row.pending_signaled_target == "done"

    t2 = await rec.reconcile()
    assert t2.data["signals_emitted"] == 1
    row = await rec._store.get("fake", "h1")
    assert row.last_signaled_outcome == "done"
    assert row.last_delivery_status == "ok"
    assert row.last_delivery_attempts == 1
    assert row.pending_signal_id is None
    assert t2.data["transitions"][0]["delivery_status"] == "ok"


@pytest.mark.asyncio
async def test_coalesced_counts_as_delivered(tmp_path):
    provider = _FakeProvider()
    provider.set("h1", Outcome.DONE)
    dispatcher = _CapturingDispatcher(status_override=Status.COALESCED)
    agent = await _make_agent(tmp_path, provider, dispatcher)
    rec = WaitReconciler(agent)

    await rec.reconcile()
    t = await rec.reconcile()
    assert t.data["signals_emitted"] == 1
    row = await rec._store.get("fake", "h1")
    assert row.last_signaled_outcome == "done"
    assert row.last_delivery_status == "coalesced"


@pytest.mark.asyncio
async def test_corrected_native_status_resignals_within_same_outcome(tmp_path):
    """Regression (codex Wave 2 P2): when a provider's native status changes
    but maps to the SAME generic Outcome (talon finished_unknown -> failed,
    both FAILED), the corrected transition must re-signal — dedup is on the
    outcome+native-status token, not the bare outcome."""
    provider = _FakeProvider()
    provider.set("h1", Outcome.FAILED, summary="finished, no exit code",
                 data={"status": "finished_unknown"})
    dispatcher = _CapturingDispatcher()
    agent = await _make_agent(tmp_path, provider, dispatcher)
    rec = WaitReconciler(agent)

    # First terminal state: enqueue + harvest delivered.
    await rec.reconcile()
    await rec.reconcile()
    row = await rec._store.get("fake", "h1")
    assert row.last_signaled_outcome == "failed:finished_unknown"
    assert len(dispatcher.signals) == 1

    # The late exit sidecar lands: same Outcome.FAILED, different native status.
    provider.set("h1", Outcome.FAILED, summary="exit code 2",
                 data={"status": "failed"})
    t = await rec.reconcile()
    assert t.data["signals_enqueued"] == 1  # re-signaled, not suppressed
    await rec.reconcile()
    row = await rec._store.get("fake", "h1")
    assert row.last_signaled_outcome == "failed:failed"
    assert len(dispatcher.signals) == 2


# ---------------------------------------------------------------------------
# Soft / hard fail classification
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_soft_fail_does_not_lock_and_retries_with_fresh_attempt(tmp_path):
    provider = _FakeProvider()
    provider.set("h1", Outcome.DONE)
    quiet = _CapturingDispatcher(
        status_override=Status.DROPPED_QUIET_HOURS,
        error_override="inside quiet window",
    )
    agent = await _make_agent(tmp_path, provider, quiet)
    rec = WaitReconciler(agent)

    # Tick 1 enqueues (attempt 1). Tick 2 harvests the soft fail AND
    # Phase 1 re-enqueues (attempt 2). Etc.
    await rec.reconcile()
    t = await rec.reconcile()
    assert t.data["signals_emitted"] == 0
    assert t.data["signals_soft_failed"] == 1
    row = await rec._store.get("fake", "h1")
    assert row.last_signaled_outcome is None, "soft fail must NOT lock signaled"
    assert row.last_delivery_status == "dropped_quiet_hours"
    assert row.last_delivery_error == "inside quiet window"

    # Each retry gets a unique dedupe_key (attempt suffix) so the
    # coalescing window doesn't swallow it as COALESCED.
    keys = [s.dedupe_key for s in quiet.signals]
    assert len(set(keys)) == len(keys)
    assert all(k.endswith(f":attempt-{i+1}") for i, k in enumerate(keys))

    # Swap to a working dispatcher — the next pair finally delivers.
    agent.dispatcher = _CapturingDispatcher()
    await rec.reconcile()
    harvest = await rec.reconcile()
    assert harvest.data["signals_emitted"] == 1
    row = await rec._store.get("fake", "h1")
    assert row.last_signaled_outcome == "done"
    assert row.last_delivery_status == "ok"


@pytest.mark.asyncio
async def test_hard_fail_locks_signaled(tmp_path):
    provider = _FakeProvider()
    provider.set("h1", Outcome.DONE)
    bad = _CapturingDispatcher(
        status_override=Status.DROPPED_VALIDATION,
        error_override="schema mismatch",
    )
    agent = await _make_agent(tmp_path, provider, bad)
    rec = WaitReconciler(agent)

    await rec.reconcile()
    t = await rec.reconcile()
    assert t.data["signals_emitted"] == 0
    assert t.data["signals_hard_failed"] == 1
    row = await rec._store.get("fake", "h1")
    # Hard fail locks signaled — a retry would just re-fail identically.
    assert row.last_signaled_outcome == "done"
    assert row.last_delivery_status == "dropped_validation"
    assert row.last_delivery_error == "schema mismatch"
    tr = t.data["transitions"][0]
    assert tr["delivery_status"] == "dropped_validation"
    assert tr["delivery_error"] == "schema mismatch"


@pytest.mark.asyncio
async def test_dispatcher_raises_records_soft_failure(tmp_path):
    provider = _FakeProvider()
    provider.set("h1", Outcome.DONE)

    class _BrokenDispatcher:
        async def enqueue_signal(self, signal):
            raise RuntimeError("boom")

    agent = await _make_agent(tmp_path, provider, _BrokenDispatcher())
    rec = WaitReconciler(agent)

    # enqueue raises synchronously → soft-fail recorded in the same tick.
    t = await rec.reconcile()
    assert t.data["signals_emitted"] == 0
    assert t.data["signals_soft_failed"] == 1
    row = await rec._store.get("fake", "h1")
    assert row.last_signaled_outcome is None
    assert row.last_delivery_status == "dispatcher_raised"
    assert "RuntimeError" in row.last_delivery_error


# ---------------------------------------------------------------------------
# Retry cap
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_retry_cap_locks_after_max_attempts(tmp_path):
    provider = _FakeProvider()
    provider.set("h1", Outcome.DONE)
    broken = _CapturingDispatcher(
        status_override=Status.FAILED, error_override="upstream dead",
    )
    agent = await _make_agent(tmp_path, provider, broken)
    rec = WaitReconciler(agent)

    # Each enqueue/harvest pair burns one attempt. Run enough pairs to
    # exceed the cap; the reconciler must synthesize max_attempts_exceeded
    # rather than enqueue forever.
    for _ in range(MAX_DELIVERY_ATTEMPTS + 2):
        await rec.reconcile()
        await rec.reconcile()

    row = await rec._store.get("fake", "h1")
    assert row.last_signaled_outcome == "done"
    assert row.last_delivery_status in ("failed", "max_attempts_exceeded")
    assert row.last_delivery_attempts <= MAX_DELIVERY_ATTEMPTS


# ---------------------------------------------------------------------------
# Restart-lost harvest
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_pending_lost_at_restart_re_detects(tmp_path):
    """A durable pending row with no in-memory handle (Kestrel restarted
    mid-flight) must be swept as lost_at_restart (NOT locked) so the same
    tick re-detects the still-terminal handle and re-enqueues."""
    provider = _FakeProvider()
    provider.set("h1", Outcome.DONE)
    dispatcher = _CapturingDispatcher()
    agent = await _make_agent(tmp_path, provider, dispatcher)
    rec = WaitReconciler(agent)

    # Seed a durable pending row directly (simulating a pre-restart enqueue)
    # with NO in-memory task in rec._pending_signal_tasks.
    await rec._store.record_pending(
        "fake", "h1", signal_id="stale-id", target="done", attempts=1,
    )
    assert ("fake", "h1") not in rec._pending_signal_tasks

    t = await rec.reconcile()
    # Phase 0 swept the stale row → soft fail. Phase 1 re-detected the
    # still-terminal handle and enqueued a fresh attempt.
    assert t.data["signals_soft_failed"] == 1
    assert t.data["signals_enqueued"] == 1
    row = await rec._store.get("fake", "h1")
    assert row.last_signaled_outcome is None
    # A fresh (non-stale) pending id is set and an in-memory handle exists.
    assert row.pending_signal_id != "stale-id"
    assert ("fake", "h1") in rec._pending_signal_tasks


# ---------------------------------------------------------------------------
# No dispatcher / in-flight
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_no_dispatcher_does_not_mark_signaled(tmp_path):
    provider = _FakeProvider()
    provider.set("h1", Outcome.DONE)
    agent = await _make_agent(tmp_path, provider, dispatcher=None)
    rec = WaitReconciler(agent)

    t1 = await rec.reconcile()
    assert t1.data["signals_emitted"] == 0
    assert t1.data["signals_enqueued"] == 0
    assert t1.data["signals_skipped_no_dispatcher"] == 1
    # Must NOT have locked signaled — a later dispatcher should still fire.
    row = await rec._store.get("fake", "h1")
    assert row is None or row.last_signaled_outcome is None

    # Wire a real dispatcher: next pair delivers.
    dispatcher = _CapturingDispatcher()
    agent.dispatcher = dispatcher
    t2 = await rec.reconcile()
    assert t2.data["signals_enqueued"] == 1
    t3 = await rec.reconcile()
    assert t3.data["signals_emitted"] == 1
    assert len(dispatcher.signals) == 1


@pytest.mark.asyncio
async def test_does_not_re_enqueue_while_pending(tmp_path):
    provider = _FakeProvider()
    provider.set("h1", Outcome.DONE)
    pending = _CapturingDispatcher(pending=True)
    agent = await _make_agent(tmp_path, provider, pending)
    rec = WaitReconciler(agent)

    t1 = await rec.reconcile()
    assert t1.data["signals_enqueued"] == 1

    t2 = await rec.reconcile()
    # Task still in flight → not harvested, not re-enqueued.
    assert t2.data["signals_enqueued"] == 0
    assert t2.data["signals_emitted"] == 0
    assert len(pending.signals) == 1


# ---------------------------------------------------------------------------
# Source routing + skip-poll-only providers
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_provider_signal_name_routes_source(tmp_path):
    provider = _FakeProvider(signal="talon.job_complete")
    provider.set("job-7", Outcome.DONE)
    dispatcher = _CapturingDispatcher()
    agent = await _make_agent(tmp_path, provider, dispatcher)
    rec = WaitReconciler(agent)

    await rec.reconcile()
    assert dispatcher.signals[0].source == "talon.job_complete"


@pytest.mark.asyncio
async def test_poll_only_provider_is_skipped(tmp_path):
    """A provider that is NOT a MonitorableWaitable (no active_handles) must
    be skipped by enumeration — it stays valid as a blocking-wait provider."""

    class _PollOnly:
        kind = "pollonly"
        signal = None

        async def poll(self, handle):
            return WaitStatus(Outcome.DONE, "x", data={})

    db = await AsyncDatabase.sqlite(str(tmp_path / "agent.db"))
    registry = WaitRegistry()
    registry.register(_PollOnly())
    dispatcher = _CapturingDispatcher()
    agent = SimpleNamespace(
        did="did:test:agent",
        agent_id="did:test:agent",
        _raw_storage=SimpleNamespace(db=db),
        wait_registry=registry,
        dispatcher=dispatcher,
    )
    rec = WaitReconciler(agent)
    t = await rec.reconcile()
    assert t.data["scanned"] == 0
    assert dispatcher.signals == []


# ---------------------------------------------------------------------------
# Explicit watched-waits (mode="signal")
# ---------------------------------------------------------------------------


class _PollOnlyProvider:
    """A poll-only Waitable (NO active_handles) — e.g. TaskWaitable. It can
    only be woken via the explicit watched-handles path, proving "every
    waitable is wakeable if it is async" without auto-waking all tasks."""

    signal = "wait.complete"

    def __init__(self, kind="task", states=None):
        self.kind = kind
        self._states = states or {}

    def set(self, handle, outcome, *, summary="done", data=None):
        self._states[handle] = (outcome, summary, data or {})

    async def poll(self, handle):
        entry = self._states.get(handle)
        if entry is None:
            return WaitStatus(Outcome.PENDING, f"pending {handle}", data={})
        outcome, summary, data = entry
        return WaitStatus(outcome, summary, data=dict(data))


async def _make_agent_with_provider(tmp_path, provider, dispatcher):
    db = await AsyncDatabase.sqlite(str(tmp_path / "agent.db"))
    registry = WaitRegistry()
    registry.register(provider)
    agent = SimpleNamespace(
        did="did:test:agent",
        agent_id="did:test:agent",
        _raw_storage=SimpleNamespace(db=db),
        wait_registry=registry,
        dispatcher=dispatcher,
    )
    return agent


@pytest.mark.asyncio
async def test_watched_poll_only_provider_emits_on_terminal(tmp_path):
    """A watched poll-only provider (no active_handles) gets a wait.complete
    signal when its handle reaches a terminal outcome."""
    provider = _PollOnlyProvider(kind="task")
    provider.set("t1", Outcome.DONE, summary="task done", data={"y": 2})
    dispatcher = _CapturingDispatcher()
    agent = await _make_agent_with_provider(tmp_path, provider, dispatcher)
    rec = WaitReconciler(agent)

    # Register the explicit watch (what wait(target, mode="signal") does).
    await rec._store.start_watch("task", "t1")

    t1 = await rec.reconcile()
    assert t1.data["signals_enqueued"] == 1
    sig = dispatcher.signals[0]
    assert sig.payload["kind"] == "task"
    assert sig.payload["handle"] == "t1"
    assert sig.payload["outcome"] == "done"
    assert sig.payload["y"] == 2

    t2 = await rec.reconcile()
    assert t2.data["signals_emitted"] == 1
    row = await rec._store.get("task", "t1")
    assert row.last_signaled_outcome == "done"


@pytest.mark.asyncio
async def test_watched_non_terminal_stays_watched_and_emits_nothing(tmp_path):
    provider = _PollOnlyProvider(kind="task")
    provider.set("t1", Outcome.PENDING, summary="still working")
    dispatcher = _CapturingDispatcher()
    agent = await _make_agent_with_provider(tmp_path, provider, dispatcher)
    rec = WaitReconciler(agent)
    await rec._store.start_watch("task", "t1")

    t = await rec.reconcile()
    assert t.data["signals_enqueued"] == 0
    assert t.data["signals_emitted"] == 0
    assert dispatcher.signals == []
    # The watch is still active for next tick.
    watched = await rec._store.list_watched()
    assert {(w.kind, w.handle) for w in watched} == {("task", "t1")}


@pytest.mark.asyncio
async def test_watch_stops_emitting_after_delivery(tmp_path):
    provider = _PollOnlyProvider(kind="task")
    provider.set("t1", Outcome.DONE)
    dispatcher = _CapturingDispatcher()
    agent = await _make_agent_with_provider(tmp_path, provider, dispatcher)
    rec = WaitReconciler(agent)
    await rec._store.start_watch("task", "t1")

    await rec.reconcile()  # enqueue
    await rec.reconcile()  # harvest/deliver
    assert len(dispatcher.signals) == 1
    # Delivered → drops out of list_watched (last_signaled_outcome set).
    assert await rec._store.list_watched() == []
    # Further ticks must not re-emit.
    t = await rec.reconcile()
    assert t.data["signals_enqueued"] == 0
    assert t.data["signals_emitted"] == 0
    assert len(dispatcher.signals) == 1


@pytest.mark.asyncio
async def test_handle_both_active_and_watched_processed_once(tmp_path):
    """A MonitorableWaitable handle that is ALSO explicitly watched must be
    polled/emitted exactly once per tick (the active_handles loop wins; the
    watched loop skips the already-processed key)."""
    provider = _FakeProvider()  # monitorable: active_handles == states keys
    provider.set("h1", Outcome.DONE)
    dispatcher = _CapturingDispatcher()
    agent = await _make_agent(tmp_path, provider, dispatcher)
    rec = WaitReconciler(agent)
    # Watch the SAME handle the active_handles loop will also surface.
    await rec._store.start_watch("fake", "h1")

    t1 = await rec.reconcile()
    # Exactly one enqueue despite being in both wake sources.
    assert t1.data["signals_enqueued"] == 1
    assert len(dispatcher.signals) == 1
    # scanned counts the handle once (active loop), watched loop skipped it.
    assert t1.data["scanned"] == 1

    t2 = await rec.reconcile()
    assert t2.data["signals_emitted"] == 1
    assert len(dispatcher.signals) == 1


@pytest.mark.asyncio
async def test_watched_unregistered_provider_is_skipped(tmp_path):
    """A watch whose provider kind isn't registered (feature unloaded) is
    skipped — no crash, the row stays for when the provider returns."""
    db = await AsyncDatabase.sqlite(str(tmp_path / "agent.db"))
    registry = WaitRegistry()  # empty — no providers
    dispatcher = _CapturingDispatcher()
    agent = SimpleNamespace(
        did="did:test:agent",
        agent_id="did:test:agent",
        _raw_storage=SimpleNamespace(db=db),
        wait_registry=registry,
        dispatcher=dispatcher,
    )
    rec = WaitReconciler(agent)
    await rec._store.start_watch("ghost", "g1")

    t = await rec.reconcile()
    assert t.data["scanned"] == 0
    assert dispatcher.signals == []
    # Row preserved.
    assert {(w.kind, w.handle) for w in await rec._store.list_watched()} == {
        ("ghost", "g1")
    }


# ---------------------------------------------------------------------------
# Singleton helper
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_wait_reconcile_caches_singleton(tmp_path):
    provider = _FakeProvider()
    provider.set("h1", Outcome.DONE)
    dispatcher = _CapturingDispatcher()
    agent = await _make_agent(tmp_path, provider, dispatcher)

    await run_wait_reconcile(agent)
    rec1 = agent._wait_reconciler
    assert isinstance(rec1, WaitReconciler)
    # The in-memory pending map must survive across calls (same instance),
    # so tick 1's enqueue is harvested by tick 2.
    await run_wait_reconcile(agent)
    assert agent._wait_reconciler is rec1
    row = await rec1._store.get("fake", "h1")
    assert row.last_signaled_outcome == "done"
