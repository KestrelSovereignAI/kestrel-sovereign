"""Fallback wait-reconcile driver + reconcile serialization (#2729 P1).

The wait reconciler turns durable ``mode="signal"`` watches into cognition
wakes. Historically it was driven ONLY by the ``wait_reconcile`` cron seeded by
the OPTIONAL ``SchedulerFeature`` — so a valid minimal profile (Peers + Wait,
no Scheduler) could accept a signal watch that then never woke.

This suite pins the two halves of the fix:

1. **The mandatory WaitFeature owns a fallback driver** that drives the
   reconciler when no scheduler cron keeps it fresh, and stands down when one
   does (staleness-gated, so no double-driving in the common case).
2. **Reconcile ticks are serialized** so the two possible drivers can never
   race into a duplicate wake for one terminal transition.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from kestrel_sdk.signals import Status
from kestrel_sdk.tools import Outcome, WaitStatus

from kestrel_sovereign.features.wait.feature import WaitFeature
from kestrel_sovereign.storage.async_wait_signal_store import WaitSignalStore
from kestrel_sovereign.waits.engine import WaitRegistry
from kestrel_sovereign.waits.reconciler import WaitReconciler, run_wait_reconcile

AGENT_ID = "did:test:fallback"


class _CapturingDispatcher:
    """Collects enqueued signals; each task resolves OK (harvested next tick)."""

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


class _TerminalProvider:
    """Poll-only provider whose handle is already terminal (DONE)."""

    kind = "x"
    signal = None

    def __init__(self):
        self.polls = 0

    async def poll(self, handle):
        self.polls += 1
        # Yield so two concurrent reconciles genuinely interleave — this is
        # exactly the window the serialization lock has to close.
        await asyncio.sleep(0)
        return WaitStatus(Outcome.DONE, f"{handle} done", data={"handle": handle})


class _FakeReconciler:
    """Exposes only the staleness surface the fallback driver reads."""

    def __init__(self, since):
        self._since = since

    def seconds_since_last_reconcile(self):
        return self._since


@pytest.fixture
async def db(tmp_path, sqlite_database_factory):
    return await sqlite_database_factory(tmp_path / "agent.db")


def _make_agent(db, dispatcher=None, registry=None):
    return SimpleNamespace(
        did=AGENT_ID,
        agent_id=AGENT_ID,
        _raw_storage=SimpleNamespace(db=db),
        wait_registry=registry if registry is not None else WaitRegistry(),
        dispatcher=dispatcher or _CapturingDispatcher(),
    )


# ---------------------------------------------------------------------------
# Staleness gating — the fallback defers to a live scheduler cron
# ---------------------------------------------------------------------------


def test_fallback_due_when_no_reconciler_yet():
    feature = WaitFeature(agent=None)
    assert feature._fallback_reconcile_due(None) is True


def test_fallback_due_when_never_reconciled():
    feature = WaitFeature(agent=None)
    assert feature._fallback_reconcile_due(_FakeReconciler(None)) is True


def test_fallback_stands_down_when_recently_reconciled():
    """A scheduler cron at 60s keeps ``since`` below the 90s threshold, so the
    fallback loop must NOT drive (no double-driving)."""
    feature = WaitFeature(agent=None)
    assert feature._fallback_reconcile_due(_FakeReconciler(5.0)) is False
    assert (
        feature._fallback_reconcile_due(
            _FakeReconciler(feature._FALLBACK_RECONCILE_STALE_SECONDS - 1)
        )
        is False
    )


def test_fallback_due_when_stale():
    """No scheduler keeping it fresh → the last tick ages past the threshold →
    the fallback drives."""
    feature = WaitFeature(agent=None)
    stale = feature._FALLBACK_RECONCILE_STALE_SECONDS + 10
    assert feature._fallback_reconcile_due(_FakeReconciler(stale)) is True


# ---------------------------------------------------------------------------
# WaitFeature owns the fallback background task
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_post_all_features_loaded_starts_fallback_task():
    started = []

    def _track(coro, *, name):
        task = asyncio.ensure_future(coro)
        started.append((name, task))
        return task

    agent = SimpleNamespace(
        wait_registry=WaitRegistry(),
        _track_background_task=_track,
    )
    feature = WaitFeature(agent=None)
    feature.agent = agent

    await feature.post_all_features_loaded(agent)
    try:
        assert [name for name, _ in started] == ["wait_fallback_reconcile"]
        # Recorded as a feature-owned task so shutdown/disable cancels it.
        assert feature._owned_background_tasks
    finally:
        for _, task in started:
            task.cancel()
        await asyncio.gather(*(t for _, t in started), return_exceptions=True)


@pytest.mark.asyncio
async def test_post_all_features_loaded_noop_without_wait_registry():
    """Standalone / no wait engine → no background task is started."""
    started = []

    def _track(coro, *, name):  # pragma: no cover - must not be called
        started.append(name)
        coro.close()
        return None

    agent = SimpleNamespace(wait_registry=None, _track_background_task=_track)
    feature = WaitFeature(agent=None)
    feature.agent = agent
    await feature.post_all_features_loaded(agent)
    assert started == []


# ---------------------------------------------------------------------------
# Reconcile serialization — two drivers, exactly one wake
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_concurrent_reconciles_emit_one_signal(db):
    """The scheduler cron and the fallback loop can both call reconcile. The
    lock must ensure two overlapping ticks do NOT both enqueue a wake for the
    same terminal handle (#2729 P1)."""
    dispatcher = _CapturingDispatcher()
    registry = WaitRegistry()
    registry.register(_TerminalProvider())
    agent = _make_agent(db, dispatcher, registry)

    reconciler = WaitReconciler(agent)
    agent._wait_reconciler = reconciler
    await reconciler._store.start_watch("x", "h1")

    # Two concurrent reconcile ticks racing on the same watched handle.
    await asyncio.gather(reconciler.reconcile(), reconciler.reconcile())

    assert len(dispatcher.signals) == 1
    assert dispatcher.signals[0].payload["handle"] == "h1"

    # And it completes exactly once thereafter.
    await reconciler.reconcile()
    await reconciler.reconcile()
    assert len(dispatcher.signals) == 1


@pytest.mark.asyncio
async def test_reconcile_stamps_last_reconcile_time(db):
    """After a tick, the staleness clock is set so the fallback driver can see
    the reconciler is fresh."""
    agent = _make_agent(db)
    reconciler = WaitReconciler(agent)
    assert reconciler.seconds_since_last_reconcile() is None
    await reconciler.reconcile()
    since = reconciler.seconds_since_last_reconcile()
    assert since is not None and since >= 0


@pytest.mark.asyncio
async def test_fallback_loop_drives_reconcile_when_no_scheduler(db):
    """End-to-end: with the poll interval shrunk to near-zero and no scheduler
    keeping it fresh, the fallback loop actually drives a reconcile tick that
    wakes on a terminal watched handle."""
    dispatcher = _CapturingDispatcher()
    registry = WaitRegistry()
    registry.register(_TerminalProvider())
    agent = _make_agent(db, dispatcher, registry)

    reconciler = WaitReconciler(agent)
    agent._wait_reconciler = reconciler
    await reconciler._store.start_watch("x", "h1")

    feature = WaitFeature(agent=None)
    feature.agent = agent
    # Drive quickly and treat anything as stale so the loop runs every tick.
    feature._FALLBACK_RECONCILE_POLL_SECONDS = 0
    feature._FALLBACK_RECONCILE_STALE_SECONDS = 0

    task = asyncio.ensure_future(feature._fallback_reconcile_loop(agent))
    try:
        # Wait on the actual condition with a bounded wall-clock timeout, NOT
        # a fixed number of event-loop turns. The reconcile is aiosqlite-backed
        # (its awaits bounce to a worker thread and back), so it needs real
        # time and many loop iterations to reach the enqueue — a fixed
        # ``sleep(0)`` count was not enough under loaded CI and flaked here
        # (#2729). Polling the condition on wall-clock is deterministic; the
        # reconciler dedups the transition (``last_signaled_outcome``), so the
        # count can never exceed one however many ticks run before we observe.
        async def _await_first_signal():
            while not dispatcher.signals:
                await asyncio.sleep(0.01)

        await asyncio.wait_for(_await_first_signal(), timeout=5.0)
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    assert len(dispatcher.signals) == 1
    assert dispatcher.signals[0].payload["handle"] == "h1"
