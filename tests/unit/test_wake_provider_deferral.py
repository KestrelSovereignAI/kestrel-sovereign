"""#3302: a wake retried through a provider's advised rate-limit wait.

Emma, 2026-09-15: a Talon job ended at 15:56; at 15:56:36 the plan route began
answering 429 "advised waiting 9203s". The reconciler retried the job's
``talon.job_complete`` wake once a minute, ten times, inside a window the
provider had said would last two and a half hours, then locked it as
``max_attempts_exceeded``. When the route came back the wake was already gone,
and the only way to find the lock was ``sqlite3``.

The retry cap is right for a signal the dispatcher will always reject. It is
wrong for a provider that has named its own retry time. These tests pin:

  * the dispatcher records the provider's ``retry_at`` for a cognition turn
    that failed on a declined advised wait — and records nothing otherwise;
  * the reconciler parks such a wake until then WITHOUT spending an attempt,
    re-emits it afterwards, and labels the late wake as late;
  * a dispatcher-side failure with no advice still spends the cap exactly as
    before (the loop guard is not weakened);
  * locked and parked wakes are listed by ``wait_status``.

The dispatcher, source registration, reconciler and store are production code;
only the agent body and the wait provider are doubles.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from kestrel_sdk.signals import Signal, SignalMode, Status
from kestrel_sdk.tools import Outcome, WaitStatus
from kestrel_sdk.tools.result import ToolResultStatus

from kestrel_sovereign.features.wait.feature import WaitFeature
from kestrel_sovereign.llm.error_handling import LLMAllProvidersFailedError
from kestrel_sovereign.llm.retry import AdvisedWaitExceedsRetryBudget
from kestrel_sovereign.signals import (
    OrderedLockManager,
    SignalDispatcher,
    SignalLogStore,
    SourceRegistry,
)
from kestrel_sovereign.signals.sources.wait import (
    build_wait_complete_registration,
)
from kestrel_sovereign.storage.async_wait_signal_store import (
    DEFERRED_RATE_LIMITED,
    MAX_ATTEMPTS_EXCEEDED,
    WaitSignalStore,
)
from kestrel_sovereign.storage.db import SQLiteBackend
from kestrel_sovereign.waits.engine import WaitRegistry
from kestrel_sovereign.waits.reconciler import MAX_DELIVERY_ATTEMPTS, WaitReconciler


class _Throttle(Exception):
    status_code = 429


def _all_routes_declined(retry_at: datetime) -> LLMAllProvidersFailedError:
    """The exception ``LLMService`` raises when every route declined a wait."""
    declined = AdvisedWaitExceedsRetryBudget(
        _Throttle("429 rate_limit_error"),
        advised_seconds=9203,
        budget_seconds=120,
        retry_at=retry_at,
    )
    aggregate = LLMAllProvidersFailedError({"anthropic:plan": declined})
    aggregate.declined_wait = declined
    return aggregate


class _Agent:
    """A dispatcher agent whose cognition fails the way the test says."""

    did = "did:test:3302"
    agent_name = "kestrel"

    def __init__(self):
        self.background_tasks: list[asyncio.Task] = []
        self.turns = 0
        # None -> the turn succeeds; an exception -> the turn raises it.
        self.failure: BaseException | None = None

    async def process_input(self, prompt: str, **kwargs):
        self.turns += 1
        if self.failure is not None:
            raise self.failure
        return "Wake turn ran."

    def _track_background_task(self, coro, *, name: str):
        task = asyncio.create_task(coro, name=name)
        self.background_tasks.append(task)
        return task


class _FinishedJob:
    """A MonitorableWaitable over one failed job — Emma's Talon job."""

    kind = "example"
    signal = "wait.complete"

    async def active_handles(self):
        return ["job-1"]

    async def poll(self, handle):
        return WaitStatus(
            Outcome.FAILED, "job failed", data={"job_id": handle, "status": "failed"}
        )


@pytest.fixture
async def rig(tmp_path, sqlite_database_factory):
    backend = SQLiteBackend(str(tmp_path / "signal_log.db"))
    await backend.connect()
    log_store = SignalLogStore(backend)
    await log_store.initialize()
    sources = SourceRegistry()
    sources.register(build_wait_complete_registration())
    db = await sqlite_database_factory(tmp_path / "agent.db")

    agent = _Agent()
    dispatcher = SignalDispatcher(
        agent=agent,
        registry=sources,
        lock_manager=OrderedLockManager(),
        store=log_store,
    )
    emitted: list[Signal] = []
    real_enqueue = dispatcher.enqueue_signal

    async def _capture(signal):
        emitted.append(signal)
        return await real_enqueue(signal)

    dispatcher.enqueue_signal = _capture
    waits = WaitRegistry()
    waits.register(_FinishedJob())
    host = SimpleNamespace(
        did=agent.did,
        agent_id=agent.did,
        _raw_storage=SimpleNamespace(db=db),
        wait_registry=waits,
        dispatcher=dispatcher,
    )
    reconciler = WaitReconciler(host)
    host._wait_reconciler = reconciler

    yield SimpleNamespace(
        agent=agent,
        dispatcher=dispatcher,
        reconciler=reconciler,
        host=host,
        db=db,
        emitted=emitted,
    )

    pending = [t for t in agent.background_tasks if not t.done()]
    if pending:
        await asyncio.gather(*pending, return_exceptions=True)
    await backend.close()


async def _tick(r):
    """One reconcile tick, then let the dispatches it enqueued settle."""
    result = await r.reconciler.reconcile()
    pending = [t for t in r.agent.background_tasks if not t.done()]
    if pending:
        await asyncio.gather(*pending, return_exceptions=True)
    return result


async def _row(r):
    return await r.reconciler._store.get("example", "job-1")


# ---------------------------------------------------------------------------
# The dispatcher keeps the provider's number instead of flattening it to text
# ---------------------------------------------------------------------------


def _cognition_signal(agent) -> Signal:
    return Signal(
        source="wait.complete",
        kind="inbound",
        mode=SignalMode.COGNITION,
        payload={"kind": "example", "handle": "job-x", "outcome": "failed"},
        target_agent=agent.did,
    )


@pytest.mark.asyncio
async def test_dispatcher_records_the_provider_advised_retry_time(rig):
    retry_at = datetime.now(timezone.utc) + timedelta(seconds=9203)
    rig.agent.failure = _all_routes_declined(retry_at)
    signal = _cognition_signal(rig.agent)

    result = await rig.dispatcher.dispatch_signal(signal)

    assert result.status is Status.FAILED
    assert rig.dispatcher.cognition_retry_at(signal.id) == retry_at


@pytest.mark.asyncio
async def test_an_ordinary_cognition_failure_carries_no_retry_time(rig):
    rig.agent.failure = RuntimeError("table has no column named request_generation")
    signal = _cognition_signal(rig.agent)

    result = await rig.dispatcher.dispatch_signal(signal)

    assert result.status is Status.FAILED
    assert rig.dispatcher.cognition_retry_at(signal.id) is None, (
        "only a dated decline is advice; any other failure must still spend "
        "the retry cap"
    )


@pytest.mark.asyncio
async def test_a_route_with_its_own_non_throttle_failure_is_not_advice(rig):
    """If one route failed for a reason a retry could clear at once, the
    aggregate names no reset time (``common_declined_wait``), so the wake is
    not parked."""
    aggregate = LLMAllProvidersFailedError({"openai:api": ConnectionError("reset")})
    aggregate.declined_wait = None
    rig.agent.failure = aggregate
    signal = _cognition_signal(rig.agent)

    await rig.dispatcher.dispatch_signal(signal)

    assert rig.dispatcher.cognition_retry_at(signal.id) is None


# ---------------------------------------------------------------------------
# The reconciler parks the wake instead of burning the cap
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_rate_limited_wake_is_parked_without_spending_an_attempt(rig):
    """Emma's incident: the route is throttled for 9203s. Every minute inside
    that window used to spend an attempt; after ten the wake was locked."""
    retry_at = datetime.now(timezone.utc) + timedelta(seconds=9203)
    rig.agent.failure = _all_routes_declined(retry_at)

    await _tick(rig)  # enqueue attempt 1; its turn fails on the 429
    harvest = await _tick(rig)  # harvest: parked, not soft-failed

    row = await _row(rig)
    assert row.last_delivery_status == DEFERRED_RATE_LIMITED
    assert row.last_delivery_attempts == 0, "the declined attempt is refunded"
    assert row.delivery_deferrals == 1
    assert row.deferred_until_utc() == retry_at
    assert row.last_signaled_outcome is None, "a parked wake is not delivered"
    assert row.pending_signal_id is None
    assert "429" in (row.last_delivery_error or "")
    assert harvest.data["signals_deferred"] == 1
    assert harvest.data["signals_soft_failed"] == 0
    assert harvest.data["signals_enqueued"] == 0, "nothing re-emits inside the window"

    # Many more ticks than the cap, all inside the provider's window.
    for _ in range(MAX_DELIVERY_ATTEMPTS + 5):
        tick = await _tick(rig)
        assert tick.data["signals_parked"] == 1
        assert tick.data["signals_enqueued"] == 0

    row = await _row(rig)
    assert rig.agent.turns == 1, "no futile turn was attempted inside the window"
    assert row.last_delivery_status == DEFERRED_RATE_LIMITED
    assert row.last_signaled_outcome is None, "the wake was never locked away"


@pytest.mark.asyncio
async def test_parked_wake_is_redelivered_after_the_reset_and_says_it_is_late(rig):
    retry_at = datetime.now(timezone.utc) + timedelta(seconds=9203)
    rig.agent.failure = _all_routes_declined(retry_at)
    await _tick(rig)
    await _tick(rig)
    first = rig.emitted[0]

    # The provider's reset passes and the route comes back.
    await rig.db.execute(
        "UPDATE wait_signal_state SET delivery_deferred_until = ?",
        ((datetime.now(timezone.utc) - timedelta(seconds=1)).replace(tzinfo=None),),
    )
    rig.agent.failure = None

    resend = await _tick(rig)
    assert resend.data["signals_enqueued"] == 1
    assert len(rig.emitted) == 2
    late = rig.emitted[1]
    assert late.payload["delivery_attempt"] == 1, "a deferral is not an attempt"
    assert late.payload["delivery_deferrals"] == 1
    assert late.payload["delivery_previous_status"] == DEFERRED_RATE_LIMITED, (
        "a wake delayed by hours must not read as a first delivery (#3105)"
    )
    assert late.payload["delivery_previous_attempt_at"]
    assert late.dedupe_key != first.dedupe_key, (
        "the refunded attempt number must not collide with the parked attempt"
    )
    assert first.payload["delivery_deferrals"] == 0

    await _tick(rig)
    row = await _row(rig)
    assert row.last_signaled_outcome == "failed:failed", "delivered and locked"
    assert row.last_delivery_status.startswith("ok_")
    assert row.delivery_deferred_until is None
    assert rig.agent.turns == 2


@pytest.mark.asyncio
async def test_repeated_advised_waits_never_exhaust_the_cap(rig):
    """A provider that keeps naming short resets is honored every time; the
    cap is not the protection for this case, time is."""
    rig.agent.failure = _all_routes_declined(
        datetime.now(timezone.utc) - timedelta(seconds=1)  # already due
    )

    for _ in range(MAX_DELIVERY_ATTEMPTS + 3):
        await _tick(rig)

    row = await _row(rig)
    assert rig.agent.turns > MAX_DELIVERY_ATTEMPTS
    assert row.last_delivery_status != MAX_ATTEMPTS_EXCEEDED
    assert row.last_signaled_outcome is None
    assert row.last_delivery_attempts <= 1
    assert row.delivery_deferrals >= MAX_DELIVERY_ATTEMPTS


@pytest.mark.asyncio
async def test_ordinary_failure_still_locks_at_the_cap_and_keeps_its_error(rig):
    """The loop guard is unchanged for the case it was written for (#3292's
    shape: a failure that no amount of waiting fixes)."""
    rig.agent.failure = RuntimeError("table has no column named request_generation")

    for _ in range(MAX_DELIVERY_ATTEMPTS + 2):
        await _tick(rig)

    row = await _row(rig)
    assert row.last_delivery_status == MAX_ATTEMPTS_EXCEEDED
    assert row.last_signaled_outcome == "failed:failed"
    assert row.delivery_deferrals == 0
    assert rig.agent.turns == MAX_DELIVERY_ATTEMPTS
    assert "request_generation" in (row.last_delivery_error or ""), (
        "the lock keeps the last real failure: it is the only account of why "
        "the wake never landed"
    )


# ---------------------------------------------------------------------------
# Locked and parked wakes are visible to the agent
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_wait_status_lists_locked_and_parked_wakes(rig):
    store = rig.reconciler._store
    now = datetime.now(timezone.utc)
    # Locked after the cap.
    await store.record_pending(
        "talon", "4f08bfc5", signal_id="s1", target="failed:failed",
        attempts=MAX_DELIVERY_ATTEMPTS, attempt_at=now,
    )
    await store.record_delivery(
        "talon", "4f08bfc5", delivery_status="failed",
        delivery_error="LLMAllProvidersFailedError: 429", attempt_at=now,
    )
    await store.record_delivery(
        "talon", "4f08bfc5", delivery_status=MAX_ATTEMPTS_EXCEEDED,
        delivery_error="LLMAllProvidersFailedError: 429",
        signaled_outcome="failed:failed", attempt_at=now,
    )
    # Parked until the provider's reset.
    reset = now + timedelta(hours=2)
    await store.record_pending(
        "talon", "bd2583ac", signal_id="s2", target="done:complete",
        attempts=1, attempt_at=now,
    )
    await store.record_deferral(
        "talon", "bd2583ac", target="done:complete", retry_at=reset,
        delivery_error="429", attempt_at=now,
    )
    # Delivered normally — not listed.
    await store.record_pending(
        "talon", "ok-job", signal_id="s3", target="done:complete",
        attempts=1, attempt_at=now,
    )
    await store.record_delivery(
        "talon", "ok-job", delivery_status="ok_queued",
        signaled_outcome="done:complete", attempt_at=now,
    )

    feature = WaitFeature(agent=None)
    feature.agent = rig.host
    result = await feature.wait_status()

    assert result.status is ToolResultStatus.OK
    locked = result.data["locked"]
    deferred = result.data["deferred"]
    assert [e["ref"] for e in locked] == ["talon:4f08bfc5"]
    assert locked[0]["delivery_attempts"] == MAX_DELIVERY_ATTEMPTS
    assert "429" in locked[0]["last_error"]
    assert [e["ref"] for e in deferred] == ["talon:bd2583ac"]
    assert deferred[0]["deferred_until"]
    assert deferred[0]["delivery_deferrals"] == 1
    assert "talon:4f08bfc5" in result.confirmation
    assert "LOCKED" in result.confirmation
    assert "ok-job" not in result.confirmation


@pytest.mark.asyncio
async def test_wait_status_says_so_when_nothing_is_undelivered(rig):
    feature = WaitFeature(agent=None)
    feature.agent = rig.host

    result = await feature.wait_status()

    assert result.status is ToolResultStatus.OK
    assert result.data == {"locked": [], "deferred": []}


@pytest.mark.asyncio
async def test_wait_status_rejects_an_out_of_range_limit(rig):
    feature = WaitFeature(agent=None)
    feature.agent = rig.host

    assert (await feature.wait_status(limit=0)).status is ToolResultStatus.ERROR
    assert (
        await feature.wait_status(limit=WaitFeature._MAX_STATUS_WAKES + 1)
    ).status is ToolResultStatus.ERROR


@pytest.mark.asyncio
async def test_wait_status_without_a_wait_engine_fails_loudly():
    result = await WaitFeature(agent=None).wait_status()
    assert result.status is ToolResultStatus.ERROR


# ---------------------------------------------------------------------------
# Store semantics of a deferral
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_deferral_refunds_only_the_transition_it_belongs_to(
    tmp_path, sqlite_database_factory,
):
    db = await sqlite_database_factory(tmp_path / "wait_store.db")
    store = WaitSignalStore(db, agent_id="did:test:agent")
    now = datetime.now(timezone.utc)
    await store.record_pending(
        "talon", "job-1", signal_id="s1", target="failed:failed",
        attempts=3, attempt_at=now,
    )

    # A deferral naming another transition refunds nothing.
    await store.record_deferral(
        "talon", "job-1", target="failed:finished_unknown",
        retry_at=now + timedelta(hours=1), attempt_at=now,
    )
    assert (await store.get("talon", "job-1")).last_delivery_attempts == 3

    await store.record_deferral(
        "talon", "job-1", target="failed:failed",
        retry_at=now + timedelta(hours=1), attempt_at=now,
    )
    row = await store.get("talon", "job-1")
    assert row.last_delivery_attempts == 2
    assert row.delivery_deferrals == 2

    # Dispatching again spends the deferral; a NEW transition restarts the
    # deferral count just as it restarts the attempt count (#3105).
    await store.record_pending(
        "talon", "job-1", signal_id="s2", target="failed:failed",
        attempts=3, attempt_at=now,
    )
    row = await store.get("talon", "job-1")
    assert row.delivery_deferred_until is None
    assert row.delivery_deferrals == 2
    await store.record_pending(
        "talon", "job-1", signal_id="s3", target="done:complete",
        attempts=1, attempt_at=now,
    )
    assert (await store.get("talon", "job-1")).delivery_deferrals == 0


@pytest.mark.asyncio
async def test_deferral_columns_migrate_onto_a_legacy_table(
    tmp_path, sqlite_database_factory,
):
    """``wait_signal_state`` is CREATE TABLE IF NOT EXISTS: only the ALTERs
    give an existing database the deferral columns. A legacy row reads as
    "nothing parked, never deferred", which is true of it."""
    import sqlite3

    db_path = tmp_path / "legacy_wait_state.db"
    legacy = sqlite3.connect(db_path)
    legacy.executescript(
        """
        CREATE TABLE wait_signal_state (
            agent_id TEXT NOT NULL DEFAULT '',
            kind TEXT NOT NULL,
            handle TEXT NOT NULL,
            last_signaled_outcome TEXT,
            last_delivery_status TEXT,
            last_surface_status TEXT,
            last_delivery_error TEXT,
            last_delivery_attempts INTEGER NOT NULL DEFAULT 0,
            last_delivery_attempt_at TIMESTAMP,
            attempts_signaled_target TEXT NOT NULL DEFAULT '',
            last_attempt_started_at TIMESTAMP,
            pending_signal_id TEXT,
            pending_signaled_target TEXT,
            pending_signal_enqueued_at TIMESTAMP,
            watching INTEGER NOT NULL DEFAULT 0,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (agent_id, kind, handle)
        );
        INSERT INTO wait_signal_state
            (agent_id, kind, handle, last_delivery_status, last_delivery_attempts)
        VALUES ('did:legacy', 'talon', 'h-old', 'max_attempts_exceeded', 10);
        """
    )
    legacy.commit()
    legacy.close()

    database = await sqlite_database_factory(db_path)
    store = WaitSignalStore(database, agent_id="did:legacy")
    row = await store.get("talon", "h-old")

    assert row.delivery_deferred_until is None
    assert row.delivery_deferrals == 0
    assert [r.handle for r in await store.list_undelivered()] == ["h-old"], (
        "a wake locked before this fix is listed too — that is the backlog "
        "an agent most needs to find"
    )


def test_wait_status_description_names_the_real_lock_status():
    """The description is a literal (the static inventory cannot evaluate an
    f-string), so pin it to the status the reconciler actually writes."""
    tool = next(
        t
        for t in WaitFeature(SimpleNamespace(wait_registry=WaitRegistry())).get_tools()
        if t.name == "wait_status"
    )
    assert f"`{MAX_ATTEMPTS_EXCEEDED}`" in tool.schema.description
