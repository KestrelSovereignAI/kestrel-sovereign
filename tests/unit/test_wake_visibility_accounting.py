"""#2922: ``delivery_status`` must stop conflating "persisted" with "surfaced".

#2877 shipped session binding and the ``USER_VISIBLE`` + ``result_summary``
emit, but the wait reconciler still recorded a bare ``ok`` whenever the
dispatcher accepted a wake. ``ok`` therefore measured *persistence* — a turn
was written down somewhere — while reading as *visibility*. That is the exact
self-reporting failure that let a stranded wake report success for months: the
system said it worked, the observer's chat was blank, and nothing in the ledger
disagreed.

Two things make these tests different from the ones that let the gap through:

  1. **The dispatcher is real.** The reconciler suite's ``_CapturingDispatcher``
     has no ``emit_event`` and no surface ledger, so it could never observe an
     emit failing — a stub cannot fail the way production fails.
  2. **The event manager is real.** Production
     ``EventManagerMixin.emit_event`` CATCHES every listener exception and
     returns normally, so a test that monkeypatches ``emit_event`` to raise is
     testing a failure mode that does not exist. Here the listeners themselves
     raise, exactly as a broken SSE forwarder does, and the assertion is that
     the swallowed failure still reaches the ledger.

On the ceiling of the claims under test: ``queued`` means the event was
accepted by a live listener — for ``/notifications/sse``, admission to a
server-side ``asyncio.Queue``. The browser can still drop it; see
``tests/frontend/signal_completed_wake.test.mjs`` ("a wake bound to another
conversation is not painted here"). So no state here is named "surfaced", and
:func:`test_no_verdict_claims_the_user_saw_anything` guards that on purpose.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from kestrel_sdk.tools import Outcome, WaitStatus

from kestrel_sovereign.agent.event_manager import (
    EVENT_ACCEPTED,
    EVENT_BUFFERED,
    EVENT_REJECTED,
    EventManagerMixin,
)
from kestrel_sovereign.signals import (
    SURFACE_BUFFERED,
    SURFACE_EMIT_FAILED,
    SURFACE_NOT_APPLICABLE,
    SURFACE_QUEUED,
    SURFACE_REJECTED,
    SURFACE_UNKNOWN,
    OrderedLockManager,
    SignalDispatcher,
    SignalLogStore,
    SourceRegistry,
)
from kestrel_sovereign.signals.sources.wait import (
    build_wait_complete_registration,
)
from kestrel_sovereign.storage.db import SQLiteBackend
from kestrel_sovereign.waits.engine import WaitRegistry
from kestrel_sovereign.waits.reconciler import (
    VISIBILITY_QUEUED,
    VISIBILITY_UNBOUND,
    VISIBILITY_UNKNOWN,
    VISIBILITY_UNSURFACED,
    WaitReconciler,
)


# ---------------------------------------------------------------------------
# Doubles — only the agent body and the wait provider, never the seam under test
# ---------------------------------------------------------------------------


class _EventAgent(EventManagerMixin):
    """A dispatcher agent whose event bus is the REAL ``EventManagerMixin``.

    The point of inheriting rather than stubbing: production ``emit_event``
    swallows listener exceptions and (before #2922) returned ``None``, so a
    caller could not distinguish "every forwarder failed" from success. Any
    stand-in that raises out of ``emit_event`` would test a failure mode the
    real bus never produces.
    """

    did = "did:test:2922"
    agent_name = "kestrel"

    def __init__(self, response: str = "Wake turn ran."):
        self._event_listeners: list = []
        self._pending_task_notifications: list = []
        self.background_tasks: list[asyncio.Task] = []
        self.process_input_sessions: list = []
        self._response = response

    async def process_input(self, prompt: str, **kwargs):
        self.process_input_sessions.append(kwargs.get("session_id"))
        return self._response

    def _track_background_task(self, coro, *, name: str):
        task = asyncio.create_task(coro, name=name)
        self.background_tasks.append(task)
        return task


class _ReceiptlessAgent(_EventAgent):
    """An agent whose ``emit_event`` predates the #2922 receipt.

    Feature/host stand-ins in the wild return ``None``. The dispatcher must
    read that as *unobservable*, not as a successful delivery.
    """

    async def emit_event(self, event_type: str, data: dict) -> None:
        self.emitted = getattr(self, "emitted", [])
        self.emitted.append((event_type, data))
        return None


class _RaisingEmitAgent(_EventAgent):
    """An agent whose ``emit_event`` itself blows up (not just a listener)."""

    async def emit_event(self, event_type: str, data: dict):
        raise RuntimeError("event bus is down")


class _JobProvider:
    """A MonitorableWaitable over one finished provider-owned job.

    Declares ``origin_session_id`` so the reconciler can bind the wake — the
    provider-side half of #2877, and the precondition for a wake having a chat
    window to surface into at all.
    """

    kind = "example"
    signal = "wait.complete"

    def __init__(self, *, origin_session_id: str | None = "chat-sess-1"):
        self._origin = origin_session_id

    async def active_handles(self):
        return ["job-1"]

    async def poll(self, handle):
        return WaitStatus(
            Outcome.DONE,
            "job finished",
            data={"job_id": handle, "status": "complete"},
        )

    async def origin_session_id(self, handle):
        return self._origin


@pytest.fixture
async def rig(tmp_path, sqlite_database_factory):
    """Real dispatcher + real source registration + real reconciler.

    Only the agent body and the wait provider are doubles, and neither sits on
    the seam under test: the emit path, the surface ledger and the delivery
    ledger are all production code.
    """
    backend = SQLiteBackend(str(tmp_path / "signal_log.db"))
    await backend.connect()
    store = SignalLogStore(backend)
    await store.initialize()

    registry = SourceRegistry()
    registry.register(build_wait_complete_registration())

    db = await sqlite_database_factory(tmp_path / "agent.db")

    async def build(agent, *, origin_session_id="chat-sess-1"):
        dispatcher = SignalDispatcher(
            agent=agent,
            registry=registry,
            lock_manager=OrderedLockManager(),
            store=store,
        )
        wait_registry = WaitRegistry()
        wait_registry.register(_JobProvider(origin_session_id=origin_session_id))
        reconciler = WaitReconciler(
            SimpleNamespace(
                did=agent.did,
                agent_id=agent.did,
                _raw_storage=SimpleNamespace(db=db),
                wait_registry=wait_registry,
                dispatcher=dispatcher,
            )
        )
        return SimpleNamespace(
            agent=agent, dispatcher=dispatcher, reconciler=reconciler
        )

    agents: list = []

    async def make(agent=None, *, origin_session_id="chat-sess-1"):
        agent = agent if agent is not None else _EventAgent()
        agents.append(agent)
        return await build(agent, origin_session_id=origin_session_id)

    yield make

    for agent in agents:
        pending = [t for t in agent.background_tasks if not t.done()]
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
    await backend.close()


async def _drain(agent) -> None:
    """Let the dispatch tasks (and their outcome writers) settle.

    ``dispatch_signal`` joins its own outcome writers before returning, so once
    these tasks are done the surface record for each signal already exists —
    the reconciler's next harvest never races the emit it is reporting on.
    """
    pending = [t for t in agent.background_tasks if not t.done()]
    if pending:
        await asyncio.gather(*pending, return_exceptions=True)


async def _cycle(rig_obj):
    """One enqueue tick, then the harvest tick that records the outcome."""
    await rig_obj.reconciler.reconcile()
    await _drain(rig_obj.agent)
    harvest = await rig_obj.reconciler.reconcile()
    await _drain(rig_obj.agent)
    return harvest


# ---------------------------------------------------------------------------
# The receipt itself — production emit_event, production failure mode
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_emit_event_reports_rejection_it_used_to_swallow():
    """A listener that raises is logged and skipped — and now COUNTED.

    Swallowing is correct behavior (one broken forwarder must not deny the
    event to the others, and a UI notification is never worth failing the work
    that produced it). Swallowing SILENTLY is the bug: it left the caller
    unable to tell total failure from success.
    """
    agent = _EventAgent()

    async def _broken(event_type, data):
        raise ConnectionError("SSE client vanished")

    agent.add_event_listener(_broken)
    receipt = await agent.emit_event("signal_completed", {"a": 1})

    assert receipt.outcome == EVENT_REJECTED
    assert receipt.listeners == 1
    assert receipt.accepted == 0
    assert receipt.rejected == 1


@pytest.mark.asyncio
async def test_emit_event_reports_partial_acceptance():
    agent = _EventAgent()
    got: list = []

    async def _ok(event_type, data):
        got.append((event_type, data))

    async def _broken(event_type, data):
        raise RuntimeError("boom")

    agent.add_event_listener(_broken)
    agent.add_event_listener(_ok)
    receipt = await agent.emit_event("signal_completed", {"a": 1})

    assert receipt.outcome == EVENT_ACCEPTED
    assert (receipt.accepted, receipt.rejected) == (1, 1)
    assert len(got) == 1, "a raising listener must not deny the event to others"


@pytest.mark.asyncio
async def test_emit_event_reports_buffering_when_nobody_is_connected():
    agent = _EventAgent()
    receipt = await agent.emit_event("signal_completed", {"a": 1})

    assert receipt.outcome == EVENT_BUFFERED
    assert receipt.listeners == 0
    assert agent.get_pending_events() == [("signal_completed", {"a": 1})], (
        "buffering for replay is preserved (#1551) — it is simply not a "
        "delivery to anyone yet"
    )


# ---------------------------------------------------------------------------
# Reconciler accounting over the real rail
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_bound_wake_reaching_a_live_listener_records_ok_queued(rig):
    r = await rig()
    seen: list = []

    async def _forward(event_type, data):
        seen.append((event_type, data))

    r.agent.add_event_listener(_forward)

    harvest = await _cycle(r)

    assert [e for e in seen if e[0] == "signal_completed"], (
        "the bound wake must reach the SSE forwarder at all"
    )
    row = await r.reconciler._store.get("example", "job-1")
    assert row.last_delivery_status == "ok_queued"
    assert row.last_surface_status == SURFACE_QUEUED
    assert harvest.data["signals_persisted"] == 1
    assert harvest.data["signals_queued"] == 1
    assert harvest.data["signals_unsurfaced"] == 0
    assert harvest.data["signals_visibility_unknown"] == 0


@pytest.mark.asyncio
async def test_every_forwarder_failing_is_recorded_unsurfaced_not_ok(rig):
    """The #2922 headline case, in its production shape.

    ``EventManagerMixin.emit_event`` catches the listener's exception and
    returns normally, so the dispatcher sees a clean call. Before the receipt,
    that clean return became ``last_delivery_status='ok'`` — a wake no consumer
    ever received, reported as delivered.
    """
    r = await rig()
    attempts: list = []

    async def _broken(event_type, data):
        attempts.append(event_type)
        raise ConnectionError("SSE stream closed mid-write")

    r.agent.add_event_listener(_broken)

    harvest = await _cycle(r)

    assert attempts == ["signal_completed"], "the emit must actually be tried"
    row = await r.reconciler._store.get("example", "job-1")
    assert row.last_delivery_status == "ok_unsurfaced", (
        "every forwarder failed — recording this as ok is the bug"
    )
    assert row.last_surface_status == SURFACE_REJECTED
    assert harvest.data["signals_persisted"] == 1, "the turn DID run and persist"
    assert harvest.data["signals_unsurfaced"] == 1
    assert harvest.data["signals_queued"] == 0
    # The transition still locks, so the wake is not re-emitted forever: the
    # fix is honest accounting, not a retry loop.
    assert row.last_signaled_outcome == "done:complete"


@pytest.mark.asyncio
async def test_one_surviving_forwarder_still_counts_as_queued(rig):
    r = await rig()
    delivered: list = []

    async def _broken(event_type, data):
        raise RuntimeError("half the fleet is down")

    async def _ok(event_type, data):
        delivered.append(event_type)

    r.agent.add_event_listener(_broken)
    r.agent.add_event_listener(_ok)

    harvest = await _cycle(r)

    assert delivered == ["signal_completed"]
    row = await r.reconciler._store.get("example", "job-1")
    assert row.last_delivery_status == "ok_queued"
    assert harvest.data["signals_queued"] == 1


@pytest.mark.asyncio
async def test_no_listener_connected_is_unsurfaced_even_though_buffered(rig):
    """Buffering keeps the event replayable (#1551) — it is not a delivery.

    A headless host with no browser attached is the ordinary case for this;
    reporting it as ``ok`` is what made "the agent woke but I never saw it"
    indistinguishable from success.
    """
    r = await rig()

    harvest = await _cycle(r)

    row = await r.reconciler._store.get("example", "job-1")
    assert row.last_delivery_status == "ok_unsurfaced"
    assert row.last_surface_status == SURFACE_BUFFERED
    assert harvest.data["signals_unsurfaced"] == 1
    assert r.agent.get_pending_events(), (
        "the event is still queued for replay to the next client"
    )


@pytest.mark.asyncio
async def test_unbound_wake_is_recorded_unbound_not_ok(rig):
    """No origin session -> built INTERNAL -> no emit was ever attempted.

    Correct behavior (unattended cron/CLI work has no chat window, and the
    notifications stream is pinned to the agent rather than a session), but it
    is not a delivery to a user and must not read like one.
    """
    r = await rig(origin_session_id=None)
    seen: list = []

    async def _forward(event_type, data):
        seen.append(event_type)

    r.agent.add_event_listener(_forward)

    harvest = await _cycle(r)

    row = await r.reconciler._store.get("example", "job-1")
    assert row.last_delivery_status == "ok_unbound"
    assert row.last_surface_status == SURFACE_NOT_APPLICABLE
    assert harvest.data["signals_unbound"] == 1
    assert "signal_completed" not in seen, (
        "an INTERNAL wake is log-only; emitting it would paint a turn into "
        "whichever pane happens to be open"
    )
    assert r.agent.process_input_sessions == [None], "still wakes, unattended"


@pytest.mark.asyncio
async def test_receiptless_emit_event_is_visibility_unknown_not_ok(rig):
    """An emitter that reports nothing yields *unknown*, never credit.

    This is #2877's attempt-2 P2 carried forward: report unknown rather than
    asserting an outcome from ledger contents alone.
    """
    r = await rig(_ReceiptlessAgent())

    harvest = await _cycle(r)

    row = await r.reconciler._store.get("example", "job-1")
    assert row.last_delivery_status == "ok_visibility_unknown"
    assert row.last_surface_status == SURFACE_UNKNOWN
    assert harvest.data["signals_visibility_unknown"] == 1
    assert harvest.data["signals_queued"] == 0


@pytest.mark.asyncio
async def test_emit_that_raises_outright_is_unsurfaced(rig):
    r = await rig(_RaisingEmitAgent())

    harvest = await _cycle(r)

    row = await r.reconciler._store.get("example", "job-1")
    assert row.last_delivery_status == "ok_unsurfaced"
    assert row.last_surface_status == SURFACE_EMIT_FAILED
    assert harvest.data["signals_unsurfaced"] == 1


@pytest.mark.asyncio
async def test_lost_surface_record_reports_unknown_rather_than_ok(rig):
    """The ledger is in-memory and bounded, so a record can be gone by harvest.

    A missing record is *unobservable*, which is neither success nor failure.
    Filling that gap optimistically would reintroduce the whole bug through the
    back door.
    """
    r = await rig()

    async def _forward(event_type, data):
        return None

    r.agent.add_event_listener(_forward)

    await r.reconciler.reconcile()
    await _drain(r.agent)
    r.dispatcher._surface_records.clear()  # restart / eviction
    harvest = await r.reconciler.reconcile()
    await _drain(r.agent)

    row = await r.reconciler._store.get("example", "job-1")
    assert row.last_delivery_status == "ok_visibility_unknown"
    assert row.last_surface_status is None
    assert harvest.data["signals_visibility_unknown"] == 1


@pytest.mark.asyncio
async def test_dispatcher_without_a_surface_ledger_reports_unknown(rig):
    """A foreign/legacy dispatcher exposes no ``surface_record``.

    The reconciler must degrade to *unknown*, not to the pre-#2922 ``ok``.
    """
    r = await rig()

    async def _forward(event_type, data):
        return None

    r.agent.add_event_listener(_forward)

    await r.reconciler.reconcile()
    await _drain(r.agent)
    # Hide the ledger the way an older dispatcher would: no such attribute.
    r.reconciler._agent.dispatcher = SimpleNamespace(
        enqueue_signal=r.dispatcher.enqueue_signal
    )
    harvest = await r.reconciler.reconcile()
    await _drain(r.agent)

    row = await r.reconciler._store.get("example", "job-1")
    assert row.last_delivery_status == "ok_visibility_unknown"
    assert harvest.data["signals_visibility_unknown"] == 1


# ---------------------------------------------------------------------------
# The honesty ceiling
# ---------------------------------------------------------------------------


def test_no_verdict_claims_the_user_saw_anything():
    """Queue acceptance is the strongest verdict, and it is not a render.

    ``/notifications/sse`` accepting an event means it landed on a server-side
    queue. ``chat.js`` still discards a wake whose ``session_id`` is not the
    open pane's conversation — proven in
    ``tests/frontend/signal_completed_wake.test.mjs``. So the honest ceiling is
    "queued", and this test exists to fail loudly if a later change promotes it
    to a claim the server cannot support.
    """
    verdicts = {
        VISIBILITY_QUEUED,
        VISIBILITY_UNSURFACED,
        VISIBILITY_UNBOUND,
        VISIBILITY_UNKNOWN,
    }
    assert VISIBILITY_QUEUED == "queued"
    positive = verdicts - {VISIBILITY_UNSURFACED}
    for word in ("surfaced", "seen", "rendered", "displayed", "read"):
        assert not any(word in v for v in positive), (
            f"{word!r} asserts something no server-side observation can prove"
        )


@pytest.mark.asyncio
async def test_surface_ledger_is_bounded(rig):
    """The ledger is diagnostic provenance, not a second audit trail."""
    r = await rig()
    r.dispatcher._MAX_SURFACE_RECORDS = 3
    for i in range(5):
        r.dispatcher._record_surface(f"sig-{i}", SURFACE_QUEUED)

    assert len(r.dispatcher._surface_records) == 3
    assert r.dispatcher.surface_record("sig-0") is None
    assert r.dispatcher.surface_record("sig-4").status == SURFACE_QUEUED
