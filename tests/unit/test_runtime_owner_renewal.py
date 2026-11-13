"""#4001: an unretired runtime owner is renewed whatever else stands.

A dispatcher's runtime owner is what tells a peer that the leases it holds are
live. Each review round of #4001 found another way its renewal stopped while
the owner still leased work: a shutdown that refused or failed, maintenance
that stalled, native denial of the agent's own execution scope, a plain
retained control error, and on SQLite the shared work connection's
cancellation-cleanup fence. Each time renewal shared a path with work that can
fail or wait: it ran inside durable admission (``_admit_durable_operation``),
in the owner's own execution scope, in the same task as the heartbeat's
maintenance, on the connection the agent's work uses, and it ended when
shutdown began. Once the stale threshold passed, a peer's recovery took a
lease the live owner still held, and the same work ran twice.

Renewal is now one fixed write, started in a task of its own on every tick,
outside durable admission, on no execution scope the agent's work can fence
and, on SQLite, on a connection of its own. It stops only once the owner is
durably retired. These run the real heartbeat timer on both backends. Every
obstacle in the parametrized case closes durable admission before the stale
threshold is crossed, so renewal moved back under admission fails each one.
That renewal on PostgreSQL no longer depends on the owner leasing core
cognition is pinned by ``tests/integration/test_runtime_owner_liveness_pg.py``.
"""

from __future__ import annotations

import asyncio
import contextvars
import copy
import sqlite3
from datetime import timedelta

import pytest

from kestrel_sovereign.signals.sources.a2a_task_submitted import submitted_input_identity
from tests.unit.test_a2a_native_input_fencing import (
    close_input_dispatcher,
    input_dispatcher,
    snapshot,
    wake,
)
from tests.unit.test_a2a_native_task_manager import manager as manager  # noqa: F401
from tests.unit.test_a2a_native_task_manager import request

pytestmark = [pytest.mark.asyncio, pytest.mark.timeout(60)]

WORKFLOW_CONSUMER = "workflows:wait:4001"
STALE_AFTER = timedelta(seconds=1.2)


@pytest.fixture(autouse=True)
def fast_owner_heartbeat(monkeypatch):
    """A stale threshold short enough to cross several times in one test."""
    from kestrel_sovereign.signals import SignalDispatcher

    monkeypatch.setitem(
        SignalDispatcher.__init__.__kwdefaults__, "runtime_owner_stale_after", STALE_AFTER
    )


async def _owner_row(dispatcher, store=None):
    """This owner's ``(heartbeat_at, stopped_at)`` as ``store`` reads it."""
    from kestrel_sovereign.signals.durable import DurableSignalStore

    store = store or dispatcher._durable_store
    row = await store.backend.fetch_one(
        f"SELECT heartbeat_at, stopped_at FROM {DurableSignalStore.RUNTIME_OWNERS} "
        "WHERE agent_id = ? AND owner_id = ?",
        (dispatcher._agent.did, dispatcher._durable_delivery_owner),
    )
    return tuple(
        None if value is None else store.from_timestamp_field(value) for value in row
    )


async def _submission_wake(manager, dispatcher, consumer, task):
    """The task's submission wake, persisted for this agent's consumers."""
    from kestrel_sovereign.signals.sources.a2a_task_submitted import SOURCE_NAME

    identity = submitted_input_identity(task.id, 0)
    await dispatcher.dispatch_signal(
        copy.deepcopy(wake(await snapshot(manager, task), manager)),
        source_event_id=identity,
    )
    return await dispatcher.get_durable_delivery_for_source_event(
        consumer_id=consumer, source=SOURCE_NAME, source_event_id=identity
    )


async def held_workflow_lease(manager, dispatcher, consumer, task):
    """A live lease this owner holds on the wake's event that is not core
    cognition: what a peer takes once the owner looks dead."""
    from kestrel_sovereign.signals import DurableConsumerRegistration

    await dispatcher.register_durable_consumer(
        DurableConsumerRegistration(
            consumer_id=WORKFLOW_CONSUMER, source="a2a.task_submitted",
            agent_id=dispatcher._agent.did, max_attempts=0, lease_seconds=3600,
        )
    )
    original = await _submission_wake(manager, dispatcher, consumer, task)
    held = await dispatcher.claim_durable_delivery_for_event(
        consumer_id=WORKFLOW_CONSUMER, event_id=original.event_id,
        executor_id=dispatcher._durable_delivery_owner,
    )
    assert held is not None and held.status == "leased"
    return original, held


async def assert_unretired_owner_stays_live(dispatcher, held, store=None):
    """Past the stale threshold the owner was renewed and kept its lease.

    ``store`` is the peer that reads the owner and tries to recover its
    lease; by default one sharing the dispatcher's own backend."""
    before = await _owner_row(dispatcher, store)
    assert before[1] is None
    await asyncio.sleep(STALE_AFTER.total_seconds() * 2)
    after = await _owner_row(dispatcher, store)
    assert after[1] is None
    assert after[0] > before[0], "the unretired owner stopped heartbeating"
    store = store or dispatcher._durable_store
    recovered = await store.recover_abandoned_leases(
        agent_id=dispatcher._agent.did,
        recovering_owner_id="dispatcher:peer-4001",
        stale_before=store.now_utc() - STALE_AFTER,
    )
    assert recovered == 0, "a peer reclaimed a lease the unretired owner holds"
    (lease,) = await store.list_deliveries(
        agent_id=dispatcher._agent.did, consumer_id=held.consumer_id
    )
    assert (lease.status, lease.lease_owner, lease.lease_token) == (
        "leased", dispatcher._durable_delivery_owner, held.lease_token,
    )


async def assert_retired_owner_stays_retired(dispatcher):
    """Once retired, nothing renews the owner or revives it."""
    retired = await _owner_row(dispatcher)
    assert retired[1] is not None
    assert dispatcher._runtime_owner_heartbeat_timer is None
    assert dispatcher._runtime_owner_heartbeat_task is None
    assert dispatcher._runtime_owner_renewal_task is None
    await asyncio.sleep(STALE_AFTER.total_seconds())
    assert await _owner_row(dispatcher) == retired


async def _admission_outcome(dispatcher):
    """What durable admission does now for a task like the renewal's, one with
    an empty context: ``None`` if it admitted, else what it raised."""

    async def admit():
        async with dispatcher._admit_durable_operation():
            pass

    task = asyncio.get_running_loop().create_task(admit(), context=contextvars.Context())
    done, _ = await asyncio.wait({task}, timeout=5)
    assert done, "durable admission neither admitted nor refused"
    return task.exception()


async def admission_refused(dispatcher):
    """Wait until durable admission refuses new work (self-review of #4040).

    Each obstacle must close admission before the stale threshold is
    crossed. Otherwise renewal moved back under ``_admit_durable_operation()``
    would still run, and the case could not tell."""

    async def refused():
        while await _admission_outcome(dispatcher) is None:
            await asyncio.sleep(0.01)

    await asyncio.wait_for(refused(), 5)


def fence_the_agents_own_scope(agent, dispatcher, monkeypatch):
    """Give the agent the execution scope a booted agent binds on every
    resident timer, and so on every heartbeat it starts."""
    from kestrel_sovereign.execution_custody import (
        ExecutionCustody,
        ProcessRuntimeExecutionFence,
    )
    from kestrel_sovereign.kestrel_agent import KestrelAgent

    agent._execution_custody = ExecutionCustody(ProcessRuntimeExecutionFence(agent.did))
    monkeypatch.setattr(
        type(agent), "_runtime_owner_context",
        KestrelAgent._runtime_owner_context, raising=False,
    )
    if dispatcher._runtime_owner_heartbeat_timer is not None:
        dispatcher._runtime_owner_heartbeat_timer.cancel()
        dispatcher._runtime_owner_heartbeat_timer = None
    dispatcher._schedule_runtime_owner_heartbeat()


RENEWAL_OBSTACLES = (
    "refused_shutdown",
    "failed_shutdown",
    "stalled_maintenance",
    "native_denial",
    "retained_control",
)


@pytest.mark.parametrize("obstacle", RENEWAL_OBSTACLES)
async def test_an_unretired_owner_is_renewed_whatever_else_stands(
    manager, monkeypatch, obstacle  # noqa: F811 - the imported fixture
):
    """Whatever keeps the owner unretired, the real timer renews it past the
    stale threshold and a peer's recovery reclaims nothing it holds. Once the
    obstacle clears, the owner retires and nothing renews or revives it."""
    from kestrel_sdk.storage.database.interface import QueryError

    from kestrel_sovereign.execution_custody import ExecutionCommitOutcomeError

    task = await manager.create_task(request(), creator_agent_id="did:creator")
    agent, dispatcher, consumer = await input_dispatcher(manager)
    released = asyncio.Event()
    stop_mark_faults = []
    shutting = None
    closed = False
    try:
        if obstacle == "native_denial":
            fence_the_agents_own_scope(agent, dispatcher, monkeypatch)
        _, held = await held_workflow_lease(manager, dispatcher, consumer, task)

        if obstacle == "refused_shutdown":
            # A writer's commit outcome is unknown: shutdown refuses to
            # release an owner whose storage control is uncertain.
            dispatcher._outcome_log_control_error = ExecutionCommitOutcomeError("unknown")
            with pytest.raises(BaseException):
                await close_input_dispatcher(agent, dispatcher)
        elif obstacle == "failed_shutdown":
            store = dispatcher._durable_store
            release = store.release_initial_reservations
            stop_mark_faults.append(QueryError("database is locked"))

            async def failing_stop_mark(**kwargs):
                if kwargs.get("mark_owner_stopped", True) and stop_mark_faults:
                    raise stop_mark_faults.pop()
                return await release(**kwargs)

            monkeypatch.setattr(store, "release_initial_reservations", failing_stop_mark)
            with pytest.raises(QueryError):
                await close_input_dispatcher(agent, dispatcher)
        elif obstacle == "stalled_maintenance":
            entered = asyncio.Event()

            async def stalled_sweep():
                entered.set()
                await released.wait()
                return 0

            monkeypatch.setattr(dispatcher, "_recover_abandoned_leases", stalled_sweep)
            await asyncio.wait_for(entered.wait(), 5)
            # Shutdown closes admission, then waits for the stalled pass it
            # admitted earlier to drain.
            shutting = asyncio.create_task(dispatcher.shutdown_durable_delivery())
        elif obstacle == "native_denial":
            agent._execution_custody.preserve_commit_uncertainty("unknown")
        else:
            dispatcher._outcome_log_control_error = ExecutionCommitOutcomeError("unknown")

        if obstacle in ("refused_shutdown", "failed_shutdown"):
            assert dispatcher._durable_shutdown
            assert dispatcher._durable_runtime_owner_registered
        await admission_refused(dispatcher)

        await assert_unretired_owner_stays_live(dispatcher, held)

        # The obstacle still stands: renewal went on beside it, with
        # admission closed throughout.
        assert await _admission_outcome(dispatcher) is not None
        if obstacle in ("refused_shutdown", "failed_shutdown"):
            assert dispatcher._durable_runtime_owner_registered
        elif obstacle == "stalled_maintenance":
            assert not dispatcher._runtime_owner_heartbeat_task.done()
            assert not shutting.done()
        elif obstacle == "native_denial":
            with pytest.raises(ExecutionCommitOutcomeError):
                agent._execution_custody.require_work()
        else:
            assert dispatcher._outcome_log_control_error is not None

        released.set()
        dispatcher._outcome_log_control_error = None
        if shutting is not None:
            await asyncio.wait_for(shutting, 10)
        closed = True
        await close_input_dispatcher(agent, dispatcher)
        assert not dispatcher._durable_runtime_owner_registered
        await assert_retired_owner_stays_retired(dispatcher)
    finally:
        released.set()
        dispatcher._outcome_log_control_error = None
        if shutting is not None and not shutting.done():
            await asyncio.wait({shutting}, timeout=10)
        if not closed:
            await close_input_dispatcher(agent, dispatcher)


async def fence_the_shared_sqlite_connection(backend):
    """Leave SQLite's shared work connection fenced, as the backend does.

    A write transaction is cancelled. The backend hands its rollback to the
    retained cleanup drain, and that rollback runs but loses its
    acknowledgement. Not knowing whether the abandoned work was undone, the
    backend refuses every later write on that connection until it is
    replaced. Returns the error the fence retains."""
    from kestrel_sovereign.signals.durable import DurableSignalStore

    connection = backend._connection
    rollback = connection.rollback
    lost = sqlite3.OperationalError("disk I/O error")
    lost.sqlite_errorcode = sqlite3.SQLITE_IOERR
    lost.sqlite_errorname = "SQLITE_IOERR"

    async def rollback_then_lose_its_acknowledgement():
        del connection.rollback
        await rollback()
        raise lost

    connection.rollback = rollback_then_lose_its_acknowledgement
    writing = asyncio.Event()

    async def abandoned_write():
        async with backend.transaction():
            await backend.execute(
                f"UPDATE {DurableSignalStore.RUNTIME_OWNERS} "
                "SET updated_at = updated_at WHERE agent_id = ?",
                ("did:none",),
            )
            writing.set()
            await asyncio.Event().wait()

    writer = asyncio.create_task(abandoned_write())
    try:
        await asyncio.wait_for(writing.wait(), 5)
    finally:
        writer.cancel()
        await asyncio.gather(writer, return_exceptions=True)
    drain = backend._cancelled_write_drain
    if drain is not None:
        await asyncio.wait({drain}, timeout=5)
    assert "rollback" not in vars(connection), "the abandoned write was never rolled back"
    assert backend._cancelled_write_drain is None
    assert backend.write_connection_requires_reconnect
    return backend._cancelled_write_drain_error


@pytest.mark.parametrize("native_a2a_backend_factory", ["sqlite"], indirect=True)
async def test_a_fenced_shared_sqlite_connection_does_not_stop_renewal(
    manager, native_a2a_backend_factory  # noqa: F811 - the imported fixture
):
    """Self-review of #4040: SQLite's shared work connection is fenced after
    a cancelled write's rollback ran but lost its acknowledgement, so every
    write on it is refused until it is replaced. The owner is renewed on a
    connection of its own, so a peer reaching the file through its own
    connection never finds it stale, and the fence keeps its original error.

    SQLite only: PostgreSQL has no one shared work connection to fence. Its
    renewal runs on a pool checkout that admits no work, which
    ``tests/integration/test_runtime_owner_liveness_pg.py`` pins."""
    from kestrel_sovereign.signals.durable import DurableSignalStore
    from kestrel_sovereign.storage.db.interface import ConnectionError

    task = await manager.create_task(request(), creator_agent_id="did:creator")
    agent, dispatcher, consumer = await input_dispatcher(manager)
    backend = dispatcher._durable_store.backend
    peer_backend = native_a2a_backend_factory()
    await peer_backend.connect()
    peer = DurableSignalStore(peer_backend)
    closed = False
    try:
        _, held = await held_workflow_lease(manager, dispatcher, consumer, task)
        fence = await fence_the_shared_sqlite_connection(backend)
        assert fence is not None
        # Everything else this owner writes on that connection is refused.
        with pytest.raises(ConnectionError, match="cancellation cleanup failed"):
            await dispatcher._durable_store.heartbeat_runtime_owner(
                agent_id=agent.did, owner_id=dispatcher._durable_delivery_owner,
            )

        await assert_unretired_owner_stays_live(dispatcher, held, store=peer)

        # Renewal left the fence and its original error as they were.
        assert backend._cancelled_write_drain_error is fence
        assert backend.write_connection_requires_reconnect

        # Replacing the connection lifts the fence; the owner then retires.
        await backend.connect()
        closed = True
        await close_input_dispatcher(agent, dispatcher)
        assert not dispatcher._durable_runtime_owner_registered
        retired = await _owner_row(dispatcher, peer)
        assert retired[1] is not None
        await asyncio.sleep(STALE_AFTER.total_seconds())
        assert await _owner_row(dispatcher, peer) == retired
    finally:
        if not closed:
            if backend.write_connection_requires_reconnect:
                await backend.connect()
            await close_input_dispatcher(agent, dispatcher)
        await peer_backend.close()


@pytest.mark.parametrize("mark", ["failed", "unacknowledged"])
async def test_an_owner_whose_stop_mark_fails_keeps_heartbeating_until_it_retires(
    manager, monkeypatch, mark  # noqa: F811 - the imported fixture
):
    """Shutdown stops the owner heartbeat, then marks the owner stopped. If
    that mark fails (``failed``), the owner is not retired and still leases
    work, so its heartbeat resumes until a retry retires it, rather than
    leaving a peer free to reclaim the lease. If the mark committed but its
    acknowledgement was lost (``unacknowledged``), the owner is retired: the
    resumed heartbeat must not renew or revive it. No heartbeat starts while
    the owner is being marked stopped (#4001)."""
    from kestrel_sdk.storage.database.interface import QueryError

    task = await manager.create_task(request(), creator_agent_id="did:creator")
    agent, dispatcher, consumer = await input_dispatcher(manager)
    closed = False
    try:
        _, held = await held_workflow_lease(manager, dispatcher, consumer, task)
        store = dispatcher._durable_store
        release = store.release_initial_reservations
        marks = []

        async def faulted_stop_mark(**kwargs):
            if not kwargs.get("mark_owner_stopped", True):
                return await release(**kwargs)
            # A timer firing now finds the owner being retired.
            dispatcher._start_runtime_owner_heartbeat()
            marks.append(dispatcher._runtime_owner_heartbeat_task)
            if len(marks) > 1:
                return await release(**kwargs)
            if mark == "unacknowledged":
                await release(**kwargs)
            raise QueryError("database is locked")

        monkeypatch.setattr(store, "release_initial_reservations", faulted_stop_mark)
        with pytest.raises(QueryError):
            await close_input_dispatcher(agent, dispatcher)
        assert marks == [None]
        assert dispatcher._durable_runtime_owner_registered

        if mark == "failed":
            await assert_unretired_owner_stays_live(dispatcher, held)
        else:
            stopped = await _owner_row(dispatcher)
            assert stopped[1] is not None
            await asyncio.sleep(STALE_AFTER.total_seconds() * 2)
            assert await _owner_row(dispatcher) == stopped, (
                "a heartbeat renewed an owner whose retirement committed"
            )

        closed = True
        await close_input_dispatcher(agent, dispatcher)
        assert marks == [None, None]
        assert not dispatcher._durable_runtime_owner_registered
        await assert_retired_owner_stays_retired(dispatcher)
    finally:
        if not closed:
            await close_input_dispatcher(agent, dispatcher)


async def test_stopping_the_heartbeat_leaves_no_heartbeat_unjoined(
    manager, monkeypatch  # noqa: F811 - the imported fixture
):
    """Self-review of #4040: stopping the heartbeat joins the heartbeat, then
    the renewal. A heartbeat that ends during that stop arms the next tick;
    were that timer left armed while a slow renewal is joined, it would fire
    and start a heartbeat nobody joins, which then sweeps a store its owner
    is closing. The timer is cancelled after each join."""
    from kestrel_sovereign.signals import SignalDispatcher

    monkeypatch.setitem(
        SignalDispatcher.__init__.__kwdefaults__, "runtime_owner_stale_after",
        timedelta(seconds=0.09),
    )
    agent, dispatcher, _ = await input_dispatcher(manager)
    sweeping, release_sweep = asyncio.Event(), asyncio.Event()
    renewing, release_renewal = asyncio.Event(), asyncio.Event()
    block_renewal = {"armed": False}
    store = dispatcher._durable_store
    renew = store.retain_unretired_runtime_owner

    async def stalled_sweep():
        sweeping.set()
        await release_sweep.wait()
        return 0

    async def slow_renewal(**kwargs):
        if block_renewal["armed"]:
            renewing.set()
            await release_renewal.wait()
        return await renew(**kwargs)

    monkeypatch.setattr(dispatcher, "_recover_abandoned_leases", stalled_sweep)
    monkeypatch.setattr(store, "retain_unretired_runtime_owner", slow_renewal)
    stopping = None
    try:
        await asyncio.wait_for(sweeping.wait(), 5)
        block_renewal["armed"] = True
        # The next tick starts a renewal while the stalled maintenance runs.
        await asyncio.wait_for(renewing.wait(), 5)
        stalled = dispatcher._runtime_owner_heartbeat_task
        assert stalled is not None and not stalled.done()

        stopping = asyncio.create_task(dispatcher._stop_runtime_owner_heartbeat())
        await asyncio.sleep(0)
        release_sweep.set()
        await asyncio.wait_for(asyncio.shield(stalled), 5)
        # Several intervals pass while the renewal is still being joined.
        await asyncio.sleep(0.2)
        release_renewal.set()
        await asyncio.wait_for(stopping, 5)

        assert dispatcher._runtime_owner_heartbeat_timer is None
        assert dispatcher._runtime_owner_heartbeat_task is None
        assert dispatcher._runtime_owner_renewal_task is None
        unjoined = [
            task.get_name() for task in agent.tasks
            if not task.done() and "owner_heartbeat" in task.get_name()
        ]
        assert unjoined == [], f"a heartbeat started while stopping was never joined: {unjoined}"
    finally:
        release_sweep.set()
        release_renewal.set()
        if stopping is not None and not stopping.done():
            await asyncio.wait({stopping}, timeout=5)
        await close_input_dispatcher(agent, dispatcher)


async def test_a_tick_that_loses_the_race_to_retirement_neither_fails_it_nor_writes(
    manager, monkeypatch  # noqa: F811 - the imported fixture
):
    """Self-review of #4040: a tick started its renewal and heartbeat just
    before retirement began. The renewal then finds the owner being retired
    and writes nothing, and the heartbeat that awaits it ends the same way.
    Retirement treats that as the normal end of a heartbeat, not a failure,
    and retires the owner."""
    agent, dispatcher, _ = await input_dispatcher(manager)
    store = dispatcher._durable_store
    renew = store.retain_unretired_runtime_owner
    writes = []

    async def counted_renewal(**kwargs):
        writes.append(kwargs)
        return await renew(**kwargs)

    monkeypatch.setattr(store, "retain_unretired_runtime_owner", counted_renewal)
    retired = False
    try:
        # Both tasks are created but have not run when retirement begins.
        dispatcher._start_runtime_owner_heartbeat()
        assert dispatcher._runtime_owner_renewal_task is not None
        assert dispatcher._runtime_owner_heartbeat_task is not None
        await dispatcher._retire_runtime_owner()
        retired = True

        assert writes == [], "a renewal wrote after retirement began"
        assert not dispatcher._durable_runtime_owner_registered
        assert (await _owner_row(dispatcher))[1] is not None
        assert dispatcher._runtime_owner_heartbeat_task is None
        assert dispatcher._runtime_owner_renewal_task is None
    finally:
        if not retired:
            dispatcher._runtime_owner_retiring = False
        await close_input_dispatcher(agent, dispatcher)


@pytest.mark.parametrize("native_a2a_backend_factory", ["sqlite"], indirect=True)
async def test_an_owner_whose_renewal_keeps_failing_is_reported_loudly_once(
    manager, native_a2a_backend_factory, caplog  # noqa: F811 - the imported fixture
):
    """Operator P3 on PR #4050: an unretired owner whose backend has closed
    fails to renew on every tick, and each failure was logged at ERROR, twice
    with the heartbeat that joined it. The first renewal failure for an owner
    and connection is logged at ERROR, later ones at DEBUG, and every one is
    counted. Once a renewal succeeds, a later failure is loud again."""
    import logging

    agent, dispatcher, _ = await input_dispatcher(manager)
    backend = dispatcher._durable_store.backend
    loop = asyncio.get_running_loop()

    async def failures_reach(count):
        deadline = loop.time() + 15
        while dispatcher.runtime_owner_renewal_failure_count < count:
            assert loop.time() < deadline, "renewal stopped failing or stopped running"
            await asyncio.sleep(0.02)

    def reports(fragment):
        return [
            record.levelno for record in caplog.records
            if fragment in record.getMessage()
        ]

    try:
        with caplog.at_level(logging.DEBUG, logger="kestrel_sovereign.signals.dispatcher"):
            assert dispatcher.runtime_owner_renewal_failure_count == 0
            await backend.close()
            await failures_reach(3)
            assert reports("runtime-owner renewal failed").count(logging.ERROR) == 1
            assert len(reports("runtime-owner renewal failed")) >= 3
            assert logging.ERROR not in reports("runtime-owner heartbeat failed")

            await backend.connect()
            deadline = loop.time() + 15
            while dispatcher._runtime_owner_renewal_failure_reported is not None:
                assert loop.time() < deadline, "renewal never recovered"
                await asyncio.sleep(0.02)
            recovered = dispatcher.runtime_owner_renewal_failure_count
            caplog.clear()
            await backend.close()
            await failures_reach(recovered + 2)
            assert reports("runtime-owner renewal failed").count(logging.ERROR) == 1
    finally:
        if not backend.is_connected:
            await backend.connect()
        await close_input_dispatcher(agent, dispatcher)
