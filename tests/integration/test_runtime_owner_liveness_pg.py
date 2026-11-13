"""#4001: on PostgreSQL an unretired owner's renewal never depends on its leases.

The PostgreSQL renewal used to refresh an owner with retained cleanup debt only
while that owner still leased core cognition. When a cognition delivery's
terminal write committed and its acknowledgement was lost, the row was no
longer leased but its debt stood: the renewal matched nothing, so the owner
stopped heartbeating, and a peer reclaimed a live lease the owner still held
for another consumer. The same work then ran twice.

Renewal is now one fixed write that refreshes any existing, unstopped owner,
whatever it leases. It checks out from a one-connection pool of its own,
reserved when the owner registers: ordinary work holding every connection of
the operational and advisory pools cannot hold it back, and closing the
backend joins a renewal in flight. Uses only TEST_POSTGRES_URL and owns a
unique schema; no shared rows change.
"""

from __future__ import annotations

import asyncio
import os
from datetime import timedelta

import pytest

from kestrel_sovereign.a2a.scope import NativeRecipientScope
from kestrel_sovereign.execution_custody import (
    ExecutionCommitOutcomeError,
    execution_commit_outcome,
)
from kestrel_sovereign.signals.durable import DurableSignalStore
from kestrel_sovereign.storage.db.postgres import PostgresBackend
from tests.unit.test_a2a_native_input_fencing import close_input_dispatcher, input_dispatcher
from tests.unit.test_a2a_native_task_manager import request
from tests.unit.test_runtime_owner_renewal import (
    STALE_AFTER,
    assert_retired_owner_stays_retired,
    assert_unretired_owner_stays_live,
    held_workflow_lease,
)
from tests.utils.native_a2a import manager_for
from tests.utils.postgres_schema import disposable_postgres_schema, with_search_path

pytestmark = [pytest.mark.asyncio, pytest.mark.integration, pytest.mark.timeout(60)]


@pytest.fixture
async def native_pg_scope():
    """A native task manager on its own schema, and that schema's URL."""
    url = os.environ.get("TEST_POSTGRES_URL")
    if not url:
        pytest.skip("TEST_POSTGRES_URL required")
    admin = PostgresBackend(url, min_pool_size=1, max_pool_size=2)
    await admin.connect()
    try:
        async with disposable_postgres_schema(admin, "owner_liveness") as schema:
            schema_url = with_search_path(url, schema)
            backend = PostgresBackend(schema_url, min_pool_size=1, max_pool_size=2)
            await backend.connect()
            manager = manager_for(
                backend, NativeRecipientScope("did:kestrel:recipient", "tenant")
            )
            try:
                await manager.initialize()
                yield manager, schema_url
            finally:
                await manager.close()
    finally:
        await admin.close()


@pytest.fixture
async def native_pg(native_pg_scope):
    return native_pg_scope[0]


@pytest.fixture
def native_pg_url(native_pg, native_pg_scope):
    """The URL of ``native_pg``'s schema.

    It requests ``native_pg`` so that every case reaching PostgreSQL through
    this module is a ``native_pg`` consumer: that fixture name is what
    selects a case into the required native PostgreSQL CI lane
    (tests/shared/postgres_requirement.py). A case requesting only
    ``native_pg_scope`` is not selected there, and the generic integration
    lane has no TEST_POSTGRES_URL, so it skips in both.
    """
    return native_pg_scope[1]


async def _heartbeat_at(dispatcher):
    store = dispatcher._durable_store
    value = await store.backend.fetch_val(
        f"SELECT heartbeat_at FROM {DurableSignalStore.RUNTIME_OWNERS} "
        "WHERE agent_id = ? AND owner_id = ?",
        (dispatcher._agent.did, dispatcher._durable_delivery_owner),
    )
    return store.from_timestamp_field(value)


async def test_retained_debt_without_a_cognition_lease_keeps_the_owner_live(native_pg):
    """With row debt retained for a cognition delivery whose terminal write
    committed, and no leased cognition row, the owner's ``heartbeat_at`` keeps
    advancing across heartbeats, and a peer's recovery of stale owners
    reclaims none of its live leases. The debt still stands: renewing admits
    no cognition and settles nothing."""
    manager = native_pg
    task = await manager.create_task(request(), creator_agent_id="did:creator")
    agent, dispatcher, consumer = await input_dispatcher(manager)
    closed = False
    try:
        # Only explicit heartbeats in this test, so each is observed.
        await dispatcher._stop_runtime_owner_heartbeat()
        original, held = await held_workflow_lease(manager, dispatcher, consumer, task)
        claimed = await dispatcher.claim_durable_delivery_for_event(
            consumer_id=consumer, event_id=original.event_id,
            executor_id=dispatcher._durable_delivery_owner,
        )
        assert claimed is not None and claimed.status == "leased"

        backend = dispatcher._durable_store.backend
        fail = backend.fail_cognition_delivery

        async def committed_then_lost(**kwargs):
            assert await fail(**kwargs)
            raise ExecutionCommitOutcomeError("unknown")

        backend.fail_cognition_delivery = committed_then_lost
        try:
            with pytest.raises(BaseException) as lost:
                await dispatcher._terminalize_failed_cognition(
                    claimed, ExecutionCommitOutcomeError("unknown")
                )
        finally:
            backend.fail_cognition_delivery = fail
        assert execution_commit_outcome(lost.value) == "unknown"
        assert set(dispatcher._retained_cognition_control_debt) == {claimed.delivery_id}
        (row,) = await dispatcher.list_durable_deliveries(consumer_id=consumer)
        assert row.status == "failed", "the owner must lease no core cognition"

        beats = [await _heartbeat_at(dispatcher)]
        for _ in range(3):
            await asyncio.sleep(0.02)
            await dispatcher._heartbeat_runtime_owner()
            beats.append(await _heartbeat_at(dispatcher))
        assert beats == sorted(beats) and len(set(beats)) == len(beats), beats

        # A peer treats every owner not heard from since the debt stood as
        # dead. This owner was heard from.
        store = dispatcher._durable_store
        recovered = await store.recover_abandoned_leases(
            agent_id=agent.did,
            recovering_owner_id="dispatcher:peer-4001",
            stale_before=beats[0] + timedelta(microseconds=1),
        )
        assert recovered == 0, f"a peer reclaimed this live owner's lease; beats={beats}"
        (lease,) = await dispatcher.list_durable_deliveries(consumer_id=held.consumer_id)
        assert (lease.status, lease.lease_owner, lease.lease_token) == (
            "leased", dispatcher._durable_delivery_owner, held.lease_token,
        )
        assert set(dispatcher._retained_cognition_control_debt) == {claimed.delivery_id}

        # A committed CAS cannot be settled again: the debt stands for an
        # operator. Retire the owner past it.
        dispatcher._retained_cognition_control_debt.clear()
        closed = True
        await close_input_dispatcher(agent, dispatcher)
        assert not dispatcher._durable_runtime_owner_registered
    finally:
        dispatcher._retained_cognition_control_debt.clear()
        if not closed:
            await close_input_dispatcher(agent, dispatcher)


async def _hold_every_connection(pool):
    """Check out every connection ``pool`` can open, as stalled work would."""
    held = [await pool.acquire() for _ in range(pool.get_max_size())]
    with pytest.raises(asyncio.TimeoutError):
        await pool.acquire(timeout=0.2)
    return held


async def test_ordinary_work_holding_every_pool_connection_does_not_stop_renewal(
    native_pg, native_pg_url, monkeypatch
):
    """Self-review of #4040: ordinary work holds every connection of the
    operational pool and of the advisory pool. Renewal checks out from its
    own one-connection pool, reserved when the owner registered, so the real
    heartbeat timer keeps renewing the owner past the stale threshold and a
    peer's recovery, run from a backend of its own, reclaims nothing the
    owner holds. Renewal on either shared pool would wait for a connection
    until its bound and never write."""
    from kestrel_sovereign.signals import SignalDispatcher

    monkeypatch.setitem(
        SignalDispatcher.__init__.__kwdefaults__, "runtime_owner_stale_after", STALE_AFTER
    )
    manager, schema_url = native_pg, native_pg_url
    task = await manager.create_task(request(), creator_agent_id="did:creator")
    agent, dispatcher, consumer = await input_dispatcher(manager)
    backend = dispatcher._durable_store.backend
    peer_backend = PostgresBackend(schema_url, min_pool_size=1, max_pool_size=1)
    await peer_backend.connect()
    peer = DurableSignalStore(peer_backend)
    held_connections = []
    closed = False
    try:
        _, held = await held_workflow_lease(manager, dispatcher, consumer, task)
        renewal_pool = backend._runtime_owner_renewal_pool
        assert renewal_pool is not None, "registration reserved no renewal connection"
        assert (renewal_pool.get_max_size(), renewal_pool.get_size()) == (1, 1)
        advisory_pool = await backend._ensure_advisory_pool()
        assert len({id(backend._pool), id(advisory_pool), id(renewal_pool)}) == 3
        for pool in (backend._pool, advisory_pool):
            held_connections.append((pool, await _hold_every_connection(pool)))

        await assert_unretired_owner_stays_live(dispatcher, held, store=peer)

        for pool, connections in held_connections:
            for connection in connections:
                await pool.release(connection)
        held_connections.clear()
        closed = True
        await close_input_dispatcher(agent, dispatcher)
        assert not dispatcher._durable_runtime_owner_registered
        await assert_retired_owner_stays_retired(dispatcher)
    finally:
        for pool, connections in held_connections:
            for connection in connections:
                await pool.release(connection)
        if not closed:
            await close_input_dispatcher(agent, dispatcher)
        await peer_backend.close()


async def test_closing_the_backend_joins_a_renewal_in_flight(native_pg_url, monkeypatch):
    """Self-review of #4040: closing the backend while a renewal is under way
    waits for that renewal, which commits, and only then closes renewal's
    pool. Closing it under the renewal would end the renewal, and only
    retirement may. A renewal after close gets no pool back."""
    from kestrel_sovereign.storage.db.interface import ConnectionError

    schema_url = native_pg_url
    backend = PostgresBackend(schema_url, min_pool_size=1, max_pool_size=1)
    await backend.connect()
    store = DurableSignalStore(backend)
    owner = {"agent_id": "did:kestrel:close-4001", "owner_id": "dispatcher:close-4001"}
    renewing = closing = None
    release = asyncio.Event()
    try:
        await store.initialize()
        await store.register_runtime_owner(**owner)
        entered = asyncio.Event()
        renewal_pool = backend._runtime_owner_renewal_pool_locked

        async def paused_renewal_pool():
            entered.set()
            await release.wait()
            return await renewal_pool()

        monkeypatch.setattr(backend, "_runtime_owner_renewal_pool_locked", paused_renewal_pool)
        renewing = asyncio.create_task(store.retain_unretired_runtime_owner(**owner))
        await asyncio.wait_for(entered.wait(), 5)
        closing = asyncio.create_task(backend.close())
        await asyncio.sleep(0.2)
        assert not closing.done(), "close did not wait for the renewal in flight"
        release.set()
        assert await asyncio.wait_for(renewing, 5) is True
        await asyncio.wait_for(closing, 5)
        assert backend._runtime_owner_renewal_pool is None
        monkeypatch.undo()
        with pytest.raises(ConnectionError):
            await store.retain_unretired_runtime_owner(**owner)
        assert backend._runtime_owner_renewal_pool is None
    finally:
        release.set()
        for pending in (renewing, closing):
            if pending is not None and not pending.done():
                await asyncio.wait({pending}, timeout=5)
        if closing is None or not closing.done() or closing.exception() is not None:
            await backend.close()


async def test_cancelling_close_while_it_joins_a_renewal_still_closes_every_pool(
    native_pg_url, monkeypatch
):
    """Self-review of #4040: close joins a renewal in flight by taking
    renewal's lock, and that wait was cancellable. A close cancelled during
    it raised before any pool closer ran, leaving renewal's pool, the
    advisory pool and the operational pool open and attached. Close now
    keeps the cancellation, lets the renewal commit, closes every pool, then
    raises it."""
    schema_url = native_pg_url
    backend = PostgresBackend(schema_url, min_pool_size=1, max_pool_size=1)
    await backend.connect()
    store = DurableSignalStore(backend)
    owner = {"agent_id": "did:kestrel:cancel-4040", "owner_id": "dispatcher:cancel-4040"}
    renewing = closing = None
    release = asyncio.Event()
    try:
        await store.initialize()
        await store.register_runtime_owner(**owner)
        await backend._ensure_advisory_pool()
        attributes = ("_runtime_owner_renewal_pool", "_advisory_pool", "_pool")
        pools = {attribute: getattr(backend, attribute) for attribute in attributes}
        assert all(pool is not None for pool in pools.values())
        entered = asyncio.Event()
        renewal_pool = backend._runtime_owner_renewal_pool_locked

        async def paused_renewal_pool():
            entered.set()
            await release.wait()
            return await renewal_pool()

        monkeypatch.setattr(backend, "_runtime_owner_renewal_pool_locked", paused_renewal_pool)
        renewing = asyncio.create_task(store.retain_unretired_runtime_owner(**owner))
        await asyncio.wait_for(entered.wait(), 5)
        closing = asyncio.create_task(backend.close())
        await asyncio.sleep(0.2)
        assert not closing.done(), "close did not wait for the renewal in flight"
        closing.cancel()
        await asyncio.sleep(0.2)
        assert not closing.done(), "a cancelled close stopped joining the renewal"
        assert not any(pool.is_closing() for pool in pools.values())

        release.set()
        assert await asyncio.wait_for(renewing, 5) is True
        done, _ = await asyncio.wait({closing}, timeout=10)
        assert closing in done, "close never finished once the renewal had"
        assert closing.cancelled(), "close did not deliver the caller's cancellation"
        for attribute, pool in pools.items():
            assert pool.is_closing(), f"close left {attribute} open"
            assert getattr(backend, attribute) is None, f"close left {attribute} attached"
    finally:
        release.set()
        for pending in (renewing, closing):
            if pending is not None and not pending.done():
                await asyncio.wait({pending}, timeout=5)
        if any(getattr(backend, attribute, None) is not None for attribute in (
            "_runtime_owner_renewal_pool", "_advisory_pool", "_pool",
        )):
            await backend.close()
