"""SQLite/PostgreSQL parity for the durable signal delivery ledger."""

from __future__ import annotations

import asyncio
from uuid import uuid4

import pytest

from kestrel_sdk.signals import Signal, SignalMode
from kestrel_sovereign.signals import (
    DurableConsumerRegistration,
    DurableSignalStore,
)


def _signal(agent_id: str) -> Signal:
    return Signal(
        source="provider.message",
        kind="inbound",
        mode=SignalMode.ACTION,
        payload={"workflow": "wf-1", "message": "normalized"},
        target_agent=agent_id,
    )


async def _independent_backend(db_backend):
    """Connect a second real backend instance to the same durable ledger."""
    if db_backend.backend_type == "sqlite":
        from kestrel_sovereign.storage.db.sqlite import SQLiteBackend

        backend = SQLiteBackend(db_backend.db_path)
    else:
        from kestrel_sovereign.storage.db.postgres import PostgresBackend

        backend = PostgresBackend(db_backend._dsn)
    await backend.connect()
    return backend


def _is_scope_handoff_lock(query: str) -> bool:
    return (
        "pg_advisory_xact_lock" in query
        or query.strip() == "DELETE FROM durable_signal_consumers WHERE 0"
    )


async def _assert_one_pending_delivery(
    store: DurableSignalStore, *, agent_id: str, event_id: str
) -> None:
    deliveries = await store.list_deliveries(agent_id=agent_id)
    assert [delivery.event_id for delivery in deliveries] == [event_id]


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_durable_delivery_claim_is_scoped_and_single_owner_on_both_backends(db_backend):
    store = DurableSignalStore(db_backend)
    await store.initialize()
    # The hosted PostgreSQL fixture shares its database between test runs, so
    # keep this durable scope unique instead of colliding with retained ledger
    # rows from an earlier invocation.
    agent_id = f"did:test:durable-parity:{uuid4()}"
    await store.register_consumer(
        DurableConsumerRegistration(
            consumer_id="workflow-wait",
            source="provider.message",
            agent_id=agent_id,
            correlation_selector="payload.workflow=wf-1",
        )
    )
    event = _signal(agent_id)
    assert (await store.persist_signal(
        event, agent_id=agent_id, source_event_id="provider-event-1", retention_days=7
    )).created
    # The same explicit provider ID is the durable dedup key on both engines.
    assert not (await store.persist_signal(
        _signal(agent_id), agent_id=agent_id,
        source_event_id="provider-event-1", retention_days=7
    )).created

    first, second = await asyncio.gather(
        store.claim_delivery(
            agent_id=agent_id, consumer_id="workflow-wait", executor_id="worker-a"
        ),
        store.claim_delivery(
            agent_id=agent_id, consumer_id="workflow-wait", executor_id="worker-b"
        ),
    )
    claims = [delivery for delivery in (first, second) if delivery is not None]
    assert len(claims) == 1
    delivery = claims[0]
    assert delivery.event.payload["message"] == "normalized"
    assert await store.ack_delivery(
        agent_id=agent_id,
        consumer_id="workflow-wait",
        delivery_id=delivery.delivery_id,
        lease_token=delivery.lease_token,
    )


@pytest.mark.asyncio
@pytest.mark.dual_backend
@pytest.mark.parametrize(
    "first",
    ("registration", "persistence"),
    ids=("registration-first", "persistence-first"),
)
async def test_durable_registration_persistence_handoff_is_atomic_across_instances(
    db_backend, first
):
    """Exercise both handoff orders through the public durable-store API.

    The gates pause a real transaction after it has acquired its handoff
    primitive.  They prove the competing backend instance cannot take its
    first stale read until the owner commits, which is the SQLite failure mode
    that a same-instance or private-helper test cannot expose.
    """
    peer_backend = await _independent_backend(db_backend)
    try:
        registration_backend, persistence_backend = (
            (db_backend, peer_backend)
            if first == "registration"
            else (peer_backend, db_backend)
        )
        registration_store = DurableSignalStore(registration_backend)
        persistence_store = DurableSignalStore(persistence_backend)
        await registration_store.initialize()
        await persistence_store.initialize()

        agent_id = f"did:test:durable-handoff:{uuid4()}"
        registration = DurableConsumerRegistration(
            consumer_id="workflow-wait",
            source="provider.message",
            agent_id=agent_id,
            correlation_selector="payload.workflow=wf-1",
        )
        event = _signal(agent_id)

        if first == "registration":
            registration_read = asyncio.Event()
            allow_registration_write = asyncio.Event()
            original_fetch_one = registration_backend.fetch_one

            async def pause_after_registration_read(query, params=()):
                row = await original_fetch_one(query, params)
                if (
                    f"FROM {DurableSignalStore.CONSUMERS}" in query
                    and "consumer_id" in query
                ):
                    registration_read.set()
                    await allow_registration_write.wait()
                return row

            registration_backend.fetch_one = pause_after_registration_read
            registration_task = asyncio.create_task(
                registration_store.register_consumer(registration)
            )
            await asyncio.wait_for(registration_read.wait(), timeout=5)

            persistence_started = asyncio.Event()
            original_handoff = (
                persistence_backend.fetch_val
                if persistence_backend.backend_type == "postgres"
                else persistence_backend.execute
            )

            async def note_persistence_handoff(query, params=()):
                if _is_scope_handoff_lock(query):
                    persistence_started.set()
                return await original_handoff(query, params)

            if persistence_backend.backend_type == "postgres":
                persistence_backend.fetch_val = note_persistence_handoff
            else:
                persistence_backend.execute = note_persistence_handoff
            persistence_task = asyncio.create_task(
                persistence_store.persist_signal(
                    event,
                    agent_id=agent_id,
                    source_event_id=f"handoff-{uuid4()}",
                    retention_days=7,
                )
            )
            await asyncio.wait_for(persistence_started.wait(), timeout=5)
            assert not persistence_task.done()
            allow_registration_write.set()
            _, persisted = await asyncio.wait_for(
                asyncio.gather(registration_task, persistence_task), timeout=5
            )
        else:
            persistence_lookup = asyncio.Event()
            allow_persistence_lookup = asyncio.Event()
            original_fetch_all = persistence_backend.fetch_all

            async def pause_before_persistence_consumer_lookup(query, params=()):
                if (
                    f"FROM {DurableSignalStore.CONSUMERS}" in query
                    and "max_attempts" in query
                ):
                    persistence_lookup.set()
                    await allow_persistence_lookup.wait()
                return await original_fetch_all(query, params)

            persistence_backend.fetch_all = pause_before_persistence_consumer_lookup
            persistence_task = asyncio.create_task(
                persistence_store.persist_signal(
                    event,
                    agent_id=agent_id,
                    source_event_id=f"handoff-{uuid4()}",
                    retention_days=7,
                )
            )
            await asyncio.wait_for(persistence_lookup.wait(), timeout=5)

            registration_started = asyncio.Event()
            original_handoff = (
                registration_backend.fetch_val
                if registration_backend.backend_type == "postgres"
                else registration_backend.execute
            )

            async def note_registration_handoff(query, params=()):
                if _is_scope_handoff_lock(query):
                    registration_started.set()
                return await original_handoff(query, params)

            if registration_backend.backend_type == "postgres":
                registration_backend.fetch_val = note_registration_handoff
            else:
                registration_backend.execute = note_registration_handoff
            registration_task = asyncio.create_task(
                registration_store.register_consumer(registration)
            )
            await asyncio.wait_for(registration_started.wait(), timeout=5)
            assert not registration_task.done()
            allow_persistence_lookup.set()
            persisted, _ = await asyncio.wait_for(
                asyncio.gather(persistence_task, registration_task), timeout=5
            )

        assert persisted.created
        await _assert_one_pending_delivery(
            registration_store, agent_id=agent_id, event_id=event.id
        )
    finally:
        await peer_backend.close()
