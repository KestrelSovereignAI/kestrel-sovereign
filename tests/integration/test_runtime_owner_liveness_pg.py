"""#4001: on PostgreSQL an unretired owner's renewal never depends on its leases.

The PostgreSQL renewal used to refresh an owner with retained cleanup debt only
while that owner still leased core cognition. When a cognition delivery's
terminal write committed and its acknowledgement was lost, the row was no
longer leased but its debt stood: the renewal matched nothing, so the owner
stopped heartbeating, and a peer reclaimed a live lease the owner still held
for another consumer. The same work then ran twice.

Renewal is now one fixed write that refreshes any existing, unstopped owner,
whatever it leases. Uses only TEST_POSTGRES_URL and owns a unique schema; no
shared rows change.
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
from tests.unit.test_runtime_owner_renewal import held_workflow_lease
from tests.utils.native_a2a import manager_for
from tests.utils.postgres_schema import disposable_postgres_schema, with_search_path

pytestmark = [pytest.mark.asyncio, pytest.mark.integration, pytest.mark.timeout(60)]


@pytest.fixture
async def native_pg():
    url = os.environ.get("TEST_POSTGRES_URL")
    if not url:
        pytest.skip("TEST_POSTGRES_URL required")
    admin = PostgresBackend(url, min_pool_size=1, max_pool_size=2)
    await admin.connect()
    try:
        async with disposable_postgres_schema(admin, "owner_liveness") as schema:
            backend = PostgresBackend(
                with_search_path(url, schema), min_pool_size=1, max_pool_size=2
            )
            await backend.connect()
            manager = manager_for(
                backend, NativeRecipientScope("did:kestrel:recipient", "tenant")
            )
            try:
                await manager.initialize()
                yield manager
            finally:
                await manager.close()
    finally:
        await admin.close()


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
