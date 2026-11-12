"""#4001: a settlement's native control never takes the owner heartbeat away.

The settlement pass ends the inbound A2A task whose submission wake ended
undelivered. When that write's commit outcome is unknown, the pass retains it
as the delivery's settlement control: readiness fails and owner retirement
refuses until a pass finds nothing owed. That control guards no lease, since
its row already stands. Before the fix it still put the owner heartbeat into
its cleanup branch, and on PostgreSQL that branch refreshes the owner only
while it leases core cognition. A live owner whose only control was a
settlement's therefore stopped heartbeating, and a peer reclaimed its live
leases, so the same work ran twice.

Uses only TEST_POSTGRES_URL and owns a unique schema; no shared rows change.
"""

from __future__ import annotations

import asyncio
import os
from datetime import timedelta

import pytest
from kestrel_sdk.native_a2a.types import TaskState

from kestrel_sovereign.a2a.scope import NativeRecipientScope
from kestrel_sovereign.execution_custody import ExecutionCommitOutcomeError
from kestrel_sovereign.signals import DurableConsumerRegistration
from kestrel_sovereign.signals.durable import DurableSignalStore
from kestrel_sovereign.storage.db.postgres import PostgresBackend
from tests.unit.test_a2a_native_input_fencing import (
    close_input_dispatcher,
    failed_submission_wake,
    fault_task_writes,
    heartbeat,
    input_dispatcher,
    snapshot,
)
from tests.unit.test_a2a_native_task_manager import request
from tests.utils.native_a2a import manager_for
from tests.utils.postgres_schema import disposable_postgres_schema, with_search_path

pytestmark = [pytest.mark.asyncio, pytest.mark.integration, pytest.mark.timeout(60)]

WORKFLOW_CONSUMER = "workflows:wait:4001"


@pytest.fixture
async def native_pg():
    url = os.environ.get("TEST_POSTGRES_URL")
    if not url:
        pytest.skip("TEST_POSTGRES_URL required")
    admin = PostgresBackend(url, min_pool_size=1, max_pool_size=2)
    await admin.connect()
    try:
        async with disposable_postgres_schema(admin, "settlement_owner") as schema:
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


async def test_a_settlement_control_keeps_the_owner_heartbeating(native_pg):
    """With a settlement's commit outcome unknown and no leased cognition
    row, the owner's ``heartbeat_at`` keeps advancing across heartbeats, and
    a peer's recovery of stale owners reclaims none of its live leases.
    Readiness and retirement still refuse while the control stands."""
    manager = native_pg
    task = await manager.create_task(request(), creator_agent_id="did:creator")
    agent, dispatcher, consumer = await input_dispatcher(manager)
    closed = False
    try:
        # A live lease this owner holds that is not core cognition: what a
        # peer would steal if this owner looked dead.
        await dispatcher.register_durable_consumer(
            DurableConsumerRegistration(
                consumer_id=WORKFLOW_CONSUMER,
                source="a2a.task_submitted",
                agent_id=agent.did,
                max_attempts=0,
                lease_seconds=3600,
            )
        )
        failed = await failed_submission_wake(manager, dispatcher, consumer, task)
        held = await dispatcher.claim_durable_delivery_for_event(
            consumer_id=WORKFLOW_CONSUMER,
            event_id=failed.event_id,
            executor_id=dispatcher._durable_delivery_owner,
        )
        assert held is not None and held.status == "leased"

        heartbeats = 3
        faults = [ExecutionCommitOutcomeError("unknown") for _ in range(heartbeats + 1)]
        fault_task_writes(manager, faults)

        await heartbeat(dispatcher)
        assert dispatcher.unresolved_cognition_control == (failed.delivery_id,)
        assert not dispatcher._retained_cognition_control_debt
        beats = [await _heartbeat_at(dispatcher)]
        for _ in range(heartbeats - 1):
            await asyncio.sleep(0.02)
            await heartbeat(dispatcher)
            beats.append(await _heartbeat_at(dispatcher))
            # Each pass retried the fenced write and met the unknown commit
            # again: the control stands, and readiness still fails.
            assert dispatcher.unresolved_cognition_control == (failed.delivery_id,)

        # A peer treats every owner not heard from since the control first
        # stood as dead. This owner was heard from.
        store = dispatcher._durable_store
        recovered = await store.recover_abandoned_leases(
            agent_id=agent.did,
            recovering_owner_id="dispatcher:peer-4001",
            stale_before=beats[0] + timedelta(microseconds=1),
        )
        assert recovered == 0, f"a peer reclaimed this live owner's lease; beats={beats}"
        assert beats == sorted(beats) and len(set(beats)) == len(beats), beats
        assert len(faults) == 1
        (lease,) = await dispatcher.list_durable_deliveries(consumer_id=WORKFLOW_CONSUMER)
        assert (lease.status, lease.lease_owner, lease.lease_token) == (
            "leased", dispatcher._durable_delivery_owner, held.lease_token,
        )
        assert (await snapshot(manager, task)).task.status.state == (
            TaskState.TASK_STATE_SUBMITTED
        )

        # Owner retirement refuses while the control stands...
        with pytest.raises(BaseException):
            await close_input_dispatcher(agent, dispatcher)
        assert faults == []
        assert dispatcher.unresolved_cognition_control == (failed.delivery_id,)
        assert dispatcher._durable_runtime_owner_registered

        # ...and completes once a retry of the fenced write returns.
        closed = True
        await close_input_dispatcher(agent, dispatcher)
        assert dispatcher.unresolved_cognition_control == ()
        assert (await snapshot(manager, task)).task.status.state == (
            TaskState.TASK_STATE_FAILED
        )
    finally:
        if not closed:
            await close_input_dispatcher(agent, dispatcher)
