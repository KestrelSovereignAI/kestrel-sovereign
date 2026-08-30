"""Cross-replica live-work authority for cooperative Stop (#3152)."""

import asyncio
from uuid import uuid4

import pytest

from kestrel_sovereign.agent.request_lifecycle import RequestLifecycleMixin
from kestrel_sovereign.stop import (
    DistributedInvocationRegistry,
    DistributedInvocationStore,
    StopDisposition,
    StopOutcome,
    StopReceiptStore,
    StopRequest,
    StopScope,
)


class _ReplicaAgent(RequestLifecycleMixin):
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self._current_request_id = None


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_distributed_protocol_has_sqlite_postgres_parity(db_backend):
    from kestrel_sovereign.storage.async_database import AsyncDatabase

    store = DistributedInvocationStore(AsyncDatabase(db_backend))
    await store.ensure_schema()
    suffix = uuid4().hex
    generation_id = f"generation-{suffix}"
    agent_id = f"did:test:agent-{suffix}"
    turn_id = f"turn-{suffix}"
    owner_id = f"owner-{suffix}"

    assert await store.register(
        generation_id=generation_id,
        agent_id=agent_id,
        turn_id=turn_id,
        owner_id=owner_id,
    )
    ticket = await store.mark_turn(agent_id, turn_id)
    assert ticket.generation_ids == (generation_id,)
    assert len(await store.remaining(ticket.generation_ids)) == 1
    await store.complete(generation_id, owner_id)
    assert await store.remaining(ticket.generation_ids) == ()
    assert (
        await store.register(
            generation_id=f"retry-{generation_id}",
            agent_id=agent_id,
            turn_id=turn_id,
            owner_id=owner_id,
        )
        is False
    )


async def _shared_registries(tmp_path):
    from kestrel_sovereign.storage.async_database import AsyncDatabase

    path = tmp_path / "distributed-stop.db"
    first_db = await AsyncDatabase.sqlite(str(path))
    second_db = await AsyncDatabase.sqlite(str(path))
    first_store = DistributedInvocationStore(first_db)
    second_store = DistributedInvocationStore(second_db)
    await first_store.ensure_schema()
    first = DistributedInvocationRegistry(first_store, poll_seconds=0.01)
    second = DistributedInvocationRegistry(second_store, poll_seconds=0.01)
    first.start()
    second.start()
    return first_db, second_db, first_store, first, second


async def _wait_until_registered(store, expected: int = 1) -> None:
    for _ in range(100):
        rows = await store._db.fetchall(
            "SELECT generation_id FROM stop_active_invocations"
        )
        if len(rows) == expected:
            return
        await asyncio.sleep(0.01)
    raise AssertionError("distributed invocation registration did not appear")


@pytest.mark.asyncio
async def test_stop_on_replica_b_cancels_invocation_owned_by_replica_a(tmp_path):
    first_db, second_db, store, replica_a, replica_b = await _shared_registries(
        tmp_path
    )
    agent = _ReplicaAgent("did:test:shared-agent")
    replica_a.attach(agent)
    entered = asyncio.Event()

    async def cognition():
        agent.register_active_request("turn-across-replicas")
        assert await agent.await_durable_request_admission("turn-across-replicas")
        agent.bind_request_operation("turn-across-replicas", asyncio.current_task())
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            agent._cleanup_cancelled_request("turn-across-replicas")

    operation = asyncio.create_task(cognition())
    try:
        await entered.wait()
        await _wait_until_registered(store)

        ticket = await replica_b.request_turn(
            "did:test:shared-agent", "turn-across-replicas"
        )
        disposition = await replica_b.wait_for_stop(ticket, timeout_seconds=1.0)

        assert disposition is StopDisposition.STOPPED
        with pytest.raises(asyncio.CancelledError):
            await operation
        assert await store.remaining(ticket.generation_ids) == ()
        durable = await first_db.fetchone(
            "SELECT turn_digest FROM stop_invocation_fences"
        )
        assert durable is not None
        assert durable[0] != "turn-across-replicas"
    finally:
        if not operation.done():
            operation.cancel()
            await asyncio.gather(operation, return_exceptions=True)
        await replica_a.close()
        await replica_b.close()
        await first_db.close()
        await second_db.close()


@pytest.mark.asyncio
async def test_exact_stop_fence_wins_before_remote_registration(tmp_path):
    first_db, second_db, store, replica_a, replica_b = await _shared_registries(
        tmp_path
    )
    agent = _ReplicaAgent("did:test:shared-agent")
    replica_a.attach(agent)
    try:
        ticket = await replica_b.request_turn(
            "did:test:shared-agent", "turn-not-registered-yet"
        )
        assert await replica_b.wait_for_stop(ticket) is (
            StopDisposition.ALREADY_COMPLETE
        )

        agent.register_active_request("turn-not-registered-yet")
        admitted = await agent.await_durable_request_admission(
            "turn-not-registered-yet"
        )

        assert admitted is False
        assert agent.is_request_cancelled("turn-not-registered-yet") is True
        agent._cleanup_cancelled_request("turn-not-registered-yet")
        assert await store._db.fetchone("SELECT 1 FROM stop_active_invocations") is None
    finally:
        await replica_a.close()
        await replica_b.close()
        await first_db.close()
        await second_db.close()


@pytest.mark.asyncio
async def test_unjoined_remote_owner_is_unreachable_not_already_complete(tmp_path):
    first_db, second_db, store, replica_a, replica_b = await _shared_registries(
        tmp_path
    )
    # Stop the relay while preserving its durable owner row, modeling an
    # unresponsive process whose database connection and lease have not yet
    # been retired.
    agent = _ReplicaAgent("did:test:shared-agent")
    replica_a.attach(agent)
    agent.register_active_request("wedged-turn")
    assert await agent.await_durable_request_admission("wedged-turn")
    await _wait_until_registered(store)
    replica_a._relay_task.cancel()
    await asyncio.gather(replica_a._relay_task, return_exceptions=True)
    replica_a._relay_task = None
    try:
        ticket = await replica_b.request_turn("did:test:shared-agent", "wedged-turn")
        assert (
            await replica_b.wait_for_stop(ticket, timeout_seconds=0.05)
            is StopDisposition.UNREACHABLE
        )
    finally:
        agent._cleanup_cancelled_request("wedged-turn")
        await replica_a.close()
        await replica_b.close()
        await first_db.close()
        await second_db.close()


@pytest.mark.asyncio
async def test_transient_completion_failure_is_retried_until_row_is_removed(
    tmp_path,
):
    first_db, second_db, store, replica_a, replica_b = await _shared_registries(
        tmp_path
    )
    agent = _ReplicaAgent("did:test:retry-completion-agent")
    replica_a.attach(agent)
    original_complete = store.complete
    attempts = 0

    async def flaky_complete(generation_id, owner_id):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("transient database outage")
        await original_complete(generation_id, owner_id)

    store.complete = flaky_complete
    try:
        assert await replica_a.register(agent, "retry-completion-turn", 1)
        generation_id = replica_a._by_local_generation[
            (id(agent), "retry-completion-turn", 1)
        ]

        replica_a.complete_soon(agent, "retry-completion-turn", 1)
        for _ in range(100):
            if generation_id not in replica_a._active:
                break
            await asyncio.sleep(0.01)

        assert attempts == 2
        assert generation_id not in replica_a._active
        assert await store.remaining((generation_id,)) == ()
    finally:
        await replica_a.close()
        await replica_b.close()
        await first_db.close()
        await second_db.close()


@pytest.mark.asyncio
async def test_acknowledged_receipt_fences_direct_non_http_invocation(tmp_path):
    first_db, second_db, _store, replica_a, replica_b = await _shared_registries(
        tmp_path
    )
    request = StopRequest(
        scope=StopScope.TURN,
        actor_id="did:test:operator",
        target="durably-stopped-turn",
        target_agent_id="did:test:shared-agent",
        correlation_id="prior-stop-operation",
    )
    await StopReceiptStore(first_db).persist(
        request,
        (
            StopOutcome(
                scope=StopScope.TURN,
                requested_target=request.target,
                resolved_target=request.target,
                agent_id="did:test:shared-agent",
                disposition=StopDisposition.ALREADY_COMPLETE,
                correlation_id=request.correlation_id,
            ),
        ),
    )
    agent = _ReplicaAgent("did:test:shared-agent")
    replica_a.attach(agent)
    try:
        agent.register_active_request("durably-stopped-turn")
        assert (
            await agent.await_durable_request_admission("durably-stopped-turn") is False
        )
        agent._cleanup_cancelled_request("durably-stopped-turn")
    finally:
        await replica_a.close()
        await replica_b.close()
        await first_db.close()
        await second_db.close()
