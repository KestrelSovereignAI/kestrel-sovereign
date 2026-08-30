"""Cross-replica live-work authority for cooperative Stop (#3152)."""

import asyncio
from unittest.mock import MagicMock, call
from uuid import uuid4

import pytest

from kestrel_sovereign.agent.request_lifecycle import (
    RequestCompletionDisposition,
    RequestLifecycleMixin,
)
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
    exact_ticket = await store.mark_generation(generation_id, agent_id)
    assert exact_ticket.generation_ids == (generation_id,)
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


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_owner_lease_expiry_has_sqlite_postgres_parity(db_backend):
    from kestrel_sovereign.storage.async_database import AsyncDatabase

    store = DistributedInvocationStore(AsyncDatabase(db_backend))
    await store.ensure_schema()
    suffix = uuid4().hex
    generation_id = f"expired-{suffix}"
    owner_id = f"owner-{suffix}"
    assert await store.register(
        generation_id=generation_id,
        agent_id=f"did:test:agent-{suffix}",
        turn_id=f"turn-{suffix}",
        owner_id=owner_id,
    )
    stale = "2000-01-01T00:00:00.000+00:00"
    await store._db.execute(
        "UPDATE stop_active_invocations SET heartbeat_at = ? "
        "WHERE generation_id = ?",
        (stale, generation_id),
    )

    poll = await store.poll_owner(owner_id, lease_seconds=0.03)
    reaped = await store.reap_expired(
        (generation_id,), lease_seconds=0.03
    )

    assert poll.live_generation_ids == ()
    assert reaped == (generation_id,)
    assert len(await store.remaining((generation_id,))) == 1
    await store.complete(generation_id, owner_id)
    assert await store.remaining((generation_id,)) == ()


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
async def test_public_turn_alias_routes_stop_to_remote_request_generation(tmp_path):
    first_db, second_db, store, replica_a, replica_b = await _shared_registries(
        tmp_path
    )
    agent = _ReplicaAgent("did:test:public-turn-agent")
    replica_a.attach(agent)
    agent.cancel_current_request = MagicMock(return_value=True)
    try:
        assert await replica_a.register(agent, "private-request", 1)
        generation_id = replica_a._by_local_generation[
            (id(agent), "private-request", 1)
        ]
        assert await replica_a.bind_public_turn(
            agent,
            "private-request",
            1,
            "public-turn",
        )

        ticket = await replica_b.request_public_turn(
            agent.agent_id,
            "public-turn",
        )
        assert ticket.generation_ids == (generation_id,)
        for _ in range(100):
            if agent.cancel_current_request.called:
                break
            await asyncio.sleep(0.01)

        agent.cancel_current_request.assert_called_once_with(
            request_id="private-request",
            generation=1,
        )
    finally:
        await replica_a.close()
        await replica_b.close()
        await first_db.close()
        await second_db.close()


@pytest.mark.asyncio
async def test_public_turn_stop_wins_race_before_alias_binding(tmp_path):
    first_db, second_db, _store, replica_a, replica_b = await _shared_registries(
        tmp_path
    )
    agent = _ReplicaAgent("did:test:public-turn-race")
    try:
        assert await replica_a.register(agent, "private-race-request", 1)
        ticket = await replica_b.request_public_turn(
            agent.agent_id,
            "public-race-turn",
        )
        assert ticket.generation_ids == ()

        assert not await replica_a.bind_public_turn(
            agent,
            "private-race-request",
            1,
            "public-race-turn",
        )
    finally:
        await replica_a.close()
        await replica_b.close()
        await first_db.close()
        await second_db.close()


@pytest.mark.asyncio
async def test_registration_finishing_during_close_is_retired_and_refused(tmp_path):
    from kestrel_sovereign.storage.async_database import AsyncDatabase

    db = await AsyncDatabase.sqlite(str(tmp_path / "registration-close.db"))
    store = DistributedInvocationStore(db)
    await store.ensure_schema()
    registry = DistributedInvocationRegistry(store, poll_seconds=0.01)
    agent = _ReplicaAgent("did:test:closing-registration")
    entered = asyncio.Event()
    release = asyncio.Event()
    original_register = store.register

    async def delayed_register(**kwargs):
        admitted = await original_register(**kwargs)
        entered.set()
        await release.wait()
        return admitted

    store.register = delayed_register
    registration = asyncio.create_task(registry.register(agent, "closing-turn", 1))
    try:
        await entered.wait()
        closing = asyncio.create_task(registry.close())
        await asyncio.sleep(0)
        assert registry._closing is True
        release.set()

        assert await registration is False
        await closing
        assert await db.fetchone(
            "SELECT 1 FROM stop_active_invocations LIMIT 1"
        ) is None
    finally:
        release.set()
        if not registration.done():
            registration.cancel()
            await asyncio.gather(registration, return_exceptions=True)
        if not registry._closing:
            await registry.close()
        await db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("pruned", [False, True])
async def test_abandoned_cleanup_preserves_indeterminate_durable_generation(
    tmp_path,
    pruned,
):
    first_db, second_db, store, replica_a, replica_b = await _shared_registries(
        tmp_path
    )
    agent = _ReplicaAgent("did:test:abandoned-agent")
    replica_a.attach(agent)
    turn_id = f"abandoned-{'pruned' if pruned else 'active'}"
    try:
        generation = agent.register_active_request(turn_id)
        assert await agent.await_durable_request_admission(turn_id)
        generation_id = replica_a._by_local_generation[
            (id(agent), turn_id, generation)
        ]
        if pruned:
            agent._active_request_started_at[turn_id] -= 1000
            assert agent.prune_stale_active_requests(900) == [turn_id]

        agent._cleanup_cancelled_request(
            turn_id,
            disposition=RequestCompletionDisposition.ABANDONED,
        )
        await asyncio.sleep(0.05)

        ticket = await replica_b.request_turn(agent.agent_id, turn_id)
        assert ticket.generation_ids == (generation_id,)
        assert (
            await replica_b.wait_for_stop(ticket, timeout_seconds=0.05)
            is StopDisposition.UNREACHABLE
        )
    finally:
        await replica_a.close()
        await replica_b.close()
        await first_db.close()
        await second_db.close()


@pytest.mark.asyncio
async def test_exact_generation_stop_marks_no_request_id_fence(tmp_path):
    first_db, second_db, store, replica_a, replica_b = await _shared_registries(
        tmp_path
    )
    agent = _ReplicaAgent("did:test:exact-generation-agent")
    try:
        assert await replica_a.register(agent, "reused-turn", 1)
        assert await replica_a.register(agent, "reused-turn", 2)
        first_generation = replica_a._by_local_generation[
            (id(agent), "reused-turn", 1)
        ]
        second_generation = replica_a._by_local_generation[
            (id(agent), "reused-turn", 2)
        ]

        ticket = await replica_a.request_generation(agent, "reused-turn", 1)

        assert ticket.generation_ids == (first_generation,)
        assert await store.remaining((first_generation,))
        assert await store.remaining((second_generation,))
        assert await first_db.fetchone(
            "SELECT 1 FROM stop_invocation_fences"
        ) is None
    finally:
        await replica_a.close()
        await replica_b.close()
        await first_db.close()
        await second_db.close()


@pytest.mark.asyncio
async def test_relay_cancels_each_selected_local_generation_exactly(tmp_path):
    first_db, second_db, _store, replica_a, replica_b = await _shared_registries(
        tmp_path
    )
    agent = _ReplicaAgent("did:test:relay-generation-agent")
    agent.cancel_current_request = MagicMock(return_value=True)
    try:
        assert await replica_a.register(agent, "reused-turn", 1)
        assert await replica_a.register(agent, "reused-turn", 2)

        await replica_b.request_turn(agent.agent_id, "reused-turn")
        for _ in range(100):
            if agent.cancel_current_request.call_count >= 2:
                break
            await asyncio.sleep(0.01)

        assert call(request_id="reused-turn", generation=1) in (
            agent.cancel_current_request.call_args_list
        )
        assert call(request_id="reused-turn", generation=2) in (
            agent.cancel_current_request.call_args_list
        )
        assert call(request_id="reused-turn") not in (
            agent.cancel_current_request.call_args_list
        )
    finally:
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
async def test_expired_crashed_owner_stays_unreachable_until_owner_cleanup(tmp_path):
    from kestrel_sovereign.storage.async_database import AsyncDatabase

    path = tmp_path / "expired-owner.db"
    first_db = await AsyncDatabase.sqlite(str(path))
    second_db = await AsyncDatabase.sqlite(str(path))
    store = DistributedInvocationStore(first_db)
    waiter_store = DistributedInvocationStore(second_db)
    await store.ensure_schema()
    generation_id = "crashed-generation"
    try:
        assert await store.register(
            generation_id=generation_id,
            agent_id="did:test:crashed-agent",
            turn_id="crashed-turn",
            owner_id="crashed-owner",
        )
        await first_db.execute(
            "UPDATE stop_active_invocations SET heartbeat_at = ? "
            "WHERE generation_id = ?",
            ("2000-01-01T00:00:00.000+00:00", generation_id),
        )
        waiter = DistributedInvocationRegistry(
            waiter_store,
            poll_seconds=0.01,
            owner_lease_seconds=0.03,
        )
        ticket = await waiter.request_turn(
            "did:test:crashed-agent", "crashed-turn"
        )

        assert (
            await waiter.wait_for_stop(ticket, timeout_seconds=0.2)
            is StopDisposition.UNREACHABLE
        )
        assert len(await store.remaining((generation_id,))) == 1

        repeated = await waiter.request_turn(
            "did:test:crashed-agent", "crashed-turn"
        )
        assert repeated.generation_ids == (generation_id,)
        repeated_agent = await waiter.request_agent("did:test:crashed-agent")
        assert repeated_agent.generation_ids == (generation_id,)

        await store.complete(generation_id, "wrong-owner")
        assert len(await store.remaining((generation_id,))) == 1
        await store.complete(generation_id, "crashed-owner")
        assert await store.remaining((generation_id,)) == ()
        assert (
            await waiter.wait_for_stop(ticket, timeout_seconds=0.2)
            is StopDisposition.STOPPED
        )
    finally:
        await first_db.close()
        await second_db.close()


@pytest.mark.asyncio
async def test_expired_owner_cannot_revive_its_heartbeat(tmp_path):
    from kestrel_sovereign.storage.async_database import AsyncDatabase

    db = await AsyncDatabase.sqlite(str(tmp_path / "non-revivable-owner.db"))
    store = DistributedInvocationStore(db)
    await store.ensure_schema()
    try:
        assert await store.register(
            generation_id="expired-generation",
            agent_id="did:test:expired-agent",
            turn_id="expired-turn",
            owner_id="expired-owner",
        )
        stale = "2000-01-01T00:00:00.000+00:00"
        await db.execute(
            "UPDATE stop_active_invocations SET heartbeat_at = ?",
            (stale,),
        )

        poll = await store.poll_owner(
            "expired-owner", lease_seconds=0.03
        )
        row = await db.fetchone(
            "SELECT heartbeat_at FROM stop_active_invocations "
            "WHERE generation_id = ?",
            ("expired-generation",),
        )

        assert poll.live_generation_ids == ()
        assert poll.stop_generation_ids == ()
        assert row == (stale,)
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_owner_that_cannot_renew_self_fences_and_refuses_new_work(tmp_path):
    first_db, second_db, store, replica_a, replica_b = await _shared_registries(
        tmp_path
    )
    replica_a._owner_lease_seconds = 0.04
    agent = _ReplicaAgent("did:test:self-fenced-agent")
    replica_a.attach(agent)
    entered = asyncio.Event()

    async def cognition():
        agent.register_active_request("partitioned-turn")
        assert await agent.await_durable_request_admission("partitioned-turn")
        agent.bind_request_operation("partitioned-turn", asyncio.current_task())
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            agent._cleanup_cancelled_request("partitioned-turn")

    operation = asyncio.create_task(cognition())
    original_poll = store.poll_owner

    async def unavailable_poll(*args, **kwargs):
        raise RuntimeError("database partition")

    try:
        await entered.wait()
        store.poll_owner = unavailable_poll
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(operation, timeout=0.5)
        assert replica_a._lease_lost is True
        assert await replica_a.register(agent, "later-turn", 2) is False
    finally:
        store.poll_owner = original_poll
        if not operation.done():
            operation.cancel()
            await asyncio.gather(operation, return_exceptions=True)
        await replica_a.close()
        await replica_b.close()
        await first_db.close()
        await second_db.close()


@pytest.mark.asyncio
async def test_idle_registry_starts_a_fresh_owner_lease_for_later_work(tmp_path):
    from kestrel_sovereign.storage.async_database import AsyncDatabase

    db = await AsyncDatabase.sqlite(str(tmp_path / "idle-owner-lease.db"))
    store = DistributedInvocationStore(db)
    await store.ensure_schema()
    registry = DistributedInvocationRegistry(
        store,
        poll_seconds=0.01,
        owner_lease_seconds=0.03,
    )
    agent = _ReplicaAgent("did:test:idle-agent")
    try:
        assert await registry.register(agent, "first-turn", 1)
        registry.complete_soon(agent, "first-turn", 1)
        for _ in range(100):
            if not registry._active:
                break
            await asyncio.sleep(0.01)
        assert registry._active == {}
        assert registry._last_heartbeat_monotonic is None

        await asyncio.sleep(0.04)

        assert await registry.register(agent, "later-turn", 2)
        assert registry._lease_lost is False
    finally:
        await registry.close()
        await db.close()


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
