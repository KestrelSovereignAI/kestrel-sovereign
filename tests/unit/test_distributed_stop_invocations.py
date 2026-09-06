"""Cross-replica live-work authority for cooperative Stop (#3152)."""

import asyncio
from unittest.mock import MagicMock
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
        request_generation=1,
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
            request_generation=2,
        )
        is False
    )


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_public_turn_uuid_binding_has_sqlite_postgres_parity(db_backend):
    from kestrel_sovereign.storage.async_database import AsyncDatabase

    store = DistributedInvocationStore(AsyncDatabase(db_backend))
    await store.ensure_schema()
    suffix = uuid4().hex
    generation_id = f"public-generation-{suffix}"
    owner_id = f"public-owner-{suffix}"
    agent_id = f"did:test:public-agent-{suffix}"
    public_turn_id = f"public-turn-{suffix}"

    assert await store.register(
        generation_id=generation_id,
        agent_id=agent_id,
        turn_id=f"private-request-{suffix}",
        owner_id=owner_id,
        request_generation=1,
    )
    assert await store.bind_public_turn(
        generation_id=generation_id,
        owner_id=owner_id,
        agent_id=agent_id,
        turn_id=public_turn_id,
    )
    ticket = await store.mark_public_turn(agent_id, public_turn_id)

    assert ticket.generation_ids == (generation_id,)
    await store.complete(generation_id, owner_id)
    assert await store.remaining(ticket.generation_ids) == ()


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
        request_generation=1,
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
async def test_public_turn_on_replica_b_cancels_uuid_bound_on_replica_a(tmp_path):
    first_db, second_db, _store, replica_a, replica_b = await _shared_registries(
        tmp_path
    )
    agent = _ReplicaAgent("did:test:shared-public-agent")
    agent.cancel_current_request = MagicMock(return_value=True)
    replica_a.attach(agent)
    try:
        assert await replica_a.register(agent, "private-request", 1)
        assert await replica_a.bind_public_turn(
            agent,
            "public-turn",
            "private-request",
            1,
        )

        ticket = await replica_b.request_public_turn(
            agent.agent_id,
            "public-turn",
        )
        for _ in range(100):
            if agent.cancel_current_request.called:
                break
            await asyncio.sleep(0.01)

        assert len(ticket.generation_ids) == 1
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
async def test_public_turn_fence_wins_before_durable_binding(tmp_path):
    first_db, second_db, _store, replica_a, replica_b = await _shared_registries(
        tmp_path
    )
    agent = _ReplicaAgent("did:test:public-bind-race")
    replica_a.attach(agent)
    try:
        assert await replica_a.register(agent, "private-request", 1)
        ticket = await replica_b.request_public_turn(
            agent.agent_id,
            "public-turn-not-bound-yet",
        )

        assert ticket.generation_ids == ()
        assert not await replica_a.bind_public_turn(
            agent,
            "public-turn-not-bound-yet",
            "private-request",
            1,
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
            request_generation=1,
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
            request_generation=1,
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
async def test_lease_loss_after_insert_retries_provisional_generation_cleanup(
    tmp_path,
):
    """A published row remains locally owned until durable deletion succeeds."""

    from kestrel_sovereign.storage.async_database import AsyncDatabase

    db = await AsyncDatabase.sqlite(str(tmp_path / "provisional-cleanup.db"))
    store = DistributedInvocationStore(db)
    await store.ensure_schema()
    registry = DistributedInvocationRegistry(store, poll_seconds=0.01)
    agent = _ReplicaAgent("did:test:provisional-cleanup")
    original_register = store.register
    original_complete = store.complete
    attempts = 0

    async def lose_lease_after_insert(**kwargs):
        admitted = await original_register(**kwargs)
        registry._lease_lost = True
        return admitted

    async def flaky_complete(generation_id, owner_id):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("transient database outage")
        await original_complete(generation_id, owner_id)

    store.register = lose_lease_after_insert
    store.complete = flaky_complete
    try:
        assert await registry.register(agent, "provisional-turn", 1) is False
        for _ in range(100):
            rows = await db.fetchall(
                "SELECT generation_id FROM stop_active_invocations"
            )
            if not rows and not registry._active:
                break
            await asyncio.sleep(0.01)

        assert attempts == 2
        assert rows == []
        assert registry._active == {}
        assert registry._by_local_generation == {}
    finally:
        await registry.close()
        await db.close()


@pytest.mark.asyncio
async def test_public_turn_receipt_does_not_fence_same_named_request_id(tmp_path):
    """Public turn and private request addresses occupy separate namespaces."""

    from kestrel_sovereign.storage.async_database import AsyncDatabase

    db = await AsyncDatabase.sqlite(str(tmp_path / "address-namespaces.db"))
    try:
        store = DistributedInvocationStore(db)
        await store.ensure_schema()
        request = StopRequest(
            scope=StopScope.TURN,
            actor_id="did:test:operator",
            target="shared-address",
            target_agent_id="did:test:shared-agent",
            correlation_id="public-turn-stop",
            target_is_turn_id=True,
        )
        await StopReceiptStore(db).persist(
            request,
            (
                StopOutcome(
                    scope=StopScope.TURN,
                    requested_target=request.target,
                    resolved_target="private-request",
                    agent_id="did:test:shared-agent",
                    disposition=StopDisposition.ALREADY_COMPLETE,
                    correlation_id=request.correlation_id,
                ),
            ),
        )

        assert await store.register(
            generation_id="direct-request-generation",
            agent_id="did:test:shared-agent",
            turn_id="shared-address",
            owner_id="direct-request-owner",
            request_generation=1,
        )
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_exact_generation_stop_does_not_cancel_or_fence_reused_request(
    tmp_path,
):
    """A captured public-turn generation stays narrower than its request ID."""

    from kestrel_sovereign.storage.async_database import AsyncDatabase

    db = await AsyncDatabase.sqlite(str(tmp_path / "exact-generation.db"))
    try:
        store = DistributedInvocationStore(db)
        await store.ensure_schema()
        common = {
            "agent_id": "did:test:reused-request",
            "turn_id": "reused-request",
            "owner_id": "request-owner",
        }
        assert await store.register(
            generation_id="generation-one",
            request_generation=1,
            **common,
        )
        assert await store.register(
            generation_id="generation-two",
            request_generation=2,
            **common,
        )
        assert await store.bind_public_turn(
            generation_id="generation-one",
            owner_id=common["owner_id"],
            agent_id=common["agent_id"],
            turn_id="public-generation-one",
        )

        ticket = await store.mark_public_turn(
            common["agent_id"],
            "public-generation-one",
        )
        rows = await db.fetchall(
            "SELECT generation_id, stop_requested "
            "FROM stop_active_invocations ORDER BY generation_id"
        )

        assert ticket.generation_ids == ("generation-one",)
        assert rows == [
            ("generation-one", 1),
            ("generation-two", 0),
        ]
        await store.complete("generation-one", common["owner_id"])
        assert await store.register(
            generation_id="generation-three",
            request_generation=3,
            **common,
        )
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_public_turn_stop_selects_its_durable_uuid_across_replicas(tmp_path):
    """Replica-local integer generations cannot identify distributed work."""

    from kestrel_sovereign.storage.async_database import AsyncDatabase

    db = await AsyncDatabase.sqlite(str(tmp_path / "public-turn-uuid.db"))
    store = DistributedInvocationStore(db)
    await store.ensure_schema()
    agent_id = "did:test:public-turn-uuid"
    try:
        # Both replicas legitimately start at local generation 1 and receive
        # the same caller retry ID. Only the server-owned durable UUID can
        # distinguish the public turns they actually created.
        assert await store.register(
            generation_id="durable-generation-a",
            agent_id=agent_id,
            turn_id="shared-request-id",
            owner_id="replica-a",
            request_generation=1,
        )
        assert await store.register(
            generation_id="durable-generation-b",
            agent_id=agent_id,
            turn_id="shared-request-id",
            owner_id="replica-b",
            request_generation=1,
        )
        assert await store.bind_public_turn(
            generation_id="durable-generation-a",
            owner_id="replica-a",
            agent_id=agent_id,
            turn_id="public-turn-a",
        )
        assert await store.bind_public_turn(
            generation_id="durable-generation-b",
            owner_id="replica-b",
            agent_id=agent_id,
            turn_id="public-turn-b",
        )

        ticket = await store.mark_public_turn(agent_id, "public-turn-a")
        rows = await db.fetchall(
            "SELECT generation_id, stop_requested "
            "FROM stop_active_invocations ORDER BY generation_id"
        )

        assert ticket.generation_ids == ("durable-generation-a",)
        assert rows == [
            ("durable-generation-a", 1),
            ("durable-generation-b", 0),
        ]
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_distributed_store_preserves_whitespace_only_request_id(tmp_path):
    """Durable Stop accepts every opaque ID accepted by invocation entry."""

    from kestrel_sovereign.storage.async_database import AsyncDatabase

    db = await AsyncDatabase.sqlite(str(tmp_path / "opaque-request-id.db"))
    store = DistributedInvocationStore(db)
    await store.ensure_schema()
    try:
        assert await store.register(
            generation_id="opaque-request-generation",
            agent_id="did:test:opaque-request",
            turn_id=" ",
            owner_id="opaque-request-owner",
            request_generation=1,
        )
        ticket = await store.mark_turn("did:test:opaque-request", " ")
        assert ticket.generation_ids == ("opaque-request-generation",)
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_existing_distributed_schema_gains_public_turn_binding(tmp_path):
    """An additive migration upgrades databases created before the binding."""

    from kestrel_sovereign.storage.async_database import AsyncDatabase

    db = await AsyncDatabase.sqlite(str(tmp_path / "legacy-stop-schema.db"))
    try:
        await db.execute(
            "CREATE TABLE stop_active_invocations ("
            "generation_id TEXT NOT NULL PRIMARY KEY, "
            "agent_id TEXT NOT NULL, turn_digest TEXT NOT NULL, "
            "request_generation INTEGER NOT NULL, owner_id TEXT NOT NULL, "
            "stop_requested INTEGER NOT NULL DEFAULT 0, "
            "registered_at TEXT NOT NULL, heartbeat_at TEXT NOT NULL)"
        )
        await db.execute(
            "CREATE TABLE stop_unresolved_invocations ("
            "generation_id TEXT NOT NULL PRIMARY KEY, "
            "agent_id TEXT NOT NULL, turn_digest TEXT NOT NULL, "
            "request_generation INTEGER NOT NULL, owner_id TEXT NOT NULL, "
            "expired_at TEXT NOT NULL)"
        )

        store = DistributedInvocationStore(db)
        await store.ensure_schema()
        active_columns = {
            str(row[1])
            for row in await db.fetchall(
                "PRAGMA table_info(stop_active_invocations)"
            )
        }
        unresolved_columns = {
            str(row[1])
            for row in await db.fetchall(
                "PRAGMA table_info(stop_unresolved_invocations)"
            )
        }

        assert "public_turn_digest" in active_columns
        assert "public_turn_digest" in unresolved_columns
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_distributed_relay_cancels_only_ticketed_local_generation(tmp_path):
    """Relay delivery must retain the generation selected by mark_turn."""

    from kestrel_sovereign.storage.async_database import AsyncDatabase

    db = await AsyncDatabase.sqlite(str(tmp_path / "relay-generation.db"))
    store = DistributedInvocationStore(db)
    await store.ensure_schema()
    registry = DistributedInvocationRegistry(store, poll_seconds=0.01)
    agent = _ReplicaAgent("did:test:relay-generation")
    agent.cancel_current_request = MagicMock(return_value=True)
    try:
        assert await registry.register(agent, "reused-request", 1)
        assert await registry.register(agent, "reused-request", 2)
        assert await registry.bind_public_turn(
            agent,
            "public-generation-one",
            "reused-request",
            1,
        )
        await store.mark_public_turn(
            agent.agent_id,
            "public-generation-one",
        )
        registry.start()
        for _ in range(100):
            if agent.cancel_current_request.called:
                break
            await asyncio.sleep(0.01)

        agent.cancel_current_request.assert_called_once_with(
            request_id="reused-request",
            generation=1,
        )
    finally:
        await registry.close()
        await db.close()


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
