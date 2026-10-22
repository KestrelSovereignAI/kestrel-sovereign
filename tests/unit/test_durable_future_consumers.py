"""Persisted future-only consumers use atomic fan-out without history scans."""

from dataclasses import replace
from datetime import timedelta

import pytest
import pytest_asyncio
from kestrel_sdk.signals import Signal, SignalMode

from kestrel_sovereign.signals.durable import (
    ACKNOWLEDGED,
    FAILED,
    DurableConsumerRegistration,
    DurableSignalStore,
)
from kestrel_sovereign.storage.db.interface import DatabaseError, TransactionError


@pytest_asyncio.fixture
async def ledger(native_a2a_backend_factory):
    backend = native_a2a_backend_factory()
    await backend.connect()
    store = DurableSignalStore(backend)
    try:
        await store.initialize()
        yield store
    finally:
        await backend.close()


def _registration(*, replay_retained=True):
    return DurableConsumerRegistration(
        consumer_id="owned-consumer",
        source="future.test",
        agent_id="did:future-consumer",
        replay_retained=replay_retained,
    )


async def _persist(
    store, event_id, *, source="future.test", agent_id="did:future-consumer"
):
    return await store.persist_signal(
        Signal(
            source=source,
            kind="work",
            mode=SignalMode.ACTION,
            payload={"event": event_id},
            target_agent=agent_id,
        ),
        agent_id=agent_id,
        source_event_id=event_id,
        retention_days=30,
    )


async def _claim(store):
    return await store.claim_delivery(
        agent_id="did:future-consumer",
        consumer_id="owned-consumer",
        executor_id="original-executor",
    )


async def test_future_only_contract_survives_restart_without_history_scans(
    ledger, monkeypatch
):
    await _persist(ledger, "old")
    actual = ledger.backend.fetch_all
    scans = []

    async def fetch(query, params=()):
        rows = await actual(query, params)
        if "FROM durable_signal_events" in query and "retention_until >=" in query:
            scans.extend(rows)
        return rows

    monkeypatch.setattr(ledger.backend, "fetch_all", fetch)
    registration = _registration(replay_retained=False)
    await ledger.register_consumer(registration)
    assert await _claim(ledger) is None
    await _persist(ledger, "new")
    restarted = DurableSignalStore(ledger.backend)
    await restarted.initialize()
    await restarted.register_consumer(registration)
    delivery = await _claim(restarted)
    assert delivery is not None and delivery.event.source_event_id == "new"
    assert await restarted.ack_delivery(
        agent_id=delivery.agent_id,
        consumer_id=delivery.consumer_id,
        delivery_id=delivery.delivery_id,
        lease_token=delivery.lease_token,
    )
    assert await _claim(restarted) is None
    assert not scans, "future-only consumer decoded pre-registration history"


async def test_future_delivery_uses_commit_handoff_instead_of_a_clock_watermark(
    ledger, monkeypatch
):
    await _persist(ledger, "old")
    await ledger.register_consumer(_registration(replay_retained=False))
    regressed_clock = ledger.now_utc() - timedelta(days=3)
    monkeypatch.setattr(ledger, "now_utc", lambda: regressed_clock)
    await _persist(ledger, "late-arrival")
    delivery = await _claim(ledger)
    assert delivery is not None and delivery.event.source_event_id == "late-arrival"


async def test_default_consumer_still_replays_its_retained_history(ledger):
    await _persist(ledger, "retained")
    await ledger.register_consumer(
        DurableConsumerRegistration(
            consumer_id="owned-consumer", source="future.test", agent_id="did:future-consumer"
        )
    )
    delivery = await _claim(ledger)
    assert delivery is not None and delivery.event.source_event_id == "retained"


async def test_replay_mode_is_an_immutable_consumer_contract(ledger):
    registration = _registration(replay_retained=False)
    await ledger.register_consumer(registration)
    with pytest.raises((ValueError, TransactionError), match="registration conflicts"):
        await ledger.register_consumer(replace(registration, replay_retained=True))
    assert await ledger.backend.fetch_val(
        "SELECT replay_retained FROM durable_signal_consumers"
    ) == ledger.to_bool_param(False)


async def test_existing_consumer_schema_evolves_with_replay_enabled(ledger):
    await ledger.register_consumer(_registration())
    await ledger.backend.execute(
        "ALTER TABLE durable_signal_consumers DROP COLUMN replay_retained"
    )
    await DurableSignalStore(ledger.backend).initialize()
    assert await ledger.backend.fetch_val(
        "SELECT replay_retained FROM durable_signal_consumers"
    ) == ledger.to_bool_param(True)
    await ledger.register_consumer(_registration())


async def test_invalid_stored_replay_mode_cannot_disable_history_silently(ledger):
    await ledger.register_consumer(_registration())
    query = "UPDATE durable_signal_consumers SET replay_retained = ?"
    if ledger.is_postgres:
        with pytest.raises(DatabaseError):
            await ledger.backend.execute(query, (2,))
    else:
        await ledger.backend.execute(query, (2,))
        with pytest.raises(RuntimeError, match="replay policy is invalid"):
            await _claim(ledger)


@pytest.mark.parametrize("value", [None, 0, 1, "false"])
async def test_replay_policy_requires_an_explicit_boolean(ledger, value):
    with pytest.raises(ValueError, match="replay_retained must be a bool"):
        await ledger.register_consumer(_registration(replay_retained=value))
    assert await ledger.backend.fetch_val(
        "SELECT COUNT(*) FROM durable_signal_consumers"
    ) == 0


async def test_due_inventory_is_bounded_scoped_and_never_changes_original_leases(ledger):
    registration = _registration(replay_retained=False)
    for index in range(3):
        await ledger.register_consumer(
            replace(registration, consumer_id=f"consumer-{index}")
        )
    await _persist(ledger, "materialized")
    query = {"agent_id": registration.agent_id, "source": registration.source, "limit": 2}
    assert await ledger.due_consumer_ids(**query) == ["consumer-0", "consumer-1"]
    assert await ledger.due_consumer_ids(**{**query, "agent_id": "did:other"}) == []
    assert await ledger.due_consumer_ids(**{**query, "source": "other.test"}) == []
    claimed = []
    for index in range(2):
        claimed.append(
            await ledger.claim_delivery(
                agent_id=registration.agent_id,
                consumer_id=f"consumer-{index}",
                executor_id="original-executor",
            )
        )
    assert await ledger.due_consumer_ids(**query) == ["consumer-2"]
    assert await ledger.ack_delivery(
        agent_id=registration.agent_id,
        consumer_id=claimed[1].consumer_id,
        delivery_id=claimed[1].delivery_id,
        lease_token=claimed[1].lease_token,
    )
    await ledger.deactivate_consumer(
        agent_id=registration.agent_id, consumer_id="consumer-2"
    )
    assert await ledger.due_consumer_ids(**query) == []
    lease = claimed[0]
    before = await ledger.backend.fetch_one(
        "SELECT status, lease_token, lease_expires_at "
        "FROM durable_signal_deliveries WHERE delivery_id = ?",
        (lease.delivery_id,),
    )
    expired = lease.lease_expires_at + timedelta(seconds=1)
    assert await ledger.due_consumer_ids(**query, now=expired) == ["consumer-0"]
    assert await ledger.backend.fetch_one(
        "SELECT status, lease_token, lease_expires_at "
        "FROM durable_signal_deliveries WHERE delivery_id = ?",
        (lease.delivery_id,),
    ) == before, "a read-only due hint changed the original lease"


async def test_retry_inventory_respects_its_original_next_attempt_boundary(ledger):
    await ledger.register_consumer(_registration(replay_retained=False))
    await _persist(ledger, "retry")
    delivery = await _claim(ledger)
    released = await ledger.nack_delivery(
        agent_id=delivery.agent_id,
        consumer_id=delivery.consumer_id,
        delivery_id=delivery.delivery_id,
        lease_token=delivery.lease_token,
        error="retained test refusal",
        retry_delay=timedelta(seconds=60),
    )
    query = {"agent_id": delivery.agent_id, "source": "future.test", "limit": 1}
    assert await ledger.due_consumer_ids(**query) == []
    assert await ledger.due_consumer_ids(
        **query, now=released.next_attempt_at
    ) == [delivery.consumer_id]


@pytest.mark.parametrize("limit", [True, 0, -1, 1.5])
async def test_due_inventory_requires_an_explicit_positive_limit(ledger, limit):
    with pytest.raises(ValueError, match="positive integer"):
        await ledger.due_consumer_ids(
            agent_id="did:future-consumer", source="future.test", limit=limit
        )


async def test_retention_returns_exact_scoped_ids_with_a_finite_source_batch(ledger):
    # Older foreign rows would win the bounded sweep if either scope filter
    # were lost; inserting them after the owned batch would hide that defect.
    other_source = await _persist(ledger, "other-source", source="other.test")
    other_agent = await _persist(ledger, "other-agent", agent_id="did:other")
    events = [await _persist(ledger, f"retained-{index}") for index in range(3)]
    future = ledger.now_utc() + timedelta(days=31)
    deleted = await ledger.purge_expired(
        agent_id="did:future-consumer", source="future.test", limit=2, now=future
    )
    expected = {event.event_id for event in events}
    assert isinstance(deleted, tuple) and len(deleted) == 2
    assert set(deleted) <= expected
    assert await ledger.backend.fetch_val(
        "SELECT COUNT(*) FROM durable_signal_events"
    ) == 3
    rest = await ledger.purge_expired(agent_id="did:future-consumer", now=future)
    assert set(rest) == (expected - set(deleted)) | {other_source.event_id}
    assert await ledger.backend.fetch_all(
        "SELECT event_id FROM durable_signal_events"
    ) == [(other_agent.event_id,)]


async def test_retention_ids_and_the_callers_reference_write_roll_back_together(ledger):
    event = await _persist(ledger, "original-reference")
    await ledger.backend.execute(
        "CREATE TABLE native_retention_references (event_id TEXT PRIMARY KEY)"
    )
    future = ledger.now_utc() + timedelta(days=31)
    with pytest.raises(TransactionError):
        async with ledger.backend.transaction():
            deleted = await ledger.purge_expired(
                agent_id="did:future-consumer", source="future.test", limit=1, now=future
            )
            assert deleted == (event.event_id,)
            await ledger.backend.execute(
                "INSERT INTO native_retention_references VALUES (?)", (deleted[0],)
            )
            raise ValueError("original enclosing reference transaction refused")
    assert await ledger.backend.fetch_all(
        "SELECT event_id FROM durable_signal_events"
    ) == [(event.event_id,)]
    assert await ledger.backend.fetch_all("SELECT * FROM native_retention_references") == []


@pytest.mark.parametrize("limit", [True, 0, -1, 1.5])
async def test_retention_refuses_an_invalid_batch_before_deleting(ledger, limit):
    event = await _persist(ledger, "retained")
    with pytest.raises(ValueError, match="positive integer"):
        await ledger.purge_expired(agent_id="did:future-consumer", limit=limit)
    assert await ledger.backend.fetch_all(
        "SELECT event_id FROM durable_signal_events"
    ) == [(event.event_id,)]


async def test_idle_work_and_retention_probes_have_narrow_delivery_indexes(
    ledger, monkeypatch
):
    if ledger.is_postgres:
        definitions = await ledger.backend.fetch_all(
            "SELECT indexdef FROM pg_indexes WHERE schemaname = current_schema() "
            "AND tablename = 'durable_signal_deliveries'"
        )
        assert any("(event_id, status)" in row[0] for row in definitions)
        assert any("_live_due" in row[0] and "WHERE" in row[0] for row in definitions)
        return
    actual = ledger.backend.fetch_all
    queries = []

    async def fetch(query, params=()):
        result = await actual(query, params)
        queries.append((query, params))
        return result

    monkeypatch.setattr(ledger.backend, "fetch_all", fetch)
    await ledger.due_consumer_ids(
        agent_id="did:future-consumer", source="future.test", limit=1
    )
    query, params = queries[-1]
    plan = await actual("EXPLAIN QUERY PLAN " + query, params)
    assert any("SEARCH d" in row[-1] and "_live_due" in row[-1] for row in plan), plan
    plan = await actual(
        "EXPLAIN QUERY PLAN SELECT 1 FROM durable_signal_deliveries d "
        "WHERE d.event_id = ? AND d.status NOT IN ('acknowledged', 'failed', 'terminal_ackable')",
        ("retained-event",),
    )
    assert any("SEARCH d" in row[-1] and "_event" in row[-1] for row in plan), plan


async def test_completion_requires_the_original_ack_and_finishes_the_consumer_atomically(
    ledger,
):
    registration = _registration(replay_retained=False)
    await ledger.register_consumer(registration)
    for index in range(2):
        await _persist(ledger, f"owed-{index}")
    delivery = await _claim(ledger)
    parameters = {
        "agent_id": delivery.agent_id,
        "consumer_id": delivery.consumer_id,
        "delivery_id": delivery.delivery_id,
        "lease_token": delivery.lease_token,
        "complete_consumer": True,
    }
    assert not await ledger.ack_delivery(**{**parameters, "lease_token": "foreign"})
    assert await ledger.has_active_consumer(delivery.agent_id, delivery.consumer_id)
    with pytest.raises(TransactionError):
        async with ledger.backend.transaction():
            assert await ledger.ack_delivery(**parameters)
            assert not await ledger.has_active_consumer(
                delivery.agent_id, delivery.consumer_id
            )
            raise ValueError("original completion transaction refused")
    assert await ledger.has_active_consumer(delivery.agent_id, delivery.consumer_id)
    assert await ledger.ack_delivery(**parameters)
    assert not await ledger.has_active_consumer(delivery.agent_id, delivery.consumer_id)
    statuses = await ledger.backend.fetch_all(
        "SELECT status FROM durable_signal_deliveries ORDER BY created_at, delivery_id"
    )
    assert sorted(row[0] for row in statuses) == sorted([ACKNOWLEDGED, FAILED])
    await _persist(ledger, "after-completion")
    assert await ledger.backend.fetch_val(
        "SELECT COUNT(*) FROM durable_signal_deliveries"
    ) == 2


@pytest.mark.parametrize("value", [None, 0, 1, "true"])
async def test_consumer_completion_requires_an_explicit_boolean(ledger, value):
    with pytest.raises(ValueError, match="explicit bool"):
        await ledger.ack_delivery(
            agent_id="did:future-consumer",
            consumer_id="owned-consumer",
            delivery_id="no-delivery",
            lease_token="no-token",
            complete_consumer=value,
        )
