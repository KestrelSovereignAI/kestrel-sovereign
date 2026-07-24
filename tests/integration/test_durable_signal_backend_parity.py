"""SQLite/PostgreSQL parity for the durable signal delivery ledger."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pytest

from kestrel_sdk.signals import (
    RedactionPolicy,
    Signal,
    SignalMode,
    SourceRegistration,
    Status,
    Trust,
)
from kestrel_sovereign.signals import (
    DurableConsumerRegistration,
    DurableSignalStore,
    OrderedLockManager,
    SignalDispatcher,
    SignalLogStore,
    SourceRegistry,
)


def _signal(agent_id: str) -> Signal:
    return Signal(
        source="provider.message",
        kind="inbound",
        mode=SignalMode.ACTION,
        payload={"workflow": "wf-1", "message": "normalized"},
        target_agent=agent_id,
    )


class _DispatcherAgent:
    """Minimal live-dispatch agent for the PostgreSQL commit-boundary race."""

    def __init__(self, did: str, privacy_config) -> None:
        self.did = did
        self.privacy_config = privacy_config
        self._privacy_transition_lock = asyncio.Lock()
        self.tasks: list[asyncio.Task] = []

    def _get_privacy_transition_lock(self):
        return self._privacy_transition_lock

    def _track_background_task(self, coro, *, name: str):
        task = asyncio.create_task(coro, name=name)
        self.tasks.append(task)
        return task

    async def process_input(self, prompt: str):  # pragma: no cover - ACTION only
        return prompt


async def _ephemeral_dispatcher(db_backend, *, agent_id: str):
    """Build the actual dispatcher path with a payload-eliding projection."""
    from kestrel_sovereign.privacy import get_privacy_preset

    agent = _DispatcherAgent(agent_id, get_privacy_preset("ephemeral"))
    registry = SourceRegistry()

    async def handler(payload):
        return {"handled": payload}

    registry.register(
        SourceRegistration(
            name="provider.message",
            schema=lambda payload: payload,
            default_mode=SignalMode.ACTION,
            allowed_modes=frozenset({SignalMode.ACTION}),
            handler=handler,
            trust=Trust.TRUSTED,
            log_redaction=RedactionPolicy(summarize=lambda payload: "<redacted>"),
            retention_days=7,
        )
    )
    log_store = SignalLogStore(db_backend)
    await log_store.initialize()
    dispatcher = SignalDispatcher(
        agent=agent,
        registry=registry,
        lock_manager=OrderedLockManager(),
        store=log_store,
    )
    await dispatcher.initialize_durable_delivery()
    return agent, dispatcher


async def _finish_dispatcher(agent: _DispatcherAgent, dispatcher: SignalDispatcher):
    """Drain audit tasks before the shared backend fixture is torn down."""
    dispatcher.shutdown()
    if agent.tasks:
        await asyncio.gather(*agent.tasks, return_exceptions=True)


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
async def test_durable_retry_and_lease_expiry_transitions_work_on_both_backends(
    db_backend,
):
    """Timestamp-bearing failure transitions retain PostgreSQL type parity."""
    store = DurableSignalStore(db_backend)
    await store.initialize()
    agent_id = f"did:test:durable-retry:{uuid4()}"
    consumer_id = "workflow-wait"
    await store.register_consumer(
        DurableConsumerRegistration(
            consumer_id=consumer_id,
            source="provider.message",
            agent_id=agent_id,
            max_attempts=2,
            lease_seconds=1,
        )
    )
    await store.persist_signal(
        _signal(agent_id),
        agent_id=agent_id,
        source_event_id=f"retry-{uuid4()}",
        retention_days=7,
    )
    now = datetime.now(timezone.utc)
    first = await store.claim_delivery(
        agent_id=agent_id,
        consumer_id=consumer_id,
        executor_id="worker-a",
        now=now,
    )
    assert first is not None
    retried = await store.nack_delivery(
        agent_id=agent_id,
        consumer_id=consumer_id,
        delivery_id=first.delivery_id,
        lease_token=first.lease_token,
        error="transient dependency outage",
        retry_delay=timedelta(seconds=1),
        now=now,
    )
    assert retried is not None
    assert retried.status == "retry"

    second = await store.claim_delivery(
        agent_id=agent_id,
        consumer_id=consumer_id,
        executor_id="worker-b",
        now=now + timedelta(seconds=1),
    )
    assert second is not None
    assert second.attempts == 2
    assert await store.claim_delivery(
        agent_id=agent_id,
        consumer_id=consumer_id,
        executor_id="worker-c",
        now=now + timedelta(seconds=2),
    ) is None
    terminal = await store.get_delivery(
        agent_id=agent_id,
        consumer_id=consumer_id,
        delivery_id=second.delivery_id,
    )
    assert terminal is not None
    assert terminal.status == "failed"
    assert terminal.last_error == "lease expired before acknowledgement"


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_initial_volatile_reservation_blocks_a_peer_on_both_backends(db_backend):
    """The initial live owner is established before a marker event is visible."""
    peer_backend = await _independent_backend(db_backend)
    try:
        emitting_store = DurableSignalStore(db_backend)
        peer_store = DurableSignalStore(peer_backend)
        await emitting_store.initialize()
        await peer_store.initialize()
        agent_id = f"did:test:durable-initial-lease:{uuid4()}"
        consumer_id = "workflow-wait"
        await emitting_store.register_consumer(
            DurableConsumerRegistration(
                consumer_id=consumer_id,
                source="provider.message",
                agent_id=agent_id,
                correlation_selector="payload.workflow=wf-1",
                lease_seconds=10,
            )
        )
        secret = "initial-lease-customer@example.com"
        marker_event = _signal(agent_id)
        marker_event.payload = {"_privacy_gated": "none"}
        installed_before_commit = []
        persisted = await emitting_store.persist_signal(
            marker_event,
            agent_id=agent_id,
            source_event_id=f"initial-lease-{uuid4()}",
            retention_days=7,
            transient_selector_payload={"workflow": "wf-1", "message": secret},
            initial_lease_owner="emitting-dispatcher",
            before_commit=installed_before_commit.append,
        )
        assert installed_before_commit == [persisted]
        assert len(persisted.initial_leases) == 1
        reservation = persisted.initial_leases[0]

        # A separate store sees the committed row but not a claimable
        # delivery. It must wait for the owning dispatcher to transfer or for
        # lease expiry recovery, never receive the transient selector payload.
        assert await peer_store.claim_delivery(
            agent_id=agent_id,
            consumer_id=consumer_id,
            executor_id="peer-worker",
        ) is None
        owned = await emitting_store.claim_initial_delivery(
            agent_id=agent_id,
            consumer_id=consumer_id,
            delivery_id=reservation.delivery_id,
            initial_lease_owner="emitting-dispatcher",
            initial_lease_token=reservation.lease_token,
            executor_id="owner-worker",
        )
        assert owned is not None
        assert owned.event.payload == {"_privacy_gated": "none"}
        row = await db_backend.fetch_one(
            "SELECT payload FROM durable_signal_events WHERE event_id = ?",
            (marker_event.id,),
        )
        assert row is not None
        assert secret not in str(row[0])
    finally:
        await peer_backend.close()


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_same_dispatcher_initial_claim_waits_for_postgres_commit_boundary(
    db_backend,
):
    """A local claimant must not discard a sidecar before its row commits.

    PostgreSQL claims run in a separate transaction.  Pause the emitting
    transaction after the synchronous sidecar callback, make the same
    dispatcher perform its ordinary durable poll (which cannot see the row),
    then release commit.  The claim must transfer the now-visible initial
    lease and receive the raw live payload.
    """
    if db_backend.backend_type != "postgres":
        pytest.skip("PostgreSQL-specific separate-transaction visibility race")

    agent_id = f"did:test:durable-local-commit-race:{uuid4()}"
    consumer_id = "workflow-wait"
    agent, dispatcher = await _ephemeral_dispatcher(db_backend, agent_id=agent_id)
    original_transaction = db_backend.transaction
    original_claim_delivery = dispatcher._durable_store.claim_delivery
    commit_boundary = asyncio.Event()
    sidecar_installed = asyncio.Event()
    ordinary_claim_missed = asyncio.Event()
    release_commit = asyncio.Event()
    original_schedule = dispatcher._schedule_transient_durable_handoff_expiry

    @asynccontextmanager
    async def pause_after_before_commit():
        async with original_transaction():
            yield
            # The raw sidecar is installed, but this PostgreSQL transaction
            # has not committed its event/delivery rows yet.
            commit_boundary.set()
            await release_commit.wait()

    def note_sidecar_install(delivery_id, expires_at):
        original_schedule(delivery_id, expires_at)
        sidecar_installed.set()

    async def note_uncommitted_ordinary_claim(**kwargs):
        delivery = await original_claim_delivery(**kwargs)
        if delivery is None:
            ordinary_claim_missed.set()
        return delivery

    db_backend.transaction = pause_after_before_commit
    dispatcher._schedule_transient_durable_handoff_expiry = note_sidecar_install
    dispatcher._durable_store.claim_delivery = note_uncommitted_ordinary_claim
    try:
        await dispatcher.register_durable_consumer(
            DurableConsumerRegistration(
                consumer_id=consumer_id,
                source="provider.message",
                agent_id=agent_id,
                correlation_selector="payload.workflow=wf-1",
                lease_seconds=10,
            )
        )
        secret = "same-dispatcher-commit-boundary@example.com"
        event = _signal(agent_id)
        event.payload["message"] = secret
        dispatch_task = asyncio.create_task(
            dispatcher.dispatch_signal(
                event,
                source_event_id=f"same-dispatcher-commit-race:{uuid4()}",
            )
        )
        await asyncio.wait_for(sidecar_installed.wait(), timeout=5)
        await asyncio.wait_for(commit_boundary.wait(), timeout=5)

        claim_task = asyncio.create_task(
            dispatcher.claim_durable_delivery(
                consumer_id=consumer_id,
                executor_id="local-worker",
            )
        )
        await asyncio.wait_for(ordinary_claim_missed.wait(), timeout=5)
        # The caller has performed the separate-transaction poll but must wait
        # on the same local handoff lock that protects the commit boundary.
        assert not claim_task.done()

        release_commit.set()
        assert (await dispatch_task).status is Status.OK
        delivery = await claim_task
        assert delivery is not None
        assert delivery.event.payload == {"workflow": "wf-1", "message": secret}
        row = await db_backend.fetch_one(
            "SELECT payload FROM durable_signal_events WHERE event_id = ?",
            (event.id,),
        )
        assert row is not None
        assert secret not in str(row[0])
    finally:
        release_commit.set()
        db_backend.transaction = original_transaction
        dispatcher._schedule_transient_durable_handoff_expiry = original_schedule
        dispatcher._durable_store.claim_delivery = original_claim_delivery
        await _finish_dispatcher(agent, dispatcher)


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_initial_reservation_lease_starts_after_handoff_contention_on_both_backends(
    db_backend,
):
    """The initial owner retains a live lease after a blocked handoff lock."""
    peer_backend = await _independent_backend(db_backend)
    release_handoff = asyncio.Event()
    holder = None
    persistence_task = None
    try:
        emitting_store = DurableSignalStore(db_backend)
        blocking_store = DurableSignalStore(peer_backend)
        await emitting_store.initialize()
        await blocking_store.initialize()
        agent_id = f"did:test:durable-initial-contention:{uuid4()}"
        consumer_id = "workflow-wait"
        await emitting_store.register_consumer(
            DurableConsumerRegistration(
                consumer_id=consumer_id,
                source="provider.message",
                agent_id=agent_id,
                lease_seconds=1,
            )
        )
        handoff_acquired = asyncio.Event()

        async def hold_scope_handoff() -> None:
            async with peer_backend.transaction():
                await blocking_store._lock_scope_handoff(
                    agent_id=agent_id, source="provider.message"
                )
                handoff_acquired.set()
                await release_handoff.wait()

        holder = asyncio.create_task(hold_scope_handoff())
        await asyncio.wait_for(handoff_acquired.wait(), timeout=2)
        marker_event = _signal(agent_id)
        marker_event.payload = {"_privacy_gated": "none"}
        persistence_task = asyncio.create_task(
            emitting_store.persist_signal(
                marker_event,
                agent_id=agent_id,
                source_event_id=f"initial-contention-{uuid4()}",
                retention_days=7,
                initial_lease_owner="emitting-dispatcher",
            )
        )
        # The pre-fix method-entry timestamp is now older than the whole
        # initial lease while this transaction waits for the handoff lock.
        await asyncio.sleep(1.1)
        release_handoff.set()
        await holder
        persisted = await persistence_task
        assert len(persisted.initial_leases) == 1
        reservation = persisted.initial_leases[0]
        assert reservation.lease_expires_at > datetime.now(timezone.utc)
        assert await emitting_store.claim_initial_delivery(
            agent_id=agent_id,
            consumer_id=consumer_id,
            delivery_id=reservation.delivery_id,
            initial_lease_owner="emitting-dispatcher",
            initial_lease_token=reservation.lease_token,
            executor_id="owner-worker",
        ) is not None
    finally:
        release_handoff.set()
        if holder is not None:
            await asyncio.gather(holder, return_exceptions=True)
        if persistence_task is not None:
            await asyncio.gather(persistence_task, return_exceptions=True)
        await peer_backend.close()


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
