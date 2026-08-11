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
from kestrel_sovereign.features.channels.route_ownership import ChannelRouteOwnershipStore


def _signal(agent_id: str) -> Signal:
    return Signal(
        source="provider.message",
        kind="inbound",
        mode=SignalMode.ACTION,
        payload={"workflow": "wf-1", "message": "normalized"},
        target_agent=agent_id,
    )


def _dispatcher_owner_id() -> str:
    """Return the same managed-owner shape a production dispatcher emits."""

    return f"dispatcher:{uuid4().hex}"


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
    await dispatcher.shutdown_durable_delivery()
    if agent.tasks:
        await asyncio.gather(*agent.tasks, return_exceptions=True)


async def _cancel_and_drain(*tasks: asyncio.Task | None) -> None:
    """Cancel test choreography tasks before releasing their shared backend."""
    live_tasks = [task for task in tasks if task is not None]
    for task in live_tasks:
        if not task.done():
            task.cancel()
    if live_tasks:
        await asyncio.wait_for(
            asyncio.gather(*live_tasks, return_exceptions=True), timeout=5
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
async def test_schema_bootstrap_is_safe_under_independent_backend_contention(db_backend):
    """Fresh/additive durable and route bootstrap serializes across processes."""

    peer_backend = await _independent_backend(db_backend)
    try:
        first = DurableSignalStore(db_backend)
        second = DurableSignalStore(peer_backend)
        first_routes = ChannelRouteOwnershipStore(db_backend)
        second_routes = ChannelRouteOwnershipStore(peer_backend)
        await asyncio.wait_for(
            asyncio.gather(
                first.initialize(), second.initialize(),
                first_routes.initialize(), second_routes.initialize(),
            ),
            timeout=10,
        )
        assert await db_backend.fetch_val(
            "SELECT COUNT(*) FROM durable_signal_event_integrity"
        ) == 0
        claim = await first_routes.claim(
            channel_type="telegram",
            canonical_route_identity=f"telegram-bot:bootstrap-{uuid4().hex}",
            agent_id="did:test:bootstrap",
        )
        assert claim is not None
    finally:
        await peer_backend.close()


@pytest.mark.asyncio
@pytest.mark.dual_backend
@pytest.mark.parametrize("exact_event_claim", (False, True), ids=("ordinary", "exact"))
async def test_implicit_claim_clock_follows_contended_scope_handoff_on_both_backends(
    db_backend, monkeypatch, exact_event_claim
):
    """A delayed claim takes its implicit lease clock after the real DB lock."""

    peer_backend = await _independent_backend(db_backend)
    store = DurableSignalStore(db_backend)
    peer_store = DurableSignalStore(peer_backend)
    await store.initialize()
    await peer_store.initialize()
    agent_id = f"did:test:durable-claim-clock:{uuid4()}"
    consumer = DurableConsumerRegistration(
        consumer_id="clock-worker",
        source="provider.message",
        agent_id=agent_id,
        lease_seconds=1,
    )
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    clock = {"now": base}
    monkeypatch.setattr(store, "now_utc", lambda: clock["now"])
    await store.register_consumer(consumer)
    persisted = await store.persist_signal(
        _signal(agent_id),
        agent_id=agent_id,
        source_event_id=f"claim-clock:{uuid4()}",
        retention_days=7,
    )
    lock_acquired = asyncio.Event()
    entered_handoff = asyncio.Event()
    release_lock = asyncio.Event()
    original_handoff = store._lock_scope_handoff

    async def observe_handoff(**kwargs):
        entered_handoff.set()
        await original_handoff(**kwargs)

    monkeypatch.setattr(store, "_lock_scope_handoff", observe_handoff)

    async def hold_peer_scope() -> None:
        async with peer_backend.transaction():
            await peer_store._lock_scope_handoff(
                agent_id=agent_id, source=consumer.source
            )
            lock_acquired.set()
            await release_lock.wait()

    blocker = asyncio.create_task(hold_peer_scope())
    try:
        await asyncio.wait_for(lock_acquired.wait(), timeout=5)
        if exact_event_claim:
            claim = asyncio.create_task(
                store.claim_delivery_for_event(
                    agent_id=agent_id,
                    consumer_id=consumer.consumer_id,
                    event_id=persisted.event_id,
                    executor_id="worker",
                )
            )
        else:
            claim = asyncio.create_task(
                store.claim_delivery(
                    agent_id=agent_id,
                    consumer_id=consumer.consumer_id,
                    executor_id="worker",
                )
            )
        await asyncio.wait_for(entered_handoff.wait(), timeout=5)
        clock["now"] = base + timedelta(seconds=2)
        release_lock.set()
        claimed = await asyncio.wait_for(claim, timeout=5)
        assert claimed is not None
        assert claimed.lease_expires_at == base + timedelta(seconds=3)
    finally:
        release_lock.set()
        await _cancel_and_drain(blocker)
        await peer_backend.close()


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_registration_backfill_clock_follows_contended_scope_handoff_on_both_backends(
    db_backend, monkeypatch
):
    """Backfill evaluates retention after the real database serialization point."""

    peer_backend = await _independent_backend(db_backend)
    store = DurableSignalStore(db_backend)
    peer_store = DurableSignalStore(peer_backend)
    await store.initialize()
    await peer_store.initialize()
    agent_id = f"did:test:durable-registration-clock:{uuid4()}"
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    clock = {"now": base}
    monkeypatch.setattr(store, "now_utc", lambda: clock["now"])
    persisted = await store.persist_signal(
        _signal(agent_id),
        agent_id=agent_id,
        source_event_id=f"registration-clock:{uuid4()}",
        retention_days=7,
    )
    await db_backend.execute(
        "UPDATE durable_signal_events SET retention_until = ? WHERE event_id = ?",
        (base + timedelta(seconds=1), persisted.event_id),
    )
    consumer = DurableConsumerRegistration(
        consumer_id="late-clock-worker",
        source="provider.message",
        agent_id=agent_id,
    )
    lock_acquired = asyncio.Event()
    entered_handoff = asyncio.Event()
    release_lock = asyncio.Event()
    original_handoff = store._lock_scope_handoff

    async def observe_handoff(**kwargs):
        entered_handoff.set()
        await original_handoff(**kwargs)

    monkeypatch.setattr(store, "_lock_scope_handoff", observe_handoff)

    async def hold_peer_scope() -> None:
        async with peer_backend.transaction():
            await peer_store._lock_scope_handoff(
                agent_id=agent_id, source=consumer.source
            )
            lock_acquired.set()
            await release_lock.wait()

    blocker = asyncio.create_task(hold_peer_scope())
    try:
        await asyncio.wait_for(lock_acquired.wait(), timeout=5)
        registration = asyncio.create_task(store.register_consumer(consumer))
        await asyncio.wait_for(entered_handoff.wait(), timeout=5)
        clock["now"] = base + timedelta(seconds=2)
        release_lock.set()
        await asyncio.wait_for(registration, timeout=5)
        assert await store.list_deliveries(agent_id=agent_id) == []
    finally:
        release_lock.set()
        await _cancel_and_drain(blocker)
        await peer_backend.close()


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
        assert len(persisted.initial_reservations) == 1
        reservation = persisted.initial_reservations[0]

        # A separate store sees the committed row but not a claimable
        # delivery. It must wait for the owning dispatcher to transfer or for
        # lease expiry recovery, never receive the transient selector payload.
        assert await peer_store.claim_delivery(
            agent_id=agent_id,
            consumer_id=consumer_id,
            executor_id="peer-worker",
        ) is None
        activated = await emitting_store.activate_initial_delivery(
            agent_id=agent_id,
            consumer_id=consumer_id,
            delivery_id=reservation.delivery_id,
            initial_lease_owner="emitting-dispatcher",
            initial_lease_token=reservation.reservation_token,
        )
        assert activated is not None
        owned = await emitting_store.claim_initial_delivery(
            agent_id=agent_id,
            consumer_id=consumer_id,
            delivery_id=reservation.delivery_id,
            initial_lease_owner="emitting-dispatcher",
            initial_lease_token=reservation.reservation_token,
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
async def test_unactivated_reservation_survives_a_long_paused_commit_and_owner_activation_wins(
    db_backend,
):
    """No delivery lease exists until the post-commit owner activation."""
    peer_backend = await _independent_backend(db_backend)
    release_commit = asyncio.Event()
    persistence_task = None
    original_transaction = db_backend.transaction
    try:
        emitting_store = DurableSignalStore(db_backend)
        peer_store = DurableSignalStore(peer_backend)
        await emitting_store.initialize()
        await peer_store.initialize()
        agent_id = f"did:test:durable-paused-commit:{uuid4()}"
        consumer_id = "workflow-wait"
        owner_id = _dispatcher_owner_id()
        await emitting_store.register_consumer(
            DurableConsumerRegistration(
                consumer_id=consumer_id,
                source="provider.message",
                agent_id=agent_id,
                lease_seconds=1,
            )
        )
        await emitting_store.register_runtime_owner(
            agent_id=agent_id, owner_id=owner_id
        )
        sidecar_installed = asyncio.Event()
        commit_paused = asyncio.Event()

        @asynccontextmanager
        async def pause_after_precommit_callback():
            async with original_transaction():
                yield
                commit_paused.set()
                await release_commit.wait()

        db_backend.transaction = pause_after_precommit_callback
        marker_event = _signal(agent_id)
        marker_event.payload = {"_privacy_gated": "none"}
        persistence_task = asyncio.create_task(
            emitting_store.persist_signal(
                marker_event,
                agent_id=agent_id,
                source_event_id=f"paused-commit-{uuid4()}",
                retention_days=7,
                transient_selector_payload={
                    "workflow": "wf-1",
                    "message": "commit-paused-customer@example.com",
                },
                initial_lease_owner=owner_id,
                before_commit=lambda _persisted: sidecar_installed.set(),
            )
        )
        await asyncio.wait_for(sidecar_installed.wait(), timeout=2)
        await asyncio.wait_for(commit_paused.wait(), timeout=2)
        # The event transaction is still invisible. Sleeping beyond the
        # consumer's one-second lease cannot expire a reservation because it
        # does not have a lease deadline yet.
        await asyncio.sleep(1.1)
        release_commit.set()
        persisted = await persistence_task
        reservation = persisted.initial_reservations[0]
        unactivated = await emitting_store.get_delivery(
            agent_id=agent_id,
            consumer_id=consumer_id,
            delivery_id=reservation.delivery_id,
        )
        assert unactivated is not None
        assert unactivated.status == "initial_reserved"
        assert unactivated.lease_expires_at is None

        peer_claim, activated = await asyncio.gather(
            peer_store.claim_delivery(
                agent_id=agent_id,
                consumer_id=consumer_id,
                executor_id="peer-after-visibility",
            ),
            emitting_store.activate_initial_delivery(
                agent_id=agent_id,
                consumer_id=consumer_id,
                delivery_id=reservation.delivery_id,
                initial_lease_owner=owner_id,
                initial_lease_token=reservation.reservation_token,
            ),
        )
        assert peer_claim is None
        assert activated is not None
        assert activated.lease_expires_at > datetime.now(timezone.utc)
    finally:
        release_commit.set()
        db_backend.transaction = original_transaction
        if persistence_task is not None:
            await asyncio.gather(persistence_task, return_exceptions=True)
        await peer_backend.close()


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_startup_recovers_only_a_stale_unactivated_owner_as_marker_work(
    db_backend,
):
    """A crash before activation replays only the durable privacy marker."""
    peer_backend = await _independent_backend(db_backend)
    recovering_agent = None
    recovering_dispatcher = None
    try:
        emitting_store = DurableSignalStore(db_backend)
        await emitting_store.initialize()
        agent_id = f"did:test:durable-stale-owner:{uuid4()}"
        consumer_id = "workflow-wait"
        # Recovery deliberately considers only managed dispatcher owners;
        # arbitrary executor namespaces must never be stolen as crashed hosts.
        owner_id = _dispatcher_owner_id()
        now = datetime.now(timezone.utc)
        await emitting_store.register_consumer(
            DurableConsumerRegistration(
                consumer_id=consumer_id,
                source="provider.message",
                agent_id=agent_id,
            )
        )
        await emitting_store.register_runtime_owner(
            agent_id=agent_id,
            owner_id=owner_id,
            now=now - timedelta(minutes=10),
        )
        marker_event = _signal(agent_id)
        marker_event.payload = {"_privacy_gated": "none"}
        secret = "crash-before-activation@example.com"
        persisted = await emitting_store.persist_signal(
            marker_event,
            agent_id=agent_id,
            source_event_id=f"stale-owner-{uuid4()}",
            retention_days=7,
            transient_selector_payload={"workflow": "wf-1", "message": secret},
            initial_lease_owner=owner_id,
        )
        reservation = persisted.initial_reservations[0]
        # Startup initializes a distinct runtime owner and invokes the
        # owner-aware recovery path. The old owner heartbeat is ten minutes
        # stale, so the marker becomes ordinary retry work before any generic
        # worker can claim it.
        recovering_agent, recovering_dispatcher = await _ephemeral_dispatcher(
            peer_backend, agent_id=agent_id
        )
        recovered = await recovering_dispatcher._durable_store.get_delivery(
            agent_id=agent_id,
            consumer_id=consumer_id,
            delivery_id=reservation.delivery_id,
        )
        assert recovered is not None
        assert recovered.status == "retry"
        replay = await recovering_dispatcher.claim_durable_delivery(
            consumer_id=consumer_id,
            executor_id="restart-worker",
        )
        assert replay is not None
        assert replay.event.payload == {"_privacy_gated": "none"}
        assert secret not in str(replay.event.payload)
    finally:
        if recovering_agent is not None and recovering_dispatcher is not None:
            await _finish_dispatcher(recovering_agent, recovering_dispatcher)
        await peer_backend.close()


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_live_runtime_owner_cannot_be_recovered_by_a_concurrent_dispatcher(
    db_backend,
):
    """Owner-aware recovery never steals a contemporaneously live emitter."""
    peer_backend = await _independent_backend(db_backend)
    try:
        emitting_store = DurableSignalStore(db_backend)
        peer_store = DurableSignalStore(peer_backend)
        await emitting_store.initialize()
        await peer_store.initialize()
        agent_id = f"did:test:durable-live-owner:{uuid4()}"
        consumer_id = "workflow-wait"
        owner_id = _dispatcher_owner_id()
        peer_owner_id = _dispatcher_owner_id()
        now = datetime.now(timezone.utc)
        await emitting_store.register_consumer(
            DurableConsumerRegistration(
                consumer_id=consumer_id,
                source="provider.message",
                agent_id=agent_id,
            )
        )
        await emitting_store.register_runtime_owner(
            agent_id=agent_id, owner_id=owner_id, now=now
        )
        await peer_store.register_runtime_owner(
            agent_id=agent_id, owner_id=peer_owner_id, now=now
        )
        marker_event = _signal(agent_id)
        marker_event.payload = {"_privacy_gated": "none"}
        persisted = await emitting_store.persist_signal(
            marker_event,
            agent_id=agent_id,
            source_event_id=f"live-owner-{uuid4()}",
            retention_days=7,
            initial_lease_owner=owner_id,
        )
        reservation = persisted.initial_reservations[0]
        assert await peer_store.recover_abandoned_initial_reservations(
            agent_id=agent_id,
            recovering_owner_id=peer_owner_id,
            stale_before=now - timedelta(seconds=1),
            now=now,
        ) == 0
        assert await peer_store.claim_delivery(
            agent_id=agent_id,
            consumer_id=consumer_id,
            executor_id="peer-worker",
        ) is None
        assert await emitting_store.activate_initial_delivery(
            agent_id=agent_id,
            consumer_id=consumer_id,
            delivery_id=reservation.delivery_id,
            initial_lease_owner=owner_id,
            initial_lease_token=reservation.reservation_token,
            now=now,
        ) is not None
    finally:
        await peer_backend.close()


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_delayed_runtime_heartbeat_cannot_regress_owner_liveness_or_release_work(
    db_backend,
):
    """A heartbeat sampled before a wait must not undo newer liveness.

    Explicit times model a heartbeat which sampled ``older`` before blocking,
    while a newer owner touch completed first.  It then completes late on the
    same owner row.  Recovery from a concurrent dispatcher must still see the
    newer heartbeat and leave the initial reservation untouched.
    """
    peer_backend = await _independent_backend(db_backend)
    try:
        emitting_store = DurableSignalStore(db_backend)
        peer_store = DurableSignalStore(peer_backend)
        await emitting_store.initialize()
        await peer_store.initialize()
        agent_id = f"did:test:durable-monotonic-heartbeat:{uuid4()}"
        consumer_id = "workflow-wait"
        owner_id = _dispatcher_owner_id()
        recovering_owner_id = _dispatcher_owner_id()
        base = datetime(2040, 1, 1, tzinfo=timezone.utc)
        older = base + timedelta(seconds=1)
        newer = base + timedelta(seconds=10)
        recovery_now = base + timedelta(seconds=11)
        await emitting_store.register_consumer(
            DurableConsumerRegistration(
                consumer_id=consumer_id,
                source="provider.message",
                agent_id=agent_id,
            )
        )
        await emitting_store.register_runtime_owner(
            agent_id=agent_id, owner_id=owner_id, now=base
        )
        await peer_store.register_runtime_owner(
            agent_id=agent_id, owner_id=recovering_owner_id, now=newer
        )
        marker_event = _signal(agent_id)
        marker_event.payload = {"_privacy_gated": "none"}
        persisted = await emitting_store.persist_signal(
            marker_event,
            agent_id=agent_id,
            source_event_id=f"monotonic-heartbeat-{uuid4()}",
            retention_days=7,
            initial_lease_owner=owner_id,
        )
        reservation = persisted.initial_reservations[0]

        # A newer activation/heartbeat touch wins; the delayed heartbeat that
        # was sampled earlier must not overwrite it at commit time.
        await emitting_store.heartbeat_runtime_owner(
            agent_id=agent_id, owner_id=owner_id, now=newer
        )
        await emitting_store.heartbeat_runtime_owner(
            agent_id=agent_id, owner_id=owner_id, now=older
        )

        assert await peer_store.recover_abandoned_initial_reservations(
            agent_id=agent_id,
            recovering_owner_id=recovering_owner_id,
            stale_before=base + timedelta(seconds=5),
            now=recovery_now,
        ) == 0
        remaining = await peer_store.get_delivery(
            agent_id=agent_id,
            consumer_id=consumer_id,
            delivery_id=reservation.delivery_id,
        )
        assert remaining is not None
        assert remaining.status == "initial_reserved"
        assert await peer_store.claim_delivery(
            agent_id=agent_id,
            consumer_id=consumer_id,
            executor_id="peer-worker",
        ) is None
    finally:
        await peer_backend.close()


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_postgres_recovery_waits_for_overlapping_live_runtime_heartbeat(
    db_backend,
):
    """A PostgreSQL recovery cannot classify a heartbeat mid-transaction stale."""

    if db_backend.backend_type != "postgres":
        pytest.skip("PostgreSQL advisory-lock overlap regression")

    peer_backend = await _independent_backend(db_backend)
    heartbeat_task = None
    recovery_task = None
    try:
        emitting_store = DurableSignalStore(db_backend)
        recovering_store = DurableSignalStore(peer_backend)
        await emitting_store.initialize()
        await recovering_store.initialize()
        agent_id = f"did:test:durable-heartbeat-overlap:{uuid4()}"
        consumer_id = "workflow-wait"
        owner_id = _dispatcher_owner_id()
        recovering_owner_id = _dispatcher_owner_id()
        base = datetime.now(timezone.utc)
        stale = base - timedelta(minutes=5)
        now = base + timedelta(seconds=1)
        await emitting_store.register_consumer(
            DurableConsumerRegistration(
                consumer_id=consumer_id,
                source="provider.message",
                agent_id=agent_id,
            )
        )
        await emitting_store.register_runtime_owner(
            agent_id=agent_id, owner_id=owner_id, now=stale
        )
        await recovering_store.register_runtime_owner(
            agent_id=agent_id, owner_id=recovering_owner_id, now=now
        )
        marker = _signal(agent_id)
        persisted = await emitting_store.persist_signal(
            marker,
            agent_id=agent_id,
            source_event_id=f"heartbeat-overlap:{uuid4()}",
            retention_days=7,
            initial_lease_owner=owner_id,
        )
        reservation = persisted.initial_reservations[0]
        activated = await emitting_store.activate_initial_delivery(
            agent_id=agent_id,
            consumer_id=consumer_id,
            delivery_id=reservation.delivery_id,
            initial_lease_owner=owner_id,
            initial_lease_token=reservation.reservation_token,
            now=stale,
        )
        assert activated is not None and activated.status == "leased"

        heartbeat_updated = asyncio.Event()
        release_heartbeat = asyncio.Event()
        original_touch = emitting_store._touch_runtime_owner_locked

        async def pause_after_owner_touch(**kwargs):
            await original_touch(**kwargs)
            heartbeat_updated.set()
            await release_heartbeat.wait()

        emitting_store._touch_runtime_owner_locked = pause_after_owner_touch
        heartbeat_task = asyncio.create_task(
            emitting_store.heartbeat_runtime_owner(
                agent_id=agent_id, owner_id=owner_id, now=now
            )
        )
        await asyncio.wait_for(heartbeat_updated.wait(), timeout=2)
        recovery_task = asyncio.create_task(
            recovering_store.recover_abandoned_leases(
                agent_id=agent_id,
                recovering_owner_id=recovering_owner_id,
                stale_before=base,
                now=now,
            )
        )
        await asyncio.sleep(0.05)
        assert recovery_task.done() is False

        release_heartbeat.set()
        await heartbeat_task
        assert await recovery_task == 0
        delivery = await recovering_store.get_delivery(
            agent_id=agent_id,
            consumer_id=consumer_id,
            delivery_id=reservation.delivery_id,
        )
        assert delivery is not None and delivery.status == "leased"
    finally:
        if heartbeat_task is not None and not heartbeat_task.done():
            heartbeat_task.cancel()
        if recovery_task is not None and not recovery_task.done():
            recovery_task.cancel()
        await asyncio.gather(
            *(task for task in (heartbeat_task, recovery_task) if task is not None),
            return_exceptions=True,
        )
        await peer_backend.close()


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_same_dispatcher_initial_claim_waits_for_postgres_commit_boundary(
    db_backend,
):
    """A local claimant must not discard a sidecar before its row commits.

    PostgreSQL claims run in a separate transaction.  Pause the emitting
    transaction after the synchronous sidecar callback, start the same
    dispatcher's durable claim, then release commit.  Whether PostgreSQL
    blocks that claim on the source handoff lock or its initial-handoff lock,
    it must transfer the now-visible reservation and receive the raw payload.
    """
    if db_backend.backend_type != "postgres":
        pytest.skip("PostgreSQL-specific separate-transaction visibility race")

    agent_id = f"did:test:durable-local-commit-race:{uuid4()}"
    consumer_id = "workflow-wait"
    agent, dispatcher = await _ephemeral_dispatcher(db_backend, agent_id=agent_id)
    original_transaction = db_backend.transaction
    original_fetch_val = db_backend.fetch_val
    commit_boundary = asyncio.Event()
    claim_scope_waiting = asyncio.Event()
    release_commit = asyncio.Event()
    dispatch_task = None
    claim_task = None

    @asynccontextmanager
    async def pause_after_before_commit():
        async with original_transaction():
            yield
            # The raw sidecar is installed, but this PostgreSQL transaction
            # has not committed its event/delivery rows yet.
            commit_boundary.set()
            await release_commit.wait()

    try:
        await asyncio.wait_for(
            dispatcher.register_durable_consumer(
                DurableConsumerRegistration(
                    consumer_id=consumer_id,
                    source="provider.message",
                    agent_id=agent_id,
                    correlation_selector="payload.workflow=wf-1",
                    lease_seconds=10,
                )
            ),
            timeout=5,
        )
        # Register before installing the barrier.  The backend seam wraps every
        # transaction, and pausing registration here would block the test
        # before it starts the emitting transaction it is meant to observe.
        db_backend.transaction = pause_after_before_commit
        secret = "same-dispatcher-commit-boundary@example.com"
        event = _signal(agent_id)
        event.payload["message"] = secret
        dispatch_task = asyncio.create_task(
            dispatcher.dispatch_signal(
                event,
                source_event_id=f"same-dispatcher-commit-race:{uuid4()}",
            )
        )
        await asyncio.wait_for(commit_boundary.wait(), timeout=5)
        assert len(dispatcher._transient_durable_handoffs) == 1

        async def note_claim_scope_wait(query, params=()):
            if _is_scope_handoff_lock(query):
                claim_scope_waiting.set()
            return await original_fetch_val(query, params)

        # Install only after persistence reaches its pre-commit barrier; its
        # own earlier acquisition of this same lock is not the observation.
        db_backend.fetch_val = note_claim_scope_wait

        claim_task = asyncio.create_task(
            dispatcher.claim_durable_delivery(
                consumer_id=consumer_id,
                executor_id="local-worker",
            )
        )
        await asyncio.wait_for(claim_scope_waiting.wait(), timeout=5)
        # The source-level database handoff lock may serialize the ordinary
        # poll before it can observe the uncommitted row. If it does not, the
        # local handoff lock still protects the sidecar-to-reservation transfer.
        assert not claim_task.done()

        release_commit.set()
        assert (
            await asyncio.wait_for(asyncio.shield(dispatch_task), timeout=5)
        ).status is Status.OK
        delivery = await asyncio.wait_for(asyncio.shield(claim_task), timeout=5)
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
        db_backend.fetch_val = original_fetch_val
        try:
            await _cancel_and_drain(claim_task, dispatch_task)
        finally:
            await _finish_dispatcher(agent, dispatcher)


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_initial_reservation_lease_starts_only_after_post_commit_activation_on_both_backends(
    db_backend,
):
    """A delayed persistence publishes no lease until the owner activates it."""
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
        # The old implementation could spend a full lease duration before it
        # even acquired this handoff lock. A reservation must still carry no
        # live deadline when its transaction later commits.
        await asyncio.sleep(1.1)
        release_handoff.set()
        await holder
        persisted = await persistence_task
        assert len(persisted.initial_reservations) == 1
        reservation = persisted.initial_reservations[0]
        unactivated = await emitting_store.get_delivery(
            agent_id=agent_id,
            consumer_id=consumer_id,
            delivery_id=reservation.delivery_id,
        )
        assert unactivated is not None
        assert unactivated.status == "initial_reserved"
        assert unactivated.lease_expires_at is None
        assert await blocking_store.claim_delivery(
            agent_id=agent_id,
            consumer_id=consumer_id,
            executor_id="peer-before-activation",
        ) is None
        activated = await emitting_store.activate_initial_delivery(
            agent_id=agent_id,
            consumer_id=consumer_id,
            delivery_id=reservation.delivery_id,
            initial_lease_owner="emitting-dispatcher",
            initial_lease_token=reservation.reservation_token,
        )
        assert activated is not None
        assert activated.lease_expires_at > datetime.now(timezone.utc)
        assert await emitting_store.claim_initial_delivery(
            agent_id=agent_id,
            consumer_id=consumer_id,
            delivery_id=reservation.delivery_id,
            initial_lease_owner="emitting-dispatcher",
            initial_lease_token=reservation.reservation_token,
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
async def test_initial_worker_transfer_does_not_publish_an_expired_lease_after_contention(
    db_backend,
):
    """The initial owner lease is rechecked after the real write/row lock.

    A deterministic clock advances while another connection holds the exact
    serialization point.  The old transfer sampled before waiting and returned
    a worker lease already expired at the advanced clock; the fixed path sees
    the initial owner lease expired and declines the handoff.
    """
    peer_backend = await _independent_backend(db_backend)
    release_lock = asyncio.Event()
    lock_acquired = asyncio.Event()
    transfer_waiting = asyncio.Event()
    holder = None
    try:
        store = DurableSignalStore(db_backend)
        peer_store = DurableSignalStore(peer_backend)
        await store.initialize()
        await peer_store.initialize()
        agent_id = f"did:test:durable-initial-transfer:{uuid4()}"
        consumer_id = "workflow-wait"
        owner_id = _dispatcher_owner_id()
        base = datetime(2040, 1, 1, tzinfo=timezone.utc)
        current_time = {"value": base}
        store.now_utc = lambda: current_time["value"]
        await store.register_consumer(
            DurableConsumerRegistration(
                consumer_id=consumer_id,
                source="provider.message",
                agent_id=agent_id,
                lease_seconds=1,
            )
        )
        await store.register_runtime_owner(
            agent_id=agent_id, owner_id=owner_id, now=base
        )
        marker_event = _signal(agent_id)
        marker_event.payload = {"_privacy_gated": "none"}
        persisted = await store.persist_signal(
            marker_event,
            agent_id=agent_id,
            source_event_id=f"initial-transfer-contention:{uuid4()}",
            retention_days=7,
            initial_lease_owner=owner_id,
        )
        reservation = persisted.initial_reservations[0]
        assert await store.activate_initial_delivery(
            agent_id=agent_id,
            consumer_id=consumer_id,
            delivery_id=reservation.delivery_id,
            initial_lease_owner=owner_id,
            initial_lease_token=reservation.reservation_token,
            now=base,
        ) is not None

        async def hold_transfer_serialization_point() -> None:
            async with peer_backend.transaction():
                if peer_backend.backend_type == "postgres":
                    await peer_backend.fetch_one(
                        "SELECT delivery_id FROM durable_signal_deliveries "
                        "WHERE delivery_id = ? FOR UPDATE",
                        (reservation.delivery_id,),
                    )
                else:
                    await peer_backend.execute(
                        "UPDATE durable_signal_deliveries "
                        "SET updated_at = updated_at WHERE delivery_id = ?",
                        (reservation.delivery_id,),
                    )
                lock_acquired.set()
                await release_lock.wait()

        holder = asyncio.create_task(hold_transfer_serialization_point())
        await asyncio.wait_for(lock_acquired.wait(), timeout=5)

        if db_backend.backend_type == "postgres":
            original_fetch_one = db_backend.fetch_one

            async def observe_transfer_lock(query, params=()):
                if "FOR UPDATE" in query and "durable_signal_deliveries" in query:
                    transfer_waiting.set()
                return await original_fetch_one(query, params)

            db_backend.fetch_one = observe_transfer_lock
        else:
            original_execute = db_backend.execute

            async def observe_transfer_lock(query, params=()):
                if (
                    "UPDATE durable_signal_deliveries" in query
                    and "SET updated_at = updated_at" in query
                ):
                    transfer_waiting.set()
                return await original_execute(query, params)

            db_backend.execute = observe_transfer_lock

        try:
            transfer_task = asyncio.create_task(
                store.claim_initial_delivery(
                    agent_id=agent_id,
                    consumer_id=consumer_id,
                    delivery_id=reservation.delivery_id,
                    initial_lease_owner=owner_id,
                    initial_lease_token=reservation.reservation_token,
                    executor_id="owner-worker",
                )
            )
            await asyncio.wait_for(transfer_waiting.wait(), timeout=5)
            current_time["value"] = base + timedelta(seconds=2)
            release_lock.set()
            assert await transfer_task is None
        finally:
            if db_backend.backend_type == "postgres":
                db_backend.fetch_one = original_fetch_one
            else:
                db_backend.execute = original_execute

        row = await store.get_delivery(
            agent_id=agent_id,
            consumer_id=consumer_id,
            delivery_id=reservation.delivery_id,
        )
        assert row is not None
        assert row.status == "leased"
        assert row.lease_owner == owner_id
        assert row.attempts == 0
    finally:
        release_lock.set()
        if holder is not None:
            await asyncio.gather(holder, return_exceptions=True)
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
