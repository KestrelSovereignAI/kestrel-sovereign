"""Durable signal consumer contract: replay, scoped leasing, and retention."""

from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock

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
    ACKNOWLEDGED,
    FAILED,
    LEASED,
    RETRY,
    DurableConsumerRegistration,
    DurableSignalStore,
    OrderedLockManager,
    SignalDispatcher,
    SignalLogStore,
    SourceRegistry,
)
from kestrel_sovereign.signals.sources.channels import (
    build_channel_message_registration,
)
from kestrel_sovereign.storage.db import SQLiteBackend


class _Agent:
    def __init__(self, did: str):
        self._did = did
        self.tasks: list[asyncio.Task] = []
        self.action_calls = 0
        self.action_payloads: list[dict] = []
        self._privacy_transition_lock = asyncio.Lock()

    @property
    def did(self) -> str:
        return self._did

    async def process_input(self, prompt: str):  # pragma: no cover - ACTION only
        return prompt

    def _get_privacy_transition_lock(self):
        """Mirror the production agent seam used by durable persistence."""
        return self._privacy_transition_lock

    def _track_background_task(self, coro, *, name: str):
        task = asyncio.create_task(coro, name=name)
        self.tasks.append(task)
        return task


def _registration(agent: _Agent, source: str = "provider.message") -> SourceRegistration:
    async def handler(payload):
        agent.action_calls += 1
        agent.action_payloads.append(dict(payload))
        return {"handled": payload["message"]}

    def normalize(payload):
        # This represents a source-owned sanitizer/schema canonicalization.
        return {"message": str(payload["message"]).strip(), "workflow": str(payload["workflow"])}

    return SourceRegistration(
        name=source,
        schema=normalize,
        default_mode=SignalMode.ACTION,
        allowed_modes=frozenset({SignalMode.ACTION}),
        handler=handler,
        trust=Trust.TRUSTED,
        log_redaction=RedactionPolicy(summarize=lambda payload: "<redacted>"),
        retention_days=7,
    )


def _signal(*, agent_id: str, source: str = "provider.message", message: str = " hello ", workflow: str = "wf-1") -> Signal:
    return Signal(
        source=source,
        kind="inbound",
        mode=SignalMode.ACTION,
        payload={"message": message, "workflow": workflow},
        target_agent=agent_id,
    )


async def _dispatcher(path, did: str):
    backend = SQLiteBackend(str(path))
    await backend.connect()
    log_store = SignalLogStore(backend)
    await log_store.initialize()
    agent = _Agent(did)
    registry = SourceRegistry()
    registry.register(_registration(agent))
    dispatcher = SignalDispatcher(
        agent=agent,
        registry=registry,
        lock_manager=OrderedLockManager(),
        store=log_store,
    )
    await dispatcher.initialize_durable_delivery()
    return backend, agent, dispatcher


async def _close(backend, agent: _Agent) -> None:
    pending = [task for task in agent.tasks if not task.done()]
    if pending:
        await asyncio.gather(*pending, return_exceptions=True)
    await backend.close()


@pytest.mark.asyncio
async def test_committed_signal_replays_after_restart_with_normalized_payload(tmp_path):
    path = tmp_path / "durable.db"
    backend, agent, dispatcher = await _dispatcher(path, "did:agent:one")
    consumer = DurableConsumerRegistration(
        consumer_id="workflow-wait",
        source="provider.message",
        agent_id=agent.did,
        correlation_selector="payload.workflow=wf-1",
    )
    await dispatcher.register_durable_consumer(consumer)

    result = await dispatcher.dispatch_signal(
        _signal(agent_id=agent.did), source_event_id="provider-evt-1"
    )
    assert result.status is Status.OK
    await _close(backend, agent)  # Simulates a process death before claim/ack.

    backend2, agent2, dispatcher2 = await _dispatcher(path, "did:agent:one")
    await dispatcher2.register_durable_consumer(consumer)  # startup is idempotent
    delivery = await dispatcher2.claim_durable_delivery(
        consumer_id="workflow-wait", executor_id="workflow-runner-1"
    )
    assert delivery is not None
    assert delivery.status == LEASED
    assert delivery.event.payload == {"message": "hello", "workflow": "wf-1"}
    assert delivery.event.causation_chain[-1]["signal_id"] == result.signal_id
    assert await dispatcher2.ack_durable_delivery(
        consumer_id="workflow-wait",
        delivery_id=delivery.delivery_id,
        lease_token=delivery.lease_token,
    )
    assert (await dispatcher2.list_durable_deliveries())[0].status == ACKNOWLEDGED
    await _close(backend2, agent2)


@pytest.mark.asyncio
async def test_source_event_dedup_prevents_duplicate_delivery_and_side_effect(tmp_path):
    backend, agent, dispatcher = await _dispatcher(tmp_path / "dedup.db", "did:agent:one")
    await dispatcher.register_durable_consumer(
        DurableConsumerRegistration(
            consumer_id="all-messages", source="provider.message", agent_id=agent.did
        )
    )

    first = await dispatcher.dispatch_signal(
        _signal(agent_id=agent.did), source_event_id="provider-evt-1"
    )
    duplicate = await dispatcher.dispatch_signal(
        _signal(agent_id=agent.did, message="duplicate"), source_event_id="provider-evt-1"
    )

    assert first.status is Status.OK
    assert duplicate.status is Status.COALESCED
    assert agent.action_calls == 1
    deliveries = await dispatcher.list_durable_deliveries()
    assert len(deliveries) == 1
    assert deliveries[0].event.event_id == first.signal_id
    await _close(backend, agent)


@pytest.mark.asyncio
async def test_two_executors_cannot_claim_the_same_delivery(tmp_path):
    path = tmp_path / "claim-race.db"
    backend_a, agent_a, dispatcher_a = await _dispatcher(path, "did:agent:one")
    backend_b, agent_b, dispatcher_b = await _dispatcher(path, "did:agent:one")
    consumer = DurableConsumerRegistration(
        consumer_id="workflow-wait", source="provider.message", agent_id=agent_a.did
    )
    await dispatcher_a.register_durable_consumer(consumer)
    await dispatcher_b.register_durable_consumer(consumer)
    assert (await dispatcher_a.dispatch_signal(
        _signal(agent_id=agent_a.did), source_event_id="provider-evt-1"
    )).status is Status.OK

    first, second = await asyncio.gather(
        dispatcher_a.claim_durable_delivery(
            consumer_id="workflow-wait", executor_id="executor-a"
        ),
        dispatcher_b.claim_durable_delivery(
            consumer_id="workflow-wait", executor_id="executor-b"
        ),
    )
    claimed = [delivery for delivery in (first, second) if delivery is not None]
    assert len(claimed) == 1
    assert claimed[0].status == LEASED
    await _close(backend_a, agent_a)
    await _close(backend_b, agent_b)


@pytest.mark.asyncio
async def test_volatile_sidecar_is_installed_before_commit_and_reserved_from_peer(tmp_path):
    """A peer cannot steal a just-committed marker before its owner claims raw data."""
    from kestrel_sovereign.privacy import get_privacy_preset

    path = tmp_path / "volatile-commit-boundary.db"
    backend_a, agent_a, dispatcher_a = await _dispatcher(path, "did:agent:one")
    backend_b, agent_b, dispatcher_b = await _dispatcher(path, "did:agent:one")
    agent_a.privacy_config = get_privacy_preset("ephemeral")
    agent_b.privacy_config = get_privacy_preset("ephemeral")
    consumer = DurableConsumerRegistration(
        consumer_id="workflow-wait", source="provider.message", agent_id=agent_a.did
    )
    secret = "commit-boundary-customer@example.com"
    await dispatcher_a.register_durable_consumer(consumer)
    await dispatcher_b.register_durable_consumer(consumer)

    original_transaction = backend_a.transaction
    original_schedule = dispatcher_a._schedule_transient_durable_handoff_expiry
    sidecar_installed = asyncio.Event()
    commit_boundary = asyncio.Event()
    release_commit = asyncio.Event()

    @asynccontextmanager
    async def pause_after_sidecar_before_commit():
        async with original_transaction():
            yield
            # ``persist_signal`` has returned from its before-commit callback,
            # but the event row is not visible to the peer connection yet.
            commit_boundary.set()
            await release_commit.wait()

    def note_sidecar_install(delivery_id, expires_at):
        original_schedule(delivery_id, expires_at)
        sidecar_installed.set()

    backend_a.transaction = pause_after_sidecar_before_commit
    dispatcher_a._schedule_transient_durable_handoff_expiry = note_sidecar_install

    peer_ready = asyncio.Event()
    allow_peer_claim = asyncio.Event()
    original_peer_fetch_one = backend_b.fetch_one

    async def pause_peer_after_consumer_read(query, params=()):
        row = await original_peer_fetch_one(query, params)
        if (
            "FROM durable_signal_consumers" in query
            and "consumer_id" in query
        ):
            peer_ready.set()
            await allow_peer_claim.wait()
        return row

    backend_b.fetch_one = pause_peer_after_consumer_read
    try:
        dispatch_task = asyncio.create_task(
            dispatcher_a.dispatch_signal(
                _signal(agent_id=agent_a.did, message=secret),
                source_event_id="volatile-commit-boundary",
            )
        )
        await asyncio.wait_for(commit_boundary.wait(), timeout=2)
        await asyncio.wait_for(sidecar_installed.wait(), timeout=2)
        assert len(dispatcher_a._transient_durable_handoffs) == 1

        peer_claim_task = asyncio.create_task(
            dispatcher_b.claim_durable_delivery(
                consumer_id=consumer.consumer_id, executor_id="peer-worker"
            )
        )
        await asyncio.wait_for(peer_ready.wait(), timeout=2)
        assert not peer_claim_task.done()

        # Force the peer's actual lease attempt to run only after the emitting
        # transaction commits.  The row is already leased to dispatcher A, so
        # worker B must not see a claimable marker-only delivery.
        release_commit.set()
        assert (await dispatch_task).status is Status.OK
        allow_peer_claim.set()
        assert await peer_claim_task is None

        owned = await dispatcher_a.claim_durable_delivery(
            consumer_id=consumer.consumer_id, executor_id="owner-worker"
        )
        assert owned is not None
        assert owned.event.payload == {"message": secret, "workflow": "wf-1"}
        durable_event = await backend_a.fetch_one(
            "SELECT payload FROM durable_signal_events WHERE event_id = ?",
            (owned.event_id,),
        )
        durable_delivery = await backend_a.fetch_one(
            "SELECT lease_owner, lease_token FROM durable_signal_deliveries "
            "WHERE delivery_id = ?",
            (owned.delivery_id,),
        )
        assert durable_event is not None and secret not in durable_event[0]
        assert durable_delivery is not None and secret not in repr(durable_delivery)
    finally:
        backend_a.transaction = original_transaction
        dispatcher_a._schedule_transient_durable_handoff_expiry = original_schedule
        backend_b.fetch_one = original_peer_fetch_one
        dispatcher_a.shutdown()
        dispatcher_b.shutdown()
        await _close(backend_a, agent_a)
        await _close(backend_b, agent_b)


@pytest.mark.asyncio
async def test_concurrent_local_initial_claims_preserve_the_winner_payload(tmp_path):
    """A losing local reservation transfer cannot erase the winner's sidecar."""
    from kestrel_sovereign.privacy import get_privacy_preset

    backend, agent, dispatcher = await _dispatcher(
        tmp_path / "volatile-local-claim-race.db", "did:agent:one"
    )
    agent.privacy_config = get_privacy_preset("ephemeral")
    consumer = DurableConsumerRegistration(
        consumer_id="workflow-wait", source="provider.message", agent_id=agent.did
    )
    transferred = asyncio.Event()
    allow_winner_return = asyncio.Event()
    original_claim_initial = dispatcher._durable_store.claim_initial_delivery

    async def pause_successful_transfer(**kwargs):
        delivery = await original_claim_initial(**kwargs)
        if delivery is not None:
            transferred.set()
            await allow_winner_return.wait()
        return delivery

    try:
        await dispatcher.register_durable_consumer(consumer)
        secret = "local-claim-race-customer@example.com"
        assert (
            await dispatcher.dispatch_signal(
                _signal(agent_id=agent.did, message=secret),
                source_event_id="volatile-local-claim-race",
            )
        ).status is Status.OK
        dispatcher._durable_store.claim_initial_delivery = pause_successful_transfer

        winner_task = asyncio.create_task(
            dispatcher.claim_durable_delivery(
                consumer_id=consumer.consumer_id, executor_id="worker-a"
            )
        )
        await asyncio.wait_for(transferred.wait(), timeout=2)

        loser_task = asyncio.create_task(
            dispatcher.claim_durable_delivery(
                consumer_id=consumer.consumer_id, executor_id="worker-b"
            )
        )
        # The second claimant has made its ordinary durable poll, but cannot
        # reach the initial transfer while the first owns the local handoff.
        await asyncio.sleep(0)
        assert not loser_task.done()
        assert len(dispatcher._transient_durable_handoffs) == 1

        allow_winner_return.set()
        winner = await winner_task
        loser = await loser_task
        assert winner is not None
        assert winner.event.payload == {"message": secret, "workflow": "wf-1"}
        assert loser is None
    finally:
        dispatcher._durable_store.claim_initial_delivery = original_claim_initial
        dispatcher.shutdown()
        await _close(backend, agent)


@pytest.mark.asyncio
async def test_initial_reservation_starts_after_sqlite_handoff_contention(tmp_path):
    """A blocked writer cannot commit an initial lease that has already expired."""
    from kestrel_sovereign.privacy import get_privacy_preset

    path = tmp_path / "volatile-reservation-contention.db"
    backend, agent, dispatcher = await _dispatcher(path, "did:agent:one")
    peer_backend = SQLiteBackend(str(path))
    await peer_backend.connect()
    agent.privacy_config = get_privacy_preset("ephemeral")
    consumer = DurableConsumerRegistration(
        consumer_id="workflow-wait",
        source="provider.message",
        agent_id=agent.did,
        lease_seconds=1,
    )
    writer_acquired = asyncio.Event()
    release_writer = asyncio.Event()

    async def hold_sqlite_writer() -> None:
        async with peer_backend.transaction():
            # This is the same no-op write DurableSignalStore uses to acquire
            # its cross-connection SQLite handoff lock.
            await peer_backend.execute("DELETE FROM durable_signal_consumers WHERE 0")
            writer_acquired.set()
            await release_writer.wait()

    holder = None
    try:
        await dispatcher.register_durable_consumer(consumer)
        holder = asyncio.create_task(hold_sqlite_writer())
        await asyncio.wait_for(writer_acquired.wait(), timeout=2)
        dispatch_task = asyncio.create_task(
            dispatcher.dispatch_signal(
                _signal(
                    agent_id=agent.did,
                    message="contended-reservation-customer@example.com",
                ),
                source_event_id="volatile-reservation-contention",
            )
        )
        # The old method-entry timestamp would now be more than a full lease
        # old before it could acquire the handoff writer lock.
        await asyncio.sleep(1.1)
        release_writer.set()
        await holder
        assert (await dispatch_task).status is Status.OK

        owned = await dispatcher.claim_durable_delivery(
            consumer_id=consumer.consumer_id, executor_id="owner-worker"
        )
        assert owned is not None
        assert owned.event.payload == {
            "message": "contended-reservation-customer@example.com",
            "workflow": "wf-1",
        }
    finally:
        release_writer.set()
        if holder is not None:
            await asyncio.gather(holder, return_exceptions=True)
        dispatcher.shutdown()
        await _close(backend, agent)
        await peer_backend.close()


@pytest.mark.asyncio
async def test_notify_resume_expires_and_reschedules_volatile_handoffs(tmp_path):
    """Host resume reconciles sidecars against UTC, not frozen monotonic timers."""
    from kestrel_sovereign.privacy import get_privacy_preset

    backend, agent, dispatcher = await _dispatcher(
        tmp_path / "volatile-resume-reconciliation.db", "did:agent:one"
    )
    agent.privacy_config = get_privacy_preset("ephemeral")
    consumer = DurableConsumerRegistration(
        consumer_id="workflow-wait", source="provider.message", agent_id=agent.did
    )
    try:
        await dispatcher.register_durable_consumer(consumer)
        for event_id, message in (
            ("resume-expired", "expired-on-resume@example.com"),
            ("resume-live", "live-on-resume@example.com"),
        ):
            assert (
                await dispatcher.dispatch_signal(
                    _signal(agent_id=agent.did, message=message),
                    source_event_id=event_id,
                )
            ).status is Status.OK

        expired = await dispatcher.claim_durable_delivery(
            consumer_id=consumer.consumer_id, executor_id="worker-expired"
        )
        live = await dispatcher.claim_durable_delivery(
            consumer_id=consumer.consumer_id, executor_id="worker-live"
        )
        assert expired is not None and live is not None
        expired_handoff = dispatcher._transient_durable_handoffs[expired.delivery_id]
        live_handoff = dispatcher._transient_durable_handoffs[live.delivery_id]
        expired_handoff.expires_at = datetime.now(timezone.utc) - timedelta(seconds=1)
        live_handoff.expires_at = datetime.now(timezone.utc) + timedelta(seconds=30)
        prior_live_timer = dispatcher._transient_durable_handoff_timers[live.delivery_id]

        dispatcher.notify_resume(3600.0)

        assert expired.delivery_id not in dispatcher._transient_durable_handoffs
        assert live.delivery_id in dispatcher._transient_durable_handoffs
        assert prior_live_timer.cancelled()
        assert (
            dispatcher._transient_durable_handoff_timers[live.delivery_id]
            is not prior_live_timer
        )
    finally:
        dispatcher.shutdown()
        await _close(backend, agent)


@pytest.mark.asyncio
async def test_volatile_sidecars_are_discarded_when_the_event_transaction_rolls_back(tmp_path):
    """Pre-commit sidecars cannot survive a failed event transaction."""
    from kestrel_sovereign.privacy import get_privacy_preset

    backend, agent, dispatcher = await _dispatcher(tmp_path / "volatile-rollback.db", "did:agent:one")
    agent.privacy_config = get_privacy_preset("ephemeral")
    await dispatcher.register_durable_consumer(
        DurableConsumerRegistration(
            consumer_id="workflow-wait", source="provider.message", agent_id=agent.did
        )
    )
    original_transaction = backend.transaction
    sidecar_installed = asyncio.Event()
    original_schedule = dispatcher._schedule_transient_durable_handoff_expiry

    @asynccontextmanager
    async def rollback_after_before_commit():
        async with original_transaction():
            yield
            # This runs after the store invokes its pre-commit sidecar hook.
            raise RuntimeError("force durable event rollback")

    def note_sidecar_install(delivery_id, expires_at):
        original_schedule(delivery_id, expires_at)
        sidecar_installed.set()

    backend.transaction = rollback_after_before_commit
    dispatcher._schedule_transient_durable_handoff_expiry = note_sidecar_install
    try:
        result = await dispatcher.dispatch_signal(
            _signal(agent_id=agent.did, message="rollback-secret@example.com"),
            source_event_id="volatile-rollback",
        )
        assert result.status is Status.FAILED
        assert sidecar_installed.is_set()
        assert dispatcher._transient_durable_handoffs == {}
        assert await backend.fetch_one(
            "SELECT event_id FROM durable_signal_events WHERE agent_id = ?",
            (agent.did,),
        ) is None
    finally:
        backend.transaction = original_transaction
        dispatcher._schedule_transient_durable_handoff_expiry = original_schedule
        dispatcher.shutdown()
        await _close(backend, agent)


@pytest.mark.asyncio
async def test_expired_volatile_reservation_replays_only_the_durable_marker_after_restart(tmp_path):
    """A crashed initial owner yields marker-only replay once its lease expires."""
    from kestrel_sovereign.privacy import get_privacy_preset

    path = tmp_path / "volatile-expired-restart.db"
    backend_a, agent_a, dispatcher_a = await _dispatcher(path, "did:agent:one")
    agent_a.privacy_config = get_privacy_preset("ephemeral")
    consumer = DurableConsumerRegistration(
        consumer_id="workflow-wait",
        source="provider.message",
        agent_id=agent_a.did,
        lease_seconds=1,
    )
    secret = "expired-owner-customer@example.com"
    try:
        await dispatcher_a.register_durable_consumer(consumer)
        assert (await dispatcher_a.dispatch_signal(
            _signal(agent_id=agent_a.did, message=secret),
            source_event_id="volatile-expired-restart",
        )).status is Status.OK
        reserved = (await dispatcher_a.list_durable_deliveries())[0]
        assert reserved.status == LEASED
        assert reserved.lease_expires_at is not None

        # A process death drops raw state but cannot synchronously rewrite its
        # durable lease. A fresh worker must wait for that lease to expire.
        dispatcher_a.shutdown()
        assert dispatcher_a._transient_durable_handoffs == {}
    finally:
        await _close(backend_a, agent_a)

    backend_b, agent_b, dispatcher_b = await _dispatcher(path, "did:agent:one")
    agent_b.privacy_config = get_privacy_preset("ephemeral")
    try:
        await dispatcher_b.register_durable_consumer(consumer)
        replayed = await dispatcher_b._durable_store.claim_delivery(
            agent_id=agent_b.did,
            consumer_id=consumer.consumer_id,
            executor_id="restart-worker",
            now=reserved.lease_expires_at + timedelta(microseconds=1),
        )
        assert replayed is not None
        assert replayed.event.payload == {"_privacy_gated": "none"}
        assert secret not in json.dumps(replayed.event.payload)
    finally:
        dispatcher_b.shutdown()
        await _close(backend_b, agent_b)


@pytest.mark.asyncio
async def test_agent_scope_is_enforced_for_registration_selection_and_ack(tmp_path):
    path = tmp_path / "tenant-scope.db"
    backend_a, agent_a, dispatcher_a = await _dispatcher(path, "did:agent:a")
    backend_b, agent_b, dispatcher_b = await _dispatcher(path, "did:agent:b")
    with pytest.raises(PermissionError):
        await dispatcher_a.register_durable_consumer(
            DurableConsumerRegistration(
                consumer_id="wrong-tenant", source="provider.message", agent_id=agent_b.did
            )
        )
    await dispatcher_a.register_durable_consumer(
        DurableConsumerRegistration(
            consumer_id="workflow-wait", source="provider.message", agent_id=agent_a.did
        )
    )
    await dispatcher_b.register_durable_consumer(
        DurableConsumerRegistration(
            consumer_id="workflow-wait", source="provider.message", agent_id=agent_b.did
        )
    )
    # A source attached to agent A cannot inject a delivery into agent B's
    # scoped ledger merely by setting a foreign target on its envelope.  The
    # dispatcher owns the durable scope; target_agent remains envelope data.
    foreign_target = await dispatcher_a.dispatch_signal(
        _signal(agent_id=agent_b.did), source_event_id="provider-evt-forged"
    )
    assert foreign_target.status is Status.OK
    assert await dispatcher_b.claim_durable_delivery(
        consumer_id="workflow-wait", executor_id="executor-b"
    ) is None
    owned = await dispatcher_a.claim_durable_delivery(
        consumer_id="workflow-wait", executor_id="executor-a"
    )
    assert owned is not None
    assert owned.event.agent_id == agent_a.did
    assert owned.event.target_agent == agent_b.did
    assert not await dispatcher_b.ack_durable_delivery(
        consumer_id="workflow-wait",
        delivery_id=owned.delivery_id,
        lease_token=owned.lease_token,
    )
    await _close(backend_a, agent_a)
    await _close(backend_b, agent_b)


@pytest.mark.asyncio
async def test_nack_retry_lease_expiry_terminal_failure_and_retention_are_observable(tmp_path):
    backend = SQLiteBackend(str(tmp_path / "failure-state.db"))
    await backend.connect()
    store = DurableSignalStore(backend)
    await store.initialize()
    agent_id = "did:agent:one"
    await store.register_consumer(
        DurableConsumerRegistration(
            consumer_id="workflow-wait",
            source="provider.message",
            agent_id=agent_id,
            max_attempts=2,
            lease_seconds=1,
        )
    )
    signal = _signal(agent_id=agent_id)
    await store.persist_signal(
        signal, agent_id=agent_id, source_event_id="evt-1", retention_days=1
    )
    now = datetime.now(timezone.utc)
    claimed = await store.claim_delivery(
        agent_id=agent_id, consumer_id="workflow-wait", executor_id="worker", now=now
    )
    assert claimed is not None
    retry = await store.nack_delivery(
        agent_id=agent_id,
        consumer_id="workflow-wait",
        delivery_id=claimed.delivery_id,
        lease_token=claimed.lease_token,
        error="temporary outage",
        retry_delay=timedelta(seconds=5),
        now=now,
    )
    assert retry is not None and retry.status == RETRY
    assert await store.claim_delivery(
        agent_id=agent_id,
        consumer_id="workflow-wait",
        executor_id="worker",
        now=now + timedelta(seconds=4),
    ) is None
    reclaimed = await store.claim_delivery(
        agent_id=agent_id,
        consumer_id="workflow-wait",
        executor_id="worker",
        now=now + timedelta(seconds=5),
    )
    assert reclaimed is not None and reclaimed.attempts == 2
    assert await store.claim_delivery(
        agent_id=agent_id,
        consumer_id="workflow-wait",
        executor_id="other-worker",
        now=now + timedelta(seconds=7),
    ) is None
    terminal = await store.get_delivery(
        agent_id=agent_id, consumer_id="workflow-wait", delivery_id=reclaimed.delivery_id
    )
    assert terminal is not None
    assert terminal.status == FAILED
    assert "lease expired" in terminal.last_error
    # Retention does not erase non-terminal work, but it removes retained
    # terminal history after its configured lifetime.
    assert await store.purge_expired(
        agent_id=agent_id, now=now + timedelta(days=2)
    ) == 1
    assert await store.list_deliveries(agent_id=agent_id) == []
    await backend.close()


@pytest.mark.asyncio
async def test_dispatcher_retention_sweep_preserves_other_agents_history(tmp_path):
    """A shared backend must never let agent A purge agent B's ledger."""
    path = tmp_path / "tenant-retention.db"
    backend_a, agent_a, dispatcher_a = await _dispatcher(path, "did:agent:a")
    backend_b, agent_b, dispatcher_b = await _dispatcher(path, "did:agent:b")
    try:
        for agent, dispatcher in (
            (agent_a, dispatcher_a),
            (agent_b, dispatcher_b),
        ):
            await dispatcher.register_durable_consumer(
                DurableConsumerRegistration(
                    consumer_id="workflow-wait",
                    source="provider.message",
                    agent_id=agent.did,
                )
            )
            result = await dispatcher.dispatch_signal(
                _signal(agent_id=agent.did),
                source_event_id=f"expired-{agent.did}",
            )
            assert result.status is Status.OK
            delivery = await dispatcher.claim_durable_delivery(
                consumer_id="workflow-wait", executor_id="worker"
            )
            assert delivery is not None
            assert await dispatcher.ack_durable_delivery(
                consumer_id="workflow-wait",
                delivery_id=delivery.delivery_id,
                lease_token=delivery.lease_token,
            )

        past = datetime.now(timezone.utc) - timedelta(seconds=1)
        await backend_a.execute(
            "UPDATE durable_signal_events SET retention_until = ?",
            (past,),
        )

        assert await dispatcher_a.purge_expired_durable_deliveries() == 1
        remaining = await backend_b.fetch_all(
            "SELECT agent_id FROM durable_signal_events ORDER BY agent_id"
        )
        assert remaining == [(agent_b.did,)]
    finally:
        await _close(backend_a, agent_a)
        await _close(backend_b, agent_b)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("privacy_preset", "storage_marker"),
    (
        ("ephemeral", "none"),
        ("isolated", "temp"),
        ("deidentified", "deidentified"),
    ),
)
async def test_volatile_privacy_materializes_payload_correlated_wait_without_persisting_payload(
    tmp_path, privacy_preset, storage_marker
):
    """A volatile payload selector matches once, then replays only its safe row."""
    from kestrel_sovereign.privacy import get_privacy_preset

    path = tmp_path / f"volatile-wait-{privacy_preset}.db"
    agent_id = "did:agent:volatile"
    workflow_id = f"workflow-{privacy_preset}-42"
    secret = f"customer-{privacy_preset}@example.com"
    consumer = DurableConsumerRegistration(
        consumer_id="workflow-wait",
        source="provider.message",
        agent_id=agent_id,
        correlation_selector=f"payload.workflow={workflow_id}",
    )

    backend, agent, dispatcher = await _dispatcher(path, agent_id)
    agent.privacy_config = get_privacy_preset(privacy_preset)
    try:
        await dispatcher.register_durable_consumer(consumer)
        live_signal = _signal(
            agent_id=agent_id,
            message=f" {secret} ",
            workflow=workflow_id,
        )
        result = await dispatcher.dispatch_signal(
            live_signal,
            source_event_id=f"volatile-{privacy_preset}",
        )

        assert result.status is Status.OK
        # The durable selector, normal source handler, and an immediate durable
        # claim receive the normalized signal, not the privacy projection used
        # for the durable event row.
        assert agent.action_payloads == [
            {"message": secret, "workflow": workflow_id}
        ]
        live_delivery = await dispatcher.claim_durable_delivery(
            consumer_id=consumer.consumer_id,
            executor_id="live-workflow-runner",
        )
        assert live_delivery is not None
        assert live_delivery.event.payload == {
            "message": secret,
            "workflow": workflow_id,
        }
        # A non-terminal retry leaves the same process's live handoff intact;
        # after shutdown, only the durable marker can be replayed.
        retry = await dispatcher.nack_durable_delivery(
            consumer_id=consumer.consumer_id,
            delivery_id=live_delivery.delivery_id,
            lease_token=live_delivery.lease_token,
            error="simulate process restart before retry",
        )
        assert retry is not None and retry.status == RETRY
        assert live_delivery.delivery_id in dispatcher._transient_durable_handoffs
        assert [delivery.event_id for delivery in await dispatcher.list_durable_deliveries()] == [
            result.signal_id
        ]
        row = await backend.fetch_one(
            "SELECT payload FROM durable_signal_events WHERE agent_id = ?",
            (agent_id,),
        )
        assert row is not None
        assert json.loads(row[0]) == {"_privacy_gated": storage_marker}
        assert secret not in row[0]
        assert workflow_id not in row[0]
        dispatcher.shutdown()
        assert dispatcher._transient_durable_handoffs == {}
    finally:
        await _close(backend, agent)

    # Restart registration backfills from the projected row (which cannot
    # satisfy the payload selector), but the already-materialized delivery is
    # durable and therefore remains replayable exactly once.
    backend2, agent2, dispatcher2 = await _dispatcher(path, agent_id)
    agent2.privacy_config = get_privacy_preset(privacy_preset)
    try:
        await dispatcher2.register_durable_consumer(consumer)
        replayed = await dispatcher2.claim_durable_delivery(
            consumer_id=consumer.consumer_id,
            executor_id="workflow-runner",
        )
        assert replayed is not None
        assert replayed.event.payload == {"_privacy_gated": storage_marker}
        assert await dispatcher2.ack_durable_delivery(
            consumer_id=consumer.consumer_id,
            delivery_id=replayed.delivery_id,
            lease_token=replayed.lease_token,
        )
    finally:
        await _close(backend2, agent2)


@pytest.mark.asyncio
async def test_live_handoff_is_discarded_on_terminal_outcomes_and_lease_expiry(tmp_path):
    """Volatile payloads cannot outlive terminal or leased delivery state."""
    from kestrel_sovereign.privacy import get_privacy_preset

    backend, agent, dispatcher = await _dispatcher(tmp_path / "handoff-clear.db", "did:agent:one")
    agent.privacy_config = get_privacy_preset("ephemeral")
    consumer = DurableConsumerRegistration(
        consumer_id="workflow-wait",
        source="provider.message",
        agent_id=agent.did,
    )
    try:
        await dispatcher.register_durable_consumer(consumer)
        assert (await dispatcher.dispatch_signal(
            _signal(agent_id=agent.did, message="terminal-secret"),
            source_event_id="terminal-handoff",
        )).status is Status.OK
        acknowledged = await dispatcher.claim_durable_delivery(
            consumer_id=consumer.consumer_id,
            executor_id="live-workflow-runner",
        )
        assert acknowledged is not None
        assert acknowledged.delivery_id in dispatcher._transient_durable_handoffs
        assert await dispatcher.ack_durable_delivery(
            consumer_id=consumer.consumer_id,
            delivery_id=acknowledged.delivery_id,
            lease_token=acknowledged.lease_token,
        )
        assert acknowledged.delivery_id not in dispatcher._transient_durable_handoffs

        assert (await dispatcher.dispatch_signal(
            _signal(agent_id=agent.did, message="terminal-failure-secret"),
            source_event_id="terminal-failure-handoff",
        )).status is Status.OK
        terminal_failure = await dispatcher.claim_durable_delivery(
            consumer_id=consumer.consumer_id,
            executor_id="live-workflow-runner",
        )
        assert terminal_failure is not None
        assert terminal_failure.delivery_id in dispatcher._transient_durable_handoffs
        failed = await dispatcher.nack_durable_delivery(
            consumer_id=consumer.consumer_id,
            delivery_id=terminal_failure.delivery_id,
            lease_token=terminal_failure.lease_token,
            error="terminal worker failure",
            terminal=True,
        )
        assert failed is not None and failed.status == FAILED
        assert terminal_failure.delivery_id not in dispatcher._transient_durable_handoffs

        assert (await dispatcher.dispatch_signal(
            _signal(agent_id=agent.did, message="expired-secret"),
            source_event_id="expired-handoff",
        )).status is Status.OK
        expiring = await dispatcher.claim_durable_delivery(
            consumer_id=consumer.consumer_id,
            executor_id="live-workflow-runner",
        )
        assert expiring is not None
        handoff = dispatcher._transient_durable_handoffs[expiring.delivery_id]
        handoff.expires_at = datetime.now(timezone.utc) - timedelta(seconds=1)
        dispatcher._expire_transient_durable_handoff(
            expiring.delivery_id, handoff.expires_at
        )
        assert expiring.delivery_id not in dispatcher._transient_durable_handoffs
    finally:
        dispatcher.shutdown()
        await _close(backend, agent)


@pytest.mark.asyncio
async def test_privacy_transition_waits_for_projection_and_durable_commit(tmp_path):
    """A NORMAL projection cannot commit after an EPHEMERAL transition wins."""
    from kestrel_sovereign.privacy import get_privacy_preset

    backend, agent, dispatcher = await _dispatcher(tmp_path / "privacy-race.db", "did:agent:one")
    agent.privacy_config = get_privacy_preset("normal")
    secret = "normal-before-transition@example.com"
    persist_entered = asyncio.Event()
    allow_persist = asyncio.Event()
    persist_committed = asyncio.Event()
    original_execute = backend.execute

    async def stall_event_insert(query, params=()):
        if "INSERT OR IGNORE INTO durable_signal_events" in query:
            persist_entered.set()
            await allow_persist.wait()
            result = await original_execute(query, params)
            persist_committed.set()
            return result
        return await original_execute(query, params)

    backend.execute = stall_event_insert

    async def transition_to_ephemeral():
        async with agent._get_privacy_transition_lock():
            # This point is reachable only after the dispatch's projection and
            # durable INSERT completed under the same transition lock.
            assert persist_committed.is_set()
            agent.privacy_config = get_privacy_preset("ephemeral")

    try:
        dispatch_task = asyncio.create_task(
            dispatcher.dispatch_signal(
                _signal(agent_id=agent.did, message=secret),
                source_event_id="normal-to-ephemeral-race",
            )
        )
        await asyncio.wait_for(persist_entered.wait(), timeout=2)
        transition_task = asyncio.create_task(transition_to_ephemeral())
        await asyncio.sleep(0)
        assert not transition_task.done()

        allow_persist.set()
        result = await dispatch_task
        await transition_task
        assert result.status is Status.OK
        assert agent.privacy_config.is_ephemeral()

        row = await backend.fetch_one(
            "SELECT payload FROM durable_signal_events WHERE agent_id = ?",
            (agent.did,),
        )
        assert row is not None
        # The event was committed while the old NORMAL policy still held the
        # transition lock; the mode change was not allowed to overtake it.
        assert secret in row[0]
    finally:
        backend.execute = original_execute
        dispatcher.shutdown()
        await _close(backend, agent)


@pytest.mark.asyncio
async def test_volatile_payload_selector_is_scoped_to_its_shared_database_tenant(tmp_path):
    """Transient matching cannot materialize another tenant's durable wait."""
    from kestrel_sovereign.privacy import get_privacy_preset

    path = tmp_path / "volatile-tenant-scope.db"
    backend_a, agent_a, dispatcher_a = await _dispatcher(path, "did:agent:a")
    backend_b, agent_b, dispatcher_b = await _dispatcher(path, "did:agent:b")
    agent_a.privacy_config = get_privacy_preset("ephemeral")
    agent_b.privacy_config = get_privacy_preset("ephemeral")
    workflow_id = "shared-workflow-42"
    secret = "tenant-a-customer@example.com"
    try:
        for agent, dispatcher in (
            (agent_a, dispatcher_a),
            (agent_b, dispatcher_b),
        ):
            await dispatcher.register_durable_consumer(
                DurableConsumerRegistration(
                    consumer_id="workflow-wait",
                    source="provider.message",
                    agent_id=agent.did,
                    correlation_selector=f"payload.workflow={workflow_id}",
                )
            )

        # A foreign envelope target does not change the owner of the durable
        # scope. The payload match belongs only to dispatcher A's tenant.
        result = await dispatcher_a.dispatch_signal(
            _signal(
                agent_id=agent_b.did,
                message=secret,
                workflow=workflow_id,
            ),
            source_event_id="tenant-a-volatile-event",
        )
        assert result.status is Status.OK
        assert [delivery.event_id for delivery in await dispatcher_a.list_durable_deliveries()] == [
            result.signal_id
        ]
        assert await dispatcher_b.claim_durable_delivery(
            consumer_id="workflow-wait", executor_id="tenant-b-runner"
        ) is None

        rows = await backend_a.fetch_all(
            "SELECT agent_id, payload FROM durable_signal_events ORDER BY agent_id"
        )
        assert rows == [(agent_a.did, '{"_privacy_gated": "none"}')]
        assert secret not in rows[0][1]
        assert workflow_id not in rows[0][1]
    finally:
        await _close(backend_a, agent_a)
        await _close(backend_b, agent_b)


@pytest.mark.asyncio
async def test_anonymous_selector_uses_the_persisted_redacted_payload_before_and_after_restart(
    tmp_path,
):
    """ANONYMOUS selector matching is identical for initial and replayed delivery."""
    from kestrel_sovereign.privacy import get_privacy_preset

    path = tmp_path / "anonymous-selector.db"
    agent_id = "did:agent:anonymous"
    secret = "customer@example.com"
    consumer = DurableConsumerRegistration(
        consumer_id="workflow-wait",
        source="provider.message",
        agent_id=agent_id,
        correlation_selector="payload.message=[EMAIL_REDACTED]",
    )

    backend, agent, dispatcher = await _dispatcher(path, agent_id)
    agent.privacy_config = get_privacy_preset("anonymous")
    try:
        await dispatcher.register_durable_consumer(consumer)
        result = await dispatcher.dispatch_signal(
            _signal(agent_id=agent_id, message=secret),
            source_event_id="anonymous-selector-event",
        )

        assert result.status is Status.OK
        # The selector is evaluated against the anonymized event during the
        # persistence transaction, so its delivery exists before any claim.
        deliveries = await dispatcher.list_durable_deliveries()
        assert len(deliveries) == 1
        assert deliveries[0].event_id == result.signal_id
        assert deliveries[0].event.payload == {
            "message": "[EMAIL_REDACTED]",
            "workflow": "wf-1",
        }
        row = await backend.fetch_one(
            "SELECT payload FROM durable_signal_events WHERE agent_id = ?",
            (agent_id,),
        )
        assert row is not None
        assert secret not in row[0]
        assert json.loads(row[0]) == deliveries[0].event.payload
    finally:
        await _close(backend, agent)

    # Registering after a restart sees the same redacted payload.  It must not
    # backfill a second delivery or change the event a worker claims.
    backend2, agent2, dispatcher2 = await _dispatcher(path, agent_id)
    agent2.privacy_config = get_privacy_preset("anonymous")
    try:
        await dispatcher2.register_durable_consumer(consumer)
        assert len(await dispatcher2.list_durable_deliveries()) == 1
        replayed = await dispatcher2.claim_durable_delivery(
            consumer_id=consumer.consumer_id,
            executor_id="workflow-runner",
        )
        assert replayed is not None
        assert replayed.event.payload == {
            "message": "[EMAIL_REDACTED]",
            "workflow": "wf-1",
        }
    finally:
        await _close(backend2, agent2)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("privacy_preset", "expected_storage", "expects_anonymization"),
    (
        ("ephemeral", "none", False),
        ("isolated", "temp", False),
        ("deidentified", "deidentified", False),
        ("anonymous", None, True),
        ("normal", None, False),
        ("public", None, False),
    ),
)
async def test_channel_signal_durable_payload_honors_agent_privacy_contract(
    tmp_path, privacy_preset, expected_storage, expects_anonymization
):
    """The real dispatcher must not bypass channel privacy at its DB boundary."""
    from kestrel_sovereign.privacy import get_privacy_preset

    backend = SQLiteBackend(str(tmp_path / f"privacy-{privacy_preset}.db"))
    await backend.connect()
    agent = _Agent("did:agent:privacy")
    agent.privacy_config = get_privacy_preset(privacy_preset)
    store = SignalLogStore(backend)
    await store.initialize()
    registry = SourceRegistry()
    registry.register(build_channel_message_registration())
    dispatcher = SignalDispatcher(
        agent=agent,
        registry=registry,
        lock_manager=OrderedLockManager(),
        store=store,
    )
    try:
        result = await dispatcher.dispatch_signal(
            Signal(
                source="channel.message",
                kind="inbound",
                mode=SignalMode.COGNITION,
                payload={
                    "message_id": "channel-event-1",
                    "channel_type": "telegram",
                    "sender": "alice@example.com",
                    "recipient": "bot",
                    "content": "Contact alice@example.com for the secret",
                    "metadata": {"reply_to": "alice@example.com"},
                },
                target_agent=agent.did,
            ),
            source_event_id="channel-event-1",
        )
        assert result.status is Status.OK
        row = await backend.fetch_one(
            "SELECT payload FROM durable_signal_events WHERE agent_id = ?",
            (agent.did,),
        )
        assert row is not None
        persisted_payload = json.loads(row[0])
        if expected_storage is not None:
            assert persisted_payload == {"_privacy_gated": expected_storage}
            assert "alice@example.com" not in row[0]
        elif expects_anonymization:
            assert "alice@example.com" not in row[0]
            assert "[EMAIL_REDACTED]" in row[0]
        else:
            assert persisted_payload["content"] == "Contact alice@example.com for the secret"
            assert persisted_payload["sender"] == "alice@example.com"
    finally:
        await _close(backend, agent)


@pytest.mark.asyncio
async def test_postgres_registration_handoff_uses_a_transaction_scoped_scope_lock():
    """Pin the PostgreSQL primitive; concurrency belongs to integration tests."""

    class _PostgresBackend:
        backend_type = "postgres"
        fetch_val = AsyncMock(return_value=None)

    store = DurableSignalStore(_PostgresBackend())
    await store._lock_scope_handoff(
        agent_id="did:agent:one", source="provider.message"
    )

    _PostgresBackend.fetch_val.assert_awaited_once_with(
        "SELECT pg_advisory_xact_lock(hashtextextended(?, 0))",
        ("durable-signal:did:agent:one:provider.message",),
    )


@pytest.mark.asyncio
async def test_sqlite_registration_handoff_reserves_the_writer_slot_before_reads():
    """Pin the SQLite primitive used by the cross-instance integration race."""

    class _SQLiteBackend:
        backend_type = "sqlite"
        execute = AsyncMock(return_value=0)

    store = DurableSignalStore(_SQLiteBackend())
    await store._lock_scope_handoff(
        agent_id="did:agent:one", source="provider.message"
    )

    _SQLiteBackend.execute.assert_awaited_once_with(
        "DELETE FROM durable_signal_consumers WHERE 0"
    )
