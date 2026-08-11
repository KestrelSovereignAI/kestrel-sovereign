"""Durable signal consumer contract: replay, scoped leasing, and retention."""

from __future__ import annotations

import asyncio
import json
import time
from contextlib import asynccontextmanager, suppress
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from kestrel_sdk.signals import (
    RateLimit,
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
    TERMINAL_ACKABLE,
    DurableAdmissionDisposition,
    DurableConsumerRegistration,
    DurableSignalStore,
    OrderedLockManager,
    SignalDispatcher,
    SignalLogStore,
    SourceRegistry,
)
from kestrel_sovereign.signals import dispatcher as dispatcher_module
from kestrel_sovereign.signals.sources.channels import (
    DURABLE_COGNITION_CONSUMER_ID,
    DURABLE_COGNITION_MARKER,
    DURABLE_COGNITION_MARKER_VALUE,
    DURABLE_TERMINAL_CONSUMER_ID,
    DURABLE_TERMINAL_MARKER,
    DURABLE_TERMINAL_MARKER_VALUE,
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

    async def process_input(self, prompt: str):
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


def _channel_signal(agent_id: str, message_id: str) -> Signal:
    return Signal(
        source="channel.message",
        kind="inbound",
        mode=SignalMode.COGNITION,
        payload={
            DURABLE_COGNITION_MARKER: DURABLE_COGNITION_MARKER_VALUE,
            "message_id": message_id,
            "channel_type": "telegram",
            "sender": "555",
            "recipient": "42",
            "content": f"message {message_id}",
            "metadata": {},
        },
        target_agent=agent_id,
        caller="555",
        dedupe_key=f"telegram:{message_id}",
    )


async def _channel_dispatcher(path, did: str, *, rate_limit: RateLimit | None = None):
    backend = SQLiteBackend(str(path))
    await backend.connect()
    store = SignalLogStore(backend)
    await store.initialize()
    agent = _Agent(did)
    registry = SourceRegistry()
    registration = build_channel_message_registration()
    if rate_limit is not None:
        registration = replace(registration, rate_limit=rate_limit)
    registry.register(registration)
    dispatcher = SignalDispatcher(
        agent=agent,
        registry=registry,
        lock_manager=OrderedLockManager(),
        store=store,
    )
    await dispatcher.initialize_durable_delivery()
    return backend, agent, dispatcher


async def _assert_sync_shutdown_drained(
    backend: SQLiteBackend,
    agent: _Agent,
    dispatcher: SignalDispatcher,
    completion: asyncio.Task[None],
) -> None:
    """Verify the sync seam reached the normal durable teardown terminal state."""
    from tests.utils.aiosqlite_workers import aiosqlite_worker

    await asyncio.wait_for(completion, timeout=1.0)
    owner = await backend.fetch_one(
        "SELECT stopped_at FROM durable_signal_runtime_owners "
        "WHERE agent_id = ? AND owner_id = ?",
        (agent.did, dispatcher._durable_delivery_owner),
    )
    assert owner is not None and owner[0] is not None
    assert dispatcher._durable_runtime_owner_registered is False
    assert dispatcher._durable_runtime_owner_registration_started is False
    assert dispatcher._durable_active_admissions == 0
    assert dispatcher._durable_admissions_drained.is_set()
    assert dispatcher._transient_durable_handoffs == {}
    assert dispatcher._transient_durable_handoff_timers == {}
    assert dispatcher._post_commit_reservation_repairs == set()
    assert dispatcher._runtime_owner_heartbeat_timer is None
    assert dispatcher._runtime_owner_heartbeat_task is None
    assert all(task.done() for task in agent.tasks)
    assert await backend.fetch_val(
        "SELECT COUNT(*) FROM durable_signal_deliveries "
        "WHERE status = 'initial_reserved'"
    ) == 0

    connection = backend._connection
    assert connection is not None
    worker = aiosqlite_worker(connection)
    await backend.close()
    assert not worker.is_alive()


async def _finish_sync_shutdown_test(
    backend: SQLiteBackend,
    agent: _Agent,
    dispatcher: SignalDispatcher,
) -> None:
    """Join a sync-started teardown before an assertion failure closes SQLite."""
    if dispatcher._durable_shutdown_completion is not None:
        await dispatcher.shutdown_durable_delivery()
    elif not dispatcher._durable_shutdown:
        await dispatcher.shutdown_durable_delivery()
    if backend._connection is not None:
        await _close(backend, agent)


async def _assert_late_durable_calls_fail_safely(
    dispatcher: SignalDispatcher,
    *,
    agent: _Agent,
    consumer: DurableConsumerRegistration,
) -> None:
    """Assert every public durable entry point is closed before store access."""
    with pytest.raises(RuntimeError, match="shutting down"):
        await dispatcher.initialize_durable_delivery()
    with pytest.raises(RuntimeError, match="shutting down"):
        await dispatcher.register_durable_consumer(consumer)
    with pytest.raises(RuntimeError, match="shutting down"):
        await dispatcher.claim_durable_delivery(
            consumer_id=consumer.consumer_id, executor_id="late-worker"
        )
    with pytest.raises(RuntimeError, match="shutting down"):
        await dispatcher.ack_durable_delivery(
            consumer_id=consumer.consumer_id,
            delivery_id="late-delivery",
            lease_token="late-token",
        )
    with pytest.raises(RuntimeError, match="shutting down"):
        await dispatcher.nack_durable_delivery(
            consumer_id=consumer.consumer_id,
            delivery_id="late-delivery",
            lease_token="late-token",
            error="late",
        )
    with pytest.raises(RuntimeError, match="shutting down"):
        await dispatcher.list_durable_deliveries()
    with pytest.raises(RuntimeError, match="shutting down"):
        await dispatcher.purge_expired_durable_deliveries()

    result = await dispatcher.dispatch_signal(
        _signal(agent_id=agent.did, message="late-dispatch")
    )
    assert result.status is Status.FAILED
    assert result.error == "Durable signal delivery is shutting down"

    handle = await dispatcher.enqueue_signal(
        _signal(agent_id=agent.did, message="late-enqueue")
    )
    enqueued = await handle.task
    assert enqueued.status is Status.FAILED
    assert enqueued.error == "Durable signal delivery is shutting down"


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
async def test_cursor_owned_cognition_rate_limit_keeps_durable_retry_after_admission(tmp_path):
    """A rate limit cannot turn committed Telegram work into a duplicate loss."""
    backend, agent, dispatcher = await _channel_dispatcher(
        tmp_path / "cursor-rate-limit.db",
        "did:agent:one",
        rate_limit=RateLimit(per_minute=1, per_hour=300),
    )
    consumer = DurableConsumerRegistration(
        consumer_id="core.channel-cognition-v1",
        source="channel.message",
        agent_id=agent.did,
        correlation_selector=(
            f"payload.{DURABLE_COGNITION_MARKER}="
            f"{DURABLE_COGNITION_MARKER_VALUE}"
        ),
        max_attempts=0,
    )
    try:
        await dispatcher.register_durable_consumer(consumer)

        first = await dispatcher.enqueue_durable_cognition(
            _channel_signal(agent.did, "first"),
            source_event_id="telegram:update:first",
            consumer_id=consumer.consumer_id,
        )
        assert (
            await first.wait_for_durable_admission()
        ).disposition is DurableAdmissionDisposition.COMMITTED
        assert (await first.wait()).status is Status.OK

        limited = await dispatcher.enqueue_durable_cognition(
            _channel_signal(agent.did, "limited"),
            source_event_id="telegram:update:limited",
            consumer_id=consumer.consumer_id,
        )
        assert (
            await limited.wait_for_durable_admission()
        ).disposition is DurableAdmissionDisposition.COMMITTED
        assert (await limited.wait()).status is Status.DROPPED_RATE_LIMIT
        limited_delivery = next(
            delivery
            for delivery in await dispatcher.list_durable_deliveries(
                consumer_id=consumer.consumer_id
            )
            if delivery.event.source_event_id == "telegram:update:limited"
        )
        assert limited_delivery.status == RETRY
        assert limited_delivery.max_attempts == 0

        # Reset test-only policy windows rather than sleeping, then drive the
        # retained delivery again. The provider had already received a durable
        # admission receipt; this duplicate only wakes the durable executor.
        dispatcher._rate.reset()
        dispatcher._coalescing.reset()
        retried = await dispatcher.enqueue_durable_cognition(
            _channel_signal(agent.did, "limited"),
            source_event_id="telegram:update:limited",
            consumer_id=consumer.consumer_id,
        )
        assert (
            await retried.wait_for_durable_admission()
        ).disposition is DurableAdmissionDisposition.DUPLICATE
        assert (await retried.wait()).status is Status.OK
        completed = next(
            delivery
            for delivery in await dispatcher.list_durable_deliveries(
                consumer_id=consumer.consumer_id
            )
            if delivery.event.source_event_id == "telegram:update:limited"
        )
        assert completed.status == ACKNOWLEDGED
    finally:
        await dispatcher.shutdown_durable_delivery()
        await _close(backend, agent)


@pytest.mark.asyncio
async def test_cursor_owned_admission_resolves_before_a_hung_cognition_turn(tmp_path):
    """A Telegram cursor ACK never waits for the full durable cognition turn."""

    backend, agent, dispatcher = await _channel_dispatcher(
        tmp_path / "cursor-immediate-admission.db", "did:agent:one"
    )
    consumer = DurableConsumerRegistration(
        consumer_id=DURABLE_COGNITION_CONSUMER_ID,
        source="channel.message",
        agent_id=agent.did,
        correlation_selector=(
            f"payload.{DURABLE_COGNITION_MARKER}="
            f"{DURABLE_COGNITION_MARKER_VALUE}"
        ),
        max_attempts=0,
    )
    cognition_started = asyncio.Event()
    release_cognition = asyncio.Event()

    async def hung_cognition(_prompt: str):
        cognition_started.set()
        await release_cognition.wait()
        return "eventually processed"

    agent.process_input = hung_cognition
    try:
        await dispatcher.register_durable_consumer(consumer)
        handle = await dispatcher.enqueue_durable_cognition(
            _channel_signal(agent.did, "hung-turn"),
            source_event_id="telegram:update:hung-turn",
            consumer_id=consumer.consumer_id,
        )

        receipt = await asyncio.wait_for(handle.wait_for_durable_admission(), timeout=0.1)
        assert receipt.disposition is DurableAdmissionDisposition.COMMITTED
        await asyncio.wait_for(cognition_started.wait(), timeout=1)
        assert handle.task.done() is False
        delivery = (await dispatcher.list_durable_deliveries())[0]
        assert delivery.status == LEASED

        release_cognition.set()
        assert (await handle.wait()).status is Status.OK
        delivery = (await dispatcher.list_durable_deliveries())[0]
        assert delivery.status == ACKNOWLEDGED
    finally:
        release_cognition.set()
        await dispatcher.shutdown_durable_delivery()
        await _close(backend, agent)


@pytest.mark.asyncio
@pytest.mark.parametrize("privacy_preset", ("ephemeral", "isolated", "deidentified"))
async def test_privacy_elided_first_cognition_delivery_is_claimed_and_acked(
    tmp_path, privacy_preset
):
    """The local transient reservation transfers to the first cursor callback."""
    from kestrel_sovereign.privacy import get_privacy_preset

    backend, agent, dispatcher = await _channel_dispatcher(
        tmp_path / f"privacy-first-claim-{privacy_preset}.db", "did:agent:one"
    )
    agent.privacy_config = get_privacy_preset(privacy_preset)
    consumer = DurableConsumerRegistration(
        consumer_id="core.channel-cognition-v1",
        source="channel.message",
        agent_id=agent.did,
        correlation_selector=(
            f"payload.{DURABLE_COGNITION_MARKER}="
            f"{DURABLE_COGNITION_MARKER_VALUE}"
        ),
        max_attempts=0,
    )
    source_event_id = f"telegram:update:privacy-first:{privacy_preset}"
    try:
        await dispatcher.register_durable_consumer(consumer)
        handle = await dispatcher.enqueue_durable_cognition(
            _channel_signal(agent.did, f"privacy-{privacy_preset}"),
            source_event_id=source_event_id,
            consumer_id=consumer.consumer_id,
        )

        assert (
            await handle.wait_for_durable_admission()
        ).disposition is DurableAdmissionDisposition.COMMITTED
        assert (await handle.wait()).status is Status.OK
        delivery = (await dispatcher.list_durable_deliveries())[0]
        assert delivery.status == ACKNOWLEDGED
        assert delivery.event.payload in (
            {"_privacy_gated": "none"},
            {"_privacy_gated": "temp"},
            {"_privacy_gated": "deidentified"},
        )
        row = await backend.fetch_one(
            "SELECT payload FROM durable_signal_events WHERE event_id = ?",
            (delivery.event.event_id,),
        )
        assert row is not None
        assert "message privacy-" not in row[0]
    finally:
        await dispatcher.shutdown_durable_delivery()
        await _close(backend, agent)


@pytest.mark.asyncio
async def test_cognition_retry_uses_canonical_persisted_channel_input(tmp_path):
    """A changed provider duplicate cannot replace stored cognition content."""
    backend, agent, dispatcher = await _channel_dispatcher(
        tmp_path / "canonical-cognition-retry.db", "did:agent:one"
    )
    consumer = DurableConsumerRegistration(
        consumer_id="core.channel-cognition-v1",
        source="channel.message",
        agent_id=agent.did,
        correlation_selector=(
            f"payload.{DURABLE_COGNITION_MARKER}="
            f"{DURABLE_COGNITION_MARKER_VALUE}"
        ),
        max_attempts=0,
    )
    prompts: list[str] = []

    async def fail_once_then_record(prompt: str):
        prompts.append(prompt)
        if len(prompts) == 1:
            raise RuntimeError("temporary cognition failure")
        return "ok"

    agent.process_input = fail_once_then_record
    original = _channel_signal(agent.did, "canonical")
    original.payload["content"] = "original Telegram content"
    changed = _channel_signal(agent.did, "canonical")
    changed.payload["content"] = "attacker replacement content"
    source_event_id = "telegram:update:canonical"
    try:
        await dispatcher.register_durable_consumer(consumer)
        first = await dispatcher.enqueue_durable_cognition(
            original,
            source_event_id=source_event_id,
            consumer_id=consumer.consumer_id,
        )
        first_result = await first.wait()
        assert first_result.status is Status.FAILED
        assert (await dispatcher.list_durable_deliveries())[0].status == RETRY

        # The first failed turn recorded the normal coalescing key. A provider
        # retry is allowed to retry its durable delivery, so reset only this
        # in-memory policy window for the regression's second callback.
        dispatcher._coalescing.reset()
        retry = await dispatcher.enqueue_durable_cognition(
            changed,
            source_event_id=source_event_id,
            consumer_id=consumer.consumer_id,
        )
        retry_result = await retry.wait()
        assert retry_result.status is Status.OK
        # A delivery's source event remains durable identity, but a provider
        # retry is a distinct dispatch/outcome with the same ID its public
        # handle promised to the callback owner.
        assert retry_result.signal_id == retry.signal_id
        assert retry_result.signal_id != first_result.signal_id
        deliveries = await dispatcher.list_durable_deliveries()
        assert deliveries[0].event.event_id == first.signal_id
        assert deliveries[0].event.source_event_id == source_event_id
        outcome_ids = {
            row[0]
            for row in await backend.fetch_all(
                "SELECT id FROM signal_log WHERE source = 'channel.message'"
            )
        }
        assert {first_result.signal_id, retry_result.signal_id}.issubset(outcome_ids)
        assert len(prompts) == 2
        assert "original Telegram content" in prompts[1]
        assert "attacker replacement content" not in prompts[1]
        assert (await dispatcher.list_durable_deliveries())[0].status == ACKNOWLEDGED
    finally:
        await dispatcher.shutdown_durable_delivery()
        await _close(backend, agent)


@pytest.mark.asyncio
async def test_terminal_channel_noop_redelivery_remains_provider_ackable_after_lost_ack(
    tmp_path, monkeypatch
):
    """A terminal no-op is a durable receipt; a normal FAILED delivery is not."""

    backend, agent, dispatcher = await _channel_dispatcher(
        tmp_path / "terminal-channel-redelivery.db", "did:agent:one"
    )
    consumer = DurableConsumerRegistration(
        consumer_id=DURABLE_COGNITION_CONSUMER_ID,
        source="channel.message",
        agent_id=agent.did,
        correlation_selector=(
            f"payload.{DURABLE_COGNITION_MARKER}="
            f"{DURABLE_COGNITION_MARKER_VALUE}"
        ),
        max_attempts=0,
    )

    async def terminal_noop(signal, _registration, start):
        return dispatcher._failure_result(
            signal,
            start,
            error="terminal validation refusal",
            status=Status.DROPPED_VALIDATION,
        )

    monkeypatch.setattr(dispatcher, "_route_after_durable_persistence", terminal_noop)
    terminal_source_event = "telegram:update:terminal-noop"
    try:
        await dispatcher.register_durable_consumer(consumer)
        first = await dispatcher.enqueue_durable_cognition(
            _channel_signal(agent.did, "terminal-noop"),
            source_event_id=terminal_source_event,
            consumer_id=consumer.consumer_id,
        )
        assert (await first.wait()).status is Status.DROPPED_VALIDATION
        assert (
            await first.wait_for_durable_admission()
        ).disposition is DurableAdmissionDisposition.COMMITTED
        stored = await dispatcher.get_durable_delivery_for_event(
            consumer_id=consumer.consumer_id,
            event_id=first.signal_id,
        )
        assert stored is not None and stored.status == TERMINAL_ACKABLE

        # Simulate a provider ACK lost after Core has reached the proven
        # terminal no-op. The source replays the identical provider identity.
        redelivery = await dispatcher.enqueue_durable_cognition(
            _channel_signal(agent.did, "terminal-noop"),
            source_event_id=terminal_source_event,
            consumer_id=consumer.consumer_id,
        )
        assert (
            await redelivery.wait_for_durable_admission()
        ).disposition is DurableAdmissionDisposition.TERMINAL
        assert (await redelivery.wait()).status is Status.COALESCED
        replayed = await dispatcher.get_durable_delivery_for_event(
            consumer_id=consumer.consumer_id,
            event_id=first.signal_id,
        )
        assert replayed is not None and replayed.status == TERMINAL_ACKABLE

        # A caller marking an ordinary worker failure terminal must still not
        # advance a provider cursor when it redelivers the same source event.
        ordinary = _channel_signal(agent.did, "ordinary-terminal-failure")
        persisted = await dispatcher._durable_store.persist_signal(
            ordinary,
            agent_id=agent.did,
            source_event_id="telegram:update:ordinary-terminal-failure",
            retention_days=7,
        )
        claimed = await dispatcher.claim_durable_delivery_for_event(
            consumer_id=consumer.consumer_id,
            event_id=persisted.event_id,
            executor_id=dispatcher._durable_delivery_owner,
        )
        assert claimed is not None
        failed = await dispatcher.nack_durable_delivery(
            consumer_id=consumer.consumer_id,
            delivery_id=claimed.delivery_id,
            lease_token=claimed.lease_token or "",
            error="ordinary terminal worker failure",
            terminal=True,
        )
        assert failed is not None and failed.status == FAILED

        ordinary_redelivery = await dispatcher.enqueue_durable_cognition(
            _channel_signal(agent.did, "ordinary-terminal-failure"),
            source_event_id="telegram:update:ordinary-terminal-failure",
            consumer_id=consumer.consumer_id,
        )
        assert (
            await ordinary_redelivery.wait_for_durable_admission()
        ).disposition is DurableAdmissionDisposition.NOT_ADMITTED
        assert (await ordinary_redelivery.wait()).status is Status.FAILED

        # A deduplicated source event with its selected row absent has no
        # durable retry owner at all. It must be just as non-ACKable as an
        # ordinary FAILED row.
        missing = _channel_signal(agent.did, "selected-delivery-missing")
        missing_persistence = await dispatcher._durable_store.persist_signal(
            missing,
            agent_id=agent.did,
            source_event_id="telegram:update:selected-delivery-missing",
            retention_days=7,
        )
        await backend.execute(
            "DELETE FROM durable_signal_deliveries "
            "WHERE agent_id = ? AND consumer_id = ? AND event_id = ?",
            (agent.did, consumer.consumer_id, missing_persistence.event_id),
        )
        missing_redelivery = await dispatcher.enqueue_durable_cognition(
            _channel_signal(agent.did, "selected-delivery-missing"),
            source_event_id="telegram:update:selected-delivery-missing",
            consumer_id=consumer.consumer_id,
        )
        assert (
            await missing_redelivery.wait_for_durable_admission()
        ).disposition is DurableAdmissionDisposition.NOT_ADMITTED
        assert (await missing_redelivery.wait()).status is Status.FAILED
    finally:
        await dispatcher.shutdown_durable_delivery()
        await _close(backend, agent)


@pytest.mark.asyncio
@pytest.mark.parametrize("max_attempts", (0, 1), ids=("unlimited", "exhausted"))
async def test_expired_terminal_nack_uses_managed_token_before_provider_receipt(
    tmp_path, monkeypatch, max_attempts
):
    """A late terminal NACK is ACKable only when its fallback row proves it."""

    backend, agent, dispatcher = await _channel_dispatcher(
        tmp_path / f"expired-terminal-nack-{max_attempts}.db", "did:agent:one"
    )
    consumer = DurableConsumerRegistration(
        consumer_id=DURABLE_COGNITION_CONSUMER_ID,
        source="channel.message",
        agent_id=agent.did,
        correlation_selector=(
            f"payload.{DURABLE_COGNITION_MARKER}="
            f"{DURABLE_COGNITION_MARKER_VALUE}"
        ),
        max_attempts=max_attempts,
    )

    async def terminal_noop(signal, _registration, start):
        return dispatcher._failure_result(
            signal,
            start,
            error="terminal validation refusal after lease expiry",
            status=Status.DROPPED_VALIDATION,
        )

    async def expired_nack(**kwargs):
        # ``nack_delivery`` returns None for an expired exact lease. The
        # managed-owner fallback below is the only permitted late transition.
        return await dispatcher._durable_store.nack_delivery(
            agent_id=agent.did,
            now=datetime.now(timezone.utc) + timedelta(days=1),
            **kwargs,
        )

    monkeypatch.setattr(dispatcher, "_route_after_durable_persistence", terminal_noop)
    monkeypatch.setattr(dispatcher, "nack_durable_delivery", expired_nack)
    source_event_id = f"telegram:update:expired-terminal:{max_attempts}"
    try:
        await dispatcher.register_durable_consumer(consumer)
        first = await dispatcher.enqueue_durable_cognition(
            _channel_signal(agent.did, "terminal"),
            source_event_id=source_event_id,
            consumer_id=consumer.consumer_id,
        )
        assert (await first.wait()).status is Status.DROPPED_VALIDATION
        assert (
            await first.wait_for_durable_admission()
        ).disposition is DurableAdmissionDisposition.COMMITTED
        stored = await dispatcher.get_durable_delivery_for_event(
            consumer_id=consumer.consumer_id,
            event_id=first.signal_id,
        )
        assert stored is not None and stored.status == TERMINAL_ACKABLE

        # A provider that lost the receipt sees the durable terminal row, not
        # a fresh cognition execution.
        redelivery = await dispatcher.enqueue_durable_cognition(
            _channel_signal(agent.did, "terminal"),
            source_event_id=source_event_id,
            consumer_id=consumer.consumer_id,
        )
        assert (
            await redelivery.wait_for_durable_admission()
        ).disposition is DurableAdmissionDisposition.TERMINAL

        async def ordinary_failure(signal, _registration, start):
            return dispatcher._failure_result(
                signal,
                start,
                error="ordinary cognition failure after lease expiry",
                status=Status.FAILED,
            )

        monkeypatch.setattr(
            dispatcher, "_route_after_durable_persistence", ordinary_failure
        )
        ordinary = await dispatcher.enqueue_durable_cognition(
            _channel_signal(agent.did, "ordinary"),
            source_event_id=f"{source_event_id}:ordinary",
            consumer_id=consumer.consumer_id,
        )
        assert (await ordinary.wait()).status is Status.FAILED
        assert (
            await ordinary.wait_for_durable_admission()
        ).disposition is DurableAdmissionDisposition.COMMITTED
        ordinary_delivery = await dispatcher.get_durable_delivery_for_event(
            consumer_id=consumer.consumer_id,
            event_id=ordinary.signal_id,
        )
        assert ordinary_delivery is not None
        assert ordinary_delivery.status == (RETRY if max_attempts == 0 else FAILED)
    finally:
        await dispatcher.shutdown_durable_delivery()
        await _close(backend, agent)


@pytest.mark.asyncio
async def test_legacy_channel_redelivery_upgrades_only_matching_normal_event(tmp_path):
    """One matching retry repairs an origin/main row without bulk backfill."""

    backend, agent, dispatcher = await _channel_dispatcher(
        tmp_path / "legacy-channel-redelivery.db", "did:agent:one"
    )
    consumer = DurableConsumerRegistration(
        consumer_id=DURABLE_COGNITION_CONSUMER_ID,
        source="channel.message",
        agent_id=agent.did,
        correlation_selector=(
            f"payload.{DURABLE_COGNITION_MARKER}="
            f"{DURABLE_COGNITION_MARKER_VALUE}"
        ),
        max_attempts=0,
    )
    source_event_id = "telegram:update:origin-main"
    # This is the retained normal ledger shape from before the cognition
    # marker/consumer: no marker, no delivery, and no protected caller field.
    legacy = _channel_signal(agent.did, "origin-main")
    legacy.payload.pop(DURABLE_COGNITION_MARKER)
    attempts = 0

    async def fail_once_then_succeed(_prompt: str):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("retry the upgraded delivery")
        return "ok"

    agent.process_input = fail_once_then_succeed
    try:
        persisted = await dispatcher._durable_store.persist_signal(
            legacy,
            agent_id=agent.did,
            source_event_id=source_event_id,
            retention_days=14,
        )
        assert persisted.created is True
        await dispatcher.register_durable_consumer(consumer)
        assert await dispatcher.list_durable_deliveries() == []

        mismatched = _channel_signal(agent.did, "origin-main")
        mismatched.payload["content"] = "not the retained canonical message"
        rejected = await dispatcher.enqueue_durable_cognition(
            mismatched,
            source_event_id=source_event_id,
            consumer_id=consumer.consumer_id,
        )
        assert (await rejected.wait()).status is Status.FAILED
        # The legacy upgrade returned False for this non-canonical redelivery.
        # With no selected delivery to prove retry ownership, the provider
        # cursor must remain unchanged rather than treating event dedupe as an
        # ACK receipt.
        assert (
            await rejected.wait_for_durable_admission()
        ).disposition is DurableAdmissionDisposition.NOT_ADMITTED
        assert await dispatcher.list_durable_deliveries() == []
        row = await backend.fetch_one(
            "SELECT caller_identity FROM durable_signal_events WHERE event_id = ?",
            (persisted.event_id,),
        )
        assert row == (None,)

        first = await dispatcher.enqueue_durable_cognition(
            _channel_signal(agent.did, "origin-main"),
            source_event_id=source_event_id,
            consumer_id=consumer.consumer_id,
        )
        assert (await first.wait()).status is Status.FAILED
        delivery = (await dispatcher.list_durable_deliveries())[0]
        assert delivery.event.event_id == persisted.event_id
        assert delivery.status == RETRY
        row = await backend.fetch_one(
            "SELECT caller_identity FROM durable_signal_events WHERE event_id = ?",
            (persisted.event_id,),
        )
        assert row is not None and row[0] is not None and "555" not in row[0]

        dispatcher._coalescing.reset()
        retry = await dispatcher.enqueue_durable_cognition(
            _channel_signal(agent.did, "origin-main"),
            source_event_id=source_event_id,
            consumer_id=consumer.consumer_id,
        )
        assert (await retry.wait()).status is Status.OK
        assert attempts == 2
        assert (await dispatcher.list_durable_deliveries())[0].status == ACKNOWLEDGED
    finally:
        await dispatcher.shutdown_durable_delivery()
        await _close(backend, agent)


@pytest.mark.asyncio
@pytest.mark.parametrize("expired", (False, True))
async def test_legacy_redelivery_upgrade_respects_retention_boundary(tmp_path, expired):
    """Legacy repair accepts the exact retention boundary but never revives expiry."""

    backend = SQLiteBackend(str(tmp_path / f"legacy-retention-{expired}.db"))
    await backend.connect()
    store = DurableSignalStore(backend)
    await store.initialize()
    agent_id = "did:agent:one"
    boundary = datetime(2030, 1, 1, tzinfo=timezone.utc)
    store.now_utc = lambda: boundary  # type: ignore[method-assign]
    consumer = DurableConsumerRegistration(
        consumer_id=DURABLE_COGNITION_CONSUMER_ID,
        source="channel.message",
        agent_id=agent_id,
        correlation_selector=(
            f"payload.{DURABLE_COGNITION_MARKER}="
            f"{DURABLE_COGNITION_MARKER_VALUE}"
        ),
        max_attempts=0,
    )
    source_event_id = f"telegram:update:legacy-retention:{expired}"
    legacy = _channel_signal(agent_id, f"legacy-retention:{expired}")
    legacy.payload.pop(DURABLE_COGNITION_MARKER)
    try:
        persisted = await store.persist_signal(
            legacy,
            agent_id=agent_id,
            source_event_id=source_event_id,
            retention_days=0,
        )
        await store.register_consumer(consumer)
        if expired:
            await backend.execute(
                "UPDATE durable_signal_events SET retention_until = ? WHERE event_id = ?",
                (
                    store.to_timestamp_param(boundary - timedelta(microseconds=1)),
                    persisted.event_id,
                ),
            )

        upgraded = await store.upgrade_legacy_delivery_for_redelivery(
            agent_id=agent_id,
            consumer_id=consumer.consumer_id,
            event_id=persisted.event_id,
            source_event_id=source_event_id,
            expected_signal=legacy,
            caller_identity_factory=lambda: "v1:test-protected-caller",
        )
        assert upgraded is (not expired)
        deliveries = await store.list_deliveries(
            agent_id=agent_id, consumer_id=consumer.consumer_id
        )
        assert len(deliveries) == (0 if expired else 1)
        if not expired:
            assert deliveries[0].created_at == boundary
    finally:
        await backend.close()


@pytest.mark.asyncio
async def test_elided_row_allows_verified_live_redelivery_after_privacy_becomes_normal(
    tmp_path,
):
    """The persisted MAC row, not the retry's mode, authorizes live content."""
    from kestrel_sovereign.privacy import get_privacy_preset

    backend, agent, dispatcher = await _channel_dispatcher(
        tmp_path / "ephemeral-to-normal-live-retry.db", "did:agent:one"
    )
    agent.privacy_config = get_privacy_preset("ephemeral")
    consumer = DurableConsumerRegistration(
        consumer_id="core.channel-cognition-v1",
        source="channel.message",
        agent_id=agent.did,
        correlation_selector=(
            f"payload.{DURABLE_COGNITION_MARKER}="
            f"{DURABLE_COGNITION_MARKER_VALUE}"
        ),
        max_attempts=0,
    )
    source_event_id = "telegram:update:ephemeral-to-normal"
    prompts: list[str] = []

    async def fail_once_then_record(prompt: str):
        prompts.append(prompt)
        if len(prompts) == 1:
            raise RuntimeError("retry after privacy transition")
        return "ok"

    agent.process_input = fail_once_then_record
    original = _channel_signal(agent.did, "ephemeral-to-normal")
    original.payload["content"] = "live private content survives the verified retry"
    try:
        await dispatcher.register_durable_consumer(consumer)
        first = await dispatcher.enqueue_durable_cognition(
            original,
            source_event_id=source_event_id,
            consumer_id=consumer.consumer_id,
        )
        assert (await first.wait()).status is Status.FAILED
        persisted = await backend.fetch_one(
            "SELECT payload FROM durable_signal_events WHERE event_id = ?",
            (first.signal_id,),
        )
        binding = await backend.fetch_one(
            "SELECT integrity_binding FROM durable_signal_event_integrity WHERE event_id = ?",
            (first.signal_id,),
        )
        assert persisted is not None and "live private content" not in persisted[0]
        assert binding is not None

        # The redelivery happened after a legitimate mode transition. It must
        # compare against the row's HMAC and retain the live content instead of
        # reconstructing the old marker-only payload.
        agent.privacy_config = get_privacy_preset("normal")
        dispatcher._coalescing.reset()
        retry_signal = _channel_signal(agent.did, "ephemeral-to-normal")
        retry_signal.payload["content"] = "live private content survives the verified retry"
        retry = await dispatcher.enqueue_durable_cognition(
            retry_signal,
            source_event_id=source_event_id,
            consumer_id=consumer.consumer_id,
        )
        assert (await retry.wait()).status is Status.OK
        assert "live private content survives the verified retry" in prompts[-1]
    finally:
        await dispatcher.shutdown_durable_delivery()
        await _close(backend, agent)


@pytest.mark.asyncio
async def test_durable_caller_identity_is_tenant_bound_encrypted_and_reconstructed(tmp_path):
    """A retry uses the stored canonical caller, never a duplicate's claim."""
    backend, agent, dispatcher = await _channel_dispatcher(
        tmp_path / "durable-caller-identity.db", "did:agent:one"
    )
    consumer = DurableConsumerRegistration(
        consumer_id="core.channel-cognition-v1",
        source="channel.message",
        agent_id=agent.did,
        correlation_selector=(
            f"payload.{DURABLE_COGNITION_MARKER}="
            f"{DURABLE_COGNITION_MARKER_VALUE}"
        ),
        max_attempts=0,
    )
    source_event_id = "telegram:update:caller-identity"
    original = _channel_signal(agent.did, "caller-identity")
    original.caller = "telegram-user-original"
    changed = _channel_signal(agent.did, "caller-identity")
    changed.caller = "telegram-user-attacker"
    attempts = 0

    async def fail_once(_prompt: str):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("retry")
        return "ok"

    agent.process_input = fail_once
    try:
        await dispatcher.register_durable_consumer(consumer)
        first = await dispatcher.enqueue_durable_cognition(
            original, source_event_id=source_event_id, consumer_id=consumer.consumer_id
        )
        assert (await first.wait()).status is Status.FAILED
        row = await backend.fetch_one(
            "SELECT caller_identity FROM durable_signal_events WHERE event_id = ?",
            (first.signal_id,),
        )
        assert row is not None and row[0] is not None
        assert "telegram-user-original" not in row[0]
        delivery = (await dispatcher.list_durable_deliveries())[0]
        assert dispatcher._signal_from_durable_event(
            delivery.event, dispatch_signal=changed
        ).caller == "telegram-user-original"

        dispatcher._coalescing.reset()
        retry = await dispatcher.enqueue_durable_cognition(
            changed, source_event_id=source_event_id, consumer_id=consumer.consumer_id
        )
        assert (await retry.wait()).status is Status.OK
    finally:
        await dispatcher.shutdown_durable_delivery()
        await _close(backend, agent)


@pytest.mark.asyncio
async def test_keyless_channel_ingress_uses_nonsecret_versioned_caller_label(
    monkeypatch, tmp_path
):
    """Keyless NORMAL storage admits Telegram without retaining its raw caller twice."""

    monkeypatch.delenv("KESTREL_DATA_KEY", raising=False)
    backend, agent, dispatcher = await _channel_dispatcher(
        tmp_path / "keyless-channel-ingress.db", "did:agent:one"
    )
    consumer = DurableConsumerRegistration(
        consumer_id=DURABLE_COGNITION_CONSUMER_ID,
        source="channel.message",
        agent_id=agent.did,
        correlation_selector=(
            f"payload.{DURABLE_COGNITION_MARKER}="
            f"{DURABLE_COGNITION_MARKER_VALUE}"
        ),
        max_attempts=0,
    )
    source_event_id = "telegram:update:keyless-initial-ingress"
    original = _channel_signal(agent.did, "keyless-initial-ingress")
    original.caller = "telegram-user-555"
    try:
        await dispatcher.register_durable_consumer(consumer)
        handle = await dispatcher.enqueue_durable_cognition(
            original,
            source_event_id=source_event_id,
            consumer_id=consumer.consumer_id,
        )
        assert (
            await handle.wait_for_durable_admission()
        ).disposition is DurableAdmissionDisposition.COMMITTED
        assert (await handle.wait()).status is Status.OK

        row = await backend.fetch_one(
            "SELECT caller_identity FROM durable_signal_events WHERE event_id = ?",
            (handle.signal_id,),
        )
        assert row is not None
        assert row[0].startswith("v2:opaque:")
        assert "telegram-user-555" not in row[0]
        delivery = (await dispatcher.list_durable_deliveries())[0]
        recovered = dispatcher._signal_from_durable_event(
            delivery.event, dispatch_signal=original
        )
        assert recovered.caller == row[0]
        assert recovered.caller != original.caller
        assert delivery.status == ACKNOWLEDGED
    finally:
        await dispatcher.shutdown_durable_delivery()
        await _close(backend, agent)


@pytest.mark.asyncio
@pytest.mark.parametrize("recovery_failure", ("missing-key", "rotated-key", "corrupt-cipher"))
async def test_caller_recovery_failure_after_claim_releases_retry_and_logs_outcome(
    monkeypatch, tmp_path, recovery_failure
):
    """A claimed delivery is NACKed and audited when caller recovery fails."""

    backend, agent, dispatcher = await _channel_dispatcher(
        tmp_path / f"caller-recovery-{recovery_failure}.db", "did:agent:one"
    )
    consumer = DurableConsumerRegistration(
        consumer_id=DURABLE_COGNITION_CONSUMER_ID,
        source="channel.message",
        agent_id=agent.did,
        correlation_selector=(
            f"payload.{DURABLE_COGNITION_MARKER}="
            f"{DURABLE_COGNITION_MARKER_VALUE}"
        ),
        max_attempts=0,
    )
    source_event_id = f"telegram:update:caller-recovery:{recovery_failure}"

    async def fail_first_turn(_prompt: str):
        raise RuntimeError("leave one retryable delivery")

    agent.process_input = fail_first_turn
    try:
        await dispatcher.register_durable_consumer(consumer)
        first = await dispatcher.enqueue_durable_cognition(
            _channel_signal(agent.did, f"caller-recovery-{recovery_failure}"),
            source_event_id=source_event_id,
            consumer_id=consumer.consumer_id,
        )
        assert (await first.wait()).status is Status.FAILED
        delivery = (await dispatcher.list_durable_deliveries())[0]
        assert delivery.status == RETRY

        if recovery_failure == "missing-key":
            monkeypatch.setattr(dispatcher_module, "get_agent_fernet", lambda _agent: None)
        elif recovery_failure == "rotated-key":
            monkeypatch.setenv("KESTREL_DATA_KEY", "caller-recovery-rotated-key")
        else:
            await backend.execute(
                "UPDATE durable_signal_events SET caller_identity = ? WHERE event_id = ?",
                ("v1:corrupt-ciphertext", delivery.event_id),
            )

        dispatcher._coalescing.reset()
        retry = await dispatcher.enqueue_durable_cognition(
            _channel_signal(agent.did, f"caller-recovery-{recovery_failure}"),
            source_event_id=source_event_id,
            consumer_id=consumer.consumer_id,
        )
        assert (
            await retry.wait_for_durable_admission()
        ).disposition is DurableAdmissionDisposition.DUPLICATE
        assert (await retry.wait()).status is Status.FAILED
        recovered = (await dispatcher.list_durable_deliveries())[0]
        # Attempt two proves failure happened after this retry claimed the
        # durable lease; RETRY proves its exact lease was released, not ACKed.
        assert recovered.attempts == 2
        assert recovered.status == RETRY
        outcome = await backend.fetch_one(
            "SELECT id FROM signal_log WHERE id = ?", (retry.signal_id,)
        )
        assert outcome == (retry.signal_id,)
    finally:
        await dispatcher.shutdown_durable_delivery()
        await _close(backend, agent)


@pytest.mark.asyncio
async def test_elided_retry_integrity_is_keyed_to_the_agent_hierarchy(
    monkeypatch, tmp_path
):
    """An unkeyed payload hash would accept this key-rotation reproduction."""
    from kestrel_sovereign.privacy import get_privacy_preset

    monkeypatch.setenv("KESTREL_DATA_KEY", "durable-integrity-key-one")
    backend, agent, dispatcher = await _channel_dispatcher(
        tmp_path / "keyed-elided-integrity.db", "did:agent:one"
    )
    agent.privacy_config = get_privacy_preset("ephemeral")
    consumer = DurableConsumerRegistration(
        consumer_id="core.channel-cognition-v1",
        source="channel.message",
        agent_id=agent.did,
        correlation_selector=(
            f"payload.{DURABLE_COGNITION_MARKER}="
            f"{DURABLE_COGNITION_MARKER_VALUE}"
        ),
        max_attempts=0,
    )
    source_event_id = "telegram:update:keyed-integrity"
    original = _channel_signal(agent.did, "keyed-integrity")
    try:
        await dispatcher.register_durable_consumer(consumer)
        first = await dispatcher.enqueue_durable_cognition(
            original, source_event_id=source_event_id, consumer_id=consumer.consumer_id
        )
        assert (await first.wait()).status is Status.OK

        # Reopen the delivery only for this integrity proof. A different
        # hierarchy key must reject byte-for-byte identical live data; an
        # unkeyed SHA-256 binding would incorrectly accept it.
        delivery = (await dispatcher.list_durable_deliveries())[0]
        await backend.execute(
            "UPDATE durable_signal_deliveries SET status = ?, acknowledged_at = NULL, "
            "terminal_at = NULL, next_attempt_at = NULL WHERE delivery_id = ?",
            (RETRY, delivery.delivery_id),
        )
        monkeypatch.setenv("KESTREL_DATA_KEY", "durable-integrity-key-two")
        dispatcher._coalescing.reset()
        retry = await dispatcher.enqueue_durable_cognition(
            _channel_signal(agent.did, "keyed-integrity"),
            source_event_id=source_event_id,
            consumer_id=consumer.consumer_id,
        )
        assert (await retry.wait()).status is Status.FAILED
    finally:
        await dispatcher.shutdown_durable_delivery()
        await _close(backend, agent)


@pytest.mark.asyncio
async def test_privacy_elided_retry_requires_identical_live_input_after_restart(tmp_path):
    """An elided row binds restarts to the original normalized Telegram event."""
    from kestrel_sovereign.privacy import get_privacy_preset

    path = tmp_path / "privacy-integrity-retry.db"
    consumer_id = "core.channel-cognition-v1"
    source_event_id = "telegram:update:privacy-integrity"
    backend_a, agent_a, dispatcher_a = await _channel_dispatcher(path, "did:agent:one")
    agent_a.privacy_config = get_privacy_preset("ephemeral")
    consumer = DurableConsumerRegistration(
        consumer_id=consumer_id,
        source="channel.message",
        agent_id=agent_a.did,
        correlation_selector=(
            f"payload.{DURABLE_COGNITION_MARKER}="
            f"{DURABLE_COGNITION_MARKER_VALUE}"
        ),
        max_attempts=0,
    )

    async def fail_cognition(_prompt: str):
        raise RuntimeError("retry after restart")

    agent_a.process_input = fail_cognition
    original = _channel_signal(agent_a.did, "privacy-integrity")
    original.payload["content"] = "the original private content"
    try:
        await dispatcher_a.register_durable_consumer(consumer)
        first = await dispatcher_a.enqueue_durable_cognition(
            original,
            source_event_id=source_event_id,
            consumer_id=consumer_id,
        )
        assert (await first.wait()).status is Status.FAILED
        assert (await dispatcher_a.list_durable_deliveries())[0].status == RETRY
        row = await backend_a.fetch_one(
            "SELECT payload, caller_identity FROM durable_signal_events WHERE agent_id = ?",
            (agent_a.did,),
        )
        assert row is not None and "original private content" not in row[0]
        assert row[1] is None
        await dispatcher_a.shutdown_durable_delivery()
    finally:
        await _close(backend_a, agent_a)

    backend_b, agent_b, dispatcher_b = await _channel_dispatcher(path, "did:agent:one")
    agent_b.privacy_config = get_privacy_preset("ephemeral")
    prompts: list[str] = []

    async def record_cognition(prompt: str):
        prompts.append(prompt)
        return "ok"

    agent_b.process_input = record_cognition
    changed = _channel_signal(agent_b.did, "privacy-integrity")
    changed.caller = "different-telegram-caller"
    changed.payload["content"] = "the original private content"
    identical = _channel_signal(agent_b.did, "privacy-integrity")
    identical.payload["content"] = "the original private content"
    try:
        await dispatcher_b.register_durable_consumer(consumer)
        rejected = await dispatcher_b.enqueue_durable_cognition(
            changed,
            source_event_id=source_event_id,
            consumer_id=consumer_id,
        )
        assert (await rejected.wait()).status is Status.FAILED
        assert prompts == []
        assert (await dispatcher_b.list_durable_deliveries())[0].status == RETRY

        accepted = await dispatcher_b.enqueue_durable_cognition(
            identical,
            source_event_id=source_event_id,
            consumer_id=consumer_id,
        )
        assert (await accepted.wait()).status is Status.OK
        assert len(prompts) == 1
        assert "the original private content" in prompts[0]
        assert (await dispatcher_b.list_durable_deliveries())[0].status == ACKNOWLEDGED
    finally:
        await dispatcher_b.shutdown_durable_delivery()
        await _close(backend_b, agent_b)


@pytest.mark.asyncio
async def test_cognition_renews_short_durable_lease_until_turn_completes(tmp_path):
    """An unbounded cognition turn cannot redeliver merely after 60 seconds."""
    backend, agent, dispatcher = await _channel_dispatcher(
        tmp_path / "cognition-short-lease.db", "did:agent:one"
    )
    consumer = DurableConsumerRegistration(
        consumer_id="core.channel-cognition-v1",
        source="channel.message",
        agent_id=agent.did,
        correlation_selector=(
            f"payload.{DURABLE_COGNITION_MARKER}="
            f"{DURABLE_COGNITION_MARKER_VALUE}"
        ),
        max_attempts=0,
        lease_seconds=1,
    )
    started = asyncio.Event()
    release = asyncio.Event()

    async def long_turn(_prompt: str):
        started.set()
        await release.wait()
        return "ok"

    agent.process_input = long_turn
    try:
        await dispatcher.register_durable_consumer(consumer)
        handle = await dispatcher.enqueue_durable_cognition(
            _channel_signal(agent.did, "short-lease"),
            source_event_id="telegram:update:short-lease",
            consumer_id=consumer.consumer_id,
        )
        await asyncio.wait_for(started.wait(), timeout=1)
        await asyncio.sleep(1.2)
        assert await dispatcher.claim_durable_delivery(
            consumer_id=consumer.consumer_id, executor_id="second-executor"
        ) is None
        release.set()
        assert (await handle.wait()).status is Status.OK
        assert (await dispatcher.list_durable_deliveries())[0].status == ACKNOWLEDGED
    finally:
        release.set()
        await dispatcher.shutdown_durable_delivery()
        await _close(backend, agent)


@pytest.mark.asyncio
async def test_renewal_loss_cancels_cognition_before_a_second_executor_retries(tmp_path):
    """Lease loss cannot leave the original turn live beside a retry worker."""

    path = tmp_path / "cognition-renewal-loss.db"
    backend_a, agent_a, dispatcher_a = await _channel_dispatcher(path, "did:agent:one")
    backend_b, agent_b, dispatcher_b = await _channel_dispatcher(path, "did:agent:one")
    consumer = DurableConsumerRegistration(
        consumer_id=DURABLE_COGNITION_CONSUMER_ID,
        source="channel.message",
        agent_id=agent_a.did,
        correlation_selector=(
            f"payload.{DURABLE_COGNITION_MARKER}="
            f"{DURABLE_COGNITION_MARKER_VALUE}"
        ),
        max_attempts=0,
        lease_seconds=1,
    )
    first_started = asyncio.Event()
    first_cancelled = asyncio.Event()
    active_turns = 0
    peak_active_turns = 0

    async def first_turn(_prompt: str):
        nonlocal active_turns, peak_active_turns
        active_turns += 1
        peak_active_turns = max(peak_active_turns, active_turns)
        first_started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            first_cancelled.set()
            raise
        finally:
            active_turns -= 1

    async def second_turn(_prompt: str):
        nonlocal active_turns, peak_active_turns
        assert first_cancelled.is_set()
        active_turns += 1
        peak_active_turns = max(peak_active_turns, active_turns)
        active_turns -= 1
        return "retry complete"

    renewal_rejected = asyncio.Event()

    async def reject_renewal(**_kwargs):
        renewal_rejected.set()
        return None

    agent_a.process_input = first_turn
    agent_b.process_input = second_turn
    dispatcher_a.renew_durable_delivery_lease = reject_renewal  # type: ignore[method-assign]
    try:
        await dispatcher_a.register_durable_consumer(consumer)
        await dispatcher_b.register_durable_consumer(consumer)
        first = await dispatcher_a.enqueue_durable_cognition(
            _channel_signal(agent_a.did, "renewal-loss"),
            source_event_id="telegram:update:renewal-loss",
            consumer_id=consumer.consumer_id,
        )
        await asyncio.wait_for(first_started.wait(), timeout=1)
        await asyncio.wait_for(renewal_rejected.wait(), timeout=2)
        assert (await first.wait()).status is Status.FAILED
        await asyncio.wait_for(first_cancelled.wait(), timeout=1)
        assert (await dispatcher_a.list_durable_deliveries())[0].status == RETRY

        retry = await dispatcher_b.enqueue_durable_cognition(
            _channel_signal(agent_b.did, "renewal-loss"),
            source_event_id="telegram:update:renewal-loss",
            consumer_id=consumer.consumer_id,
        )
        assert (await retry.wait()).status is Status.OK
        assert peak_active_turns == 1
        assert (await dispatcher_b.list_durable_deliveries())[0].status == ACKNOWLEDGED
    finally:
        await dispatcher_a.shutdown_durable_delivery()
        await dispatcher_b.shutdown_durable_delivery()
        await _close(backend_a, agent_a)
        await _close(backend_b, agent_b)


@pytest.mark.asyncio
async def test_renewal_loss_quarantines_cancellation_resistant_cognition_until_it_settles(
    tmp_path,
):
    """A live managed owner fences expired work while its cancelled turn drains."""

    path = tmp_path / "cancellation-resistant-cognition.db"
    backend_a, agent_a, dispatcher_a = await _channel_dispatcher(path, "did:agent:one")
    backend_b, agent_b, dispatcher_b = await _channel_dispatcher(path, "did:agent:one")
    consumer = DurableConsumerRegistration(
        consumer_id=DURABLE_COGNITION_CONSUMER_ID,
        source="channel.message",
        agent_id=agent_a.did,
        correlation_selector=(
            f"payload.{DURABLE_COGNITION_MARKER}="
            f"{DURABLE_COGNITION_MARKER_VALUE}"
        ),
        max_attempts=0,
        lease_seconds=1,
    )
    cognition_started = asyncio.Event()
    cancellation_suppressed = asyncio.Event()
    allow_cognition_exit = asyncio.Event()
    exact_release_finished = asyncio.Event()
    renewal_rejected = asyncio.Event()

    async def cancellation_resistant_turn(_prompt: str):
        cognition_started.set()
        try:
            await allow_cognition_exit.wait()
        except asyncio.CancelledError:
            cancellation_suppressed.set()
            await allow_cognition_exit.wait()
        return "late completion"

    async def retry_turn(_prompt: str):
        return "safe retry"

    async def reject_renewal(**_kwargs):
        renewal_rejected.set()
        return None

    agent_a.process_input = cancellation_resistant_turn
    agent_b.process_input = retry_turn
    dispatcher_a.renew_durable_delivery_lease = reject_renewal  # type: ignore[method-assign]
    original_release = dispatcher_a.release_durable_delivery_after_task

    async def observe_exact_release(**kwargs):
        try:
            return await original_release(**kwargs)
        finally:
            exact_release_finished.set()

    dispatcher_a.release_durable_delivery_after_task = observe_exact_release  # type: ignore[method-assign]
    first = None
    try:
        await dispatcher_a.register_durable_consumer(consumer)
        await dispatcher_b.register_durable_consumer(consumer)
        first = await dispatcher_a.enqueue_durable_cognition(
            _channel_signal(agent_a.did, "cancellation-resistant"),
            source_event_id="telegram:update:cancellation-resistant",
            consumer_id=consumer.consumer_id,
        )
        await asyncio.wait_for(cognition_started.wait(), timeout=1)
        await asyncio.wait_for(renewal_rejected.wait(), timeout=2)

        first_result = await asyncio.wait_for(first.wait(), timeout=1)
        assert first_result.status is Status.FAILED
        assert cancellation_suppressed.is_set()
        assert dispatcher_a.retained_durable_cognition_task_count == 1
        leased = (await dispatcher_a.list_durable_deliveries())[0]
        assert leased.status == LEASED
        assert leased.lease_expires_at is not None

        # Advance only the ledger clock: no wall-clock sleep is needed to show
        # that an expired lease owned by a heartbeat-live dispatcher remains
        # fenced while its retained cognition task still runs.
        assert await dispatcher_b._durable_store.claim_delivery_for_event(
            agent_id=agent_b.did,
            consumer_id=consumer.consumer_id,
            event_id=leased.event.event_id,
            executor_id=dispatcher_b._durable_delivery_owner,
            now=leased.lease_expires_at + timedelta(microseconds=1),
            runtime_owner_stale_before=(
                leased.lease_expires_at - timedelta(minutes=1)
            ),
        ) is None

        allow_cognition_exit.set()
        await asyncio.wait_for(exact_release_finished.wait(), timeout=1)
        assert dispatcher_a.retained_durable_cognition_task_count == 0
        assert (await dispatcher_a.list_durable_deliveries())[0].status == RETRY

        retry = await dispatcher_b.enqueue_durable_cognition(
            _channel_signal(agent_b.did, "cancellation-resistant"),
            source_event_id="telegram:update:cancellation-resistant",
            consumer_id=consumer.consumer_id,
        )
        assert (await retry.wait()).status is Status.OK
        assert (await dispatcher_b.list_durable_deliveries())[0].status == ACKNOWLEDGED
        assert await backend_a.fetch_val(
            "SELECT COUNT(*) FROM signal_log WHERE id = ?", (first.signal_id,)
        ) == 1
    finally:
        allow_cognition_exit.set()
        await dispatcher_a.shutdown_durable_delivery()
        await dispatcher_b.shutdown_durable_delivery()
        await _close(backend_a, agent_a)
        await _close(backend_b, agent_b)


@pytest.mark.asyncio
async def test_shutdown_keeps_repeatedly_cancelled_cognition_owner_live_until_peer_safe(
    tmp_path, monkeypatch
):
    """Bounded unload cannot mark a still-acting cognition owner reclaimable."""
    monkeypatch.setattr(dispatcher_module, "_DURABLE_COGNITION_CANCELLATION_GRACE", 0.01)
    path = tmp_path / "shutdown-cognition-owner-fence.db"
    backend_a, agent_a, dispatcher_a = await _channel_dispatcher(path, "did:agent:one")
    backend_b, agent_b, dispatcher_b = await _channel_dispatcher(path, "did:agent:one")
    # Shorten only the test's owner-staleness window. The fence must continue
    # its private heartbeat after ordinary shutdown closes public admission.
    dispatcher_a._runtime_owner_stale_after = timedelta(milliseconds=45)
    consumer = DurableConsumerRegistration(
        consumer_id=DURABLE_COGNITION_CONSUMER_ID,
        source="channel.message",
        agent_id=agent_a.did,
        correlation_selector=(
            f"payload.{DURABLE_COGNITION_MARKER}="
            f"{DURABLE_COGNITION_MARKER_VALUE}"
        ),
        max_attempts=0,
        lease_seconds=1,
    )
    started = asyncio.Event()
    repeated_cancellation = asyncio.Event()
    allow_exit = asyncio.Event()
    cancellation_count = 0

    async def hostile_turn(_prompt: str):
        nonlocal cancellation_count
        started.set()
        while True:
            try:
                await allow_exit.wait()
                return "settled after hostile cancellation"
            except asyncio.CancelledError:
                cancellation_count += 1
                if cancellation_count >= 3:
                    repeated_cancellation.set()

    async def reject_renewal(**_kwargs):
        return None

    agent_a.process_input = hostile_turn
    dispatcher_a.renew_durable_delivery_lease = reject_renewal  # type: ignore[method-assign]
    try:
        await dispatcher_a.register_durable_consumer(consumer)
        await dispatcher_b.register_durable_consumer(consumer)
        first = await dispatcher_a.enqueue_durable_cognition(
            _channel_signal(agent_a.did, "shutdown-owner-fence"),
            source_event_id="telegram:update:shutdown-owner-fence",
            consumer_id=consumer.consumer_id,
        )
        await asyncio.wait_for(started.wait(), timeout=1)
        assert (await asyncio.wait_for(first.wait(), timeout=1)).status is Status.FAILED
        leased = (await dispatcher_a.list_durable_deliveries())[0]

        # The public shutdown is bounded. It returns with a live ownership
        # fence while the task keeps suppressing cancellation in the process.
        assert await asyncio.wait_for(dispatcher_a.shutdown_durable_delivery(), timeout=1) is False
        await asyncio.wait_for(repeated_cancellation.wait(), timeout=1)
        assert dispatcher_a.durable_shutdown_owner_fenced is True
        owner = await backend_a.fetch_one(
            "SELECT stopped_at FROM durable_signal_runtime_owners WHERE agent_id = ? AND owner_id = ?",
            (agent_a.did, dispatcher_a._durable_delivery_owner),
        )
        assert owner is not None and owner[0] is None

        # Crossing multiple stale windows must not make retained work
        # reclaimable: the fence timer runs a private owner heartbeat even
        # though public durable APIs remain closed.
        await asyncio.sleep(0.12)
        assert await dispatcher_b._durable_store.recover_abandoned_leases(
            agent_id=agent_b.did,
            recovering_owner_id=dispatcher_b._durable_delivery_owner,
            stale_before=datetime.now(timezone.utc)
            - dispatcher_a._runtime_owner_stale_after,
        ) == 0

        # Even after its lease's nominal deadline, a peer cannot reclaim while
        # the original coroutine could still make a side effect.
        assert await dispatcher_b._durable_store.claim_delivery_for_event(
            agent_id=agent_b.did,
            consumer_id=consumer.consumer_id,
            event_id=leased.event.event_id,
            executor_id=dispatcher_b._durable_delivery_owner,
            now=leased.lease_expires_at + timedelta(microseconds=1),
            runtime_owner_stale_before=leased.lease_expires_at - timedelta(minutes=1),
        ) is None

        allow_exit.set()
        await asyncio.wait_for(dispatcher_a.wait_for_durable_shutdown_release(), timeout=1)
        assert dispatcher_a.durable_shutdown_owner_fenced is False
        owner = await backend_a.fetch_one(
            "SELECT stopped_at FROM durable_signal_runtime_owners WHERE agent_id = ? AND owner_id = ?",
            (agent_a.did, dispatcher_a._durable_delivery_owner),
        )
        assert owner is not None and owner[0] is not None

        # Now recovery is legitimate: the old task is terminal before the peer
        # can obtain the delivery again.
        assert await dispatcher_b._durable_store.recover_abandoned_leases(
            agent_id=agent_b.did,
            recovering_owner_id=dispatcher_b._durable_delivery_owner,
            stale_before=datetime.now(timezone.utc),
        ) == 1
        assert await dispatcher_b._durable_store.claim_delivery_for_event(
            agent_id=agent_b.did,
            consumer_id=consumer.consumer_id,
            event_id=leased.event.event_id,
            executor_id=dispatcher_b._durable_delivery_owner,
        ) is not None
    finally:
        allow_exit.set()
        await dispatcher_a.wait_for_durable_shutdown_release()
        await dispatcher_b.shutdown_durable_delivery()
        await _close(backend_a, agent_a)
        await _close(backend_b, agent_b)


@pytest.mark.asyncio
async def test_durable_cognition_nested_and_background_dispatches_keep_one_audit_each(tmp_path):
    """Nested dispatch ContextVars cannot append outcomes into the outer lease audit."""
    backend, agent, dispatcher = await _channel_dispatcher(
        tmp_path / "isolated-deferred-outcomes.db", "did:agent:one"
    )
    dispatcher._registry.register(_registration(agent))
    consumer = DurableConsumerRegistration(
        consumer_id=DURABLE_COGNITION_CONSUMER_ID,
        source="channel.message",
        agent_id=agent.did,
        correlation_selector=(
            f"payload.{DURABLE_COGNITION_MARKER}="
            f"{DURABLE_COGNITION_MARKER_VALUE}"
        ),
        max_attempts=0,
    )
    awaited = _signal(agent_id=agent.did, message="awaited nested")
    created = _signal(agent_id=agent.did, message="created nested")
    enqueued = _signal(agent_id=agent.did, message="enqueued nested")

    async def cognition_turn(_prompt: str):
        assert (await dispatcher.dispatch_signal(awaited)).status is Status.OK
        created_task = asyncio.create_task(dispatcher.dispatch_signal(created))
        assert (await created_task).status is Status.OK
        handle = await dispatcher.enqueue_signal(enqueued)
        assert (await handle.task).status is Status.OK
        return "outer complete"

    agent.process_input = cognition_turn
    try:
        await dispatcher.register_durable_consumer(consumer)
        outer = await dispatcher.enqueue_durable_cognition(
            _channel_signal(agent.did, "isolated-deferred-outcomes"),
            source_event_id="telegram:update:isolated-deferred-outcomes",
            consumer_id=consumer.consumer_id,
        )
        assert (await outer.wait()).status is Status.OK
        for signal_id in (outer.signal_id, awaited.id, created.id, enqueued.id):
            assert await backend.fetch_val(
                "SELECT COUNT(*) FROM signal_log WHERE id = ?", (signal_id,)
            ) == 1
    finally:
        await dispatcher.shutdown_durable_delivery()
        await _close(backend, agent)


@pytest.mark.asyncio
@pytest.mark.parametrize("acknowledged", (True, False))
async def test_simultaneous_route_success_and_lease_loss_finalizes_once(
    tmp_path, acknowledged
):
    """A same-turn route/loss race bases one final outcome on the exact ACK."""

    backend, agent, dispatcher = await _channel_dispatcher(
        tmp_path / "simultaneous-route-and-loss.db", "did:agent:one"
    )
    consumer = DurableConsumerRegistration(
        consumer_id=DURABLE_COGNITION_CONSUMER_ID,
        source="channel.message",
        agent_id=agent.did,
        correlation_selector=(
            f"payload.{DURABLE_COGNITION_MARKER}="
            f"{DURABLE_COGNITION_MARKER_VALUE}"
        ),
        max_attempts=0,
    )
    acknowledgements = []
    nacks = []

    @asynccontextmanager
    async def simultaneous_lease_loss(_delivery):
        lost = asyncio.get_running_loop().create_future()
        asyncio.get_running_loop().call_soon(
            lost.set_result, "forced simultaneous renewal loss"
        )
        yield lost

    original_ack = dispatcher.ack_durable_delivery
    original_nack = dispatcher.nack_durable_delivery

    async def observe_ack(**kwargs):
        acknowledgements.append(kwargs)
        if acknowledged:
            return await original_ack(**kwargs)
        return False

    async def observe_nack(**kwargs):
        nacks.append(kwargs)
        return await original_nack(**kwargs)

    dispatcher._renew_durable_cognition_lease = simultaneous_lease_loss  # type: ignore[method-assign]
    dispatcher.ack_durable_delivery = observe_ack  # type: ignore[method-assign]
    dispatcher.nack_durable_delivery = observe_nack  # type: ignore[method-assign]
    try:
        await dispatcher.register_durable_consumer(consumer)
        handle = await dispatcher.enqueue_durable_cognition(
            _channel_signal(agent.did, "simultaneous-completion"),
            source_event_id="telegram:update:simultaneous-completion",
            consumer_id=consumer.consumer_id,
        )
        admission = await handle.wait_for_durable_admission()
        result = await handle.wait()
        assert result.status is (Status.OK if acknowledged else Status.FAILED)
        # Admission records the committed delivery before either terminal ACK
        # outcome races the lease-renewal result.
        assert admission.disposition is DurableAdmissionDisposition.COMMITTED
        assert len(acknowledgements) == 1
        assert nacks == []
        assert (await dispatcher.list_durable_deliveries())[0].status == (
            ACKNOWLEDGED if acknowledged else RETRY
        )
        assert await backend.fetch_val(
            "SELECT COUNT(*) FROM signal_log WHERE id = ?", (result.signal_id,)
        ) == 1
    finally:
        await dispatcher.shutdown_durable_delivery()
        await _close(backend, agent)


@pytest.mark.asyncio
async def test_cursor_owned_cognition_recovers_after_restart_before_cognition_ack(tmp_path):
    """A persisted ACKed callback is drained after restart without redelivery."""
    path = tmp_path / "cursor-restart.db"
    backend_a, agent_a, dispatcher_a = await _channel_dispatcher(path, "did:agent:one")
    consumer = DurableConsumerRegistration(
        consumer_id="core.channel-cognition-v1",
        source="channel.message",
        agent_id=agent_a.did,
        correlation_selector=(
            f"payload.{DURABLE_COGNITION_MARKER}="
            f"{DURABLE_COGNITION_MARKER_VALUE}"
        ),
        max_attempts=0,
    )
    started = asyncio.Event()

    async def interrupted_cognition(_prompt: str):
        started.set()
        raise RuntimeError("process died after provider ACK")

    agent_a.process_input = interrupted_cognition
    try:
        await dispatcher_a.register_durable_consumer(consumer)
        interrupted = await dispatcher_a.enqueue_durable_cognition(
            _channel_signal(agent_a.did, "restart"),
            source_event_id="telegram:update:restart",
            consumer_id=consumer.consumer_id,
        )
        await asyncio.wait_for(started.wait(), timeout=1)
        # The provider receives this receipt before the route returns. Once
        # cognition NACKs, no provider redelivery is manufactured below: the
        # restarted owner must find the persisted retry itself.
        assert (
            await interrupted.wait_for_durable_admission()
        ).disposition is DurableAdmissionDisposition.COMMITTED
        assert (await interrupted.wait()).status is Status.FAILED
        assert (await dispatcher_a.list_durable_deliveries())[0].status == RETRY
    finally:
        await dispatcher_a.shutdown_durable_delivery()
        await _close(backend_a, agent_a)

    backend_b, agent_b, dispatcher_b = await _channel_dispatcher(path, "did:agent:one")
    try:
        assert (await dispatcher_b.list_durable_deliveries())[0].status == RETRY
        recovered = asyncio.Event()

        async def recovered_cognition(_prompt: str):
            recovered.set()

        agent_b.process_input = recovered_cognition
        await dispatcher_b.register_durable_consumer(consumer)
        await dispatcher_b.start_durable_cognition_consumer(consumer.consumer_id)
        for _ in range(100):
            if recovered.is_set():
                break
            await asyncio.sleep(0.01)
        assert recovered.is_set(), (
            (await dispatcher_b.list_durable_deliveries())[0],
            dispatcher_b._durable_cognition_drainers,
            dispatcher_b._durable_cognition_drain_timers,
        )
        for _ in range(100):
            delivery = (await dispatcher_b.list_durable_deliveries())[0]
            if delivery.status == ACKNOWLEDGED:
                break
            await asyncio.sleep(0.01)
        delivery = (await dispatcher_b.list_durable_deliveries())[0]
        assert delivery.status == ACKNOWLEDGED
        assert delivery.attempts == 2
    finally:
        await dispatcher_b.shutdown_durable_delivery()
        await _close(backend_b, agent_b)


@pytest.mark.asyncio
async def test_malformed_telegram_terminal_is_durable_without_cognition(tmp_path):
    backend, agent, dispatcher = await _channel_dispatcher(
        tmp_path / "terminal-ingress.db", "did:agent:one"
    )
    consumer = DurableConsumerRegistration(
        consumer_id=DURABLE_TERMINAL_CONSUMER_ID,
        source="channel.message",
        agent_id=agent.did,
        correlation_selector=(
            f"payload.{DURABLE_TERMINAL_MARKER}={DURABLE_TERMINAL_MARKER_VALUE}"
        ),
        max_attempts=0,
    )
    signal = _channel_signal(agent.did, "malformed")
    signal.payload.pop(DURABLE_COGNITION_MARKER)
    signal.payload[DURABLE_TERMINAL_MARKER] = DURABLE_TERMINAL_MARKER_VALUE
    try:
        await dispatcher.register_durable_consumer(consumer)
        handle = await dispatcher.enqueue_durable_terminal(
            signal,
            source_event_id="telegram:update:malformed",
            consumer_id=consumer.consumer_id,
        )
        assert (
            await handle.wait_for_durable_admission()
        ).disposition is DurableAdmissionDisposition.TERMINAL
        assert (await handle.wait()).status is Status.OK
        delivery = (await dispatcher.list_durable_deliveries())[0]
        assert delivery.status == TERMINAL_ACKABLE
    finally:
        await dispatcher.shutdown_durable_delivery()
        await _close(backend, agent)


@pytest.mark.asyncio
async def test_live_cognition_nack_is_drained_without_provider_redelivery(tmp_path):
    backend, agent, dispatcher = await _channel_dispatcher(
        tmp_path / "live-nack-drain.db", "did:agent:one"
    )
    consumer = DurableConsumerRegistration(
        consumer_id=DURABLE_COGNITION_CONSUMER_ID,
        source="channel.message",
        agent_id=agent.did,
        correlation_selector=(
            f"payload.{DURABLE_COGNITION_MARKER}={DURABLE_COGNITION_MARKER_VALUE}"
        ),
        max_attempts=0,
    )
    attempts = 0
    recovered = asyncio.Event()

    async def fail_once_then_recover(_prompt: str):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("first cognition attempt failed")
        recovered.set()

    agent.process_input = fail_once_then_recover
    try:
        await dispatcher.register_durable_consumer(consumer)
        await dispatcher.start_durable_cognition_consumer(consumer.consumer_id)
        initial = await dispatcher.enqueue_durable_cognition(
            _channel_signal(agent.did, "live-nack"),
            source_event_id="telegram:update:live-nack",
            consumer_id=consumer.consumer_id,
        )
        assert (
            await initial.wait_for_durable_admission()
        ).disposition is DurableAdmissionDisposition.COMMITTED
        assert (await initial.wait()).status is Status.FAILED
        try:
            await asyncio.wait_for(recovered.wait(), timeout=1)
        except TimeoutError:
            delivery = (await dispatcher.list_durable_deliveries())[0]
            raise AssertionError(
                "durable cognition retry was not drained: "
                f"status={delivery.status} attempts={delivery.attempts} "
                f"next_attempt_at={delivery.next_attempt_at} "
                f"error={delivery.last_error!r}"
            )
        deadline = time.monotonic() + 1
        while True:
            delivery = (await dispatcher.list_durable_deliveries())[0]
            if delivery.status == ACKNOWLEDGED or time.monotonic() >= deadline:
                break
            await asyncio.sleep(0.01)
        assert delivery.status == ACKNOWLEDGED
        assert delivery.attempts == 2
    finally:
        await dispatcher.shutdown_durable_delivery()
        await _close(backend, agent)


@pytest.mark.asyncio
async def test_two_executors_cannot_claim_the_same_delivery(tmp_path):
    path = tmp_path / "claim-race.db"
    backend_a, agent_a, dispatcher_a = await _dispatcher(path, "did:agent:one")
    backend_b, agent_b, dispatcher_b = await _dispatcher(path, "did:agent:one")
    consumer = DurableConsumerRegistration(
        consumer_id="workflow-wait",
        source="provider.message",
        agent_id=agent_a.did,
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
        consumer_id="workflow-wait",
        source="provider.message",
        agent_id=agent_a.did,
        lease_seconds=1,
    )
    secret = "commit-boundary-customer@example.com"
    await dispatcher_a.register_durable_consumer(consumer)
    await dispatcher_b.register_durable_consumer(consumer)

    original_transaction = backend_a.transaction
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

    backend_a.transaction = pause_after_sidecar_before_commit

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
        assert len(dispatcher_a._transient_durable_handoffs) == 1

        # The old implementation would have published a one-second lease that
        # expired while this transaction remained uncommitted. The reservation
        # carries no lease deadline here, so a pause longer than that duration
        # cannot make the post-commit owner activation stale.
        await asyncio.sleep(1.1)

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
        backend_b.fetch_one = original_peer_fetch_one
        await dispatcher_a.shutdown_durable_delivery()
        await dispatcher_b.shutdown_durable_delivery()
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
        await dispatcher.shutdown_durable_delivery()
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
        await dispatcher.shutdown_durable_delivery()
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
        await dispatcher.shutdown_durable_delivery()
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

    @asynccontextmanager
    async def rollback_after_before_commit():
        async with original_transaction():
            yield
            # This runs after the store invokes its pre-commit sidecar hook.
            raise RuntimeError("force durable event rollback")

    backend.transaction = rollback_after_before_commit
    try:
        result = await dispatcher.dispatch_signal(
            _signal(agent_id=agent.did, message="rollback-secret@example.com"),
            source_event_id="volatile-rollback",
        )
        assert result.status is Status.FAILED
        assert dispatcher._transient_durable_handoffs == {}
        assert await backend.fetch_one(
            "SELECT event_id FROM durable_signal_events WHERE agent_id = ?",
            (agent.did,),
        ) is None
    finally:
        backend.transaction = original_transaction
        await dispatcher.shutdown_durable_delivery()
        await _close(backend, agent)


@pytest.mark.asyncio
async def test_graceful_shutdown_requeues_unactivated_volatile_reservation_as_marker(
    tmp_path,
):
    """Shutdown drops raw state before releasing its own reservation to retry."""
    from kestrel_sovereign.privacy import get_privacy_preset
    from kestrel_sovereign.signals.dispatcher import _TransientDurableHandoff

    backend, agent, dispatcher = await _dispatcher(
        tmp_path / "volatile-graceful-shutdown.db", "did:agent:one"
    )
    agent.privacy_config = get_privacy_preset("ephemeral")
    consumer = DurableConsumerRegistration(
        consumer_id="workflow-wait", source="provider.message", agent_id=agent.did
    )
    secret = "graceful-shutdown-customer@example.com"
    try:
        await dispatcher.register_durable_consumer(consumer)
        marker_event = _signal(agent_id=agent.did)
        marker_event.payload = {"_privacy_gated": "none"}
        persisted = await dispatcher._durable_store.persist_signal(
            marker_event,
            agent_id=agent.did,
            source_event_id="volatile-graceful-shutdown",
            retention_days=7,
            transient_selector_payload={"workflow": "wf-1", "message": secret},
            initial_lease_owner=dispatcher._durable_delivery_owner,
        )
        reservation = persisted.initial_reservations[0]
        retention_until = persisted.retention_until
        assert retention_until is not None
        dispatcher._transient_durable_handoffs[reservation.delivery_id] = (
            _TransientDurableHandoff(
                payload={"workflow": "wf-1", "message": secret},
                consumer_id=consumer.consumer_id,
                created_at=reservation.created_at,
                retention_until=retention_until,
                expires_at=retention_until,
                initial_lease_token=reservation.reservation_token,
            )
        )

        await dispatcher.shutdown_durable_delivery()

        assert dispatcher._transient_durable_handoffs == {}
        released = await dispatcher._durable_store.get_delivery(
            agent_id=agent.did,
            consumer_id=consumer.consumer_id,
            delivery_id=reservation.delivery_id,
        )
        assert released is not None
        assert released.status == RETRY
        replay = await dispatcher._durable_store.claim_delivery(
            agent_id=agent.did,
            consumer_id=consumer.consumer_id,
            executor_id="restart-worker",
        )
        assert replay is not None
        assert replay.event.payload == {"_privacy_gated": "none"}
        assert secret not in json.dumps(replay.event.payload)
    finally:
        await _close(backend, agent)


@pytest.mark.asyncio
@pytest.mark.parametrize("failure_phase", ("before_update", "after_update"))
async def test_committed_initial_activation_failure_requeues_only_marker_and_drops_sidecar(
    tmp_path, failure_phase
):
    """A committed event cannot strand its raw sidecar when activation fails.

    The after-update case simulates the activation's conditional UPDATE
    committing before the following delivery readback fails.  That row is
    briefly ``LEASED`` to its initial owner/token and must still be released
    as marker-only retry work.
    """
    from kestrel_sovereign.privacy import get_privacy_preset

    backend, agent, dispatcher = await _dispatcher(
        tmp_path / f"volatile-activation-{failure_phase}.db", "did:agent:one"
    )
    agent.privacy_config = get_privacy_preset("ephemeral")
    consumer = DurableConsumerRegistration(
        consumer_id="workflow-wait", source="provider.message", agent_id=agent.did
    )
    secret = f"activation-{failure_phase}-customer@example.com"
    original_activate = dispatcher._durable_store.activate_initial_delivery
    original_readback = dispatcher._durable_store._delivery_for_lease_locked
    try:
        await dispatcher.register_durable_consumer(consumer)
        if failure_phase == "before_update":

            async def fail_before_activation(**_kwargs):
                raise RuntimeError("injected activation write failure")

            dispatcher._durable_store.activate_initial_delivery = fail_before_activation
        else:

            async def fail_after_activation_update(**_kwargs):
                raise RuntimeError("injected activation readback failure")

            dispatcher._durable_store._delivery_for_lease_locked = (
                fail_after_activation_update
            )

        result = await dispatcher.dispatch_signal(
            _signal(agent_id=agent.did, message=secret),
            source_event_id=f"volatile-activation-{failure_phase}",
        )
        assert result.status is Status.FAILED
        assert "activation failed after commit" in (result.error or "")
        assert dispatcher._transient_durable_handoffs == {}

        stored = await dispatcher.list_durable_deliveries(
            consumer_id=consumer.consumer_id
        )
        assert len(stored) == 1
        assert stored[0].status == RETRY
        assert stored[0].lease_owner is None
        assert stored[0].lease_token is None
        assert stored[0].lease_expires_at is None

        # The injected readback fault is specific to activation; restore the
        # normal store read path before asserting ordinary marker replay.
        dispatcher._durable_store._delivery_for_lease_locked = original_readback
        replay = await dispatcher.claim_durable_delivery(
            consumer_id=consumer.consumer_id, executor_id="recovery-worker"
        )
        assert replay is not None
        assert replay.event.payload == {"_privacy_gated": "none"}
        assert secret not in json.dumps(replay.event.payload)
    finally:
        dispatcher._durable_store.activate_initial_delivery = original_activate
        dispatcher._durable_store._delivery_for_lease_locked = original_readback
        await dispatcher.shutdown_durable_delivery()
        await _close(backend, agent)


class _PostCommitAbort(BaseException):
    """Deliberate non-``Exception`` failure for cancellation-safety coverage."""


@pytest.mark.asyncio
@pytest.mark.parametrize("boundary", ("persist", "activate-first", "activate-second"))
@pytest.mark.parametrize("failure_kind", ("cancelled", "base-exception"))
async def test_post_commit_reservation_repair_survives_every_await_boundary_and_repeated_cancellation(
    tmp_path, boundary, failure_kind
):
    """Committed volatile work is marker-only after every activation boundary.

    The persistence boundary is the awkward one: the store's transaction can
    commit after its before-commit callback installs raw sidecars, then its
    caller can be cancelled before it receives ``DurableEventPersistence``.
    The two activation boundaries cover a partially activated multi-consumer
    batch.  In every case, a second and third cancellation land while the
    shielded repair task is awaiting its first durable release.
    """
    from kestrel_sovereign.privacy import get_privacy_preset

    backend, agent, dispatcher = await _dispatcher(
        tmp_path / f"post-commit-{boundary}-{failure_kind}.db", "did:agent:one"
    )
    agent.privacy_config = get_privacy_preset("ephemeral")
    consumers = (
        DurableConsumerRegistration(
            consumer_id="workflow-wait-a",
            source="provider.message",
            agent_id=agent.did,
        ),
        DurableConsumerRegistration(
            consumer_id="workflow-wait-b",
            source="provider.message",
            agent_id=agent.did,
        ),
    )
    original_persist = dispatcher._durable_store.persist_signal
    original_activate = dispatcher._durable_store.activate_initial_delivery
    original_abandon = dispatcher._durable_store.abandon_initial_reservation
    dispatch_task = None
    activation_count = 0
    repeated_cancellation_injected = False

    async def _raise_at_boundary() -> None:
        if failure_kind == "cancelled":
            asyncio.current_task().cancel()
            await asyncio.sleep(0)
        raise _PostCommitAbort(f"injected after committed {boundary} await")

    async def cancel_after_persist(*args, **kwargs):
        persistence = await original_persist(*args, **kwargs)
        if boundary == "persist" and persistence.created:
            await _raise_at_boundary()
        return persistence

    async def cancel_after_activation(**kwargs):
        nonlocal activation_count
        delivery = await original_activate(**kwargs)
        activation_count += 1
        if boundary == f"activate-{'first' if activation_count == 1 else 'second'}":
            await _raise_at_boundary()
        return delivery

    async def cancel_dispatch_again_during_repair(**kwargs):
        nonlocal repeated_cancellation_injected
        if not repeated_cancellation_injected:
            assert dispatch_task is not None
            repeated_cancellation_injected = True
            dispatch_task.cancel()
            dispatch_task.cancel()
        return await original_abandon(**kwargs)

    dispatcher._durable_store.persist_signal = cancel_after_persist
    dispatcher._durable_store.activate_initial_delivery = cancel_after_activation
    dispatcher._durable_store.abandon_initial_reservation = (
        cancel_dispatch_again_during_repair
    )
    try:
        for consumer in consumers:
            await dispatcher.register_durable_consumer(consumer)

        dispatch_task = asyncio.create_task(
            dispatcher.dispatch_signal(
                _signal(
                    agent_id=agent.did,
                    message=f"{boundary}-{failure_kind}-customer@example.com",
                ),
                source_event_id=f"post-commit-{boundary}-{failure_kind}",
            )
        )
        expected_error = (
            asyncio.CancelledError
            if failure_kind == "cancelled"
            else _PostCommitAbort
        )
        with pytest.raises(expected_error):
            await dispatch_task

        assert repeated_cancellation_injected
        assert dispatcher._transient_durable_handoffs == {}
        assert dispatcher._transient_durable_handoff_timers == {}
        deliveries = await dispatcher.list_durable_deliveries()
        assert len(deliveries) == 2
        assert all(delivery.status == RETRY for delivery in deliveries)
        assert all(delivery.lease_owner is None for delivery in deliveries)
        assert all(delivery.lease_token is None for delivery in deliveries)
        assert all(delivery.lease_expires_at is None for delivery in deliveries)

        # A normal shutdown must still synchronously stop the runtime owner;
        # the cancellation path cannot leak a live liveness record behind it.
        dispatcher._durable_store.persist_signal = original_persist
        dispatcher._durable_store.activate_initial_delivery = original_activate
        dispatcher._durable_store.abandon_initial_reservation = original_abandon
        await dispatcher.shutdown_durable_delivery()
        owner = await backend.fetch_one(
            "SELECT stopped_at FROM durable_signal_runtime_owners "
            "WHERE agent_id = ? AND owner_id = ?",
            (agent.did, dispatcher._durable_delivery_owner),
        )
        assert owner is not None and owner[0] is not None
        assert dispatcher._durable_runtime_owner_registered is False
    finally:
        dispatcher._durable_store.persist_signal = original_persist
        dispatcher._durable_store.activate_initial_delivery = original_activate
        dispatcher._durable_store.abandon_initial_reservation = original_abandon
        if not dispatcher._durable_shutdown:
            await dispatcher.shutdown_durable_delivery()
        await _close(backend, agent)


@pytest.mark.asyncio
@pytest.mark.parametrize("failure_kind", ("cancelled", "base-exception"))
async def test_real_sqlite_commit_boundary_repairs_ambiguous_persistence_outcome(
    tmp_path, failure_kind
):
    """A driver commit that succeeds before it raises cannot strand a row.

    This drives the production SQLite transaction implementation rather than
    wrapping ``persist_signal`` after it returns.  The patched ``commit``
    finishes the aiosqlite worker operation, then injects cancellation (or a
    non-``Exception`` failure) while the backend context manager is still
    unwinding.  ``DurableSignalStore`` consequently invokes ``on_rollback``
    despite the event being visible; the dispatcher must retain and repair the
    owner/token capability instead of trusting that callback as proof of a
    rollback.
    """
    from kestrel_sovereign.privacy import get_privacy_preset

    backend, agent, dispatcher = await _dispatcher(
        tmp_path / f"sqlite-commit-boundary-{failure_kind}.db", "did:agent:one"
    )
    agent.privacy_config = get_privacy_preset("ephemeral")
    consumer = DurableConsumerRegistration(
        consumer_id="workflow-wait", source="provider.message", agent_id=agent.did
    )
    connection = backend._connection
    assert connection is not None
    original_commit = connection.commit
    original_persist = dispatcher._durable_store.persist_signal
    persistence_in_flight = False

    async def fail_after_committed_sqlite_write():
        await original_commit()
        if persistence_in_flight:
            if failure_kind == "cancelled":
                task = asyncio.current_task()
                assert task is not None
                task.cancel()
                await asyncio.sleep(0)
            raise _PostCommitAbort("injected after SQLite commit completed")

    async def persist_with_armed_commit(*args, **kwargs):
        nonlocal persistence_in_flight
        persistence_in_flight = True
        try:
            return await original_persist(*args, **kwargs)
        finally:
            persistence_in_flight = False

    try:
        await dispatcher.register_durable_consumer(consumer)
        connection.commit = fail_after_committed_sqlite_write
        dispatcher._durable_store.persist_signal = persist_with_armed_commit

        expected = (
            asyncio.CancelledError
            if failure_kind == "cancelled"
            else _PostCommitAbort
        )
        with pytest.raises(expected):
            await dispatcher.dispatch_signal(
                _signal(
                    agent_id=agent.did,
                    message=f"sqlite-commit-{failure_kind}@example.com",
                ),
                source_event_id=f"sqlite-commit-boundary-{failure_kind}",
            )

        # The repair is awaited before the original error/cancellation escapes:
        # no raw sidecar and no initial reservation can outlive this boundary.
        assert dispatcher._transient_durable_handoffs == {}
        assert dispatcher._transient_durable_handoff_timers == {}
        connection.commit = original_commit
        dispatcher._durable_store.persist_signal = original_persist
        deliveries = await dispatcher.list_durable_deliveries(
            consumer_id=consumer.consumer_id
        )
        assert len(deliveries) == 1
        delivery = deliveries[0]
        assert delivery.status == RETRY
        assert delivery.lease_owner is None
        assert delivery.lease_token is None
        assert delivery.lease_expires_at is None
    finally:
        connection.commit = original_commit
        dispatcher._durable_store.persist_signal = original_persist
        if not dispatcher._durable_shutdown:
            await dispatcher.shutdown_durable_delivery()
        await _close(backend, agent)


@pytest.mark.asyncio
async def test_heartbeat_recovers_reservation_that_becomes_stale_after_restart(tmp_path):
    """A fresh crash is retried after its owner becomes stale, without restart.

    The first recovery pass deliberately leaves the just-crashed owner's
    reservation alone.  The restarted dispatcher must sweep it on a later
    owner heartbeat; generic delivery claims continue to ignore the reserved
    row until that owner-aware recovery makes it marker-only retry work.
    """
    path = tmp_path / "volatile-recovery-after-stale.db"
    backend = SQLiteBackend(str(path))
    await backend.connect()
    log_store = SignalLogStore(backend)
    await log_store.initialize()
    crashed_store = DurableSignalStore(backend)
    await crashed_store.initialize()
    agent = _Agent("did:agent:one")
    registry = SourceRegistry()
    registry.register(_registration(agent))
    dispatcher = SignalDispatcher(
        agent=agent,
        registry=registry,
        lock_manager=OrderedLockManager(),
        store=log_store,
        durable_store=DurableSignalStore(backend),
        runtime_owner_stale_after=timedelta(minutes=2),
    )
    consumer = DurableConsumerRegistration(
        consumer_id="workflow-wait", source="provider.message", agent_id=agent.did
    )
    # Recovery is intentionally restricted to the dispatcher ownership
    # namespace. A public executor lease must never be inferred stale merely
    # because it lacks a runtime-owner heartbeat.
    crashed_owner = "dispatcher:crashed-owner"
    try:
        await crashed_store.register_consumer(consumer)
        await crashed_store.register_runtime_owner(
            agent_id=agent.did, owner_id=crashed_owner
        )
        marker_event = _signal(agent_id=agent.did)
        marker_event.payload = {"_privacy_gated": "none"}
        persisted = await crashed_store.persist_signal(
            marker_event,
            agent_id=agent.did,
            source_event_id="fresh-crash-before-activation",
            retention_days=7,
            initial_lease_owner=crashed_owner,
        )
        reservation = persisted.initial_reservations[0]

        # This is the immediate restart: the old heartbeat is still fresh, so
        # startup recovery correctly preserves the unactivated reservation.
        await dispatcher.initialize_durable_delivery()
        initial = await crashed_store.get_delivery(
            agent_id=agent.did,
            consumer_id=consumer.consumer_id,
            delivery_id=reservation.delivery_id,
        )
        assert initial is not None and initial.status == "initial_reserved"
        assert await dispatcher.claim_durable_delivery(
            consumer_id=consumer.consumer_id, executor_id="before-stale"
        ) is None

        # No second dispatcher restart occurs. Advance the crashed owner's
        # persisted liveness past the threshold, then drive the same coroutine
        # the scheduled heartbeat invokes. This avoids a wall-clock sleep while
        # exercising recovery after the initially-fresh startup pass.
        stale_at = datetime.now(timezone.utc) - timedelta(minutes=3)
        await backend.execute(
            "UPDATE durable_signal_runtime_owners SET heartbeat_at = ? "
            "WHERE agent_id = ? AND owner_id = ?",
            (
                crashed_store.to_timestamp_param(stale_at),
                agent.did,
                crashed_owner,
            ),
        )
        await dispatcher._heartbeat_runtime_owner()
        recovered = await crashed_store.get_delivery(
            agent_id=agent.did,
            consumer_id=consumer.consumer_id,
            delivery_id=reservation.delivery_id,
        )
        assert recovered is not None and recovered.status == RETRY
        replay = await dispatcher.claim_durable_delivery(
            consumer_id=consumer.consumer_id, executor_id="after-stale"
        )
        assert replay is not None
        assert replay.event.payload == {"_privacy_gated": "none"}
    finally:
        await dispatcher.shutdown_durable_delivery()
        await _close(backend, agent)


@pytest.mark.asyncio
async def test_runtime_owner_heartbeat_failure_keeps_a_bounded_future_retry(tmp_path):
    """A transient owner-heartbeat exception cannot silently abandon liveness."""
    backend, agent, dispatcher = await _dispatcher(
        tmp_path / "owner-heartbeat-retry.db", "did:agent:one"
    )
    original = dispatcher._durable_store.heartbeat_runtime_owner
    attempts = 0

    async def fail_once(*args, **kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("temporary durable store outage")
        return await original(*args, **kwargs)

    dispatcher._durable_store.heartbeat_runtime_owner = fail_once
    try:
        dispatcher._start_runtime_owner_heartbeat()
        task = dispatcher._runtime_owner_heartbeat_task
        assert task is not None
        await asyncio.gather(task, return_exceptions=True)
        await asyncio.sleep(0)

        retry_timer = dispatcher._runtime_owner_heartbeat_timer
        assert attempts == 1
        assert dispatcher._runtime_owner_heartbeat_failures == 1
        assert retry_timer is not None and not retry_timer.cancelled()
        # The retry is deliberately bounded below the normal stale-owner
        # cadence, so a storage outage is observable without a hot loop.
        assert 0 < retry_timer.when() - asyncio.get_running_loop().time() <= (
            dispatcher._runtime_owner_stale_after.total_seconds() / 3
        )
    finally:
        dispatcher._durable_store.heartbeat_runtime_owner = original
        await dispatcher.shutdown_durable_delivery()
        await _close(backend, agent)


@pytest.mark.asyncio
async def test_recovery_never_steals_a_public_executor_lease(tmp_path):
    """Only proven stale managed dispatcher owners are recovery candidates."""
    backend, agent, dispatcher = await _dispatcher(
        tmp_path / "public-executor-lease.db", "did:agent:one"
    )
    consumer = DurableConsumerRegistration(
        consumer_id="workflow-wait", source="provider.message", agent_id=agent.did
    )
    try:
        await dispatcher.register_durable_consumer(consumer)
        assert (
            await dispatcher.dispatch_signal(
                _signal(agent_id=agent.did), source_event_id="public-executor-lease"
            )
        ).status is Status.OK
        claimed = await dispatcher.claim_durable_delivery(
            consumer_id=consumer.consumer_id,
            executor_id="public-executor:external-worker",
        )
        assert claimed is not None and claimed.status == LEASED

        released = await dispatcher._durable_store.recover_abandoned_leases(
            agent_id=agent.did,
            recovering_owner_id=dispatcher._durable_delivery_owner,
            stale_before=datetime.now(timezone.utc) + timedelta(days=1),
        )
        assert released == 0
        still_owned = await dispatcher.get_durable_delivery_for_event(
            consumer_id=consumer.consumer_id, event_id=claimed.event.event_id
        )
        assert still_owned is not None
        assert still_owned.status == LEASED
        assert still_owned.lease_owner == "public-executor:external-worker"
    finally:
        await dispatcher.shutdown_durable_delivery()
        await _close(backend, agent)


@pytest.mark.asyncio
async def test_partial_durable_init_marks_registered_owner_stopped_on_teardown(tmp_path):
    """Recovery failure after registration cannot leak a live owner record."""
    backend = SQLiteBackend(str(tmp_path / "partial-owner-init.db"))
    await backend.connect()
    log_store = SignalLogStore(backend)
    await log_store.initialize()
    agent = _Agent("did:agent:one")
    registry = SourceRegistry()
    registry.register(_registration(agent))
    dispatcher = SignalDispatcher(
        agent=agent,
        registry=registry,
        lock_manager=OrderedLockManager(),
        store=log_store,
    )
    original_recovery = dispatcher._durable_store.recover_abandoned_initial_reservations

    async def fail_startup_recovery(**_kwargs):
        raise RuntimeError("injected startup recovery failure")

    dispatcher._durable_store.recover_abandoned_initial_reservations = (
        fail_startup_recovery
    )
    try:
        with pytest.raises(RuntimeError, match="injected startup recovery failure"):
            await dispatcher.initialize_durable_delivery()
        assert dispatcher._durable_initialized is False
        assert dispatcher._durable_runtime_owner_registered is True

        await dispatcher.shutdown_durable_delivery()

        owner = await backend.fetch_one(
            "SELECT stopped_at FROM durable_signal_runtime_owners "
            "WHERE agent_id = ? AND owner_id = ?",
            (agent.did, dispatcher._durable_delivery_owner),
        )
        assert owner is not None and owner[0] is not None
        assert dispatcher._durable_runtime_owner_registered is False
    finally:
        dispatcher._durable_store.recover_abandoned_initial_reservations = (
            original_recovery
        )
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
        completion = dispatcher_a._durable_shutdown_completion
        assert completion is not None
        await completion
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
        await dispatcher_b.shutdown_durable_delivery()
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
@pytest.mark.parametrize("exact_event_claim", (False, True), ids=("ordinary", "exact"))
async def test_implicit_claim_clock_is_sampled_after_contended_scope_handoff(
    tmp_path, exact_event_claim
):
    """A contended implicit claim gets a full lease from the serialized clock."""

    path = tmp_path / f"contended-claim-{exact_event_claim}.db"
    backend = SQLiteBackend(str(path))
    peer = SQLiteBackend(str(path))
    await backend.connect()
    await peer.connect()
    store = DurableSignalStore(backend)
    await store.initialize()
    agent_id = "did:agent:contended"
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    clock = {"now": base}
    store.now_utc = lambda: clock["now"]  # type: ignore[method-assign]
    consumer = DurableConsumerRegistration(
        consumer_id="contended-worker",
        source="provider.message",
        agent_id=agent_id,
        lease_seconds=1,
    )
    await store.register_consumer(consumer)
    persisted = await store.persist_signal(
        _signal(agent_id=agent_id),
        agent_id=agent_id,
        source_event_id="provider:contended-claim",
        retention_days=7,
    )
    entered_handoff = asyncio.Event()
    writer_acquired = asyncio.Event()
    release_writer = asyncio.Event()
    original_handoff = store._lock_scope_handoff

    async def observe_handoff(**kwargs):
        entered_handoff.set()
        await original_handoff(**kwargs)

    store._lock_scope_handoff = observe_handoff  # type: ignore[method-assign]

    async def hold_peer_writer():
        async with peer.transaction():
            await peer.execute("DELETE FROM durable_signal_consumers WHERE 0")
            writer_acquired.set()
            await release_writer.wait()

    blocker = asyncio.create_task(hold_peer_writer())
    try:
        await asyncio.wait_for(writer_acquired.wait(), timeout=1)
        if exact_event_claim:
            claim_task = asyncio.create_task(
                store.claim_delivery_for_event(
                    agent_id=agent_id,
                    consumer_id=consumer.consumer_id,
                    event_id=persisted.event_id,
                    executor_id="worker",
                )
            )
        else:
            claim_task = asyncio.create_task(
                store.claim_delivery(
                    agent_id=agent_id,
                    consumer_id=consumer.consumer_id,
                    executor_id="worker",
                )
            )
        await asyncio.wait_for(entered_handoff.wait(), timeout=1)
        # The method is now blocked at the database handoff, after its caller
        # began but before it owns the serialization point.
        clock["now"] = base + timedelta(seconds=2)
        release_writer.set()
        claimed = await asyncio.wait_for(claim_task, timeout=1)
        assert claimed is not None
        assert claimed.lease_expires_at == base + timedelta(seconds=3)
    finally:
        release_writer.set()
        await asyncio.gather(blocker, return_exceptions=True)
        await peer.close()
        await backend.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("exact_event_claim", (False, True), ids=("ordinary", "exact"))
async def test_implicit_claim_clock_waits_for_selected_delivery_serialization(
    tmp_path, exact_event_claim
):
    """A row lock longer than the lease cannot publish an expired implicit claim."""

    backend = SQLiteBackend(str(tmp_path / f"row-claim-{exact_event_claim}.db"))
    await backend.connect()
    store = DurableSignalStore(backend)
    await store.initialize()
    agent_id = "did:agent:row-contention"
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    clock = {"now": base}
    store.now_utc = lambda: clock["now"]  # type: ignore[method-assign]
    consumer = DurableConsumerRegistration(
        consumer_id="row-contention-worker",
        source="provider.message",
        agent_id=agent_id,
        lease_seconds=1,
    )
    await store.register_consumer(consumer)
    persisted = await store.persist_signal(
        _signal(agent_id=agent_id),
        agent_id=agent_id,
        source_event_id="provider:row-contention",
        retention_days=7,
    )
    selected = asyncio.Event()
    release_selected = asyncio.Event()
    original_lock = store._lock_claimable_delivery

    async def hold_selected_delivery(**kwargs):
        delivery_id = await original_lock(**kwargs)
        assert delivery_id is not None
        selected.set()
        await release_selected.wait()
        return delivery_id

    store._lock_claimable_delivery = hold_selected_delivery  # type: ignore[method-assign]
    try:
        if exact_event_claim:
            claim_task = asyncio.create_task(
                store.claim_delivery_for_event(
                    agent_id=agent_id,
                    consumer_id=consumer.consumer_id,
                    event_id=persisted.event_id,
                    executor_id="worker",
                )
            )
        else:
            claim_task = asyncio.create_task(
                store.claim_delivery(
                    agent_id=agent_id,
                    consumer_id=consumer.consumer_id,
                    executor_id="worker",
                )
            )
        await asyncio.wait_for(selected.wait(), timeout=1)
        # This models a PostgreSQL SELECT .. FOR UPDATE blocked beyond the
        # entire nominal lease. The later clock must determine the lease.
        clock["now"] = base + timedelta(seconds=2)
        release_selected.set()
        claimed = await asyncio.wait_for(claim_task, timeout=1)
        assert claimed is not None
        assert claimed.lease_expires_at == base + timedelta(seconds=3)

        # Explicit caller time remains an exact deterministic contract even
        # when row serialization waits.
        explicit = base + timedelta(seconds=10)
        assert await store.nack_delivery(
            agent_id=agent_id,
            consumer_id=consumer.consumer_id,
            delivery_id=claimed.delivery_id,
            lease_token=claimed.lease_token or "",
            error="make explicit retry due",
            now=clock["now"],
        ) is not None
        explicit_claim = (
            await store.claim_delivery_for_event(
                agent_id=agent_id,
                consumer_id=consumer.consumer_id,
                event_id=persisted.event_id,
                executor_id="worker-explicit",
                now=explicit,
            )
            if exact_event_claim
            else await store.claim_delivery(
                agent_id=agent_id,
                consumer_id=consumer.consumer_id,
                executor_id="worker-explicit",
                now=explicit,
            )
        )
        assert explicit_claim is not None
        assert explicit_claim.lease_expires_at == explicit + timedelta(seconds=1)
    finally:
        release_selected.set()
        await backend.close()


@pytest.mark.asyncio
async def test_registration_backfill_clock_is_sampled_after_contended_scope_handoff(tmp_path):
    """Registration does not backfill an event that expires while it waits for a writer."""

    path = tmp_path / "contended-registration-backfill.db"
    backend = SQLiteBackend(str(path))
    peer = SQLiteBackend(str(path))
    await backend.connect()
    await peer.connect()
    store = DurableSignalStore(backend)
    await store.initialize()
    agent_id = "did:agent:contended-registration"
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    clock = {"now": base}
    store.now_utc = lambda: clock["now"]  # type: ignore[method-assign]
    persisted = await store.persist_signal(
        _signal(agent_id=agent_id),
        agent_id=agent_id,
        source_event_id="provider:contended-registration",
        retention_days=7,
    )
    await backend.execute(
        "UPDATE durable_signal_events SET retention_until = ? WHERE event_id = ?",
        (base + timedelta(seconds=1), persisted.event_id),
    )
    consumer = DurableConsumerRegistration(
        consumer_id="late-registration",
        source="provider.message",
        agent_id=agent_id,
    )
    entered_handoff = asyncio.Event()
    writer_acquired = asyncio.Event()
    release_writer = asyncio.Event()
    original_handoff = store._lock_scope_handoff

    async def observe_handoff(**kwargs):
        entered_handoff.set()
        await original_handoff(**kwargs)

    store._lock_scope_handoff = observe_handoff  # type: ignore[method-assign]

    async def hold_peer_writer():
        async with peer.transaction():
            await peer.execute("DELETE FROM durable_signal_consumers WHERE 0")
            writer_acquired.set()
            await release_writer.wait()

    blocker = asyncio.create_task(hold_peer_writer())
    try:
        await asyncio.wait_for(writer_acquired.wait(), timeout=1)
        registration_task = asyncio.create_task(store.register_consumer(consumer))
        await asyncio.wait_for(entered_handoff.wait(), timeout=1)
        clock["now"] = base + timedelta(seconds=2)
        release_writer.set()
        await asyncio.wait_for(registration_task, timeout=1)
        assert await store.list_deliveries(agent_id=agent_id) == []
    finally:
        release_writer.set()
        await asyncio.gather(blocker, return_exceptions=True)
        await peer.close()
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
        await dispatcher.shutdown_durable_delivery()
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
        await dispatcher.shutdown_durable_delivery()
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
        await dispatcher.shutdown_durable_delivery()
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
async def test_postgres_durable_owner_and_recovery_binds_preserve_aware_utc_instants():
    """TIMESTAMPTZ owner/lease operations retain instants, unlike naive TIMESTAMP binds."""
    from kestrel_sovereign.storage.db.postgres import PostgresBackend

    captured: list[tuple[str, tuple]] = []

    class _PostgresBackend:
        backend_type = "postgres"
        fetch_val = AsyncMock(return_value=None)

        @asynccontextmanager
        async def transaction(self):
            yield

        async def execute(self, query, params=()):
            captured.append((query, PostgresBackend._strip_tz(params)))
            return 1

    store = DurableSignalStore(_PostgresBackend())
    central = datetime(2026, 8, 10, 7, tzinfo=timezone(timedelta(hours=-5)))
    expected = datetime(2026, 8, 10, 12, tzinfo=timezone.utc)

    await store.heartbeat_runtime_owner(
        agent_id="did:agent:one", owner_id="dispatcher:owner", now=central
    )
    await store.recover_abandoned_leases(
        agent_id="did:agent:one",
        recovering_owner_id="dispatcher:recovery",
        stale_before=central - timedelta(minutes=2),
        now=central,
    )

    bound_instants = [
        value
        for _query, params in captured
        for value in params
        if isinstance(value, datetime)
    ]
    assert bound_instants
    assert all(value.tzinfo == timezone.utc for value in bound_instants)
    assert expected in bound_instants
    assert all("::TIMESTAMPTZ" in query for query, _params in captured[1:])

    # The generic Postgres path retains its legacy naive TIMESTAMP behavior.
    naive = datetime(2026, 8, 10, 12)
    legacy_aware = datetime(2026, 8, 10, 7, tzinfo=timezone(timedelta(hours=-5)))
    assert PostgresBackend._strip_tz((naive, legacy_aware)) == (
        naive,
        legacy_aware.replace(tzinfo=None),
    )

    class _SQLiteBackend:
        backend_type = "sqlite"

    assert DurableSignalStore(_SQLiteBackend()).to_timestamp_param(central) == central.isoformat()


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


def test_sync_shutdown_without_a_running_loop_fails_before_claiming_cleanup():
    """The compatibility seam must not claim a durable release it cannot run."""
    dispatcher = SignalDispatcher(
        agent=SimpleNamespace(did="did:agent:no-loop"),
        registry=SourceRegistry(),
        lock_manager=OrderedLockManager(),
        store=SimpleNamespace(backend=object()),
    )

    with pytest.raises(RuntimeError, match="requires a running event loop"):
        dispatcher.shutdown()

    assert dispatcher._durable_shutdown is False
    assert dispatcher._durable_shutdown_completion is None


@pytest.mark.asyncio
async def test_sync_shutdown_drains_an_admitted_persistence_before_releasing_owner(
    tmp_path,
):
    """The sync seam linearizes with a real SQLite event transaction."""
    backend, agent, dispatcher = await _dispatcher(
        tmp_path / "sync-shutdown-admitted-persistence.db", "did:agent:one"
    )
    consumer = DurableConsumerRegistration(
        consumer_id="workflow-wait", source="provider.message", agent_id=agent.did
    )
    original_persist = dispatcher._durable_store.persist_signal
    persist_entered = asyncio.Event()
    allow_persist = asyncio.Event()
    dispatch_task = None
    async_shutdown_task = None
    completion = None
    backend_closed = False

    async def block_persist(*args, **kwargs):
        persist_entered.set()
        await allow_persist.wait()
        return await original_persist(*args, **kwargs)

    dispatcher._durable_store.persist_signal = block_persist
    try:
        await dispatcher.register_durable_consumer(consumer)
        dispatch_task = asyncio.create_task(
            dispatcher.dispatch_signal(
                _signal(agent_id=agent.did, message="sync-admitted-persistence"),
                source_event_id="sync-admitted-persistence",
            )
        )
        await asyncio.wait_for(persist_entered.wait(), timeout=1.0)

        dispatcher.shutdown()
        completion = dispatcher._durable_shutdown_completion
        assert completion is not None and not completion.done()
        dispatcher.shutdown()
        assert dispatcher._durable_shutdown_completion is completion
        async_shutdown_task = asyncio.create_task(dispatcher.shutdown_durable_delivery())
        await asyncio.sleep(0)
        assert dispatcher._durable_shutdown_completion is completion
        assert not async_shutdown_task.done()
        await _assert_late_durable_calls_fail_safely(
            dispatcher, agent=agent, consumer=consumer
        )

        allow_persist.set()
        assert (await dispatch_task).status is Status.OK
        await async_shutdown_task
        await _assert_sync_shutdown_drained(backend, agent, dispatcher, completion)
        backend_closed = True
    finally:
        allow_persist.set()
        dispatcher._durable_store.persist_signal = original_persist
        if dispatch_task is not None and not dispatch_task.done():
            await dispatch_task
        if async_shutdown_task is not None and not async_shutdown_task.done():
            await async_shutdown_task
        if not backend_closed:
            await _finish_sync_shutdown_test(backend, agent, dispatcher)


@pytest.mark.asyncio
async def test_sync_shutdown_drains_an_admitted_outcome_log_before_releasing_owner(
    tmp_path,
):
    """The sync seam cannot discard sidecars while an accepted log is live."""
    backend, agent, dispatcher = await _dispatcher(
        tmp_path / "sync-shutdown-admitted-log.db", "did:agent:one"
    )
    consumer = DurableConsumerRegistration(
        consumer_id="workflow-wait", source="provider.message", agent_id=agent.did
    )
    original_append = dispatcher._store.append
    append_entered = asyncio.Event()
    allow_append = asyncio.Event()
    dispatch_task = None
    completion = None
    backend_closed = False

    async def block_append(*args, **kwargs):
        append_entered.set()
        await allow_append.wait()
        return await original_append(*args, **kwargs)

    dispatcher._store.append = block_append
    try:
        dispatch_task = asyncio.create_task(
            dispatcher.dispatch_signal(
                _signal(agent_id=agent.did, message="sync-admitted-log"),
                source_event_id="sync-admitted-log",
            )
        )
        await asyncio.wait_for(append_entered.wait(), timeout=1.0)

        dispatcher.shutdown()
        completion = dispatcher._durable_shutdown_completion
        assert completion is not None and not completion.done()
        assert dispatcher._outcome_log_tasks
        await _assert_late_durable_calls_fail_safely(
            dispatcher, agent=agent, consumer=consumer
        )

        allow_append.set()
        assert (await dispatch_task).status is Status.OK
        await _assert_sync_shutdown_drained(backend, agent, dispatcher, completion)
        backend_closed = True
    finally:
        allow_append.set()
        dispatcher._store.append = original_append
        if dispatch_task is not None and not dispatch_task.done():
            await dispatch_task
        if not backend_closed:
            await _finish_sync_shutdown_test(backend, agent, dispatcher)


@pytest.mark.asyncio
async def test_sync_shutdown_drains_an_admitted_initialization_before_releasing_owner(
    tmp_path,
):
    """An owner registration admitted before sync shutdown is always stopped."""
    backend = SQLiteBackend(str(tmp_path / "sync-shutdown-initialization.db"))
    await backend.connect()
    log_store = SignalLogStore(backend)
    await log_store.initialize()
    agent = _Agent("did:agent:one")
    registry = SourceRegistry()
    registry.register(_registration(agent))
    dispatcher = SignalDispatcher(
        agent=agent,
        registry=registry,
        lock_manager=OrderedLockManager(),
        store=log_store,
    )
    original_register = dispatcher._durable_store.register_runtime_owner
    registration_entered = asyncio.Event()
    allow_registration = asyncio.Event()
    initialization_task = None
    completion = None
    backend_closed = False

    async def block_registration(*args, **kwargs):
        registration_entered.set()
        await allow_registration.wait()
        return await original_register(*args, **kwargs)

    dispatcher._durable_store.register_runtime_owner = block_registration
    try:
        initialization_task = asyncio.create_task(dispatcher.initialize_durable_delivery())
        await asyncio.wait_for(registration_entered.wait(), timeout=1.0)

        dispatcher.shutdown()
        completion = dispatcher._durable_shutdown_completion
        assert completion is not None and not completion.done()

        allow_registration.set()
        await initialization_task
        await _assert_sync_shutdown_drained(backend, agent, dispatcher, completion)
        backend_closed = True
    finally:
        allow_registration.set()
        dispatcher._durable_store.register_runtime_owner = original_register
        if initialization_task is not None and not initialization_task.done():
            await initialization_task
        if not backend_closed:
            await _finish_sync_shutdown_test(backend, agent, dispatcher)


@pytest.mark.asyncio
async def test_shutdown_waits_for_admitted_dispatch_then_rejects_late_storage_calls(
    tmp_path,
):
    """A dispatch admitted before closure finishes before owner release.

    The persistence barrier makes this a real-SQLite race: shutdown has
    closed admission while the event transaction has not yet begun.  It must
    not release the owner or permit storage close until the admitted dispatch
    has committed and its lifecycle admission has drained.
    """
    from tests.utils.aiosqlite_workers import aiosqlite_worker

    backend, agent, dispatcher = await _dispatcher(
        tmp_path / "shutdown-admitted-dispatch.db", "did:agent:one"
    )
    consumer = DurableConsumerRegistration(
        consumer_id="workflow-wait", source="provider.message", agent_id=agent.did
    )
    original_persist = dispatcher._durable_store.persist_signal
    persist_entered = asyncio.Event()
    allow_persist = asyncio.Event()
    dispatch_task = None
    shutdown_task = None
    backend_closed = False

    async def persist_after_shutdown_admission(*args, **kwargs):
        persist_entered.set()
        await allow_persist.wait()
        return await original_persist(*args, **kwargs)

    dispatcher._durable_store.persist_signal = persist_after_shutdown_admission
    try:
        await dispatcher.register_durable_consumer(consumer)
        dispatch_task = asyncio.create_task(
            dispatcher.dispatch_signal(
                _signal(agent_id=agent.did, message="admitted-before-shutdown"),
                source_event_id="admitted-before-shutdown",
            )
        )
        await asyncio.wait_for(persist_entered.wait(), timeout=1.0)

        shutdown_task = asyncio.create_task(dispatcher.shutdown_durable_delivery())
        await asyncio.sleep(0)
        assert not shutdown_task.done()
        assert dispatcher._durable_runtime_owner_registered is True

        # The close linearization point has passed.  None of these calls may
        # reach the real database while shutdown waits for the admitted write.
        await _assert_late_durable_calls_fail_safely(
            dispatcher, agent=agent, consumer=consumer
        )

        allow_persist.set()
        result = await dispatch_task
        assert result.status is Status.OK
        await shutdown_task

        owner = await backend.fetch_one(
            "SELECT stopped_at FROM durable_signal_runtime_owners "
            "WHERE agent_id = ? AND owner_id = ?",
            (agent.did, dispatcher._durable_delivery_owner),
        )
        assert owner is not None and owner[0] is not None
        assert dispatcher._durable_runtime_owner_registered is False
        assert dispatcher._transient_durable_handoffs == {}
        assert dispatcher._transient_durable_handoff_timers == {}
        assert dispatcher._post_commit_reservation_repairs == set()
        assert dispatcher._runtime_owner_heartbeat_timer is None
        assert all(task.done() for task in agent.tasks)
        assert await backend.fetch_val(
            "SELECT COUNT(*) FROM durable_signal_deliveries "
            "WHERE status = 'initial_reserved'"
        ) == 0

        # The lifecycle guard remains effective after the actual SQLite
        # connection is closed; a regression would now fail with a backend
        # connection error instead of the deliberate shutdown outcome.
        connection = backend._connection
        assert connection is not None
        worker = aiosqlite_worker(connection)
        await backend.close()
        backend_closed = True
        assert not worker.is_alive()
        await _assert_late_durable_calls_fail_safely(
            dispatcher, agent=agent, consumer=consumer
        )
    finally:
        allow_persist.set()
        dispatcher._durable_store.persist_signal = original_persist
        if dispatch_task is not None and not dispatch_task.done():
            await dispatch_task
        if shutdown_task is not None and not shutdown_task.done():
            await shutdown_task
        if not dispatcher._durable_shutdown:
            await dispatcher.shutdown_durable_delivery()
        if not backend_closed:
            await _close(backend, agent)


@pytest.mark.asyncio
async def test_shutdown_serializes_an_inflight_runtime_owner_registration(tmp_path):
    """A registration that crosses shutdown is released before close returns."""
    backend = SQLiteBackend(str(tmp_path / "shutdown-owner-registration.db"))
    await backend.connect()
    log_store = SignalLogStore(backend)
    await log_store.initialize()
    agent = _Agent("did:agent:one")
    registry = SourceRegistry()
    registry.register(_registration(agent))
    dispatcher = SignalDispatcher(
        agent=agent,
        registry=registry,
        lock_manager=OrderedLockManager(),
        store=log_store,
    )
    consumer = DurableConsumerRegistration(
        consumer_id="workflow-wait", source="provider.message", agent_id=agent.did
    )
    original_register = dispatcher._durable_store.register_runtime_owner
    registration_entered = asyncio.Event()
    allow_registration = asyncio.Event()
    initialization_task = None
    shutdown_task = None

    async def register_after_shutdown_admission(*args, **kwargs):
        registration_entered.set()
        await allow_registration.wait()
        return await original_register(*args, **kwargs)

    dispatcher._durable_store.register_runtime_owner = register_after_shutdown_admission
    try:
        initialization_task = asyncio.create_task(
            dispatcher.initialize_durable_delivery()
        )
        await asyncio.wait_for(registration_entered.wait(), timeout=1.0)

        shutdown_task = asyncio.create_task(dispatcher.shutdown_durable_delivery())
        await asyncio.sleep(0)
        assert not shutdown_task.done()

        # Closing admission blocks a concurrent public initialization as well
        # as a full durable API operation while the first registration owns its
        # pre-close admission.
        with pytest.raises(RuntimeError, match="shutting down"):
            await dispatcher.initialize_durable_delivery()
        await _assert_late_durable_calls_fail_safely(
            dispatcher, agent=agent, consumer=consumer
        )

        allow_registration.set()
        await initialization_task
        await shutdown_task

        owner = await backend.fetch_one(
            "SELECT stopped_at FROM durable_signal_runtime_owners "
            "WHERE agent_id = ? AND owner_id = ?",
            (agent.did, dispatcher._durable_delivery_owner),
        )
        assert owner is not None and owner[0] is not None
        assert dispatcher._durable_initialized is True
        assert dispatcher._durable_runtime_owner_registered is False
        assert dispatcher._runtime_owner_heartbeat_timer is None
        assert dispatcher._transient_durable_handoffs == {}
        assert dispatcher._post_commit_reservation_repairs == set()
    finally:
        allow_registration.set()
        dispatcher._durable_store.register_runtime_owner = original_register
        if initialization_task is not None and not initialization_task.done():
            await initialization_task
        if shutdown_task is not None and not shutdown_task.done():
            await shutdown_task
        if not dispatcher._durable_shutdown:
            await dispatcher.shutdown_durable_delivery()
        await _close(backend, agent)


@pytest.mark.asyncio
async def test_shutdown_drains_an_admitted_owner_heartbeat_before_release(tmp_path):
    """A queued heartbeat cannot revive the owner after teardown begins."""
    backend, agent, dispatcher = await _dispatcher(
        tmp_path / "shutdown-owner-heartbeat.db", "did:agent:one"
    )
    original_heartbeat = dispatcher._durable_store.heartbeat_runtime_owner
    heartbeat_entered = asyncio.Event()
    allow_heartbeat = asyncio.Event()
    heartbeat_task = None
    shutdown_task = None

    async def heartbeat_after_shutdown_admission(*args, **kwargs):
        heartbeat_entered.set()
        await allow_heartbeat.wait()
        return await original_heartbeat(*args, **kwargs)

    dispatcher._durable_store.heartbeat_runtime_owner = heartbeat_after_shutdown_admission
    try:
        heartbeat_task = asyncio.create_task(dispatcher._heartbeat_runtime_owner())
        await asyncio.wait_for(heartbeat_entered.wait(), timeout=1.0)

        shutdown_task = asyncio.create_task(dispatcher.shutdown_durable_delivery())
        await asyncio.sleep(0)
        assert not shutdown_task.done()

        allow_heartbeat.set()
        await heartbeat_task
        await shutdown_task

        owner = await backend.fetch_one(
            "SELECT stopped_at FROM durable_signal_runtime_owners "
            "WHERE agent_id = ? AND owner_id = ?",
            (agent.did, dispatcher._durable_delivery_owner),
        )
        assert owner is not None and owner[0] is not None

        # A timer callback that wins the event-loop turn after closure reaches
        # the same gate and cannot issue a write against closed storage.
        with pytest.raises(RuntimeError, match="shutting down"):
            await dispatcher._heartbeat_runtime_owner()
    finally:
        allow_heartbeat.set()
        dispatcher._durable_store.heartbeat_runtime_owner = original_heartbeat
        if heartbeat_task is not None and not heartbeat_task.done():
            await heartbeat_task
        if shutdown_task is not None and not shutdown_task.done():
            await shutdown_task
        if not dispatcher._durable_shutdown:
            await dispatcher.shutdown_durable_delivery()
        await _close(backend, agent)


@pytest.mark.asyncio
async def test_shutdown_retry_survives_cancelled_caller(tmp_path):
    """Caller cancellation cannot strand the owned teardown task."""
    backend, agent, dispatcher = await _dispatcher(
        tmp_path / "shutdown-cancel-retry.db", "did:agent:one"
    )
    original_release = dispatcher._durable_store.release_initial_reservations
    release_entered = asyncio.Event()
    allow_release = asyncio.Event()
    shutdown_task = None

    async def block_release(*args, **kwargs):
        release_entered.set()
        await allow_release.wait()
        return await original_release(*args, **kwargs)

    dispatcher._durable_store.release_initial_reservations = block_release
    try:
        shutdown_task = asyncio.create_task(dispatcher.shutdown_durable_delivery())
        await asyncio.wait_for(release_entered.wait(), timeout=1.0)

        shutdown_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await shutdown_task

        # The caller was cancelled, not the dispatcher-owned cleanup.  A
        # retry must join it instead of re-raising a cached CancelledError.
        completion = dispatcher._durable_shutdown_completion
        assert completion is not None and not completion.done()
        assert dispatcher._durable_runtime_owner_registration_started is True

        allow_release.set()
        await dispatcher.shutdown_durable_delivery()

        owner = await backend.fetch_one(
            "SELECT stopped_at FROM durable_signal_runtime_owners "
            "WHERE agent_id = ? AND owner_id = ?",
            (agent.did, dispatcher._durable_delivery_owner),
        )
        assert owner is not None and owner[0] is not None
        assert dispatcher._durable_runtime_owner_registered is False
        assert dispatcher._durable_runtime_owner_registration_started is False
    finally:
        allow_release.set()
        dispatcher._durable_store.release_initial_reservations = original_release
        if shutdown_task is not None and not shutdown_task.done():
            await shutdown_task
        if dispatcher._durable_shutdown_completion is not None:
            await dispatcher.shutdown_durable_delivery()
        elif not dispatcher._durable_shutdown:
            await dispatcher.shutdown_durable_delivery()
        await _close(backend, agent)


@pytest.mark.asyncio
async def test_shutdown_retry_restarts_after_release_failure(tmp_path):
    """A failed cleanup attempt remains retryable until owner release succeeds."""
    backend, agent, dispatcher = await _dispatcher(
        tmp_path / "shutdown-failure-retry.db", "did:agent:one"
    )
    original_release = dispatcher._durable_store.release_initial_reservations
    attempts = 0

    async def fail_once(*args, **kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("injected owner-release failure")
        return await original_release(*args, **kwargs)

    dispatcher._durable_store.release_initial_reservations = fail_once
    try:
        with pytest.raises(RuntimeError, match="owner-release failure"):
            await dispatcher.shutdown_durable_delivery()
        failed_completion = dispatcher._durable_shutdown_completion
        assert failed_completion is not None and failed_completion.done()
        assert dispatcher._durable_runtime_owner_registration_started is True

        await dispatcher.shutdown_durable_delivery()
        assert attempts == 2
        assert dispatcher._durable_shutdown_completion is not failed_completion
        assert dispatcher._durable_runtime_owner_registered is False
        assert dispatcher._durable_runtime_owner_registration_started is False
    finally:
        dispatcher._durable_store.release_initial_reservations = original_release
        if dispatcher._durable_shutdown_completion is not None:
            try:
                await dispatcher.shutdown_durable_delivery()
            except RuntimeError:
                # The injected release failure remains observable above; this
                # cleanup branch is only for an assertion failure before retry.
                pass
        elif not dispatcher._durable_shutdown:
            await dispatcher.shutdown_durable_delivery()
        await _close(backend, agent)


@pytest.mark.asyncio
async def test_queued_heartbeat_shutdown_rejection_is_harvested(tmp_path):
    """A timer callback queued before closure cannot leak an exception."""
    backend, agent, dispatcher = await _dispatcher(
        tmp_path / "shutdown-queued-heartbeat.db", "did:agent:one"
    )
    heartbeat_done = asyncio.Event()
    heartbeat_task = None
    try:
        # This invokes the same callback a heartbeat timer uses, then closes
        # admission before the newly-created task receives its first turn.
        dispatcher._start_runtime_owner_heartbeat()
        heartbeat_task = dispatcher._runtime_owner_heartbeat_task
        assert heartbeat_task is not None
        heartbeat_task.add_done_callback(lambda _task: heartbeat_done.set())
        dispatcher._durable_shutdown = True

        await asyncio.wait_for(heartbeat_done.wait(), timeout=1.0)
        assert heartbeat_task.done()
        # ``Task._log_traceback`` is set only while an exception has not been
        # retrieved. The dispatcher callback must harvest the expected gate
        # rejection before the production task tracker drops its reference.
        assert not heartbeat_task._log_traceback
    finally:
        if not dispatcher._durable_shutdown:
            await dispatcher.shutdown_durable_delivery()
        elif dispatcher._durable_shutdown_completion is None:
            await dispatcher.shutdown_durable_delivery()
        await _close(backend, agent)


@pytest.mark.asyncio
async def test_shutdown_rejects_child_task_inherited_from_an_admitted_dispatch(tmp_path):
    """Child task context cannot turn a closed gate into nested admission."""
    backend = SQLiteBackend(str(tmp_path / "shutdown-inherited-child-task.db"))
    await backend.connect()
    log_store = SignalLogStore(backend)
    await log_store.initialize()
    agent = _Agent("did:agent:one")
    registry = SourceRegistry()
    child_created = asyncio.Event()
    allow_child = asyncio.Event()
    child_task = None

    async def register_after_parent_dispatch() -> None:
        await allow_child.wait()
        await dispatcher.register_durable_consumer(
            DurableConsumerRegistration(
                consumer_id="late-child",
                source="child.spawn",
                agent_id=agent.did,
            )
        )

    async def handler(_payload):
        nonlocal child_task
        child_task = asyncio.create_task(register_after_parent_dispatch())
        child_created.set()
        return {"handled": True}

    registry.register(
        SourceRegistration(
            name="child.spawn",
            schema=lambda payload: payload,
            default_mode=SignalMode.ACTION,
            allowed_modes=frozenset({SignalMode.ACTION}),
            handler=handler,
            trust=Trust.TRUSTED,
            log_redaction=RedactionPolicy(summarize=lambda _payload: "<redacted>"),
            retention_days=7,
        )
    )
    dispatcher = SignalDispatcher(
        agent=agent,
        registry=registry,
        lock_manager=OrderedLockManager(),
        store=log_store,
    )
    try:
        result = await dispatcher.dispatch_signal(
            Signal(
                source="child.spawn",
                kind="inbound",
                mode=SignalMode.ACTION,
                payload={},
                target_agent=agent.did,
            )
        )
        assert result.status is Status.OK
        await asyncio.wait_for(child_created.wait(), timeout=1.0)
        assert child_task is not None

        await dispatcher.shutdown_durable_delivery()
        allow_child.set()
        with pytest.raises(RuntimeError, match="shutting down"):
            await child_task

        # The inherited ContextVar must not allow a post-release consumer row.
        assert await backend.fetch_val(
            "SELECT COUNT(*) FROM durable_signal_consumers "
            "WHERE agent_id = ? AND consumer_id = ?",
            (agent.did, "late-child"),
        ) == 0
    finally:
        allow_child.set()
        if child_task is not None and not child_task.done():
            await child_task
        if not dispatcher._durable_shutdown:
            await dispatcher.shutdown_durable_delivery()
        await _close(backend, agent)


@pytest.mark.asyncio
async def test_shutdown_allows_child_with_completed_parent_admission(tmp_path):
    """A child may tear down after inheriting its completed parent's context.

    ``asyncio.create_task`` copies the dispatch admission ContextVar.  The
    copied tuple belongs to the completed parent task, not the child, so it
    must not prevent the child from owning shutdown and releasing the runtime
    owner.
    """
    backend = SQLiteBackend(str(tmp_path / "shutdown-inherited-completed-parent.db"))
    await backend.connect()
    log_store = SignalLogStore(backend)
    await log_store.initialize()
    agent = _Agent("did:agent:one")
    child_created = asyncio.Event()
    allow_child_shutdown = asyncio.Event()
    child_task = None

    async def shutdown_after_parent_dispatch() -> None:
        await allow_child_shutdown.wait()
        await dispatcher.shutdown_durable_delivery()

    async def handler(_payload):
        nonlocal child_task
        child_task = asyncio.create_task(shutdown_after_parent_dispatch())
        child_created.set()
        return {"handled": True}

    registry = SourceRegistry()
    registry.register(
        SourceRegistration(
            name="child.shutdown",
            schema=lambda payload: payload,
            default_mode=SignalMode.ACTION,
            allowed_modes=frozenset({SignalMode.ACTION}),
            handler=handler,
            trust=Trust.TRUSTED,
            log_redaction=RedactionPolicy(summarize=lambda _payload: "<redacted>"),
            retention_days=7,
        )
    )
    dispatcher = SignalDispatcher(
        agent=agent,
        registry=registry,
        lock_manager=OrderedLockManager(),
        store=log_store,
    )
    await dispatcher.initialize_durable_delivery()

    try:
        result = await dispatcher.dispatch_signal(
            Signal(
                source="child.shutdown",
                kind="inbound",
                mode=SignalMode.ACTION,
                payload={},
                target_agent=agent.did,
            )
        )
        assert result.status is Status.OK
        await asyncio.wait_for(child_created.wait(), timeout=1.0)
        assert child_task is not None

        # The parent dispatch has released its admission before the child
        # starts shutdown, but the child still inherited the stale ContextVar.
        allow_child_shutdown.set()
        await child_task

        owner = await backend.fetch_one(
            "SELECT stopped_at FROM durable_signal_runtime_owners "
            "WHERE agent_id = ? AND owner_id = ?",
            (agent.did, dispatcher._durable_delivery_owner),
        )
        assert owner is not None and owner[0] is not None
        assert dispatcher._durable_runtime_owner_registered is False
    finally:
        allow_child_shutdown.set()
        if child_task is not None and not child_task.done():
            await child_task
        if not dispatcher._durable_shutdown:
            await dispatcher.shutdown_durable_delivery()
        await _close(backend, agent)


@pytest.mark.asyncio
async def test_shutdown_drains_committed_reservation_repair_before_owner_release(tmp_path):
    """A committed volatile reservation cannot be dropped during shutdown."""
    from kestrel_sovereign.privacy import get_privacy_preset

    backend, agent, dispatcher = await _dispatcher(
        tmp_path / "shutdown-committed-repair.db", "did:agent:one"
    )
    agent.privacy_config = get_privacy_preset("ephemeral")
    consumer = DurableConsumerRegistration(
        consumer_id="workflow-wait", source="provider.message", agent_id=agent.did
    )
    original_activate = dispatcher._durable_store.activate_initial_delivery
    original_abandon = dispatcher._durable_store.abandon_initial_reservation
    repair_entered = asyncio.Event()
    allow_repair = asyncio.Event()
    dispatch_task = None
    shutdown_task = None

    async def fail_after_committed_activation(**kwargs):
        await original_activate(**kwargs)
        raise RuntimeError("inject committed activation failure")

    async def block_committed_repair(**kwargs):
        repair_entered.set()
        await allow_repair.wait()
        return await original_abandon(**kwargs)

    dispatcher._durable_store.activate_initial_delivery = fail_after_committed_activation
    dispatcher._durable_store.abandon_initial_reservation = block_committed_repair
    try:
        await dispatcher.register_durable_consumer(consumer)
        dispatch_task = asyncio.create_task(
            dispatcher.dispatch_signal(
                _signal(agent_id=agent.did, message="must-repair-before-close"),
                source_event_id="must-repair-before-close",
            )
        )
        await asyncio.wait_for(repair_entered.wait(), timeout=1.0)
        assert dispatcher._post_commit_reservation_repairs

        shutdown_task = asyncio.create_task(dispatcher.shutdown_durable_delivery())
        await asyncio.sleep(0)
        assert not shutdown_task.done()

        allow_repair.set()
        result = await dispatch_task
        assert result.status is Status.FAILED
        await shutdown_task

        row = await backend.fetch_one(
            "SELECT status, lease_owner, lease_token, lease_expires_at "
            "FROM durable_signal_deliveries WHERE agent_id = ? "
            "AND consumer_id = ?",
            (agent.did, consumer.consumer_id),
        )
        assert row is not None
        assert row[0] == RETRY
        assert row[1:] == (None, None, None)
        assert await backend.fetch_val(
            "SELECT COUNT(*) FROM durable_signal_deliveries "
            "WHERE status = 'initial_reserved'"
        ) == 0
        assert dispatcher._transient_durable_handoffs == {}
        assert dispatcher._transient_durable_handoff_timers == {}
        assert dispatcher._post_commit_reservation_repairs == set()

        owner = await backend.fetch_one(
            "SELECT stopped_at FROM durable_signal_runtime_owners "
            "WHERE agent_id = ? AND owner_id = ?",
            (agent.did, dispatcher._durable_delivery_owner),
        )
        assert owner is not None and owner[0] is not None
    finally:
        allow_repair.set()
        dispatcher._durable_store.activate_initial_delivery = original_activate
        dispatcher._durable_store.abandon_initial_reservation = original_abandon
        if dispatch_task is not None and not dispatch_task.done():
            await dispatch_task
        if shutdown_task is not None and not shutdown_task.done():
            await shutdown_task
        if not dispatcher._durable_shutdown:
            await dispatcher.shutdown_durable_delivery()
        await _close(backend, agent)


@pytest.mark.asyncio
async def test_repeated_cancellation_releases_admission_without_waiting_on_lifecycle_lock(
    tmp_path,
):
    """An admitted dispatch cannot leak its count when cancellation repeats.

    The old release path awaited ``_durable_lifecycle_lock`` from its
    ``finally``.  The second cancellation below landed at exactly that await,
    leaving shutdown permanently blocked on a leaked admission.
    """
    backend, agent, dispatcher = await _dispatcher(
        tmp_path / "admission-release-cancellation.db", "did:agent:one"
    )
    original_append = dispatcher._store.append
    append_entered = asyncio.Event()
    allow_append = asyncio.Event()
    dispatch_task = None

    async def block_append(*args, **kwargs):
        append_entered.set()
        await allow_append.wait()
        return await original_append(*args, **kwargs)

    dispatcher._store.append = block_append
    try:
        dispatch_task = asyncio.create_task(
            dispatcher.dispatch_signal(
                _signal(agent_id=agent.did, message="cancelled-at-release"),
                source_event_id="cancelled-at-release",
            )
        )
        await asyncio.wait_for(append_entered.wait(), timeout=1.0)

        await dispatcher._durable_lifecycle_lock.acquire()
        dispatch_task.cancel()
        await asyncio.sleep(0)
        # This is harmless with synchronous release, but it interrupted the
        # old ``async with lifecycle_lock`` finalizer before it decremented.
        dispatch_task.cancel()
        allow_append.set()
        dispatcher._durable_lifecycle_lock.release()

        with pytest.raises(asyncio.CancelledError):
            await dispatch_task
        await asyncio.wait_for(dispatcher._drain_outcome_log_tasks(), timeout=1.0)
        assert dispatcher._durable_active_admissions == 0
        assert dispatcher._durable_admissions_drained.is_set()

        await asyncio.wait_for(dispatcher.shutdown_durable_delivery(), timeout=1.0)
        assert dispatcher._durable_runtime_owner_registered is False
    finally:
        allow_append.set()
        dispatcher._store.append = original_append
        if dispatcher._durable_lifecycle_lock.locked():
            dispatcher._durable_lifecycle_lock.release()
        if dispatch_task is not None and not dispatch_task.done():
            with suppress(asyncio.CancelledError):
                await dispatch_task
        if not dispatcher._durable_shutdown:
            await dispatcher.shutdown_durable_delivery()
        await _close(backend, agent)


@pytest.mark.asyncio
async def test_queued_outcome_log_cancellation_reconciles_its_admission(tmp_path):
    """A writer cancelled before its first turn cannot hold shutdown open."""
    backend, agent, dispatcher = await _dispatcher(
        tmp_path / "queued-outcome-log-cancellation.db", "did:agent:one"
    )
    queued_log = None
    try:
        async with dispatcher._admit_durable_operation():
            dispatcher._success(
                _signal(agent_id=agent.did, message="never-written"),
                time.monotonic(),
                _registration(agent),
            )
            queued_log = next(iter(dispatcher._outcome_log_tasks))
            queued_log.cancel()

        await asyncio.wait_for(dispatcher._durable_admissions_drained.wait(), timeout=1.0)
        assert queued_log.done()
        assert dispatcher._durable_active_admissions == 0
        assert await backend.fetch_val("SELECT COUNT(*) FROM signal_log") == 0

        await asyncio.wait_for(dispatcher.shutdown_durable_delivery(), timeout=1.0)
    finally:
        if not dispatcher._durable_shutdown:
            await dispatcher.shutdown_durable_delivery()
        await _close(backend, agent)


@pytest.mark.asyncio
async def test_in_worker_outcome_log_cancellation_drains_before_sqlite_close(tmp_path):
    """A cancelled writer finishes its accepted SQLite append before close."""
    from tests.utils.aiosqlite_workers import aiosqlite_worker

    backend, agent, dispatcher = await _dispatcher(
        tmp_path / "in-worker-outcome-log-cancellation.db", "did:agent:one"
    )
    connection = backend._connection
    assert connection is not None
    worker = aiosqlite_worker(connection)
    original_append = dispatcher._store.append
    append_entered = asyncio.Event()
    allow_append = asyncio.Event()
    outcome_log = None
    shutdown_task = None

    async def block_append(*args, **kwargs):
        append_entered.set()
        await allow_append.wait()
        return await original_append(*args, **kwargs)

    dispatcher._store.append = block_append
    try:
        async with dispatcher._admit_durable_operation():
            dispatcher._success(
                _signal(agent_id=agent.did, message="must-finish-writing"),
                time.monotonic(),
                _registration(agent),
            )
            outcome_log = next(iter(dispatcher._outcome_log_tasks))
            await asyncio.wait_for(append_entered.wait(), timeout=1.0)
            outcome_log.cancel()
            await asyncio.sleep(0)
            assert not outcome_log.done()

        shutdown_task = asyncio.create_task(dispatcher.shutdown_durable_delivery())
        await asyncio.sleep(0)
        assert not shutdown_task.done()

        allow_append.set()
        with pytest.raises(asyncio.CancelledError):
            await outcome_log
        await asyncio.wait_for(shutdown_task, timeout=1.0)

        assert await backend.fetch_val("SELECT COUNT(*) FROM signal_log") == 1
        await backend.close()
        assert not worker.is_alive()
    finally:
        allow_append.set()
        dispatcher._store.append = original_append
        if outcome_log is not None and not outcome_log.done():
            with suppress(asyncio.CancelledError):
                await outcome_log
        if shutdown_task is not None and not shutdown_task.done():
            await shutdown_task
        if not dispatcher._durable_shutdown:
            await dispatcher.shutdown_durable_delivery()
        if backend._connection is not None:
            await backend.close()
