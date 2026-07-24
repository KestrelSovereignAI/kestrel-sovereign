"""Durable signal consumer contract: replay, scoped leasing, and retention."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

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
from kestrel_sovereign.storage.db import SQLiteBackend


class _Agent:
    def __init__(self, did: str):
        self._did = did
        self.tasks: list[asyncio.Task] = []
        self.action_calls = 0

    @property
    def did(self) -> str:
        return self._did

    async def process_input(self, prompt: str):  # pragma: no cover - ACTION only
        return prompt

    def _track_background_task(self, coro, *, name: str):
        task = asyncio.create_task(coro, name=name)
        self.tasks.append(task)
        return task


def _registration(agent: _Agent, source: str = "provider.message") -> SourceRegistration:
    async def handler(payload):
        agent.action_calls += 1
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
    assert await store.purge_expired(now=now + timedelta(days=2)) == 1
    assert await store.list_deliveries(agent_id=agent_id) == []
    await backend.close()
