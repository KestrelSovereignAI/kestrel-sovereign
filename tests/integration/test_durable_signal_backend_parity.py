"""SQLite/PostgreSQL parity for the durable signal delivery ledger."""

from __future__ import annotations

import asyncio

import pytest

from kestrel_sdk.signals import Signal, SignalMode
from kestrel_sovereign.signals import (
    DurableConsumerRegistration,
    DurableSignalStore,
)


def _signal(agent_id: str) -> Signal:
    return Signal(
        source="provider.message",
        kind="inbound",
        mode=SignalMode.ACTION,
        payload={"workflow": "wf-1", "message": "normalized"},
        target_agent=agent_id,
    )


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_durable_delivery_claim_is_scoped_and_single_owner_on_both_backends(db_backend):
    store = DurableSignalStore(db_backend)
    await store.initialize()
    agent_id = "did:test:durable-parity"
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
