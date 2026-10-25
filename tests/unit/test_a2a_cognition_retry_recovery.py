"""An original A2A retry fence must not prevent replacement-host boot.

Real task stores, privacy projection, dispatchers and native connections are
used. The recipient protocol fixture supplies the cognition body.
"""

import asyncio
import copy
from dataclasses import replace
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
import pytest_asyncio
from kestrel_sdk.native_a2a.types import TaskState

from kestrel_sovereign.signals import ACKNOWLEDGED, RETRY, DurableConsumerRegistration
from kestrel_sovereign.signals.sources import a2a, a2a_task_submitted
from tests.unit.test_a2a_native_input_fencing import close_input_dispatcher, input_dispatcher
from tests.unit.test_a2a_native_task_manager import manager as manager, request
from tests.utils.native_a2a import manager_for


@pytest.fixture(params=[False, True], ids=["normal", "ephemeral"])
def private(request):
    return request.param


@pytest.fixture(params=[False, True], ids=["submitted", "complete"])
def completed(request):
    return request.param


@pytest_asyncio.fixture
async def retry_pair(
    manager, native_a2a_backend_factory, private, completed,
):
    task = await manager.create_task(request(), creator_agent_id="did:creator")
    if completed:
        await manager.update_status(task.id, TaskState.TASK_STATE_COMPLETED, recipient_agent_id=manager.did)
    candidate = await manager.get_task_snapshot(task.id, recipient_agent_id=manager.did)
    source = a2a if completed else a2a_task_submitted
    signal = (a2a.build_signal_for_completed_task(candidate, manager.did) if completed
              else a2a_task_submitted.build_signal_for_submitted_task(candidate, manager.did))
    consumer_id = source.DURABLE_COGNITION_CONSUMER_ID
    identity = task.id if completed else a2a_task_submitted.submitted_input_identity(task.id, 0)
    agent_a, dispatcher_a, _ = await input_dispatcher(manager, private=private)
    backend_b = native_a2a_backend_factory()
    await backend_b.connect()
    manager_b = manager_for(backend_b, manager.task_store.scope)
    await manager_b.initialize()
    agent_b = dispatcher_b = None
    try:
        await dispatcher_a.register_durable_consumer(DurableConsumerRegistration(
            consumer_id=consumer_id, source=source.SOURCE_NAME,
            agent_id=manager.did, max_attempts=0,
        ))
        # Actual ingress creates the delivery, including the private marker
        # and original integrity binding when content is elided.
        await dispatcher_a.dispatch_signal(copy.deepcopy(signal), source_event_id=identity)
        original = await dispatcher_a.get_durable_delivery_for_source_event(
            consumer_id=consumer_id, source=source.SOURCE_NAME, source_event_id=identity,
        )
        assert original is not None
        original = await dispatcher_a.claim_durable_delivery_for_event(
            consumer_id=consumer_id, event_id=original.event_id,
            executor_id=dispatcher_a._durable_delivery_owner,
        )
        assert original is not None and original.lease_token
        retried = await dispatcher_a.nack_durable_delivery(
            consumer_id=consumer_id, delivery_id=original.delivery_id,
            lease_token=original.lease_token, error="original retry",
        )
        assert retried is not None and retried.status == RETRY
        agent_b, dispatcher_b, _ = await input_dispatcher(manager_b, private=private)
        await dispatcher_b.register_durable_consumer(DurableConsumerRegistration(
            consumer_id=consumer_id, source=source.SOURCE_NAME,
            agent_id=manager.did, max_attempts=0,
        ))
        agent_b.process_input = AsyncMock(return_value="recovered")
        yield SimpleNamespace(
            agent_a=agent_a, dispatcher_a=dispatcher_a, agent_b=agent_b, dispatcher_b=dispatcher_b,
            signal=signal, original=original, consumer_id=consumer_id, identity=identity, source=source,
        )
    finally:
        await close_input_dispatcher(agent_a, dispatcher_a)
        if dispatcher_b is not None:
            await close_input_dispatcher(agent_b, dispatcher_b)
        await manager_b.close()


@pytest.mark.parametrize("recovery", ["stopped", "stale"])
async def test_boot_reconcile_preserves_original_a2a_retry_until_owner_retires(retry_pair, recovery):
    pair = retry_pair
    # Actual EventManager reconciliation sees the old owner's retained RETRY,
    # not a mocked admission receipt. This covers a live sibling/rolling stop
    # and the still-fresh owner row left behind by a crashed process.
    await pair.agent_b.reconcile_a2a_cognition_wakes()
    retained = await pair.dispatcher_b.get_durable_delivery_for_source_event(
        consumer_id=pair.consumer_id, source=pair.source.SOURCE_NAME, source_event_id=pair.identity,
    )
    assert retained.status == RETRY
    assert (retained.lease_owner, retained.lease_token) == (pair.original.lease_owner, pair.original.lease_token)
    pair.agent_b.process_input.assert_not_awaited()
    await pair.dispatcher_b.start_durable_cognition_consumer(pair.consumer_id)
    if recovery == "stopped":
        await close_input_dispatcher(pair.agent_a, pair.dispatcher_a)
    else:
        await pair.dispatcher_a._stop_runtime_owner_heartbeat()
        pair.dispatcher_b._runtime_owner_stale_after = timedelta(0)
    await pair.dispatcher_b._heartbeat_runtime_owner()
    async with asyncio.timeout(10):
        while True:
            retained = await pair.dispatcher_b.get_durable_delivery_for_source_event(
                consumer_id=pair.consumer_id, source=pair.source.SOURCE_NAME, source_event_id=pair.identity,
            )
            if retained.status == ACKNOWLEDGED:
                break
            await asyncio.sleep(0.01)
    pair.agent_b.process_input.assert_awaited_once()
    assert len(await pair.dispatcher_b.list_durable_deliveries(consumer_id=pair.consumer_id)) == 1


async def refused_retry(pair, signal):
    from kestrel_sovereign.signals import DurableAdmissionDisposition

    signal = replace(signal, id="sig_" + uuid4().hex)
    handle = await pair.dispatcher_b.enqueue_durable_cognition(
        signal, consumer_id=pair.consumer_id, source_event_id=pair.identity,
    )
    receipt = await handle.wait_for_durable_admission()
    result = await handle.wait()
    assert receipt.disposition is DurableAdmissionDisposition.NOT_ADMITTED, result.error
    assert not receipt.acknowledged
    pair.agent_b.process_input.assert_not_awaited()
    retained = await pair.dispatcher_b.get_durable_delivery_for_source_event(
        consumer_id=pair.consumer_id, source=pair.source.SOURCE_NAME, source_event_id=pair.identity,
    )
    assert retained.status == RETRY
    assert (retained.lease_owner, retained.lease_token) == (pair.original.lease_owner, pair.original.lease_token)


@pytest.mark.parametrize("missing", ["hook", "source"])
async def test_owned_retry_without_authoritative_rehydration_is_not_checkpointable(retry_pair, monkeypatch, missing):
    if missing == "hook":
        monkeypatch.setattr(retry_pair.agent_b, "rehydrate_durable_cognition_signal", None)
    else:
        monkeypatch.setattr(retry_pair.agent_b, "durable_rehydratable_sources", frozenset())
    await refused_retry(retry_pair, retry_pair.signal)


@pytest.mark.parametrize("private", [True], ids=["ephemeral"])
async def test_owned_retry_does_not_bypass_original_private_integrity(retry_pair):
    signal = retry_pair.signal
    field = "result_summary" if retry_pair.source is a2a else "request_content"
    changed = replace(signal, payload={**signal.payload, field: "changed original content"})
    await refused_retry(retry_pair, changed)


async def test_original_channel_retry_does_not_advance_provider_cursor(tmp_path, private):
    from kestrel_sovereign.privacy import get_privacy_preset
    from kestrel_sovereign.signals.sources import channels
    from tests.unit.test_durable_signal_delivery import _channel_dispatcher, _channel_signal, _close

    path = tmp_path / "channel-retry.db"
    backend_a, agent_a, dispatcher_a = await _channel_dispatcher(path, "did:channel:retry")
    backend_b, agent_b, dispatcher_b = await _channel_dispatcher(path, agent_a.did)
    consumer_id = channels.DURABLE_COGNITION_CONSUMER_ID
    identity = "telegram:retained-retry"
    signal = _channel_signal(agent_a.did, "retained-retry")
    try:
        for agent, dispatcher in ((agent_a, dispatcher_a), (agent_b, dispatcher_b)):
            if private:
                agent.privacy_config = get_privacy_preset("ephemeral")
            await dispatcher.register_durable_consumer(DurableConsumerRegistration(
                consumer_id=consumer_id, source=channels.SOURCE_NAME,
                agent_id=agent.did, max_attempts=0,
            ))
        await dispatcher_a.dispatch_signal(copy.deepcopy(signal), source_event_id=identity)
        original = await dispatcher_a.get_durable_delivery_for_source_event(
            consumer_id=consumer_id, source=channels.SOURCE_NAME, source_event_id=identity,
        )
        original = await dispatcher_a.claim_durable_delivery_for_event(
            consumer_id=consumer_id, event_id=original.event_id,
            executor_id=dispatcher_a._durable_delivery_owner,
        )
        assert original is not None and original.lease_token
        assert (await dispatcher_a.nack_durable_delivery(
            consumer_id=consumer_id, delivery_id=original.delivery_id,
            lease_token=original.lease_token, error="original channel retry",
        )).status == RETRY
        # A declared hook alone cannot widen the native A2A outbox contract.
        # This hook is a stand-in and must never be invoked by this refusal.
        agent_b.durable_rehydratable_sources = frozenset({channels.SOURCE_NAME})
        agent_b.rehydrate_durable_cognition_signal = AsyncMock(return_value=signal)
        agent_b.process_input = AsyncMock()
        await refused_retry(SimpleNamespace(
            dispatcher_b=dispatcher_b, agent_b=agent_b, consumer_id=consumer_id,
            identity=identity, source=channels, original=original,
        ), signal)
        agent_b.rehydrate_durable_cognition_signal.assert_not_awaited()
    finally:
        await close_input_dispatcher(agent_a, dispatcher_a)
        await close_input_dispatcher(agent_b, dispatcher_b)
        await _close(backend_a, agent_a)
        await _close(backend_b, agent_b)
