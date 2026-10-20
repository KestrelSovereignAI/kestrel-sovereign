"""Real PostgreSQL ingress retirement and channel history (#3768)."""

import asyncio

import pytest

from kestrel_sovereign.execution_custody import (
    ExecutionAuthorityError, ExecutionCustody, bind_execution_custody,
    current_execution_custody, require_execution_work,
)
from kestrel_sovereign.kestrel_agent import KestrelAgent
from kestrel_sovereign.privacy import get_privacy_preset
from kestrel_sovereign.signals import (
    ACKNOWLEDGED, DurableConsumerRegistration, OrderedLockManager,
    SignalDispatcher, SignalLogStore, SourceRegistry,
)
from kestrel_sovereign.signals.sources.channels import (
    DURABLE_COGNITION_CONSUMER_ID, DURABLE_COGNITION_MARKER,
    DURABLE_COGNITION_MARKER_VALUE, build_channel_message_registration,
)
from kestrel_sovereign.storage.async_database import AsyncDatabase
from kestrel_sovereign.storage.async_storage import AsyncStorage
from tests.integration.test_execution_authority_postgres import (
    GenerationFence, native_pg as native_pg,
)
from tests.unit.test_channel_timestamp_contract import assert_channel_timestamp_roundtrip
from tests.unit.test_durable_signal_delivery import _channel_signal

pytestmark = [pytest.mark.asyncio, pytest.mark.integration]


@pytest.mark.parametrize("offset", [0, -5, 9])
async def test_channel_history_real_postgres_utc_roundtrip(native_pg, offset):
    backend, _, _ = native_pg
    await assert_channel_timestamp_roundtrip(backend, offset)


@pytest.mark.parametrize("ready_at_admission", [False, True])
async def test_real_pg_cognition_and_audit_survive_http_admission_retirement(
    native_pg, tmp_path, ready_at_admission,
):
    backend, control, schema = native_pg
    proceed, entered = asyncio.Event(), asyncio.Event()

    class ControlledAgent(KestrelAgent):
        async def process_input(self, prompt):
            require_execution_work(self)
            assert current_execution_custody() == (self._execution_custody,)
            entered.set()
            await proceed.wait()
            await self.storage.db.execute("INSERT INTO effects (id, value) VALUES (2, 'resident')")
            return "controlled, provider-free response"

    agent = ControlledAgent(did="did:test:pg-ingress", storage_path=str(tmp_path / "unused.db"))
    agent.privacy_config = get_privacy_preset("normal")
    agent._execution_custody = ExecutionCustody(GenerationFence())
    backend._execution_custody = agent._execution_custody
    storage = AsyncStorage(backend=backend)
    storage.db = AsyncDatabase(backend)
    agent.storage = agent._raw_storage = storage
    if ready_at_admission:
        agent._runtime_publication_ready.set()
    logs = SignalLogStore(backend)
    await logs.initialize()
    registry = SourceRegistry()
    registry.register(build_channel_message_registration())
    dispatcher = SignalDispatcher(
        agent=agent, registry=registry, lock_manager=OrderedLockManager(), store=logs,
    )
    agent.dispatcher = dispatcher
    consumer = DURABLE_COGNITION_CONSUMER_ID
    try:
        await dispatcher.initialize_durable_delivery()
        await dispatcher.register_durable_consumer(DurableConsumerRegistration(
            consumer_id=consumer, source="channel.message", agent_id=agent.did,
            correlation_selector=f"payload.{DURABLE_COGNITION_MARKER}={DURABLE_COGNITION_MARKER_VALUE}",
            max_attempts=0,
        ))
        with bind_execution_custody(GenerationFence()) as ingress:
            handle = await dispatcher.enqueue_durable_cognition(
                _channel_signal(agent.did, "native"),
                source_event_id="telegram:update:native", consumer_id=consumer,
            )
            receipt = await asyncio.wait_for(handle.wait_for_durable_admission(), 5)
            assert receipt.acknowledged and receipt.resident_owned
            if not ready_at_admission:
                assert not entered.is_set(), "effects ran before real READY"
        with pytest.raises(ExecutionAuthorityError, match="admission retired"):
            ingress.require_work()
        agent._runtime_publication_ready.set()
        await asyncio.wait_for(entered.wait(), 5)
        proceed.set()
        await asyncio.wait_for(handle.wait(), 10)
        delivery = await dispatcher.get_durable_delivery_for_event(
            consumer_id=consumer, event_id=receipt.signal_id,
        )
        assert delivery.status == ACKNOWLEDGED
        assert await control.fetchval(f'SELECT COUNT(*) FROM "{schema}".effects WHERE id=2') == 1
        assert dispatcher._outcome_log_control_error is None
        assert await control.fetchval(f'SELECT COUNT(*) FROM "{schema}".signal_log') >= 1
    finally:
        proceed.set()
        agent._runtime_publication_ready.set()
        await dispatcher.shutdown_durable_delivery()
        tasks = tuple(agent._background_tasks)
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
