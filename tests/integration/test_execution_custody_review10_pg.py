"""Native control terminalization uses the existing backend and exact lease CAS."""

import asyncio
from datetime import datetime, timedelta, timezone

import pytest

from kestrel_sovereign.execution_custody import (
    ExecutionAuthorityError, ExecutionCommitOutcomeError, ExecutionCustody,
)
from kestrel_sovereign.signals.sources.channels import DURABLE_COGNITION_CONSUMER_ID
from tests.integration.test_execution_authority_postgres import native_pg as _native_pg, GenerationFence


pytestmark = [pytest.mark.asyncio, pytest.mark.integration]
native_pg = _native_pg


async def test_native_cleanup_heartbeat_cannot_insert_revive_or_touch_successor(native_pg):
    from kestrel_sovereign.signals.durable import DurableSignalStore

    backend, controller, schema = native_pg
    await DurableSignalStore(backend).initialize()
    await backend.execute(
        "INSERT INTO durable_signal_runtime_owners "
        "(agent_id,owner_id,heartbeat_at,stopped_at) VALUES "
        "('original','dispatcher:original',NOW()-INTERVAL '1 hour',NULL), "
        "('original','dispatcher:stopped',NOW()-INTERVAL '1 hour',NOW()), "
        "('other','dispatcher:successor',NOW()-INTERVAL '1 hour',NULL)"
    )
    # An existing managed owner without a matching leased cognition is not
    # eligible for this cleanup metadata path (nor can stopped owners revive).
    scope = ExecutionCustody(GenerationFence())
    backend._execution_custody = scope
    scope.revoke("original generation uncertain")
    assert not await backend.retain_cognition_cleanup_owner(agent_id="original", owner_id="dispatcher:original")
    assert not await backend.retain_cognition_cleanup_owner(agent_id="original", owner_id="dispatcher:missing")
    # Use controller-only fixture setup for a real, FK-valid managed delivery.
    await controller.execute(f'SET search_path TO "{schema}", public')
    await controller.execute(
        "INSERT INTO durable_signal_consumers "
        "(agent_id,consumer_id,source,max_attempts,lease_seconds,active) "
        "VALUES ('original',$1,'channel.message',0,10,TRUE)", DURABLE_COGNITION_CONSUMER_ID,
    )
    await controller.execute(
        "INSERT INTO durable_signal_events "
        "(event_id,agent_id,target_agent,source,kind,mode,visibility,payload,"
        "urgency,causation_chain,arrived_at,retention_until,source_sequence) "
        "VALUES ('event','original','original','channel.message','message','cognition',"
        "'internal','{}','normal','[]',NOW(),NOW()+INTERVAL '1 day',1)"
    )
    await controller.execute(
        "INSERT INTO durable_signal_deliveries "
        "(delivery_id,event_id,agent_id,consumer_id,status,max_attempts,lease_owner,lease_token) "
        "VALUES ('delivery','event','original',$1,'leased',0,'dispatcher:original','original-token')",
        DURABLE_COGNITION_CONSUMER_ID,
    )
    assert await backend.retain_cognition_cleanup_owner(agent_id="original", owner_id="dispatcher:original")
    assert not await backend.retain_cognition_cleanup_owner(agent_id="original", owner_id="dispatcher:stopped")
    assert await controller.fetchval("SELECT heartbeat_at > NOW()-INTERVAL '1 minute' FROM durable_signal_runtime_owners WHERE agent_id='original' AND owner_id='dispatcher:original'")
    await controller.execute("UPDATE durable_signal_deliveries SET lease_owner='dispatcher:stopped' WHERE delivery_id='delivery'")
    assert not await backend.retain_cognition_cleanup_owner(agent_id="original", owner_id="dispatcher:stopped")
    assert await controller.fetchval("SELECT stopped_at IS NOT NULL FROM durable_signal_runtime_owners WHERE owner_id='dispatcher:stopped'")
    assert await controller.fetchval("SELECT heartbeat_at < NOW()-INTERVAL '30 minutes' FROM durable_signal_runtime_owners WHERE owner_id='dispatcher:successor'")
    with pytest.raises(ExecutionAuthorityError):
        await backend.execute("INSERT INTO effects VALUES (1, 'denied')")


@pytest.mark.parametrize("control", ["denied", "unknown", "committed"])
async def test_native_cognition_terminal_cas_never_revives_work_or_successor(native_pg, control):
    backend, controller, schema = native_pg
    await backend.execute(
        "CREATE TABLE durable_signal_deliveries (agent_id TEXT, consumer_id TEXT, "
        "delivery_id TEXT PRIMARY KEY, status TEXT, next_attempt_at TIMESTAMPTZ, "
        "lease_owner TEXT, lease_token TEXT, lease_expires_at TIMESTAMPTZ, "
        "last_error TEXT, terminal_at TIMESTAMPTZ, updated_at TIMESTAMPTZ)"
    )
    await backend.execute(
        "INSERT INTO durable_signal_deliveries "
        "(agent_id,consumer_id,delivery_id,status,lease_owner,lease_token) "
        "VALUES ('original',?,'selected','leased','dispatcher:original','token'), "
        "('other',?,'successor','leased','dispatcher:successor','new-token')",
        (DURABLE_COGNITION_CONSUMER_ID, DURABLE_COGNITION_CONSUMER_ID),
    )
    scope = ExecutionCustody(GenerationFence())
    backend._execution_custody = scope
    if control == "denied":
        scope.revoke("original generation retired")
    else:
        scope.preserve_commit_uncertainty(control)
    with pytest.raises(ExecutionAuthorityError):
        await backend.execute("INSERT INTO effects VALUES (1, 'denied')")
    arguments = dict(
        agent_id="original", consumer_id=DURABLE_COGNITION_CONSUMER_ID,
        delivery_id="selected", owner_id="dispatcher:original", lease_token="token",
        error="execution_control_unresolved",
    )
    assert await backend.fail_cognition_delivery(**arguments)
    assert not await backend.fail_cognition_delivery(**arguments)
    assert not await backend.fail_cognition_delivery(**(arguments | {"delivery_id": "successor"}))
    row = await controller.fetchrow(f'SELECT * FROM "{schema}".durable_signal_deliveries WHERE delivery_id=\'selected\'')
    assert row["status"] == "failed"
    assert row["lease_token"] is None and row["next_attempt_at"] is None
    assert row["terminal_at"] is not None
    assert await controller.fetchval(f'SELECT status FROM "{schema}".durable_signal_deliveries WHERE delivery_id=\'successor\'') == "leased"
    with pytest.raises(ExecutionAuthorityError):
        await backend.fetch_val("SELECT 1")
    assert await controller.fetchval(f'SELECT count(*) FROM "{schema}".effects') == 0


@pytest.mark.parametrize("retained", [False, True])
@pytest.mark.parametrize("carrier", ["direct", "cancel", "stop", "self-fence"])
@pytest.mark.parametrize("outcome", ["unknown", "committed"])
async def test_actual_native_cognition_keeps_uncertain_committed_effect_nonretryable(native_pg, retained, carrier, outcome):
    from kestrel_sovereign.signals import (
        DurableConsumerRegistration, OrderedLockManager, SignalDispatcher,
        SignalLogStore, SourceRegistry,
    )
    from kestrel_sovereign.signals.sources.channels import build_channel_message_registration
    from tests.unit.test_durable_signal_delivery import _Agent, _channel_signal

    backend, _, _ = native_pg
    store = SignalLogStore(backend)
    await store.initialize()
    agent = _Agent("did:test:native-uncertain-cognition")
    registry = SourceRegistry()
    registry.register(build_channel_message_registration())
    dispatcher = SignalDispatcher(
        agent=agent, registry=registry, lock_manager=OrderedLockManager(), store=store,
    )
    consumer = DurableConsumerRegistration(
        consumer_id=DURABLE_COGNITION_CONSUMER_ID, source="channel.message",
        agent_id=agent.did, max_attempts=0, lease_seconds=10,
    )
    entered, release = asyncio.Event(), asyncio.Event()
    calls = 0
    handle = None
    from kestrel_sovereign.agent.invocation import InvocationCancelledError, InvocationSelfFencedError
    error = ExecutionCommitOutcomeError(outcome)
    if carrier != "direct":
        error_type = {"cancel": asyncio.CancelledError, "stop": InvocationCancelledError,
                      "self-fence": InvocationSelfFencedError}[carrier]
        wrapper = error_type("Stop raced irreversible effect")
        wrapper.__cause__ = error
        error = wrapper

    async def process_input(prompt):
        nonlocal calls
        calls += 1
        await backend.execute("INSERT INTO effects VALUES (1, 'committed')")
        entered.set()
        if retained:
            try:
                await release.wait()
            except asyncio.CancelledError:
                await release.wait()
        raise error

    agent.process_input = process_input
    try:
        await dispatcher.register_durable_consumer(consumer)
        handle = await dispatcher.enqueue_durable_cognition(
            _channel_signal(agent.did, "uncertain"),
            source_event_id="telegram:update:uncertain", consumer_id=consumer.consumer_id,
        )
        await asyncio.wait_for(entered.wait(), 5)
        if retained:
            handle.task.cancel()
            await asyncio.gather(handle.task, return_exceptions=True)
            release.set()
            for task in tuple(dispatcher._retained_durable_cognition_tasks):
                await asyncio.gather(task, return_exceptions=True)
            await asyncio.sleep(0)
            await dispatcher._drain_retained_durable_cognition_cleanup_tasks()
        else:
            with pytest.raises(type(error)) as caught:
                await handle.wait()
            assert caught.value is error
        rows = await dispatcher.list_durable_deliveries()
        assert len(rows) == 1
        assert rows[0].status == "failed"
        assert "execution_control_unresolved" in rows[0].last_error
        assert rows[0].next_attempt_at is None
        assert await dispatcher._durable_store.claim_delivery_for_event(
            agent_id=agent.did, consumer_id=consumer.consumer_id,
            event_id=rows[0].event.event_id, executor_id="dispatcher:replacement",
            now=datetime.now(timezone.utc) + timedelta(hours=1),
        ) is None
        assert await backend.fetch_val("SELECT count(*) FROM effects") == 1
        assert calls == 1
    finally:
        release.set()
        if handle is not None:
            handle.task.cancel()
            await asyncio.gather(handle.task, return_exceptions=True)
        await dispatcher.shutdown_durable_delivery()
        for task in agent.tasks:
            task.cancel()
        await asyncio.gather(*agent.tasks, return_exceptions=True)
