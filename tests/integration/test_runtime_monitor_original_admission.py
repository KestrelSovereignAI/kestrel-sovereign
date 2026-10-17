"""A refused owner monitor cannot orphan an actual leased native component.

Actual Agent construction, ledger ownership, PostgreSQL effects and cleanup CAS
are used. The held SQL producer is an explicit private dispatcher component,
not a cognition turn, initialized Agent, Stop completion or READY acceptance.
"""
import asyncio
import inspect
from datetime import datetime, timedelta, timezone

import pytest
from kestrel_sdk.signals import Signal, SignalMode

from kestrel_sovereign._async_ownership import await_owned_task, create_owned_task
from kestrel_sovereign.multi_agent.agent_manager import AgentManager
from kestrel_sovereign.signals import (
    DurableConsumerRegistration, DurableSignalStore, OrderedLockManager,
    SignalDispatcher, SignalLogStore, SourceRegistry,
)
from kestrel_sovereign.signals.sources.channels import DURABLE_COGNITION_CONSUMER_ID
from tests.integration.test_budget_refund_original_control import registered_cleanup_component
from tests.integration.test_execution_authority_postgres import native_pg as native_pg
from tests.unit.test_host_context_native_cleanup import _actual_operation


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["allocation", "callback-before", "callback-after"])
async def test_monitor_refusal_keeps_bounded_retry_and_same_accepted_owner(
    native_pg, tmp_path, monkeypatch, mode,
):
    backend, observer, schema = native_pg
    child, db = await registered_cleanup_component(AgentManager(base_data_dir=tmp_path), schema, tmp_path, monkeypatch)
    store = DurableSignalStore(backend)
    dispatcher = SignalDispatcher(
        agent=child, registry=SourceRegistry(), lock_manager=OrderedLockManager(),
        store=SignalLogStore(backend), durable_store=store,
        runtime_owner_stale_after=timedelta(seconds=60),
    )
    await dispatcher.initialize_durable_delivery()
    await dispatcher._stop_runtime_owner_heartbeat()
    consumer = DURABLE_COGNITION_CONSUMER_ID
    await store.register_consumer(DurableConsumerRegistration(
        consumer_id=consumer, source="native.monitor", agent_id=child.did,
    ))
    await store.persist_signal(
        Signal(source="native.monitor", kind="component", mode=SignalMode.COGNITION,
               target_agent=child.did, payload={}),
        agent_id=child.did, source_event_id="original-monitor", retention_days=1,
    )
    delivery = await store.claim_delivery(
        agent_id=child.did, consumer_id=consumer,
        executor_id=dispatcher._durable_delivery_owner,
    )
    assert delivery is not None
    entered, release = asyncio.Event(), asyncio.Event()
    async def native_producer():
        await backend.execute("INSERT INTO effects VALUES (227, 'original monitor protected component')")
        entered.set()
        await release.wait()
    producer = create_owned_task(native_producer(), retain_originals=True)
    dispatcher._retained_durable_cognition_tasks.add(producer)
    await asyncio.wait_for(entered.wait(), 5)
    stale = datetime.now(timezone.utc) - timedelta(seconds=120)
    loop, previous = asyncio.get_running_loop(), asyncio.get_running_loop().get_task_factory()
    refusal = RuntimeError("actual original monitor " + mode + " refused")
    refused, accepted = [], []
    class RefusedCallback(asyncio.Task):
        def add_done_callback(self, callback, *, context=None):
            if getattr(callback, "__name__", "") == "_finish_runtime_owner_heartbeat":
                if mode == "callback-after":
                    super().add_done_callback(callback, context=context)
                raise refusal
            return super().add_done_callback(callback, context=context)
    def factory(loop, coroutine, **kwargs):
        operation = _actual_operation(coroutine)
        if getattr(operation, "cr_code", None) is SignalDispatcher._heartbeat_runtime_owner.__code__:
            if mode == "allocation":
                refused.append((coroutine, operation))
                raise refusal
            task = RefusedCallback(coroutine, loop=loop, **kwargs)
            accepted.append(task)
            return task
        return previous(loop, coroutine, **kwargs) if previous else asyncio.Task(coroutine, loop=loop, **kwargs)
    try:
        await backend.execute(
            f"UPDATE {store.RUNTIME_OWNERS} SET heartbeat_at=? WHERE agent_id=? AND owner_id=?",
            (store.to_timestamp_param(stale), child.did, dispatcher._durable_delivery_owner),
        )
        loop.set_task_factory(factory)
        with pytest.raises(RuntimeError) as caught:
            dispatcher._start_runtime_owner_heartbeat()
        loop.set_task_factory(previous)
        assert caught.value is refusal
        retry = dispatcher._runtime_owner_heartbeat_timer
        assert retry is not None and not retry.cancelled(), "monitor admission/publication refusal abandoned original owner liveness"
        assert 0 < retry.when() - loop.time() <= dispatcher._runtime_owner_stale_after.total_seconds() / 3
        assert all(inspect.getcoroutinestate(coro) == inspect.CORO_CLOSED for pair in refused for coro in pair)
        retry.cancel()
        if accepted:
            original = accepted[0]
            assert dispatcher._runtime_owner_heartbeat_task is original and not original.done()
            dispatcher._start_runtime_owner_heartbeat()
            assert dispatcher._runtime_owner_heartbeat_task is original, "retry replaced accepted original monitor before physical settlement"
            assert len(accepted) == 1
            outcome = await await_owned_task(original)
            assert outcome.error is None
            await asyncio.sleep(0)
            if dispatcher._runtime_owner_heartbeat_task is original:
                timer = dispatcher._runtime_owner_heartbeat_timer
                assert timer is not None
                timer.cancel()
                dispatcher._start_runtime_owner_heartbeat()
            assert dispatcher._runtime_owner_heartbeat_task is None, "completed monitor without published callback was not harvested by retry"
        else:
            dispatcher._start_runtime_owner_heartbeat()
            original = dispatcher._runtime_owner_heartbeat_task
            assert original is not None
            outcome = await await_owned_task(original)
            assert outcome.error is None
            await asyncio.sleep(0)
        assert not producer.done() and not producer.cancelling()
        recovered = await store.recover_abandoned_leases(
            agent_id=child.did, recovering_owner_id="dispatcher:independent-monitor-reader",
            stale_before=datetime.now(timezone.utc) - timedelta(seconds=60),
        )
        assert recovered == 0, "retry left actual live native lease reclaimable as stale"
        retained = await store.get_delivery(agent_id=child.did, consumer_id=consumer, delivery_id=delivery.delivery_id)
        assert retained.lease_token == delivery.lease_token and retained.lease_owner == dispatcher._durable_delivery_owner
        assert await observer.fetchval(f'SELECT count(*) FROM "{schema}".effects WHERE id=227') == 1
    finally:
        loop.set_task_factory(previous)
        release.set()
        await await_owned_task(producer)
        dispatcher._retained_durable_cognition_tasks.discard(producer)
        for task in accepted:
            await await_owned_task(task)
        for pair in refused:
            for coroutine in pair:
                coroutine.close()
        await dispatcher.shutdown_durable_delivery()
        await child.llm_service.close()
        await db.close()
