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
from kestrel_sovereign.execution_custody import _explicit_control_chain, execution_cleanup_debts, is_execution_control_error
from kestrel_sovereign.multi_agent.agent_manager import AgentManager
from kestrel_sovereign.signals import (
    DurableConsumerRegistration, DurableSignalStore, OrderedLockManager,
    SignalDispatcher, SignalLogStore, SourceRegistry,
)
from kestrel_sovereign.signals.sources.channels import DURABLE_COGNITION_CONSUMER_ID
from tests.integration.test_budget_refund_original_control import registered_cleanup_component
from tests.integration.test_execution_authority_postgres import native_pg as native_pg
from tests.unit.test_host_context_native_cleanup import _actual_operation
from tests.integration.test_execution_custody_review56_pg import lose_native_ack
from tests.shared.original_task_admission import assert_unknown_task_admission


@pytest.mark.asyncio
@pytest.mark.parametrize("eager", [False, True] if hasattr(asyncio, "eager_task_factory") else [False])
@pytest.mark.parametrize("mode", [
    "allocation", "callback-before", "callback-after",
    "tracker-before", "tracker-after", "canonical-before", "canonical-after",
    "tracker-control-before", "tracker-control-after",
    "tracker-control-stop-before", "tracker-control-stop-after",
])
async def test_monitor_refusal_keeps_bounded_retry_and_same_accepted_owner(
    native_pg, tmp_path, monkeypatch, mode, eager,
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
    native_fault_context = monkeypatch.context()
    native_fault = native_fault_context.__enter__()
    native_receipts = None
    native_original = None
    admission_error = admission_receipt = None
    class RefusedCallback(asyncio.Task):
        def add_done_callback(self, callback, *, context=None):
            target = {"tracker": "completed", "canonical": "close_unstarted"}.get(
                mode.split("-")[0], "_finish_runtime_owner_heartbeat",
            )
            if getattr(callback, "__name__", "") == target:
                if mode.endswith("-after"):
                    super().add_done_callback(callback, context=context)
                raise refusal
            return super().add_done_callback(callback, context=context)
    def factory(loop, coroutine, **kwargs):
        operation = _actual_operation(coroutine)
        if getattr(operation, "cr_code", None) is SignalDispatcher._heartbeat_runtime_owner.__code__:
            if mode == "allocation":
                refused.append((coroutine, operation))
                raise refusal
            options = {"eager_start": True} if eager else {}
            task = RefusedCallback(coroutine, loop=loop, **kwargs, **options)
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
        if mode == "allocation":
            admission_receipt = assert_unknown_task_admission(caught.value, refusal)
            admission_error = dispatcher._outcome_log_control_error
            assert_unknown_task_admission(admission_error, refusal, admission=admission_receipt)
            assert dispatcher._runtime_owner_heartbeat_task is None
        else:
            assert caught.value is refusal
        retry = dispatcher._runtime_owner_heartbeat_timer
        assert retry is not None and not retry.cancelled(), "monitor admission/publication refusal abandoned original owner liveness"
        assert 0 < retry.when() - loop.time() <= dispatcher._runtime_owner_stale_after.total_seconds() / 3
        assert all(inspect.getcoroutinestate(coro) == inspect.CORO_CLOSED for pair in refused for coro in pair)
        retry.cancel()
        if accepted:
            original = accepted[0]
            assert dispatcher._runtime_owner_heartbeat_task is original and not original.done(), "monitor lost original accepted Task across tracker publication"
            dispatcher._start_runtime_owner_heartbeat()
            assert dispatcher._runtime_owner_heartbeat_task is original, "retry replaced accepted original monitor before physical settlement"
            assert len(accepted) == 1
            if "-control-" in mode:
                native_receipts = lose_native_ack(
                    native_fault, boundary="commit",
                    original=OSError("actual cleanup-owner CAS COMMIT ACK lost"),
                )
            stopped = None
            if "-stop-" in mode:
                with pytest.raises(BaseException) as stopped:
                    await dispatcher._stop_runtime_owner_heartbeat()
            outcome = await await_owned_task(original)
            if "-control-" in mode:
                native_original = outcome.error
                assert native_receipts == [True] and is_execution_control_error(native_original)
                original_graph = tuple(_explicit_control_chain(native_original))
                original_debts = tuple(execution_cleanup_debts(native_original))
                if stopped is not None:
                    assert native_original in tuple(_explicit_control_chain(stopped.value))
                native_fault_context.__exit__(None, None, None)
                native_fault = None
            else:
                assert outcome.error is (refusal if mode.startswith("canonical") else None)
            await asyncio.sleep(0)
            if dispatcher._runtime_owner_heartbeat_task is original:
                timer = dispatcher._runtime_owner_heartbeat_timer
                assert timer is not None
                timer.cancel()
                dispatcher._start_runtime_owner_heartbeat()
            assert dispatcher._runtime_owner_heartbeat_task is None, "completed monitor without published callback was not harvested by retry"
            assert original not in child._background_tasks, "terminal monitor harvester did not finish original Agent bookkeeping"
            if native_original is not None:
                with pytest.raises(BaseException) as retained:
                    child._raise_shutdown_control(control_only=True)
                assert native_original in tuple(_explicit_control_chain(retained.value))
                assert native_original in tuple(_explicit_control_chain(dispatcher._outcome_log_control_error))
                assert tuple(_explicit_control_chain(native_original)) == original_graph
                assert tuple(execution_cleanup_debts(native_original)) == original_debts
            # A delayed duplicate callback may never retire its successor.
            # Canonical refusal ran no SQL; the fresh attempt must refresh CAS.
            if dispatcher._runtime_owner_heartbeat_timer is not None:
                dispatcher._runtime_owner_heartbeat_timer.cancel()
            dispatcher._start_runtime_owner_heartbeat()
            successor = dispatcher._runtime_owner_heartbeat_task
            assert successor is not None and successor is not original
            dispatcher._finish_runtime_owner_heartbeat(original)
            assert dispatcher._runtime_owner_heartbeat_task is successor, "old monitor callback retired successor owner"
            assert (await await_owned_task(successor)).error is None
            await asyncio.sleep(0)
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
        if "-stop-" in mode:
            release.set()
            await await_owned_task(producer)
            dispatcher._retained_durable_cognition_tasks.discard(producer)
            for _ in range(2):
                with pytest.raises(BaseException) as repeated:
                    await dispatcher.shutdown_durable_delivery()
                assert native_original in tuple(_explicit_control_chain(repeated.value)), "repeated shutdown lost original monitor control"
    finally:
        loop.set_task_factory(previous)
        if native_fault is not None:
            native_fault_context.__exit__(None, None, None)
        release.set()
        await await_owned_task(producer)
        dispatcher._retained_durable_cognition_tasks.discard(producer)
        for task in accepted:
            await await_owned_task(task)
        for pair in refused:
            for coroutine in pair:
                coroutine.close()
        try:
            try:
                await dispatcher.shutdown_durable_delivery()
            except BaseException as error:
                if mode == "allocation":
                    if admission_error is not None:
                        assert admission_error in tuple(_explicit_control_chain(error))
                    assert_unknown_task_admission(error, refusal, admission=admission_receipt)
                else:
                    assert native_original is not None and native_original in tuple(_explicit_control_chain(error))
            else:
                assert mode != "allocation", "shutdown erased unresolved original admission"
        finally:
            try:
                await child.llm_service.close()
            finally:
                await db.close()
