"""Native Hold rollback retains original read and exact-token release control.

Private dispatcher components and real PG ledger/CAS are used, not cognition,
Agent initialization/READY, external Hold authority or a live tenant.
"""
import asyncio
from datetime import datetime, timezone

import pytest
from kestrel_sdk.signals import Signal, SignalMode

from kestrel_sovereign._async_ownership import await_owned_task, create_owned_task
from kestrel_sovereign.execution_custody import (
    _explicit_control_chain, bind_execution_custody, execution_cleanup_debts,
    execution_commit_outcome, is_execution_control_error,
)
from kestrel_sovereign.multi_agent.agent_manager import AgentManager
from kestrel_sovereign.signals import (
    DurableConsumerRegistration, DurableSignalStore, OrderedLockManager,
    SignalDispatcher, SignalLogStore, SourceRegistry,
)
from kestrel_sovereign.signals.dispatcher import _TransientDurableHandoff
from tests.integration.test_budget_refund_original_control import registered_cleanup_component
from tests.integration.test_execution_authority_postgres import GenerationFence, native_pg as native_pg
from tests.integration.test_execution_custody_review56_pg import lose_native_ack


@pytest.mark.asyncio
@pytest.mark.parametrize("route", ["postclaim", "initial"])
@pytest.mark.parametrize("read_cancel", [False, True])
@pytest.mark.parametrize("release_cancel", [False, True])
async def test_hold_rollback_never_discards_native_release_after_read_failure(
    native_pg, tmp_path, monkeypatch, route, read_cancel, release_cancel,
):
    backend, observer, schema = native_pg
    child, db = await registered_cleanup_component(AgentManager(base_data_dir=tmp_path), schema, tmp_path, monkeypatch)
    store = DurableSignalStore(backend)
    dispatcher = SignalDispatcher(
        agent=child, registry=SourceRegistry(), lock_manager=OrderedLockManager(),
        store=SignalLogStore(backend), durable_store=store,
    )
    await dispatcher.initialize_durable_delivery()
    await dispatcher._stop_runtime_owner_heartbeat()
    consumer = "original-hold-rollback"
    await store.register_consumer(DurableConsumerRegistration(
        consumer_id=consumer, source="native.hold", agent_id=child.did,
    ))
    persisted = await store.persist_signal(
        Signal(source="native.hold", kind="component", mode=SignalMode.COGNITION,
               target_agent=child.did, payload={}),
        agent_id=child.did, source_event_id="original-hold", retention_days=1,
        **({"transient_selector_payload": {}, "initial_lease_owner": dispatcher._durable_delivery_owner}
           if route == "initial" else {}),
    )
    if route == "initial":
        reservation = persisted.initial_reservations[0]
        delivery = await store.activate_initial_delivery(
            agent_id=child.did, consumer_id=consumer, delivery_id=reservation.delivery_id,
            initial_lease_owner=dispatcher._durable_delivery_owner,
            initial_lease_token=reservation.reservation_token,
        )
        now = datetime.now(timezone.utc)
        dispatcher._transient_durable_handoffs[delivery.delivery_id] = _TransientDurableHandoff(
            payload={}, consumer_id=consumer, created_at=now,
            retention_until=persisted.retention_until, expires_at=delivery.lease_expires_at,
            initial_lease_token=delivery.lease_token,
        )
    else:
        delivery = await store.claim_delivery(
            agent_id=child.did, consumer_id=consumer,
            executor_id=dispatcher._durable_delivery_owner, cognition_admission_pending=True,
        )
        dispatcher._track_pending_cognition_admission(delivery, executor_id=dispatcher._durable_delivery_owner)
    originals, graphs, debts, reads = [], [], [], []
    read_failure = RuntimeError("actual Hold observer read refused after original claim")
    actual_release = store.release_delivery_for_hold
    operation = None

    async def failed_hold(_consumer):
        await backend.fetch_val("SELECT generation FROM authority WHERE id=1")
        if read_cancel:
            asyncio.current_task().cancel("actual original Hold reader withdrew")
            try:
                await asyncio.sleep(0)
            except asyncio.CancelledError as error:
                reads.append(error)
                raise
        reads.append(read_failure)
        raise read_failure

    async def native_release(**kwargs):
        with bind_execution_custody(GenerationFence()), monkeypatch.context() as ack:
            receipts = lose_native_ack(ack, boundary="commit", original=OSError("actual Hold rollback COMMIT ACK lost"))
            try:
                return await actual_release(**kwargs)
            except BaseException as error:
                assert receipts == [True]
                if release_cancel:
                    asyncio.current_task().cancel("actual original Hold release observer withdrew")
                    try:
                        await asyncio.sleep(0)
                    except asyncio.CancelledError as cancellation:
                        cancellation.__cause__ = error
                        error = cancellation
                originals.append(error)
                graphs.append(tuple(_explicit_control_chain(error)))
                debts.append(tuple(execution_cleanup_debts(error)))
                raise error

    monkeypatch.setattr(dispatcher, "_durable_claim_deferred_by_hold", failed_hold)
    monkeypatch.setattr(store, "release_delivery_for_hold", native_release)
    try:
        if route == "postclaim":
            work = dispatcher._fence_claimed_durable_delivery_after_hold_race(consumer_id=consumer, delivery=delivery)
        else:
            work = dispatcher.claim_durable_delivery_for_event(
                consumer_id=consumer, event_id=persisted.event_id, executor_id="unaccepted-executor",
            )
        operation = create_owned_task(work, retain_originals=True)
        outcome = await await_owned_task(operation)
        chain = tuple(_explicit_control_chain(outcome.error))
        assert originals and originals[0] in chain, "Hold rollback discarded actual native release control"
        assert reads and reads[0] in chain, "Hold rollback discarded actual original read withdrawal"
        assert execution_commit_outcome(outcome.error) == "unknown"
        for _ in range(3):
            with pytest.raises(BaseException) as blocked:
                async with dispatcher._admit_durable_operation():
                    pytest.fail("unresolved native Hold rollback admitted fresh work")
            assert originals[0] in tuple(_explicit_control_chain(blocked.value))
            assert reads[0] not in tuple(_explicit_control_chain(blocked.value)), "private Hold reader contaminated shared writer latch"
        assert tuple(_explicit_control_chain(originals[0])) == graphs[0]
        assert tuple(execution_cleanup_debts(originals[0])) == debts[0]
        row = await observer.fetchrow(f'SELECT status,attempts,lease_token FROM "{schema}".durable_signal_deliveries WHERE delivery_id=$1', delivery.delivery_id)
        assert tuple(row) == ("retry", 0, None), "exact Hold rollback did not physically settle attempt-neutral"
        assert delivery.delivery_id not in dispatcher._pending_cognition_admissions
        assert await observer.fetchval(f'SELECT count(*) FROM "{schema}".durable_signal_events') == 1
    finally:
        if operation is not None:
            await await_owned_task(operation)
        try:
            await dispatcher.shutdown_durable_delivery()
        except BaseException as error:
            assert is_execution_control_error(error)
        await db.close()
