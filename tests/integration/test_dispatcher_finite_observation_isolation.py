"""A real PG repair's current observation C is never shared native history."""
import asyncio

import pytest
from kestrel_sdk.signals import Signal, SignalMode

from kestrel_sovereign import _async_ownership as ownership
from kestrel_sovereign._async_ownership import await_owned_task, create_owned_task
from kestrel_sovereign.execution_custody import _explicit_control_chain
from kestrel_sovereign.signals import DurableConsumerRegistration
from tests.integration.test_execution_authority_postgres import native_pg as _native_pg
from tests.integration.test_execution_custody_review18_pg import setup_delivery, join_owned
from tests.integration.test_maintenance_original_retirement import _original_control
from tests.utils.original_failure import OriginalFailure

native_pg = _native_pg


@pytest.mark.asyncio
@pytest.mark.parametrize("installed", [False, True])
async def test_reservation_writer_shared_source_excludes_actual_join_withdrawal(
    native_pg, monkeypatch, installed,
):
    source = OriginalFailure.capture(await _original_control(native_pg[0], monkeypatch))
    agent, dispatcher, _delivery, _ = await setup_delivery(native_pg)
    store, observer = dispatcher._durable_store, native_pg[1]
    await store.register_consumer(DurableConsumerRegistration(
        consumer_id="finite-writer-observer", source="native.finite.writer", agent_id=agent.did,
    ))
    persisted = await store.persist_signal(
        Signal(source="native.finite.writer", kind="component", mode=SignalMode.COGNITION, target_agent=agent.did, payload={}),
        agent_id=agent.did, source_event_id="finite-writer-observer", retention_days=1,
        transient_selector_payload={}, initial_lease_owner=dispatcher._durable_delivery_owner,
    )
    assert len(persisted.initial_reservations) == 1
    identity = persisted.initial_reservations[0].delivery_id
    before = await observer.fetchrow("SELECT * FROM durable_signal_deliveries WHERE delivery_id=$1", identity)
    entered, release, attempts, withdrawals = asyncio.Event(), asyncio.Event(), [], []
    # Put the genuine native source at the publication boundary while the
    # original repair itself succeeds. Raising it inside the repair would
    # pre-populate the shared latch and miss the exceptional join's gap.
    publication = source.error
    native_abandon, native_join = store.abandon_initial_reservation, ownership.await_owned_task

    class FailedPublication(set):
        def add(self, task):
            if installed:
                super().add(task)
            raise publication

    async def abandon(**kwargs):
        result = await native_abandon(**kwargs)
        attempts.append(result)
        entered.set()
        await release.wait()
        return result

    owner = None

    async def observe(task, *args, **kwargs):
        result = await native_join(task, *args, **kwargs)
        if asyncio.current_task() is owner and result.cancellation is not None:
            withdrawals.append(result.cancellation)
        return result

    monkeypatch.setattr(store, "abandon_initial_reservation", abandon)
    monkeypatch.setattr(ownership, "await_owned_task", observe)
    dispatcher._post_commit_reservation_repairs = FailedPublication()
    owner = create_owned_task(dispatcher._repair_post_commit_reservations(persisted), retain_originals=True)
    try:
        await asyncio.wait_for(entered.wait(), 5)
        assert await observer.fetchrow("SELECT * FROM durable_signal_deliveries WHERE delivery_id=$1", identity) != before
        owner.cancel("actual reservation repair join observer withdrew")
        await asyncio.sleep(0)
        release.set()
        result = await await_owned_task(owner)
        source.assert_retained(result.error)
        chain = tuple(_explicit_control_chain(result.error))
        assert publication in chain and withdrawals and all(error in chain for error in withdrawals)
        source.assert_retained(dispatcher._outcome_log_control_error)
        shared_chain = tuple(_explicit_control_chain(dispatcher._outcome_log_control_error))
        assert all(error not in shared_chain for error in withdrawals)
        assert len(attempts) == 1
    finally:
        release.set()
        await await_owned_task(owner)
        await join_owned(agent, dispatcher)
