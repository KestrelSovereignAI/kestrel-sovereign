"""Late ordinary retained route cannot swallow native release control evidence."""

import asyncio
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock

import asyncpg

import pytest

from kestrel_sovereign.execution_custody import ExecutionCustody, execution_commit_outcome
from tests.integration.test_execution_authority_postgres import GenerationFence, native_pg as _native_pg
from tests.integration.test_execution_custody_review18_pg import setup_delivery, join_owned, route
from tests.unit.test_execution_custody_review15 import control_error

pytestmark = [pytest.mark.asyncio, pytest.mark.integration]
native_pg = _native_pg


def delayed_native_commit_ack(monkeypatch, original):
    committed, draining, release = asyncio.Event(), asyncio.Event(), asyncio.Event()
    native_exit = asyncpg.transaction.Transaction.__aexit__

    async def delayed_ack(transaction, error_type, error, traceback):
        result = await native_exit(transaction, error_type, error, traceback)
        if error_type is None:
            # Actual PostgreSQL COMMIT happened; the child still owns its ACK.
            committed.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                draining.set()
                await release.wait()
                raise original
        return result

    monkeypatch.setattr(asyncpg.transaction.Transaction, "__aexit__", delayed_ack)
    return committed, draining, release, native_exit


@pytest.mark.parametrize("control", ["denied", "unknown", "committed"])
@pytest.mark.parametrize("carrier", ["direct", "wrapped", "cancel"])
@pytest.mark.parametrize("receipt_committed", [False, True])
async def test_native_late_retained_release_preserves_original_control(native_pg, control, carrier, receipt_committed):
    backend, controller, _schema = native_pg
    agent, dispatcher, delivery, _signal = await setup_delivery(native_pg)
    error = control_error(control, carrier)
    original_release = dispatcher.release_durable_delivery_after_task
    scope = ExecutionCustody(GenerationFence())
    agent._execution_custody = backend._execution_custody = scope

    async def release(**kwargs):
        if receipt_committed:
            await original_release(**kwargs)
        scope.revoke("late original receipt control")
        raise error

    dispatcher.release_durable_delivery_after_task = AsyncMock(side_effect=release)
    try:
        with pytest.raises(BaseException) as caught:
            await dispatcher._release_retained_durable_cognition_task(delivery)
        assert caught.value is error
        state = await controller.fetchval("SELECT status FROM durable_signal_deliveries WHERE delivery_id='delivery1'")
        # Exact cleanup also covers retry releases that committed before
        # their original control evidence was delivered to the caller.
        assert state == "failed"
        assert not dispatcher._retained_cognition_control_debt
    finally:
        await join_owned(agent, dispatcher)


@pytest.mark.parametrize("control", ["denied", "unknown", "committed"])
@pytest.mark.parametrize("carrier", ["direct", "wrapped", "cancel"])
@pytest.mark.parametrize("repeat", [False, True])
@pytest.mark.parametrize("phase", ["body", "closer"])
async def test_native_renewal_commit_ack_is_joined_under_repeated_closer_cancel(native_pg, monkeypatch, control, carrier, repeat, phase):
    backend, controller, _schema = native_pg
    agent, dispatcher, delivery, _signal = await setup_delivery(native_pg)
    delivery = replace(delivery, lease_expires_at=datetime.now(timezone.utc)+timedelta(seconds=0.03))
    original = control_error(control, carrier)
    scope = ExecutionCustody(GenerationFence())
    backend._execution_custody = scope
    committed, draining, release, native_exit = delayed_native_commit_ack(monkeypatch, original)
    children = []

    async def renew(**kwargs):
        children.append(asyncio.current_task())
        await backend.execute("INSERT INTO effects VALUES (1, 'native renewal committed')")

    dispatcher.renew_durable_delivery_lease = renew
    observed = []

    async def close():
        async with dispatcher._renew_durable_cognition_lease(delivery) as loss:
            observed.append(loss)
            await committed.wait()
            if phase == "body":
                await asyncio.Event().wait()

    task = asyncio.create_task(close())
    try:
        if phase == "body":
            await committed.wait()
            task.cancel()
        await draining.wait()
        assert await controller.fetchval("SELECT count(*) FROM effects") == 1
        task.cancel()
        await asyncio.sleep(0)
        if repeat:
            task.cancel()
            await asyncio.sleep(0)
        assert not task.done(), "native renewal commit escaped its cancelled closer"
        release.set()
        with pytest.raises(BaseException) as caught:
            await task
        assert caught.value is observed[0].result()
        assert execution_commit_outcome(caught.value) == "unknown"
        error = caught.value
        while error is not original and error.__cause__ is not None:
            error = error.__cause__
        assert error is original
        assert all(child.done() for child in children)
        assert await controller.fetchval("SELECT count(*) FROM effects") == 1
    finally:
        release.set()
        await asyncio.gather(task, *children, return_exceptions=True)
        monkeypatch.setattr(asyncpg.transaction.Transaction, "__aexit__", native_exit)
        await join_owned(agent, dispatcher)


@pytest.mark.parametrize("carrier", ["direct", "wrapped", "cancel"])
@pytest.mark.parametrize("stubborn", [False, True])
async def test_cancelled_native_dispatch_retains_late_renewal_commit_control(native_pg, monkeypatch, carrier, stubborn):
    backend, controller, _schema = native_pg
    agent, dispatcher, delivery, signal = await setup_delivery(native_pg)
    delivery = replace(delivery, lease_expires_at=datetime.now(timezone.utc)+timedelta(seconds=0.03))
    original = control_error("unknown", carrier)
    scope = ExecutionCustody(GenerationFence())
    backend._execution_custody = agent._execution_custody = scope
    committed, draining, release, native_exit = delayed_native_commit_ack(monkeypatch, original)
    entered, cancelled, finish = asyncio.Event(), asyncio.Event(), asyncio.Event()
    route_children = []

    async def turn(*args):
        route_children.append(asyncio.current_task())
        entered.set()
        try:
            await finish.wait()
        except asyncio.CancelledError:
            cancelled.set()
            if not stubborn:
                raise
            await finish.wait()

    async def renew(**kwargs):
        await entered.wait()
        await backend.execute("INSERT INTO effects VALUES (1, 'native dispatch renewal committed')")

    dispatcher._route_after_durable_persistence = turn
    dispatcher.renew_durable_delivery_lease = renew
    task = asyncio.create_task(route(dispatcher, delivery, signal))
    try:
        await committed.wait()
        task.cancel()  # Dispatch is waiting for its route or renewal Future.
        await draining.wait()
        task.cancel()  # Repeated cancellation while the native ACK is draining.
        await asyncio.sleep(0)
        assert not task.done()
        release.set()
        with pytest.raises(BaseException) as caught:
            await task
        assert execution_commit_outcome(caught.value) == "unknown"
        error = caught.value
        while error is not original and error.__cause__ is not None:
            error = error.__cause__
        assert error is original
        monkeypatch.setattr(asyncpg.transaction.Transaction, "__aexit__", native_exit)
        await asyncio.wait_for(cancelled.wait(), timeout=1)
        if stubborn:
            assert dispatcher._retained_cognition_control_debt == {delivery.delivery_id: (delivery, caught.value)}
            assert await controller.fetchval("SELECT status FROM durable_signal_deliveries WHERE delivery_id='delivery1'") == "leased"
        finish.set()
        await asyncio.gather(*route_children, return_exceptions=True)
        await asyncio.sleep(0)
        await dispatcher._drain_retained_durable_cognition_cleanup_tasks()
        assert not dispatcher._retained_cognition_control_debt
        assert await controller.fetchval("SELECT status FROM durable_signal_deliveries WHERE delivery_id='delivery1'") == "failed"
        assert await controller.fetchval("SELECT count(*) FROM effects") == 1
    finally:
        release.set()
        finish.set()
        await asyncio.gather(task, *route_children, return_exceptions=True)
        monkeypatch.setattr(asyncpg.transaction.Transaction, "__aexit__", native_exit)
        await asyncio.sleep(0)
        await join_owned(agent, dispatcher)
