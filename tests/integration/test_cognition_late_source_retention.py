"""Every existing cognition debt writer shares sources, not finite withdrawals."""
import asyncio

import pytest

from kestrel_sovereign import _async_ownership as ownership
from kestrel_sovereign._async_ownership import await_owned_task, create_owned_task
from kestrel_sovereign.execution_custody import _explicit_control_chain, execution_terminal_error
from tests.integration.test_execution_authority_postgres import native_pg as _native_pg
from tests.integration.test_execution_custody_review18_pg import setup_delivery, join_owned
from tests.integration.test_maintenance_original_retirement import _original_control
from tests.utils.original_failure import OriginalFailure

native_pg = _native_pg


@pytest.mark.asyncio
@pytest.mark.parametrize("writer", ["initial-pending", "completion", "cleanup-harvest"])
@pytest.mark.parametrize("source_cancel", [False, True])
async def test_late_cognition_writer_retains_only_native_source_and_exact_original(
    native_pg, monkeypatch, writer, source_cancel,
):
    source = OriginalFailure.capture(await _original_control(native_pg[0], monkeypatch, source_cancel))
    agent, dispatcher, delivery, _ = await setup_delivery(native_pg)
    observer, schema = native_pg[1:]
    sibling = await observer.fetchrow("SELECT * FROM durable_signal_deliveries WHERE delivery_id='delivery2'")
    route_entered, route_release = asyncio.Event(), asyncio.Event()
    terminal_entered, terminal_release = asyncio.Event(), asyncio.Event()
    owner_entered, owner_release = asyncio.Event(), asyncio.Event()
    withdrawals, route_withdrawals = [], []
    actual_terminal, actual_join = dispatcher._apply_cognition_control_terminal, ownership.await_owned_task
    async def terminal(*args):
        if writer == "cleanup-harvest":
            await actual_terminal(*args)
        terminal_entered.set()
        await terminal_release.wait()
        if writer != "cleanup-harvest":
            await actual_terminal(*args)
    async def observe(task, *args, **kwargs):
        result = await actual_join(task, *args, **kwargs)
        if result.cancellation is not None:
            withdrawals.append(result.cancellation)
        return result
    async def route():
        await observer.execute('INSERT INTO "' + schema + '".effects VALUES ($1,$2)', 378, "actual retained component route")
        route_entered.set()
        try:
            await route_release.wait()
        except asyncio.CancelledError as error:
            route_withdrawals.append(error)
            if writer == "completion":
                raise execution_terminal_error(source.error, error, retain_originals=True, isolate_control_originals=True)
            await route_release.wait()
    monkeypatch.setattr(dispatcher, "_apply_cognition_control_terminal", terminal)
    monkeypatch.setattr(ownership, "await_owned_task", observe)
    routing_task = create_owned_task(route(), retain_originals=True)
    owner = None
    cleanup = None
    try:
        await asyncio.wait_for(route_entered.wait(), 5)
        if writer == "initial-pending":
            async def settle():
                owner_entered.set()
                try:
                    await owner_release.wait()
                except asyncio.CancelledError as error:
                    withdrawals.append(error)
                    finite = execution_terminal_error(source.error, error, retain_originals=True, isolate_control_originals=True)
                    raise await dispatcher._settle_cognition_control(delivery, finite, routing_task)
            owner = create_owned_task(settle(), retain_originals=True)
            await asyncio.wait_for(owner_entered.wait(), 5)
            owner.cancel("actual pending cognition ingress withdrew")
            finite = await await_owned_task(owner)
            assert withdrawals and all(error in tuple(_explicit_control_chain(finite.error)) for error in withdrawals)
            assert routing_task in dispatcher._retained_durable_cognition_tasks and not routing_task.done()
        else:
            dispatcher._retain_durable_cognition_task(
                routing_task, delivery=delivery,
                control_error=None if writer == "completion" else source.error,
            )
            if writer == "completion":
                routing_task.cancel("actual original route finite withdrawal")
            else:
                route_release.set()
            await asyncio.wait_for(terminal_entered.wait(), 5)
            if writer == "completion":
                finite = await await_owned_task(routing_task)
                withdrawals.extend(route_withdrawals)
                assert withdrawals and all(error in tuple(_explicit_control_chain(finite.error)) for error in withdrawals)
            else:
                assert len(dispatcher._retained_durable_cognition_cleanup_tasks) == 1
                cleanup = next(iter(dispatcher._retained_durable_cognition_cleanup_tasks))
                assert await observer.fetchval("SELECT status FROM durable_signal_deliveries WHERE delivery_id='delivery1'") == "failed"
                cleanup.cancel("actual original terminalizer join observer withdrew")
                await asyncio.sleep(0)
                terminal_release.set()
                finite = await await_owned_task(cleanup)
                async with asyncio.timeout(5):
                    while cleanup in dispatcher._retained_durable_cognition_cleanup_tasks:
                        await asyncio.sleep(0)
                assert withdrawals and all(error in tuple(_explicit_control_chain(finite.error)) for error in withdrawals)
        shared = dispatcher._retained_cognition_control_debt[delivery.delivery_id][1]
        source.assert_retained(shared)
        assert all(error not in tuple(_explicit_control_chain(shared)) for error in withdrawals)
        assert await observer.fetchval('SELECT count(*) FROM "' + schema + '".effects WHERE id=378') == 1
        assert await observer.fetchrow("SELECT * FROM durable_signal_deliveries WHERE delivery_id='delivery2'") == sibling
        source.assert_retained(source.error)
    finally:
        route_release.set()
        terminal_release.set()
        owner_release.set()
        if owner is not None:
            await await_owned_task(owner)
        await await_owned_task(routing_task)
        if cleanup is not None:
            await await_owned_task(cleanup)
        await join_owned(agent, dispatcher)
