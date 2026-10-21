"""Actual current shutdown withdrawals never become a shared source graph."""
import asyncio

import pytest

from kestrel_sovereign._async_ownership import await_owned_task, create_owned_task, pending_owned_retirement_error
from kestrel_sovereign.execution_custody import _explicit_control_chain
from kestrel_sovereign.host_features import context as context_module
from kestrel_sovereign.host_features.context import SovereignHostContext
from kestrel_sovereign import _async_ownership as ownership_module
from kestrel_sovereign.storage.db.sqlite import SQLiteBackend
from tests.integration.test_execution_authority_postgres import native_pg as native_pg
from tests.integration.test_host_onboarding_original_admission import onboarding_owner as onboarding_owner
from tests.integration.test_maintenance_original_retirement import _original_control
from tests.utils.original_failure import OriginalFailure


@pytest.mark.asyncio
async def test_shutdown_waiter_keeps_actual_c_without_mutating_pending_native_source(
    onboarding_owner, monkeypatch,
):
    _manager, agent, database = onboarding_owner
    connection = database.backend._connection
    worker = getattr(connection, "_thread", connection)
    entered, release, withdrawals = asyncio.Event(), asyncio.Event(), []

    async def original_close():
        await database.close()
        entered.set()
        await release.wait()

    closer = create_owned_task(original_close(), retain_originals=True)
    await asyncio.wait_for(entered.wait(), 5)
    source = OriginalFailure.capture(pending_owned_retirement_error(
        connection, [closer], message="actual SQLite close acknowledgment still pending",
    ))
    agent._remember_shutdown_control(source.error)
    actual_join = ownership_module.await_owned_task

    async def join(task, *args, **kwargs):
        result = await actual_join(task, *args, **kwargs)
        if task is closer and result.cancellation is not None:
            withdrawals.append(result.cancellation)
        return result

    monkeypatch.setattr(ownership_module, "await_owned_task", join)
    waiter = create_owned_task(agent.wait_for_shutdown_completion(), retain_originals=True)
    try:
        await asyncio.sleep(0)
        waiter.cancel("actual native shutdown completion observer withdrew")
        await asyncio.sleep(0)
        release.set()
        result = await await_owned_task(waiter)
        source.assert_retained(result.error)
        assert withdrawals and all(error in tuple(_explicit_control_chain(result.error)) for error in withdrawals)
        assert all(error not in tuple(_explicit_control_chain(vars(agent).get("_shutdown_control"))) for error in withdrawals)
        assert closer.done() and not worker.is_alive()
        later = await await_owned_task(create_owned_task(agent.wait_for_shutdown_completion(), retain_originals=True))
        source.assert_retained(later.error)
        assert all(error not in tuple(_explicit_control_chain(later.error)) for error in withdrawals)
    finally:
        release.set()
        await await_owned_task(closer)
        await await_owned_task(waiter)


@pytest.mark.asyncio
@pytest.mark.parametrize("native_opening", [False, True])
async def test_standalone_hold_physical_drain_keeps_original_shutdown_c(
    native_pg, onboarding_owner, monkeypatch, native_opening,
):
    _manager, agent, _database = onboarding_owner
    source = OriginalFailure.capture(await _original_control(native_pg[0], monkeypatch)) if native_opening else None
    if source is not None:
        agent._remember_shutdown_control(source.error)
    backend = SQLiteBackend(":memory:")
    await backend.connect()
    connection = backend._connection
    worker = getattr(connection, "_thread", connection)
    context = SovereignHostContext(db=backend)
    agent._standalone_hold_context = context
    entered, release, withdrawals = asyncio.Event(), asyncio.Event(), []
    native_close = connection.close

    async def close():
        await native_close()
        entered.set()
        await release.wait()

    actual_join = context_module.await_owned_task
    async def join(*args, **kwargs):
        result = await actual_join(*args, **kwargs)
        if result.cancellation is not None:
            withdrawals.append(result.cancellation)
        return result

    monkeypatch.setattr(connection, "close", close)
    monkeypatch.setattr(context_module, "await_owned_task", join)
    owner = create_owned_task(agent.shutdown(), retain_originals=True)
    try:
        await asyncio.wait_for(entered.wait(), 5)
        owner.cancel("actual standalone Hold drain observer withdrew")
        await asyncio.sleep(0)
        release.set()
        result = await await_owned_task(owner)
        if source is not None:
            source.assert_retained(result.error)
        assert withdrawals and all(error in tuple(_explicit_control_chain(result.error)) for error in withdrawals)
        assert all(error not in tuple(_explicit_control_chain(vars(agent).get("_shutdown_control"))) for error in withdrawals)
        assert backend._connection is None and not worker.is_alive()
        assert agent._standalone_hold_context is None
    finally:
        release.set()
        await await_owned_task(owner)
        await backend.close()
