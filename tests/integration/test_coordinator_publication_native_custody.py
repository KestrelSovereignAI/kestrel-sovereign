"""Coordinator admissions retain accepted Tasks before native disposition.

Real PG source/rows and SQLite worker lifecycle. Route/shutdown callbacks are
declared component seams; not provider, funds, genesis or deployed acceptance.
"""
import asyncio

import pytest
from kestrel_sdk.signals import SignalResult, Status

from kestrel_sovereign._async_ownership import await_owned_task, create_owned_task
from kestrel_sovereign.execution_custody import execution_cleanup_debts
from tests.integration.test_execution_authority_postgres import native_pg as _native_pg
from tests.integration.test_execution_custody_review18_pg import setup_delivery, join_owned, route
from tests.integration.test_host_onboarding_original_admission import onboarding_owner as _onboarding_owner
from tests.integration.test_maintenance_original_retirement import _original_control
from tests.utils.native_admission_fault import NativeAdmissionFault
from tests.utils.original_failure import OriginalFailure

native_pg = _native_pg
onboarding_owner = _onboarding_owner


@pytest.mark.asyncio
@pytest.mark.parametrize("surface", ["server-phase", "route", "waiter"])
@pytest.mark.parametrize("mode", ["refuse", "publish-before", "publish-after"])
@pytest.mark.parametrize("eager", [False, True])
@pytest.mark.parametrize("source_cancel", [False, True])
async def test_coordinator_publication_joins_exact_owner_before_disposition(
    native_pg, onboarding_owner, monkeypatch, surface, mode, eager, source_cancel,
):
    from kestrel_sovereign import server
    from kestrel_sovereign.kestrel_agent import await_agent_shutdown_completion
    _, real_agent, database = onboarding_owner
    source = OriginalFailure.capture(await _original_control(native_pg[0], monkeypatch, source_cancel))
    connection = database.backend._connection
    worker = getattr(connection, "_thread", connection)
    entered, release, attempted = asyncio.Event(), asyncio.Event(), asyncio.Event()
    calls, continuation = [], None
    agent = dispatcher = None
    async def close_phase():
        calls.append(asyncio.current_task())
        await database.close()
    if surface == "server-phase":
        operation = server._run_lifespan_shutdown_phase("actual-test-phase", close_phase, resources=(database,))
        target = close_phase.__code__
    elif surface == "route":
        agent, dispatcher, delivery, signal = await setup_delivery(native_pg)
        sibling = await native_pg[1].fetchrow("SELECT * FROM durable_signal_deliveries WHERE delivery_id='delivery2'")
        async def actual_route(*args):
            calls.append(asyncio.current_task())
            await native_pg[1].execute("INSERT INTO effects VALUES (491, 'declared original route effect')")
            entered.set()
            await release.wait()
            return SignalResult(signal_id=signal.id, mode=signal.mode, status=Status.FAILED, duration_ms=0, error="declared route ended")
        monkeypatch.setattr(dispatcher, "_route_after_durable_persistence", actual_route)
        operation = route(dispatcher, delivery, signal)
        target = None  # Nested production route's exact qualname below.
    else:
        async def native_tail():
            entered.set()
            await release.wait()
            await database.close()
        continuation = create_owned_task(native_tail(), retain_originals=True)
        real_agent._durable_shutdown_continuation = continuation
        await asyncio.wait_for(entered.wait(), 5)
        operation = await_agent_shutdown_completion(real_agent)
        target = type(real_agent).wait_for_shutdown_completion.__code__
    def predicate(op):
        code = getattr(op, "cr_code", None)
        matched = code is target if target is not None else (code is not None and code.co_name == "route_durable_cognition")
        if matched:
            attempted.set()
        return matched
    fault = NativeAdmissionFault(predicate, source.error, mode, eager=eager)
    owner = create_owned_task(operation, retain_originals=True)
    try:
        with fault:
            if surface == "waiter":
                await asyncio.wait_for(attempted.wait(), 5)
                done, _ = await asyncio.wait({owner}, timeout=0.03)
                assert not done, "failed waiter admission abandoned the original live continuation"
                assert worker.is_alive() and database.backend._connection is connection
                release.set()
            result = await await_owned_task(owner)
        fault.assert_terminal_unstarted()
        failure = result.result[1] if surface == "server-phase" else result.error
        source.assert_retained(failure)
        if surface == "server-phase":
            assert not calls and database.backend._connection is connection and worker.is_alive()
            assert any(resource is database and task is fault.task for resource, task in execution_cleanup_debts(failure))
            assert result.result[2] is None
        elif surface == "route":
            assert not calls
            assert await native_pg[1].fetchval("SELECT count(*) FROM effects WHERE id=491") == 0
            assert await native_pg[1].fetchval("SELECT status FROM durable_signal_deliveries WHERE delivery_id='delivery1'") == "failed"
            assert await native_pg[1].fetchrow("SELECT * FROM durable_signal_deliveries WHERE delivery_id='delivery2'") == sibling
        else:
            assert continuation.done() and database.backend._connection is None and not worker.is_alive()
    finally:
        release.set()
        await await_owned_task(owner)
        for task in fault.accepted:
            await await_owned_task(task)
        if continuation is not None:
            await await_owned_task(continuation)
        if dispatcher is not None:
            await join_owned(agent, dispatcher)
