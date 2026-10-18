"""Bounded background state must not decorate a genuine earlier source."""
import asyncio

import asyncpg
import pytest

from kestrel_sovereign._async_ownership import await_owned_task, create_owned_task
from kestrel_sovereign.execution_custody import _explicit_control_chain, execution_cleanup_debts
from tests.integration.test_execution_authority_postgres import native_pg as native_pg
from tests.integration.test_execution_authority_postgres import native_pg as _work_pg
from tests.integration.test_host_onboarding_original_admission import onboarding_owner as onboarding_owner
from tests.integration.test_maintenance_original_retirement import _original_control
from tests.utils.original_failure import OriginalFailure

work_pg = _work_pg


@pytest.mark.asyncio
async def test_background_native_facts_remain_private_and_bounded(
    native_pg, work_pg, onboarding_owner, monkeypatch,
):
    _, agent, _database = onboarding_owner
    source = OriginalFailure.capture(await _original_control(native_pg[0], monkeypatch))
    backend, observer, _schema = work_pg
    sessions, closers = [], []
    original_close = asyncpg.Connection.close
    physical = OSError("actual independent background native close ACK failed")

    async def close(connection, *args, **kwargs):
        result = await original_close(connection, *args, **kwargs)
        if sessions and connection is sessions[0]:
            closers.append(asyncio.current_task())
            raise physical
        return result

    monkeypatch.setattr(asyncpg.Connection, "close", close)

    async def acquire_and_fail():
        async with backend.advisory_locks([(210, 319)]) as lease:
            sessions.append(lease._connection._con)
            raise OSError("actual independent advisory body failed")

    outcome = await await_owned_task(create_owned_task(acquire_and_fail(), retain_originals=True))
    native_debts = execution_cleanup_debts(outcome.error)
    assert closers and closers[0].done() and native_debts
    assert await observer.fetchval("SELECT pg_try_advisory_lock(210,319)")
    await observer.execute("SELECT pg_advisory_unlock(210,319)")

    async def publish(error):
        async def failure():
            raise error
        completed = asyncio.Event()
        task = agent._track_background_task(failure(), name="original-background-native")
        task.add_done_callback(lambda _task: completed.set())
        await completed.wait()
        await await_owned_task(task)

    await publish(source.error)
    await publish(outcome.error)
    source.assert_retained(agent._shutdown_control)
    assert all(any(resource is other_resource and task is other_task
                   for other_resource, other_task in execution_cleanup_debts(agent._shutdown_control))
               for resource, task in native_debts)
    bounded = len(tuple(_explicit_control_chain(agent._shutdown_control)))
    for _ in range(10):
        await publish(outcome.error)
        source.assert_retained(agent._shutdown_control)
        assert len(tuple(_explicit_control_chain(agent._shutdown_control))) == bounded
