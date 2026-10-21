"""A repeated source object still needs a PRIVATE advisory-retirement carrier."""
import asyncio

import asyncpg
import pytest

from kestrel_sovereign._async_ownership import await_owned_task, create_owned_task
from kestrel_sovereign.execution_custody import execution_cleanup_debts
from tests.integration.test_execution_authority_postgres import native_pg as native_pg
from tests.integration.test_execution_authority_postgres import native_pg as _work_pg
from tests.integration.test_maintenance_original_retirement import _original_control
from tests.utils.original_failure import OriginalFailure

work_pg = _work_pg


@pytest.mark.asyncio
@pytest.mark.parametrize("source_cancel", [False, True])
async def test_same_native_source_is_not_decorated_by_advisory_close_debt(
    native_pg, work_pg, monkeypatch, source_cancel,
):
    source = OriginalFailure.capture(await _original_control(native_pg[0], monkeypatch, source_cancel))
    backend, observer, _schema = work_pg
    native_close = asyncpg.Connection.close
    sessions, closers = [], []
    async def close(connection, *args, **kwargs):
        result = await native_close(connection, *args, **kwargs)
        if sessions and connection is sessions[0][1]:
            closers.append(asyncio.current_task())
            raise source.error
        return result
    monkeypatch.setattr(asyncpg.Connection, "close", close)
    async def invoke():
        async with backend.advisory_locks([(209,316)]) as lease:
            proxy = lease._connection
            sessions.append((proxy, proxy._con))
            raise source.error
    owner = create_owned_task(invoke(), retain_originals=True)
    try:
        result = await await_owned_task(owner)
        source.assert_retained(result.error)
        assert len(closers) == 1 and closers[0].done() and sessions[0][1].is_closed()
        assert any(getattr(resource, "context_manager", None) is sessions[0][0] and task is closers[0]
                   for resource, task in execution_cleanup_debts(result.error))
        assert await observer.fetchval("SELECT pg_try_advisory_lock(209,316)")
        assert await observer.fetchval("SELECT pg_advisory_unlock(209,316)")
    finally:
        await await_owned_task(owner)
        await observer.execute("SELECT pg_advisory_unlock(209,316)")
