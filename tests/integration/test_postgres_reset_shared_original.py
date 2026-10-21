"""A real reset must not append its private retirement to an earlier receipt."""
import asyncio

import asyncpg
import pytest

from kestrel_sovereign._async_ownership import await_owned_task, create_owned_task
from kestrel_sovereign.execution_custody import execution_cleanup_debts, ExecutionCustody, bind_execution_custody_snapshot
from tests.integration.test_execution_authority_postgres import native_pg as native_pg
from tests.integration.test_execution_authority_postgres import native_pg as _cleanup_pg
from tests.integration.test_execution_authority_postgres import GenerationFence
from tests.integration.test_maintenance_original_retirement import _original_control
from tests.utils.original_failure import OriginalFailure

cleanup_pg = _cleanup_pg


@pytest.mark.asyncio
@pytest.mark.parametrize("source_cancel", [False, True])
async def test_actual_reset_retains_shared_original_without_mutating_source(
    native_pg, cleanup_pg, monkeypatch, source_cancel,
):
    source = OriginalFailure.capture(await _original_control(native_pg[0], monkeypatch, source_cancel))
    backend, observer, schema = cleanup_pg
    actual_reset, owners = asyncpg.pool.PoolAcquireContext.__aexit__, []
    async def reset(checkout, *args):
        result = await actual_reset(checkout, *args)
        if checkout.pool is backend._pool:
            owners.append((checkout, asyncio.current_task()))
            raise source.error
        return result
    monkeypatch.setattr(asyncpg.pool.PoolAcquireContext, "__aexit__", reset)
    async def invoke():
        with bind_execution_custody_snapshot((ExecutionCustody(GenerationFence()),)):
            return await backend.execute("INSERT INTO effects VALUES (307, 'actual reset source preservation')")
    owner = create_owned_task(invoke(), retain_originals=True)
    result = await await_owned_task(owner)
    assert await observer.fetchval(f'SELECT count(*) FROM "{schema}".effects WHERE id=307') == 1
    assert len(owners) == 1 and owners[0][1].done()
    source.assert_retained(result.error)
    assert any(resource.context_manager is owners[0][0] and task is owners[0][1]
               for resource, task in execution_cleanup_debts(result.error)
               if hasattr(resource, "context_manager"))
