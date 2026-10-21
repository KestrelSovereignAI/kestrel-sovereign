"""Supported adapters and savepoints retain independent native exit evidence."""
import asyncio
from contextlib import asynccontextmanager

import asyncpg
import pytest

from kestrel_sovereign._async_ownership import await_owned_task, create_owned_task
from kestrel_sovereign.execution_custody import (
    ExecutionCommitOutcomeError, ExecutionCustody, _explicit_control_chain, bind_execution_custody_snapshot,
)
from kestrel_sovereign.storage.db.postgres import PostgresPoolAdapter
from tests.integration.test_execution_authority_postgres import GenerationFence, native_pg as native_pg
from tests.integration.test_execution_authority_postgres import native_pg as _cleanup_pg
from tests.integration.test_execution_authority_postgres import native_pg as _work_pg
from tests.integration.test_maintenance_original_retirement import _original_control
from tests.utils.original_failure import OriginalFailure

cleanup_pg, work_pg = _cleanup_pg, _work_pg


@pytest.mark.asyncio
@pytest.mark.parametrize("door", ["adapter", "savepoint"])
@pytest.mark.parametrize("source_cancel", [False, True])
@pytest.mark.parametrize("native_cleanup", [False, True])
async def test_outer_pg_finalizer_keeps_both_original_graphs(
    native_pg, cleanup_pg, work_pg, monkeypatch, door, source_cancel, native_cleanup,
):
    original = await _original_control(native_pg[0], monkeypatch, source_cancel)
    # Exercise the actual native receipt too: an ordinary QueryError wrapper
    # incidentally protects its cause at the adapter boundary. That is not
    # protection for a shared native receipt which escapes without wrapping.
    if not source_cancel:
        original = next(error for error in _explicit_control_chain(original)
                        if isinstance(error, ExecutionCommitOutcomeError))
    source = OriginalFailure.capture(original)
    cleanup = OriginalFailure.capture(await _original_control(cleanup_pg[0], monkeypatch)) if native_cleanup else OriginalFailure.capture(OSError("independent native-savepoint/adapter exit observation"))
    backend, observer, schema = work_pg
    pool, exits = backend._pool, []
    actual_exit = asyncpg.transaction.Transaction.__aexit__
    async def exit_transaction(transaction, *args):
        result = await actual_exit(transaction, *args)
        exits.append((transaction, asyncio.current_task()))
        if door == "savepoint" and transaction._nested:
            raise cleanup.error
        return result
    @asynccontextmanager
    async def wrapper(native):
        try:
            async with native as connection:
                yield connection
        finally:
            raise cleanup.error
    monkeypatch.setattr(asyncpg.transaction.Transaction, "__aexit__", exit_transaction)
    if door == "adapter":
        backend._pool = PostgresPoolAdapter(pool, wrapper)
    async def invoke():
        with bind_execution_custody_snapshot((ExecutionCustody(GenerationFence()),)):
            async with backend.transaction():
                if door == "savepoint":
                    async with backend.transaction():
                        await backend.execute("INSERT INTO effects VALUES (311, 'actual savepoint body')")
                        raise source.error
                await backend.execute("INSERT INTO effects VALUES (311, 'actual adapter body')")
                raise source.error
    owner = create_owned_task(invoke(), retain_originals=True)
    try:
        result = await await_owned_task(owner)
        source.assert_retained(result.error)
        cleanup.assert_retained(result.error)
        assert cleanup.error in tuple(_explicit_control_chain(result.error))
        assert exits and all(task.done() for _context, task in exits)
        assert await observer.fetchval(f'SELECT count(*) FROM "{schema}".effects WHERE id=311') == 0
    finally:
        backend._pool = pool
        await await_owned_task(owner)
