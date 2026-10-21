"""Actual COMMIT/rollback exits do not rewrite shared source evidence."""
import asyncio

import asyncpg
import pytest

from kestrel_sovereign._async_ownership import await_owned_task, create_owned_task
from kestrel_sovereign.execution_custody import (
    ExecutionCustody, _explicit_control_chain, bind_execution_custody_snapshot,
    execution_cleanup_debts, execution_commit_outcome,
)
from kestrel_sovereign.storage.db import postgres as postgres_module
from tests.integration.test_execution_authority_postgres import GenerationFence, native_pg as native_pg
from tests.integration.test_execution_authority_postgres import native_pg as _cleanup_pg
from tests.integration.test_maintenance_original_retirement import _original_control
from tests.utils.original_failure import OriginalFailure

cleanup_pg = _cleanup_pg


@pytest.mark.asyncio
@pytest.mark.parametrize("commit", [False, True])
@pytest.mark.parametrize("source_cancel", [False, True])
async def test_native_transaction_exit_preserves_sources_and_actual_withdrawal(
    native_pg, cleanup_pg, monkeypatch, commit, source_cancel,
):
    source = OriginalFailure.capture(await _original_control(native_pg[0], monkeypatch, source_cancel))
    backend, observer, schema = cleanup_pg
    physical = OriginalFailure.capture(OSError("independent actual native exit ACK failed"))
    entered, release, exits, withdrawals = asyncio.Event(), asyncio.Event(), [], []
    native_exit, actual_join = asyncpg.transaction.Transaction.__aexit__, postgres_module.await_owned_task
    async def exit_transaction(transaction, *args):
        result = await native_exit(transaction, *args)
        if not transaction._nested and not exits:
            exits.append((transaction, asyncio.current_task()))
            entered.set()
            await release.wait()
            raise physical.error
        return result
    async def join(*args, **kwargs):
        result = await actual_join(*args, **kwargs)
        if result.cancellation is not None:
            withdrawals.append(result.cancellation)
        return result
    monkeypatch.setattr(asyncpg.transaction.Transaction, "__aexit__", exit_transaction)
    monkeypatch.setattr(postgres_module, "await_owned_task", join)
    async def invoke():
        with bind_execution_custody_snapshot((ExecutionCustody(GenerationFence()),)):
            async with backend.transaction():
                await backend.execute("INSERT INTO effects VALUES (310, 'actual exit composition')")
                if not commit:
                    raise source.error
    owner = create_owned_task(invoke(), retain_originals=True)
    try:
        await asyncio.wait_for(entered.wait(), 5)
        owner.cancel("actual native transaction exit observer withdrew")
        await asyncio.sleep(0)
        release.set()
        result = await await_owned_task(owner)
        if not commit:
            source.assert_retained(result.error)
        physical.assert_retained(result.error)
        assert withdrawals and all(error in tuple(_explicit_control_chain(result.error)) for error in withdrawals)
        assert len(exits) == 1 and exits[0][1].done()
        assert execution_commit_outcome(result.error) == "unknown"
        if not commit:
            assert any(getattr(resource, "context_manager", None) is exits[0][0] and task is exits[0][1]
                       for resource, task in execution_cleanup_debts(result.error))
        assert await observer.fetchval(f'SELECT count(*) FROM "{schema}".effects WHERE id=310') == int(commit)
    finally:
        release.set()
        await await_owned_task(owner)
