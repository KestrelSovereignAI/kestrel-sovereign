"""Successful native COMMIT remains a control receipt after observer withdrawal."""

import asyncio
from uuid import uuid4

import asyncpg
import pytest

from kestrel_sovereign._async_ownership import await_owned_task, create_owned_task
from kestrel_sovereign.execution_custody import (
    ExecutionCustody,
    ProcessRuntimeExecutionFence,
    bind_execution_custody_snapshot,
    execution_cleanup_debts,
    execution_commit_outcome,
    execution_denial_scopes,
    execution_terminal_error,
    is_execution_control_error,
    require_execution_work,
)
from kestrel_sovereign.features.scheduler import runner as scheduler_module
from kestrel_sovereign.features.scheduler.runner import SchedulerRunner
from kestrel_sovereign.storage.async_database import AsyncDatabase
from tests.integration.test_execution_authority_postgres import native_pg as native_pg
from tests.integration.test_scheduler_postgres_claims import (
    _activate_protocol_for_test_agent,
)
from tests.unit.test_scheduler_occurrence_liveness import _seed_due


@pytest.mark.asyncio
@pytest.mark.parametrize("with_denial", [False, True])
async def test_original_acknowledged_commit_cannot_be_classified_as_local_denial(
    native_pg, monkeypatch, with_denial
):
    backend, observer, schema = native_pg
    db = AsyncDatabase(backend)
    did = "did:web:committed-control-" + uuid4().hex
    runtime = ExecutionCustody(ProcessRuntimeExecutionFence("original-committed-host"))
    committed, release, effects, notifications = (
        asyncio.Event(),
        asyncio.Event(),
        [],
        [],
    )
    native_exit = asyncpg.transaction.Transaction.__aexit__
    tick, occurrences, locked = None, [], False

    async def executor(name, args):
        effects.append(asyncio.current_task())
        try:
            async with db.transaction():
                await db.execute(
                    "INSERT INTO effects VALUES (21, 'original committed')"
                )
        except BaseException as original:
            if not with_denial:
                raise
            assert execution_commit_outcome(original) == "committed"
            state = scheduler_module._rollout_renewal_state.get()
            expiry = create_owned_task(state.lost.wait())
            observation = await await_owned_task(expiry)
            with pytest.raises(BaseException) as denied:
                require_execution_work()
            raise execution_terminal_error(
                original,
                denied.value,
                observation.cancellation,
                retain_originals=True,
            )
        pytest.fail(
            "withdrawn native observer incorrectly resumed ordinary publication"
        )

    async def held_original_ack(transaction, error_type, error, tb):
        result = await native_exit(transaction, error_type, error, tb)
        if (
            effects
            and not committed.is_set()
            and error_type is None
            and not transaction._nested
            and asyncio.current_task().get_name()
            == effects[0].get_name() + ":postgres-original-transaction-exit"
        ):
            committed.set()
            await release.wait()
        return result

    await _activate_protocol_for_test_agent(db, did, monkeypatch)
    runner = SchedulerRunner(
        db,
        did,
        executor,
        lease_seconds=2 if with_denial else 20,
        on_protocol_failure=notifications.append,
    )
    await runner._ensure_tables()
    await _seed_due(db, uuid4().hex, "original-committed-control", agent_id=did)
    monkeypatch.setattr(asyncpg.transaction.Transaction, "__aexit__", held_original_ack)
    try:
        with bind_execution_custody_snapshot((runtime,)):
            tick = create_owned_task(runner._tick())
        await asyncio.wait_for(committed.wait(), 5)
        await await_owned_task(tick)
        occurrences = [entry.task for entry in runner._occurrences.values()]
        assert len(effects) == len(occurrences) == 1
        assert (
            await observer.fetchval(
                f'SELECT COUNT(*) FROM "{schema}".effects WHERE id=21'
            )
            == 1
        )
        if with_denial:
            await observer.execute("BEGIN")
            locked = True
            await observer.fetchval(
                f'SELECT id FROM "{schema}".scheduled_tasks WHERE agent_id=$1 FOR UPDATE',
                did,
            )
        effects[0].cancel("original executor observer withdrew after COMMIT")
        release.set()
        outcome = await asyncio.wait_for(await_owned_task(effects[0]), 5)
        assert (
            outcome.error is not None
            and execution_commit_outcome(outcome.error) == "committed"
        )
        assert bool(execution_denial_scopes(outcome.error)) is with_denial
        assert not execution_cleanup_debts(outcome.error)
        if locked:
            await observer.execute("ROLLBACK")
            locked = False
        for task in occurrences:
            await asyncio.wait_for(await_owned_task(task), 5)
        await asyncio.sleep(0)
        assert runner.readiness_failure is not None, (
            "acknowledged original COMMIT control was silently localized as an empty denial"
        )
        assert execution_commit_outcome(runner.readiness_failure) == "committed"
        assert notifications and all(
            execution_commit_outcome(error) == "committed" for error in notifications
        )
        assert all(
            receipt is not outcome.error and execution_commit_outcome(receipt) is None
            for receipt in runner._occurrence_control_receipts
        )
        runtime.require_work()
    finally:
        release.set()
        if locked:
            await observer.execute("ROLLBACK")
        if tick is not None:
            await await_owned_task(tick)
        for task in occurrences:
            if not task.done():
                task.cancel()
            await await_owned_task(task)
        try:
            await runner.stop()
        except BaseException as error:
            if not is_execution_control_error(error):
                raise
