"""Original Scheduler deadline loss at native evidence ACK, before scope yield."""

import asyncio
from contextlib import asynccontextmanager
from uuid import uuid4

import asyncpg
import pytest

from kestrel_sovereign._async_ownership import await_owned_task, create_owned_task
from kestrel_sovereign.execution_custody import (
    ExecutionCustody,
    ProcessRuntimeExecutionFence,
    bind_execution_custody_snapshot,
    execution_commit_outcome,
    execution_denial_scopes,
    is_execution_control_error,
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
async def test_original_deadline_expiry_before_scope_yield_is_not_host_failure(
    native_pg, monkeypatch
):
    backend, observer, schema = native_pg
    db = AsyncDatabase(backend)
    did, schedule = "did:web:entry-expiry-" + uuid4().hex, uuid4().hex
    runtime = ExecutionCustody(ProcessRuntimeExecutionFence("original-pre-entry-host"))
    committed, release, executor_entered = asyncio.Event(), asyncio.Event(), []
    captured, notifications, receipts = [], [], []
    native_exit = asyncpg.transaction.Transaction.__aexit__
    marking = []
    tick, occurrences, locked = None, [], False

    class Prepared:
        @asynccontextmanager
        async def prepare_scheduled(self, execution):
            executor_entered.append(True)

            async def dispatch():
                await db.execute("INSERT INTO effects VALUES (67, 'must-not-enter')")

            yield dispatch

    await _activate_protocol_for_test_agent(db, did, monkeypatch)
    runner = SchedulerRunner(
        db, did, Prepared(), lease_seconds=2, on_protocol_failure=notifications.append
    )
    await runner._ensure_tables()
    await _seed_due(db, schedule, "original-entry-expiry", agent_id=did)
    original_marker = runner._mark_effect_started

    async def marker(*args, **kwargs):
        state = scheduler_module._rollout_renewal_state.get()
        captured.append(state)
        marking.append(asyncio.current_task().get_name())
        return await original_marker(*args, **kwargs)

    async def held_original_ack(transaction, error_type, error, tb):
        result = await native_exit(transaction, error_type, error, tb)
        if (
            marking
            and not committed.is_set()
            and error_type is None
            and not transaction._nested
            and asyncio.current_task().get_name()
            == marking[0] + ":postgres-original-transaction-exit"
        ):
            committed.set()
            await release.wait()
        return result

    monkeypatch.setattr(runner, "_mark_effect_started", marker)
    monkeypatch.setattr(asyncpg.transaction.Transaction, "__aexit__", held_original_ack)
    try:
        with bind_execution_custody_snapshot((runtime,)):
            tick = create_owned_task(runner._tick())
        await asyncio.wait_for(committed.wait(), 5)
        await await_owned_task(tick)
        occurrences = [entry.task for entry in runner._occurrences.values()]
        assert len(occurrences) == len(captured) == 1
        assert (
            await observer.fetchval(
                f'SELECT status FROM "{schema}".task_execution_log WHERE task_id=$1',
                schedule,
            )
            == "executing"
        )
        await observer.execute("BEGIN")
        locked = True
        await observer.fetchval(
            f'SELECT id FROM "{schema}".scheduled_tasks WHERE id=$1 FOR UPDATE',
            schedule,
        )
        await asyncio.wait_for(captured[0].lost.wait(), 5)
        release.set()
        await observer.execute("ROLLBACK")
        locked = False
        for task in occurrences:
            await asyncio.wait_for(await_owned_task(task), 5)
        await asyncio.sleep(0)
        receipts = runner._occurrence_control_receipts
        assert receipts and execution_denial_scopes(receipts[0])
        assert execution_commit_outcome(receipts[0]) is None
        assert runner.readiness_failure is None and notifications == [], (
            "original deadline entry denial became a shared host failure before scope yield"
        )
        assert executor_entered == []
        assert await observer.fetchval(f'SELECT COUNT(*) FROM "{schema}".effects') == 0
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
