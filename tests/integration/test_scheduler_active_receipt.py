"""Native confirmed occurrence expiry is not a shared-host protocol outage."""

import asyncio
from uuid import uuid4

import pytest
from fastapi import FastAPI

from kestrel_sovereign._async_ownership import await_owned_task, create_owned_task
from kestrel_sovereign.execution_custody import (
    ExecutionCustody,
    ProcessRuntimeExecutionFence,
    bind_execution_custody_snapshot,
    current_execution_custody,
    execution_cleanup_debts,
    execution_commit_outcome,
    execution_denial_scopes,
    require_execution_work,
)
from kestrel_sovereign.features.scheduler.runner import SchedulerRunner
from kestrel_sovereign.server import _latch_scheduler_readiness_failure
from kestrel_sovereign.storage.async_database import AsyncDatabase
from tests.integration.test_execution_authority_postgres import native_pg as native_pg
from tests.integration.test_scheduler_postgres_claims import (
    _activate_protocol_for_test_agent,
)
from tests.unit.test_scheduler_occurrence_liveness import _seed_due
from tests.utils.scheduler_ticks import tick_and_settle

pytestmark = [pytest.mark.asyncio, pytest.mark.integration]


async def test_historical_original_receipt_cannot_override_different_active_occurrence(
    native_pg, monkeypatch
):
    backend, observer, schema = native_pg
    db = AsyncDatabase(backend)
    did, schedule = "did:web:expiry-" + uuid4().hex, uuid4().hex
    app = FastAPI()
    runtime = ExecutionCustody(ProcessRuntimeExecutionFence("original-host-runtime"))
    entered, cancelled = asyncio.Event(), asyncio.Event()
    receipts, scopes, healthy_calls = [], [], []

    async def executor(name, _):
        if name == "surface-other-original":
            active = current_execution_custody()
            denied = execution_denial_scopes(receipts[0])
            assert denied and all(scope not in active for scope in denied)
            assert await db.fetchval("SELECT COUNT(*) FROM effects") == 0
            raise receipts[0]
        if name != "expired":
            await db.execute("INSERT INTO effects VALUES (71, 'fresh occurrence')")
            healthy_calls.append(name)
            return "ok"
        scopes.extend(current_execution_custody())
        # Cross an actual guarded native read before holding this occurrence.
        assert await db.fetchval("SELECT COUNT(*) FROM effects") == 0
        entered.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.set()
            try:
                require_execution_work()
            except BaseException as original:
                receipts.append(original)
                raise

    def host_failure(error):
        _latch_scheduler_readiness_failure(app, "protocol", error)

    await _activate_protocol_for_test_agent(db, did, monkeypatch)
    runner = SchedulerRunner(
        db,
        did,
        executor,
        lease_seconds=2,
        owner_id="native-expiry-proof",
        on_protocol_failure=host_failure,
    )
    await runner._ensure_tables()
    await _seed_due(db, schedule, "expired", agent_id=did)
    tick, occurrences, locked = None, [], False
    try:
        with bind_execution_custody_snapshot((runtime,)):
            tick = create_owned_task(runner._tick())
        await asyncio.wait_for(entered.wait(), 5)
        await await_owned_task(tick)
        occurrences = [entry.task for entry in runner._occurrences.values()]
        assert len(occurrences) == 1
        await observer.execute("BEGIN")
        locked = True
        # The original persisted row/token stays unchanged while renewal is
        # blocked. The runner's actual confirmed deadline must still expire.
        await observer.fetchval(
            f'SELECT id FROM "{schema}".scheduled_tasks WHERE id=$1 FOR UPDATE',
            schedule,
        )
        await asyncio.wait_for(cancelled.wait(), 5)
        await observer.execute("ROLLBACK")
        locked = False
        for task in occurrences:
            await asyncio.wait_for(await_owned_task(task), 5)
        assert receipts and execution_denial_scopes(receipts[0])
        assert execution_commit_outcome(receipts[0]) is None
        assert not execution_cleanup_debts(receipts[0])
        assert not getattr(app.state, "scheduler_readiness_failures", ())
        assert runner.readiness_failure is None
        runtime.require_work()
        assert any(item is receipts[0] for item in runner._occurrence_control_receipts)
        with pytest.raises(BaseException):
            with bind_execution_custody_snapshot(scopes):
                require_execution_work()
        with bind_execution_custody_snapshot((runtime,)):
            await _seed_due(db, uuid4().hex, "surface-other-original", agent_id=did)
            await tick_and_settle(runner)
        assert runner.readiness_failure is receipts[0], (
            "historical receipt overrode the different active occurrence's original scopes"
        )
        assert getattr(app.state, "scheduler_readiness_failures", ())
        assert await observer.fetchval(f'SELECT COUNT(*) FROM "{schema}".effects') == 0
    finally:
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
            assert execution_denial_scopes(error) is not None
