"""Retained owner liveness and scheduler occurrence evidence survive denial."""

import asyncio
from datetime import datetime, timedelta, timezone

import pytest

from kestrel_sovereign.execution_custody import (
    ExecutionAuthorityError, ExecutionCustody, bind_execution_custody,
)
from kestrel_sovereign.features.scheduler.runner import (
    SCHEDULER_PROTOCOL_VERSION, ScheduledTask, SchedulerRunner,
    ScheduledTaskOwnerUnavailable, SchedulerDispatchNotReady, SchedulerFeatureUnavailable,
)
from kestrel_sovereign.kestrel_agent import KestrelAgent
from kestrel_sovereign.signals import (
    OrderedLockManager, SignalDispatcher, SignalLogStore, SourceRegistry,
)
from kestrel_sovereign.storage.async_database import AsyncDatabase
from tests.integration.test_execution_authority_postgres import (
    GenerationFence, native_pg as _native_pg,
)
from tests.integration.test_execution_custody_review13_pg import _seed_owned_deliveries
from tests.unit.test_durable_signal_delivery import _Agent

pytestmark = [pytest.mark.asyncio, pytest.mark.integration]
native_pg = _native_pg


@pytest.mark.parametrize("phase", ["direct", "timer"])
async def test_running_retained_cognition_keeps_native_heartbeat_without_shutdown_or_debt(native_pg, phase):
    backend, controller, schema = native_pg
    await _seed_owned_deliveries(backend, controller, schema)
    class Agent(_Agent):
        _execution_custody = None
        _runtime_owner_context = KestrelAgent._runtime_owner_context
    agent = Agent("original")
    dispatcher = SignalDispatcher(agent=agent, registry=SourceRegistry(),
                                  lock_manager=OrderedLockManager(), store=SignalLogStore(backend))
    dispatcher._durable_delivery_owner = "dispatcher:original"
    await dispatcher.initialize_durable_delivery()
    await dispatcher._stop_runtime_owner_heartbeat()
    retained = asyncio.create_task(asyncio.Event().wait())
    dispatcher._retained_durable_cognition_tasks.add(retained)
    scope = ExecutionCustody(GenerationFence())
    agent._execution_custody = backend._execution_custody = scope
    scope.revoke("original runtime lost")
    before = await controller.fetchrow("SELECT * FROM durable_signal_deliveries WHERE delivery_id='delivery1'")
    try:
        assert not dispatcher._durable_shutdown
        assert not dispatcher._retained_cognition_control_debt
        assert not dispatcher.durable_shutdown_owner_fenced
        if phase == "direct":
            await dispatcher._heartbeat_runtime_owner()
        first = await controller.fetchval("SELECT heartbeat_at FROM durable_signal_runtime_owners WHERE owner_id='dispatcher:original'")
        dispatcher._runtime_owner_stale_after = timedelta(seconds=0.03)
        dispatcher._schedule_runtime_owner_heartbeat(retry=True)
        assert dispatcher._runtime_owner_heartbeat_timer is not None
        async with asyncio.timeout(3):
            while await controller.fetchval(
                "SELECT heartbeat_at FROM durable_signal_runtime_owners WHERE owner_id='dispatcher:original'"
            ) <= first:
                await asyncio.sleep(0.01)
        assert not retained.done()
        assert await controller.fetchrow("SELECT * FROM durable_signal_deliveries WHERE delivery_id='delivery1'") == before
        assert await controller.fetchval("SELECT heartbeat_at < NOW()-INTERVAL '30 minutes' FROM durable_signal_runtime_owners WHERE owner_id='dispatcher:successor'")
        assert not dispatcher._retained_cognition_control_debt
        assert not dispatcher._durable_shutdown
    finally:
        await dispatcher._stop_runtime_owner_heartbeat()
        retained.cancel()
        await asyncio.gather(retained, return_exceptions=True)
        dispatcher._retained_durable_cognition_tasks.discard(retained)
        for task in agent.tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*agent.tasks, return_exceptions=True)


@pytest.mark.parametrize("carrier", ["direct", "wrapped", "owner", "not-ready", "feature"])
async def test_scheduler_child_denial_after_committed_effect_keeps_original_occurrence(native_pg, carrier):
    backend, _controller, _schema = native_pg
    db = AsyncDatabase(backend)
    agent_id = "did:example:review17"
    calls = []
    async def effect(_name, _args):
        calls.append(True)
        await backend.execute("INSERT INTO effects VALUES (1, 'already committed')")
        # The narrower child ends before the outer scheduler catches its
        # exception. The still-live scheduler must not terminalize it as failed.
        try:
            with bind_execution_custody(GenerationFence()) as child:
                child.revoke("child authority lost after committed effect")
                child.require_work()
        except ExecutionAuthorityError as control:
            if carrier == "direct":
                raise
            error = (
                SchedulerDispatchNotReady(agent_id, "effect")
                if carrier == "not-ready"
                else {
                    "wrapped": RuntimeError, "owner": ScheduledTaskOwnerUnavailable,
                    "feature": SchedulerFeatureUnavailable,
                }[carrier]("cause-carried child denial")
            )
            raise error from control

    runner = SchedulerRunner(db, agent_id, effect, owner_id="original-review17")
    await runner._ensure_tables()
    due = (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat()
    await db.execute("""INSERT INTO scheduled_tasks
        (id,agent_id,task_name,cron_expression,enabled,next_run_at,created_at,idempotency_key,scheduler_protocol_version)
        VALUES ('child-denial',?,'effect','* * * * *',1,?,?,'review17-base',?)""",
                     (agent_id, due, due, SCHEDULER_PROTOCOL_VERSION))
    task = ScheduledTask.from_row((await runner._due_rows(datetime.now(timezone.utc)))[0])
    claimed = await runner._claim(task, datetime.now(timezone.utc))
    assert claimed is not None
    identity = await db.fetchone("SELECT claim_execution_id,claim_scheduled_for FROM scheduled_tasks WHERE id='child-denial'")
    await runner._execute_claim(claimed)
    assert calls == [True]
    assert await db.fetchval("SELECT status FROM task_execution_log WHERE task_id='child-denial'") == "executing"
    assert await db.fetchone("SELECT claim_execution_id,claim_scheduled_for FROM scheduled_tasks WHERE id='child-denial'") == identity
    assert await db.fetchval("SELECT next_run_at FROM scheduled_tasks WHERE id='child-denial'") == task.next_run_at
    assert await backend.fetch_val("SELECT count(*) FROM effects") == 1
