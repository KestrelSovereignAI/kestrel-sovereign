"""Drive one scheduler poll and wait for the occurrences it admitted.

Since #3465 ``SchedulerRunner._tick`` only admits due occurrences onto
runner-owned tasks and returns; the poll loop never waits for them. A test
that asserts on an occurrence's outcome therefore waits for it explicitly.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta, timezone
from uuid import uuid4
from typing import Any


async def settle_occurrences(runner: Any) -> None:
    """Wait until no occurrence the runner admitted is still in flight.

    Cancelling the waiter cancels those occurrences, as cancelling a tick
    that awaited its batch once did.
    """

    while True:
        pending = [
            occurrence.task
            for occurrence in runner._occurrences.values()
            if not occurrence.task.done()
        ]
        if not pending:
            return
        await asyncio.gather(*pending, return_exceptions=True)


async def tick_and_settle(runner: Any) -> None:
    """Run one poll, then wait for every occurrence still in flight."""

    await runner._tick()
    await settle_occurrences(runner)


async def dispatch_scheduled_tool(feature: Any, task_name: str, arguments: dict) -> Any:
    """Run an explicit test intent under a real, revocable SQLite occurrence.

    Retain the target's exact result/error for component assertions; the real
    Runner still owns claims, leases, result recording and physical retirement.
    """
    from kestrel_sovereign.features.scheduler.runner import SCHEDULER_PROTOCOL_VERSION, SchedulerRunner

    observed = {}

    async def execute(name, supplied):
        try:
            observed['result'] = await feature._lookup_and_run_tool(name, supplied)
            return observed['result']
        except BaseException as error:
            observed['error'] = error
            raise

    owner = feature.agent
    database = owner.tool_effect_store.database
    runner = SchedulerRunner(database, owner.did, execute, owner_id='owned-component-' + uuid4().hex)
    try:
        await runner._ensure_tables()
        identity = uuid4().hex
        due = (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat()
        await database.execute('''INSERT INTO scheduled_tasks
            (id,agent_id,task_name,cron_expression,args_json,enabled,next_run_at,
             created_at,schedule_kind,run_at,timezone_name,misfire_policy,
             idempotency_key,scheduler_protocol_version)
            VALUES (?,?,?,?,?,1,?,?,'one_shot',?,'UTC','skip',?,?)''',
            (identity,owner.did,task_name,'',json.dumps(arguments),due,due,due,
             identity,SCHEDULER_PROTOCOL_VERSION))
        await tick_and_settle(runner)
        if 'error' in observed:
            raise observed['error']
        assert 'result' in observed, 'The native scheduler did not dispatch its admitted occurrence'
        return observed['result']
    finally:
        await runner.stop()
