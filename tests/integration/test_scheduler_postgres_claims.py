"""Scheduler claim race on the production PostgreSQL backend.

The unit suite proves the same predicate with independent SQLite connections.
This integration test runs under ``db_backend`` in CI where PostgreSQL is
available, so two runner transactions use separate PostgreSQL connections and
the database re-evaluates the claim predicate under its row lock.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pytest

from kestrel_sovereign.features.scheduler.runner import (
    SchedulerRunner,
    get_current_scheduler_execution,
)
from kestrel_sovereign.storage.async_database import AsyncDatabase


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_two_replicas_claim_one_due_occurrence_on_backend(db_backend):
    """Exactly one runner dispatches one occurrence, including on Postgres."""
    db = AsyncDatabase(db_backend)
    agent_id = f"scheduler-claim:{uuid4()}"
    task_id = f"scheduler-task:{uuid4()}"
    seen = []

    async def executor(task_name, args):
        seen.append(get_current_scheduler_execution())
        # Leave enough overlap for both transactions to contend on the same
        # occurrence instead of merely polling serially.
        await asyncio.sleep(0.02)
        return "ok"

    first = SchedulerRunner(db, agent_id, executor, owner_id="replica-a")
    second = SchedulerRunner(db, agent_id, executor, owner_id="replica-b")
    try:
        await first._ensure_tables()
        due = (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat()
        await db.execute(
            """
            INSERT INTO scheduled_tasks
                (id, agent_id, task_name, cron_expression, args_json, enabled,
                 next_run_at, created_at, idempotency_key)
            VALUES (?, ?, 'task', '* * * * *', '{}', 1, ?, ?, 'integration-effect')
            """,
            (task_id, agent_id, due, due),
        )
        await asyncio.gather(first._tick(), second._tick())

        assert len(seen) == 1
        rows = await db.fetchall(
            "SELECT status, attempt_count FROM task_execution_log WHERE task_id = ?",
            (task_id,),
        )
        assert rows == [("success", 1)]
    finally:
        await db.execute("DELETE FROM task_execution_log WHERE task_id = ?", (task_id,))
        await db.execute("DELETE FROM scheduled_tasks WHERE id = ?", (task_id,))
