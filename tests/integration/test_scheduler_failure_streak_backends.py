"""Repeated scheduled-task failure on both production backends (#3280).

The unit suite drives the failure streak over SQLite. This runs the same
finalize, ``schedule_list`` and health reads on PostgreSQL too, where the
claim lock, the parameter types and the correlated last-error read are each
the database's own.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from uuid import uuid4

import pytest

from kestrel_sovereign.features.health.checks import (
    check_scheduled_task_failures,
)
from kestrel_sovereign.features.scheduler.failure_streak import (
    FAILURE_STREAK_COLUMNS,
    FailureStreak,
)
from kestrel_sovereign.features.scheduler.feature import SchedulerFeature
from kestrel_sovereign.features.scheduler.runner import (
    SCHEDULER_PROTOCOL_VERSION,
    SchedulerRunner,
)
from kestrel_sovereign.storage.async_database import AsyncDatabase
from tests.integration.test_scheduler_postgres_claims import (
    _activate_protocol_for_test_agent,
)
from tests.utils.scheduler_ticks import tick_and_settle


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_failure_streak_marks_and_clears_on_backend(db_backend, monkeypatch):
    db = AsyncDatabase(db_backend)
    agent_id = f"scheduler-failure-streak:{uuid4()}"
    task_id = f"scheduler-failure-streak-task:{uuid4()}"
    outcomes = ["fail", "fail", "fail", "ok"]

    async def executor(_task_name, _args):
        if outcomes.pop(0) == "fail":
            raise RuntimeError("workflow definition not found: fleet_coding_pipeline")
        return "ok"

    async def fire():
        due = (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat()
        await db.execute(
            "UPDATE scheduled_tasks SET next_run_at = ? WHERE id = ?",
            (due, task_id),
        )
        await tick_and_settle(runner)

    async def streak() -> FailureStreak:
        return FailureStreak.from_row(
            await db.fetchone(
                f"SELECT {', '.join(FAILURE_STREAK_COLUMNS)} "
                "FROM scheduled_tasks WHERE id = ?",
                (task_id,),
            )
        )

    agent = SimpleNamespace(did=agent_id)
    feature = SchedulerFeature(agent)
    feature._db = db
    feature._agent_id = agent_id
    agent.features = {"SchedulerFeature": feature}
    try:
        await _activate_protocol_for_test_agent(db, agent_id, monkeypatch)
        runner = SchedulerRunner(
            db, agent_id, executor, owner_id="failure-streak-replica"
        )
        await runner._ensure_tables()
        now = datetime.now(timezone.utc).isoformat()
        await db.execute(
            """
            INSERT INTO scheduled_tasks
                (id, agent_id, task_name, cron_expression, args_json, enabled,
                 next_run_at, created_at, idempotency_key,
                 scheduler_protocol_version)
            VALUES (?, ?, 'signal_dispatch', '5 8 * * *', '{}', 1, ?, ?,
                    'failure-streak-effect', ?)
            """,
            (task_id, agent_id, now, now, SCHEDULER_PROTOCOL_VERSION),
        )

        for _ in range(2):
            await fire()
        assert (await streak()).count == 2
        assert (await streak()).flagged_at is None

        await fire()
        flagged = await streak()
        assert flagged.count == 3
        assert flagged.flagged_at == flagged.last_at
        assert flagged.first_at < flagged.last_at

        [listed] = (await feature.schedule_list()).data["tasks"]
        assert listed["failure_marker"]["consecutive_failures"] == 3
        assert listed["failure_marker"]["last_error"] == (
            "RuntimeError: workflow definition not found: fleet_coding_pipeline"
        )
        check = await check_scheduled_task_failures(agent, db)
        assert check["status"] == "warn"
        assert [m["task_id"] for m in check["details"]["failing"]] == [task_id]

        await fire()
        assert await streak() == FailureStreak()
        check = await check_scheduled_task_failures(agent, db)
        assert check["status"] == "pass"
    finally:
        await db.execute("DELETE FROM task_execution_log WHERE task_id = ?", (task_id,))
        await db.execute("DELETE FROM scheduled_tasks WHERE id = ?", (task_id,))
        await db.execute(
            "DELETE FROM scheduler_protocol_rollout WHERE agent_id = ?", (agent_id,)
        )
