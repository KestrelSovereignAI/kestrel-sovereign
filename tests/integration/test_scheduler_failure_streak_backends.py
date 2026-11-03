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
    hook_failure_backoff,
)
from kestrel_sovereign.features.scheduler.feature import SchedulerFeature
from kestrel_sovereign.features.scheduler.outcome import (
    PAUSE_CAUSE_COLUMNS,
    ScheduledTaskOutcome,
)
from kestrel_sovereign.features.scheduler.runner import (
    SCHEDULER_PROTOCOL_VERSION,
    SchedulerRunner,
)
from kestrel_sovereign.features.scheduler.status import scheduler_status
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


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_hook_failures_back_off_without_pausing_on_backend(
    db_backend, monkeypatch
):
    """#3749: a ``hook_failed`` tick keeps the row enabled, extends the streak,
    and holds the next attempt back by the backoff read from the database
    clock, on PostgreSQL as on SQLite."""

    db = AsyncDatabase(db_backend)
    agent_id = f"scheduler-hook-failure:{uuid4()}"
    task_id = f"scheduler-hook-failure-task:{uuid4()}"

    async def executor(task_name, _args):
        return ScheduledTaskOutcome.hook_failed(
            task_name=task_name,
            reason="Enforcing hook 'security_guard' failed (raised "
            "OperationalError: database is locked); denying by fail-closed "
            "policy.",
        )

    async def fire():
        due = (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat()
        await db.execute(
            "UPDATE scheduled_tasks SET next_run_at = ? WHERE id = ?",
            (due, task_id),
        )
        await tick_and_settle(runner)

    async def retry_delay() -> timedelta:
        next_run_at = await db.fetchval(
            "SELECT next_run_at FROM scheduled_tasks WHERE id = ?", (task_id,)
        )
        completed_at = await db.fetchval(
            "SELECT completed_at FROM task_execution_log WHERE task_id = ? "
            "ORDER BY completed_at DESC LIMIT 1",
            (task_id,),
        )
        return datetime.fromisoformat(next_run_at) - datetime.fromisoformat(
            completed_at
        )

    agent = SimpleNamespace(did=agent_id)
    feature = SchedulerFeature(agent)
    feature._db = db
    feature._agent_id = agent_id
    agent.features = {"SchedulerFeature": feature}
    try:
        await _activate_protocol_for_test_agent(db, agent_id, monkeypatch)
        runner = SchedulerRunner(
            db, agent_id, executor, owner_id="hook-failure-replica"
        )
        await runner._ensure_tables()
        now = datetime.now(timezone.utc).isoformat()
        await db.execute(
            """
            INSERT INTO scheduled_tasks
                (id, agent_id, task_name, cron_expression, args_json, enabled,
                 next_run_at, created_at, idempotency_key,
                 scheduler_protocol_version)
            VALUES (?, ?, 'restart_coordinator', '* * * * *', '{}', 1, ?, ?,
                    'hook-failure-effect', ?)
            """,
            (task_id, agent_id, now, now, SCHEDULER_PROTOCOL_VERSION),
        )

        for failures in (1, 2, 3):
            await fire()
            expected = hook_failure_backoff(failures)
            delay = await retry_delay()
            assert expected - timedelta(seconds=1) <= delay <= expected + timedelta(seconds=1)

        row = await db.fetchone(
            "SELECT enabled, terminal_status FROM scheduled_tasks WHERE id = ?",
            (task_id,),
        )
        assert (int(row[0]), row[1]) == (1, None)
        streak = FailureStreak.from_row(
            await db.fetchone(
                f"SELECT {', '.join(FAILURE_STREAK_COLUMNS)} "
                "FROM scheduled_tasks WHERE id = ?",
                (task_id,),
            )
        )
        assert streak.count == 3 and streak.flagged_at is not None
        [listed] = (await feature.schedule_list()).data["tasks"]
        assert listed["enabled"] is True
        assert listed["disablement"]["state"] == "enabled"
        assert "database is locked" in listed["failure_marker"]["last_error"]
    finally:
        await db.execute("DELETE FROM task_execution_log WHERE task_id = ?", (task_id,))
        await db.execute("DELETE FROM scheduled_tasks WHERE id = ?", (task_id,))
        await db.execute(
            "DELETE FROM scheduler_protocol_rollout WHERE agent_id = ?", (agent_id,)
        )


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_a_legacy_scheduler_pause_is_labelled_on_backend(
    db_backend, monkeypatch
):
    """#3749: the legacy-pause projection runs in both readers on PostgreSQL
    as on SQLite, and tells a pre-#2739 policy pause from an operator's."""

    db = AsyncDatabase(db_backend)
    agent_id = f"scheduler-legacy-pause:{uuid4()}"
    legacy_id = f"scheduler-legacy-pause-task:{uuid4()}"
    operator_id = f"scheduler-operator-pause-task:{uuid4()}"
    blocked_at = "2026-07-14T12:00:00.123456+00:00"

    async def noop(_task_name, _args):
        return None

    agent = SimpleNamespace(did=agent_id)
    feature = SchedulerFeature(agent)
    feature._db = db
    feature._agent_id = agent_id
    agent.features = {"SchedulerFeature": feature}
    try:
        await _activate_protocol_for_test_agent(db, agent_id, monkeypatch)
        runner = SchedulerRunner(db, agent_id, noop, owner_id="legacy-pause-replica")
        await runner._ensure_tables()
        now = datetime.now(timezone.utc).isoformat()
        for task_id, last_run_at in (
            (legacy_id, blocked_at),
            (operator_id, "2026-07-13T12:00:00.000000+00:00"),
        ):
            await db.execute(
                """
                INSERT INTO scheduled_tasks
                    (id, agent_id, task_name, cron_expression, args_json,
                     enabled, next_run_at, last_run_at, created_at,
                     scheduler_protocol_version)
                VALUES (?, ?, 'restart_coordinator', '* * * * *', '{}', 0,
                        ?, ?, ?, ?)
                """,
                (task_id, agent_id, now, last_run_at, now,
                 SCHEDULER_PROTOCOL_VERSION),
            )
            # A #2457-era blocked execution: no claimed_at.
            await db.execute(
                "INSERT INTO task_execution_log (id, task_id, agent_id, status, "
                "result_text, duration_ms, executed_at) "
                "VALUES (?, ?, ?, 'blocked', 'blocked: ...', 3, ?)",
                (f"{task_id}:block", task_id, agent_id, blocked_at),
            )

        listed = {
            task["id"]: task["disablement"]
            for task in (await feature.schedule_list()).data["tasks"]
        }
        assert (listed[legacy_id]["source"], listed[legacy_id]["reason"]) == (
            "scheduler", "blocked",
        )
        assert (listed[operator_id]["source"], listed[operator_id]["reason"]) == (
            "operator", "operator_paused",
        )
        status = await scheduler_status(db, agent_id=agent_id, poll_interval=30)
        assert status["disabled_reasons"] == {"blocked": 1, "operator_paused": 1}
        assert status["system_disabled_count"] == 1
    finally:
        for task_id in (legacy_id, operator_id):
            await db.execute(
                "DELETE FROM task_execution_log WHERE task_id = ?", (task_id,)
            )
            await db.execute("DELETE FROM scheduled_tasks WHERE id = ?", (task_id,))
        await db.execute(
            "DELETE FROM scheduler_protocol_rollout WHERE agent_id = ?", (agent_id,)
        )


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_a_pause_cause_and_its_supersession_on_backend(db_backend, monkeypatch):
    """#3873: the finalize stores a permission pause's structured cause, and a
    later registration under the same idempotency key makes the pause history,
    found through the composite index, on PostgreSQL as on SQLite."""

    db = AsyncDatabase(db_backend)
    agent_id = f"scheduler-pause-cause:{uuid4()}"
    task_id = f"scheduler-pause-cause-task:{uuid4()}"
    replacement_id = f"scheduler-pause-cause-replacement:{uuid4()}"
    idempotency_key = f"pause-cause-effect:{uuid4()}"

    async def executor(task_name, _args):
        return ScheduledTaskOutcome.blocked(
            task_name=task_name,
            decision="deny",
            hook="security_guard",
        )

    agent = SimpleNamespace(did=agent_id)
    feature = SchedulerFeature(agent)
    feature._db = db
    feature._agent_id = agent_id
    agent.features = {"SchedulerFeature": feature}
    try:
        await _activate_protocol_for_test_agent(db, agent_id, monkeypatch)
        runner = SchedulerRunner(db, agent_id, executor, owner_id="pause-cause-replica")
        await runner._ensure_tables()
        now = datetime.now(timezone.utc)
        await db.execute(
            """
            INSERT INTO scheduled_tasks
                (id, agent_id, task_name, cron_expression, args_json, enabled,
                 next_run_at, created_at, idempotency_key,
                 scheduler_protocol_version)
            VALUES (?, ?, 'workflow_trigger_cron', '* * * * *', '{}', 1, ?, ?, ?, ?)
            """,
            (
                task_id, agent_id, (now - timedelta(seconds=1)).isoformat(),
                now.isoformat(), idempotency_key, SCHEDULER_PROTOCOL_VERSION,
            ),
        )
        await tick_and_settle(runner)

        row = await db.fetchone(
            f"SELECT enabled, terminal_status, {', '.join(PAUSE_CAUSE_COLUMNS)} "
            "FROM scheduled_tasks WHERE id = ?",
            (task_id,),
        )
        assert tuple(row) == (
            0, "blocked", "security_guard", "deny", "workflow_trigger_cron",
        )
        status = await scheduler_status(db, agent_id=agent_id, poll_interval=30)
        assert status["system_disabled_count"] == 1
        [named] = status["system_disabled_schedules"]
        assert named["cause"] == {
            "hook": "security_guard", "decision": "deny",
            "tool": "workflow_trigger_cron",
        }
        # The execution's persisted text is worded from the cause alone.
        [(result_text,)] = await db.fetchall(
            "SELECT result_text FROM task_execution_log WHERE task_id = ?",
            (task_id,),
        )
        assert result_text == (
            "blocked: the permission gate stopped this tick (hook security_guard "
            "decided deny for tool workflow_trigger_cron). Set this tool's "
            f"permission to Auto, then resume the schedule. Schedule id: {task_id}."
        )

        await db.execute(
            """
            INSERT INTO scheduled_tasks
                (id, agent_id, task_name, cron_expression, args_json, enabled,
                 next_run_at, created_at, idempotency_key,
                 scheduler_protocol_version)
            VALUES (?, ?, 'workflow_trigger_cron', '* * * * *', '{}', 1, ?, ?, ?, ?)
            """,
            (
                replacement_id, agent_id, "2999-01-01T00:00:00+00:00",
                (now + timedelta(seconds=1)).isoformat(), idempotency_key,
                SCHEDULER_PROTOCOL_VERSION,
            ),
        )
        status = await scheduler_status(db, agent_id=agent_id, poll_interval=30)
        assert status["system_disabled_count"] == 0
        assert status["superseded_count"] == 1
        [superseded] = status["superseded_schedules"]
        assert (superseded["task_id"], superseded["superseded_by"]) == (
            task_id, replacement_id,
        )
    finally:
        for schedule in (task_id, replacement_id):
            await db.execute(
                "DELETE FROM task_execution_log WHERE task_id = ?", (schedule,)
            )
            await db.execute("DELETE FROM scheduled_tasks WHERE id = ?", (schedule,))
        await db.execute(
            "DELETE FROM scheduler_protocol_rollout WHERE agent_id = ?", (agent_id,)
        )
