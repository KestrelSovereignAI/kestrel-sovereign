"""Durability contracts for scheduler claims, leases, deadlines, and zones."""

import asyncio
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import ANY, AsyncMock

import pytest
from fastapi import FastAPI

from kestrel_sovereign import server
from kestrel_sovereign.features.scheduler.cron import next_run
from kestrel_sovereign.features.scheduler.runner import (
    AgentManagerHostedSchedulerExecutor,
    SchedulerExecution,
    SchedulerRunner,
    get_current_scheduler_execution,
)
from kestrel_sovereign.storage.async_database import AsyncDatabase
from kestrel_sovereign.storage.db.sqlite import SQLiteBackend


async def _database(path):
    backend = SQLiteBackend(str(path))
    await backend.connect()
    return AsyncDatabase(backend)


async def _seed_due(db, *, task_id="task-1", agent_id="agent-1", kind="cron", policy="skip"):
    due = (datetime.now(timezone.utc) - timedelta(seconds=2)).isoformat()
    await db.execute(
        """
        INSERT INTO scheduled_tasks
            (id, agent_id, task_name, cron_expression, args_json, enabled,
             next_run_at, created_at, schedule_kind, run_at, timezone_name,
             misfire_policy, idempotency_key)
        VALUES (?, ?, 'test_task', '* * * * *', '{}', 1, ?, ?, ?, ?, 'UTC', ?, 'stable-effect')
        """,
        (task_id, agent_id, due, due, kind, due if kind == "one_shot" else None, policy),
    )
    return due


@pytest.mark.asyncio
async def test_two_database_runners_claim_one_occurrence(tmp_path):
    """The compare-and-set claim is safe across independent DB connections.

    SQLite is used for a deterministic local concurrency test; the production
    PostgreSQL path uses exactly the same atomic ``UPDATE ... WHERE lease``
    predicate and transaction boundary.
    """
    db_a = await _database(tmp_path / "scheduler.db")
    db_b = await _database(tmp_path / "scheduler.db")
    calls = []

    async def executor(name, args):
        execution = get_current_scheduler_execution()
        calls.append((name, execution.id, execution.idempotency_key))
        await asyncio.sleep(0.02)
        return "done"

    runner_a = SchedulerRunner(db_a, "agent-1", executor, owner_id="replica-a")
    runner_b = SchedulerRunner(db_b, "agent-1", executor, owner_id="replica-b")
    try:
        await runner_a._ensure_tables()
        await _seed_due(db_a)
        await asyncio.gather(runner_a._tick(), runner_b._tick())

        assert len(calls) == 1
        history = await db_a.fetchall(
            "SELECT status, attempt_count, idempotency_key FROM task_execution_log WHERE task_id = ?",
            ("task-1",),
        )
        assert history == [("success", 1, calls[0][2])]
    finally:
        await db_a.close()
        await db_b.close()


@pytest.mark.asyncio
async def test_expired_lease_recovers_the_same_execution_identity(tmp_path):
    """Death before dispatch is recoverable without creating a new effect key."""
    db = await _database(tmp_path / "scheduler.db")
    seen = []

    async def executor(name, args):
        seen.append(get_current_scheduler_execution())
        return "recovered"

    runner = SchedulerRunner(db, "agent-1", executor, owner_id="recovery")
    try:
        await runner._ensure_tables()
        due = await _seed_due(db)
        expired = (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat()
        await db.execute(
            """
            UPDATE scheduled_tasks
            SET lease_owner = 'dead', lease_expires_at = ?, claim_token = 'old-token',
                claim_execution_id = 'execution-1', claim_scheduled_for = ?, attempt_count = 1
            WHERE id = ?
            """,
            (expired, due, "task-1"),
        )
        await db.execute(
            """
            INSERT INTO task_execution_log
                (id, task_id, agent_id, status, result_text, duration_ms,
                 executed_at, occurrence_at, idempotency_key, attempt_count, claimed_at)
            VALUES ('execution-1', 'task-1', 'agent-1', 'claimed', NULL, 0, ?, ?, ?, 1, ?)
            """,
            (
                due,
                due,
                SchedulerRunner._occurrence_idempotency_key("stable-effect", "task-1", due),
                due,
            ),
        )

        await runner._tick()
        assert len(seen) == 1
        assert seen[0].id == "execution-1"
        assert seen[0].attempt == 2
        assert seen[0].idempotency_key == SchedulerRunner._occurrence_idempotency_key(
            "stable-effect", "task-1", due
        )
        log = await db.fetchone(
            "SELECT status, attempt_count FROM task_execution_log WHERE id = 'execution-1'"
        )
        assert log == ("success", 2)
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_crash_before_outcome_commit_retries_with_same_idempotency_key(tmp_path):
    """Failure injection for a crash after dispatch but before outcome CAS.

    A target may observe delivery twice after this crash window, but both
    deliveries carry the same key and execution ID, so its effect boundary can
    enforce exactly-once behavior.
    """
    db = await _database(tmp_path / "scheduler.db")
    delivered = []

    async def executor(name, args):
        delivered.append(get_current_scheduler_execution())
        return "external effect sent"

    class CrashBeforeCommitRunner(SchedulerRunner):
        async def _finalize(self, *args, **kwargs):
            raise asyncio.CancelledError()

    crashing = CrashBeforeCommitRunner(db, "agent-1", executor, owner_id="dead-replica", lease_seconds=1)
    recovering = SchedulerRunner(db, "agent-1", executor, owner_id="recovery", lease_seconds=1)
    try:
        await crashing._ensure_tables()
        await _seed_due(db)
        await crashing._tick()
        row = await db.fetchone(
            "SELECT claim_execution_id, lease_expires_at FROM scheduled_tasks WHERE id = ?",
            ("task-1",),
        )
        assert row[0] == delivered[0].id
        # Simulate the crashed replica's lease elapsing without waiting.
        await db.execute(
            "UPDATE scheduled_tasks SET lease_expires_at = ? WHERE id = ?",
            ((datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat(), "task-1"),
        )
        await recovering._tick()

        assert len(delivered) == 2
        assert delivered[0].id == delivered[1].id
        assert delivered[0].idempotency_key == delivered[1].idempotency_key
        log = await db.fetchone("SELECT status, attempt_count FROM task_execution_log WHERE id = ?", (delivered[0].id,))
        assert log == ("success", 2)
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_death_during_dispatch_releases_to_recovery_with_same_key(tmp_path):
    """Failure injection for a replica disappearing while target dispatch runs."""
    db = await _database(tmp_path / "scheduler.db")
    delivered = []

    async def dies_during_dispatch(name, args):
        delivered.append(get_current_scheduler_execution())
        raise asyncio.CancelledError()

    async def recovered_dispatch(name, args):
        delivered.append(get_current_scheduler_execution())
        return "recovered"

    dying = SchedulerRunner(db, "agent-1", dies_during_dispatch, owner_id="dying", lease_seconds=1)
    recovery = SchedulerRunner(db, "agent-1", recovered_dispatch, owner_id="recovery", lease_seconds=1)
    try:
        await dying._ensure_tables()
        await _seed_due(db)
        await dying._tick()
        await db.execute(
            "UPDATE scheduled_tasks SET lease_expires_at = ? WHERE id = ?",
            ((datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat(), "task-1"),
        )
        await recovery._tick()

        assert len(delivered) == 2
        assert delivered[0].id == delivered[1].id
        assert delivered[0].idempotency_key == delivered[1].idempotency_key
        assert (await db.fetchone("SELECT status, attempt_count FROM task_execution_log WHERE id = ?", (delivered[0].id,))) == ("success", 2)
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_one_shot_deadline_becomes_terminal_after_one_fire(tmp_path):
    db = await _database(tmp_path / "scheduler.db")
    executor = AsyncMock(return_value="deadline met")
    runner = SchedulerRunner(db, "agent-1", executor)
    try:
        await runner._ensure_tables()
        await _seed_due(db, kind="one_shot", policy="fire_once")
        await runner._tick()
        await runner._tick()

        executor.assert_awaited_once_with("test_task", {})
        row = await db.fetchone(
            "SELECT enabled, next_run_at, terminal_status FROM scheduled_tasks WHERE id = ?",
            ("task-1",),
        )
        assert row == (0, None, "success")
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_fire_once_misfire_policy_executes_one_late_occurrence(tmp_path):
    db = await _database(tmp_path / "scheduler.db")
    executor = AsyncMock(return_value="caught up once")
    runner = SchedulerRunner(db, "agent-1", executor, misfire_grace_seconds=1)
    try:
        await runner._ensure_tables()
        await _seed_due(db, policy="fire_once")
        await db.execute(
            "UPDATE scheduled_tasks SET next_run_at = ? WHERE id = ?",
            ((datetime.now(timezone.utc) - timedelta(hours=1)).isoformat(), "task-1"),
        )
        await runner._tick()

        executor.assert_awaited_once_with("test_task", {})
        row = await db.fetchone(
            "SELECT enabled, next_run_at, terminal_status FROM scheduled_tasks WHERE id = ?",
            ("task-1",),
        )
        assert row[0] == 1
        assert row[1] > datetime.now(timezone.utc).isoformat()
        assert row[2] is None
    finally:
        await db.close()


def test_local_cron_dst_gap_and_fold_contract():
    chicago = "America/Chicago"
    # 02:30 does not exist on the 2026 spring-forward Sunday, so the next run
    # is Monday rather than an invented/shifted Sunday instant.
    gap = next_run("30 2 * * *", datetime(2026, 3, 8, 6, 0, tzinfo=timezone.utc), chicago)
    assert gap == datetime(2026, 3, 9, 7, 30, tzinfo=timezone.utc)
    # The fall-back 01:30 occurs twice; the scheduler chooses fold=0 (06:30Z)
    # and does not execute it again at 07:30Z.
    fold = next_run("30 1 * * *", datetime(2026, 11, 1, 5, 0, tzinfo=timezone.utc), chicago)
    assert fold == datetime(2026, 11, 1, 6, 30, tzinfo=timezone.utc)
    after_fold = next_run("30 1 * * *", fold, chicago)
    assert after_fold == datetime(2026, 11, 2, 7, 30, tzinfo=timezone.utc)


@pytest.mark.asyncio
async def test_host_executor_loads_cold_agent_once_per_target():
    dispatched = AsyncMock(return_value="woken")
    cold_agent = SimpleNamespace(
        did="agent-1",
        features={"SchedulerFeature": SimpleNamespace(_dispatch_scheduled_task=dispatched)},
    )
    manager = SimpleNamespace(list_agents=lambda: {}, load_agent=AsyncMock(return_value=cold_agent))
    executor = AgentManagerHostedSchedulerExecutor(manager, {"agent-1": ("Cold", object())})
    execution = SchedulerExecution(
        id="execution-1", schedule_id="task-1", agent_id="agent-1", task_name="test_task",
        args={"x": 1}, scheduled_for="2026-07-24T00:00:00+00:00",
        idempotency_key="effect-1", attempt=1, owner="host",
    )

    assert await executor.execute_scheduled(execution) == "woken"
    manager.load_agent.assert_awaited_once()
    dispatched.assert_awaited_once_with("test_task", {"x": 1})


@pytest.mark.asyncio
async def test_structural_host_executor_needs_no_private_marker(tmp_path):
    """Any implementation of the public hosted-executor protocol dispatches.

    The runner must not require the legacy adapter's private marker attribute;
    host integrations are intentionally free to implement the protocol without
    subclassing ``HostedSchedulerExecutor``.
    """
    db = await _database(tmp_path / "scheduler.db")
    executions = []

    class MinimalHostedExecutor:
        async def execute_scheduled(self, execution):
            executions.append(execution)
            return "dispatched"

    runner = SchedulerRunner(db, None, MinimalHostedExecutor())
    try:
        await runner._ensure_tables()
        await _seed_due(db)
        await runner._tick()

        assert len(executions) == 1
        assert executions[0].agent_id == "agent-1"
        assert executions[0].task_name == "test_task"
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_host_lifecycle_runner_claims_and_wakes_a_cold_agent(monkeypatch, tmp_path):
    """The multi-agent lifecycle owns an unscoped runner for cold schedules."""
    db = await _database(tmp_path / "scheduler.db")
    storage_instances = []

    class HostStorage:
        def __init__(self, *, backend, dsn):
            assert backend == "postgres"
            assert dsn == "postgresql://scheduler-test"
            self.db = db
            self.closed = False
            storage_instances.append(self)

        async def initialize(self):
            return None

        async def close(self):
            self.closed = True
            await db.close()

    dispatched = AsyncMock(return_value="woken")
    cold_agent = SimpleNamespace(
        did="agent-1",
        features={"SchedulerFeature": SimpleNamespace(_dispatch_scheduled_task=dispatched)},
    )
    manager = SimpleNamespace(
        local_agent_configs_by_did=AsyncMock(return_value={"agent-1": ("Cold", object())}),
        list_agents=lambda: {},
        load_agent=AsyncMock(return_value=cold_agent),
    )
    app = FastAPI()

    monkeypatch.setenv("KESTREL_DB_BACKEND", "postgres")
    monkeypatch.setenv("KESTREL_DATABASE_URL", "postgresql://scheduler-test")
    monkeypatch.setattr(
        "kestrel_sovereign.storage.async_storage.AsyncStorage", HostStorage
    )

    try:
        await server._start_host_scheduler(app, manager, object())
        runner = app.state.host_scheduler_runner
        assert runner is not None
        # Stop the background loop so this test alone drives the due occurrence.
        await runner.stop()
        await _seed_due(db)

        await runner._tick()

        manager.load_agent.assert_awaited_once_with("Cold", ANY)
        dispatched.assert_awaited_once_with("test_task", {})
        assert storage_instances[0].closed is False
    finally:
        await server._shutdown_host_scheduler(app)
