"""Durability contracts for scheduler claims, leases, deadlines, and zones."""

import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import ANY, AsyncMock

import pytest
from fastapi import FastAPI

from kestrel_sovereign import server
from kestrel_sovereign.features.scheduler import runner as scheduler_runner_module
from kestrel_sovereign.features.scheduler.cron import next_run
from kestrel_sovereign.features.scheduler.feature import SchedulerFeature
from kestrel_sovereign.features.scheduler.runner import (
    AgentManagerHostedSchedulerExecutor,
    HostedSchedulerExecutor,
    SCHEDULER_PROTOCOL_VERSION,
    SCHEDULER_ROLLOUT_ACK_ENV,
    SCHEDULER_ROLLOUT_STATE_ACTIVE,
    SCHEDULER_SCHEMA_PROVENANCE_FRESH_V2,
    SchedulerRolloutQuiescenceRequired,
    SchedulerExecution,
    SchedulerRunner,
    ScheduledTask,
    get_current_scheduler_execution,
)
from kestrel_sovereign.multi_agent.agent_manager import AgentManager
from kestrel_sovereign.multi_agent.config import LocalAgentConfig, MultiAgentConfig
from kestrel_sovereign.storage.async_database import AsyncDatabase
from kestrel_sovereign.storage.db.sqlite import SQLiteBackend


async def _database(path):
    backend = SQLiteBackend(str(path))
    await backend.connect()
    return AsyncDatabase(backend)


async def _seed_due(
    db,
    *,
    task_id="task-1",
    agent_id="agent-1",
    kind="cron",
    policy="skip",
    task_name="test_task",
):
    due = (datetime.now(timezone.utc) - timedelta(seconds=2)).isoformat()
    await db.execute(
        """
        INSERT INTO scheduled_tasks
            (id, agent_id, task_name, cron_expression, args_json, enabled,
             next_run_at, created_at, schedule_kind, run_at, timezone_name,
             misfire_policy, idempotency_key, scheduler_protocol_version)
        VALUES (?, ?, ?, '* * * * *', '{}', 1, ?, ?, ?, ?, 'UTC', ?, 'stable-effect', ?)
        """,
        (
            task_id,
            agent_id,
            task_name,
            due,
            due,
            kind,
            due if kind == "one_shot" else None,
            policy,
            SCHEDULER_PROTOCOL_VERSION,
        ),
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
async def test_file_sqlite_concurrent_bootstrap_serializes_schema_and_all_did_seeding(
    tmp_path,
):
    """Separate file connections cannot observe a partial scheduler bootstrap."""

    database_path = tmp_path / "concurrent-bootstrap.db"
    db_a = await _database(database_path)
    db_b = await _database(database_path)
    first_inside_mutations = asyncio.Event()
    release_first = asyncio.Event()
    second_inside_mutations = asyncio.Event()
    agent_ids = {"agent-a", "agent-b"}

    class FirstBootstrapRunner(SchedulerRunner):
        async def _ensure_tables_mutations(self):
            first_inside_mutations.set()
            await release_first.wait()
            return await super()._ensure_tables_mutations()

    class SecondBootstrapRunner(SchedulerRunner):
        async def _ensure_tables_mutations(self):
            second_inside_mutations.set()
            return await super()._ensure_tables_mutations()

    async def noop(_task_name, _args):
        return None

    first = FirstBootstrapRunner(
        db_a, None, noop, authorized_agent_ids=agent_ids, owner_id="sqlite-first"
    )
    second = SecondBootstrapRunner(
        db_b, None, noop, authorized_agent_ids=agent_ids, owner_id="sqlite-second"
    )
    first_task: asyncio.Task[None] | None = None
    second_task: asyncio.Task[None] | None = None
    try:
        first_task = asyncio.create_task(first._ensure_tables())
        await asyncio.wait_for(first_inside_mutations.wait(), timeout=1)

        second_task = asyncio.create_task(second._ensure_tables())
        await asyncio.sleep(0.05)
        assert not second_inside_mutations.is_set()
        assert not second_task.done()

        release_first.set()
        await asyncio.wait_for(asyncio.gather(first_task, second_task), timeout=2)

        assert await db_a.fetchone(
            "SELECT provenance, protocol_version FROM scheduler_protocol_schema "
            "WHERE singleton = 1"
        ) == (SCHEDULER_SCHEMA_PROVENANCE_FRESH_V2, SCHEDULER_PROTOCOL_VERSION)
        assert await db_a.fetchall(
            "SELECT agent_id, protocol_version, state, activation_nonce "
            "FROM scheduler_protocol_rollout ORDER BY agent_id"
        ) == [
            ("agent-a", SCHEDULER_PROTOCOL_VERSION, SCHEDULER_ROLLOUT_STATE_ACTIVE, None),
            ("agent-b", SCHEDULER_PROTOCOL_VERSION, SCHEDULER_ROLLOUT_STATE_ACTIVE, None),
        ]
        assert first._protocol_ready and second._protocol_ready
    finally:
        release_first.set()
        for owned in (first_task, second_task):
            if owned is not None and not owned.done():
                owned.cancel()
        await asyncio.gather(
            *(owned for owned in (first_task, second_task) if owned is not None),
            return_exceptions=True,
        )
        await db_a.close()
        await db_b.close()


@pytest.mark.asyncio
async def test_file_sqlite_concurrent_builtin_seeders_insert_each_default_once(
    tmp_path,
):
    """Two post-load replicas cannot duplicate core defaults for one DID."""

    database_path = tmp_path / "concurrent-default-seeding.db"
    db_a = await _database(database_path)
    db_b = await _database(database_path)
    agent_id = "did:scheduler:concurrent-defaults"

    def feature_for(db):
        agent = SimpleNamespace(
            did=agent_id,
            agent_id=agent_id,
            features={},
            storage=SimpleNamespace(db=db),
        )
        feature = SchedulerFeature(agent)
        feature._db = db
        feature._agent_id = agent_id
        return feature, agent

    first, first_agent = feature_for(db_a)
    second, second_agent = feature_for(db_b)
    runner = SchedulerRunner(
        db_a,
        agent_id,
        AsyncMock(),
        owner_id="builtin-seed-schema",
    )
    try:
        await runner._ensure_tables()
        await asyncio.gather(
            first.post_all_features_loaded(first_agent),
            second.post_all_features_loaded(second_agent),
        )

        defaults = await db_a.fetchall(
            """
            SELECT task_name, COUNT(*), MIN(idempotency_key)
            FROM scheduled_tasks
            WHERE agent_id = ?
            GROUP BY task_name
            ORDER BY task_name
            """,
            (agent_id,),
        )
        assert defaults == [
            ("backup_snapshot", 1, "scheduler:builtin:v1:backup_snapshot"),
            ("morning_signal", 1, "scheduler:builtin:v1:morning_signal"),
            ("signal_dispatch", 1, "scheduler:builtin:v1:signal_dispatch"),
            ("trash_retention", 1, "scheduler:builtin:v1:trash_retention"),
            ("wait_reconcile", 1, "scheduler:builtin:v1:wait_reconcile"),
        ]

        # The idempotent core ensure is private; the public API intentionally
        # retains duplicate task names for independently keyed user schedules.
        custom = await first.schedule_add(
            cron_expression="@hourly",
            task_name="backup_snapshot",
            idempotency_key="user:second-backup",
        )
        assert custom.status.value == "ok"
        assert await db_a.fetchone(
            """
            SELECT COUNT(*) FROM scheduled_tasks
            WHERE agent_id = ? AND task_name = 'backup_snapshot'
            """,
            (agent_id,),
        ) == (2,)
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
            SET enabled = 0, scheduler_claim_fenced = 1,
                lease_owner = 'dead', lease_expires_at = ?, claim_token = 'old-token',
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
async def test_stale_due_snapshot_recovers_durable_claim_identity_after_row_lock(
    tmp_path,
):
    """Recovery rereads a claim that appeared after the original due scan.

    This is the interleaving a worker sees when it selects an unclaimed due
    row, another replica claims it and dies, and the first worker finally
    acquires the schedule-row lock.  The first worker must retain the durable
    execution log rather than minting a second identity from its stale row.
    """

    db = await _database(tmp_path / "stale-due-recovery.db")
    seen = []

    async def executor(_name, _args):
        seen.append(get_current_scheduler_execution())
        return "recovered"

    runner = SchedulerRunner(db, "agent-1", executor, owner_id="late-worker")
    try:
        await runner._ensure_tables()
        due = await _seed_due(db)
        stale_rows = await runner._due_rows(datetime.now(timezone.utc))
        stale_task = ScheduledTask.from_row(stale_rows[0])
        assert stale_task.claim_execution_id is None

        expired = (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat()
        execution_id = "identity-published-after-due-scan"
        occurrence_key = SchedulerRunner._occurrence_idempotency_key(
            "stable-effect", "task-1", due,
        )
        await db.execute(
            """
            UPDATE scheduled_tasks
            SET enabled = 0, scheduler_claim_fenced = 1,
                lease_owner = 'dead-worker', lease_expires_at = ?,
                claim_token = 'dead-token', claim_execution_id = ?,
                claim_scheduled_for = ?, attempt_count = 1
            WHERE id = ?
            """,
            (expired, execution_id, due, "task-1"),
        )
        await db.execute(
            """
            INSERT INTO task_execution_log
                (id, task_id, agent_id, status, result_text, duration_ms,
                 executed_at, occurrence_at, idempotency_key, attempt_count,
                 claimed_at)
            VALUES (?, 'task-1', 'agent-1', 'claimed', NULL, 0,
                    ?, ?, ?, 1, ?)
            """,
            (execution_id, due, due, occurrence_key, due),
        )

        claimed = await runner._claim(stale_task, datetime.now(timezone.utc))
        assert claimed is not None
        assert claimed.claim_execution_id == execution_id
        assert claimed.attempt_count == 2
        await runner._execute_claim(claimed)

        assert [execution.id for execution in seen] == [execution_id]
        assert await db.fetchone(
            "SELECT status, attempt_count, idempotency_key "
            "FROM task_execution_log WHERE id = ?",
            (execution_id,),
        ) == ("success", 2, occurrence_key)
        assert await db.fetchone(
            "SELECT COUNT(*) FROM task_execution_log WHERE task_id = ?",
            ("task-1",),
        ) == (1,)
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
async def test_overdue_builtin_one_shot_waits_for_agent_ready_before_dispatch(
    tmp_path,
):
    """Polling cannot consume a one-shot while its built-in owner is loading."""

    db = await _database(tmp_path / "post-load-one-shot.db")
    agent_id = "did:scheduler:post-load-one-shot"
    agent = SimpleNamespace(
        did=agent_id,
        agent_id=agent_id,
        features={},
        storage=SimpleNamespace(db=db),
    )
    scheduler = SchedulerFeature(agent)
    executed = asyncio.Event()

    class RestartOwner:
        enabled = True
        name = "RestartCoordinatorFeature"

        def __init__(self):
            async def execute(**_kwargs):
                executed.set()
                return {"success": True}

            self.tool = SimpleNamespace(
                name="restart_coordinator",
                execute=execute,
            )

        def get_tools(self):
            return [self.tool]

    try:
        await scheduler.initialize()
        assert scheduler._runner is not None
        assert scheduler._runner._task is None
        deadline = (
            datetime.now(timezone.utc) - timedelta(seconds=2)
        ).isoformat()
        added = await scheduler.schedule_add_deadline(
            run_at=deadline,
            task_name="restart_coordinator",
            idempotency_key="post-load-one-shot",
        )
        assert added.status.value == "ok"

        # Even an overdue row remains untouched throughout post-load wiring.
        await asyncio.sleep(0.05)
        assert await db.fetchone(
            "SELECT COUNT(*) FROM task_execution_log WHERE task_id = ?",
            (added.data["task_id"],),
        ) == (0,)

        agent.features["RestartCoordinatorFeature"] = RestartOwner()
        await scheduler.on_agent_ready(agent)
        await asyncio.wait_for(executed.wait(), timeout=1)
        terminal_row = None
        for _ in range(100):
            terminal_row = await db.fetchone(
                """
                SELECT enabled, terminal_status
                FROM scheduled_tasks WHERE id = ?
                """,
                (added.data["task_id"],),
            )
            if terminal_row == (0, "success"):
                break
            await asyncio.sleep(0.01)
        assert terminal_row == (0, "success")
        assert await db.fetchone(
            """
            SELECT status FROM task_execution_log
            WHERE task_id = ?
            """,
            (added.data["task_id"],),
        ) == ("success",)
    finally:
        await scheduler.shutdown()
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


@pytest.mark.asyncio
async def test_claim_lease_starts_after_a_semaphore_queue_wait(tmp_path):
    """A queued task gets a fresh lease, not the stale poll timestamp."""

    db = await _database(tmp_path / "queued-claim.db")
    first_started = asyncio.Event()
    release_first = asyncio.Event()
    second_started = asyncio.Event()
    observed = {}

    async def executor(name, _args):
        if name == "first":
            first_started.set()
            await release_first.wait()
            return "first complete"
        second_started.set()
        observed["lease_expires_at"] = await db.fetchval(
            "SELECT lease_expires_at FROM scheduled_tasks WHERE id = ?",
            ("second-task",),
        )
        observed["claimed_at"] = await db.fetchval(
            "SELECT claimed_at FROM task_execution_log WHERE task_id = ?",
            ("second-task",),
        )
        return "second complete"

    runner = SchedulerRunner(
        db,
        "agent-1",
        executor,
        owner_id="queued-worker",
        max_concurrent_tasks=1,
        lease_seconds=1,
    )
    try:
        await runner._ensure_tables()
        await _seed_due(db, task_id="first-task", task_name="first")
        await _seed_due(db, task_id="second-task", task_name="second")

        tick = asyncio.create_task(runner._tick())
        await asyncio.wait_for(first_started.wait(), timeout=1)
        # This deliberately exceeds the whole configured lease interval while
        # the second due row waits behind the semaphore.
        await asyncio.sleep(1.1)
        release_first.set()
        await asyncio.wait_for(second_started.wait(), timeout=2)
        await tick

        claimed_at = datetime.fromisoformat(observed["claimed_at"])
        lease_expires_at = datetime.fromisoformat(observed["lease_expires_at"])
        assert (lease_expires_at - claimed_at).total_seconds() >= 0.9
        assert lease_expires_at > datetime.now(timezone.utc) + timedelta(seconds=0.5)
    finally:
        release_first.set()
        await db.close()


@pytest.mark.asyncio
async def test_one_second_lease_renews_strictly_before_expiry(tmp_path):
    """A one-second claim renews at about one-third, never at expiry."""

    db = await _database(tmp_path / "one-second-renewal.db")
    dispatch_started = asyncio.Event()
    release_dispatch = asyncio.Event()

    async def executor(_name, _args):
        dispatch_started.set()
        await release_dispatch.wait()
        return "done"

    runner = SchedulerRunner(
        db, "agent-1", executor, owner_id="renew-before-expiry", lease_seconds=1
    )
    try:
        await runner._ensure_tables()
        await _seed_due(db)
        tick = asyncio.create_task(runner._tick())
        await asyncio.wait_for(dispatch_started.wait(), timeout=1)
        initial_expiry = await db.fetchval(
            "SELECT lease_expires_at FROM scheduled_tasks WHERE id = ?", ("task-1",)
        )

        renewed_expiry = initial_expiry
        deadline = asyncio.get_running_loop().time() + 0.8
        while renewed_expiry == initial_expiry and asyncio.get_running_loop().time() < deadline:
            await asyncio.sleep(0.03)
            renewed_expiry = await db.fetchval(
                "SELECT lease_expires_at FROM scheduled_tasks WHERE id = ?", ("task-1",)
            )

        assert renewed_expiry != initial_expiry
        assert datetime.fromisoformat(renewed_expiry) > datetime.now(timezone.utc)
        release_dispatch.set()
        await tick
    finally:
        release_dispatch.set()
        await db.close()


@pytest.mark.asyncio
async def test_sqlite_rollout_gate_allows_executor_write_and_delays_fence(tmp_path):
    """SQLite fencing waits for an admitted effect without blocking its writes.

    The runner owns a file-backed cross-connection gate, not SQLite's global
    writer transaction, while an executor is in flight. This catches both
    regressions at once: a fencer cannot cross the external-effect boundary,
    and target storage work plus the one-second lease renewal remain writable.
    """

    db_a = await _database(tmp_path / "sqlite-rollout-gate.db")
    db_b = await _database(tmp_path / "sqlite-rollout-gate.db")
    entered_executor = asyncio.Event()
    release_executor = asyncio.Event()
    fencer_started = asyncio.Event()
    v2_effects = []

    async def executor(_name, _args):
        # This write would deadlock if admission retained SQLite's one writer
        # transaction through dispatch.
        await db_a.execute(
            "INSERT INTO scheduler_gate_effects (value) VALUES ('effect')"
        )
        v2_effects.append("effect")
        entered_executor.set()
        await release_executor.wait()
        return "done"

    runner = SchedulerRunner(
        db_a, "agent-1", executor, owner_id="sqlite-effect", lease_seconds=1
    )
    fencer = SchedulerRunner(db_b, "agent-1", AsyncMock(), owner_id="sqlite-fencer")
    tick: asyncio.Task[None] | None = None
    fence_task: asyncio.Task[None] | None = None
    try:
        await runner._ensure_tables()
        await fencer._ensure_tables()
        await db_a.execute("CREATE TABLE scheduler_gate_effects (value TEXT NOT NULL)")
        await _seed_due(db_a)

        tick = asyncio.create_task(runner._tick())
        await asyncio.wait_for(entered_executor.wait(), timeout=1)
        assert await db_a.fetchval("SELECT COUNT(*) FROM scheduler_gate_effects") == 1

        # Model an origin/main writer which knows only legacy columns. Its
        # arrival makes the active DID quiesce, but the v2 fencer must wait for
        # the already-admitted external effect to reach its terminal CAS.
        due = (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat()
        await db_b.execute(
            """
            INSERT INTO scheduled_tasks
                (id, agent_id, task_name, cron_expression, args_json, enabled,
                 next_run_at, created_at)
            VALUES ('late-legacy', 'agent-1', 'legacy', '* * * * *', '{}', 1, ?, ?)
            """,
            (due, due),
        )

        async def fence() -> None:
            fencer_started.set()
            await fencer._ensure_protocol_rollout(preexisting_schedule_table=True)

        fence_task = asyncio.create_task(fence())
        await asyncio.wait_for(fencer_started.wait(), timeout=1)
        await asyncio.sleep(0.08)
        assert not fence_task.done()
        assert v2_effects == ["effect"]

        release_executor.set()
        await asyncio.wait_for(tick, timeout=2)
        with pytest.raises(SchedulerRolloutQuiescenceRequired):
            await asyncio.wait_for(fence_task, timeout=2)
        assert await db_a.fetchone(
            "SELECT status FROM task_execution_log WHERE task_id = ?", ("task-1",)
        ) == ("success",)
        assert await db_a.fetchone(
            "SELECT enabled, scheduler_rollout_fenced FROM scheduled_tasks WHERE id = ?",
            ("late-legacy",),
        ) == (0, 1)
    finally:
        release_executor.set()
        for task in (tick, fence_task):
            if task is not None and not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
        await db_a.close()
        await db_b.close()


@pytest.mark.asyncio
async def test_sqlite_same_did_effects_share_admission_while_fence_drains_them(
    tmp_path,
):
    """Two due schedules for one DID run together before a fence takes over."""

    database_path = tmp_path / "sqlite-shared-rollout-admission.db"
    db_a = await _database(database_path)
    db_b = await _database(database_path)
    admitted_effects: list[str] = []
    both_effects_started = asyncio.Event()
    release_effects = asyncio.Event()
    transition_started = asyncio.Event()
    tick: asyncio.Task[None] | None = None
    transition: asyncio.Task[None] | None = None

    async def executor(_name, _args):
        execution = get_current_scheduler_execution()
        admitted_effects.append(execution.schedule_id)
        if len(admitted_effects) == 2:
            both_effects_started.set()
        await release_effects.wait()
        return "done"

    async def no_op(_name, _args):
        return None

    runner = SchedulerRunner(
        db_a,
        "agent-1",
        executor,
        owner_id="sqlite-shared-effects",
        max_concurrent_tasks=2,
    )
    fencer = SchedulerRunner(
        db_b, "agent-1", no_op, owner_id="sqlite-shared-transition"
    )
    try:
        await runner._ensure_tables()
        await _seed_due(db_a, task_id="shared-effect-a")
        await _seed_due(db_a, task_id="shared-effect-b")

        tick = asyncio.create_task(runner._tick())
        await asyncio.wait_for(both_effects_started.wait(), timeout=2)
        assert set(admitted_effects) == {"shared-effect-a", "shared-effect-b"}

        due = (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat()
        await db_b.execute(
            """
            INSERT INTO scheduled_tasks
                (id, agent_id, task_name, cron_expression, args_json, enabled,
                 next_run_at, created_at)
            VALUES ('shared-legacy', 'agent-1', 'legacy', '* * * * *', '{}', 1, ?, ?)
            """,
            (due, due),
        )

        async def reconcile() -> None:
            transition_started.set()
            await fencer._ensure_protocol_rollout(preexisting_schedule_table=True)

        transition = asyncio.create_task(reconcile())
        await asyncio.wait_for(transition_started.wait(), timeout=1)
        await asyncio.sleep(0.08)
        assert not transition.done()

        release_effects.set()
        await asyncio.wait_for(tick, timeout=2)
        with pytest.raises(SchedulerRolloutQuiescenceRequired):
            await asyncio.wait_for(transition, timeout=2)
        rows = await db_a.fetchall(
            "SELECT status FROM task_execution_log WHERE task_id IN (?, ?) ORDER BY task_id",
            ("shared-effect-a", "shared-effect-b"),
        )
        assert rows == [("success",), ("success",)]
    finally:
        release_effects.set()
        for task in (tick, transition):
            if task is not None and not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
        await db_a.close()
        await db_b.close()


def test_rollout_effect_advisory_key_never_uses_bootstrap_sentinel(monkeypatch):
    """A SHA-256 zero prefix cannot self-deadlock PostgreSQL bootstrap."""

    class ZeroDigest:
        def digest(self) -> bytes:
            return b"\0" * 32

    monkeypatch.setattr(
        scheduler_runner_module.hashlib,
        "sha256",
        lambda _payload: ZeroDigest(),
    )

    assert SchedulerRunner._rollout_effect_advisory_key("did:scheduler:zero") != 0


@pytest.mark.asyncio
async def test_renewal_stays_alive_while_terminal_cas_is_contended(
    monkeypatch, tmp_path
):
    """Renewal remains owned until the final schedule/log CAS can commit."""

    db = await _database(tmp_path / "terminal-cas-renewal.db")
    renewal_attempted = asyncio.Event()
    terminal_cas_entered = asyncio.Event()
    release_terminal_cas = asyncio.Event()

    class RenewalProbeRunner(SchedulerRunner):
        async def _renew_lease_once(self, task):
            renewal_attempted.set()
            return await super()._renew_lease_once(task)

    runner = RenewalProbeRunner(
        db, "agent-1", AsyncMock(return_value="done"),
        owner_id="terminal-cas-worker", lease_seconds=1,
    )
    original_execute = db.execute

    async def gated_execute(sql, params=()):
        if "UPDATE scheduled_tasks" in sql and "terminal_status = ?" in sql:
            terminal_cas_entered.set()
            await release_terminal_cas.wait()
        return await original_execute(sql, params)

    try:
        await runner._ensure_tables()
        await _seed_due(db)
        monkeypatch.setattr(db, "execute", gated_execute)
        tick = asyncio.create_task(runner._tick())
        await asyncio.wait_for(terminal_cas_entered.wait(), timeout=1)
        # The final CAS is intentionally holding the DB write transaction. A
        # first renewal still wakes and tries to renew before we permit that
        # CAS to complete; this would be impossible if it were cancelled first.
        await asyncio.wait_for(renewal_attempted.wait(), timeout=1)
        release_terminal_cas.set()
        await asyncio.wait_for(tick, timeout=2)

        assert await db.fetchone(
            "SELECT status, attempt_count FROM task_execution_log WHERE task_id = ?",
            ("task-1",),
        ) == ("success", 1)
    finally:
        release_terminal_cas.set()
        await db.close()


@pytest.mark.asyncio
async def test_attempts_reset_for_next_cron_occurrence_but_recovery_increments(tmp_path):
    """Normal occurrence progress is attempt 1; retrying one claim is attempt 2."""

    db = await _database(tmp_path / "occurrence-attempts.db")
    observed = []

    async def executor(_name, _args):
        observed.append(get_current_scheduler_execution())
        return "done"

    runner = SchedulerRunner(db, "agent-1", executor, owner_id="attempt-worker")
    try:
        await runner._ensure_tables()
        first_occurrence = await _seed_due(db)
        await runner._tick()
        assert observed[-1].attempt == 1
        assert await db.fetchone(
            "SELECT attempt_count FROM scheduled_tasks WHERE id = ?", ("task-1",)
        ) == (0,)

        second_occurrence = (
            datetime.now(timezone.utc) - timedelta(seconds=2)
        ).isoformat()
        await db.execute(
            "UPDATE scheduled_tasks SET next_run_at = ? WHERE id = ?",
            (second_occurrence, "task-1"),
        )
        await runner._tick()
        assert observed[-1].attempt == 1
        assert [row[0] for row in await db.fetchall(
            "SELECT attempt_count FROM task_execution_log WHERE task_id = ? ORDER BY rowid",
            ("task-1",),
        )] == [1, 1]

        recovery_occurrence = (
            datetime.now(timezone.utc) - timedelta(seconds=2)
        ).isoformat()
        expired = (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat()
        recovery_key = SchedulerRunner._occurrence_idempotency_key(
            "stable-effect", "task-1", recovery_occurrence
        )
        await db.execute(
            """
            UPDATE scheduled_tasks
            SET enabled = 0, scheduler_claim_fenced = 1, next_run_at = ?,
                lease_owner = 'dead-worker', lease_expires_at = ?,
                claim_token = 'old-token', claim_execution_id = 'recovery-exec',
                claim_scheduled_for = ?, attempt_count = 1
            WHERE id = ?
            """,
            (recovery_occurrence, expired, recovery_occurrence, "task-1"),
        )
        await db.execute(
            """
            INSERT INTO task_execution_log
                (id, task_id, agent_id, status, result_text, duration_ms,
                 executed_at, occurrence_at, idempotency_key, attempt_count, claimed_at)
            VALUES ('recovery-exec', 'task-1', 'agent-1', 'claimed', NULL, 0,
                    ?, ?, ?, 1, ?)
            """,
            (recovery_occurrence, recovery_occurrence, recovery_key, recovery_occurrence),
        )
        await runner._tick()

        assert observed[-1].id == "recovery-exec"
        assert observed[-1].attempt == 2
        assert await db.fetchone(
            "SELECT status, attempt_count FROM task_execution_log WHERE id = 'recovery-exec'"
        ) == ("success", 2)
        assert first_occurrence != second_occurrence
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_invalid_persisted_idempotency_key_is_disabled_with_visible_terminal_log(tmp_path):
    """A bad historical base key is preserved as a visible failed invariant."""

    db = await _database(tmp_path / "invalid-idempotency.db")
    executor = AsyncMock(return_value="must-not-run")
    runner = SchedulerRunner(db, "agent-1", executor, owner_id="invalid-key-worker")
    try:
        await runner._ensure_tables()
        await _seed_due(db)
        await db.execute(
            "UPDATE scheduled_tasks SET idempotency_key = ? WHERE id = ?",
            ("é" * 224, "task-1"),  # 448 UTF-8 bytes
        )

        await runner._tick()

        executor.assert_not_awaited()
        assert await db.fetchone(
            "SELECT enabled, terminal_status FROM scheduled_tasks WHERE id = ?",
            ("task-1",),
        ) == (0, "invalid_idempotency_key")
        log = await db.fetchone(
            "SELECT status, result_text FROM task_execution_log WHERE task_id = ?",
            ("task-1",),
        )
        assert log[0] == "invalid_idempotency_key"
        assert "448 UTF-8 bytes" in log[1]
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_schedule_add_locks_and_requires_active_rollout_control_row(tmp_path):
    """Runnable schedule writes cannot race a durable quiescing transition."""

    db = await _database(tmp_path / "schedule-writer-rollout.db")
    runner = SchedulerRunner(db, "agent-1", AsyncMock(), owner_id="schema-owner")
    agent = SimpleNamespace(did="agent-1", agent_id="agent-1", features={})
    feature = SchedulerFeature(agent)
    feature._db = db
    feature._agent_id = "agent-1"
    try:
        await runner._ensure_tables()
        added = await feature.schedule_add(
            cron_expression="@daily",
            task_name="memory_consolidate",
            idempotency_key="active-control-row",
        )
        assert added.status.value == "ok"
        assert await db.fetchone(
            "SELECT scheduler_protocol_version FROM scheduled_tasks WHERE id = ?",
            (added.data["task_id"],),
        ) == (SCHEDULER_PROTOCOL_VERSION,)

        # A concurrent legacy observation moves the DID to quiescing. The
        # feature's active-row UPDATE lock must reject new runnable rows rather
        # than relying on a stale EXISTS snapshot.
        await db.execute(
            """
            UPDATE scheduler_protocol_rollout
            SET state = 'quiescing', activation_nonce = 'test-quiescing-nonce'
            WHERE agent_id = ?
            """,
            ("agent-1",),
        )
        blocked = await feature.schedule_add(
            cron_expression="@daily",
            task_name="memory_consolidate",
            idempotency_key="must-not-persist",
        )
        assert blocked.status.value == "error"
        assert "rollout is not active" in blocked.error
        assert await db.fetchone(
            "SELECT COUNT(*) FROM scheduled_tasks WHERE agent_id = ?",
            ("agent-1",),
        ) == (1,)
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_pause_during_rollout_quiescence_survives_ack_activation(tmp_path, monkeypatch):
    """An operator pause is not mistaken for a rollout fence on activation."""

    db = await _database(tmp_path / "pause-during-rollout.db")
    runner = SchedulerRunner(db, "agent-1", AsyncMock(), owner_id="rollout-owner")
    agent = SimpleNamespace(did="agent-1", agent_id="agent-1", features={})
    feature = SchedulerFeature(agent)
    feature._db = db
    feature._agent_id = "agent-1"
    try:
        await runner._ensure_tables()
        await _seed_due(db)
        nonce = "pause-during-quiescence"
        await db.execute(
            """
            UPDATE scheduler_protocol_rollout
            SET state = 'quiescing', activation_nonce = ?
            WHERE agent_id = ?
            """,
            (nonce, "agent-1"),
        )
        async with runner._transaction():
            await runner._fence_legacy_agent_rows("agent-1", nonce)

        paused = await feature.schedule_pause(task_id="task-1")
        assert paused.data["status"] == "paused"
        assert await db.fetchone(
            """
            SELECT enabled, scheduler_rollout_fenced, scheduler_claim_fenced
            FROM scheduled_tasks WHERE id = ?
            """,
            ("task-1",),
        ) == (0, 0, 0)

        monkeypatch.setenv(SCHEDULER_ROLLOUT_ACK_ENV, nonce)
        acknowledged = SchedulerRunner(db, "agent-1", AsyncMock(), owner_id="ack")
        await acknowledged._ensure_tables()
        assert await db.fetchone(
            """
            SELECT enabled, scheduler_rollout_fenced, scheduler_claim_fenced
            FROM scheduled_tasks WHERE id = ?
            """,
            ("task-1",),
        ) == (0, 0, 0)
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_schedule_update_serializes_with_pause_and_resume_interleavings(tmp_path):
    """A definition update cannot overwrite a concurrent pause or resume."""

    db_a = await _database(tmp_path / "update-pause-race.db")
    db_b = await _database(tmp_path / "update-pause-race.db")
    runner = SchedulerRunner(db_a, "agent-1", AsyncMock(), owner_id="race-schema")
    agent = SimpleNamespace(did="agent-1", agent_id="agent-1", features={})
    update_feature = SchedulerFeature(agent)
    update_feature._db = db_a
    update_feature._agent_id = "agent-1"
    pause_feature = SchedulerFeature(agent)
    pause_feature._db = db_b
    pause_feature._agent_id = "agent-1"
    try:
        await runner._ensure_tables()
        await _seed_due(db_a)

        # First ordering: update gets the control-row lock first. Pause waits,
        # then wins last; the final state must be paused rather than the
        # update's stale enabled=1 snapshot.
        update_locked = asyncio.Event()
        release_update = asyncio.Event()
        original_update_lock = update_feature._lock_active_scheduler_rollout

        async def hold_update_lock():
            locked = await original_update_lock()
            update_locked.set()
            await release_update.wait()
            return locked

        update_feature._lock_active_scheduler_rollout = hold_update_lock
        update_task = asyncio.create_task(
            update_feature.schedule_update("task-1", "@hourly")
        )
        await asyncio.wait_for(update_locked.wait(), timeout=1)
        pause_task = asyncio.create_task(pause_feature.schedule_pause("task-1"))
        await asyncio.sleep(0.05)
        assert not pause_task.done()
        release_update.set()
        assert (await update_task).data["status"] == "updated"
        assert (await pause_task).data["status"] == "paused"
        assert await db_a.fetchone(
            "SELECT enabled, cron_expression FROM scheduled_tasks WHERE id = ?",
            ("task-1",),
        ) == (0, "@hourly")

        # Second ordering: pause gets the control row first. Update must read
        # the committed enabled=0 row *after* that lock, retain disabled state,
        # and only change the definition.
        pause_locked = asyncio.Event()
        release_pause = asyncio.Event()
        original_pause_lock = pause_feature._lock_scheduler_rollout_for_pause

        async def hold_pause_lock():
            locked = await original_pause_lock()
            pause_locked.set()
            await release_pause.wait()
            return locked

        pause_feature._lock_scheduler_rollout_for_pause = hold_pause_lock
        # Resume once so this ordering starts from a genuinely active row.
        await update_feature.schedule_resume("task-1")
        pause_task = asyncio.create_task(pause_feature.schedule_pause("task-1"))
        await asyncio.wait_for(pause_locked.wait(), timeout=1)
        update_task = asyncio.create_task(
            update_feature.schedule_update("task-1", "@weekly")
        )
        await asyncio.sleep(0.05)
        assert not update_task.done()
        release_pause.set()
        assert (await pause_task).data["status"] == "paused"
        assert (await update_task).data["status"] == "updated"
        assert await db_a.fetchone(
            "SELECT enabled, cron_expression FROM scheduled_tasks WHERE id = ?",
            ("task-1",),
        ) == (0, "@weekly")

        # The original stale-read defect also ran in the other direction:
        # schedule_update could read enabled=0, let resume commit enabled=1,
        # then overwrite it with stale disabled state. Put resume first under
        # the same control lock and prove the later update observes enabled=1.
        resume_locked = asyncio.Event()
        release_resume = asyncio.Event()
        original_resume_lock = update_feature._lock_active_scheduler_rollout

        async def hold_resume_lock():
            locked = await original_resume_lock()
            resume_locked.set()
            await release_resume.wait()
            return locked

        update_feature._lock_active_scheduler_rollout = hold_resume_lock
        resume_task = asyncio.create_task(update_feature.schedule_resume("task-1"))
        await asyncio.wait_for(resume_locked.wait(), timeout=1)
        update_task = asyncio.create_task(
            pause_feature.schedule_update("task-1", "@monthly")
        )
        await asyncio.sleep(0.05)
        assert not update_task.done()
        release_resume.set()
        assert (await resume_task).data["status"] == "resumed"
        assert (await update_task).data["status"] == "updated"
        assert await db_a.fetchone(
            "SELECT enabled, cron_expression FROM scheduled_tasks WHERE id = ?",
            ("task-1",),
        ) == (1, "@monthly")
    finally:
        await db_a.close()
        await db_b.close()


def test_legacy_generated_idempotency_base_and_occurrence_are_sdk_safe():
    """Even pathological old schedule IDs get a bounded stable generated key."""

    schedule_id = "legacy-" + ("é" * 600)
    first = SchedulerRunner._legacy_base_idempotency_key(schedule_id)
    second = SchedulerRunner._legacy_base_idempotency_key(schedule_id)
    occurrence = SchedulerRunner._occurrence_idempotency_key(
        first, schedule_id, "2026-07-25T00:00:00+00:00"
    )

    assert first == second
    assert len(first.encode("utf-8")) <= 447
    assert len(occurrence.encode("utf-8")) <= 512


@pytest.mark.asyncio
async def test_preexisting_legacy_table_requires_nonce_before_v2_execution(
    monkeypatch, tmp_path
):
    """A legacy selector cannot see a durably fenced row during rollout."""
    db = await _database(tmp_path / "legacy-scheduler.db")
    delivered = []
    due = (datetime.now(timezone.utc) - timedelta(seconds=2)).isoformat()
    await db.execute(
        """
        CREATE TABLE scheduled_tasks (
            id TEXT PRIMARY KEY,
            agent_id TEXT NOT NULL,
            task_name TEXT NOT NULL,
            cron_expression TEXT NOT NULL,
            args_json TEXT,
            enabled INTEGER,
            last_run_at TEXT,
            next_run_at TEXT,
            created_at TEXT NOT NULL
        )
        """
    )
    await db.execute(
        """
        INSERT INTO scheduled_tasks
            (id, agent_id, task_name, cron_expression, args_json, enabled,
             next_run_at, created_at)
        VALUES ('legacy-task', 'agent-1', 'legacy_task', '* * * * *', '{}', 1, ?, ?)
        """,
        (due, due),
    )

    async def executor(name, args):
        delivered.append((name, args))
        return "done"

    first = SchedulerRunner(db, "agent-1", executor, owner_id="new-replica")
    try:
        with pytest.raises(SchedulerRolloutQuiescenceRequired):
            await first._ensure_tables()

        state = await db.fetchone(
            """
            SELECT protocol_version, state, activation_nonce
            FROM scheduler_protocol_rollout WHERE agent_id = ?
            """,
            ("agent-1",),
        )
        assert state[0:2] == (SCHEDULER_PROTOCOL_VERSION, "quiescing")
        nonce = state[2]
        assert nonce
        assert await db.fetchall(
            """
            SELECT id FROM scheduled_tasks
            WHERE agent_id = ? AND enabled = 1 AND next_run_at <= ?
            """,
            ("agent-1", datetime.now(timezone.utc).isoformat()),
        ) == []
        assert await db.fetchone(
            """
            SELECT enabled, scheduler_protocol_version,
                   scheduler_rollout_fenced, scheduler_rollout_nonce
            FROM scheduled_tasks WHERE id = 'legacy-task'
            """
        ) == (0, None, 1, nonce)

        monkeypatch.setenv(SCHEDULER_ROLLOUT_ACK_ENV, nonce)
        activated = SchedulerRunner(db, "agent-1", executor, owner_id="v2-after-drain")
        await activated._ensure_tables()
        assert await db.fetchone(
            """
            SELECT enabled, scheduler_protocol_version,
                   scheduler_rollout_fenced, scheduler_rollout_nonce,
                   terminal_status
            FROM scheduled_tasks WHERE id = 'legacy-task'
            """
        ) == (
            0,
            SCHEDULER_PROTOCOL_VERSION,
            0,
            None,
            "rollout_ambiguous_legacy_occurrence",
        )

        await activated._tick()
        # This row was due when the legacy selector was fenced. Its effect may
        # already have happened in the exact select → dispatch → re-read
        # ordering of origin/main, so ACK never replays it automatically.
        assert delivered == []
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_legacy_selected_then_fenced_then_dispatched_and_disabled_reread_stays_paused(
    monkeypatch, tmp_path
):
    """Model origin/main's exact select → fence → effect → reread ordering.

    The old scheduler selected by ``agent_id + enabled + next_run_at`` before
    v2 fenced the row. It can still perform its external effect afterwards,
    but its post-dispatch enabled reread sees zero and leaves the overdue
    occurrence in place. ACK must preserve that ambiguity as a visible pause,
    never replay it on v2.
    """

    db = await _database(tmp_path / "legacy-selected-before-fence.db")
    due = (datetime.now(timezone.utc) - timedelta(seconds=2)).isoformat()
    v2_calls = AsyncMock(return_value="must-not-run")
    await db.execute(
        """
        CREATE TABLE scheduled_tasks (
            id TEXT PRIMARY KEY, agent_id TEXT NOT NULL, task_name TEXT NOT NULL,
            cron_expression TEXT NOT NULL, args_json TEXT, enabled INTEGER,
            last_run_at TEXT, next_run_at TEXT, created_at TEXT NOT NULL
        )
        """
    )
    await db.execute(
        """
        INSERT INTO scheduled_tasks
            (id, agent_id, task_name, cron_expression, args_json, enabled,
             next_run_at, created_at)
        VALUES ('legacy-task', 'agent-1', 'legacy', '* * * * *', '{}', 1, ?, ?)
        """,
        (due, due),
    )
    first = SchedulerRunner(db, "agent-1", v2_calls, owner_id="v2-fence")
    try:
        # Exact legacy due selector, before the v2 rollout process starts.
        selected = await db.fetchone(
            """
            SELECT id, task_name, args_json FROM scheduled_tasks
            WHERE agent_id = ? AND enabled = 1 AND next_run_at <= ?
            ORDER BY next_run_at ASC
            """,
            ("agent-1", datetime.now(timezone.utc).isoformat()),
        )
        assert selected == ("legacy-task", "legacy", "{}")

        with pytest.raises(SchedulerRolloutQuiescenceRequired):
            await first._ensure_tables()
        nonce = await db.fetchval(
            "SELECT activation_nonce FROM scheduler_protocol_rollout WHERE agent_id = ?",
            ("agent-1",),
        )
        assert nonce

        # The old worker already owns the selected occurrence, so it can make
        # its effect after v2's fence. These are the literal origin/main
        # post-effect statements: when the reread sees enabled=0 it does not
        # reschedule, but it *does* unconditionally stamp last_run_at.
        legacy_effects = [selected[0]]
        post_effect = await db.fetchone(
            "SELECT cron_expression, enabled FROM scheduled_tasks WHERE id = ?",
            ("legacy-task",),
        )
        assert post_effect == ("* * * * *", 0)
        legacy_last_run = datetime.now(timezone.utc).isoformat()
        assert await db.execute(
            "UPDATE scheduled_tasks SET last_run_at = ? WHERE id = ?",
            (legacy_last_run, "legacy-task"),
        ) == 1
        assert await db.fetchone(
            "SELECT last_run_at, next_run_at FROM scheduled_tasks WHERE id = ?",
            ("legacy-task",),
        ) == (legacy_last_run, due)

        # The unconditional timestamp change invalidates the first snapshot,
        # so its acknowledgement must rotate instead of activating rows under
        # a stale drain proof.
        monkeypatch.setenv(SCHEDULER_ROLLOUT_ACK_ENV, nonce)
        stale_ack = SchedulerRunner(db, "agent-1", v2_calls, owner_id="v2-stale-ack")
        with pytest.raises(SchedulerRolloutQuiescenceRequired):
            await stale_ack._ensure_tables()
        fresh_nonce = await db.fetchval(
            "SELECT activation_nonce FROM scheduler_protocol_rollout WHERE agent_id = ?",
            ("agent-1",),
        )
        assert fresh_nonce and fresh_nonce != nonce

        monkeypatch.setenv(SCHEDULER_ROLLOUT_ACK_ENV, fresh_nonce)
        acknowledged = SchedulerRunner(db, "agent-1", v2_calls, owner_id="v2-fresh-ack")
        await acknowledged._ensure_tables()
        assert legacy_effects == ["legacy-task"]
        assert await db.fetchone(
            "SELECT enabled, last_run_at, next_run_at, terminal_status FROM scheduled_tasks WHERE id = ?",
            ("legacy-task",),
        ) == (0, legacy_last_run, due, "rollout_ambiguous_legacy_occurrence")
        await acknowledged._tick()
        v2_calls.assert_not_awaited()
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_legacy_enabled_reread_before_fence_cannot_conditionally_reschedule_afterward(
    monkeypatch, tmp_path
):
    """The old conditional reschedule is a no-op if the fence wins second."""

    db = await _database(tmp_path / "legacy-reread-before-fence.db")
    due = (datetime.now(timezone.utc) - timedelta(seconds=2)).isoformat()
    future = (datetime.now(timezone.utc) + timedelta(minutes=5)).isoformat()
    v2_calls = AsyncMock(return_value="must-not-run")
    await db.execute(
        """
        CREATE TABLE scheduled_tasks (
            id TEXT PRIMARY KEY, agent_id TEXT NOT NULL, task_name TEXT NOT NULL,
            cron_expression TEXT NOT NULL, args_json TEXT, enabled INTEGER,
            last_run_at TEXT, next_run_at TEXT, created_at TEXT NOT NULL
        )
        """
    )
    await db.execute(
        """
        INSERT INTO scheduled_tasks
            (id, agent_id, task_name, cron_expression, args_json, enabled,
             next_run_at, created_at)
        VALUES ('legacy-task', 'agent-1', 'legacy', '* * * * *', '{}', 1, ?, ?)
        """,
        (due, due),
    )
    first = SchedulerRunner(db, "agent-1", v2_calls, owner_id="v2-fence")
    try:
        # This is origin/main's literal pre-fence post-effect enabled reread.
        assert await db.fetchone(
            "SELECT cron_expression, enabled FROM scheduled_tasks WHERE id = ?",
            ("legacy-task",),
        ) == ("* * * * *", 1)
        with pytest.raises(SchedulerRolloutQuiescenceRequired):
            await first._ensure_tables()
        nonce = await db.fetchval(
            "SELECT activation_nonce FROM scheduler_protocol_rollout WHERE agent_id = ?",
            ("agent-1",),
        )
        assert nonce

        # Origin/main's conditional post-dispatch update cannot undo the
        # fence. Preserve the literal enabled predicate in this regression.
        rescheduled = await db.execute(
            """
            UPDATE scheduled_tasks
            SET last_run_at = ?, next_run_at = ?
            WHERE id = ? AND enabled = 1
            """,
            (datetime.now(timezone.utc).isoformat(), future, "legacy-task"),
        )
        assert rescheduled == 0

        monkeypatch.setenv(SCHEDULER_ROLLOUT_ACK_ENV, nonce)
        acknowledged = SchedulerRunner(db, "agent-1", v2_calls, owner_id="v2-ack")
        await acknowledged._ensure_tables()
        assert await db.fetchone(
            "SELECT enabled, next_run_at, terminal_status FROM scheduled_tasks WHERE id = ?",
            ("legacy-task",),
        ) == (0, due, "rollout_ambiguous_legacy_occurrence")
        await acknowledged._tick()
        v2_calls.assert_not_awaited()
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_preexisting_empty_table_requires_quiesced_nonce(tmp_path):
    """An empty legacy table is still unsafe while an old DID is deployed."""

    db = await _database(tmp_path / "empty-legacy-scheduler.db")
    await db.execute(
        """
        CREATE TABLE scheduled_tasks (
            id TEXT PRIMARY KEY,
            agent_id TEXT NOT NULL,
            task_name TEXT NOT NULL,
            cron_expression TEXT NOT NULL,
            args_json TEXT,
            enabled INTEGER,
            last_run_at TEXT,
            next_run_at TEXT,
            created_at TEXT NOT NULL
        )
        """
    )
    runner = SchedulerRunner(db, "agent-1", AsyncMock(), owner_id="v2")
    try:
        with pytest.raises(SchedulerRolloutQuiescenceRequired):
            await runner._ensure_tables()

        state = await db.fetchone(
            """
            SELECT protocol_version, state, activation_nonce
            FROM scheduler_protocol_rollout WHERE agent_id = ?
            """,
            ("agent-1",),
        )
        assert state[0:2] == (SCHEDULER_PROTOCOL_VERSION, "quiescing")
        assert state[2]
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_legacy_insert_omitting_protocol_on_v2_schema_is_quiesced(
    tmp_path,
):
    """A default-less protocol column keeps an old writer provenance-visible."""
    db = await _database(tmp_path / "v2-schema-legacy-writer.db")
    runner = SchedulerRunner(db, "agent-1", AsyncMock(), owner_id="v2")
    try:
        await runner._ensure_tables()
        due = (datetime.now(timezone.utc) - timedelta(seconds=2)).isoformat()
        # Simulate origin/main SQL: it cannot name any v2-only column.
        await db.execute(
            """
            INSERT INTO scheduled_tasks
                (id, agent_id, task_name, cron_expression, args_json, enabled,
                 next_run_at, created_at)
            VALUES ('legacy-write', 'agent-1', 'task', '* * * * *', '{}', 1, ?, ?)
            """,
            (due, due),
        )

        with pytest.raises(SchedulerRolloutQuiescenceRequired):
            await runner._tick()

        assert await db.fetchone(
            """
            SELECT enabled, scheduler_protocol_version, scheduler_rollout_fenced
            FROM scheduled_tasks WHERE id = 'legacy-write'
            """
        ) == (0, None, 1)
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_legacy_update_of_fenced_baseline_rotates_nonce_before_ack(
    monkeypatch, tmp_path
):
    """An old reschedule of an existing row cannot reuse its old nonce."""

    db = await _database(tmp_path / "baseline-update-rotation.db")
    due = (datetime.now(timezone.utc) - timedelta(seconds=2)).isoformat()
    await db.execute(
        """
        CREATE TABLE scheduled_tasks (
            id TEXT PRIMARY KEY, agent_id TEXT NOT NULL, task_name TEXT NOT NULL,
            cron_expression TEXT NOT NULL, args_json TEXT, enabled INTEGER,
            last_run_at TEXT, next_run_at TEXT, created_at TEXT NOT NULL
        )
        """
    )
    await db.execute(
        """
        INSERT INTO scheduled_tasks
            (id, agent_id, task_name, cron_expression, args_json, enabled,
             next_run_at, created_at)
        VALUES ('legacy-task', 'agent-1', 'task', '* * * * *', '{}', 1, ?, ?)
        """,
        (due, due),
    )
    initial = SchedulerRunner(db, "agent-1", AsyncMock(), owner_id="first-v2")
    try:
        with pytest.raises(SchedulerRolloutQuiescenceRequired):
            await initial._ensure_tables()
        old_nonce = await db.fetchval(
            "SELECT activation_nonce FROM scheduler_protocol_rollout WHERE agent_id = ?",
            ("agent-1",),
        )
        # Simulate a legacy poller that read the occurrence before fencing and
        # rescheduled the same row without knowing v2-only columns.  Its nonce
        # remains stamped, so the snapshot must be what detects this write.
        legacy_next = (datetime.now(timezone.utc) + timedelta(minutes=5)).isoformat()
        await db.execute(
            """
            UPDATE scheduled_tasks
            SET enabled = 1, last_run_at = ?, next_run_at = ?
            WHERE id = 'legacy-task' AND agent_id = 'agent-1'
            """,
            (datetime.now(timezone.utc).isoformat(), legacy_next),
        )

        monkeypatch.setenv(SCHEDULER_ROLLOUT_ACK_ENV, old_nonce)
        retry = SchedulerRunner(db, "agent-1", AsyncMock(), owner_id="retry-v2")
        with pytest.raises(SchedulerRolloutQuiescenceRequired):
            await retry._ensure_tables()

        new_nonce = await db.fetchval(
            "SELECT activation_nonce FROM scheduler_protocol_rollout WHERE agent_id = ?",
            ("agent-1",),
        )
        assert new_nonce and new_nonce != old_nonce
        row = await db.fetchone(
            """
            SELECT enabled, scheduler_rollout_nonce, scheduler_rollout_snapshot
            FROM scheduled_tasks WHERE id = 'legacy-task'
            """
        )
        assert row[0:2] == (0, new_nonce)
        assert row[2]
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_unknown_row_during_quiescence_rotates_nonce_before_old_ack(
    monkeypatch, tmp_path
):
    """A later legacy write invalidates an acknowledgement of the old fence."""
    db = await _database(tmp_path / "nonce-rotation.db")
    due = (datetime.now(timezone.utc) - timedelta(seconds=2)).isoformat()
    await db.execute(
        """
        CREATE TABLE scheduled_tasks (
            id TEXT PRIMARY KEY, agent_id TEXT NOT NULL, task_name TEXT NOT NULL,
            cron_expression TEXT NOT NULL, args_json TEXT, enabled INTEGER,
            last_run_at TEXT, next_run_at TEXT, created_at TEXT NOT NULL
        )
        """
    )
    await db.execute(
        """
        INSERT INTO scheduled_tasks
            (id, agent_id, task_name, cron_expression, args_json, enabled,
             next_run_at, created_at)
        VALUES ('first', 'agent-1', 'task', '* * * * *', '{}', 1, ?, ?)
        """,
        (due, due),
    )
    initial = SchedulerRunner(db, "agent-1", AsyncMock(), owner_id="first-v2")
    try:
        with pytest.raises(SchedulerRolloutQuiescenceRequired):
            await initial._ensure_tables()
        old_nonce = await db.fetchval(
            "SELECT activation_nonce FROM scheduler_protocol_rollout WHERE agent_id = ?",
            ("agent-1",),
        )
        # This is a new legacy write after the first fencing pass: no protocol
        # and no baseline nonce. Presenting the old acknowledgement must not
        # convert or re-enable either row.
        await db.execute(
            """
            INSERT INTO scheduled_tasks
                (id, agent_id, task_name, cron_expression, args_json, enabled,
                 next_run_at, created_at)
            VALUES ('late-legacy', 'agent-1', 'task', '* * * * *', '{}', 1, ?, ?)
            """,
            (due, due),
        )
        monkeypatch.setenv(SCHEDULER_ROLLOUT_ACK_ENV, old_nonce)
        retry = SchedulerRunner(db, "agent-1", AsyncMock(), owner_id="retry-v2")
        with pytest.raises(SchedulerRolloutQuiescenceRequired):
            await retry._ensure_tables()

        new_nonce = await db.fetchval(
            "SELECT activation_nonce FROM scheduler_protocol_rollout WHERE agent_id = ?",
            ("agent-1",),
        )
        assert new_nonce and new_nonce != old_nonce
        assert await db.fetchall(
            "SELECT enabled, scheduler_rollout_nonce FROM scheduled_tasks WHERE agent_id = ?",
            ("agent-1",),
        ) == [(0, new_nonce), (0, new_nonce)]
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_rollout_fence_revokes_live_claim_and_recovers_stable_occurrence_after_ack(
    monkeypatch, tmp_path
):
    """A fence covers enabled=0 claims and stale finalization cannot re-enable.

    This is the claim-bearing counterpart to the legacy selector tests. The
    fence revokes the live owner/token before ACK, preserves the occurrence
    identity for v2 recovery, and rejects the pre-fence worker's terminal CAS.
    """

    db = await _database(tmp_path / "claim-fence-recovery.db")
    delivered = []

    async def executor(_name, _args):
        delivered.append(get_current_scheduler_execution())
        return "recovered"

    runner = SchedulerRunner(db, "agent-1", executor, owner_id="old-v2")
    try:
        await runner._ensure_tables()
        due = await _seed_due(db)
        task = ScheduledTask.from_row(
            (await runner._due_rows(datetime.now(timezone.utc)))[0]
        )
        claimed = await runner._claim(task, datetime.now(timezone.utc))
        assert claimed is not None
        nonce = "claim-fence-nonce"

        # Simulate a late legacy observation that moves this DID to a new
        # quiescing epoch while a v2 claim exists but has not dispatched.
        await db.execute(
            """
            UPDATE scheduler_protocol_rollout
            SET state = 'quiescing', activation_nonce = ?
            WHERE agent_id = ?
            """,
            (nonce, "agent-1"),
        )
        async with runner._transaction():
            await runner._fence_legacy_agent_rows("agent-1", nonce)

        fenced = await db.fetchone(
            """
            SELECT enabled, scheduler_rollout_fenced, scheduler_claim_fenced,
                   lease_owner, lease_expires_at, claim_token,
                   claim_execution_id, claim_scheduled_for
            FROM scheduled_tasks WHERE id = ?
            """,
            ("task-1",),
        )
        assert fenced == (
            0,
            1,
            1,
            None,
            None,
            None,
            claimed.claim_execution_id,
            due,
        )

        monkeypatch.setenv(SCHEDULER_ROLLOUT_ACK_ENV, nonce)
        recovery = SchedulerRunner(db, "agent-1", executor, owner_id="new-v2")
        await recovery._ensure_tables()
        assert await db.fetchone(
            """
            SELECT enabled, scheduler_rollout_fenced, scheduler_claim_fenced,
                   claim_execution_id, claim_scheduled_for
            FROM scheduled_tasks WHERE id = ?
            """,
            ("task-1",),
        ) == (0, 0, 1, claimed.claim_execution_id, due)

        # The worker that held the old token cannot re-enable a recurring row
        # after ACK; its token was durably revoked by the fence.
        await runner._finalize(
            claimed,
            SchedulerExecution(
                id=claimed.claim_execution_id,
                schedule_id=claimed.id,
                agent_id=claimed.agent_id,
                task_name=claimed.task_name,
                args=claimed.args,
                scheduled_for=claimed.next_run_at,
                idempotency_key=SchedulerRunner._occurrence_idempotency_key(
                    claimed.idempotency_key, claimed.id, claimed.next_run_at
                ),
                attempt=claimed.attempt_count,
                owner="old-v2",
            ),
            status="success",
            result_text="stale",
            duration_ms=0,
            outcome_signal=None,
            ran=True,
        )
        assert await db.fetchone(
            "SELECT enabled, claim_token FROM scheduled_tasks WHERE id = ?",
            ("task-1",),
        ) == (0, None)

        await recovery._tick()
        assert len(delivered) == 1
        assert delivered[0].id == claimed.claim_execution_id
        assert delivered[0].attempt == 2
        assert await db.fetchone(
            "SELECT status, attempt_count FROM task_execution_log WHERE id = ?",
            (claimed.claim_execution_id,),
        ) == ("success", 2)
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
async def test_hosted_dispatch_refuses_a_soft_disabled_scheduler_feature():
    """A live host cannot invoke a persisted task through a disabled feature."""

    dispatched = AsyncMock(return_value="must-not-run")
    agent = SimpleNamespace(
        did="agent-1",
        features={
            "SchedulerFeature": SimpleNamespace(
                enabled=False,
                _dispatch_scheduled_task=dispatched,
            )
        },
    )
    executor = HostedSchedulerExecutor(AsyncMock(return_value=agent))
    execution = SchedulerExecution(
        id="execution-disabled-feature",
        schedule_id="task-disabled-feature",
        agent_id="agent-1",
        task_name="custom_tool",
        args={"value": "must-not-effect"},
        scheduled_for="2026-07-25T00:00:00+00:00",
        idempotency_key="disabled-feature-effect",
        attempt=1,
        owner="host",
    )

    with pytest.raises(RuntimeError, match="no enabled SchedulerFeature"):
        await executor.execute_scheduled(execution)
    dispatched.assert_not_awaited()


@pytest.mark.asyncio
async def test_host_runner_skips_claim_for_soft_disabled_scheduler_feature(tmp_path):
    """A disabled warm host feature leaves its due custom-tool row unclaimed."""

    db = await _database(tmp_path / "host-disabled-scheduler-feature.db")
    agent_id = "did:scheduler:disabled-feature"
    dispatched = AsyncMock(return_value="must-not-run")
    feature = SimpleNamespace(
        enabled=False,
        _dispatch_scheduled_task=dispatched,
    )
    manager = AgentManager()
    manager._agents["Disabled"] = SimpleNamespace(
        did=agent_id,
        features={"SchedulerFeature": feature},
    )
    manager._agent_names[agent_id] = "Disabled"
    manager._seed_scheduler_authority(
        {agent_id: ("Disabled", LocalAgentConfig(data_dir="disabled", port=8801))}
    )
    runner = SchedulerRunner(
        db,
        None,
        AgentManagerHostedSchedulerExecutor(manager),
        authorized_agent_ids={agent_id},
        is_agent_authorized=manager.is_scheduler_agent_authorized,
        owner_id="host-disabled-scheduler-feature",
    )
    try:
        await runner._ensure_tables()
        await _seed_due(
            db,
            task_id="disabled-custom-tool",
            agent_id=agent_id,
            task_name="custom_tool",
        )

        await runner._tick()

        dispatched.assert_not_awaited()
        assert await db.fetchone(
            """
            SELECT enabled, scheduler_claim_fenced, claim_token
            FROM scheduled_tasks WHERE id = ?
            """,
            ("disabled-custom-tool",),
        ) == (1, 0, None)
        assert await db.fetchone(
            "SELECT COUNT(*) FROM task_execution_log WHERE task_id = ?",
            ("disabled-custom-tool",),
        ) == (0,)
    finally:
        await db.close()


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
async def test_prepared_agent_manager_cold_load_precedes_admission_and_holds_delete_lock(
    tmp_path,
):
    """Cold schema bootstrap never runs under admission, and DELETE waits.

    The load deliberately executes additive DDL. If the runner took its
    PostgreSQL-style admission transaction before preparation, that operation
    would need an ``ACCESS EXCLUSIVE`` table lock while its own token read held
    ``ACCESS SHARE``. The order probe makes the regression deterministic on
    SQLite too; the SQL itself remains valid on PostgreSQL.
    """

    db = await _database(tmp_path / "prepared-cold-load.db")
    lifecycle_lock = asyncio.Lock()
    authority = {"active": True}
    preparation_complete = asyncio.Event()
    admission_entered = asyncio.Event()
    dispatch_started = asyncio.Event()
    allow_dispatch_finish = asyncio.Event()
    config = object()

    async def dispatch(_task_name, _args):
        assert lifecycle_lock.locked()
        dispatch_started.set()
        await allow_dispatch_finish.wait()
        return "dispatched"

    cold_agent = SimpleNamespace(
        did="agent-1",
        features={
            "SchedulerFeature": SimpleNamespace(
                _dispatch_scheduled_task=dispatch,
            )
        },
    )

    async def load_agent(name, loaded_config, **kwargs):
        assert name == "Cold"
        assert loaded_config is config
        assert kwargs == {
            "expected_agent_id": "agent-1",
            "scheduler_lifecycle_lock_held": True,
        }
        assert lifecycle_lock.locked()
        await db.execute(
            "ALTER TABLE scheduled_tasks ADD COLUMN cold_bootstrap_marker TEXT"
        )
        preparation_complete.set()
        return cold_agent

    manager = SimpleNamespace(
        list_agents=lambda: {},
        load_agent=load_agent,
        scheduler_lifecycle_lock=lambda _agent_id: lifecycle_lock,
        scheduler_authority_for=lambda _agent_id: (
            ("Cold", config) if authority["active"] else None
        ),
    )

    class AdmissionProbeRunner(SchedulerRunner):
        @asynccontextmanager
        async def _active_dispatch_admission(self, task):
            assert preparation_complete.is_set()
            admission_entered.set()
            async with super()._active_dispatch_admission(task) as admitted:
                yield admitted

    runner = AdmissionProbeRunner(
        db,
        "agent-1",
        AgentManagerHostedSchedulerExecutor(manager, {"agent-1": ("Cold", config)}),
        is_agent_authorized=lambda _agent_id: authority["active"],
        owner_id="prepared-cold-load",
    )

    async def delete_agent():
        async with lifecycle_lock:
            authority["active"] = False

    tick: asyncio.Task[None] | None = None
    deletion: asyncio.Task[None] | None = None
    try:
        await runner._ensure_tables()
        await _seed_due(db)
        tick = asyncio.create_task(runner._tick())
        await asyncio.wait_for(admission_entered.wait(), timeout=1)
        await asyncio.wait_for(dispatch_started.wait(), timeout=1)
        assert await db.fetchone(
            "SELECT 1 FROM pragma_table_info('scheduled_tasks') WHERE name = ?",
            ("cold_bootstrap_marker",),
        ) == (1,)

        deletion = asyncio.create_task(delete_agent())
        await asyncio.sleep(0)
        assert not deletion.done()
        assert authority["active"] is True

        allow_dispatch_finish.set()
        await asyncio.wait_for(tick, timeout=1)
        assert await asyncio.wait_for(deletion, timeout=1) is None
        assert authority["active"] is False
        assert not lifecycle_lock.locked()
    finally:
        allow_dispatch_finish.set()
        for owned in (tick, deletion):
            if owned is not None and not owned.done():
                owned.cancel()
        await asyncio.gather(
            *(owned for owned in (tick, deletion) if owned is not None),
            return_exceptions=True,
        )
        await db.close()


@pytest.mark.asyncio
async def test_host_executor_rejects_cold_agent_did_mismatch_before_dispatch():
    """A nonconforming manager cannot cross-route a claimed tenant DID."""

    dispatched = AsyncMock(return_value="must-not-run")
    wrong_agent = SimpleNamespace(
        did="agent-b",
        features={"SchedulerFeature": SimpleNamespace(_dispatch_scheduled_task=dispatched)},
    )
    manager = SimpleNamespace(
        list_agents=lambda: {}, load_agent=AsyncMock(return_value=wrong_agent)
    )
    executor = AgentManagerHostedSchedulerExecutor(manager, {"agent-a": ("Cold", object())})
    execution = SchedulerExecution(
        id="execution-1", schedule_id="task-1", agent_id="agent-a", task_name="test_task",
        args={}, scheduled_for="2026-07-24T00:00:00+00:00",
        idempotency_key="effect-1", attempt=1, owner="host",
    )

    with pytest.raises(RuntimeError, match="does not match"):
        await executor.execute_scheduled(execution)

    manager.load_agent.assert_awaited_once_with(
        "Cold",
        ANY,
        expected_agent_id="agent-a",
        scheduler_lifecycle_lock_held=True,
    )
    dispatched.assert_not_awaited()


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

    runner = SchedulerRunner(
        db, None, MinimalHostedExecutor(), authorized_agent_ids={"agent-1"}
    )
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
async def test_host_runner_cannot_claim_or_advance_foreign_fleet_rows(tmp_path):
    """Host authority scopes due selection and claim before a lease exists."""
    db = await _database(tmp_path / "scheduler.db")
    executor = AsyncMock(return_value="local delivery")
    runner = SchedulerRunner(
        db, None, executor, authorized_agent_ids={"agent-1"}, owner_id="host-a"
    )
    try:
        await runner._ensure_tables()
        local_due = await _seed_due(db, task_id="local-task", agent_id="agent-1")
        foreign_due = await _seed_due(
            db, task_id="foreign-task", agent_id="did:foreign:fleet"
        )

        await runner._tick()

        executor.assert_awaited_once_with("test_task", {})
        foreign = await db.fetchone(
            """
            SELECT enabled, last_run_at, next_run_at, lease_owner,
                   lease_expires_at, claim_token, claim_execution_id,
                   attempt_count, terminal_status, terminal_at
            FROM scheduled_tasks WHERE id = ?
            """,
            ("foreign-task",),
        )
        assert foreign == (1, None, foreign_due, None, None, None, None, 0, None, None)
        assert await db.fetchall(
            "SELECT status FROM task_execution_log WHERE task_id = ?",
            ("foreign-task",),
        ) == []

        # The next local occurrence is in the future, and the foreign row is
        # still due. A second host tick must neither redeliver local work nor
        # terminalize the row belonging to another fleet.
        await runner._tick()
        executor.assert_awaited_once()
        assert await db.fetchone(
            "SELECT next_run_at FROM scheduled_tasks WHERE id = ?",
            ("foreign-task",),
        ) == (foreign_due,)
        assert local_due < datetime.now(timezone.utc).isoformat()
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_renewal_exception_before_effect_fails_closed(tmp_path, caplog):
    """A renewal failure observed before dispatch never invokes the effect."""
    db = await _database(tmp_path / "scheduler.db")
    renewal_started = asyncio.Event()
    executor = AsyncMock()

    async def fail_renewal(_task):
        renewal_started.set()
        raise RuntimeError("renewal backend unavailable")

    runner = SchedulerRunner(db, "agent-1", executor, owner_id="host-a")
    runner._renew_lease = fail_renewal
    try:
        await runner._ensure_tables()
        await _seed_due(db)

        with caplog.at_level("ERROR", logger="kestrel_sovereign.features.scheduler.runner"):
            await runner._tick()

        await asyncio.wait_for(renewal_started.wait(), timeout=1)
        executor.assert_not_awaited()
        assert await db.fetchall(
            "SELECT status, attempt_count FROM task_execution_log WHERE task_id = ?",
            ("task-1",),
        ) == [("claimed", 1)]
        assert any("lease renewal failed" in record.getMessage() for record in caplog.records)

        await runner._tick()
        executor.assert_not_awaited()
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_renewal_loss_during_preparation_never_enters_effect(tmp_path):
    """The renewal monitor runs during cold preparation and gates admission."""

    db = await _database(tmp_path / "renewal-loss-preparation.db")
    preparation_started = asyncio.Event()
    allow_preparation_finish = asyncio.Event()
    renewal_started = asyncio.Event()
    lose_lease = asyncio.Event()
    dispatched = AsyncMock(return_value="must-not-run")

    class PreparedExecutor:
        @asynccontextmanager
        async def prepare_scheduled(self, _execution):
            preparation_started.set()
            await allow_preparation_finish.wait()

            async def dispatch():
                return await dispatched()

            yield dispatch

    async def renewal_until_lost(_task):
        renewal_started.set()
        await lose_lease.wait()

    runner = SchedulerRunner(
        db, "agent-1", PreparedExecutor(), owner_id="renewal-preparation"
    )
    runner._renew_lease = renewal_until_lost
    tick: asyncio.Task[None] | None = None
    try:
        await runner._ensure_tables()
        await _seed_due(db)
        tick = asyncio.create_task(runner._tick())
        await asyncio.wait_for(preparation_started.wait(), timeout=1)
        await asyncio.wait_for(renewal_started.wait(), timeout=1)

        lose_lease.set()
        allow_preparation_finish.set()
        await asyncio.wait_for(tick, timeout=1)

        dispatched.assert_not_awaited()
        assert await db.fetchall(
            "SELECT status, attempt_count FROM task_execution_log WHERE task_id = ?",
            ("task-1",),
        ) == [("claimed", 1)]
    finally:
        allow_preparation_finish.set()
        if tick is not None and not tick.done():
            tick.cancel()
            await asyncio.gather(tick, return_exceptions=True)
        await db.close()


@pytest.mark.asyncio
async def test_renewal_loss_during_effect_cancels_owned_work_without_finalizing(tmp_path):
    """An expired/revoked owner cannot publish a terminal result after effect."""

    db = await _database(tmp_path / "renewal-loss-effect.db")
    renewal_started = asyncio.Event()
    lose_lease = asyncio.Event()
    effect_started = asyncio.Event()
    effect_cancelled = asyncio.Event()
    execution_ids = []

    async def renewal_until_lost(_task):
        renewal_started.set()
        await lose_lease.wait()

    async def long_effect(_task_name, _args):
        execution = get_current_scheduler_execution()
        assert execution is not None
        execution_ids.append(execution.id)
        effect_started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            # The runner revokes this scope only after it has joined the owned
            # effect task, so cancellation cannot strand a trusted context.
            assert get_current_scheduler_execution() is execution
            effect_cancelled.set()
            raise

    runner = SchedulerRunner(
        db, "agent-1", long_effect, owner_id="renewal-effect"
    )
    runner._renew_lease = renewal_until_lost
    tick: asyncio.Task[None] | None = None
    try:
        await runner._ensure_tables()
        await _seed_due(db)
        tick = asyncio.create_task(runner._tick())
        await asyncio.wait_for(effect_started.wait(), timeout=1)
        await asyncio.wait_for(renewal_started.wait(), timeout=1)

        lose_lease.set()
        await asyncio.wait_for(effect_cancelled.wait(), timeout=1)
        await asyncio.wait_for(tick, timeout=1)

        assert execution_ids
        assert get_current_scheduler_execution() is None
        assert await db.fetchall(
            "SELECT status, attempt_count FROM task_execution_log WHERE task_id = ?",
            ("task-1",),
        ) == [("claimed", 1)]
    finally:
        if tick is not None and not tick.done():
            tick.cancel()
            await asyncio.gather(tick, return_exceptions=True)
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
        assert runner._authorized_agent_ids == ("agent-1",)
        # Stop the background loop so this test alone drives the due occurrence.
        await runner.stop()
        await _seed_due(db)

        await runner._tick()

        manager.load_agent.assert_awaited_once_with(
            "Cold",
            ANY,
            expected_agent_id="agent-1",
            scheduler_lifecycle_lock_held=True,
        )
        dispatched.assert_awaited_once_with("test_task", {})
        assert storage_instances[0].closed is False
    finally:
        await server._shutdown_host_scheduler(app)


@pytest.mark.asyncio
async def test_empty_shared_postgres_host_starts_for_first_runtime_tenant(
    monkeypatch,
    tmp_path,
):
    """An empty configured fleet still exposes the dynamic registration seam."""

    db = await _database(tmp_path / "empty-host-scheduler.db")

    class HostStorage:
        def __init__(self, *, backend, dsn):
            assert backend == "postgres"
            self.db = db

        async def initialize(self):
            return None

        async def close(self):
            await db.close()

    manager = AgentManager(base_data_dir=tmp_path)
    manager.set_scheduler_polling_managed_by_host(True)
    app = FastAPI()
    app.state.agent_manager = manager
    monkeypatch.setenv("KESTREL_DB_BACKEND", "postgres")
    monkeypatch.setenv("KESTREL_DATABASE_URL", "postgresql://scheduler-test")
    monkeypatch.setattr(
        "kestrel_sovereign.storage.async_storage.AsyncStorage",
        HostStorage,
    )

    try:
        await server._start_host_scheduler(
            app,
            manager,
            MultiAgentConfig(agents={}),
        )
        runner = app.state.host_scheduler_runner
        assert runner is not None
        assert await runner._current_authorized_agent_ids() == ()
        assert manager._scheduler_tenant_registration_hook is not None
    finally:
        await server._shutdown_host_scheduler(app)


@pytest.mark.asyncio
async def test_host_managed_feature_cannot_claim_while_remove_revokes_authority(
    tmp_path,
):
    """A shared-PG tenant has no stale scoped poller during DELETE."""

    db = await _database(tmp_path / "host-managed-removal.db")
    agent_id = "did:scheduler:host-managed-removal"
    await SchedulerRunner(
        db,
        agent_id,
        AsyncMock(),
        owner_id="host-managed-schema",
    )._ensure_tables()

    shutdown_started = asyncio.Event()
    release_shutdown = asyncio.Event()

    async def shutdown():
        shutdown_started.set()
        await release_shutdown.wait()

    agent = SimpleNamespace(
        did=agent_id,
        agent_id=agent_id,
        features={},
        storage=SimpleNamespace(db=db),
        shutdown=shutdown,
        _scheduler_polling_managed_by_host=True,
    )
    feature = SchedulerFeature(agent)
    manager = AgentManager()
    config = LocalAgentConfig(data_dir="managed", port=8801)
    removal: asyncio.Task[bool] | None = None
    try:
        await feature.initialize()
        agent.features["SchedulerFeature"] = feature
        assert feature._runner is None
        await _seed_due(
            db,
            task_id="host-managed-due",
            agent_id=agent_id,
        )

        manager._agents["Managed"] = agent
        manager._agent_names[agent_id] = "Managed"
        manager._seed_scheduler_authority(
            {agent_id: ("Managed", config)}
        )
        removal = asyncio.create_task(manager.remove_agent("Managed"))
        await asyncio.wait_for(shutdown_started.wait(), timeout=1)
        assert not manager.is_scheduler_agent_authorized(agent_id)

        # The ready hook is harmless even when it races removal: there is no
        # scoped runner capable of claiming from stale per-agent authority.
        await feature.on_agent_ready(agent)
        await asyncio.sleep(0.05)
        assert await db.fetchone(
            """
            SELECT enabled, lease_owner FROM scheduled_tasks
            WHERE id = 'host-managed-due'
            """
        ) == (1, None)
        assert await db.fetchone(
            """
            SELECT COUNT(*) FROM task_execution_log
            WHERE task_id = 'host-managed-due'
            """
        ) == (0,)

        release_shutdown.set()
        assert await removal is True
    finally:
        release_shutdown.set()
        if removal is not None and not removal.done():
            removal.cancel()
            await asyncio.gather(removal, return_exceptions=True)
        await db.close()
