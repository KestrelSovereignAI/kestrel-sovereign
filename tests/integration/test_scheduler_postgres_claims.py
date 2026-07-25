"""Scheduler claim race on the production PostgreSQL backend.

The unit suite proves the same predicate with independent SQLite connections.
This integration test runs under ``db_backend`` in CI where PostgreSQL is
available, so two runner transactions use separate PostgreSQL connections and
the database re-evaluates the claim predicate under its row lock.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from uuid import uuid4

import pytest

from kestrel_sovereign.features.scheduler.runner import (
    AgentManagerHostedSchedulerExecutor,
    SCHEDULER_PROTOCOL_VERSION,
    SCHEDULER_ROLLOUT_ACK_ENV,
    SchedulerRolloutQuiescenceRequired,
    SchedulerRunner,
    ScheduledTask,
    get_current_scheduler_execution,
)
from kestrel_sovereign.features.scheduler.feature import SchedulerFeature
from kestrel_sovereign.storage.async_database import AsyncDatabase


@asynccontextmanager
async def _single_connection_scheduler_database(db_backend):
    """Yield a real PostgreSQL scheduler DB with one operational connection.

    The scheduler's advisory gates must queue independently of this pool: an
    admitted effect renews/finalizes through the one query connection while a
    fence waits on the matching per-DID gate. This deliberately exercises the
    smallest supported operational-pool topology.
    """

    from kestrel_sovereign.storage.db.postgres import PostgresBackend

    backend = PostgresBackend(
        db_backend._dsn,
        min_pool_size=1,
        max_pool_size=1,
    )
    await backend.connect()
    try:
        yield AsyncDatabase(backend)
    finally:
        await backend.close()


async def _activate_protocol_for_test_agent(db, agent_id, monkeypatch) -> None:
    """Bring a unique test DID through the real rollout control transition."""

    async def noop(_task_name, _args):
        return None

    bootstrap = SchedulerRunner(db, agent_id, noop, owner_id="rollout-bootstrap")
    try:
        await bootstrap._ensure_tables()
    except SchedulerRolloutQuiescenceRequired:
        nonce = await db.fetchval(
            "SELECT activation_nonce FROM scheduler_protocol_rollout WHERE agent_id = ?",
            (agent_id,),
        )
        assert nonce
        # The constructor deliberately captures the acknowledgement to model a
        # process restart; create a fresh candidate after the durable nonce is
        # observed rather than mutating runner internals in this backend test.
        monkeypatch.setenv(SCHEDULER_ROLLOUT_ACK_ENV, nonce)
        acknowledged = SchedulerRunner(
            db, agent_id, noop, owner_id="rollout-acknowledged"
        )
        await acknowledged._ensure_tables()


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_two_replicas_claim_one_due_occurrence_on_backend(db_backend, monkeypatch):
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

    try:
        await _activate_protocol_for_test_agent(db, agent_id, monkeypatch)
        first = SchedulerRunner(db, agent_id, executor, owner_id="replica-a")
        second = SchedulerRunner(db, agent_id, executor, owner_id="replica-b")
        await first._ensure_tables()
        await second._ensure_tables()
        due = (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat()
        await db.execute(
            """
            INSERT INTO scheduled_tasks
                (id, agent_id, task_name, cron_expression, args_json, enabled,
                 next_run_at, created_at, idempotency_key,
                 scheduler_protocol_version)
            VALUES (?, ?, 'task', '* * * * *', '{}', 1, ?, ?, 'integration-effect', ?)
            """,
            (task_id, agent_id, due, due, SCHEDULER_PROTOCOL_VERSION),
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
        await db.execute("DELETE FROM scheduler_protocol_rollout WHERE agent_id = ?", (agent_id,))


@pytest.mark.asyncio
@pytest.mark.dual_backend
@pytest.mark.parametrize(
    (
        "schedule_kind",
        "cron_expression",
        "expected_enabled",
        "expected_terminal_status",
        "expected_execution_status",
    ),
    [
        ("cron", "* * * * *", 1, None, "success"),
        ("one_shot", "* * * * *", 0, "success", "success"),
        ("cron", "not-a-cron", 0, "invalid_cron", "failed"),
    ],
)
async def test_finalization_terminal_timestamp_condition_is_backend_portable(
    db_backend,
    monkeypatch,
    schedule_kind,
    cron_expression,
    expected_enabled,
    expected_terminal_status,
    expected_execution_status,
):
    """Finalization accepts both branches of its terminal timestamp condition.

    PostgreSQL requires a real boolean for ``CASE WHEN $N`` while SQLite
    accepts the same value through its integer boolean representation. Exercise
    recurring, terminal, and invalid-cron schedules against both production
    backends.
    """

    db = AsyncDatabase(db_backend)
    agent_id = f"scheduler-finalize:{uuid4()}"
    task_id = f"scheduler-finalize-task:{uuid4()}"

    async def executor(_task_name, _args):
        return "ok"

    try:
        await _activate_protocol_for_test_agent(db, agent_id, monkeypatch)
        runner = SchedulerRunner(db, agent_id, executor, owner_id="finalize-owner")
        await runner._ensure_tables()
        due = (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat()
        await db.execute(
            """
            INSERT INTO scheduled_tasks
                (id, agent_id, task_name, cron_expression, args_json, enabled,
                 next_run_at, created_at, schedule_kind, run_at,
                 idempotency_key, scheduler_protocol_version)
            VALUES (?, ?, 'task', ?, '{}', 1, ?, ?, ?, ?,
                    'finalize-terminal', ?)
            """,
            (
                task_id,
                agent_id,
                cron_expression,
                due,
                due,
                schedule_kind,
                due if schedule_kind == "one_shot" else None,
                SCHEDULER_PROTOCOL_VERSION,
            ),
        )

        await runner._tick()

        schedule_row = await db.fetchone(
            """
            SELECT enabled, terminal_status, terminal_at, lease_owner,
                   lease_expires_at, claim_token, claim_execution_id,
                   claim_scheduled_for
            FROM scheduled_tasks WHERE id = ?
            """,
            (task_id,),
        )
        assert schedule_row is not None
        assert schedule_row[0:2] == (expected_enabled, expected_terminal_status)
        assert (schedule_row[2] is not None) is (expected_terminal_status is not None)
        assert schedule_row[3:] == (None, None, None, None, None)
        assert await db.fetchall(
            "SELECT status FROM task_execution_log WHERE task_id = ?",
            (task_id,),
        ) == [(expected_execution_status,)]
    finally:
        await db.execute("DELETE FROM task_execution_log WHERE task_id = ?", (task_id,))
        await db.execute("DELETE FROM scheduled_tasks WHERE id = ?", (task_id,))
        await db.execute("DELETE FROM scheduler_protocol_rollout WHERE agent_id = ?", (agent_id,))


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_scheduler_timestamp_predicates_are_backend_portable(db_backend):
    """Legacy and explicit timestamp forms represent the same past instant."""

    db = AsyncDatabase(db_backend)

    async def noop(_task_name, _args):
        return None

    runner = SchedulerRunner(db, "scheduler-timestamp-test", noop)
    candidate = "candidate"
    predicates = f"""
        SELECT {runner._database_due_sql(candidate)},
               {runner._database_lease_expired_sql(candidate)},
               {runner._database_lease_live_sql(candidate)}
        FROM (SELECT CAST(? AS TEXT) AS {candidate}) AS timestamp_value
    """
    past = datetime.now(timezone.utc) - timedelta(minutes=2)
    timestamp_forms = (
        # SQLite treats its historic space separator as UTC. PostgreSQL must
        # not reinterpret it through a connection-level TimeZone.
        past.strftime("%Y-%m-%d %H:%M:%S"),
        past.isoformat().replace("+00:00", "Z"),
        past.astimezone(timezone(timedelta(hours=2))).isoformat(),
    )

    async def assert_past_predicates() -> None:
        for timestamp in timestamp_forms:
            row = await db.fetchone(predicates, (timestamp,))
            assert row is not None
            assert tuple(bool(value) for value in row) == (True, True, False)

    if db_backend.backend_type == "postgres":
        async with db.transaction():
            await db.execute("SET LOCAL TIME ZONE 'America/Chicago'")
            await assert_past_predicates()
    else:
        await assert_past_predicates()


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_recovery_claim_upsert_qualifies_existing_execution_log_columns(
    db_backend, monkeypatch
):
    """A recovery updates its existing ``claimed`` log on PostgreSQL.

    This intentionally forces the ``ON CONFLICT(id) DO UPDATE`` branch in
    :meth:`SchedulerRunner._claim`. PostgreSQL exposes ambiguous target versus
    ``excluded`` references here while SQLite accepts the former unqualified
    spelling, so retain this as a real dual-backend regression.
    """

    db = AsyncDatabase(db_backend)
    agent_id = f"scheduler-recovery:{uuid4()}"
    task_id = f"scheduler-recovery-task:{uuid4()}"
    execution_id = f"scheduler-recovery-execution:{uuid4()}"
    seen = []

    async def executor(_task_name, _args):
        seen.append(get_current_scheduler_execution())
        return "recovered"

    try:
        await _activate_protocol_for_test_agent(db, agent_id, monkeypatch)
        runner = SchedulerRunner(db, agent_id, executor, owner_id="recovery-owner")
        await runner._ensure_tables()
        due = (datetime.now(timezone.utc) - timedelta(seconds=2)).isoformat()
        expired = (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat()
        base_key = "integration-recovery"
        occurrence_key = SchedulerRunner._occurrence_idempotency_key(
            base_key, task_id, due
        )
        await db.execute(
            """
            INSERT INTO scheduled_tasks
                (id, agent_id, task_name, cron_expression, args_json, enabled,
                 next_run_at, created_at, idempotency_key,
                 scheduler_protocol_version, scheduler_claim_fenced,
                 lease_owner, lease_expires_at, claim_token,
                 claim_execution_id, claim_scheduled_for, attempt_count)
            VALUES (?, ?, 'task', '* * * * *', '{}', 0, ?, ?, ?, ?, 1,
                    'dead-owner', ?, 'dead-token', ?, ?, 1)
            """,
            (
                task_id,
                agent_id,
                due,
                due,
                base_key,
                SCHEDULER_PROTOCOL_VERSION,
                expired,
                execution_id,
                due,
            ),
        )
        await db.execute(
            """
            INSERT INTO task_execution_log
                (id, task_id, agent_id, status, result_text, duration_ms,
                 executed_at, occurrence_at, idempotency_key, attempt_count,
                 claimed_at)
            VALUES (?, ?, ?, 'claimed', NULL, 0, ?, ?, ?, 1, ?)
            """,
            (execution_id, task_id, agent_id, due, due, occurrence_key, due),
        )

        await runner._tick()

        assert len(seen) == 1
        assert seen[0].id == execution_id
        assert seen[0].attempt == 2
        assert await db.fetchone(
            "SELECT status, attempt_count FROM task_execution_log WHERE id = ?",
            (execution_id,),
        ) == ("success", 2)
    finally:
        await db.execute("DELETE FROM task_execution_log WHERE task_id = ?", (task_id,))
        await db.execute("DELETE FROM scheduled_tasks WHERE id = ?", (task_id,))
        await db.execute("DELETE FROM scheduler_protocol_rollout WHERE agent_id = ?", (agent_id,))


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_stale_due_snapshot_reuses_claim_published_before_row_lock(
    db_backend, monkeypatch,
):
    """The real backend keeps one execution identity across stale recovery.

    This exercises the actual lock/reread ordering on PostgreSQL as well as
    SQLite: the worker's due result has no claim, but the durable row gains an
    expired claim before that worker enters ``_claim``.
    """

    db = AsyncDatabase(db_backend)
    agent_id = f"scheduler-stale-recovery:{uuid4()}"
    task_id = f"scheduler-stale-recovery-task:{uuid4()}"
    execution_id = f"scheduler-stale-recovery-execution:{uuid4()}"
    seen = []

    async def executor(_task_name, _args):
        seen.append(get_current_scheduler_execution())
        return "recovered"

    try:
        await _activate_protocol_for_test_agent(db, agent_id, monkeypatch)
        runner = SchedulerRunner(db, agent_id, executor, owner_id="stale-owner")
        await runner._ensure_tables()
        due = (datetime.now(timezone.utc) - timedelta(seconds=2)).isoformat()
        expired = (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat()
        base_key = "integration-stale-recovery"
        occurrence_key = SchedulerRunner._occurrence_idempotency_key(
            base_key, task_id, due,
        )
        await db.execute(
            """
            INSERT INTO scheduled_tasks
                (id, agent_id, task_name, cron_expression, args_json, enabled,
                 next_run_at, created_at, idempotency_key,
                 scheduler_protocol_version)
            VALUES (?, ?, 'task', '* * * * *', '{}', 1, ?, ?, ?, ?)
            """,
            (task_id, agent_id, due, due, base_key, SCHEDULER_PROTOCOL_VERSION),
        )
        stale_rows = await runner._due_rows(datetime.now(timezone.utc))
        stale_task = ScheduledTask.from_row(stale_rows[0])
        assert stale_task.claim_execution_id is None

        await db.execute(
            """
            UPDATE scheduled_tasks
            SET enabled = 0, scheduler_claim_fenced = 1,
                lease_owner = 'dead-owner', lease_expires_at = ?,
                claim_token = 'dead-token', claim_execution_id = ?,
                claim_scheduled_for = ?, attempt_count = 1
            WHERE id = ?
            """,
            (expired, execution_id, due, task_id),
        )
        await db.execute(
            """
            INSERT INTO task_execution_log
                (id, task_id, agent_id, status, result_text, duration_ms,
                 executed_at, occurrence_at, idempotency_key, attempt_count,
                 claimed_at)
            VALUES (?, ?, ?, 'claimed', NULL, 0, ?, ?, ?, 1, ?)
            """,
            (execution_id, task_id, agent_id, due, due, occurrence_key, due),
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
            (task_id,),
        ) == (1,)
    finally:
        await db.execute("DELETE FROM task_execution_log WHERE task_id = ?", (task_id,))
        await db.execute("DELETE FROM scheduled_tasks WHERE id = ?", (task_id,))
        await db.execute("DELETE FROM scheduler_protocol_rollout WHERE agent_id = ?", (agent_id,))


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_postgres_claim_uses_statement_time_after_real_row_lock_stall(
    db_backend, monkeypatch
):
    """A blocked claimant cannot publish a lease based on an old host clock.

    This deliberately holds the exact ``scheduled_tasks`` row with a separate
    real PostgreSQL connection for longer than a one-second lease.  The claim
    is also passed a deliberately ancient poll timestamp.  Its persisted lease
    must still begin after the lock is released according to PostgreSQL's own
    wall clock.
    """

    if db_backend.backend_type != "postgres":
        pytest.skip("requires PostgreSQL row-level lock semantics")
    db = AsyncDatabase(db_backend)
    agent_id = f"scheduler-clock-claim:{uuid4()}"
    task_id = f"scheduler-clock-claim-task:{uuid4()}"

    async def noop(_task_name, _args):
        return "ok"

    try:
        await _activate_protocol_for_test_agent(db, agent_id, monkeypatch)
        runner = SchedulerRunner(
            db, agent_id, noop, owner_id="clock-claim", lease_seconds=1
        )
        await runner._ensure_tables()
        due = (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat()
        await db.execute(
            """
            INSERT INTO scheduled_tasks
                (id, agent_id, task_name, cron_expression, args_json, enabled,
                 next_run_at, created_at, idempotency_key,
                 scheduler_protocol_version)
            VALUES (?, ?, 'task', '* * * * *', '{}', 1, ?, ?, 'clock-claim', ?)
            """,
            (task_id, agent_id, due, due, SCHEDULER_PROTOCOL_VERSION),
        )
        task = ScheduledTask.from_row((await runner._due_rows(datetime.now(timezone.utc)))[0])
        pool = db_backend._pool
        assert pool is not None
        async with pool.acquire() as locked_connection:
            async with locked_connection.transaction():
                await locked_connection.execute(
                    "SELECT id FROM scheduled_tasks WHERE id = $1 AND agent_id = $2 FOR UPDATE",
                    task_id,
                    agent_id,
                )
                claim_task = asyncio.create_task(
                    runner._claim(task, datetime(1970, 1, 1, tzinfo=timezone.utc))
                )
                await asyncio.sleep(1.1)
                assert not claim_task.done()

        claimed = await asyncio.wait_for(claim_task, timeout=2)
        assert claimed is not None
        lease_expires_at = await db.fetchval(
            "SELECT lease_expires_at FROM scheduled_tasks WHERE id = ?", (task_id,)
        )
        database_now = await db.fetchval("SELECT clock_timestamp()")
        lease = datetime.fromisoformat(lease_expires_at).astimezone(timezone.utc)
        if database_now.tzinfo is None:
            database_now = database_now.replace(tzinfo=timezone.utc)
        else:
            database_now = database_now.astimezone(timezone.utc)
        assert lease > database_now + timedelta(seconds=0.5)

        await runner._execute_claim(claimed)
    finally:
        await db.execute("DELETE FROM task_execution_log WHERE task_id = ?", (task_id,))
        await db.execute("DELETE FROM scheduled_tasks WHERE id = ?", (task_id,))
        await db.execute("DELETE FROM scheduler_protocol_rollout WHERE agent_id = ?", (agent_id,))


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_postgres_long_executor_renews_while_admission_gate_is_held(
    db_backend, monkeypatch
):
    """A long-running effect renews without retaining the control-row lock."""

    if db_backend.backend_type != "postgres":
        pytest.skip("requires PostgreSQL advisory-lock semantics")

    db = AsyncDatabase(db_backend)
    agent_id = f"scheduler-admission-renew:{uuid4()}"
    task_id = f"scheduler-admission-renew-task:{uuid4()}"
    executor_started = asyncio.Event()
    release_executor = asyncio.Event()
    tick: asyncio.Task[None] | None = None

    async def executor(_task_name, _args):
        executor_started.set()
        await release_executor.wait()
        return "ok"

    try:
        await _activate_protocol_for_test_agent(db, agent_id, monkeypatch)
        runner = SchedulerRunner(
            db, agent_id, executor, owner_id="admission-renew", lease_seconds=1
        )
        await runner._ensure_tables()
        due = (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat()
        await db.execute(
            """
            INSERT INTO scheduled_tasks
                (id, agent_id, task_name, cron_expression, args_json, enabled,
                 next_run_at, created_at, idempotency_key,
                 scheduler_protocol_version)
            VALUES (?, ?, 'task', '* * * * *', '{}', 1, ?, ?,
                    'admission-renew', ?)
            """,
            (task_id, agent_id, due, due, SCHEDULER_PROTOCOL_VERSION),
        )
        tick = asyncio.create_task(runner._tick())
        await asyncio.wait_for(executor_started.wait(), timeout=2)
        initial = await db.fetchval(
            "SELECT lease_expires_at FROM scheduled_tasks WHERE id = ?", (task_id,)
        )
        assert isinstance(initial, str)

        renewed: str | None = None
        for _ in range(30):
            candidate = await db.fetchval(
                "SELECT lease_expires_at FROM scheduled_tasks WHERE id = ?", (task_id,)
            )
            if isinstance(candidate, str) and candidate != initial:
                renewed = candidate
                break
            await asyncio.sleep(0.05)
        assert renewed is not None

        # Unlike the replaced row-lock admission epoch, the control row is
        # available while the effect is running. The per-DID advisory gate,
        # rather than this row, owns the rollout/effect exclusion.
        pool = db_backend._pool
        assert pool is not None
        async with pool.acquire() as contender:
            async with contender.transaction():
                control = await contender.fetchrow(
                    """
                    SELECT agent_id FROM scheduler_protocol_rollout
                    WHERE agent_id = $1 FOR UPDATE NOWAIT
                    """,
                    agent_id,
                )
        assert control is not None

        release_executor.set()
        await asyncio.wait_for(tick, timeout=3)
        assert await db.fetchone(
            "SELECT status FROM task_execution_log WHERE task_id = ?", (task_id,)
        ) == ("success",)
    finally:
        release_executor.set()
        if tick is not None and not tick.done():
            tick.cancel()
            await asyncio.gather(tick, return_exceptions=True)
        await db.execute("DELETE FROM task_execution_log WHERE task_id = ?", (task_id,))
        await db.execute("DELETE FROM scheduled_tasks WHERE id = ?", (task_id,))
        await db.execute("DELETE FROM scheduler_protocol_rollout WHERE agent_id = ?", (agent_id,))


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_postgres_scheduled_mutator_does_not_deadlock_rollout_admission(
    db_backend, monkeypatch
):
    """A scheduled pause runs on another PG connection without deadlocking.

    This is the production shape that the prior implementation wedged: the
    runner held ``scheduler_protocol_rollout`` in its parent transaction while
    user code invoked ``SchedulerFeature.schedule_pause`` through the normal
    pooled database path.  The mutation waited on the runner's control-row
    lock while the runner awaited the mutation.  The advisory effect gate now
    preserves rollout/effect exclusion without retaining that row lock.
    """

    if db_backend.backend_type != "postgres":
        pytest.skip("requires PostgreSQL advisory-lock semantics")

    db = AsyncDatabase(db_backend)
    agent_id = f"scheduler-admission-mutate:{uuid4()}"
    task_id = f"scheduler-admission-mutate-task:{uuid4()}"
    tick: asyncio.Task[None] | None = None
    mutation_result = []
    mutation_agent = SimpleNamespace(
        did=agent_id,
        agent_id=agent_id,
        _raw_storage=SimpleNamespace(db=db),
        features={},
    )
    mutation_feature = SchedulerFeature(mutation_agent)
    # This executor is already inside the runner's admitted effect boundary;
    # only the feature's actual scheduler mutator is relevant to the
    # regression, so avoid starting a second feature-owned polling runner.
    mutation_feature._db = db
    mutation_feature._agent_id = agent_id

    async def executor(_task_name, _args):
        mutation_result.append(await mutation_feature.schedule_pause(task_id))
        return "ok"

    try:
        await _activate_protocol_for_test_agent(db, agent_id, monkeypatch)
        runner = SchedulerRunner(
            db, agent_id, executor, owner_id="admission-renew", lease_seconds=1
        )
        await runner._ensure_tables()
        due = (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat()
        await db.execute(
            """
            INSERT INTO scheduled_tasks
                (id, agent_id, task_name, cron_expression, args_json, enabled,
                 next_run_at, created_at, idempotency_key,
                 scheduler_protocol_version)
            VALUES (?, ?, 'task', '* * * * *', '{}', 1, ?, ?,
                    'admission-renew', ?)
            """,
            (task_id, agent_id, due, due, SCHEDULER_PROTOCOL_VERSION),
        )
        tick = asyncio.create_task(runner._tick())
        await asyncio.wait_for(tick, timeout=3)
        assert len(mutation_result) == 1
        assert mutation_result[0].data["status"] == "paused"
        assert await db.fetchone(
            "SELECT enabled, claim_token FROM scheduled_tasks WHERE id = ?", (task_id,)
        ) == (False, None)
        assert await db.fetchone(
            "SELECT status FROM task_execution_log WHERE task_id = ?", (task_id,)
        ) == ("cancelled",)
    finally:
        if tick is not None and not tick.done():
            tick.cancel()
            await asyncio.gather(tick, return_exceptions=True)
        await db.execute("DELETE FROM task_execution_log WHERE task_id = ?", (task_id,))
        await db.execute("DELETE FROM scheduled_tasks WHERE id = ?", (task_id,))
        await db.execute("DELETE FROM scheduler_protocol_rollout WHERE agent_id = ?", (agent_id,))


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_postgres_rollout_transition_waits_for_admitted_effect(
    db_backend, monkeypatch
):
    """A fence cannot revoke a claim after its external effect has started.

    The transition observes a newly introduced legacy row while an admitted
    task is paused in its executor.  It must wait at the per-DID advisory gate
    until terminal compare-and-set finalization has completed, rather than
    changing the durable rollout row under the live effect.
    """

    if db_backend.backend_type != "postgres":
        pytest.skip("requires PostgreSQL advisory-lock semantics")

    db = AsyncDatabase(db_backend)
    agent_id = f"scheduler-admission-fence:{uuid4()}"
    task_id = f"scheduler-admission-fence-task:{uuid4()}"
    legacy_task_id = f"scheduler-admission-fence-legacy:{uuid4()}"
    executor_started = asyncio.Event()
    release_executor = asyncio.Event()
    tick: asyncio.Task[None] | None = None
    transition: asyncio.Task[None] | None = None

    async def executor(_task_name, _args):
        executor_started.set()
        await release_executor.wait()
        return "ok"

    async def no_op(_task_name, _args):
        return None

    try:
        await _activate_protocol_for_test_agent(db, agent_id, monkeypatch)
        runner = SchedulerRunner(
            db, agent_id, executor, owner_id="admission-fence-effect"
        )
        fencer = SchedulerRunner(
            db, agent_id, no_op, owner_id="admission-fence-transition"
        )
        await runner._ensure_tables()
        due = (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat()
        await db.execute(
            """
            INSERT INTO scheduled_tasks
                (id, agent_id, task_name, cron_expression, args_json, enabled,
                 next_run_at, created_at, idempotency_key,
                 scheduler_protocol_version)
            VALUES (?, ?, 'task', '* * * * *', '{}', 1, ?, ?,
                    'admission-fence', ?)
            """,
            (task_id, agent_id, due, due, SCHEDULER_PROTOCOL_VERSION),
        )
        tick = asyncio.create_task(runner._tick())
        await asyncio.wait_for(executor_started.wait(), timeout=2)

        # Simulate an origin/main process adding a legacy-visible row after
        # this runner was prepared. An old binary does not know the v2 marker
        # column and therefore cannot erase it from the live v2 claim above;
        # using a separate row models the real mixed-version condition.
        await db.execute(
            """
            INSERT INTO scheduled_tasks
                (id, agent_id, task_name, cron_expression, args_json, enabled,
                 next_run_at, created_at, idempotency_key)
            VALUES (?, ?, 'legacy-task', '* * * * *', '{}', 1, ?, ?,
                    'admission-fence-legacy')
            """,
            (legacy_task_id, agent_id, due, due),
        )

        async def reconcile() -> None:
            with pytest.raises(SchedulerRolloutQuiescenceRequired):
                await fencer._ensure_protocol_rollout(preexisting_schedule_table=True)

        transition = asyncio.create_task(reconcile())
        await asyncio.sleep(0.1)
        assert not transition.done()

        release_executor.set()
        await asyncio.wait_for(tick, timeout=3)
        await asyncio.wait_for(transition, timeout=3)
        assert await db.fetchone(
            "SELECT status FROM task_execution_log WHERE task_id = ?", (task_id,)
        ) == ("success",)
        rollout = await db.fetchone(
            "SELECT state, activation_nonce FROM scheduler_protocol_rollout WHERE agent_id = ?",
            (agent_id,),
        )
        assert rollout is not None
        assert rollout[0] == "quiescing"
    finally:
        release_executor.set()
        for task in (tick, transition):
            if task is not None and not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
        await db.execute("DELETE FROM task_execution_log WHERE task_id = ?", (task_id,))
        await db.execute("DELETE FROM scheduled_tasks WHERE id = ?", (task_id,))
        await db.execute("DELETE FROM scheduler_protocol_rollout WHERE agent_id = ?", (agent_id,))


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_postgres_same_did_effects_share_admission_before_transition(
    db_backend, monkeypatch
):
    """Live PostgreSQL admits sibling effects but drains both for a fence."""

    if db_backend.backend_type != "postgres":
        pytest.skip("requires PostgreSQL shared advisory-lock semantics")

    db = AsyncDatabase(db_backend)
    agent_id = f"scheduler-shared-admission:{uuid4()}"
    task_ids = (
        f"scheduler-shared-admission-a:{uuid4()}",
        f"scheduler-shared-admission-b:{uuid4()}",
    )
    legacy_task_id = f"scheduler-shared-admission-legacy:{uuid4()}"
    effect_ids: list[str] = []
    both_effects_started = asyncio.Event()
    release_effects = asyncio.Event()
    transition_started = asyncio.Event()
    tick: asyncio.Task[None] | None = None
    transition: asyncio.Task[None] | None = None

    async def executor(_task_name, _args):
        effect_ids.append(get_current_scheduler_execution().schedule_id)
        if len(effect_ids) == 2:
            both_effects_started.set()
        await release_effects.wait()
        return "ok"

    async def no_op(_task_name, _args):
        return None

    try:
        await _activate_protocol_for_test_agent(db, agent_id, monkeypatch)
        runner = SchedulerRunner(
            db,
            agent_id,
            executor,
            owner_id="shared-admission-effects",
            max_concurrent_tasks=2,
        )
        fencer = SchedulerRunner(
            db, agent_id, no_op, owner_id="shared-admission-transition"
        )
        await runner._ensure_tables()
        due = (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat()
        for task_id in task_ids:
            await db.execute(
                """
                INSERT INTO scheduled_tasks
                    (id, agent_id, task_name, cron_expression, args_json, enabled,
                     next_run_at, created_at, idempotency_key,
                     scheduler_protocol_version)
                VALUES (?, ?, 'task', '* * * * *', '{}', 1, ?, ?, ?, ?)
                """,
                (
                    task_id,
                    agent_id,
                    due,
                    due,
                    f"shared-admission-{task_id}",
                    SCHEDULER_PROTOCOL_VERSION,
                ),
            )

        tick = asyncio.create_task(runner._tick())
        await asyncio.wait_for(both_effects_started.wait(), timeout=3)
        assert set(effect_ids) == set(task_ids)

        await db.execute(
            """
            INSERT INTO scheduled_tasks
                (id, agent_id, task_name, cron_expression, args_json, enabled,
                 next_run_at, created_at, idempotency_key)
            VALUES (?, ?, 'legacy-task', '* * * * *', '{}', 1, ?, ?,
                    'shared-admission-legacy')
            """,
            (legacy_task_id, agent_id, due, due),
        )

        async def reconcile() -> None:
            transition_started.set()
            with pytest.raises(SchedulerRolloutQuiescenceRequired):
                await fencer._ensure_protocol_rollout(preexisting_schedule_table=True)

        transition = asyncio.create_task(reconcile())
        await asyncio.wait_for(transition_started.wait(), timeout=1)
        await asyncio.sleep(0.1)
        assert not transition.done()

        release_effects.set()
        await asyncio.wait_for(tick, timeout=3)
        await asyncio.wait_for(transition, timeout=3)
        rows = await db.fetchall(
            "SELECT task_id, status FROM task_execution_log WHERE agent_id = ? ORDER BY task_id",
            (agent_id,),
        )
        assert rows == [(task_id, "success") for task_id in sorted(task_ids)]
    finally:
        release_effects.set()
        for task in (tick, transition):
            if task is not None and not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
        await db.execute("DELETE FROM task_execution_log WHERE agent_id = ?", (agent_id,))
        await db.execute("DELETE FROM scheduled_tasks WHERE agent_id = ?", (agent_id,))
        await db.execute(
            "DELETE FROM scheduler_protocol_rollout WHERE agent_id = ?", (agent_id,)
        )


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_postgres_bootstrap_waits_for_admitted_effect(
    db_backend, monkeypatch
):
    """Schema bootstrap drains active effects before it proceeds with DDL."""

    if db_backend.backend_type != "postgres":
        pytest.skip("requires PostgreSQL advisory-lock semantics")

    db = AsyncDatabase(db_backend)
    agent_id = f"scheduler-admission-bootstrap:{uuid4()}"
    task_id = f"scheduler-admission-bootstrap-task:{uuid4()}"
    executor_started = asyncio.Event()
    release_executor = asyncio.Event()
    tick: asyncio.Task[None] | None = None
    bootstrap: asyncio.Task[None] | None = None

    async def executor(_task_name, _args):
        executor_started.set()
        await release_executor.wait()
        return "ok"

    async def no_op(_task_name, _args):
        return None

    try:
        await _activate_protocol_for_test_agent(db, agent_id, monkeypatch)
        runner = SchedulerRunner(
            db, agent_id, executor, owner_id="admission-bootstrap-effect"
        )
        rebootstrap = SchedulerRunner(
            db, agent_id, no_op, owner_id="admission-bootstrap-schema"
        )
        await runner._ensure_tables()
        due = (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat()
        await db.execute(
            """
            INSERT INTO scheduled_tasks
                (id, agent_id, task_name, cron_expression, args_json, enabled,
                 next_run_at, created_at, idempotency_key,
                 scheduler_protocol_version)
            VALUES (?, ?, 'task', '* * * * *', '{}', 1, ?, ?,
                    'admission-bootstrap', ?)
            """,
            (task_id, agent_id, due, due, SCHEDULER_PROTOCOL_VERSION),
        )
        tick = asyncio.create_task(runner._tick())
        await asyncio.wait_for(executor_started.wait(), timeout=2)

        bootstrap = asyncio.create_task(rebootstrap._ensure_tables())
        await asyncio.sleep(0.1)
        assert not bootstrap.done()

        release_executor.set()
        await asyncio.wait_for(tick, timeout=3)
        await asyncio.wait_for(bootstrap, timeout=3)
        assert await db.fetchone(
            "SELECT status FROM task_execution_log WHERE task_id = ?", (task_id,)
        ) == ("success",)
    finally:
        release_executor.set()
        for task in (tick, bootstrap):
            if task is not None and not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
        await db.execute("DELETE FROM task_execution_log WHERE task_id = ?", (task_id,))
        await db.execute("DELETE FROM scheduled_tasks WHERE id = ?", (task_id,))
        await db.execute("DELETE FROM scheduler_protocol_rollout WHERE agent_id = ?", (agent_id,))


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_postgres_single_query_connection_effect_renews_while_fence_waits(
    db_backend,
    monkeypatch,
):
    """A waiting fence never pins the only query connection an effect needs."""

    if db_backend.backend_type != "postgres":
        pytest.skip("requires PostgreSQL advisory-lock semantics")

    agent_id = f"scheduler-single-pool-fence:{uuid4()}"
    task_id = f"scheduler-single-pool-fence-task:{uuid4()}"
    legacy_task_id = f"scheduler-single-pool-fence-legacy:{uuid4()}"
    executor_started = asyncio.Event()
    renewed_during_effect = asyncio.Event()
    release_executor = asyncio.Event()
    tick: asyncio.Task[None] | None = None
    transition: asyncio.Task[None] | None = None

    async def executor(_task_name, _args):
        # This query runs while the effect's advisory gate is held. It would
        # deadlock if that gate consumed the lone operational pool connection.
        assert await db.fetchval("SELECT 1") == 1
        executor_started.set()
        await release_executor.wait()
        return "ok"

    async def no_op(_task_name, _args):
        return None

    async with _single_connection_scheduler_database(db_backend) as db:
        try:
            await _activate_protocol_for_test_agent(db, agent_id, monkeypatch)
            runner = SchedulerRunner(
                db,
                agent_id,
                executor,
                lease_seconds=1,
                owner_id="single-pool-effect",
            )
            fencer = SchedulerRunner(
                db,
                agent_id,
                no_op,
                owner_id="single-pool-fence",
            )
            original_renew = runner._renew_lease_once

            async def observe_renewal(task):
                renewed = await original_renew(task)
                if executor_started.is_set():
                    renewed_during_effect.set()
                return renewed

            runner._renew_lease_once = observe_renewal
            await runner._ensure_tables()
            due = (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat()
            await db.execute(
                """
                INSERT INTO scheduled_tasks
                    (id, agent_id, task_name, cron_expression, args_json, enabled,
                     next_run_at, created_at, idempotency_key,
                     scheduler_protocol_version)
                VALUES (?, ?, 'task', '* * * * *', '{}', 1, ?, ?,
                        'single-pool-fence', ?)
                """,
                (task_id, agent_id, due, due, SCHEDULER_PROTOCOL_VERSION),
            )
            tick = asyncio.create_task(runner._tick())
            await asyncio.wait_for(executor_started.wait(), timeout=3)
            await asyncio.wait_for(renewed_during_effect.wait(), timeout=3)

            # A legacy writer makes the fencer rotate active→quiescing. Its
            # gate wait must happen before it opens the one connection pool.
            await db.execute(
                """
                INSERT INTO scheduled_tasks
                    (id, agent_id, task_name, cron_expression, args_json, enabled,
                     next_run_at, created_at, idempotency_key)
                VALUES (?, ?, 'legacy-task', '* * * * *', '{}', 1, ?, ?,
                        'single-pool-fence-legacy')
                """,
                (legacy_task_id, agent_id, due, due),
            )

            async def reconcile() -> None:
                with pytest.raises(SchedulerRolloutQuiescenceRequired):
                    await fencer._ensure_protocol_rollout(
                        preexisting_schedule_table=True
                    )

            transition = asyncio.create_task(reconcile())
            await asyncio.sleep(0.1)
            assert not transition.done()

            release_executor.set()
            await asyncio.wait_for(tick, timeout=4)
            await asyncio.wait_for(transition, timeout=4)
            assert await db.fetchone(
                "SELECT status FROM task_execution_log WHERE task_id = ?",
                (task_id,),
            ) == ("success",)
            assert await db.fetchone(
                "SELECT state FROM scheduler_protocol_rollout WHERE agent_id = ?",
                (agent_id,),
            ) == ("quiescing",)
        finally:
            release_executor.set()
            for pending in (tick, transition):
                if pending is not None and not pending.done():
                    pending.cancel()
                    await asyncio.gather(pending, return_exceptions=True)
            await db.execute(
                "DELETE FROM task_execution_log WHERE task_id = ?", (task_id,)
            )
            await db.execute(
                "DELETE FROM scheduled_tasks WHERE id IN (?, ?)",
                (task_id, legacy_task_id),
            )
            await db.execute(
                "DELETE FROM scheduler_protocol_rollout WHERE agent_id = ?",
                (agent_id,),
            )


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_postgres_single_query_connection_bootstrap_legacy_fence_does_not_self_wait(
    db_backend,
    monkeypatch,
):
    """Bootstrap owns the session gate before its one-connection transaction."""

    if db_backend.backend_type != "postgres":
        pytest.skip("requires PostgreSQL advisory-lock semantics")

    agent_id = f"scheduler-single-pool-bootstrap:{uuid4()}"
    legacy_task_id = f"scheduler-single-pool-bootstrap-legacy:{uuid4()}"

    async def no_op(_task_name, _args):
        return None

    async with _single_connection_scheduler_database(db_backend) as db:
        try:
            await _activate_protocol_for_test_agent(db, agent_id, monkeypatch)
            due = (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat()
            await db.execute(
                """
                INSERT INTO scheduled_tasks
                    (id, agent_id, task_name, cron_expression, args_json, enabled,
                     next_run_at, created_at, idempotency_key)
                VALUES (?, ?, 'legacy-task', '* * * * *', '{}', 1, ?, ?,
                        'single-pool-bootstrap-legacy')
                """,
                (legacy_task_id, agent_id, due, due),
            )
            bootstrap = SchedulerRunner(
                db,
                agent_id,
                no_op,
                owner_id="single-pool-bootstrap",
            )

            with pytest.raises(SchedulerRolloutQuiescenceRequired):
                await asyncio.wait_for(bootstrap._ensure_tables(), timeout=3)
            assert await db.fetchone(
                "SELECT state FROM scheduler_protocol_rollout WHERE agent_id = ?",
                (agent_id,),
            ) == ("quiescing",)
        finally:
            await db.execute(
                "DELETE FROM task_execution_log WHERE agent_id = ?", (agent_id,)
            )
            await db.execute(
                "DELETE FROM scheduled_tasks WHERE id = ?", (legacy_task_id,)
            )
            await db.execute(
                "DELETE FROM scheduler_protocol_rollout WHERE agent_id = ?",
                (agent_id,),
            )


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_postgres_cancelled_advisory_gate_releases_the_dedicated_session(
    db_backend,
):
    """Cancelling a holder cannot strand its gate or operational pool."""

    if db_backend.backend_type != "postgres":
        pytest.skip("requires PostgreSQL advisory-lock semantics")

    agent_id = f"scheduler-cancelled-advisory:{uuid4()}"
    entered = asyncio.Event()
    hold = asyncio.Event()

    async def no_op(_task_name, _args):
        return None

    async with _single_connection_scheduler_database(db_backend) as db:
        runner = SchedulerRunner(db, agent_id, no_op, owner_id="cancelled-gate")

        async def hold_gate() -> None:
            async with runner._postgres_rollout_effect_gate(agent_id):
                entered.set()
                await hold.wait()

        holder = asyncio.create_task(hold_gate())
        try:
            await asyncio.wait_for(entered.wait(), timeout=3)
            holder.cancel()
            with pytest.raises(asyncio.CancelledError):
                await holder

            # The terminated advisory connection releases the server-side lock
            # and the one operational connection remains independently usable.
            async def reacquire_and_query() -> None:
                async with runner._postgres_rollout_effect_gate(agent_id):
                    assert await db.fetchval("SELECT 1") == 1

            await asyncio.wait_for(reacquire_and_query(), timeout=3)
        finally:
            hold.set()
            if not holder.done():
                holder.cancel()
                await asyncio.gather(holder, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_postgres_cold_schema_bootstrap_precedes_admission_and_holds_lifecycle_lock(
    db_backend,
    monkeypatch,
):
    """A real ``ALTER TABLE`` cold load cannot self-deadlock admission."""

    if db_backend.backend_type != "postgres":
        pytest.skip("requires PostgreSQL ACCESS SHARE/ACCESS EXCLUSIVE semantics")

    db = AsyncDatabase(db_backend)
    agent_id = f"scheduler-prepared-cold:{uuid4()}"
    task_id = f"scheduler-prepared-cold-task:{uuid4()}"
    marker_column = f"scheduler_prepared_{uuid4().hex}"
    lifecycle_lock = asyncio.Lock()
    authority = {"active": True}
    preparation_complete = asyncio.Event()
    dispatch_started = asyncio.Event()
    release_dispatch = asyncio.Event()
    config = object()
    tick: asyncio.Task[None] | None = None
    deletion: asyncio.Task[None] | None = None

    async def dispatch(_task_name, _args):
        assert lifecycle_lock.locked()
        dispatch_started.set()
        await release_dispatch.wait()
        return "ok"

    cold_agent = SimpleNamespace(
        did=agent_id,
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
            "expected_agent_id": agent_id,
            "scheduler_lifecycle_lock_held": True,
        }
        # This is the same table relation the admission's exact-token SELECT
        # reads. It needs ACCESS EXCLUSIVE on PostgreSQL, so the test would
        # deterministically hang/fail under the former ordering.
        await db.execute(
            f"ALTER TABLE scheduled_tasks ADD COLUMN IF NOT EXISTS {marker_column} TEXT"
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
            async with super()._active_dispatch_admission(task) as admitted:
                yield admitted

    runner = AdmissionProbeRunner(
        db,
        agent_id,
        AgentManagerHostedSchedulerExecutor(manager, {agent_id: ("Cold", config)}),
        is_agent_authorized=lambda _agent_id: authority["active"],
        owner_id="postgres-prepared-cold",
    )

    async def delete_agent():
        async with lifecycle_lock:
            authority["active"] = False

    try:
        await _activate_protocol_for_test_agent(db, agent_id, monkeypatch)
        await runner._ensure_tables()
        due = (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat()
        await db.execute(
            """
            INSERT INTO scheduled_tasks
                (id, agent_id, task_name, cron_expression, args_json, enabled,
                 next_run_at, created_at, idempotency_key,
                 scheduler_protocol_version)
            VALUES (?, ?, 'task', '* * * * *', '{}', 1, ?, ?, 'prepared-cold', ?)
            """,
            (task_id, agent_id, due, due, SCHEDULER_PROTOCOL_VERSION),
        )

        tick = asyncio.create_task(runner._tick())
        await asyncio.wait_for(dispatch_started.wait(), timeout=3)
        assert preparation_complete.is_set()

        deletion = asyncio.create_task(delete_agent())
        await asyncio.sleep(0)
        assert not deletion.done()
        assert authority["active"] is True

        release_dispatch.set()
        await asyncio.wait_for(tick, timeout=4)
        assert await asyncio.wait_for(deletion, timeout=2) is None
        assert authority["active"] is False
        assert await db.fetchone(
            "SELECT status FROM task_execution_log WHERE task_id = ?", (task_id,)
        ) == ("success",)
    finally:
        release_dispatch.set()
        for owned in (tick, deletion):
            if owned is not None and not owned.done():
                owned.cancel()
        await asyncio.gather(
            *(owned for owned in (tick, deletion) if owned is not None),
            return_exceptions=True,
        )
        await db.execute(
            f"ALTER TABLE scheduled_tasks DROP COLUMN IF EXISTS {marker_column}"
        )
        await db.execute("DELETE FROM task_execution_log WHERE task_id = ?", (task_id,))
        await db.execute("DELETE FROM scheduled_tasks WHERE id = ?", (task_id,))
        await db.execute("DELETE FROM scheduler_protocol_rollout WHERE agent_id = ?", (agent_id,))


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_postgres_renewal_does_not_resurrect_expired_token_after_row_lock_stall(
    db_backend, monkeypatch
):
    """A renewal blocked beyond expiry fails closed instead of extending stale work."""

    if db_backend.backend_type != "postgres":
        pytest.skip("requires PostgreSQL row-level lock semantics")
    db = AsyncDatabase(db_backend)
    agent_id = f"scheduler-clock-renew:{uuid4()}"
    task_id = f"scheduler-clock-renew-task:{uuid4()}"
    executor_started = asyncio.Event()
    release_executor = asyncio.Event()
    row_lock_held = asyncio.Event()
    blocked_renewal_started = asyncio.Event()
    blocked_renewal_finished = asyncio.Event()
    blocked_renewal_results: list[bool] = []

    async def executor(_task_name, _args):
        executor_started.set()
        await release_executor.wait()
        return "ok"

    class RenewalProbeRunner(SchedulerRunner):
        """Expose the renewal which begins while the external row lock is held."""

        async def _renew_lease_once(self, task):
            if not row_lock_held.is_set():
                return await super()._renew_lease_once(task)
            blocked_renewal_started.set()
            try:
                renewed = await super()._renew_lease_once(task)
                blocked_renewal_results.append(renewed)
                return renewed
            finally:
                blocked_renewal_finished.set()

    tick: asyncio.Task[None] | None = None
    locked_expiry: str | None = None
    try:
        await _activate_protocol_for_test_agent(db, agent_id, monkeypatch)
        runner = RenewalProbeRunner(
            db, agent_id, executor, owner_id="clock-renew", lease_seconds=1
        )
        await runner._ensure_tables()
        due = (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat()
        await db.execute(
            """
            INSERT INTO scheduled_tasks
                (id, agent_id, task_name, cron_expression, args_json, enabled,
                 next_run_at, created_at, idempotency_key,
                 scheduler_protocol_version)
            VALUES (?, ?, 'task', '* * * * *', '{}', 1, ?, ?, 'clock-renew', ?)
            """,
            (task_id, agent_id, due, due, SCHEDULER_PROTOCOL_VERSION),
        )
        tick = asyncio.create_task(runner._tick())
        await asyncio.wait_for(executor_started.wait(), timeout=2)
        initial = await db.fetchone(
            "SELECT lease_expires_at, claim_token FROM scheduled_tasks WHERE id = ?",
            (task_id,),
        )
        assert initial is not None
        pool = db_backend._pool
        assert pool is not None
        async with pool.acquire() as locked_connection:
            async with locked_connection.transaction():
                # This turns the historical lock-order regression into a
                # bounded failure: an admitted long executor must not retain
                # the schedule row merely because it holds the rollout epoch.
                await locked_connection.execute("SET LOCAL lock_timeout = '1500ms'")
                await locked_connection.execute(
                    "SELECT id FROM scheduled_tasks WHERE id = $1 AND agent_id = $2 FOR UPDATE",
                    task_id,
                    agent_id,
                )
                row_lock_held.set()
                await asyncio.wait_for(blocked_renewal_started.wait(), timeout=2)
                # A legitimate renewal can win just before this lock does. Hold
                # through the lease value observed *under the row lock*, rather
                # than assuming a fixed delay from executor entry, so the
                # blocked renewal is guaranteed to resume after expiry.
                locked_expiry, locked_now = await locked_connection.fetchrow(
                    """
                    SELECT lease_expires_at, clock_timestamp()
                    FROM scheduled_tasks
                    WHERE id = $1 AND agent_id = $2
                    """,
                    task_id,
                    agent_id,
                )
                assert isinstance(locked_expiry, str)
                assert isinstance(locked_now, datetime)
                expiry = datetime.fromisoformat(locked_expiry).astimezone(timezone.utc)
                if locked_now.tzinfo is None:
                    locked_now = locked_now.replace(tzinfo=timezone.utc)
                else:
                    locked_now = locked_now.astimezone(timezone.utc)
                await asyncio.sleep(
                    max(0.0, (expiry - locked_now).total_seconds()) + 0.2
                )

        row_lock_held.clear()
        await asyncio.wait_for(blocked_renewal_finished.wait(), timeout=2)
        assert blocked_renewal_results == [False]
        assert locked_expiry is not None
        row = await db.fetchone(
            "SELECT lease_expires_at, claim_token FROM scheduled_tasks WHERE id = ?",
            (task_id,),
        )
        database_now = await db.fetchval("SELECT clock_timestamp()")
        assert row is not None
        assert row[1] == initial[1]
        assert row[0] == locked_expiry
        lease = datetime.fromisoformat(row[0]).astimezone(timezone.utc)
        if database_now.tzinfo is None:
            database_now = database_now.replace(tzinfo=timezone.utc)
        else:
            database_now = database_now.astimezone(timezone.utc)
        assert lease <= database_now

        release_executor.set()
        await asyncio.wait_for(tick, timeout=3)
    finally:
        row_lock_held.clear()
        release_executor.set()
        if tick is not None and not tick.done():
            tick.cancel()
            await asyncio.gather(tick, return_exceptions=True)
        await db.execute("DELETE FROM task_execution_log WHERE task_id = ?", (task_id,))
        await db.execute("DELETE FROM scheduled_tasks WHERE id = ?", (task_id,))
        await db.execute("DELETE FROM scheduler_protocol_rollout WHERE agent_id = ?", (agent_id,))


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_host_runner_never_claims_another_fleets_rows_on_backend(db_backend, monkeypatch):
    """Host authority applies to both due selection and durable claim state."""
    db = AsyncDatabase(db_backend)
    local_agent = f"scheduler-local:{uuid4()}"
    foreign_agent = f"scheduler-foreign:{uuid4()}"
    local_task = f"scheduler-local-task:{uuid4()}"
    foreign_task = f"scheduler-foreign-task:{uuid4()}"
    seen = []

    async def executor(task_name, args):
        seen.append((task_name, args))
        return "ok"

    try:
        await _activate_protocol_for_test_agent(db, local_agent, monkeypatch)
        runner = SchedulerRunner(
            db,
            None,
            executor,
            authorized_agent_ids={local_agent},
            owner_id="authorized-host",
        )
        await runner._ensure_tables()
        due = (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat()
        for task_id, agent_id in (
            (local_task, local_agent),
            (foreign_task, foreign_agent),
        ):
            await db.execute(
                """
                INSERT INTO scheduled_tasks
                    (id, agent_id, task_name, cron_expression, args_json, enabled,
                     next_run_at, created_at, idempotency_key,
                     scheduler_protocol_version)
                VALUES (?, ?, 'task', '* * * * *', '{}', 1, ?, ?, 'integration-effect', ?)
                """,
                (task_id, agent_id, due, due, SCHEDULER_PROTOCOL_VERSION),
            )

        await runner._tick()

        assert seen == [("task", {})]
        assert await db.fetchone(
            """
            SELECT last_run_at, next_run_at, lease_owner, claim_execution_id,
                   attempt_count, terminal_status
            FROM scheduled_tasks WHERE id = ?
            """,
            (foreign_task,),
        ) == (None, due, None, None, 0, None)
        assert await db.fetchall(
            "SELECT status FROM task_execution_log WHERE task_id = ?",
            (foreign_task,),
        ) == []
    finally:
        await db.execute(
            "DELETE FROM task_execution_log WHERE task_id IN (?, ?)",
            (local_task, foreign_task),
        )
        await db.execute(
            "DELETE FROM scheduled_tasks WHERE id IN (?, ?)",
            (local_task, foreign_task),
        )
        await db.execute(
            "DELETE FROM scheduler_protocol_rollout WHERE agent_id = ?",
            (local_agent,),
        )


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_concurrent_replicas_fence_late_legacy_write_with_nonce_cas(
    db_backend, monkeypatch
):
    """The real backend makes a late legacy row rotate—not reuse—the nonce.

    This is intentionally dual-backend.  On PostgreSQL the two `ensure` calls
    use separate transaction connections, so the assertion covers the durable
    control-row CAS rather than merely one process's in-memory ordering.
    """

    db = AsyncDatabase(db_backend)
    agent_id = f"scheduler-rollout:{uuid4()}"
    task_id = f"scheduler-rollout-task:{uuid4()}"

    async def noop(_task_name, _args):
        return None

    try:
        await _activate_protocol_for_test_agent(db, agent_id, monkeypatch)
        due = (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat()
        # This insert names only legacy columns. Its nullable v2 marker proves
        # that an origin/main writer cannot be silently mistaken for v2.
        await db.execute(
            """
            INSERT INTO scheduled_tasks
                (id, agent_id, task_name, cron_expression, args_json, enabled,
                 next_run_at, created_at)
            VALUES (?, ?, 'task', '* * * * *', '{}', 1, ?, ?)
            """,
            (task_id, agent_id, due, due),
        )

        first = SchedulerRunner(db, agent_id, noop, owner_id="fence-a")
        second = SchedulerRunner(db, agent_id, noop, owner_id="fence-b")
        outcomes = await asyncio.gather(
            first._ensure_protocol_rollout(preexisting_schedule_table=True),
            second._ensure_protocol_rollout(preexisting_schedule_table=True),
            return_exceptions=True,
        )
        assert all(
            isinstance(outcome, SchedulerRolloutQuiescenceRequired)
            for outcome in outcomes
        )
        old_nonce = await db.fetchval(
            "SELECT activation_nonce FROM scheduler_protocol_rollout WHERE agent_id = ?",
            (agent_id,),
        )
        assert old_nonce
        row = await db.fetchone(
            """
            SELECT enabled, scheduler_protocol_version, scheduler_rollout_fenced,
                   scheduler_rollout_nonce, scheduler_rollout_snapshot
            FROM scheduled_tasks WHERE id = ?
            """,
            (task_id,),
        )
        assert row[0:4] == (0, None, 1, old_nonce)
        assert row[4]

        # A legacy poller that had already selected the row can reschedule that
        # same baseline row without changing its new columns. The snapshot
        # makes this visible and forces a new acknowledgement epoch.
        await db.execute(
            """
            UPDATE scheduled_tasks SET enabled = 1, last_run_at = ?, next_run_at = ?
            WHERE id = ? AND agent_id = ?
            """,
            (
                datetime.now(timezone.utc).isoformat(),
                (datetime.now(timezone.utc) + timedelta(minutes=1)).isoformat(),
                task_id,
                agent_id,
            ),
        )
        monkeypatch.setenv(SCHEDULER_ROLLOUT_ACK_ENV, old_nonce)
        ack_a = SchedulerRunner(db, agent_id, noop, owner_id="ack-a")
        ack_b = SchedulerRunner(db, agent_id, noop, owner_id="ack-b")
        outcomes = await asyncio.gather(
            ack_a._ensure_protocol_rollout(preexisting_schedule_table=True),
            ack_b._ensure_protocol_rollout(preexisting_schedule_table=True),
            return_exceptions=True,
        )
        assert all(
            isinstance(outcome, SchedulerRolloutQuiescenceRequired)
            for outcome in outcomes
        )
        new_nonce = await db.fetchval(
            "SELECT activation_nonce FROM scheduler_protocol_rollout WHERE agent_id = ?",
            (agent_id,),
        )
        assert new_nonce and new_nonce != old_nonce
        assert await db.fetchone(
            "SELECT enabled, scheduler_rollout_nonce FROM scheduled_tasks WHERE id = ?",
            (task_id,),
        ) == (0, new_nonce)
    finally:
        await db.execute("DELETE FROM task_execution_log WHERE task_id = ?", (task_id,))
        await db.execute("DELETE FROM scheduled_tasks WHERE id = ?", (task_id,))
        await db.execute("DELETE FROM scheduler_protocol_rollout WHERE agent_id = ?", (agent_id,))
