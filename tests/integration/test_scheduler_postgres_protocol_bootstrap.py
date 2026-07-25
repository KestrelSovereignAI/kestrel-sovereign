"""Fresh-fleet scheduler protocol bootstrap coverage on real backends."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import AsyncIterator
from uuid import uuid4

import pytest
from fastapi import FastAPI

from kestrel_sovereign import server
from kestrel_sovereign.features.scheduler.feature import SchedulerFeature
from kestrel_sovereign.features.scheduler.runner import (
    SCHEDULER_PROTOCOL_VERSION,
    SCHEDULER_ROLLOUT_STATE_ACTIVE,
    SCHEDULER_ROLLOUT_STATE_QUIESCING,
    SCHEDULER_SCHEMA_PROVENANCE_FRESH_V2,
    SchedulerProtocolVersionIncompatible,
    SchedulerRolloutQuiescenceRequired,
    SchedulerRunner,
)
from kestrel_sovereign.multi_agent.agent_manager import AgentManager
from kestrel_sovereign.multi_agent.config import LocalAgentConfig, MultiAgentConfig
from kestrel_sovereign.spawn.mandate import SpawnMandate
from kestrel_sovereign.storage.async_database import AsyncDatabase


@asynccontextmanager
async def _isolated_scheduler_schema(db_backend) -> AsyncIterator[AsyncDatabase]:
    """Yield a fresh scheduler namespace, including on PostgreSQL.

    The production PostgreSQL test database is shared by integration files, so
    the global provenance singleton cannot be reset there.  Keep the complete
    bootstrap inside one transaction-local schema instead.  Nested scheduler
    transactions reuse this task's PostgreSQL connection, preserving the
    temporary search path while the schema is created and inspected.
    """
    db = AsyncDatabase(db_backend)
    if db.backend_type == "sqlite":
        # The regression below must exercise an already-open caller
        # transaction on both backends.  SQLite's transaction is also the
        # normal production context in which additive bootstrap runs.
        async with db.transaction():
            yield db
        return

    schema = f"scheduler_protocol_bootstrap_{uuid4().hex}"
    await db.execute(f'CREATE SCHEMA "{schema}"')
    try:
        async with db.transaction():
            await db.execute(f'SET LOCAL search_path TO "{schema}"')
            yield db
    finally:
        await db.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')


@asynccontextmanager
async def _top_level_scheduler_databases(
    db_backend,
    *,
    count: int = 1,
) -> AsyncIterator[list[AsyncDatabase]]:
    """Yield isolated DB handles whose scheduler transactions are top-level."""

    if db_backend.backend_type == "sqlite":
        yield [AsyncDatabase(db_backend) for _ in range(count)]
        return

    import asyncpg

    from kestrel_sovereign.storage.db.postgres import PostgresBackend

    schema = f"scheduler_top_level_{uuid4().hex}"
    control = AsyncDatabase(db_backend)
    await control.execute(f'CREATE SCHEMA "{schema}"')
    pool = await asyncpg.create_pool(
        db_backend._dsn,
        min_size=max(1, count),
        max_size=max(2, count + 1),
        server_settings={"search_path": schema},
    )
    try:
        databases = [
            AsyncDatabase(
                PostgresBackend.from_pool(
                    pool,
                    advisory_dsn=db_backend._dsn,
                    advisory_connect_kwargs={"server_settings": {"search_path": schema}},
                )
            )
            for _ in range(count)
        ]
        yield databases
    finally:
        await pool.close()
        await control.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')


@asynccontextmanager
async def _public_scheduled_tasks_for_regression(
    db_backend,
) -> AsyncIterator[None]:
    """Ensure PostgreSQL has the public relation that must stay out of scope."""
    if db_backend.backend_type != "postgres":
        yield
        return

    db = AsyncDatabase(db_backend)
    existed = bool(
        await db.fetchval(
            "SELECT to_regclass(?) IS NOT NULL",
            ("public.scheduled_tasks",),
        )
    )
    if not existed:
        await db.execute("CREATE TABLE public.scheduled_tasks (id TEXT PRIMARY KEY)")
    try:
        yield
    finally:
        if not existed:
            await db.execute("DROP TABLE public.scheduled_tasks")


async def _noop_executor(_task_name, _args):
    return None


def _host_runner(db: AsyncDatabase, agent_ids: set[str]) -> SchedulerRunner:
    """Build the non-polling fleet bootstrap shape used by server startup."""
    return SchedulerRunner(
        db,
        agent_id=None,
        executor=_noop_executor,
        authorized_agent_ids=agent_ids,
        owner_id=f"protocol-bootstrap:{uuid4()}",
    )


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_scheduler_table_detection_uses_the_active_schema(db_backend):
    """A public table cannot make a fresh PostgreSQL schema look legacy."""
    agent_id = f"did:scheduler:table-detection:{uuid4()}"

    async with _public_scheduled_tasks_for_regression(db_backend):
        async with _isolated_scheduler_schema(db_backend) as db:
            if db.backend_type == "postgres":
                assert await db.fetchval(
                    "SELECT to_regclass(?) IS NOT NULL",
                    ("public.scheduled_tasks",),
                )

            runner = _host_runner(db, {agent_id})
            assert not await runner._scheduled_tasks_table_exists()

            await db.execute("CREATE TABLE scheduled_tasks (id TEXT PRIMARY KEY)")
            assert await runner._scheduled_tasks_table_exists()


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_bootstrap_keeps_existing_transaction_usable_after_additive_migration(
    db_backend,
):
    """Idempotent DDL must not abort the host's transaction.

    PostgreSQL runs this inside a transaction-local ``search_path`` schema;
    SQLite runs inside an explicit transaction.  The update after bootstrap
    is deliberately the statement that exposed the former PostgreSQL
    ``InFailedSQLTransactionError``.
    """
    agent_id = f"did:scheduler:transaction-bootstrap:{uuid4()}"

    async with _isolated_scheduler_schema(db_backend) as db:
        await _host_runner(db, {agent_id})._ensure_tables()

        # The scheduler's CREATE TABLE shape already includes every additive
        # column.  Bootstrap therefore exercises every idempotent migration
        # path while the caller's transaction remains open.
        await db.execute("UPDATE scheduled_tasks SET enabled = 1 WHERE 1 = 0")
        assert await db.fetchone("SELECT COUNT(*) FROM scheduled_tasks") == (0,)


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_fresh_host_bootstrap_seeds_all_configured_dids_before_scoped_runners(
    db_backend,
):
    """A fresh multi-DID fleet never treats its second DID as a legacy upgrade.

    This is deliberately dual-backend.  The PostgreSQL parametrization drives
    the production migration and advisory-lock implementation against a real
    backend when ``TEST_POSTGRES_URL`` is available; SQLite retains the same
    durable schema semantics for local fleets.
    """
    warm_did = f"did:scheduler:fresh-warm:{uuid4()}"
    cold_did = f"did:scheduler:fresh-cold:{uuid4()}"

    async with _isolated_scheduler_schema(db_backend) as db:
        # This models server preflight, before two autostart agents can launch
        # their own SchedulerFeature runners concurrently.
        await _host_runner(db, {warm_did, cold_did})._ensure_tables()

        assert await db.fetchone(
            "SELECT provenance, protocol_version "
            "FROM scheduler_protocol_schema WHERE singleton = 1"
        ) == (SCHEDULER_SCHEMA_PROVENANCE_FRESH_V2, SCHEDULER_PROTOCOL_VERSION)
        assert await db.fetchall(
            "SELECT agent_id, protocol_version, state, activation_nonce "
            "FROM scheduler_protocol_rollout ORDER BY agent_id"
        ) == [
            (cold_did, SCHEDULER_PROTOCOL_VERSION, SCHEDULER_ROLLOUT_STATE_ACTIVE, None),
            (warm_did, SCHEDULER_PROTOCOL_VERSION, SCHEDULER_ROLLOUT_STATE_ACTIVE, None),
        ]

        # These are the later independently-started per-agent feature runners.
        # Neither may require a legacy rollout nonce merely because another
        # fresh DID already caused the shared tables to exist.
        for agent_id in (warm_did, cold_did):
            await SchedulerRunner(
                db,
                agent_id,
                _noop_executor,
                owner_id=f"scoped-after-preflight:{agent_id}",
            )._ensure_tables()

        assert await db.fetchall(
            "SELECT agent_id, state, activation_nonce "
            "FROM scheduler_protocol_rollout ORDER BY agent_id"
        ) == [
            (cold_did, SCHEDULER_ROLLOUT_STATE_ACTIVE, None),
            (warm_did, SCHEDULER_ROLLOUT_STATE_ACTIVE, None),
        ]


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_new_did_in_fresh_shared_fleet_is_seeded_without_legacy_quiescence(
    db_backend,
):
    """A later configured DID inherits durable fresh-v2 provenance safely."""
    first_did = f"did:scheduler:first:{uuid4()}"
    added_did = f"did:scheduler:added:{uuid4()}"

    async with _isolated_scheduler_schema(db_backend) as db:
        await _host_runner(db, {first_did})._ensure_tables()

        # A later host configuration includes the old DID plus a newly added
        # tenant.  The global marker—not accidental table existence—proves the
        # new DID is part of the fresh v2 fleet.
        await _host_runner(db, {first_did, added_did})._ensure_tables()

        assert await db.fetchone(
            "SELECT provenance FROM scheduler_protocol_schema WHERE singleton = 1"
        ) == (SCHEDULER_SCHEMA_PROVENANCE_FRESH_V2,)
        assert await db.fetchall(
            "SELECT agent_id, protocol_version, state, activation_nonce "
            "FROM scheduler_protocol_rollout ORDER BY agent_id"
        ) == [
            (added_did, SCHEDULER_PROTOCOL_VERSION, SCHEDULER_ROLLOUT_STATE_ACTIVE, None),
            (first_did, SCHEDULER_PROTOCOL_VERSION, SCHEDULER_ROLLOUT_STATE_ACTIVE, None),
        ]

        await SchedulerRunner(
            db,
            added_did,
            _noop_executor,
            owner_id="newly-added-scoped-runner",
        )._ensure_tables()
        assert await db.fetchone(
            "SELECT state, activation_nonce FROM scheduler_protocol_rollout "
            "WHERE agent_id = ?",
            (added_did,),
        ) == (SCHEDULER_ROLLOUT_STATE_ACTIVE, None)


@pytest.mark.asyncio
@pytest.mark.dual_backend
@pytest.mark.parametrize("future_state", ["global", "per_did"])
async def test_newer_protocol_state_fails_before_any_scheduler_mutation(
    db_backend,
    future_state,
):
    """Older runners leave exact future durable state untouched on both DBs."""

    agent_id = f"did:scheduler:future-protocol:{uuid4()}"
    future_version = SCHEDULER_PROTOCOL_VERSION + 7
    future_nonce = "opaque-future-nonce"

    async with _isolated_scheduler_schema(db_backend) as db:
        await db.execute(
            """
            CREATE TABLE scheduler_protocol_schema (
                singleton INTEGER PRIMARY KEY,
                provenance TEXT NOT NULL,
                protocol_version INTEGER NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        await db.execute(
            """
            INSERT INTO scheduler_protocol_schema
                (singleton, provenance, protocol_version, created_at)
            VALUES (1, 'future-provenance', ?, 'future-created-at')
            """,
            (
                future_version
                if future_state == "global"
                else SCHEDULER_PROTOCOL_VERSION,
            ),
        )
        if future_state == "per_did":
            await db.execute(
                """
                CREATE TABLE scheduler_protocol_rollout (
                    agent_id TEXT PRIMARY KEY,
                    protocol_version INTEGER NOT NULL,
                    state TEXT NOT NULL,
                    activation_nonce TEXT,
                    updated_at TEXT NOT NULL
                )
                """
            )
            await db.execute(
                """
                INSERT INTO scheduler_protocol_rollout
                    (agent_id, protocol_version, state, activation_nonce, updated_at)
                VALUES (?, ?, 'future-state', ?, 'future-updated-at')
                """,
                (agent_id, future_version, future_nonce),
            )

        runner = _host_runner(db, {agent_id})
        with pytest.raises(SchedulerProtocolVersionIncompatible) as raised:
            await runner.start()

        # Readiness can expose only the safe exception type/message: never a
        # future version, tenant DID, nonce, or opaque state string.
        assert isinstance(runner.readiness_failure, SchedulerProtocolVersionIncompatible)
        message = str(raised.value)
        assert str(future_version) not in message
        assert agent_id not in message
        assert future_nonce not in message
        assert "future-state" not in message
        assert not await db.table_exists("scheduled_tasks")

        assert await db.fetchone(
            "SELECT provenance, protocol_version, created_at "
            "FROM scheduler_protocol_schema WHERE singleton = 1"
        ) == (
            "future-provenance",
            future_version if future_state == "global" else SCHEDULER_PROTOCOL_VERSION,
            "future-created-at",
        )
        if future_state == "per_did":
            assert await db.fetchone(
                "SELECT agent_id, protocol_version, state, activation_nonce, updated_at "
                "FROM scheduler_protocol_rollout WHERE agent_id = ?",
                (agent_id,),
            ) == (
                agent_id,
                future_version,
                "future-state",
                future_nonce,
                "future-updated-at",
            )


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_top_level_bootstrap_preserves_typed_future_protocol_failure(
    db_backend,
):
    """Concrete backend transactions must not hide the readiness exception."""

    agent_id = f"did:scheduler:top-level-future:{uuid4()}"
    future_version = SCHEDULER_PROTOCOL_VERSION + 1
    async with _top_level_scheduler_databases(db_backend) as databases:
        db = databases[0]
        await db.execute(
            """
            CREATE TABLE scheduler_protocol_schema (
                singleton INTEGER PRIMARY KEY,
                provenance TEXT NOT NULL,
                protocol_version INTEGER NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        await db.execute(
            """
            INSERT INTO scheduler_protocol_schema
                (singleton, provenance, protocol_version, created_at)
            VALUES (1, 'future', ?, 'future-created-at')
            """,
            (future_version,),
        )

        runner = _host_runner(db, {agent_id})
        with pytest.raises(SchedulerProtocolVersionIncompatible):
            await runner.start()

        assert isinstance(
            runner.readiness_failure,
            SchedulerProtocolVersionIncompatible,
        )
        assert not await db.table_exists("scheduled_tasks")


@pytest.mark.asyncio
@pytest.mark.dual_backend
@pytest.mark.parametrize(
    "operation",
    [
        "add",
        "deadline",
        "remove",
        "pause",
        "resume",
        "update",
        "record_outcome",
    ],
)
async def test_every_mutating_api_rejects_future_global_protocol_before_write(
    db_backend,
    operation,
):
    """Global v3 plus a still-active DID v2 cannot mutate scheduler state."""

    agent_id = f"did:scheduler:future-api:{operation}:{uuid4()}"
    future_version = SCHEDULER_PROTOCOL_VERSION + 1
    async with _isolated_scheduler_schema(db_backend) as db:
        await _host_runner(db, {agent_id})._ensure_tables()
        agent = SimpleNamespace(
            did=agent_id,
            agent_id=agent_id,
            features={},
        )
        feature = SchedulerFeature(agent)
        feature._db = db
        feature._agent_id = agent_id
        added = await feature.schedule_add(
            cron_expression="@daily",
            task_name="backup_snapshot",
            idempotency_key=f"future-api-seed:{operation}",
        )
        assert added.status.value == "ok"
        task_id = added.data["task_id"]
        if operation == "resume":
            paused = await feature.schedule_pause(task_id)
            assert paused.status.value == "ok"

        execution_id = f"future-api-execution:{uuid4()}"
        now = datetime.now(timezone.utc).isoformat()
        await db.execute(
            """
            INSERT INTO task_execution_log
                (id, task_id, agent_id, status, result_text, duration_ms,
                 executed_at, outcome_signal, attempt_count)
            VALUES (?, ?, ?, 'success', 'before', 1, ?, NULL, 1)
            """,
            (execution_id, task_id, agent_id, now),
        )
        await db.execute(
            """
            UPDATE scheduler_protocol_schema
            SET protocol_version = ?, provenance = 'future-live-transition'
            WHERE singleton = 1
            """,
            (future_version,),
        )
        schedule_before = await db.fetchall(
            """
            SELECT id, task_name, cron_expression, enabled, next_run_at,
                   scheduler_protocol_version, scheduler_rollout_fenced,
                   scheduler_claim_fenced
            FROM scheduled_tasks WHERE agent_id = ? ORDER BY id
            """,
            (agent_id,),
        )
        log_before = await db.fetchall(
            """
            SELECT id, status, result_text, outcome_signal
            FROM task_execution_log WHERE agent_id = ? ORDER BY id
            """,
            (agent_id,),
        )

        if operation == "add":
            result = await feature.schedule_add(
                cron_expression="@hourly",
                task_name="backup_snapshot",
                idempotency_key="future-api-blocked-add",
            )
        elif operation == "deadline":
            result = await feature.schedule_add_deadline(
                run_at=(
                    datetime.now(timezone.utc) + timedelta(hours=1)
                ).isoformat(),
                task_name="backup_snapshot",
                idempotency_key="future-api-blocked-deadline",
            )
        elif operation == "remove":
            result = await feature.schedule_remove(task_id)
        elif operation == "pause":
            result = await feature.schedule_pause(task_id)
        elif operation == "resume":
            result = await feature.schedule_resume(task_id)
        elif operation == "update":
            result = await feature.schedule_update(task_id, "@weekly")
        else:
            result = await feature.schedule_record_outcome(
                execution_id,
                0.75,
            )

        assert result.status.value == "error"
        assert "newer than this runner" in result.error
        assert await db.fetchall(
            """
            SELECT id, task_name, cron_expression, enabled, next_run_at,
                   scheduler_protocol_version, scheduler_rollout_fenced,
                   scheduler_claim_fenced
            FROM scheduled_tasks WHERE agent_id = ? ORDER BY id
            """,
            (agent_id,),
        ) == schedule_before
        assert await db.fetchall(
            """
            SELECT id, status, result_text, outcome_signal
            FROM task_execution_log WHERE agent_id = ? ORDER BY id
            """,
            (agent_id,),
        ) == log_before
        assert await db.fetchone(
            """
            SELECT provenance, protocol_version
            FROM scheduler_protocol_schema WHERE singleton = 1
            """
        ) == ("future-live-transition", future_version)


@pytest.mark.asyncio
@pytest.mark.dual_backend
@pytest.mark.parametrize("operation", ["pause", "remove"])
async def test_pause_and_remove_reject_future_did_without_clearing_fence(
    db_backend,
    operation,
):
    """A false v2 lock result never authorizes a future-DID mutation."""

    agent_id = f"did:scheduler:future-did-api:{operation}:{uuid4()}"
    future_version = SCHEDULER_PROTOCOL_VERSION + 1
    async with _isolated_scheduler_schema(db_backend) as db:
        await _host_runner(db, {agent_id})._ensure_tables()
        task_id = f"future-fenced-task:{uuid4()}"
        now = datetime.now(timezone.utc).isoformat()
        await db.execute(
            """
            INSERT INTO scheduled_tasks
                (id, agent_id, task_name, cron_expression, args_json, enabled,
                 next_run_at, created_at, idempotency_key,
                 scheduler_protocol_version, scheduler_rollout_fenced,
                 scheduler_rollout_nonce, scheduler_claim_fenced,
                 scheduler_rollout_fenced_at)
            VALUES (?, ?, 'backup_snapshot', '@daily', '{}', 0, ?, ?,
                    'future-fenced-effect', ?, 1, 'future-fence-nonce', 0, ?)
            """,
            (
                task_id,
                agent_id,
                now,
                now,
                SCHEDULER_PROTOCOL_VERSION,
                now,
            ),
        )
        await db.execute(
            """
            UPDATE scheduler_protocol_rollout
            SET protocol_version = ?, state = 'future-state',
                activation_nonce = 'future-control-nonce'
            WHERE agent_id = ?
            """,
            (future_version, agent_id),
        )
        feature = SchedulerFeature(
            SimpleNamespace(
                did=agent_id,
                agent_id=agent_id,
                features={},
            )
        )
        feature._db = db
        feature._agent_id = agent_id
        before = await db.fetchone(
            """
            SELECT enabled, scheduler_rollout_fenced,
                   scheduler_rollout_nonce, scheduler_rollout_fenced_at
            FROM scheduled_tasks WHERE id = ?
            """,
            (task_id,),
        )

        result = (
            await feature.schedule_pause(task_id)
            if operation == "pause"
            else await feature.schedule_remove(task_id)
        )

        assert result.status.value == "error"
        assert "newer than this runner" in result.error
        assert await db.fetchone(
            """
            SELECT enabled, scheduler_rollout_fenced,
                   scheduler_rollout_nonce, scheduler_rollout_fenced_at
            FROM scheduled_tasks WHERE id = ?
            """,
            (task_id,),
        ) == before
        assert await db.fetchone(
            """
            SELECT protocol_version, state, activation_nonce
            FROM scheduler_protocol_rollout WHERE agent_id = ?
            """,
            (agent_id,),
        ) == (
            future_version,
            "future-state",
            "future-control-nonce",
        )


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_multi_did_quiescence_fences_commit_before_bootstrap_raises(db_backend):
    """An outer bootstrap transaction commits every fence before its error."""

    first_did = f"did:scheduler:quiesce-first:{uuid4()}"
    second_did = f"did:scheduler:quiesce-second:{uuid4()}"
    async with _isolated_scheduler_schema(db_backend) as db:
        # Model the old pre-protocol relation. Both rows must be fenced in the
        # same bootstrap boundary; raising inside that transaction would roll
        # every DDL/fence back and make this assertion impossible.
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
        for ordinal, agent_id in enumerate((first_did, second_did), start=1):
            await db.execute(
                """
                INSERT INTO scheduled_tasks
                    (id, agent_id, task_name, cron_expression, args_json, enabled,
                     next_run_at, created_at)
                VALUES (?, ?, 'legacy', '* * * * *', '{}', 1,
                        '2000-01-01T00:00:00+00:00', '2000-01-01T00:00:00+00:00')
                """,
                (f"legacy-{ordinal}", agent_id),
            )

        with pytest.raises(SchedulerRolloutQuiescenceRequired):
            await _host_runner(db, {first_did, second_did})._ensure_tables()

        rollout_rows = await db.fetchall(
            "SELECT agent_id, protocol_version, state, activation_nonce "
            "FROM scheduler_protocol_rollout ORDER BY agent_id"
        )
        assert [row[:3] for row in rollout_rows] == [
            (first_did, SCHEDULER_PROTOCOL_VERSION, SCHEDULER_ROLLOUT_STATE_QUIESCING),
            (second_did, SCHEDULER_PROTOCOL_VERSION, SCHEDULER_ROLLOUT_STATE_QUIESCING),
        ]
        assert all(isinstance(row[3], str) and row[3] for row in rollout_rows)
        rows = await db.fetchall(
            "SELECT agent_id, enabled, scheduler_rollout_fenced, "
            "scheduler_rollout_nonce FROM scheduled_tasks ORDER BY agent_id"
        )
        assert [row[:3] for row in rows] == [
            (first_did, 0, 1),
            (second_did, 0, 1),
        ]
        assert all(isinstance(row[3], str) and row[3] for row in rows)


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_postgres_concurrent_bootstraps_hold_one_boundary_through_all_dids(
    db_backend,
):
    """Separate PG connections cannot enter migration/seeding concurrently."""

    if db_backend.backend_type != "postgres":
        pytest.skip("requires separate PostgreSQL pool connections")

    schema = f"scheduler_concurrent_bootstrap_{uuid4().hex}"
    control = AsyncDatabase(db_backend)
    first_db = AsyncDatabase(db_backend)
    second_db = AsyncDatabase(db_backend)
    first_inside_mutations = asyncio.Event()
    release_first = asyncio.Event()
    second_inside_mutations = asyncio.Event()
    first_did = f"did:scheduler:concurrent-first:{uuid4()}"
    second_did = f"did:scheduler:concurrent-second:{uuid4()}"
    agent_ids = {first_did, second_did}

    class FirstBootstrapRunner(SchedulerRunner):
        async def _ensure_tables_mutations(self):
            first_inside_mutations.set()
            await release_first.wait()
            return await super()._ensure_tables_mutations()

    class SecondBootstrapRunner(SchedulerRunner):
        async def _ensure_tables_mutations(self):
            second_inside_mutations.set()
            return await super()._ensure_tables_mutations()

    async def bootstrap_in_schema(db, runner):
        async with db.transaction():
            await db.execute(f'SET LOCAL search_path TO "{schema}"')
            await runner._ensure_tables()

    first_runner = FirstBootstrapRunner(
        first_db,
        None,
        _noop_executor,
        authorized_agent_ids=agent_ids,
        owner_id="postgres-bootstrap-first",
    )
    second_runner = SecondBootstrapRunner(
        second_db,
        None,
        _noop_executor,
        authorized_agent_ids=agent_ids,
        owner_id="postgres-bootstrap-second",
    )
    first_task: asyncio.Task[None] | None = None
    second_task: asyncio.Task[None] | None = None
    await control.execute(f'CREATE SCHEMA "{schema}"')
    try:
        first_task = asyncio.create_task(bootstrap_in_schema(first_db, first_runner))
        await asyncio.wait_for(first_inside_mutations.wait(), timeout=2)

        second_task = asyncio.create_task(bootstrap_in_schema(second_db, second_runner))
        await asyncio.sleep(0.1)
        assert not second_inside_mutations.is_set()
        assert not second_task.done()

        release_first.set()
        await asyncio.wait_for(asyncio.gather(first_task, second_task), timeout=5)

        async with control.transaction():
            await control.execute(f'SET LOCAL search_path TO "{schema}"')
            assert await control.fetchone(
                "SELECT provenance, protocol_version FROM scheduler_protocol_schema "
                "WHERE singleton = 1"
            ) == (SCHEDULER_SCHEMA_PROVENANCE_FRESH_V2, SCHEDULER_PROTOCOL_VERSION)
            assert await control.fetchall(
                "SELECT agent_id, protocol_version, state, activation_nonce "
                "FROM scheduler_protocol_rollout ORDER BY agent_id"
            ) == [
                (first_did, SCHEDULER_PROTOCOL_VERSION, SCHEDULER_ROLLOUT_STATE_ACTIVE, None),
                (second_did, SCHEDULER_PROTOCOL_VERSION, SCHEDULER_ROLLOUT_STATE_ACTIVE, None),
            ]
        assert first_runner._protocol_ready and second_runner._protocol_ready
    finally:
        release_first.set()
        for owned in (first_task, second_task):
            if owned is not None and not owned.done():
                owned.cancel()
        await asyncio.gather(
            *(owned for owned in (first_task, second_task) if owned is not None),
            return_exceptions=True,
        )
        await control.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_postgres_concurrent_builtin_seeders_insert_each_default_once(
    db_backend,
):
    """Independent PG connections serialize the same-DID default ensure."""

    if db_backend.backend_type != "postgres":
        pytest.skip("SQLite concurrent seeding is covered by the file-backed unit test")

    agent_id = f"did:scheduler:concurrent-defaults:{uuid4()}"
    async with _top_level_scheduler_databases(
        db_backend,
        count=2,
    ) as databases:
        first_db, second_db = databases
        await _host_runner(first_db, {agent_id})._ensure_tables()

        def feature_for(db):
            agent = SimpleNamespace(
                did=agent_id,
                agent_id=agent_id,
                features={},
            )
            feature = SchedulerFeature(agent)
            feature._db = db
            feature._agent_id = agent_id
            return feature, agent

        first, first_agent = feature_for(first_db)
        second, second_agent = feature_for(second_db)
        await asyncio.gather(
            first.post_all_features_loaded(first_agent),
            second.post_all_features_loaded(second_agent),
        )

        assert await first_db.fetchall(
            """
            SELECT task_name, COUNT(*), MIN(idempotency_key)
            FROM scheduled_tasks
            WHERE agent_id = ?
            GROUP BY task_name
            ORDER BY task_name
            """,
            (agent_id,),
        ) == [
            ("backup_snapshot", 1, "scheduler:builtin:v1:backup_snapshot"),
            ("morning_signal", 1, "scheduler:builtin:v1:morning_signal"),
            ("signal_dispatch", 1, "scheduler:builtin:v1:signal_dispatch"),
            ("trash_retention", 1, "scheduler:builtin:v1:trash_retention"),
            ("wait_reconcile", 1, "scheduler:builtin:v1:wait_reconcile"),
        ]


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_live_postgres_runtime_create_spawn_execute_remove_and_failure_rollback(
    db_backend,
    monkeypatch,
    tmp_path,
):
    """Runtime tenants join and leave one live host scheduler atomically."""

    if db_backend.backend_type != "postgres":
        pytest.skip("dynamic shared scheduler registration is PostgreSQL-only")

    async with _top_level_scheduler_databases(db_backend) as databases:
        db = databases[0]
        app = FastAPI()
        manager = AgentManager(base_data_dir=tmp_path)
        dispatch_started = asyncio.Event()
        release_dispatch = asyncio.Event()

        class HostStorage:
            def __init__(self, *, backend, dsn):
                assert backend == "postgres"
                self.db = db

            async def initialize(self):
                return None

            async def close(self):
                return None

        class TestLLMService:
            async def close(self):
                return None

        class HostedTestAgent:
            def __init__(self, *, did, **_kwargs):
                self.did = did
                self.agent_id = did
                self.features = {}
                self.dispatcher = None
                self.signal_registry = None
                self._private_key = None
                self.identity = None
                self.wallet = None

            def _set_display_name(self, _name):
                return None

            async def initialize(self):
                feature = SchedulerFeature(self)
                feature._db = db
                feature._agent_id = self.agent_id
                feature._runner = None
                await feature.post_all_features_loaded(self)

                async def dispatch(_task_name, _args):
                    if self.agent_id.endswith(":Spawned"):
                        dispatch_started.set()
                        await release_dispatch.wait()
                    return f"executed:{self.agent_id}"

                feature._dispatch_scheduled_task = dispatch
                self.features = {"SchedulerFeature": feature}
                if self.agent_id.endswith(":Broken"):
                    raise RuntimeError("forced post-seed initialization failure")

            async def shutdown(self):
                feature = self.features.get("SchedulerFeature")
                if feature is not None and callable(
                    getattr(feature, "shutdown", None)
                ):
                    await feature.shutdown()

        parent_id = f"did:scheduler:dynamic-parent:{uuid4()}"
        parent = HostedTestAgent(did=parent_id)
        parent.features = {"SchedulerFeature": SimpleNamespace()}
        manager._agents["Parent"] = parent
        manager._agent_names[parent_id] = "Parent"
        parent_config = LocalAgentConfig(
            data_dir="parent",
            port=8801,
            autostart=True,
        )
        config = MultiAgentConfig(agents={"Parent": parent_config})

        async def fake_get_agent_did(storage_dir, *, mode):
            return f"did:scheduler:runtime:{storage_dir.rsplit('/', 1)[-1]}"

        async def fake_inception(**_kwargs):
            return None

        monkeypatch.setenv("KESTREL_DB_BACKEND", "postgres")
        monkeypatch.setenv("KESTREL_DATABASE_URL", "postgresql://scheduler-test")
        monkeypatch.setattr(
            "kestrel_sovereign.storage.async_storage.AsyncStorage",
            HostStorage,
        )
        monkeypatch.setattr(
            "kestrel_sovereign.multi_agent.agent_manager.KestrelAgent",
            HostedTestAgent,
        )
        monkeypatch.setattr(
            "kestrel_sovereign.multi_agent.agent_manager.LLMService",
            TestLLMService,
        )
        monkeypatch.setattr(
            "kestrel_sovereign.multi_agent.agent_manager._get_agent_did",
            fake_get_agent_did,
        )
        monkeypatch.setattr(
            "kestrel_sovereign.inception_service.create_kestrel_identity_async",
            fake_inception,
        )
        monkeypatch.setattr(
            LocalAgentConfig,
            "validate_runtime",
            lambda self, **_kwargs: [],
        )

        tick = None
        removal = None
        try:
            await server._prepare_shared_postgres_scheduler_protocol(
                app,
                manager,
                config,
            )
            await server._start_host_scheduler(app, manager, config)
            host_runner = app.state.host_scheduler_runner
            await host_runner.stop()

            created = await manager.create_agent("Created")
            assert created.agent_id in manager.scheduler_authorized_agent_ids()

            mandate = SpawnMandate(
                parent_did=parent_id,
                purpose="dynamic scheduler regression",
                features_allowed=["SchedulerFeature"],
            )
            spawned = await manager.spawn_agent("Spawned", parent, mandate)
            spawned_id = spawned.agent_id
            assert spawned_id in manager.scheduler_authorized_agent_ids()

            assert await db.fetchall(
                """
                SELECT task_name, COUNT(*)
                FROM scheduled_tasks
                WHERE agent_id IN (?, ?)
                GROUP BY agent_id, task_name
                ORDER BY agent_id, task_name
                """,
                (created.agent_id, spawned_id),
            ) == [
                ("backup_snapshot", 1),
                ("morning_signal", 1),
                ("signal_dispatch", 1),
                ("trash_retention", 1),
                ("wait_reconcile", 1),
                ("backup_snapshot", 1),
                ("morning_signal", 1),
                ("signal_dispatch", 1),
                ("trash_retention", 1),
                ("wait_reconcile", 1),
            ]

            due = (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat()
            await db.execute(
                """
                UPDATE scheduled_tasks
                SET next_run_at = ?, enabled = 1
                WHERE agent_id = ? AND task_name = 'wait_reconcile'
                """,
                (due, spawned_id),
            )
            tick = asyncio.create_task(host_runner._tick())
            await asyncio.wait_for(dispatch_started.wait(), timeout=3)

            removal = asyncio.create_task(manager.remove_agent("Spawned"))
            await asyncio.sleep(0.05)
            assert not removal.done()
            assert spawned_id in manager.scheduler_authorized_agent_ids()

            release_dispatch.set()
            await asyncio.wait_for(tick, timeout=3)
            assert await asyncio.wait_for(removal, timeout=3) is True
            assert spawned_id not in manager.scheduler_authorized_agent_ids()
            with pytest.raises(LookupError, match="refusing to reauthorize"):
                await manager.create_agent("Spawned")

            broken_id = "did:scheduler:runtime:Broken"
            with pytest.raises(
                RuntimeError,
                match="forced post-seed initialization failure",
            ):
                await manager.create_agent("Broken")
            assert manager.scheduler_authority_for(broken_id) is None
            assert broken_id not in manager.scheduler_authorized_agent_ids()
            assert await db.fetchone(
                "SELECT COUNT(*) FROM scheduled_tasks WHERE agent_id = ?",
                (broken_id,),
            ) == (0,)
            assert await db.fetchone(
                "SELECT COUNT(*) FROM scheduler_protocol_rollout WHERE agent_id = ?",
                (broken_id,),
            ) == (0,)
        finally:
            release_dispatch.set()
            for owned in (tick, removal):
                if owned is not None and not owned.done():
                    owned.cancel()
            await asyncio.gather(
                *(owned for owned in (tick, removal) if owned is not None),
                return_exceptions=True,
            )
            await server._shutdown_host_scheduler(app)
            await manager.shutdown_all()
