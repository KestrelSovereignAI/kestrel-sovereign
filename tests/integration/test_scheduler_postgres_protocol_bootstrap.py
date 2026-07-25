"""Fresh-fleet scheduler protocol bootstrap coverage on real backends."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator
from uuid import uuid4

import pytest

from kestrel_sovereign.features.scheduler.runner import (
    SCHEDULER_PROTOCOL_VERSION,
    SCHEDULER_ROLLOUT_STATE_ACTIVE,
    SCHEDULER_SCHEMA_PROVENANCE_FRESH_V2,
    SchedulerRunner,
)
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
