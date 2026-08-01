"""The semantic migrations must agree on their marker table and their lock.

Five migrations each ensured ``semantic_schema_migrations`` existed and each
hand-rolled the same PostgreSQL advisory lock. Two problems came with that:

1. Two of the five spelled the table WITH ``completed_at`` and three WITHOUT.
   Under ``CREATE TABLE IF NOT EXISTS`` whichever migration ran first silently
   decided the schema, so the column's presence depended on execution order —
   the same latent shape as #2804.
2. ``migrate_semantic_assertion_store`` opened a *deferred* SQLite transaction
   and promoted it to writer with a ``DELETE ... WHERE 0`` no-op, while the
   other four used an IMMEDIATE transaction. Two shapes for one invariant.

Both now go through ``AsyncDatabase.migration_lock``. These tests hold that
line: the platform primitive owns the *how*, and every migration declares the
same *what*.
"""

from __future__ import annotations

import asyncio

import pytest

from kestrel_sovereign.storage.async_database import AsyncDatabase
from kestrel_sovereign.storage.sqla.migrations import (
    migrate_semantic_assertion_store,
    migrate_semantic_governed_artifacts,
    migrate_semantic_maintenance,
    migrate_semantic_validation_reports,
    migrate_semantic_vector_projection,
)


# Every migration that ensures the shared marker table exists.
MARKER_TABLE_MIGRATIONS = (
    migrate_semantic_assertion_store,
    migrate_semantic_vector_projection,
    migrate_semantic_governed_artifacts,
    migrate_semantic_validation_reports,
    migrate_semantic_maintenance,
)


async def _marker_columns(db: AsyncDatabase) -> set[str]:
    rows = await db.fetchall("PRAGMA table_info(semantic_schema_migrations)")
    return {row[1] for row in rows}


@pytest.mark.parametrize(
    "migration", MARKER_TABLE_MIGRATIONS, ids=lambda m: m.__name__
)
@pytest.mark.asyncio
async def test_marker_table_schema_does_not_depend_on_which_migration_runs_first(
    migration,
):
    """Whichever migration creates the table first, it creates the same table.

    Parameterized rather than looped so a divergent spelling names the guilty
    migration instead of failing an opaque aggregate.
    """
    db = await AsyncDatabase.sqlite(":memory:")
    try:
        # ``AsyncDatabase.sqlite`` runs ``_init_schema``, which already creates
        # the marker table — so without dropping it first every migration's own
        # ``CREATE TABLE IF NOT EXISTS`` is a no-op and this test would pass
        # whatever DDL each one carried. It did exactly that on the first
        # attempt: the mutant survived. Drop it so the migration under test is
        # genuinely the one that creates the table.
        await db.execute("DROP TABLE IF EXISTS semantic_schema_migrations", ())
        assert not await db.table_exists("semantic_schema_migrations")

        await migration(db)
        columns = await _marker_columns(db)
    finally:
        await db.close()

    assert columns == {"version", "completed_at"}, (
        f"{migration.__name__} created semantic_schema_migrations as {columns}; "
        "all migrations must use _SEMANTIC_SCHEMA_MIGRATIONS_DDL so the schema "
        "does not depend on execution order"
    )


@pytest.mark.asyncio
async def test_separate_connections_racing_one_file_both_complete(tmp_path):
    """Two *connections* to one database file both complete the migration.

    This is the case the lock exists for and the one a single-connection test
    cannot reach: within one connection the backend's write-unit lock already
    serializes everything, so a deferred transaction looks identical to an
    IMMEDIATE one. A first version of this test raced two calls on a single
    in-memory database and passed even with the lock mutated away.

    Across two connections the difference is real. A deferred ``BEGIN`` that
    reads the marker and only then tries to write must upgrade to a writer
    while the other connection holds it, which fails immediately rather than
    waiting out ``busy_timeout``. ``migration_lock`` begins IMMEDIATE on
    SQLite, so both initializers serialize and both finish.
    """
    path = str(tmp_path / "race.db")
    first = await AsyncDatabase.sqlite(path)
    second = await AsyncDatabase.sqlite(path)
    try:
        # _init_schema already ran the migration on both; clear the marker so
        # the bodies genuinely re-execute and contend.
        await first.execute("DROP TABLE IF EXISTS semantic_schema_migrations", ())

        results = await asyncio.gather(
            migrate_semantic_assertion_store(first),
            migrate_semantic_assertion_store(second),
            return_exceptions=True,
        )
        failures = [r for r in results if isinstance(r, BaseException)]
        assert not failures, f"concurrent initializers raced: {failures}"

        duplicates = await first.fetchall(
            "SELECT version, COUNT(*) FROM semantic_schema_migrations "
            "GROUP BY version HAVING COUNT(*) > 1"
        )
        assert not duplicates, f"duplicate schema markers: {duplicates}"
    finally:
        await second.close()
        await first.close()


@pytest.mark.asyncio
async def test_migrations_remain_idempotent_under_the_shared_lock():
    """Re-running every migration is a no-op, in any order.

    The lock is shared across all five deliberately — they contend on one
    table — so this also covers that holding it repeatedly does not deadlock.
    """
    db = await AsyncDatabase.sqlite(":memory:")
    try:
        for migration in MARKER_TABLE_MIGRATIONS:
            await migration(db)
        for migration in reversed(MARKER_TABLE_MIGRATIONS):
            await migration(db)

        assert await _marker_columns(db) == {"version", "completed_at"}
        assert await db.table_exists("semantic_assertions")
        assert await db.table_exists("semantic_validation_reports")
    finally:
        await db.close()
