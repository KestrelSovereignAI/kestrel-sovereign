"""#2958: the session_id backfill means the same thing on SQLite and Postgres.

The backfill is the one part of this migration that cannot be written once —
SQLite reads metadata with ``json_extract`` and Postgres with a ``jsonb``
operator, and the "not a bare integer" test has no portable spelling either.
Two spellings of one rule is exactly the shape that drifts, so both are driven
against a real pre-migration table here rather than only asserted as strings.

Postgres is skipped when no test server is configured; the SQLite leg still
runs everywhere.
"""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from typing import AsyncIterator
from uuid import uuid4

import pytest

from kestrel_sovereign.storage.async_database import AsyncDatabase
from kestrel_sovereign.storage.session_grouping import canonical_session_id


UUID_A = "3a6c1f0e-2b7d-4c8a-9e10-000000000001"

# (label, stored metadata, expected column value).
#
# The bottom block is the set that made the two dialects and Python disagree
# before the rule was stated once. Each renders differently in each reader —
# ``{"session_id": true}`` extracts as 1 in SQLite, 'true' in Postgres and
# 'True' under ``str()``; a JSON object comes back as compact JSON text from
# SQLite and as a dict repr from Python; "١٢٣" is ``str.isdigit()`` in Python
# but matches neither dialect's ASCII digit test. They are the reason both
# backfills carry an explicit JSON-type guard and an ASCII-range digit test.
CASES = [
    ("uuid", {"session_id": UUID_A}, UUID_A),
    ("bare integer", {"session_id": "1314"}, None),
    ("json number", {"session_id": 1314}, None),
    ("absent", {"role_hint": "user"}, None),
    ("empty string", {"session_id": ""}, None),
    ("json null", {"session_id": None}, None),
    ("no metadata", None, None),
    ("json true", {"session_id": True}, None),
    ("json false", {"session_id": False}, None),
    ("json object", {"session_id": {"nested": UUID_A}}, None),
    ("json array", {"session_id": [UUID_A]}, None),
    ("json float", {"session_id": 1.5}, None),
    ("unicode digits", {"session_id": "١٢٣"}, "١٢٣"),
    ("metadata not an object", [{"session_id": UUID_A}], None),
]


@asynccontextmanager
async def _isolated_schema(db_backend) -> AsyncIterator[AsyncDatabase]:
    """Give the pre-migration table an isolated SQLite DB / Postgres schema."""
    db = AsyncDatabase(db_backend)
    if db.backend_type == "sqlite":
        yield db
        return

    schema = f"session_id_column_{uuid4().hex}"
    await db.execute(f'CREATE SCHEMA "{schema}"')
    try:
        async with db.transaction():
            await db.execute(f'SET LOCAL search_path TO "{schema}"')
            yield db
    finally:
        await db.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')


async def _create_pre_migration_table(db: AsyncDatabase) -> None:
    """The table as it stood before session_id was a column."""
    id_column = (
        "BIGSERIAL PRIMARY KEY"
        if db.backend_type == "postgres"
        else "INTEGER PRIMARY KEY AUTOINCREMENT"
    )
    await db.execute(
        "CREATE TABLE conversation_history ("
        f"id {id_column}, agent_id TEXT NOT NULL DEFAULT '', "
        "role TEXT NOT NULL, content TEXT NOT NULL, metadata TEXT, "
        "created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, "
        "deleted_at TIMESTAMP DEFAULT NULL)"
    )


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_backfill_agrees_with_session_grouping_on_both_backends(db_backend):
    agent_id = f"did:test:session-column:{uuid4()}"
    async with _isolated_schema(db_backend) as db:
        await _create_pre_migration_table(db)
        for label, metadata, _expected in CASES:
            await db.execute(
                "INSERT INTO conversation_history (agent_id, role, content, metadata) "
                "VALUES (?, ?, ?, ?)",
                (
                    agent_id,
                    "user",
                    label,
                    None if metadata is None else json.dumps(metadata),
                ),
            )

        await db._migrate_conversation_session_id_column()

        rows = await db.fetchall(
            "SELECT content, metadata, session_id FROM conversation_history "
            "ORDER BY id",
            (),
        )
        assert [(content, session_id) for content, _meta, session_id in rows] == [
            (label, expected) for label, _metadata, expected in CASES
        ]
        # The same statement of the contract the unit suite makes, now against
        # whichever dialect actually ran.
        for content, metadata, session_id in rows:
            assert session_id == canonical_session_id(metadata), content


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_migration_is_idempotent_on_both_backends(db_backend):
    """Re-running must not disturb a column later writes already own.

    The schema IS the marker for this backfill, so a second run that re-derived
    from metadata would be undetectable in production — assert the column a
    newer write set survives instead.
    """
    agent_id = f"did:test:session-column:{uuid4()}"
    async with _isolated_schema(db_backend) as db:
        await _create_pre_migration_table(db)
        await db.execute(
            "INSERT INTO conversation_history (agent_id, role, content, metadata) "
            "VALUES (?, ?, ?, ?)",
            (agent_id, "user", "legacy", json.dumps({"session_id": "1314"})),
        )

        await db._migrate_conversation_session_id_column()
        # A later write stamps a column value metadata alone would not produce.
        await db.execute(
            "UPDATE conversation_history SET session_id = ? WHERE agent_id = ?",
            (UUID_A, agent_id),
        )
        await db._migrate_conversation_session_id_column()

        assert (
            await db.fetchval(
                "SELECT session_id FROM conversation_history WHERE agent_id = ?",
                (agent_id,),
            )
            == UUID_A
        )


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_index_existence_probe_answers_truthfully_on_both_backends(db_backend):
    """``ensure_index``'s gate is dialect-specific, so both spellings must work.

    A probe that always answered "missing" would leave every boot re-issuing
    the DDL under the lock (degraded, silent); one that always answered
    "present" would mean the index is never built at all — and neither shows up
    as a failure anywhere else, because ``CREATE INDEX IF NOT EXISTS`` shrugs
    either way. The Postgres leg also exercises the ``to_regclass`` search-path
    reasoning against a schema that is NOT the first on the path.
    """
    async with _isolated_schema(db_backend) as db:
        await _create_pre_migration_table(db)
        name = f"idx_probe_{uuid4().hex[:12]}"

        assert await db._index_exists(name) is False
        await db.ensure_index(name, "conversation_history", "agent_id")
        assert await db._index_exists(name) is True
        # Second call is the every-boot-after-the-first path.
        await db.ensure_index(name, "conversation_history", "agent_id")
        assert await db._index_exists(name) is True


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_the_session_index_lands_on_both_backends(db_backend):
    """The migration's own index, through whichever dialect actually ran."""
    async with _isolated_schema(db_backend) as db:
        await _create_pre_migration_table(db)
        await db._migrate_conversation_session_id_column()

        assert await db._index_exists("idx_conversation_agent_session") is True
