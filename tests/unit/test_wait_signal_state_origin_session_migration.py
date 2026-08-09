"""Legacy-database migration for ``wait_signal_state.origin_session_id`` (#2877).

The wait ledger is created with ``CREATE TABLE IF NOT EXISTS``, which never
adds a column to a table that already exists — so every deployment that ran a
pre-#2877 build keeps the old shape. Without the migration, the very first
``start_watch``/``record_pending`` after the upgrade fails on the unknown
column and the wake path breaks for exactly the agents that have been running
longest. These tests pin the reconciliation.
"""

from __future__ import annotations

import pytest
import pytest_asyncio

from kestrel_sovereign.storage.async_database import AsyncDatabase
from kestrel_sovereign.storage.async_wait_signal_store import WaitSignalStore
from kestrel_sovereign.storage.db import SQLiteBackend


@pytest_asyncio.fixture
async def legacy_db(tmp_path):
    """AsyncDatabase whose wait_signal_state predates origin_session_id."""
    raw = SQLiteBackend(str(tmp_path / "legacy-wait-signal-state.db"))
    await raw.connect()
    await raw.execute(
        "CREATE TABLE wait_signal_state ("
        "  agent_id TEXT NOT NULL,"
        "  kind TEXT NOT NULL,"
        "  handle TEXT NOT NULL,"
        "  last_signaled_outcome TEXT,"
        "  last_delivery_status TEXT,"
        "  last_delivery_error TEXT,"
        "  last_delivery_attempts INTEGER NOT NULL DEFAULT 0,"
        "  last_delivery_attempt_at TIMESTAMP,"
        "  pending_signal_id TEXT,"
        "  pending_signaled_target TEXT,"
        "  pending_signal_enqueued_at TIMESTAMP,"
        "  watching INTEGER NOT NULL DEFAULT 0,"
        "  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,"
        "  PRIMARY KEY (agent_id, kind, handle)"
        ")"
    )
    # A row written by the old build — it must survive with a NULL session.
    await raw.execute(
        "INSERT INTO wait_signal_state (agent_id, kind, handle, watching) "
        "VALUES (?, ?, ?, 1)",
        ("did:test:agent", "talon", "legacy-job"),
    )
    db = AsyncDatabase(raw)
    await db._init_schema()
    try:
        yield db
    finally:
        await db.close()


async def _columns(db: AsyncDatabase) -> set:
    rows = await db.fetchall("PRAGMA table_info('wait_signal_state')", ())
    return {row[1] for row in rows}


@pytest.mark.asyncio
async def test_init_schema_adds_origin_session_id(legacy_db):
    assert "origin_session_id" in await _columns(legacy_db)


@pytest.mark.asyncio
async def test_legacy_row_reads_back_with_no_session(legacy_db):
    """Pre-migration rows are genuinely session-less, not silently bound."""
    store = WaitSignalStore(legacy_db, "did:test:agent")
    row = await store.get("talon", "legacy-job")
    assert row is not None
    assert row.origin_session_id is None
    assert row.watching == 1


@pytest.mark.asyncio
async def test_watch_records_session_after_migration(legacy_db):
    store = WaitSignalStore(legacy_db, "did:test:agent")
    await store.start_watch("talon", "legacy-job", origin_session_id="sess-1")
    assert (await store.get("talon", "legacy-job")).origin_session_id == "sess-1"


@pytest.mark.asyncio
async def test_migration_is_idempotent(legacy_db):
    await legacy_db._init_schema()  # second pass must not raise
    assert "origin_session_id" in await _columns(legacy_db)


@pytest.mark.asyncio
async def test_fresh_db_already_has_the_column(tmp_path):
    db = await AsyncDatabase.sqlite(str(tmp_path / "fresh.db"))
    try:
        assert "origin_session_id" in await _columns(db)
    finally:
        await db.close()
