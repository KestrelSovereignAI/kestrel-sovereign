"""Regression: wait_signal_state schema drift on legacy databases (#2922).

``wait_signal_state`` gained ``last_surface_status`` and
``pending_signal_session_id`` when the reconciler's delivery accounting split
"persisted" from "surfaced". The table is created with ``CREATE TABLE IF NOT
EXISTS``, which never adds columns to an existing table, so every database
that already ran a wait reconcile keeps the old shape — and
:class:`WaitSignalStore` names both new columns in its SELECT projection and
its writes. Without the idempotent ``_migrate_add_column`` calls in
``_init_schema``, the first reconcile tick after an upgrade would fail on
``no such column: last_surface_status`` and stop waking the agent entirely.

These tests pin the migration against a genuinely pre-drift table.
"""

import pytest
import pytest_asyncio

from kestrel_sovereign.storage.async_database import AsyncDatabase
from kestrel_sovereign.storage.async_wait_signal_store import WaitSignalStore
from kestrel_sovereign.storage.db import SQLiteBackend

_MIGRATED_COLUMNS = ("last_surface_status", "pending_signal_session_id")


@pytest_asyncio.fixture
async def legacy_db(tmp_path):
    """AsyncDatabase whose wait_signal_state predates the surface columns."""
    raw = SQLiteBackend(str(tmp_path / "legacy-wait-signal-state.db"))
    await raw.connect()
    # The pre-#2922 shape, verbatim.
    await raw.execute(
        "CREATE TABLE wait_signal_state ("
        "  agent_id TEXT NOT NULL DEFAULT '',"
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
    # A row written by the old code: an in-flight wake with no recorded
    # binding and no recorded visibility.
    await raw.execute(
        "INSERT INTO wait_signal_state "
        "(agent_id, kind, handle, pending_signal_id, pending_signaled_target) "
        "VALUES (?, ?, ?, ?, ?)",
        ("did:test:legacy", "talon", "job-legacy", "sig-old", "done"),
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
async def test_init_schema_backfills_missing_columns(legacy_db):
    cols = await _columns(legacy_db)
    for col in _MIGRATED_COLUMNS:
        assert col in cols, f"{col} was not added to legacy wait_signal_state"


@pytest.mark.asyncio
async def test_legacy_row_reads_as_unknown_not_as_a_claim(legacy_db):
    """A wake enqueued before the columns existed carries no binding and no
    visibility verdict. Both read NULL — which the reconciler reports as
    unbound/unknown rather than inventing an answer."""
    store = WaitSignalStore(legacy_db, "did:test:legacy")
    row = await store.get("talon", "job-legacy")

    assert row is not None
    assert row.pending_signal_id == "sig-old"
    assert row.last_surface_status is None
    assert row.pending_signal_session_id is None


@pytest.mark.asyncio
async def test_store_writes_roundtrip_after_migration(legacy_db):
    """The real payoff: the reconciler's writes, which name both new columns,
    succeed against the upgraded legacy table."""
    store = WaitSignalStore(legacy_db, "did:test:legacy")

    await store.record_pending(
        "talon", "job-legacy",
        signal_id="sig-new", target="done", attempts=1,
        session_id="chat-sess-legacy",
    )
    row = await store.get("talon", "job-legacy")
    assert row.pending_signal_session_id == "chat-sess-legacy"

    await store.record_delivery(
        "talon", "job-legacy",
        delivery_status="ok", signaled_outcome="done",
        surface_status="surfaced",
    )
    row = await store.get("talon", "job-legacy")
    assert row.last_surface_status == "surfaced"
    assert row.pending_signal_session_id is None


@pytest.mark.asyncio
async def test_migration_is_idempotent(legacy_db):
    await legacy_db._init_schema()  # second pass must not raise
    cols = await _columns(legacy_db)
    for col in _MIGRATED_COLUMNS:
        assert col in cols


@pytest.mark.asyncio
async def test_fresh_db_already_has_columns(tmp_path):
    db = await AsyncDatabase.sqlite(str(tmp_path / "fresh.db"))
    try:
        cols = await _columns(db)
        for col in _MIGRATED_COLUMNS:
            assert col in cols
    finally:
        await db.close()
