"""Regression: agent_service_keys schema drift on legacy databases.

``agent_service_keys`` gained ``key_hash`` and quota columns after its initial
release, but the table is created via ``CREATE TABLE IF NOT EXISTS``, which
never adds columns to a table that already exists. A database created before
those columns shipped therefore kept the old shape, and
``ServiceKeyStorage.store_key`` — whose INSERT names ``key_hash, quota_limit,
quota_used, is_active`` — failed with::

    column "key_hash" of relation "agent_service_keys" does not exist

Because per-user provider-key injection is fail-open, that error was swallowed
and callers silently fell back to the shared platform key, disabling per-user
metering/caps with no hard failure. ``_init_schema`` now runs idempotent
``_migrate_add_column`` calls to reconcile the legacy shape; these tests pin
that behaviour.
"""

from typing import Iterator

import pytest
import pytest_asyncio

from kestrel_sovereign.security.service_key_storage import ServiceKeyStorage
from kestrel_sovereign.storage.async_database import AsyncDatabase
from kestrel_sovereign.storage.db import SQLiteBackend

_MIGRATED_COLUMNS = ("key_hash", "quota_limit", "quota_used", "is_active")


@pytest.fixture(autouse=True)
def _kestrel_data_key(monkeypatch) -> Iterator[None]:
    """Fixed KESTREL_DATA_KEY so ServiceKeyStorage encryption works."""
    monkeypatch.setenv("KESTREL_DATA_KEY", "test-master-key-32-bytes-fixed--")
    yield


@pytest_asyncio.fixture
async def legacy_db(tmp_path):
    """AsyncDatabase whose agent_service_keys predates key_hash/quota columns.

    The pre-existing legacy table makes ``CREATE TABLE IF NOT EXISTS`` a no-op,
    so only the migration can add the missing columns — exactly the drift a
    real upgraded deployment hits. Yields so the aiosqlite worker thread is
    closed before its event loop exits (no leaked-thread warnings).
    """
    raw = SQLiteBackend(str(tmp_path / "legacy-agent-service-keys.db"))
    await raw.connect()
    # Original pre-drift shape: no key_hash, quota_limit, quota_used, is_active.
    await raw.execute(
        "CREATE TABLE agent_service_keys ("
        "  id TEXT PRIMARY KEY,"
        "  agent_did TEXT NOT NULL,"
        "  provider_id TEXT NOT NULL,"
        "  encrypted_key TEXT NOT NULL,"
        "  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,"
        "  UNIQUE(agent_did, provider_id)"
        ")"
    )
    db = AsyncDatabase(raw)
    await db._init_schema()
    try:
        yield db
    finally:
        await db.close()


async def _columns(db: AsyncDatabase) -> set:
    rows = await db.fetchall("PRAGMA table_info('agent_service_keys')", ())
    # PRAGMA table_info columns: (cid, name, type, notnull, dflt_value, pk)
    return {row[1] for row in rows}


@pytest.mark.asyncio
async def test_init_schema_backfills_missing_columns(legacy_db):
    cols = await _columns(legacy_db)
    for col in _MIGRATED_COLUMNS:
        assert col in cols, f"{col} was not added to legacy agent_service_keys"


@pytest.mark.asyncio
async def test_store_key_roundtrips_after_migration(legacy_db):
    """The real payoff: store_key's INSERT (which names key_hash) now succeeds."""
    storage = ServiceKeyStorage(legacy_db, "did:web:agents.frinz.ai:kestrel-agent-test")

    await storage.store_key("openrouter", "sk-or-v1-secret-value")

    assert await storage.has_key("openrouter") is True
    assert await storage.get_key("openrouter") == "sk-or-v1-secret-value"


@pytest.mark.asyncio
async def test_migration_is_idempotent(legacy_db):
    """Re-running _init_schema on an already-migrated DB is a clean no-op."""
    await legacy_db._init_schema()  # second pass must not raise
    cols = await _columns(legacy_db)
    for col in _MIGRATED_COLUMNS:
        assert col in cols


@pytest.mark.asyncio
async def test_fresh_db_already_has_columns(tmp_path):
    """A DB created from CORE_SCHEMA (no legacy table) is unaffected."""
    db = await AsyncDatabase.sqlite(str(tmp_path / "fresh.db"))
    try:
        cols = await _columns(db)
        for col in _MIGRATED_COLUMNS:
            assert col in cols
    finally:
        await db.close()
