"""Regression test for #795 — deleted_at migration on pre-#770 databases.

Pre-#770 agent DBs were created without ``conversation_history.deleted_at``.
``_init_schema`` is responsible for backfilling the column on boot. If this
migration fails silently, every conversation read (which filters on
``WHERE deleted_at IS NULL``) breaks at runtime — see #795.

These tests pin two contracts:

1. An existing DB with a pre-#770 ``conversation_history`` schema (no
   ``deleted_at``) gets the column added when reopened via ``AsyncDatabase``.
2. If the underlying ALTER fails, ``_init_schema`` raises rather than
   swallowing — callers must refuse to bring the agent up rather than start
   half-broken.
"""
import sqlite3
import tempfile
from pathlib import Path

import pytest

from kestrel_sovereign.storage.async_database import AsyncDatabase


PRE_770_CONVERSATION_HISTORY = """
CREATE TABLE conversation_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    agent_id TEXT NOT NULL DEFAULT '',
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    metadata TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
"""


def _seed_pre_770_db(db_path: Path) -> None:
    """Create a DB with a pre-#770 conversation_history (no deleted_at)."""
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute(PRE_770_CONVERSATION_HISTORY)
        conn.execute(
            "INSERT INTO conversation_history (agent_id, role, content) "
            "VALUES (?, ?, ?)",
            ("test-agent", "user", "pre-migration message"),
        )
        conn.commit()
    finally:
        conn.close()


@pytest.mark.asyncio
async def test_pre_770_db_gets_deleted_at_migrated():
    """An existing DB without deleted_at must have the column added on init."""
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "pre770.db"
        _seed_pre_770_db(db_path)

        # Sanity: the seeded DB really does lack deleted_at.
        conn = sqlite3.connect(str(db_path))
        cols = {row[1] for row in conn.execute(
            "PRAGMA table_info(conversation_history)"
        )}
        conn.close()
        assert "deleted_at" not in cols, "test fixture is wrong"

        # Open via AsyncDatabase — _init_schema runs and should add the column.
        db = await AsyncDatabase.sqlite(str(db_path))
        try:
            row = await db.fetchone(
                "SELECT COUNT(*) FROM pragma_table_info('conversation_history') "
                "WHERE name = 'deleted_at'"
            )
            assert row and row[0] == 1, "deleted_at column not added by _init_schema"

            # The downstream filter that #795 reported as failing must now work.
            await db.fetchall(
                "SELECT id FROM conversation_history "
                "WHERE agent_id = ? AND deleted_at IS NULL",
                ("test-agent",),
            )
        finally:
            await db.close()


@pytest.mark.asyncio
async def test_init_schema_raises_on_alter_failure(monkeypatch):
    """If the ALTER fails, _init_schema must raise — never swallow.

    Pins the post-#795 contract that PR #861's earlier graceful-degradation
    shape explicitly does NOT survive: a failed migration leaves the schema
    broken (every read filters on deleted_at), so the agent must refuse to
    boot rather than continue with half-broken state.
    """
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "fail.db"
        _seed_pre_770_db(db_path)

        db = AsyncDatabase.__new__(AsyncDatabase)

        async def _ctor():
            return await AsyncDatabase.sqlite(str(db_path))

        # Build a real instance up to but not including _init_schema, then
        # poison _migrate_add_column to simulate a backend failure on ALTER.
        from kestrel_sovereign.storage.async_database import AsyncDatabase as AD

        orig_migrate = AD._migrate_add_column

        async def boom(self, table, column, col_def):
            raise RuntimeError("simulated backend failure on ALTER")

        monkeypatch.setattr(AD, "_migrate_add_column", boom)

        with pytest.raises(RuntimeError, match="simulated backend failure"):
            await AD.sqlite(str(db_path))
