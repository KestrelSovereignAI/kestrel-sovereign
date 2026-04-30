"""Regression test for #795: deleted_at migration silently fails on legacy DBs.

Builds a SQLite database that mirrors the pre-#770 schema (no
``deleted_at`` column, no dependent index), then opens it through
``AsyncDatabase.sqlite()`` to drive ``_init_schema``. Asserts:

* The migration adds the column.
* The dependent index ``idx_conversation_deleted_at`` is created.
* A real migration failure surfaces instead of being swallowed at
  debug level (the original bug — see issue #795).
"""
from __future__ import annotations

import sqlite3

import pytest

from kestrel_sovereign.storage.async_database import AsyncDatabase


LEGACY_SCHEMA = """
CREATE TABLE conversation_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    agent_id TEXT NOT NULL DEFAULT '',
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    metadata TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX idx_conversation_agent_id ON conversation_history(agent_id);
"""


def _seed_legacy_db(path) -> None:
    conn = sqlite3.connect(path)
    try:
        conn.executescript(LEGACY_SCHEMA)
        conn.execute(
            "INSERT INTO conversation_history (agent_id, role, content) "
            "VALUES (?, ?, ?)",
            ("did:test:legacy", "user", "pre-soft-delete row"),
        )
        conn.commit()
    finally:
        conn.close()


@pytest.mark.asyncio
async def test_legacy_db_loads_and_migrates_deleted_at(tmp_path):
    db_path = tmp_path / "legacy.db"
    _seed_legacy_db(str(db_path))

    db = await AsyncDatabase.sqlite(str(db_path))
    try:
        cols = await db.fetchall("PRAGMA table_info(conversation_history)")
        col_names = {row[1] for row in cols}
        assert "deleted_at" in col_names, (
            "Migration did not add deleted_at column on legacy DB"
        )

        idx_rows = await db.fetchall(
            "SELECT name FROM sqlite_master WHERE type='index' "
            "AND tbl_name='conversation_history'"
        )
        idx_names = {row[0] for row in idx_rows}
        assert "idx_conversation_deleted_at" in idx_names, (
            "Dependent index was not created after migration"
        )

        existing = await db.fetchall(
            "SELECT role, content, deleted_at FROM conversation_history"
        )
        assert len(existing) == 1
        assert existing[0][0] == "user"
        assert existing[0][2] is None
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_migration_failure_surfaces(tmp_path, monkeypatch):
    """Real migration failures must propagate, not get logged at debug.

    Pre-fix, ``_migrate_add_column`` swallowed every Exception so an
    underlying DB failure became invisible — and then the dependent
    index creation crashed with the misleading ``no such column`` error.
    Now the helper re-raises real failures.
    """
    db_path = tmp_path / "fresh.db"
    db = await AsyncDatabase.sqlite(str(db_path))
    try:
        async def _boom(*_args, **_kwargs):
            raise sqlite3.OperationalError("disk I/O error")

        monkeypatch.setattr(db._backend, "execute", _boom)

        with pytest.raises(sqlite3.OperationalError, match="disk I/O error"):
            await db._migrate_add_column(
                "conversation_history", "extra_col", "TEXT"
            )
    finally:
        await db.close()
