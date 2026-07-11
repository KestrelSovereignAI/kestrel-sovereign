from unittest.mock import AsyncMock, MagicMock

import pytest

from kestrel_sovereign.storage.async_database import AsyncDatabase
from kestrel_sovereign.storage.db import SQLiteBackend
from kestrel_sovereign.storage.sqla.migrations import (
    migrate_conversation_lexical_index,
)


@pytest.mark.asyncio
async def test_sqlite_legacy_schema_migration_is_idempotent(tmp_path):
    raw = SQLiteBackend(str(tmp_path / "legacy-lexical.db"))
    await raw.connect()
    db = AsyncDatabase(raw)
    try:
        await db.execute(
            "CREATE TABLE conversation_history ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, agent_id TEXT NOT NULL, "
            "role TEXT NOT NULL, content TEXT NOT NULL, deleted_at TIMESTAMP, "
            "archived_at TIMESTAMP)"
        )

        await migrate_conversation_lexical_index(db)
        await migrate_conversation_lexical_index(db)

        columns = {
            row[1] for row in await db.fetchall(
                "PRAGMA table_info('conversation_history')"
            )
        }
        tables = {
            row[0] for row in await db.fetchall(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        indexes = {
            row[0] for row in await db.fetchall(
                "SELECT name FROM sqlite_master WHERE type = 'index'"
            )
        }
        assert {"lexical_index_id", "lexical_index_version"} <= columns
        assert "conversation_lexical_tokens" in tables
        assert "idx_conversation_lexical_token_lookup" in indexes
        assert "idx_conversation_lexical_coverage" in indexes
        assert "idx_conversation_lexical_message" in indexes
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_postgres_migration_uses_additive_transactional_ddl():
    db = MagicMock()
    db.backend_type = "postgres"
    db.execute = AsyncMock()

    class Transaction:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return False

    db.transaction = MagicMock(return_value=Transaction())

    await migrate_conversation_lexical_index(db)

    statements = [call.args[0] for call in db.execute.await_args_list]
    assert any("ADD COLUMN IF NOT EXISTS lexical_index_id" in sql for sql in statements)
    assert any("ADD COLUMN IF NOT EXISTS lexical_index_version" in sql for sql in statements)
    assert any("CREATE TABLE IF NOT EXISTS conversation_lexical_tokens" in sql for sql in statements)
    assert any("idx_conversation_lexical_coverage" in sql for sql in statements)
    assert any("idx_conversation_lexical_message" in sql for sql in statements)
    db.transaction.assert_called_once_with()
