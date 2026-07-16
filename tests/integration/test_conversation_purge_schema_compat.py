"""Hard-purge behavior across additive lexical-index schema states."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any, AsyncIterator
from uuid import uuid4

import pytest

from kestrel_sovereign.storage.async_conversation_store import (
    AsyncConversationStore,
    ConversationLexicalSchemaError,
)
from kestrel_sovereign.storage.async_database import AsyncDatabase
from kestrel_sovereign.storage.db import TransactionError
from kestrel_sovereign.storage.sqla.migrations import (
    migrate_conversation_lexical_index,
)


class _RecordingAudit:
    def __init__(self) -> None:
        self.events: list[Any] = []

    async def append(self, event: Any) -> int:
        self.events.append(event)
        return len(self.events)


@asynccontextmanager
async def _isolated_schema(db_backend) -> AsyncIterator[AsyncDatabase]:
    """Give each partial-schema case an isolated SQLite DB/Postgres schema."""
    db = AsyncDatabase(db_backend)
    if db.backend_type == "sqlite":
        yield db
        return

    schema = f"purge_schema_{uuid4().hex}"
    await db.execute(f'CREATE SCHEMA "{schema}"')
    try:
        async with db.transaction():
            # The capability probe uses to_regclass(), so it observes this
            # transaction-local search path on the same task-owned connection.
            await db.execute(f'SET LOCAL search_path TO "{schema}"')
            yield db
    finally:
        await db.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')


async def _create_history_table(
    db: AsyncDatabase,
    *,
    lexical_owner_column: bool,
) -> None:
    id_column = (
        "BIGSERIAL PRIMARY KEY"
        if db.backend_type == "postgres"
        else "INTEGER PRIMARY KEY AUTOINCREMENT"
    )
    lexical_column = ", lexical_index_id TEXT" if lexical_owner_column else ""
    await db.execute(
        "CREATE TABLE conversation_history ("
        f"id {id_column}, agent_id TEXT NOT NULL DEFAULT '', "
        "role TEXT NOT NULL, content TEXT NOT NULL, metadata TEXT, "
        "created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, "
        "deleted_at TIMESTAMP DEFAULT NULL, "
        f"archived_at TIMESTAMP DEFAULT NULL{lexical_column})"
    )


async def _create_token_table(db: AsyncDatabase) -> None:
    await db.execute(
        "CREATE TABLE conversation_lexical_tokens ("
        "agent_id TEXT NOT NULL, lexical_index_id TEXT NOT NULL, "
        "token_hash TEXT NOT NULL, "
        "PRIMARY KEY (agent_id, lexical_index_id, token_hash))"
    )


@pytest.mark.asyncio
@pytest.mark.dual_backend
@pytest.mark.parametrize(
    ("lexical_owner_column", "token_table"),
    [(False, False), (True, False), (False, True), (True, True)],
    ids=["neither", "history-only", "table-only", "both"],
)
async def test_hard_purge_distinguishes_degraded_and_inconsistent_schemas(
    db_backend,
    lexical_owner_column,
    token_table,
):
    agent_id = f"did:test:purge-schema:{uuid4()}"
    async with _isolated_schema(db_backend) as db:
        await _create_history_table(
            db,
            lexical_owner_column=lexical_owner_column,
        )
        if token_table:
            await _create_token_table(db)

        columns = "agent_id, role, content"
        params: tuple[Any, ...] = (agent_id, "user", "private history")
        if lexical_owner_column:
            columns += ", lexical_index_id"
            params = (*params, "legacy-key")
        placeholders = ",".join("?" for _ in params)
        await db.execute(
            f"INSERT INTO conversation_history ({columns}) VALUES ({placeholders})",
            params,
        )
        if token_table:
            await db.execute(
                "INSERT INTO conversation_lexical_tokens "
                "(agent_id, lexical_index_id, token_hash) VALUES (?, ?, ?)",
                (agent_id, "legacy-key", "digest"),
            )

        audit = _RecordingAudit()
        store = AsyncConversationStore(
            db,
            agent_id=agent_id,
            destructive_audit=audit,
        )
        if token_table and not lexical_owner_column:
            with pytest.raises(
                ConversationLexicalSchemaError,
                match="lexical-token ownership cannot be proven",
            ):
                await store.purge_all()
            assert audit.events == []
            assert await db.fetchval("SELECT COUNT(*) FROM conversation_history") == 1
            assert (
                await db.fetchval("SELECT COUNT(*) FROM conversation_lexical_tokens")
                == 1
            )
            return

        assert await store.purge_all() == 1
        assert len(audit.events) == 1
        assert await db.fetchval("SELECT COUNT(*) FROM conversation_history") == 0
        if token_table:
            assert (
                await db.fetchval("SELECT COUNT(*) FROM conversation_lexical_tokens")
                == 0
            )


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_rolled_back_lexical_migration_keeps_hard_purge_available(
    db_backend,
    monkeypatch,
):
    agent_id = f"did:test:purge-rollback:{uuid4()}"
    async with _isolated_schema(db_backend) as db:
        await _create_history_table(db, lexical_owner_column=False)
        await db.execute(
            "INSERT INTO conversation_history (agent_id, role, content) "
            "VALUES (?, ?, ?)",
            (agent_id, "user", "must remain purgeable"),
        )

        original_execute = db.execute

        async def fail_mid_migration(sql: str, params: tuple = ()) -> int:
            if "idx_conversation_lexical_token_lookup" in sql:
                raise RuntimeError("forced additive migration failure")
            return await original_execute(sql, params)

        monkeypatch.setattr(db, "execute", fail_mid_migration)
        with pytest.raises(
            (RuntimeError, TransactionError),
            match="forced additive migration failure",
        ):
            await migrate_conversation_lexical_index(db)
        monkeypatch.setattr(db, "execute", original_execute)

        if db.backend_type == "postgres":
            assert not await db.fetchval(
                "SELECT to_regclass(?) IS NOT NULL",
                ("conversation_lexical_tokens",),
            )
            assert not await db.fetchval(
                "SELECT EXISTS ("
                "SELECT 1 FROM pg_attribute "
                "WHERE attrelid = to_regclass(?) AND attname = ? "
                "AND attnum > 0 AND NOT attisdropped)",
                ("conversation_history", "lexical_index_id"),
            )
        else:
            assert not await db.table_exists("conversation_lexical_tokens")
            assert (
                await db.fetchone(
                    "SELECT 1 FROM pragma_table_info('conversation_history') "
                    "WHERE name = ? LIMIT 1",
                    ("lexical_index_id",),
                )
                is None
            )

        audit = _RecordingAudit()
        store = AsyncConversationStore(
            db,
            agent_id=agent_id,
            destructive_audit=audit,
        )
        assert await store.purge_all() == 1
        assert len(audit.events) == 1
        assert await db.fetchval("SELECT COUNT(*) FROM conversation_history") == 0
