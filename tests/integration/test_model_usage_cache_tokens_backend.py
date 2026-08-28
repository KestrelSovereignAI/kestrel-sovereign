"""Real SQLite/PostgreSQL coverage for model cache-token usage (#3019)."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from typing import AsyncIterator
from uuid import uuid4

import pytest

from kestrel_sovereign.llm.usage_tracking import UsageTrackingMixin
from kestrel_sovereign.storage.async_database import AsyncDatabase


pytestmark = pytest.mark.integration


@asynccontextmanager
async def _fresh_schema(db_backend) -> AsyncIterator[AsyncDatabase]:
    db = AsyncDatabase(db_backend)
    if db.backend_type == "sqlite":
        yield db
        return

    schema = f"model_usage_cache_{uuid4().hex}"
    await db.execute(f'CREATE SCHEMA "{schema}"')
    try:
        async with db.transaction():
            await db.execute(f'SET LOCAL search_path TO "{schema}"')
            yield db
    finally:
        await db.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_fresh_schema_accumulates_reported_cache_tokens(db_backend) -> None:
    async with _fresh_schema(db_backend) as db:
        # Reproduce the last release's table in an otherwise fresh database so
        # both engines exercise the additive ALTER, not only greenfield DDL.
        await db.execute(
            "CREATE TABLE model_usage ("
            "model_id TEXT PRIMARY KEY, provider TEXT NOT NULL, "
            "last_used TIMESTAMP NOT NULL, use_count INTEGER DEFAULT 0, "
            "total_tokens INTEGER DEFAULT 0, "
            "created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)"
        )
        await db.execute(
            "INSERT INTO model_usage "
            "(model_id, provider, last_used, use_count, total_tokens) "
            "VALUES (?, ?, CURRENT_TIMESTAMP, 1, ?)",
            ("legacy-cache-test", "anthropic", 13),
        )
        await db._init_schema()

        tracker = UsageTrackingMixin()
        tracker._usage_db = db
        tracker._db_initialized = True

        await tracker._track_model_usage(
            "claude-cache-test",
            "anthropic",
            tokens=29,
            cache_creation_input_tokens=3_000_000_019,
            cache_read_input_tokens=7,
        )
        await tracker._track_model_usage(
            "claude-cache-test",
            "anthropic",
            tokens=5,
        )
        await tracker._track_model_usage(
            "claude-cache-test",
            "anthropic",
            tokens=4,
            cache_creation_input_tokens=2,
            cache_read_input_tokens=3,
        )

        assert await db.fetchone(
            "SELECT provider, use_count, total_tokens, "
            "cache_creation_input_tokens, cache_read_input_tokens "
            "FROM model_usage WHERE model_id = ?",
            ("claude-cache-test",),
        ) == ("anthropic", 3, 38, 3_000_000_021, 10)
        assert await db.fetchone(
            "SELECT total_tokens, cache_creation_input_tokens, "
            "cache_read_input_tokens FROM model_usage WHERE model_id = ?",
            ("legacy-cache-test",),
        ) == (13, 0, 0)

        period_start = datetime.now(UTC).replace(
            hour=0, minute=0, second=0, microsecond=0, tzinfo=None
        )
        period_end = period_start + timedelta(days=1)
        assert await db.fetchone(
            "SELECT provider, use_count, total_tokens, "
            "cache_creation_input_tokens, cache_read_input_tokens "
            "FROM model_usage_periods WHERE model_id = ? "
            "AND period_start >= ? AND period_start < ?",
            ("claude-cache-test", period_start, period_end),
        ) == ("anthropic", 3, 38, 3_000_000_021, 10)

        # Exercise the exact old-revision statement after migration. A rolling
        # deployment may keep serving from an older process while this schema
        # is live, and telemetry must not begin failing silently underneath it.
        await db.execute(
            "INSERT INTO model_usage "
            "(model_id, provider, last_used, use_count, total_tokens, created_at) "
            "VALUES (?, ?, CURRENT_TIMESTAMP, 1, ?, CURRENT_TIMESTAMP) "
            "ON CONFLICT(model_id) DO UPDATE SET "
            "last_used = CURRENT_TIMESTAMP, "
            "use_count = model_usage.use_count + 1, "
            "total_tokens = model_usage.total_tokens + ?",
            ("claude-cache-test", "anthropic", 11, 11),
        )
        assert await db.fetchone(
            "SELECT use_count, total_tokens, cache_creation_input_tokens, "
            "cache_read_input_tokens FROM model_usage WHERE model_id = ?",
            ("claude-cache-test",),
        ) == (4, 49, 3_000_000_021, 10)
        # Old writers have no cache telemetry, so they leave the new period's
        # cache totals untouched while continuing to update the lifetime row.
        assert await db.fetchone(
            "SELECT use_count, total_tokens, cache_creation_input_tokens, "
            "cache_read_input_tokens FROM model_usage_periods "
            "WHERE model_id = ? AND period_start = ?",
            ("claude-cache-test", period_start),
        ) == (3, 38, 3_000_000_021, 10)

        await tracker._track_model_usage(
            "shared-model", "openai:api", tokens=10,
            cache_read_input_tokens=8,
        )
        await tracker._track_model_usage(
            "shared-model", "openai:plan", tokens=12,
            cache_read_input_tokens=11,
        )
        assert await db.fetchall(
            "SELECT provider, total_tokens, cache_read_input_tokens "
            "FROM model_usage_periods WHERE model_id = ? "
            "AND period_start >= ? AND period_start < ? ORDER BY provider",
            ("shared-model", period_start, period_end),
        ) == [
            ("openai:api", 10, 8),
            ("openai:plan", 12, 11),
        ]


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_postgres_concurrent_period_schema_initializers_converge(
    db_backend,
) -> None:
    """Eight real pool connections must survive the first upgrade boot."""
    if db_backend.backend_type != "postgres":
        pytest.skip("PostgreSQL catalogue race")

    import asyncpg

    from kestrel_sovereign.storage.db.postgres import PostgresBackend

    schema = f"model_usage_period_race_{uuid4().hex}"
    await db_backend.execute(f'CREATE SCHEMA "{schema}"')
    pool = await asyncpg.create_pool(
        db_backend._dsn,
        min_size=4,
        max_size=12,
        server_settings={"search_path": schema},
    )
    db = AsyncDatabase(PostgresBackend.from_pool(pool))
    try:
        await asyncio.gather(
            *(db._ensure_model_usage_periods() for _ in range(8))
        )

        assert await db.fetchone(
            "SELECT COUNT(*) FROM pg_class c "
            "JOIN pg_namespace n ON n.oid = c.relnamespace "
            "WHERE n.nspname = ? AND c.relname = ? AND c.relkind = 'r'",
            (schema, "model_usage_periods"),
        ) == (1,)
        indexes = await db.fetchall(
            "SELECT indexname FROM pg_indexes "
            "WHERE schemaname = ? AND tablename = ?",
            (schema, "model_usage_periods"),
        )
        assert sum(
            row[0].startswith("idx_model_usage_periods_start_")
            for row in indexes
        ) == 1
    finally:
        await db.close()
        await pool.close()
        await db_backend.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
