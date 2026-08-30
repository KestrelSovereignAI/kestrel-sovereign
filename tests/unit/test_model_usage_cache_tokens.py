"""Cache-token observability in the model-usage store (#3019)."""

from __future__ import annotations

import asyncio
import sqlite3
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, patch

import pytest

from kestrel_sovereign.llm.usage_tracking import UsageTrackingMixin
from kestrel_sovereign.storage.async_database import AsyncDatabase
from kestrel_sovereign.storage.db.interface import QueryError, TransactionError


PRE_CACHE_SCHEMA = """
CREATE TABLE model_usage (
    model_id TEXT PRIMARY KEY,
    provider TEXT NOT NULL,
    last_used TIMESTAMP NOT NULL,
    use_count INTEGER DEFAULT 0,
    total_tokens INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
INSERT INTO model_usage
    (model_id, provider, last_used, use_count, total_tokens)
VALUES
    ('legacy-model', 'anthropic', '2026-08-01 00:00:00', 3, 120);
"""


@pytest.mark.asyncio
async def test_usage_transaction_retries_postgres_concurrent_update() -> None:
    """A poisoned PostgreSQL transaction must be replayed as one whole unit."""

    class FlakyPostgresUsageDB:
        backend_type = "postgres"

        def __init__(self) -> None:
            self.attempt = 0
            self.executed: list[tuple[int, str]] = []

        @asynccontextmanager
        async def transaction(self):
            self.attempt += 1
            try:
                yield
            except Exception as exc:
                raise TransactionError(f"Transaction failed: {exc}") from exc

        async def execute(self, sql, _params=()):
            table = (
                "period"
                if "INSERT INTO model_usage_periods" in sql
                else "lifetime"
            )
            self.executed.append((self.attempt, table))
            if self.attempt == 1 and table == "period":
                raise QueryError("Query failed: tuple concurrently updated")

    tracker = UsageTrackingMixin()
    tracker._usage_db = FlakyPostgresUsageDB()
    tracker._db_initialized = True

    await tracker._track_model_usage(
        "claude-cache-test",
        "anthropic",
        tokens=13,
        cache_read_input_tokens=8,
    )

    assert tracker._usage_db.executed == [
        (1, "lifetime"),
        (1, "period"),
        (2, "lifetime"),
        (2, "period"),
    ]


@pytest.mark.asyncio
async def test_preexisting_usage_database_gains_cache_columns_without_data_loss(
    tmp_path,
) -> None:
    db_path = tmp_path / "pre-cache-observability.db"
    connection = sqlite3.connect(db_path)
    try:
        connection.executescript(PRE_CACHE_SCHEMA)
        connection.commit()
    finally:
        connection.close()

    db = await AsyncDatabase.sqlite(str(db_path))
    try:
        columns = {
            row[1]: row for row in await db.fetchall("PRAGMA table_info(model_usage)")
        }
        assert "cache_creation_input_tokens" in columns
        assert "cache_read_input_tokens" in columns
        assert "cache_creation_input_tokens_report_count" in columns
        assert "cache_read_input_tokens_report_count" in columns
        assert columns["cache_creation_input_tokens"][2] == "BIGINT"
        assert columns["cache_read_input_tokens"][2] == "BIGINT"
        assert columns["cache_creation_input_tokens_report_count"][2] == "BIGINT"
        assert columns["cache_read_input_tokens_report_count"][2] == "BIGINT"
        # #3019 is an additive migration. Retaining the legacy model_id key is
        # what lets old-revision ON CONFLICT(model_id) writers coexist with a
        # newly migrated process during a rolling deployment.
        assert [row[1] for row in columns.values() if row[5]] == ["model_id"]
        assert await db.fetchone(
            "SELECT COUNT(*) FROM model_usage_periods"
        ) == (0,)
        assert await db.fetchone(
            "SELECT provider, use_count, total_tokens, "
            "cache_creation_input_tokens, cache_read_input_tokens, "
            "cache_creation_input_tokens_report_count, "
            "cache_read_input_tokens_report_count "
            "FROM model_usage WHERE model_id = ?",
            ("legacy-model",),
        ) == ("anthropic", 3, 120, 0, 0, 0, 0)
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_concurrent_period_schema_initializers_create_the_table_once(
    tmp_path,
) -> None:
    """The first post-upgrade request burst must not race table creation."""
    db = await AsyncDatabase.sqlite(str(tmp_path / "period-schema-race.db"))
    try:
        await db.execute("DROP TABLE model_usage_periods")
        creates: list[str] = []
        real_execute = db.execute

        async def recording_execute(sql, params=()):
            if (
                sql.lstrip().startswith("CREATE TABLE")
                and "model_usage_periods" in sql
            ):
                creates.append(sql)
                # Make the pre-lock scheduling window deterministic enough that
                # all contenders can reach it; the lock/re-probe still decides
                # whether more than one CREATE is attempted.
                await asyncio.sleep(0)
            return await real_execute(sql, params)

        with (
            patch.object(db, "execute", recording_execute),
            patch.object(db, "ensure_index", new=AsyncMock()),
        ):
            await asyncio.gather(
                *(db._ensure_model_usage_periods() for _ in range(4))
            )

        assert len(creates) == 1
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_preexisting_period_table_gains_report_availability_counts(
    tmp_path,
) -> None:
    """A development-era #3019 table must preserve rows while upgrading."""
    db = await AsyncDatabase.sqlite(str(tmp_path / "period-report-counts.db"))
    try:
        await db.execute("DROP TABLE model_usage_periods")
        await db.execute("""
            CREATE TABLE model_usage_periods (
                period_start TIMESTAMP NOT NULL,
                model_id TEXT NOT NULL,
                provider TEXT NOT NULL,
                use_count INTEGER NOT NULL DEFAULT 0,
                total_tokens BIGINT NOT NULL DEFAULT 0,
                cache_creation_input_tokens BIGINT NOT NULL DEFAULT 0,
                cache_read_input_tokens BIGINT NOT NULL DEFAULT 0,
                PRIMARY KEY (period_start, model_id, provider)
            )
        """)
        await db.execute(
            "INSERT INTO model_usage_periods "
            "(period_start, model_id, provider, use_count, total_tokens, "
            "cache_creation_input_tokens, cache_read_input_tokens) "
            "VALUES (CURRENT_TIMESTAMP, ?, ?, 2, 20, 4, 8)",
            ("development-model", "anthropic"),
        )

        await db._ensure_model_usage_periods()

        columns = {
            row[1]: row
            for row in await db.fetchall("PRAGMA table_info(model_usage_periods)")
        }
        assert columns["cache_creation_input_tokens_report_count"][2] == "BIGINT"
        assert columns["cache_read_input_tokens_report_count"][2] == "BIGINT"
        assert await db.fetchone(
            "SELECT use_count, total_tokens, cache_creation_input_tokens, "
            "cache_read_input_tokens, cache_creation_input_tokens_report_count, "
            "cache_read_input_tokens_report_count FROM model_usage_periods "
            "WHERE model_id = ?",
            ("development-model",),
        ) == (2, 20, 4, 8, 0, 0)
    finally:
        await db.close()
