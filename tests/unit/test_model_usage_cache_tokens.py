"""Cache-token observability in the model-usage store (#3019)."""

from __future__ import annotations

import asyncio
import sqlite3
from unittest.mock import AsyncMock, patch

import pytest

from kestrel_sovereign.storage.async_database import AsyncDatabase


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
        assert columns["cache_creation_input_tokens"][2] == "BIGINT"
        assert columns["cache_read_input_tokens"][2] == "BIGINT"
        # #3019 is an additive migration. Retaining the legacy model_id key is
        # what lets old-revision ON CONFLICT(model_id) writers coexist with a
        # newly migrated process during a rolling deployment.
        assert [row[1] for row in columns.values() if row[5]] == ["model_id"]
        assert await db.fetchone(
            "SELECT COUNT(*) FROM model_usage_periods"
        ) == (0,)
        assert await db.fetchone(
            "SELECT provider, use_count, total_tokens, "
            "cache_creation_input_tokens, cache_read_input_tokens "
            "FROM model_usage WHERE model_id = ?",
            ("legacy-model",),
        ) == ("anthropic", 3, 120, 0, 0)
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
