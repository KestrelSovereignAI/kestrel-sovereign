import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest
from kestrel_sdk.tools.result import ToolResultStatus

from kestrel_sovereign.features.memory.feature import MemoryFeature


def _feature_with_store(monkeypatch, store):
    feature = MemoryFeature(agent=MagicMock())
    feature.agent_id = "did:test:lexical-feature"
    feature._lexical_backfill_task = None
    feature._lexical_backfill_result = {"status": "idle"}
    monkeypatch.setattr(feature, "_get_conversation_store", lambda: store)
    return feature


@pytest.mark.asyncio
async def test_memory_index_backfill_starts_background_and_records_result(monkeypatch):
    store = MagicMock()
    store.backfill_lexical_index = AsyncMock(return_value={
        "indexed": 12,
        "remaining": 0,
        "coverage": 1.0,
    })
    feature = _feature_with_store(monkeypatch, store)

    result = await feature.memory_index_backfill(batch_size=37)
    await feature._lexical_backfill_task

    assert result.status is ToolResultStatus.OK
    store.backfill_lexical_index.assert_awaited_once_with(batch_size=37)
    assert feature._lexical_backfill_result == {
        "status": "done",
        "indexed": 12,
        "remaining": 0,
        "coverage": 1.0,
    }


@pytest.mark.asyncio
async def test_memory_index_backfill_rejects_duplicate_running_job(monkeypatch):
    gate = asyncio.Event()
    store = MagicMock()

    async def blocked_backfill(*, batch_size):
        await gate.wait()
        return {"indexed": 1, "remaining": 0, "coverage": 1.0}

    store.backfill_lexical_index = AsyncMock(side_effect=blocked_backfill)
    feature = _feature_with_store(monkeypatch, store)

    await feature.memory_index_backfill()
    duplicate = await feature.memory_index_backfill()
    gate.set()
    await feature._lexical_backfill_task

    assert duplicate.status is ToolResultStatus.OK
    assert "already running" in duplicate.confirmation
    assert store.backfill_lexical_index.await_count == 1


@pytest.mark.asyncio
async def test_memory_index_backfill_validates_batch_size(monkeypatch):
    store = MagicMock()
    feature = _feature_with_store(monkeypatch, store)

    result = await feature.memory_index_backfill(batch_size=0)

    assert result.status is ToolResultStatus.ERROR
    assert "batch_size must be in" in result.error


@pytest.mark.asyncio
async def test_shutdown_cancels_owned_backfill(monkeypatch):
    gate = asyncio.Event()
    store = MagicMock()

    async def blocked_backfill(*, batch_size):
        await gate.wait()

    store.backfill_lexical_index = AsyncMock(side_effect=blocked_backfill)
    feature = _feature_with_store(monkeypatch, store)
    await feature.memory_index_backfill()

    await feature.shutdown()

    assert feature._lexical_backfill_task.cancelled()


@pytest.mark.asyncio
async def test_memory_status_survives_unavailable_index_health(monkeypatch):
    store = MagicMock()
    store.agent_id = "did:test:lexical-feature"
    store.encryption_enabled = True
    store.get_lexical_index_health = AsyncMock(
        side_effect=RuntimeError("lexical migration incomplete")
    )
    store.get_embedding_profile_health = AsyncMock(
        side_effect=RuntimeError("profile migration incomplete")
    )
    feature = _feature_with_store(monkeypatch, store)
    feature._db = MagicMock()
    feature._db.fetchone = AsyncMock(side_effect=[(5,), (2,)])
    feature.storage = object()
    feature._memory_system = None
    feature.agent.memory_system = None

    result = await feature.memory_status()

    assert result.status is ToolResultStatus.OK
    assert result.data["total_messages"] == 5
    assert result.data["files_stored"] == 2
    assert result.data["lexical_index"]["available"] is False
    assert "migration incomplete" in result.data["lexical_index"]["error"]
    assert result.data["embedding_profiles"]["available"] is False
