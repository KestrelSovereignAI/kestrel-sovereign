"""get_episodes tool: relevance (query) vs recency (#1674 P2)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from kestrel_sovereign.features.memory.feature import MemoryFeature
from kestrel_sovereign.storage.memory_models import MemoryEpisode


def _feature_with_consolidator(monkeypatch, consolidator):
    feat = MemoryFeature(agent=MagicMock())
    # `consolidator` is a read-only property; monkeypatch it on the class so it
    # resolves our mock and auto-reverts after the test (no cross-test leak).
    monkeypatch.setattr(
        MemoryFeature, "consolidator",
        property(lambda self: consolidator), raising=True,
    )
    return feat


@pytest.mark.asyncio
async def test_get_episodes_query_uses_semantic_recall(monkeypatch):
    consolidator = MagicMock()
    consolidator.search_episodes = AsyncMock(return_value=[
        MemoryEpisode(id="sail", agent_id="a", title="Sailing",
                      summary="Lake trip", emotional_arc="calm"),
    ])
    consolidator.get_recent_episodes_for_context = AsyncMock(return_value=[])
    feat = _feature_with_consolidator(monkeypatch, consolidator)

    result = await feat.get_episodes(limit=5, query="boats on the lake")

    consolidator.search_episodes.assert_awaited_once_with(
        "boats on the lake", limit=5)
    consolidator.get_recent_episodes_for_context.assert_not_awaited()
    assert result.data["mode"] == "relevance"
    assert result.data["count"] == 1
    assert result.data["episodes"][0]["title"] == "Sailing"


@pytest.mark.asyncio
async def test_get_episodes_no_query_uses_recency(monkeypatch):
    consolidator = MagicMock()
    consolidator.search_episodes = AsyncMock(return_value=[])
    consolidator.get_recent_episodes_for_context = AsyncMock(return_value=[
        {"title": "Recent", "summary": "s", "emotional_arc": "", "timespan": "2026-06-01"},
    ])
    feat = _feature_with_consolidator(monkeypatch, consolidator)

    result = await feat.get_episodes(limit=3)

    consolidator.get_recent_episodes_for_context.assert_awaited_once_with(
        max_episodes=3)
    consolidator.search_episodes.assert_not_awaited()
    assert result.data["mode"] == "recency"


@pytest.mark.asyncio
async def test_get_episodes_blank_query_is_recency(monkeypatch):
    """A whitespace-only query is treated as no query (recency path)."""
    consolidator = MagicMock()
    consolidator.search_episodes = AsyncMock(return_value=[])
    consolidator.get_recent_episodes_for_context = AsyncMock(return_value=[])
    feat = _feature_with_consolidator(monkeypatch, consolidator)

    result = await feat.get_episodes(limit=3, query="   ")

    consolidator.search_episodes.assert_not_awaited()
    consolidator.get_recent_episodes_for_context.assert_awaited_once()
    assert result.data["mode"] == "recency"
