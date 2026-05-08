"""ToolResult contract tests for SaveFeature (#1061 wave 7).

Pins the honesty edges introduced by the migration:
  - save_stash / save_excerpt / save_item without an embedding -> PARTIAL
    (the LLM must speak that semantic recall will not find the item)
  - save_excerpt last_N when fewer messages exist -> PARTIAL with shortfall
  - recall with zero results -> OK with explicit no-match wording
  - recall_get / recall_delete on missing id -> ERROR
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from kestrel_sdk.tools.result import ToolResultStatus
from kestrel_sovereign.features.save.feature import SaveFeature


def _make_feature(store, conv_store=None):
    feat = SaveFeature(agent=MagicMock())
    feat.storage = MagicMock()
    feat._db = MagicMock()
    feat.context_manager = MagicMock()
    feat.context_manager._get_conversation_store = MagicMock(return_value=conv_store)
    feat.agent_id = "did:test:agent"
    feat._saved_items_store = store
    return feat


def _fake_item(item_id="itm-1", name="thing", item_type="structured", embedding=None):
    return SimpleNamespace(
        id=item_id,
        name=name,
        item_type=item_type,
        embedding=embedding,
    )


# ---------------------------------------------------------------------------
# save_stash
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_save_stash_with_embedding_returns_ok():
    store = AsyncMock()
    store.save_item = AsyncMock(
        return_value=_fake_item(item_id="saved-1", name="my-stash", embedding=[0.1, 0.2])
    )
    conv = AsyncMock()
    conv.list_stashes = AsyncMock(return_value=[
        {"stash_id": "s1", "name": "my-stash", "message_count": 2}
    ])
    conv.get_stashed_messages = AsyncMock(return_value=[
        {"id": 1, "role": "user", "content": "hello"},
        {"id": 2, "role": "assistant", "content": "hi"},
    ])
    feat = _make_feature(store, conv_store=conv)

    result = await feat.save_stash()

    assert result.status is ToolResultStatus.OK
    assert result.data["saved_item_id"] == "saved-1"
    assert result.data["message_count"] == 2
    assert result.data["has_embedding"] is True


@pytest.mark.asyncio
async def test_save_stash_without_embedding_returns_partial():
    store = AsyncMock()
    store.save_item = AsyncMock(
        return_value=_fake_item(item_id="saved-2", name="my-stash", embedding=None)
    )
    conv = AsyncMock()
    conv.list_stashes = AsyncMock(return_value=[
        {"stash_id": "s1", "name": "my-stash"}
    ])
    conv.get_stashed_messages = AsyncMock(return_value=[
        {"id": 1, "role": "user", "content": "x"}
    ])
    feat = _make_feature(store, conv_store=conv)

    result = await feat.save_stash()

    assert result.status is ToolResultStatus.PARTIAL
    assert result.data["has_embedding"] is False
    assert "embedding" in result.error


@pytest.mark.asyncio
async def test_save_stash_no_stashes_returns_error():
    store = AsyncMock()
    conv = AsyncMock()
    conv.list_stashes = AsyncMock(return_value=[])
    feat = _make_feature(store, conv_store=conv)

    result = await feat.save_stash()

    assert result.status is ToolResultStatus.ERROR
    assert "No stashes found" in result.error


@pytest.mark.asyncio
async def test_save_stash_empty_specific_stash_returns_error():
    store = AsyncMock()
    conv = AsyncMock()
    conv.get_stashed_messages = AsyncMock(return_value=[])
    feat = _make_feature(store, conv_store=conv)

    result = await feat.save_stash(stash_id="missing")

    assert result.status is ToolResultStatus.ERROR
    assert "missing" in result.error
    assert result.data["stash_id"] == "missing"


# ---------------------------------------------------------------------------
# save_excerpt
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_save_excerpt_last_n_full_returns_ok():
    store = AsyncMock()
    store.save_item = AsyncMock(
        return_value=_fake_item(item_id="exc-1", name="recent", embedding=[0.1])
    )
    conv = AsyncMock()
    conv.get_full_history_with_ids = AsyncMock(return_value=[
        {"id": i, "role": "user", "content": f"m{i}"} for i in range(5)
    ])
    feat = _make_feature(store, conv_store=conv)

    result = await feat.save_excerpt(target="last_3", name="recent")

    assert result.status is ToolResultStatus.OK
    assert result.data["message_count"] == 3
    assert "shortfall" not in result.data


@pytest.mark.asyncio
async def test_save_excerpt_last_n_shortfall_returns_partial():
    store = AsyncMock()
    store.save_item = AsyncMock(
        return_value=_fake_item(item_id="exc-2", name="thin", embedding=[0.1])
    )
    conv = AsyncMock()
    conv.get_full_history_with_ids = AsyncMock(return_value=[
        {"id": 1, "role": "user", "content": "only one"}
    ])
    feat = _make_feature(store, conv_store=conv)

    result = await feat.save_excerpt(target="last_5", name="thin")

    assert result.status is ToolResultStatus.PARTIAL
    assert result.data["message_count"] == 1
    assert result.data["requested_count"] == 5
    assert result.data["shortfall"] == 4
    assert "5" in result.error and "1" in result.error


@pytest.mark.asyncio
async def test_save_excerpt_invalid_target_returns_error():
    store = AsyncMock()
    conv = AsyncMock()
    feat = _make_feature(store, conv_store=conv)

    result = await feat.save_excerpt(target="garbage", name="x")

    assert result.status is ToolResultStatus.ERROR
    assert "garbage" in result.error
    assert result.data["target"] == "garbage"


@pytest.mark.asyncio
async def test_save_excerpt_no_embedding_returns_partial():
    store = AsyncMock()
    store.save_item = AsyncMock(
        return_value=_fake_item(item_id="exc-3", name="ne", embedding=None)
    )
    conv = AsyncMock()
    conv.get_messages_by_ids = AsyncMock(return_value=[
        {"id": 1, "role": "user", "content": "x"},
        {"id": 2, "role": "user", "content": "y"},
    ])
    feat = _make_feature(store, conv_store=conv)

    result = await feat.save_excerpt(target="ids:1,2", name="ne")

    assert result.status is ToolResultStatus.PARTIAL
    assert result.data["has_embedding"] is False
    assert "embedding" in result.error


# ---------------------------------------------------------------------------
# save_item
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_save_item_with_embedding_returns_ok():
    store = AsyncMock()
    store.save_item = AsyncMock(
        return_value=_fake_item(item_id="i-1", name="note", embedding=[0.5])
    )
    feat = _make_feature(store)

    result = await feat.save_item(name="note", content="some text")

    assert result.status is ToolResultStatus.OK
    assert result.data["saved_item_id"] == "i-1"
    assert result.data["item_type"] == "structured"
    assert result.data["has_embedding"] is True


@pytest.mark.asyncio
async def test_save_item_without_embedding_returns_partial():
    store = AsyncMock()
    store.save_item = AsyncMock(
        return_value=_fake_item(item_id="i-2", name="note", embedding=None)
    )
    feat = _make_feature(store)

    result = await feat.save_item(name="note", content="text")

    assert result.status is ToolResultStatus.PARTIAL
    assert result.data["has_embedding"] is False
    assert "embedding" in result.error


@pytest.mark.asyncio
async def test_save_item_propagates_store_failure():
    store = AsyncMock()
    store.save_item = AsyncMock(side_effect=RuntimeError("disk full"))
    feat = _make_feature(store)

    result = await feat.save_item(name="x", content="y")

    assert result.status is ToolResultStatus.ERROR
    assert "disk full" in result.error


# ---------------------------------------------------------------------------
# recall / recall_list
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_recall_no_matches_returns_ok_with_zero_count():
    store = AsyncMock()
    store.search = AsyncMock(return_value=[])
    feat = _make_feature(store)

    result = await feat.recall(query="anything")

    assert result.status is ToolResultStatus.OK
    assert result.data["result_count"] == 0
    assert result.data["results"] == []
    assert "no matches" in result.confirmation.lower()


@pytest.mark.asyncio
async def test_recall_with_matches_returns_ok_with_results():
    store = AsyncMock()
    store.search = AsyncMock(return_value=[
        {
            "score": 0.876,
            "item": {
                "id": "i-1",
                "name": "thing",
                "item_type": "structured",
                "summary": "hello world",
                "tags": ["x"],
                "created_at": "2026-05-08T00:00:00+00:00",
            },
        }
    ])
    feat = _make_feature(store)

    result = await feat.recall(query="thing")

    assert result.status is ToolResultStatus.OK
    assert result.data["result_count"] == 1
    assert result.data["results"][0]["score"] == 0.876


@pytest.mark.asyncio
async def test_recall_list_returns_ok():
    store = AsyncMock()
    store.list_items = AsyncMock(return_value=[
        SimpleNamespace(
            id="i-1", name="a", item_type="structured", summary="s",
            tags=[], created_at=None,
        ),
        SimpleNamespace(
            id="i-2", name="b", item_type="structured", summary="t",
            tags=[], created_at=None,
        ),
    ])
    feat = _make_feature(store)

    result = await feat.recall_list()

    assert result.status is ToolResultStatus.OK
    assert result.data["count"] == 2
    assert {it["id"] for it in result.data["items"]} == {"i-1", "i-2"}


# ---------------------------------------------------------------------------
# recall_get / recall_delete
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_recall_get_missing_returns_error():
    store = AsyncMock()
    store.get_by_id = AsyncMock(return_value=None)
    feat = _make_feature(store)

    result = await feat.recall_get(item_id="nope")

    assert result.status is ToolResultStatus.ERROR
    assert "nope" in result.error


@pytest.mark.asyncio
async def test_recall_get_returns_ok_with_dict():
    item = SimpleNamespace(to_dict=lambda: {"id": "i-1", "name": "thing"})
    store = AsyncMock()
    store.get_by_id = AsyncMock(return_value=item)
    feat = _make_feature(store)

    result = await feat.recall_get(item_id="i-1")

    assert result.status is ToolResultStatus.OK
    assert result.data["item"] == {"id": "i-1", "name": "thing"}


@pytest.mark.asyncio
async def test_recall_delete_missing_returns_error():
    store = AsyncMock()
    store.get_by_id = AsyncMock(return_value=None)
    feat = _make_feature(store)

    result = await feat.recall_delete(item_id="nope")

    assert result.status is ToolResultStatus.ERROR
    assert "nope" in result.error


@pytest.mark.asyncio
async def test_recall_delete_succeeds():
    item = SimpleNamespace(name="thing")
    store = AsyncMock()
    store.get_by_id = AsyncMock(return_value=item)
    store.delete_item = AsyncMock(return_value=None)
    feat = _make_feature(store)

    result = await feat.recall_delete(item_id="i-1")

    assert result.status is ToolResultStatus.OK
    assert result.data["deleted_id"] == "i-1"
    assert result.data["deleted_name"] == "thing"


# ---------------------------------------------------------------------------
# Storage-not-available paths
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_recall_without_store_returns_error():
    feat = _make_feature(store=None)
    feat._db = None
    result = await feat.recall(query="x")
    assert result.status is ToolResultStatus.ERROR
    assert "Storage not available" in result.error


@pytest.mark.asyncio
async def test_save_item_without_store_returns_error():
    feat = _make_feature(store=None)
    feat._db = None
    result = await feat.save_item(name="n", content="c")
    assert result.status is ToolResultStatus.ERROR
    assert "Storage not available" in result.error


# ---------------------------------------------------------------------------
# Contract: every @tool annotated -> ToolResult
# ---------------------------------------------------------------------------

def test_save_feature_passes_toolresult_contract():
    from kestrel_sovereign.tools.result_contract import (
        assert_feature_returns_tool_result,
    )

    feat = SaveFeature(agent=MagicMock())
    assert_feature_returns_tool_result(feat)
