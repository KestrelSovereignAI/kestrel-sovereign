"""
Tests for the Saved Items system.

Tests SavedItemsStore and SaveFeature functionality.
"""
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


class TestSavedItemsStore:
    """Tests for SavedItemsStore."""

    @pytest.fixture
    def mock_db(self):
        """Create a mock database."""
        db = AsyncMock()
        db.execute = AsyncMock()
        db.commit = AsyncMock()
        db.fetchone = AsyncMock(return_value=None)
        db.fetchall = AsyncMock(return_value=[])
        return db

    @pytest.mark.asyncio
    async def test_save_item_basic(self, mock_db):
        """Test saving a basic item."""
        from kestrel_sovereign.storage.saved_items_store import SavedItemsStore

        store = SavedItemsStore(mock_db, agent_id="test-agent")

        # Mock no existing item
        mock_db.fetchone.return_value = None

        item = await store.save_item(
            item_type="structured",
            name="Test Item",
            content="This is test content",
            summary="A test item",
            compute_embedding=False  # Skip embedding for test
        )

        assert item.name == "Test Item"
        assert item.item_type == "structured"
        assert item.content == "This is test content"
        assert item.summary == "A test item"
        assert item.agent_id == "test-agent"
        assert item.content_hash is not None
        mock_db.execute.assert_called_once()
        mock_db.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_save_item_with_tags(self, mock_db):
        """Test saving item with tags."""
        from kestrel_sovereign.storage.saved_items_store import SavedItemsStore

        store = SavedItemsStore(mock_db, agent_id="test-agent")
        mock_db.fetchone.return_value = None

        item = await store.save_item(
            item_type="excerpt",
            name="Tagged Item",
            content="Content with tags",
            tags=["important", "architecture"],
            compute_embedding=False
        )

        assert item.tags == ["important", "architecture"]

    @pytest.mark.asyncio
    async def test_deduplication_same_identity_updates_existing_item(self, mock_db):
        """Test that duplicate content updates an item only for the same logical identity."""
        from kestrel_sovereign.storage.saved_items_store import SavedItemsStore, SavedItem

        store = SavedItemsStore(mock_db, agent_id="test-agent")

        # Mock existing item with same content hash
        existing_row = (
            "existing-id",
            "test-agent",
            "structured",
            "Old Name",
            "Old summary",
            "Same content",
            "abc123hash",  # content_hash
            None,  # ipfs_cid
            None,  # embedding
            "manual",
            None,
            None,
            "[]",
            "{}",
            "2024-01-15T10:00:00",
            "2024-01-15T10:00:00"
        )
        mock_db.fetchall.return_value = [existing_row]
        store.update_item = AsyncMock(return_value=SavedItem.from_row(existing_row))

        item = await store.save_item(
            item_type="structured",
            name="New Name",
            content="Same content",
            source_type="manual",
            deduplicate=True,
            compute_embedding=False
        )

        # Should return updated existing item, not create new
        assert item.id == "existing-id"

    @pytest.mark.asyncio
    async def test_deduplication_does_not_merge_different_item_types(self, mock_db):
        """Same content across item types should create distinct rows."""
        from kestrel_sovereign.storage.saved_items_store import SavedItemsStore

        store = SavedItemsStore(mock_db, agent_id="test-agent")
        mock_db.fetchall.return_value = [
            (
                "existing-id",
                "test-agent",
                "stash",
                "Existing Stash",
                "Old summary",
                "Same content",
                "abc123hash",
                None,
                None,
                "manual",
                None,
                None,
                "[]",
                "{}",
                "2024-01-15T10:00:00",
                "2024-01-15T10:00:00",
            )
        ]

        item = await store.save_item(
            item_type="structured",
            name="Structured Copy",
            content="Same content",
            deduplicate=True,
            compute_embedding=False,
        )

        assert item.id != "existing-id"
        mock_db.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_list_items(self, mock_db):
        """Test listing items."""
        from kestrel_sovereign.storage.saved_items_store import SavedItemsStore

        store = SavedItemsStore(mock_db, agent_id="test-agent")

        # Mock returned rows
        mock_db.fetchall.return_value = [
            (
                "item-1", "test-agent", "stash", "Stash 1", "Summary 1",
                "content1", "hash1", None, None, "conversation", None, None,
                "[]", "{}", "2024-01-15T10:00:00", "2024-01-15T10:00:00"
            ),
            (
                "item-2", "test-agent", "excerpt", "Excerpt 1", "Summary 2",
                "content2", "hash2", None, None, "conversation", None, None,
                '["tag1"]', "{}", "2024-01-15T11:00:00", "2024-01-15T11:00:00"
            )
        ]

        items = await store.list_items()

        assert len(items) == 2
        assert items[0].id == "item-1"
        assert items[0].item_type == "stash"
        assert items[1].id == "item-2"
        assert items[1].tags == ["tag1"]

    @pytest.mark.asyncio
    async def test_list_items_by_type(self, mock_db):
        """Test listing items filtered by type."""
        from kestrel_sovereign.storage.saved_items_store import SavedItemsStore

        store = SavedItemsStore(mock_db, agent_id="test-agent")
        mock_db.fetchall.return_value = []

        await store.list_items(item_type="stash")

        # Verify the query included type filter
        call_args = mock_db.fetchall.call_args
        assert "item_type = ?" in call_args[0][0]
        assert "stash" in call_args[0][1]

    @pytest.mark.asyncio
    async def test_get_by_id(self, mock_db):
        """Test getting item by ID."""
        from kestrel_sovereign.storage.saved_items_store import SavedItemsStore

        store = SavedItemsStore(mock_db, agent_id="test-agent")

        mock_db.fetchone.return_value = (
            "item-123", "test-agent", "structured", "My Item", "Summary",
            '{"data": "test"}', "hash123", None, None, "manual", None, "recipe",
            '["food"]', '{"servings": 4}', "2024-01-15T10:00:00", "2024-01-15T10:00:00"
        )

        item = await store.get_by_id("item-123")

        assert item is not None
        assert item.id == "item-123"
        assert item.name == "My Item"
        assert item.schema_id == "recipe"
        assert item.tags == ["food"]
        assert item.metadata == {"servings": 4}

    @pytest.mark.asyncio
    async def test_get_by_id_not_found(self, mock_db):
        """Test getting non-existent item."""
        from kestrel_sovereign.storage.saved_items_store import SavedItemsStore

        store = SavedItemsStore(mock_db, agent_id="test-agent")
        mock_db.fetchone.return_value = None

        item = await store.get_by_id("nonexistent")

        assert item is None

    @pytest.mark.asyncio
    async def test_delete_item(self, mock_db):
        """Test deleting item."""
        from kestrel_sovereign.storage.saved_items_store import SavedItemsStore

        store = SavedItemsStore(mock_db, agent_id="test-agent")

        result = await store.delete_item("item-to-delete")

        assert result is True
        mock_db.execute.assert_called_once()
        call_args = mock_db.execute.call_args[0]
        assert "DELETE FROM saved_items" in call_args[0]
        assert "item-to-delete" in call_args[1]

    @pytest.mark.asyncio
    async def test_text_search_fallback(self, mock_db):
        """Test text search when embeddings unavailable."""
        from kestrel_sovereign.storage.saved_items_store import SavedItemsStore

        store = SavedItemsStore(mock_db, agent_id="test-agent")
        store._embedding_service = False  # Mark as unavailable

        mock_db.fetchall.return_value = [
            (
                "item-1", "test-agent", "stash", "Debug Session", "Debugging discussion",
                "content about debugging", "hash1", None, None, "conversation", None, None,
                "[]", "{}", "2024-01-15T10:00:00", "2024-01-15T10:00:00"
            )
        ]

        results = await store.search("debugging")

        assert len(results) == 1
        assert results[0]["item"]["name"] == "Debug Session"
        assert results[0]["score"] == 1.0  # Text search returns 1.0

    @pytest.mark.asyncio
    async def test_get_item_count(self, mock_db):
        """Test getting item count."""
        from kestrel_sovereign.storage.saved_items_store import SavedItemsStore

        store = SavedItemsStore(mock_db, agent_id="test-agent")
        mock_db.fetchone.return_value = (5,)

        count = await store.get_item_count()

        assert count == 5

    @pytest.mark.asyncio
    async def test_get_item_count_by_type(self, mock_db):
        """Test getting item count filtered by type."""
        from kestrel_sovereign.storage.saved_items_store import SavedItemsStore

        store = SavedItemsStore(mock_db, agent_id="test-agent")
        mock_db.fetchone.return_value = (3,)

        count = await store.get_item_count(item_type="stash")

        assert count == 3
        call_args = mock_db.fetchone.call_args[0]
        assert "item_type = ?" in call_args[0]


class TestSavedItemModel:
    """Tests for SavedItem dataclass."""

    def test_to_dict(self):
        """Test converting SavedItem to dict."""
        from kestrel_sovereign.storage.saved_items_store import SavedItem
        from datetime import datetime, timezone

        now = datetime.now(timezone.utc)
        item = SavedItem(
            id="test-id",
            agent_id="agent-1",
            item_type="stash",
            name="Test Stash",
            content='{"messages": []}',
            summary="A test stash",
            tags=["test"],
            metadata={"key": "value"},
            created_at=now,
            updated_at=now
        )

        d = item.to_dict()

        assert d["id"] == "test-id"
        assert d["name"] == "Test Stash"
        assert d["tags"] == ["test"]
        assert d["metadata"] == {"key": "value"}
        assert d["created_at"] == now.isoformat()

    def test_from_row(self):
        """Test creating SavedItem from database row."""
        from kestrel_sovereign.storage.saved_items_store import SavedItem

        row = (
            "row-id",
            "agent-1",
            "excerpt",
            "Excerpt Name",
            "Summary text",
            "Content here",
            "hash123",
            "ipfs://cid",
            None,  # embedding
            "conversation",
            '{"ids": [1,2,3]}',
            None,  # schema_id
            '["tag1", "tag2"]',
            '{"meta": "data"}',
            "2024-01-15T10:00:00",
            "2024-01-15T11:00:00"
        )

        item = SavedItem.from_row(row)

        assert item.id == "row-id"
        assert item.item_type == "excerpt"
        assert item.name == "Excerpt Name"
        assert item.ipfs_cid == "ipfs://cid"
        assert item.tags == ["tag1", "tag2"]
        assert item.metadata == {"meta": "data"}


class TestSavedItemTypes:
    """Tests for SavedItemType and SourceType enums."""

    def test_saved_item_types(self):
        """Test SavedItemType enum values."""
        from kestrel_sovereign.storage.saved_items_store import SavedItemType

        assert SavedItemType.STASH.value == "stash"
        assert SavedItemType.FILE.value == "file"
        assert SavedItemType.EXCERPT.value == "excerpt"
        assert SavedItemType.STRUCTURED.value == "structured"

    def test_source_types(self):
        """Test SourceType enum values."""
        from kestrel_sovereign.storage.saved_items_store import SourceType

        assert SourceType.CONVERSATION.value == "conversation"
        assert SourceType.FILE.value == "file"
        assert SourceType.URL.value == "url"
        assert SourceType.MANUAL.value == "manual"


class TestContentHash:
    """Tests for content hashing."""

    def test_compute_content_hash(self):
        """Test content hash computation."""
        from kestrel_sovereign.storage.saved_items_store import _compute_content_hash

        hash1 = _compute_content_hash("test content")
        hash2 = _compute_content_hash("test content")
        hash3 = _compute_content_hash("different content")

        assert hash1 == hash2  # Same content = same hash
        assert hash1 != hash3  # Different content = different hash
        assert len(hash1) == 64  # SHA256 hex is 64 chars


class TestEmbeddingSerialization:
    """Tests for embedding serialization."""

    def test_serialize_deserialize_embedding(self):
        """Test embedding roundtrip."""
        from kestrel_sovereign.storage.saved_items_store import (
            _serialize_embedding, _deserialize_embedding
        )

        original = [0.1, 0.2, 0.3, 0.4, 0.5]
        serialized = _serialize_embedding(original)
        deserialized = _deserialize_embedding(serialized)

        assert len(deserialized) == len(original)
        for i in range(len(original)):
            assert abs(deserialized[i] - original[i]) < 0.0001

    def test_serialize_large_embedding(self):
        """Test serializing larger embedding vector."""
        from kestrel_sovereign.storage.saved_items_store import (
            _serialize_embedding, _deserialize_embedding
        )

        # Typical embedding size (768 dimensions)
        original = [float(i) / 768 for i in range(768)]
        serialized = _serialize_embedding(original)
        deserialized = _deserialize_embedding(serialized)

        assert len(deserialized) == 768
        assert abs(deserialized[100] - original[100]) < 0.0001


class TestStashSaveIntegration:
    """Tests for stash_save method in ContextManager."""

    @pytest.fixture
    def mock_storage_with_conv(self):
        """Create mock storage with conversation store."""
        storage = MagicMock()
        conv_store = AsyncMock()
        storage.conversation = conv_store

        # Mock db for SavedItemsStore
        db = AsyncMock()
        db.execute = AsyncMock()
        db.commit = AsyncMock()
        db.fetchone = AsyncMock(return_value=None)
        storage.db = db

        return storage, conv_store

    @pytest.mark.asyncio
    async def test_stash_save_basic(self, mock_storage_with_conv):
        """Test saving a stash to long-term storage."""
        from kestrel_sovereign.agent.context_manager import ContextManager

        storage, conv_store = mock_storage_with_conv

        conv_store.list_stashes = AsyncMock(return_value=[
            {"stash_id": "abc123", "name": "test-stash", "message_count": 3}
        ])
        conv_store.get_stashed_messages = AsyncMock(return_value=[
            {"id": 1, "role": "user", "content": "Hello"},
            {"id": 2, "role": "assistant", "content": "Hi there!"},
            {"id": 3, "role": "user", "content": "How are you?"},
        ])
        conv_store.add_conversation = AsyncMock()

        manager = ContextManager(storage=storage, model="gpt-4", agent_id="test")

        # Mock the SavedItemsStore.save_item (patch at source module)
        with patch('kestrel_sovereign.storage.saved_items_store.SavedItemsStore') as MockStore:
            mock_store_instance = AsyncMock()
            mock_item = MagicMock()
            mock_item.id = "saved-item-123"
            mock_item.name = "test-stash"
            mock_item.embedding = [0.1, 0.2, 0.3]
            mock_store_instance.save_item = AsyncMock(return_value=mock_item)
            MockStore.return_value = mock_store_instance

            result = await manager.stash_save()

            assert result["success"] is True
            assert result["saved_item_id"] == "saved-item-123"
            assert result["message_count"] == 3
            assert result["has_embedding"] is True

    @pytest.mark.asyncio
    async def test_stash_save_no_stashes(self, mock_storage_with_conv):
        """Test stash_save when no stashes exist."""
        from kestrel_sovereign.agent.context_manager import ContextManager

        storage, conv_store = mock_storage_with_conv
        conv_store.list_stashes = AsyncMock(return_value=[])

        manager = ContextManager(storage=storage, model="gpt-4", agent_id="test")
        result = await manager.stash_save()

        assert result["success"] is False
        assert "No stashes found" in result["error"]

    @pytest.mark.asyncio
    async def test_stash_save_specific_stash(self, mock_storage_with_conv):
        """Test saving a specific stash by ID."""
        from kestrel_sovereign.agent.context_manager import ContextManager

        storage, conv_store = mock_storage_with_conv

        conv_store.get_stashed_messages = AsyncMock(return_value=[
            {"id": 5, "role": "user", "content": "Specific message"},
        ])
        conv_store.add_conversation = AsyncMock()

        manager = ContextManager(storage=storage, model="gpt-4", agent_id="test")

        with patch('kestrel_sovereign.storage.saved_items_store.SavedItemsStore') as MockStore:
            mock_store_instance = AsyncMock()
            mock_item = MagicMock()
            mock_item.id = "saved-xyz"
            mock_item.name = "specific-stash"
            mock_item.embedding = None
            mock_store_instance.save_item = AsyncMock(return_value=mock_item)
            MockStore.return_value = mock_store_instance

            result = await manager.stash_save(stash_id="xyz789", name="My Saved Stash")

            assert result["success"] is True
            assert result["stash_id"] == "xyz789"

            # Verify save_item was called with correct name
            call_kwargs = mock_store_instance.save_item.call_args.kwargs
            assert call_kwargs["name"] == "My Saved Stash"
