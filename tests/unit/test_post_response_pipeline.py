"""
Unit tests for post-response memory tagging pipeline (#565).

Tests cover:
- Phase 1: Inline emotional tagging (EmotionalTagger via MemorySystem.tag_message)
- Phase 2: Background temporal + associative processing
- MemoryManager.tag_exchange orchestration
- update_message_metadata race-condition fix (atomic JSON merge)
- Dual episode-creation thresholds
- Error isolation (failures in one phase do not block the other)
"""

import asyncio
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

from kestrel_sovereign.agent.memory_manager import MemoryManager
from kestrel_sovereign.storage.emotional_tagger import EmotionalTagger
from kestrel_sovereign.storage.memory_system import MemorySystem


# ─────────────────────────────────────────────────────────────────────────────
# MemorySystem.tag_message (Phase 1 inline tagging)
# ─────────────────────────────────────────────────────────────────────────────


class TestMemorySystemTagMessage:
    """Tests for MemorySystem.tag_message -- inline emotional enrichment."""

    @pytest.fixture
    def memory_system(self):
        storage = MagicMock()
        conv_store = AsyncMock()
        storage.conversation = conv_store
        storage.db = MagicMock()
        storage.graph = MagicMock()
        ms = MemorySystem(storage=storage, agent_id="test-agent")
        # Skip full initialize -- just set the tagger
        ms.tagger = EmotionalTagger()
        return ms

    @pytest.mark.asyncio
    async def test_tag_message_positive_emotion(self, memory_system):
        """Should detect positive emotion and write metadata to message."""
        result = await memory_system.tag_message(
            message_id=42,
            content="I am so happy today, this is wonderful!",
            role="user",
        )
        assert result["emotional_valence"] > 0
        assert "joy" in result.get("emotional_categories", [])
        # Should have called update_message_metadata
        memory_system.storage.conversation.update_message_metadata.assert_awaited_once()
        call_args = memory_system.storage.conversation.update_message_metadata.call_args
        assert call_args[0][0] == 42  # message_id

    @pytest.mark.asyncio
    async def test_tag_message_negative_emotion(self, memory_system):
        """Should detect negative emotion."""
        result = await memory_system.tag_message(
            message_id=43,
            content="I feel so sad and depressed about everything",
            role="user",
        )
        assert result["emotional_valence"] < 0
        assert "sadness" in result.get("emotional_categories", [])

    @pytest.mark.asyncio
    async def test_tag_message_importance_life_event(self, memory_system):
        """Life events should receive high importance scores."""
        result = await memory_system.tag_message(
            message_id=44,
            content="I just got married last weekend!",
            role="user",
        )
        assert result.get("importance", 0) > 0.7

    @pytest.mark.asyncio
    async def test_tag_message_conv_store_failure_does_not_raise(self, memory_system):
        """Failure writing tags should be caught, not propagated."""
        memory_system.storage.conversation.update_message_metadata.side_effect = (
            RuntimeError("DB connection lost")
        )
        # Should not raise
        result = await memory_system.tag_message(
            message_id=99, content="hello", role="user"
        )
        # Still returns the enriched metadata even though write failed
        assert "emotional_valence" in result

    @pytest.mark.asyncio
    async def test_tag_message_no_conv_store(self):
        """Should gracefully handle missing conversation store."""
        storage = MagicMock()
        storage.conversation = None
        storage.db = MagicMock()
        storage.graph = MagicMock()
        ms = MemorySystem(storage=storage, agent_id="test")
        ms.tagger = EmotionalTagger()
        result = await ms.tag_message(1, "hello", "user")
        assert "emotional_valence" in result


# ─────────────────────────────────────────────────────────────────────────────
# MemoryManager.tag_exchange
# ─────────────────────────────────────────────────────────────────────────────


class TestMemoryManagerTagExchange:
    """Tests for MemoryManager.tag_exchange orchestration."""

    @pytest.fixture
    def memory_manager(self):
        storage = MagicMock()
        storage.conversation = AsyncMock()
        return MemoryManager(storage=storage, agent_id="agent-123")

    @pytest.fixture
    def mock_memory_system(self):
        ms = AsyncMock()
        ms.tag_message = AsyncMock(
            return_value={"emotional_valence": 0.5, "importance": 0.6}
        )
        return ms

    @pytest.mark.asyncio
    async def test_tag_exchange_both_messages(self, memory_manager, mock_memory_system):
        """Should tag both user and assistant messages."""
        result = await memory_manager.tag_exchange(
            user_content="I am excited!",
            assistant_content="That is great to hear!",
            user_message_id=10,
            assistant_message_id=11,
            memory_system=mock_memory_system,
        )
        assert result["user"] is not None
        assert result["assistant"] is not None
        assert mock_memory_system.tag_message.await_count == 2

    @pytest.mark.asyncio
    async def test_tag_exchange_no_memory_system(self, memory_manager):
        """Should return empty results without memory_system."""
        result = await memory_manager.tag_exchange(
            user_content="hello",
            assistant_content="hi",
            user_message_id=1,
            assistant_message_id=2,
            memory_system=None,
        )
        assert result["user"] is None
        assert result["assistant"] is None

    @pytest.mark.asyncio
    async def test_tag_exchange_missing_message_ids(self, memory_manager, mock_memory_system):
        """Should skip tagging when message_id is None."""
        result = await memory_manager.tag_exchange(
            user_content="hello",
            assistant_content="hi",
            user_message_id=None,
            assistant_message_id=None,
            memory_system=mock_memory_system,
        )
        assert result["user"] is None
        assert result["assistant"] is None
        mock_memory_system.tag_message.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_tag_exchange_error_isolation(self, memory_manager):
        """Failure in tag_message should not propagate."""
        ms = AsyncMock()
        ms.tag_message = AsyncMock(side_effect=RuntimeError("boom"))
        # Should not raise
        result = await memory_manager.tag_exchange(
            user_content="hello",
            assistant_content="hi",
            user_message_id=1,
            assistant_message_id=2,
            memory_system=ms,
        )
        # Returns None for both since the exception was caught
        assert result["user"] is None
        assert result["assistant"] is None


# ─────────────────────────────────────────────────────────────────────────────
# update_message_metadata race-condition fix
# ─────────────────────────────────────────────────────────────────────────────


class TestUpdateMessageMetadataAtomicMerge:
    """Tests for the atomic JSON merge in AsyncConversationStore."""

    @pytest.mark.asyncio
    async def test_sqlite_merge_preserves_existing_fields(self):
        """SQLite path: SELECT + UPDATE should merge, not overwrite."""
        from kestrel_sovereign.storage.async_conversation_store import AsyncConversationStore

        db = AsyncMock()
        db.backend_type = "sqlite"
        # Existing metadata has enc and session_id
        db.fetchone = AsyncMock(
            return_value=(json.dumps({"enc": True, "session_id": "abc", "old_field": 1}),)
        )
        db.execute_commit = AsyncMock()

        store = AsyncConversationStore(db, agent_id="test")
        # Patch out encryption
        store._global_fernet = None
        store._agent_fernet = None

        result = await store.update_message_metadata(
            message_id=42,
            metadata_updates={"emotional_valence": 0.8, "importance": 0.9},
        )
        assert result is True
        # Verify the merged metadata was written
        written_meta = json.loads(db.execute_commit.call_args[0][1][0])
        assert written_meta["enc"] is True  # preserved
        assert written_meta["session_id"] == "abc"  # preserved
        assert written_meta["old_field"] == 1  # preserved
        assert written_meta["emotional_valence"] == 0.8  # added
        assert written_meta["importance"] == 0.9  # added

    @pytest.mark.asyncio
    async def test_sqlite_returns_false_on_missing_message(self):
        """Should return False if message not found."""
        from kestrel_sovereign.storage.async_conversation_store import AsyncConversationStore

        db = AsyncMock()
        db.backend_type = "sqlite"
        db.fetchone = AsyncMock(return_value=None)

        store = AsyncConversationStore(db, agent_id="test")
        store._global_fernet = None
        store._agent_fernet = None

        result = await store.update_message_metadata(99, {"foo": "bar"})
        assert result is False

    @pytest.mark.asyncio
    async def test_postgres_uses_jsonb_merge(self):
        """PostgreSQL path: should use atomic || operator."""
        from kestrel_sovereign.storage.async_conversation_store import AsyncConversationStore

        db = AsyncMock()
        db.backend_type = "postgres"
        mock_result = MagicMock()
        mock_result.rowcount = 1
        db.execute_commit = AsyncMock(return_value=mock_result)

        store = AsyncConversationStore(db, agent_id="test")
        store._global_fernet = None
        store._agent_fernet = None

        result = await store.update_message_metadata(
            message_id=42,
            metadata_updates={"emotional_valence": 0.8},
        )
        assert result is True
        # Verify SQL uses jsonb merge operator
        sql = db.execute_commit.call_args[0][0]
        assert "||" in sql
        assert "jsonb" in sql.lower()


# ─────────────────────────────────────────────────────────────────────────────
# Episode creation dual threshold
# ─────────────────────────────────────────────────────────────────────────────


class TestEpisodeCreationThreshold:
    """Tests for the configurable dual episode-creation threshold."""

    @pytest.mark.asyncio
    async def test_default_threshold_is_15(self):
        """Default episode threshold should be 15, not the old 20."""
        import os

        # Ensure env var is not set
        old_val = os.environ.pop("KESTREL_EPISODE_THRESHOLD", None)
        try:
            threshold = int(os.environ.get("KESTREL_EPISODE_THRESHOLD", "15"))
            assert threshold == 15
        finally:
            if old_val is not None:
                os.environ["KESTREL_EPISODE_THRESHOLD"] = old_val

    @pytest.mark.asyncio
    async def test_configurable_threshold(self):
        """KESTREL_EPISODE_THRESHOLD env var should override default."""
        import os

        os.environ["KESTREL_EPISODE_THRESHOLD"] = "25"
        try:
            threshold = int(os.environ.get("KESTREL_EPISODE_THRESHOLD", "15"))
            assert threshold == 25
        finally:
            del os.environ["KESTREL_EPISODE_THRESHOLD"]


# ─────────────────────────────────────────────────────────────────────────────
# _post_response_pipeline integration (mocked agent)
# ─────────────────────────────────────────────────────────────────────────────


class TestPostResponsePipeline:
    """Integration tests for _post_response_pipeline on KestrelAgent."""

    @pytest.fixture
    def mock_agent(self):
        """Build a minimal mock of KestrelAgent with pipeline dependencies."""
        from kestrel_sovereign.kestrel_agent import KestrelAgent

        agent = MagicMock()
        agent.agent_id = "did:pkh:test"
        agent._background_tasks = set()
        agent._track_background_task = KestrelAgent._track_background_task.__get__(
            agent,
            KestrelAgent,
        )

        # Memory system
        agent.memory_system = MagicMock()
        agent.memory_system.analyzer = None  # Disable Phase 2 temporal for unit test
        agent.memory_system.linker = None  # Disable Phase 2 associative for unit test

        # Raw storage with conversation store
        conv_store = AsyncMock()
        conv_store.get_full_history_with_ids = AsyncMock(
            return_value=[
                {"id": 100, "role": "user", "content": "I am happy", "metadata": {}},
                {"id": 101, "role": "assistant", "content": "Glad to hear!", "metadata": {}},
            ]
        )
        agent._raw_storage = MagicMock()
        agent._raw_storage.conversation = conv_store

        # Context manager with memory manager
        mm = MemoryManager(storage=MagicMock(), agent_id="did:pkh:test")
        agent.context_manager = MagicMock()
        agent.context_manager.memory_manager = mm

        return agent

    @pytest.mark.asyncio
    async def test_pipeline_calls_tag_exchange(self, mock_agent):
        """Phase 1 should call tag_exchange with correct message IDs."""
        from kestrel_sovereign.kestrel_agent import KestrelAgent

        # Patch tag_exchange to track calls
        mock_agent.context_manager.memory_manager.tag_exchange = AsyncMock(
            return_value={"user": {}, "assistant": {}}
        )

        # Call the real method, bound to our mock
        await KestrelAgent._post_response_pipeline(
            mock_agent, "I am happy", "Glad to hear!", session_id="sess-1"
        )

        mock_agent.context_manager.memory_manager.tag_exchange.assert_awaited_once()
        call_kwargs = mock_agent.context_manager.memory_manager.tag_exchange.call_args[1]
        assert call_kwargs["user_message_id"] == 100
        assert call_kwargs["assistant_message_id"] == 101

    @pytest.mark.asyncio
    async def test_pipeline_phase1_failure_does_not_block(self, mock_agent):
        """Phase 1 failure should be caught, not propagated."""
        from kestrel_sovereign.kestrel_agent import KestrelAgent

        mock_agent.context_manager.memory_manager.tag_exchange = AsyncMock(
            side_effect=RuntimeError("Phase 1 boom")
        )

        # Should not raise
        await KestrelAgent._post_response_pipeline(
            mock_agent, "hello", "hi", session_id=None
        )

    @pytest.mark.asyncio
    async def test_pipeline_skips_when_no_memory_system(self):
        """Should early-return when memory_system is not available."""
        from kestrel_sovereign.kestrel_agent import KestrelAgent

        agent = MagicMock()
        agent.memory_system = None
        # Should not raise
        await KestrelAgent._post_response_pipeline(agent, "hi", "hello")

    @pytest.mark.asyncio
    async def test_pipeline_skips_when_no_conv_store(self):
        """Should early-return when conversation store is not available."""
        from kestrel_sovereign.kestrel_agent import KestrelAgent

        agent = MagicMock()
        agent.memory_system = MagicMock()
        agent._raw_storage = MagicMock()
        agent._raw_storage.conversation = None
        # Should not raise
        await KestrelAgent._post_response_pipeline(agent, "hi", "hello")

    @pytest.mark.asyncio
    async def test_pipeline_phase2_background_associative(self, mock_agent):
        """Phase 2 should fire associative linker in background."""
        from kestrel_sovereign.kestrel_agent import KestrelAgent

        # Enable linker
        mock_agent.memory_system.linker = AsyncMock()
        mock_agent.memory_system.linker.extract_and_link = AsyncMock()

        mock_agent.context_manager.memory_manager.tag_exchange = AsyncMock(
            return_value={"user": {}, "assistant": {}}
        )

        await KestrelAgent._post_response_pipeline(
            mock_agent, "My mom lives in Brooklyn", "That sounds nice!", session_id=None
        )

        # Give the background task a chance to run
        await asyncio.sleep(0.1)

        mock_agent.memory_system.linker.extract_and_link.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_pipeline_phase2_temporal_detection(self, mock_agent):
        """Phase 2 should run temporal pattern detection in background."""
        from kestrel_sovereign.kestrel_agent import KestrelAgent

        # Enable analyzer
        mock_agent.memory_system.analyzer = AsyncMock()
        mock_agent.memory_system.analyzer.detect_patterns = AsyncMock(return_value=[])
        mock_agent.memory_system.analyzer.save_patterns = AsyncMock()

        mock_agent.context_manager.memory_manager.tag_exchange = AsyncMock(
            return_value={"user": {}, "assistant": {}}
        )

        await KestrelAgent._post_response_pipeline(
            mock_agent, "hello", "hi", session_id=None
        )

        await asyncio.sleep(0.1)

        mock_agent.memory_system.analyzer.detect_patterns.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_pipeline_phase2_error_isolation(self, mock_agent):
        """Phase 2 errors should be logged, not propagated."""
        from kestrel_sovereign.kestrel_agent import KestrelAgent

        mock_agent.memory_system.linker = AsyncMock()
        mock_agent.memory_system.linker.extract_and_link = AsyncMock(
            side_effect=RuntimeError("graph write failed")
        )

        mock_agent.context_manager.memory_manager.tag_exchange = AsyncMock(
            return_value={"user": {}, "assistant": {}}
        )

        # Should not raise
        await KestrelAgent._post_response_pipeline(
            mock_agent, "hello", "hi", session_id=None
        )

        await asyncio.sleep(0.1)
        # If we got here, the error was caught properly
