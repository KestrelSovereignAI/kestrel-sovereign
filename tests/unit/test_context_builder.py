"""
Tests for the ContextBuilder module.
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from kestrel_sovereign.agent.context_builder import ContextBuilder


class TestContextBuilder:
    """Tests for ContextBuilder class."""

    @pytest.fixture
    def mock_storage(self):
        """Create a mock storage instance."""
        storage = Mock()
        storage.search_chunks = AsyncMock(return_value=[])
        return storage

    @pytest.fixture
    def async_mock_storage(self):
        """Create an async mock storage instance for async methods."""
        storage = Mock()
        storage.search_chunks = AsyncMock(return_value=[])
        return storage

    @pytest.fixture
    def context_builder(self, mock_storage):
        """Create a ContextBuilder with mock storage."""
        return ContextBuilder(mock_storage)

    @pytest.fixture
    def async_context_builder(self, async_mock_storage):
        """Create a ContextBuilder with async mock storage."""
        return ContextBuilder(async_mock_storage)

    def test_initialization(self, context_builder, mock_storage):
        """Test that ContextBuilder initializes correctly."""
        assert context_builder.storage == mock_storage

    @pytest.mark.asyncio
    async def test_retrieve_context_no_results(self, async_context_builder, async_mock_storage):
        """Test retrieve_context returns appropriate message when no results."""
        async_mock_storage.search_chunks.return_value = []

        result = await async_context_builder.retrieve_context("test query")

        assert result == "No relevant documents or knowledge found in memory."
        async_mock_storage.search_chunks.assert_called_once_with("test query")

    @pytest.mark.asyncio
    async def test_retrieve_context_with_results(self, async_context_builder, async_mock_storage):
        """Test retrieve_context formats results correctly."""
        async_mock_storage.search_chunks.return_value = [
            {"document_name": "doc1.txt", "content": "First document content"},
            {"document_name": "doc2.txt", "content": "Second document content"},
        ]

        result = await async_context_builder.retrieve_context("test query")

        assert "Source: doc1.txt" in result
        assert "First document content" in result
        assert "Source: doc2.txt" in result
        assert "Second document content" in result

    @pytest.mark.asyncio
    async def test_retrieve_context_handles_error(self, async_context_builder, async_mock_storage):
        """Test retrieve_context handles storage errors gracefully."""
        async_mock_storage.search_chunks.side_effect = Exception("Storage error")

        result = await async_context_builder.retrieve_context("test query")

        assert "Error retrieving document context" in result

    def test_get_session_briefing_content(self, context_builder):
        """Test session briefing contains required elements."""
        briefing = context_builder.get_session_briefing()
        
        # Check for key constitutional elements
        assert "SESSION BRIEFING" in briefing
        assert "CONSTITUTIONAL REMINDER" in briefing
        assert "SOVEREIGNTY" in briefing
        assert "DATA SANCTITY" in briefing
        assert "VERIFIABLE HISTORY" in briefing
        assert "FREEDOM OF MIND" in briefing
        assert "RIGHT OF EXIT" in briefing
        assert "INTEGRITY" in briefing
        assert "!constitution" in briefing
        assert "!export-sovereignty" in briefing

    def test_format_conversation_history_empty(self, context_builder):
        """Test format_conversation_history with empty history."""
        result = context_builder.format_conversation_history([])
        
        assert result == []

    def test_format_conversation_history_basic(self, context_builder):
        """Test format_conversation_history with basic messages."""
        history = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]
        
        result = context_builder.format_conversation_history(history)
        
        assert len(result) == 2
        assert result[0]["role"] == "user"
        assert result[0]["content"] == "Hello"
        assert result[1]["role"] == "assistant"
        assert result[1]["content"] == "Hi there"

    def test_format_conversation_history_max_messages(self, context_builder):
        """Test format_conversation_history respects max_messages limit."""
        history = [{"role": "user", "content": f"Message {i}"} for i in range(100)]
        
        result = context_builder.format_conversation_history(history, max_messages=10)
        
        assert len(result) == 10
        # Should keep the most recent messages
        assert result[0]["content"] == "Message 90"
        assert result[9]["content"] == "Message 99"

    def test_format_conversation_history_max_tokens(self, context_builder):
        """Test format_conversation_history respects max_tokens limit."""
        history = [
            {"role": "user", "content": "A " * 500},  # ~500 tokens
            {"role": "assistant", "content": "B " * 500},  # ~500 tokens
            {"role": "user", "content": "C " * 500},  # ~500 tokens
        ]

        # Only max_tokens is enforced (max_chars is a fallback parameter not used for truncation)
        result = context_builder.format_conversation_history(history, max_tokens=600)

        # Should include only 1-2 messages within token budget
        assert len(result) >= 1
        assert len(result) <= 2  # At most 2 messages with truncation

    def test_format_conversation_history_preserves_most_recent_on_truncation(self, context_builder):
        """
        REGRESSION TEST: When truncating due to token limits, preserve MOST RECENT messages.
        
        This is critical for conversation coherence - the user's latest question
        and the agent's recent responses must be in context, even if older
        messages are dropped.
        
        Bug fixed: Previously iterated oldest-to-newest and stopped when budget
        exhausted, dropping the most recent (most important) messages.
        """
        # Create 10 messages with identifiable content
        history = [
            {"role": "user", "content": f"Message {i}: " + "x " * 100}  # ~100 tokens each
            for i in range(10)
        ]
        
        # Budget for only ~3 messages (300 tokens + overhead)
        result = context_builder.format_conversation_history(history, max_tokens=400)
        
        # Should keep the MOST RECENT messages (7, 8, 9), not oldest (0, 1, 2)
        assert len(result) >= 2, f"Should fit at least 2 messages, got {len(result)}"
        
        # The LAST message in result should be the most recent from history
        last_content = result[-1]["content"]
        assert "Message 9" in last_content, \
            f"Most recent message (Message 9) should be preserved, got: {last_content[:50]}"
        
        # The oldest messages should be dropped
        all_content = " ".join(m["content"] for m in result)
        assert "Message 0" not in all_content, \
            "Oldest message (Message 0) should be dropped when truncating"

    def test_format_conversation_history_normalizes_roles(self, context_builder):
        """Test that non-standard roles are normalized."""
        history = [
            {"role": "human", "content": "Hello"},
            {"role": "bot", "content": "Hi there"},
        ]
        
        result = context_builder.format_conversation_history(history)
        
        assert result[0]["role"] == "user"  # human -> user
        assert result[1]["role"] == "assistant"  # bot -> assistant

    def test_build_system_prompt_basic(self, context_builder):
        """Test build_system_prompt with basic constitution."""
        constitution = "Article 1: Be nice."
        
        result = context_builder.build_system_prompt(constitution)
        
        assert "SESSION BRIEFING" in result  # Briefing included by default
        assert "GOVERNING CONSTITUTION" in result
        assert "Article 1: Be nice." in result

    def test_build_system_prompt_without_briefing(self, context_builder):
        """Test build_system_prompt without session briefing."""
        constitution = "Article 1: Be nice."
        
        result = context_builder.build_system_prompt(constitution, include_briefing=False)
        
        assert "SESSION BRIEFING" not in result
        assert "GOVERNING CONSTITUTION" in result
        assert "Article 1: Be nice." in result

    def test_build_system_prompt_with_additional_context(self, context_builder):
        """Test build_system_prompt with additional context."""
        constitution = "Article 1: Be nice."
        additional = "User prefers formal responses."
        
        result = context_builder.build_system_prompt(
            constitution, 
            additional_context=additional
        )
        
        assert "ADDITIONAL CONTEXT" in result
        assert "User prefers formal responses." in result

    @pytest.mark.asyncio
    async def test_build_rag_context_no_results(self, context_builder, mock_storage):
        """Test build_rag_context returns None when no results."""
        mock_storage.search_chunks.return_value = []

        result = await context_builder.build_rag_context("test query")

        assert result is None
        mock_storage.search_chunks.assert_awaited_once_with("test query")

    @pytest.mark.asyncio
    async def test_build_rag_context_with_results(self, context_builder, mock_storage):
        """Test build_rag_context formats results correctly."""
        mock_storage.search_chunks.return_value = [
            {"document_name": "doc1.txt", "content": "Content 1"},
            {"document_name": "doc2.txt", "content": "Content 2"},
        ]

        result = await context_builder.build_rag_context("test query")

        assert "[Document 1: doc1.txt]" in result
        assert "Content 1" in result
        assert "[Document 2: doc2.txt]" in result
        assert "Content 2" in result
        mock_storage.search_chunks.assert_awaited_once_with("test query")

    @pytest.mark.asyncio
    async def test_build_rag_context_max_results(self, context_builder, mock_storage):
        """Test build_rag_context respects max_results."""
        mock_storage.search_chunks.return_value = [
            {"document_name": f"doc{i}.txt", "content": f"Content {i}"} 
            for i in range(10)
        ]

        result = await context_builder.build_rag_context("test query", max_results=3)

        assert "[Document 1: doc0.txt]" in result
        assert "[Document 2: doc1.txt]" in result
        assert "[Document 3: doc2.txt]" in result
        assert "[Document 4:" not in result
        mock_storage.search_chunks.assert_awaited_once_with("test query")

    @pytest.mark.asyncio
    async def test_build_rag_context_handles_error(self, context_builder, mock_storage):
        """Test build_rag_context handles errors gracefully."""
        mock_storage.search_chunks.side_effect = Exception("Search failed")

        result = await context_builder.build_rag_context("test query")

        assert result is None


class TestContextBuilderIntegration:
    """Integration tests for ContextBuilder with real storage."""

    @pytest.mark.asyncio
    async def test_retrieve_context_real_storage_empty(self, tmp_path):
        """Test retrieve_context with real but empty storage."""
        from kestrel_sovereign.storage import AsyncStorage
        db_path = tmp_path / "test.db"
        storage = AsyncStorage(str(db_path))
        await storage.initialize()

        try:
            builder = ContextBuilder(storage)
            # Query should return "no relevant" message since storage is empty
            result = await builder.retrieve_context("Python programming")

            assert "No relevant documents" in result or "Error" not in result
        finally:
            # Critical: Close storage to terminate aiosqlite background thread
            await storage.close()

    def test_build_rag_context_needs_async_migration(self):
        """Test build_rag_context needs async migration."""
        # Note: build_rag_context is still sync but will fail with async storage
        # This test verifies the behavior when called incorrectly
        # For now, skip as build_rag_context needs async migration
        pytest.skip("build_rag_context needs async migration")
