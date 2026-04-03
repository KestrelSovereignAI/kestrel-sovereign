"""
Integration tests for MemoryFeature.

Tests the memory tools with real storage to ensure they work end-to-end.
"""
import pytest
import tempfile
import os
from pathlib import Path

from kestrel_sovereign.features.memory import MemoryFeature
from kestrel_sovereign.storage import AsyncStorage


class MockAgent:
    """Minimal mock agent for testing MemoryFeature."""

    def __init__(self, storage: AsyncStorage):
        self.storage = storage
        self.did = "did:test:memory-feature-test"
        self.memory_consolidator = None
        self.memory_retriever = None


@pytest.fixture
async def temp_storage():
    """Create temporary storage for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        agent_id = "test-agent-memory-feature"

        storage = AsyncStorage(str(db_path), agent_id=agent_id)
        await storage.initialize()

        # Add some test conversations
        await storage.add_conversation("user", "Hello, I'm testing memory")
        await storage.add_conversation("assistant", "Hi! I'll remember this.")
        await storage.add_conversation("user", "My favorite color is blue")
        await storage.add_conversation("assistant", "Blue is a great color!")
        await storage.add_conversation("user", "Can you remember what we discussed?")

        yield storage

        await storage.close()


@pytest.fixture
async def memory_feature(temp_storage):
    """Create MemoryFeature with real storage."""
    agent = MockAgent(temp_storage)
    feature = MemoryFeature(agent)
    await feature.initialize()
    return feature


class TestMemoryFeatureDiscovery:
    """Tests that MemoryFeature is properly discoverable."""

    def test_feature_has_tools(self, memory_feature):
        """MemoryFeature should expose tools."""
        tools = memory_feature.get_tools()
        assert len(tools) >= 7, "Should have at least 7 memory tools"

    def test_tool_names(self, memory_feature):
        """Tools should have expected names."""
        tools = memory_feature.get_tools()
        tool_names = {t.name for t in tools}

        expected = {
            "search_memory",
            "recall_recent",
            "search_documents",
            "search_case_law",
            "get_episodes",
            "memory_status",
            "full_history_search",
            "recall_emotional",
        }
        assert expected.issubset(tool_names), f"Missing tools: {expected - tool_names}"


class TestSearchMemory:
    """Tests for search_memory tool."""

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        os.getenv("KESTREL_DATA_KEY") is not None,
        reason="Search doesn't work on encrypted content (uses SQL LIKE on encrypted text)"
    )
    @pytest.mark.asyncio
    async def test_search_finds_matching_content(self, memory_feature):
        """Should find messages containing search term.

        Note: This test is skipped when encryption is enabled because search_history()
        uses SQL LIKE on encrypted content, which won't match plaintext queries.
        """
        result = await memory_feature.search_memory("blue", limit=5)

        assert result["success"] is True
        assert result["count"] >= 1
        assert result["query"] == "blue"

    @pytest.mark.asyncio
    async def test_search_with_no_matches(self, memory_feature):
        """Should return empty results for non-matching query."""
        result = await memory_feature.search_memory("xyznonexistent123", limit=5)

        assert result["success"] is True
        assert result["count"] == 0

    @pytest.mark.asyncio
    async def test_search_limit_is_respected(self, memory_feature):
        """Should respect the limit parameter."""
        result = await memory_feature.search_memory("", limit=2)

        assert result["success"] is True
        assert result["count"] <= 2


class TestRecallRecent:
    """Tests for recall_recent tool."""

    @pytest.mark.asyncio
    async def test_recall_returns_messages(self, memory_feature):
        """Should return recent conversation messages."""
        result = await memory_feature.recall_recent(limit=10)

        assert result["success"] is True
        assert result["count"] >= 1
        assert "messages" in result
        assert len(result["messages"]) >= 1

    @pytest.mark.asyncio
    async def test_recall_limit_works(self, memory_feature):
        """Should respect the limit parameter."""
        result = await memory_feature.recall_recent(limit=2)

        assert result["success"] is True
        assert result["count"] <= 2


class TestMemoryStatus:
    """Tests for memory_status tool."""

    @pytest.mark.asyncio
    async def test_status_returns_info(self, memory_feature):
        """Should return memory system status."""
        result = await memory_feature.memory_status()

        assert result["success"] is True
        assert "total_messages" in result
        assert result["total_messages"] >= 5  # We added 5 test messages
        assert "agent_id" in result
        assert "consolidator_available" in result


class TestFullHistorySearch:
    """Tests for full_history_search tool."""

    @pytest.mark.asyncio
    async def test_full_search_finds_content(self, memory_feature):
        """Should find content via full history search."""
        result = await memory_feature.full_history_search("blue", limit=5)

        assert result["success"] is True
        assert result["count"] >= 1
        assert result["query"] == "blue"
        assert "total_searched" in result

    @pytest.mark.asyncio
    async def test_full_search_case_insensitive(self, memory_feature):
        """Search should be case insensitive."""
        result = await memory_feature.full_history_search("BLUE", limit=5)

        assert result["success"] is True
        assert result["count"] >= 1


class TestSearchDocuments:
    """Tests for search_documents tool (RAG)."""

    @pytest.mark.asyncio
    async def test_search_docs_handles_empty(self, memory_feature):
        """Should handle empty RAG gracefully."""
        result = await memory_feature.search_documents("anything", limit=5)

        # Should succeed even with no documents
        assert result["success"] is True
        assert "results" in result


class TestGetEpisodes:
    """Tests for get_episodes tool."""

    @pytest.mark.asyncio
    async def test_episodes_without_consolidator(self, memory_feature):
        """Should report consolidator not available."""
        result = await memory_feature.get_episodes(limit=5)

        # Our mock agent doesn't have a consolidator
        assert result["success"] is False
        assert "consolidator" in result["error"].lower()


class TestRecallEmotional:
    """Tests for recall_emotional tool."""

    @pytest.mark.asyncio
    async def test_recall_without_retriever(self, memory_feature):
        """Should fallback gracefully without memory_retriever."""
        result = await memory_feature.recall_emotional("memory test", mood="positive")

        # Our mock agent doesn't have memory_retriever
        assert result["success"] is False
        assert "retriever" in result["error"].lower()
        # Should have fallback results
        assert "fallback" in result


class TestCommandParsing:
    """Tests for command prefix parsing."""

    @pytest.mark.asyncio
    async def test_can_handle_memory_commands(self, memory_feature):
        """Tools should handle their command prefixes."""
        tools = memory_feature.get_tools()

        search_tool = next((t for t in tools if t.name == "search_memory"), None)
        assert search_tool is not None
        assert search_tool.can_handle_command("!memory search test")
        assert not search_tool.can_handle_command("!other command")

    @pytest.mark.asyncio
    async def test_parse_multi_word_prefix(self, memory_feature):
        """Should correctly parse args with multi-word prefix."""
        tools = memory_feature.get_tools()
        search_tool = next((t for t in tools if t.name == "search_memory"), None)

        args = search_tool.parse_command_args("!memory search blue 5")
        assert args["query"] == "blue"
        assert args["limit"] == 5
        assert isinstance(args["limit"], int)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
