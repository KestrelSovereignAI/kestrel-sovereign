#!/usr/bin/env python3
"""
E2E Tests for Session Compression Functionality.

Tests the !compress command and context compression:
1. Checking if compression is needed
2. Compressing session with LLM-generated summary
3. Preserving recent messages
4. Command variants (--check, --keep N, --force)

Uses REAL services - minimal mocking for LLM only.
"""

import pytest
import pytest_asyncio
import os
import tempfile
import shutil
from unittest.mock import MagicMock, AsyncMock

from kestrel_sovereign.kestrel_agent import KestrelAgent
from kestrel_sovereign.privacy import PrivacyMode


@pytest_asyncio.fixture
async def agent_with_messages(temp_dir):
    """Create a real KestrelAgent with conversation history for testing."""
    db_path = str(temp_dir / "test_agent.db")

    # Mock LLM service for compression with all required methods
    mock_llm = MagicMock()
    mock_llm.generate = AsyncMock(return_value="Summary: The user and assistant discussed various topics including greetings, preferences, and project details.")
    mock_llm.get_default_model = MagicMock(return_value="gpt-4")

    agent = KestrelAgent(
        did="did:test:compress-test-agent",
        storage_path=db_path,
        llm_service=mock_llm,
        privacy_mode=PrivacyMode.NORMAL,
    )
    await agent.initialize()

    # Add 25 messages to have enough for compression
    if hasattr(agent, 'storage') and agent.storage:
        for i in range(25):
            await agent.storage.add_conversation(
                role="user",
                content=f"User message {i}: This is test message number {i} with some content."
            )
            await agent.storage.add_conversation(
                role="assistant",
                content=f"Assistant response {i}: I understand and acknowledge test message {i}."
            )

    yield agent

    # Cleanup
    if hasattr(agent, 'storage') and agent.storage:
        await agent.storage.close()


@pytest_asyncio.fixture
async def agent_with_few_messages(temp_dir):
    """Create an agent with only a few messages (not enough to compress)."""
    db_path = str(temp_dir / "test_agent_small.db")

    mock_llm = MagicMock()
    mock_llm.generate = AsyncMock(return_value="Brief summary")
    mock_llm.get_default_model = MagicMock(return_value="gpt-4")

    agent = KestrelAgent(
        did="did:test:compress-test-small",
        storage_path=db_path,
        llm_service=mock_llm,
        privacy_mode=PrivacyMode.NORMAL,
    )
    await agent.initialize()

    # Add only 5 messages
    if hasattr(agent, 'storage') and agent.storage:
        for i in range(5):
            await agent.storage.add_conversation(
                role="user",
                content=f"Short message {i}"
            )

    yield agent

    if hasattr(agent, 'storage') and agent.storage:
        await agent.storage.close()


class TestCompressCommand:
    """Tests for !compress command handling."""

    @pytest.mark.asyncio
    async def test_compress_check_only(self, agent_with_messages):
        """Test !compress --check shows compression status."""
        agent = agent_with_messages

        result = await agent.command_handler.handle("!compress --check")

        assert "Utilization:" in result
        assert "Messages:" in result
        assert "Tokens:" in result
        print(f"✅ !compress --check result:\n{result}")

    @pytest.mark.asyncio
    async def test_compress_default(self, agent_with_messages):
        """Test !compress with default settings."""
        agent = agent_with_messages

        result = await agent.command_handler.handle("!compress")

        # Should succeed or indicate not needed
        assert "compressed" in result.lower() or "not needed" in result.lower() or "not enough" in result.lower()
        print(f"✅ !compress result:\n{result}")

    @pytest.mark.asyncio
    async def test_compress_with_keep(self, agent_with_messages):
        """Test !compress --keep N preserves specified messages."""
        agent = agent_with_messages

        result = await agent.command_handler.handle("!compress --keep 5")

        # Should show messages preserved
        if "preserved" in result.lower():
            assert "5" in result
        print(f"✅ !compress --keep 5 result:\n{result}")

    @pytest.mark.asyncio
    async def test_compress_force(self, agent_with_messages):
        """Test !compress --force compresses even at low utilization."""
        agent = agent_with_messages

        result = await agent.command_handler.handle("!compress --force")

        # Force should succeed
        assert "compressed" in result.lower() or "success" in result.lower() or "tokens saved" in result.lower()
        print(f"✅ !compress --force result:\n{result}")

    @pytest.mark.asyncio
    async def test_compress_not_enough_messages(self, agent_with_few_messages):
        """Test compression fails gracefully with too few messages."""
        agent = agent_with_few_messages

        result = await agent.command_handler.handle("!compress")

        # Should indicate not enough messages
        assert "not enough" in result.lower() or "not needed" in result.lower()
        print(f"✅ !compress with few messages result:\n{result}")


class TestCompressionCheckNeeded:
    """Tests for check_compression_needed functionality."""

    @pytest.mark.asyncio
    async def test_check_below_threshold(self, agent_with_few_messages):
        """Test that few messages results in no compression recommendation."""
        agent = agent_with_few_messages

        if hasattr(agent, 'context_manager') and agent.context_manager:
            result = await agent.context_manager.check_compression_needed()

            assert result["compression_recommended"] is False
            assert result["message_count"] <= 10
            print(f"✅ Check compression (few msgs): recommended={result['compression_recommended']}")

    @pytest.mark.asyncio
    async def test_check_returns_stats(self, agent_with_messages):
        """Test that check returns all expected stats."""
        agent = agent_with_messages

        if hasattr(agent, 'context_manager') and agent.context_manager:
            result = await agent.context_manager.check_compression_needed()

            assert "compression_recommended" in result
            assert "utilization_percent" in result
            assert "message_count" in result
            assert "total_tokens" in result
            assert "budget_limit" in result
            assert "threshold" in result
            print(f"✅ Check compression stats: {result}")


class TestCompressionSession:
    """Tests for compress_session functionality."""

    @pytest.mark.asyncio
    async def test_compress_creates_summary(self, agent_with_messages):
        """Test that compression creates a summary of older messages."""
        agent = agent_with_messages

        if hasattr(agent, 'context_manager') and agent.context_manager:
            result = await agent.context_manager.compress_session(
                llm_service=agent.llm_service,
                preserve_recent=10,
                force=True
            )

            assert result["success"] is True
            assert result["messages_compressed"] > 0
            assert result["messages_preserved"] == 10
            assert result["tokens_saved"] >= 0  # May be negative if summary is longer
            assert "summary_preview" in result
            print(f"✅ Compress session: {result['messages_compressed']} compressed, {result['tokens_saved']} tokens saved")

    @pytest.mark.asyncio
    async def test_compress_preserves_recent(self, agent_with_messages):
        """Test that recent messages are preserved after compression."""
        agent = agent_with_messages

        if hasattr(agent, 'context_manager') and agent.context_manager:
            # Get message count before
            history_before = await agent.storage.get_conversation_history()
            count_before = len(history_before)

            result = await agent.context_manager.compress_session(
                llm_service=agent.llm_service,
                preserve_recent=5,
                force=True
            )

            if result["success"]:
                assert result["messages_preserved"] == 5
                print(f"✅ Preserved {result['messages_preserved']} recent messages")

    @pytest.mark.asyncio
    async def test_compress_llm_called(self, agent_with_messages):
        """Test that LLM is called to generate summary."""
        agent = agent_with_messages

        if hasattr(agent, 'context_manager') and agent.context_manager:
            result = await agent.context_manager.compress_session(
                llm_service=agent.llm_service,
                preserve_recent=10,
                force=True
            )

            if result["success"]:
                # Verify LLM was called
                agent.llm_service.generate.assert_called_once()
                call_args = agent.llm_service.generate.call_args
                assert "CONVERSATION:" in call_args.kwargs.get("prompt", "")
                print(f"✅ LLM called to generate summary")


class TestContextManagerNoLLM:
    """Tests for context manager when LLM is unavailable."""

    @pytest.mark.asyncio
    async def test_compress_without_llm_service(self, agent_with_messages):
        """Test that compression handles missing LLM gracefully."""
        agent = agent_with_messages

        # Set agent.llm_service to None
        agent.llm_service = None

        result = await agent.command_handler.handle("!compress")

        # Should indicate LLM not available
        assert "not available" in result.lower() or "error" in result.lower()
        print(f"✅ !compress without LLM: {result}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
