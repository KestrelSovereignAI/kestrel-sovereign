#!/usr/bin/env python3
"""
E2E Tests for Session Compaction Functionality.

Tests the !compact command and context compaction:
1. Checking if compaction is needed
2. Compacting session with LLM-generated summary
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

    # Mock LLM service for compaction with all required methods
    mock_llm = MagicMock()
    mock_llm.generate = AsyncMock(return_value="Summary: The user and assistant discussed various topics including greetings, preferences, and project details.")
    mock_llm.get_default_model = MagicMock(return_value="gpt-4")

    agent = KestrelAgent(
        did="did:test:compact-test-agent",
        storage_path=db_path,
        llm_service=mock_llm,
        privacy_mode=PrivacyMode.NORMAL,
    )
    await agent.initialize()

    # Add 25 messages to have enough for compaction
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
    """Create an agent with only a few messages (not enough to compact)."""
    db_path = str(temp_dir / "test_agent_small.db")

    mock_llm = MagicMock()
    mock_llm.generate = AsyncMock(return_value="Brief summary")
    mock_llm.get_default_model = MagicMock(return_value="gpt-4")

    agent = KestrelAgent(
        did="did:test:compact-test-small",
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


class TestCompactCommand:
    """Tests for !compact command handling."""

    @pytest.mark.asyncio
    async def test_compact_check_only(self, agent_with_messages):
        """Test !compact --check shows compaction status."""
        agent = agent_with_messages

        result = await agent.command_handler.handle("!compact --check")

        assert "Utilization:" in result
        assert "Messages:" in result
        assert "Tokens:" in result
        print(f"✅ !compact --check result:\n{result}")

    @pytest.mark.asyncio
    async def test_compact_default(self, agent_with_messages):
        """Test !compact with default settings."""
        agent = agent_with_messages

        result = await agent.command_handler.handle("!compact")

        # Should succeed or indicate not needed
        assert "compacted" in result.lower() or "not needed" in result.lower() or "not enough" in result.lower()
        print(f"✅ !compact result:\n{result}")

    @pytest.mark.asyncio
    async def test_compact_with_keep(self, agent_with_messages):
        """Test !compact --keep N preserves specified messages."""
        agent = agent_with_messages

        result = await agent.command_handler.handle("!compact --keep 5")

        # Should show messages preserved
        if "preserved" in result.lower():
            assert "5" in result
        print(f"✅ !compact --keep 5 result:\n{result}")

    @pytest.mark.asyncio
    async def test_compact_force(self, agent_with_messages):
        """Test !compact --force compacts even at low utilization."""
        agent = agent_with_messages

        result = await agent.command_handler.handle("!compact --force")

        # Force should succeed
        assert "compacted" in result.lower() or "success" in result.lower() or "tokens saved" in result.lower()
        print(f"✅ !compact --force result:\n{result}")

    @pytest.mark.asyncio
    async def test_compact_not_enough_messages(self, agent_with_few_messages):
        """Test compaction fails gracefully with too few messages."""
        agent = agent_with_few_messages

        result = await agent.command_handler.handle("!compact")

        # Should indicate not enough messages
        assert "not enough" in result.lower() or "not needed" in result.lower()
        print(f"✅ !compact with few messages result:\n{result}")


class TestCompactionCheckNeeded:
    """Tests for check_compaction_needed functionality."""

    @pytest.mark.asyncio
    async def test_check_below_threshold(self, agent_with_few_messages):
        """Test that few messages results in no compaction recommendation."""
        agent = agent_with_few_messages

        if hasattr(agent, 'context_manager') and agent.context_manager:
            result = await agent.context_manager.check_compaction_needed()

            assert result["compaction_recommended"] is False
            assert result["message_count"] <= 10
            print(f"✅ Check compaction (few msgs): recommended={result['compaction_recommended']}")

    @pytest.mark.asyncio
    async def test_check_returns_stats(self, agent_with_messages):
        """Test that check returns all expected stats."""
        agent = agent_with_messages

        if hasattr(agent, 'context_manager') and agent.context_manager:
            result = await agent.context_manager.check_compaction_needed()

            assert "compaction_recommended" in result
            assert "utilization_percent" in result
            assert "message_count" in result
            assert "total_tokens" in result
            assert "budget_limit" in result
            assert "threshold" in result
            print(f"✅ Check compaction stats: {result}")


class TestCompactionSession:
    """Tests for compact_session functionality."""

    @pytest.mark.asyncio
    async def test_compact_creates_summary(self, agent_with_messages):
        """Test that compaction creates a summary of older messages."""
        agent = agent_with_messages

        if hasattr(agent, 'context_manager') and agent.context_manager:
            result = await agent.context_manager.compact_session(
                llm_service=agent.llm_service,
                preserve_recent=10,
                force=True
            )

            assert result["success"] is True
            assert result["messages_compacted"] > 0
            assert result["messages_preserved"] == 10
            assert result["tokens_saved"] >= 0  # May be negative if summary is longer
            assert "summary_preview" in result
            print(f"✅ Compact session: {result['messages_compacted']} compacted, {result['tokens_saved']} tokens saved")

    @pytest.mark.asyncio
    async def test_compact_preserves_recent(self, agent_with_messages):
        """Test that recent messages are preserved after compaction."""
        agent = agent_with_messages

        if hasattr(agent, 'context_manager') and agent.context_manager:
            # Get message count before
            history_before = await agent.storage.get_conversation_history()
            count_before = len(history_before)

            result = await agent.context_manager.compact_session(
                llm_service=agent.llm_service,
                preserve_recent=5,
                force=True
            )

            if result["success"]:
                assert result["messages_preserved"] == 5
                print(f"✅ Preserved {result['messages_preserved']} recent messages")

    @pytest.mark.asyncio
    async def test_compact_llm_called(self, agent_with_messages):
        """Test that LLM is called to generate summary."""
        agent = agent_with_messages

        if hasattr(agent, 'context_manager') and agent.context_manager:
            result = await agent.context_manager.compact_session(
                llm_service=agent.llm_service,
                preserve_recent=10,
                force=True
            )

            if result["success"]:
                # Verify LLM was called
                agent.llm_service.generate.assert_called_once()
                call_args = agent.llm_service.generate.call_args
                # generate is keyword-only with user_prompt (not prompt) — the
                # old prompt= call silently no-op'd compaction (#1844).
                assert "CONVERSATION:" in call_args.kwargs.get("user_prompt", "")
                print(f"✅ LLM called to generate summary")


class TestContextManagerNoLLM:
    """Tests for context manager when LLM is unavailable."""

    @pytest.mark.asyncio
    async def test_compact_without_llm_service(self, agent_with_messages):
        """Test that compaction handles missing LLM gracefully."""
        agent = agent_with_messages

        # Set agent.llm_service to None
        agent.llm_service = None

        result = await agent.command_handler.handle("!compact")

        # Should indicate LLM not available
        assert "not available" in result.lower() or "error" in result.lower()
        print(f"✅ !compact without LLM: {result}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
