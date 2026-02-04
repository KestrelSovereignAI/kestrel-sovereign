"""
Integration tests for the Bootstrap Flow.

Tests the full agent wake-up and discovery flow from end to end.
"""

import pytest
import os
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from kestrel_sovereign.bootstrap import BootstrapService, BootstrapState


class MockLLMResponse:
    """Mock LLM response."""

    def __init__(self, content):
        self.content = content


class MockLLMService:
    """Mock LLM service that returns predetermined responses."""

    def __init__(self, responses=None):
        self.responses = responses or []
        self.call_count = 0

    async def generate(self, messages, temperature=None):
        """Return next response from the list."""
        if self.call_count < len(self.responses):
            response = self.responses[self.call_count]
            self.call_count += 1
            return MockLLMResponse(response)
        return MockLLMResponse("Default response")

    async def generate_with_messages(self, messages, **kwargs):
        """Return next response from the list (same as generate)."""
        return await self.generate(messages)


@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def mock_db():
    """Create an in-memory mock database."""
    class MockDB:
        def __init__(self):
            self.data = {}

        async def fetchall(self, query, params=None):
            key = (params[0], params[1]) if params and len(params) >= 2 else None
            if key in self.data:
                return [(self.data[key],)]
            return []

        async def execute(self, query, params=None):
            if params and len(params) >= 4:
                key = (params[0], params[1])
                self.data[key] = params[2]

    return MockDB()


class TestBootstrapIntegrationFlow:
    """Test the full bootstrap flow from start to finish."""

    @pytest.mark.asyncio
    async def test_full_discovery_flow(self, temp_dir, mock_db):
        """Test complete discovery flow: wake-up -> discovery -> completion."""
        # The SOUL.md content that will be returned for generation
        soul_content_response = """# SOUL.md - You Are TestBot

## Who You Are
You're TestBot, a Kestrel agent working with Alice on coding projects.

## How You Talk
Casual and quick - no unnecessary padding.

## Core Rules
1. Keep it brief
2. Focus on coding
3. Be direct

## First Message
- "Hey Alice, what's up?"
- "Ready to code?"

## The Bottom Line
A helpful coding buddy who gets straight to the point.
"""

        # Set up LLM responses for the discovery conversation
        # Need enough responses for: discovery exchanges + SOUL.md generation
        llm_responses = [
            # Response to first user message
            "Nice to meet you, Alice! I like that - casual and quick works for me too. "
            "What kind of stuff will we be working on together?",
            # Response to second user message - should offer avatar
            "Got it - coding and analysis. Sounds fun! "
            "One more thing - would you like to give me a face? "
            "Describe how you imagine me and I'll generate an avatar.",
            # Response when user skips avatar
            "No problem! Let's get started.",
            # SOUL.md generation response (called by complete_bootstrap)
            soul_content_response,
        ]

        mock_llm = MockLLMService(llm_responses)

        # Create bootstrap service
        agent_dir = temp_dir / "agent_data" / "testbot"
        agent_dir.mkdir(parents=True)

        service = BootstrapService(
            db=mock_db,
            agent_id="did:test:123",
            agent_name="TestBot",
            llm_service=mock_llm,
            agent_data_path=agent_dir,
        )

        # Step 1: Verify initial state is PENDING
        assert await service.get_bootstrap_state() == BootstrapState.PENDING
        assert await service.is_bootstrap_needed() is True

        # Step 2: Generate wake-up message
        wake_up = await service.generate_wake_up_message()
        assert "TestBot" in wake_up
        assert "?" in wake_up  # Should ask a question

        # Transition to discovery
        await service.set_bootstrap_state(BootstrapState.DISCOVERY)

        # Step 3: First discovery exchange
        response1, complete1, avatar1 = await service.process_discovery_message(
            "Hi! I'm Alice. I prefer casual, quick communication."
        )
        assert "Alice" in response1
        assert complete1 is False  # Not complete yet

        # Step 4: Second discovery exchange
        response2, complete2, avatar2 = await service.process_discovery_message(
            "Mostly coding projects and data analysis."
        )
        assert "avatar" in response2.lower()  # Should offer avatar

        # Step 5: Skip avatar - this triggers completion
        response3, complete3, avatar3 = await service.process_discovery_message("skip")
        assert complete3 is True
        assert avatar3 is False

        # Step 6: Complete bootstrap (generates SOUL.md)
        completion_msg = await service.complete_bootstrap()
        assert "ready" in completion_msg.lower() or "meet" in completion_msg.lower()

        # Step 7: Verify SOUL.md was created
        soul_path = agent_dir / "SOUL.md"
        assert soul_path.exists()
        saved_content = soul_path.read_text()
        assert "SOUL.md" in saved_content
        assert "TestBot" in saved_content

        # Step 8: Verify state is complete
        await service.set_bootstrap_state(BootstrapState.COMPLETE)
        assert await service.is_bootstrap_needed() is False

    @pytest.mark.asyncio
    async def test_skip_discovery_flow(self, temp_dir, mock_db):
        """Test skipping discovery entirely."""
        mock_llm = MockLLMService()

        agent_dir = temp_dir / "agent_data" / "skipper"
        agent_dir.mkdir(parents=True)

        service = BootstrapService(
            db=mock_db,
            agent_id="did:test:456",
            agent_name="Skipper",
            llm_service=mock_llm,
            agent_data_path=agent_dir,
        )

        # Skip discovery
        result = await service.skip_discovery()
        assert "default" in result.lower() or "personality" in result.lower()

        # Verify SOUL.md was created with default content
        soul_path = agent_dir / "SOUL.md"
        assert soul_path.exists()

        # Verify state is complete
        assert await service.is_bootstrap_needed() is False

    @pytest.mark.asyncio
    async def test_restart_discovery_flow(self, temp_dir, mock_db):
        """Test restarting discovery after completion."""
        mock_llm = MockLLMService(["Test response"])

        agent_dir = temp_dir / "agent_data" / "restarter"
        agent_dir.mkdir(parents=True)

        service = BootstrapService(
            db=mock_db,
            agent_id="did:test:789",
            agent_name="Restarter",
            llm_service=mock_llm,
            agent_data_path=agent_dir,
        )

        # Complete bootstrap first
        await service.skip_discovery()
        assert await service.is_bootstrap_needed() is False

        # Verify SOUL.md exists
        soul_path = agent_dir / "SOUL.md"
        assert soul_path.exists()

        # Restart discovery
        result = await service.restart_discovery()
        assert "reset" in result.lower()

        # Verify state is back to pending
        assert await service.get_bootstrap_state() == BootstrapState.PENDING
        assert await service.is_bootstrap_needed() is True

        # Verify SOUL.md was deleted
        assert not soul_path.exists()

        # Verify history was cleared
        history = await service.get_discovery_history()
        assert len(history) == 0


class TestBootstrapWithExistingAgents:
    """Test bootstrap behavior with agents that already have data."""

    @pytest.mark.asyncio
    async def test_existing_agent_with_soul_md(self, temp_dir, mock_db):
        """Agents with existing SOUL.md should skip bootstrap."""
        mock_llm = MockLLMService()

        agent_dir = temp_dir / "agent_data" / "existing"
        agent_dir.mkdir(parents=True)

        # Create existing SOUL.md
        soul_path = agent_dir / "SOUL.md"
        soul_path.write_text("# Existing SOUL.md content")

        # Set bootstrap state to complete
        mock_db.data[("did:test:existing", "bootstrap_state")] = "complete"

        service = BootstrapService(
            db=mock_db,
            agent_id="did:test:existing",
            agent_name="ExistingAgent",
            llm_service=mock_llm,
            agent_data_path=agent_dir,
        )

        # Should not need bootstrap
        assert await service.is_bootstrap_needed() is False


class TestBootstrapErrorHandling:
    """Test error handling in bootstrap flow."""

    @pytest.mark.asyncio
    async def test_llm_error_during_discovery(self, temp_dir, mock_db):
        """Discovery should handle LLM errors gracefully."""

        class FailingLLM:
            async def generate(self, messages, temperature=None):
                raise Exception("LLM unavailable")

        agent_dir = temp_dir / "agent_data" / "error_test"
        agent_dir.mkdir(parents=True)

        service = BootstrapService(
            db=mock_db,
            agent_id="did:test:error",
            agent_name="ErrorTest",
            llm_service=FailingLLM(),
            agent_data_path=agent_dir,
        )

        await service.set_bootstrap_state(BootstrapState.DISCOVERY)

        # Should return a fallback response, not crash
        response, complete, avatar = await service.process_discovery_message("Test message")
        assert response is not None
        assert "trouble" in response.lower() or "more about yourself" in response.lower()

    @pytest.mark.asyncio
    async def test_missing_agent_data_path(self, mock_db):
        """Bootstrap should handle missing agent_data_path."""
        mock_llm = MockLLMService()

        service = BootstrapService(
            db=mock_db,
            agent_id="did:test:nopath",
            agent_name="NoPath",
            llm_service=mock_llm,
            agent_data_path=None,  # No path provided
        )

        # Should still work, just won't save SOUL.md
        result = await service.skip_discovery()
        assert "default" in result.lower() or "personality" in result.lower()

        # save_soul_md should return False
        saved = await service.save_soul_md("# Test")
        assert saved is False
