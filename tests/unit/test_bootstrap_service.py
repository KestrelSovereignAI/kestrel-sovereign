"""
Unit tests for the Bootstrap Service.

Tests the agent wake-up and personality discovery system.
"""

import pytest
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timezone

from kestrel_sovereign.bootstrap.service import BootstrapService, BootstrapState


class MockDB:
    """Mock database for testing."""

    def __init__(self):
        self.data = {}

    async def fetchall(self, query: str, params: tuple = None):
        """Mock fetchall."""
        key = (params[0], params[1]) if params and len(params) >= 2 else None
        if key and key in self.data:
            return [(self.data[key],)]
        return []

    async def execute(self, query: str, params: tuple = None):
        """Mock execute."""
        if params and len(params) >= 4:
            key = (params[0], params[1])
            self.data[key] = params[2]


class MockLLMService:
    """Mock LLM service for testing."""

    def __init__(self, responses=None):
        self.responses = responses or ["Hello! Nice to meet you!"]
        self.call_count = 0

    async def generate(self, messages, temperature=None):
        """Mock generate."""
        response = self.responses[min(self.call_count, len(self.responses) - 1)]
        self.call_count += 1

        class MockResponse:
            content = response

        return MockResponse()

    async def generate_with_messages(self, messages, **kwargs):
        """Mock generate_with_messages (same as generate)."""
        return await self.generate(messages)


@pytest.fixture
def mock_db():
    """Create a mock database."""
    return MockDB()


@pytest.fixture
def mock_llm():
    """Create a mock LLM service."""
    return MockLLMService()


@pytest.fixture
def temp_agent_dir(tmp_path):
    """Create a temporary agent data directory."""
    agent_dir = tmp_path / "agent_data" / "test_agent"
    agent_dir.mkdir(parents=True)
    return agent_dir


@pytest.fixture
def bootstrap_service(mock_db, mock_llm, temp_agent_dir):
    """Create a BootstrapService for testing."""
    return BootstrapService(
        db=mock_db,
        agent_id="did:pkh:eip155:1:0x123",
        agent_name="TestAgent",
        llm_service=mock_llm,
        agent_data_path=temp_agent_dir,
    )


class TestBootstrapState:
    """Tests for bootstrap state management."""

    @pytest.mark.asyncio
    async def test_initial_state_is_pending(self, bootstrap_service):
        """New agents should start in PENDING state."""
        state = await bootstrap_service.get_bootstrap_state()
        assert state == BootstrapState.PENDING

    @pytest.mark.asyncio
    async def test_set_state_to_discovery(self, bootstrap_service):
        """Should be able to transition to DISCOVERY state."""
        await bootstrap_service.set_bootstrap_state(BootstrapState.DISCOVERY)
        state = await bootstrap_service.get_bootstrap_state()
        assert state == BootstrapState.DISCOVERY

    @pytest.mark.asyncio
    async def test_set_state_to_complete(self, bootstrap_service):
        """Should be able to transition to COMPLETE state."""
        await bootstrap_service.set_bootstrap_state(BootstrapState.COMPLETE)
        state = await bootstrap_service.get_bootstrap_state()
        assert state == BootstrapState.COMPLETE

    @pytest.mark.asyncio
    async def test_is_bootstrap_needed_when_pending(self, bootstrap_service):
        """Bootstrap should be needed when state is PENDING."""
        assert await bootstrap_service.is_bootstrap_needed() is True

    @pytest.mark.asyncio
    async def test_is_bootstrap_needed_when_discovery(self, bootstrap_service):
        """Bootstrap should be needed when state is DISCOVERY."""
        await bootstrap_service.set_bootstrap_state(BootstrapState.DISCOVERY)
        assert await bootstrap_service.is_bootstrap_needed() is True

    @pytest.mark.asyncio
    async def test_is_bootstrap_not_needed_when_complete(self, bootstrap_service):
        """Bootstrap should not be needed when state is COMPLETE."""
        await bootstrap_service.set_bootstrap_state(BootstrapState.COMPLETE)
        assert await bootstrap_service.is_bootstrap_needed() is False


class TestWakeUpMessage:
    """Tests for wake-up message generation."""

    @pytest.mark.asyncio
    async def test_wake_up_message_contains_name(self, bootstrap_service):
        """Wake-up message should contain the agent's name."""
        message = await bootstrap_service.generate_wake_up_message()
        assert "TestAgent" in message

    @pytest.mark.asyncio
    async def test_wake_up_message_is_warm(self, bootstrap_service):
        """Wake-up message should have warm, friendly tone."""
        message = await bootstrap_service.generate_wake_up_message()
        # Check for characteristic warm phrases
        assert "Hey" in message or "woke up" in message
        assert "?" in message  # Should ask a question

    @pytest.mark.asyncio
    async def test_wake_up_message_asks_about_user(self, bootstrap_service):
        """Wake-up message should ask about the user."""
        message = await bootstrap_service.generate_wake_up_message()
        # Should ask for name or preferences
        assert "call you" in message.lower() or "work together" in message.lower()


class TestDiscoveryFlow:
    """Tests for the discovery conversation flow."""

    @pytest.mark.asyncio
    async def test_discovery_stores_history(self, bootstrap_service):
        """Discovery messages should be stored in history."""
        await bootstrap_service.process_discovery_message("Hi, I'm Alice!")
        history = await bootstrap_service.get_discovery_history()
        assert len(history) == 2  # User message + assistant response
        assert history[0]["role"] == "user"
        assert history[0]["content"] == "Hi, I'm Alice!"

    @pytest.mark.asyncio
    async def test_discovery_returns_response(self, bootstrap_service):
        """Discovery should return an LLM response."""
        response, is_complete, wants_avatar = await bootstrap_service.process_discovery_message(
            "Hi, I'm Alice!"
        )
        assert response is not None
        assert len(response) > 0

    @pytest.mark.asyncio
    async def test_skip_triggers_completion(self, bootstrap_service):
        """Saying 'skip' should complete discovery."""
        response, is_complete, wants_avatar = await bootstrap_service.process_discovery_message(
            "!skip-discovery"
        )
        assert is_complete is True
        assert wants_avatar is False

    @pytest.mark.asyncio
    async def test_lets_start_triggers_completion(self, bootstrap_service):
        """Saying 'let's start' should complete discovery."""
        response, is_complete, wants_avatar = await bootstrap_service.process_discovery_message(
            "let's start working"
        )
        assert is_complete is True


class TestSoulGeneration:
    """Tests for SOUL.md generation."""

    @pytest.mark.asyncio
    async def test_generate_soul_md_returns_content(self, bootstrap_service, mock_llm):
        """SOUL.md generation should return content."""
        # Set up some discovery history
        await bootstrap_service._save_discovery_history([
            {"role": "user", "content": "Hi, I'm Alice. I prefer casual communication."},
            {"role": "assistant", "content": "Nice to meet you Alice!"},
        ])

        # Mock LLM to return SOUL.md content
        mock_llm.responses = ["# SOUL.md - You Are TestAgent\n\n## Who You Are\nYou are TestAgent..."]

        soul_content = await bootstrap_service.generate_soul_md()
        assert "SOUL.md" in soul_content or "TestAgent" in soul_content

    @pytest.mark.asyncio
    async def test_generate_soul_md_uses_default_when_no_history(self, bootstrap_service):
        """SOUL.md generation should use default when no history."""
        soul_content = await bootstrap_service.generate_soul_md()
        # Should contain default template content
        assert "SOUL.md" in soul_content
        # Default template may or may not include agent name
        assert "Kestrel" in soul_content or "Default" in soul_content

    @pytest.mark.asyncio
    async def test_save_soul_md_creates_file(self, bootstrap_service, temp_agent_dir):
        """Saving SOUL.md should create the file."""
        content = "# SOUL.md - Test Content"
        result = await bootstrap_service.save_soul_md(content)
        assert result is True

        soul_path = temp_agent_dir / "SOUL.md"
        assert soul_path.exists()
        assert soul_path.read_text() == content


class TestSkipAndRestart:
    """Tests for skip and restart functionality."""

    @pytest.mark.asyncio
    async def test_skip_discovery_creates_default_soul(self, bootstrap_service, temp_agent_dir):
        """Skipping discovery should create default SOUL.md."""
        result = await bootstrap_service.skip_discovery()

        soul_path = temp_agent_dir / "SOUL.md"
        assert soul_path.exists()
        assert "default" in result.lower() or "personality" in result.lower()

    @pytest.mark.asyncio
    async def test_skip_discovery_marks_complete(self, bootstrap_service):
        """Skipping discovery should mark bootstrap as complete."""
        await bootstrap_service.skip_discovery()
        state = await bootstrap_service.get_bootstrap_state()
        assert state == BootstrapState.COMPLETE

    @pytest.mark.asyncio
    async def test_restart_discovery_clears_history(self, bootstrap_service):
        """Restarting discovery should clear history."""
        # Add some history
        await bootstrap_service._save_discovery_history([
            {"role": "user", "content": "Test message"},
        ])

        await bootstrap_service.restart_discovery()

        history = await bootstrap_service.get_discovery_history()
        assert len(history) == 0

    @pytest.mark.asyncio
    async def test_restart_discovery_resets_state(self, bootstrap_service):
        """Restarting discovery should reset state to PENDING."""
        await bootstrap_service.set_bootstrap_state(BootstrapState.COMPLETE)
        await bootstrap_service.restart_discovery()
        state = await bootstrap_service.get_bootstrap_state()
        assert state == BootstrapState.PENDING

    @pytest.mark.asyncio
    async def test_restart_discovery_deletes_soul(self, bootstrap_service, temp_agent_dir):
        """Restarting discovery should delete SOUL.md."""
        # Create a SOUL.md first
        soul_path = temp_agent_dir / "SOUL.md"
        soul_path.write_text("# Test SOUL")

        await bootstrap_service.restart_discovery()

        assert not soul_path.exists()


class TestBootstrapStatus:
    """Tests for bootstrap status reporting."""

    @pytest.mark.asyncio
    async def test_status_shows_state(self, bootstrap_service):
        """Status should show current state."""
        status = await bootstrap_service.get_bootstrap_status()
        assert "pending" in status.lower()

    @pytest.mark.asyncio
    async def test_status_shows_exchange_count(self, bootstrap_service):
        """Status should show discovery exchange count."""
        await bootstrap_service._save_discovery_history([
            {"role": "user", "content": "Message 1"},
            {"role": "assistant", "content": "Response 1"},
            {"role": "user", "content": "Message 2"},
            {"role": "assistant", "content": "Response 2"},
        ])

        status = await bootstrap_service.get_bootstrap_status()
        assert "2" in status  # 2 user exchanges

    @pytest.mark.asyncio
    async def test_status_shows_soul_existence(self, bootstrap_service, temp_agent_dir):
        """Status should show whether SOUL.md exists."""
        # No SOUL.md
        status = await bootstrap_service.get_bootstrap_status()
        assert "No" in status

        # Create SOUL.md
        soul_path = temp_agent_dir / "SOUL.md"
        soul_path.write_text("# Test")

        status = await bootstrap_service.get_bootstrap_status()
        assert "Yes" in status


class TestCompleteBootstrap:
    """Tests for bootstrap completion."""

    @pytest.mark.asyncio
    async def test_complete_bootstrap_generates_soul(self, bootstrap_service, temp_agent_dir, mock_llm):
        """Completing bootstrap should generate SOUL.md."""
        mock_llm.responses = ["# SOUL.md - You Are TestAgent\n\nGenerated content"]

        await bootstrap_service.complete_bootstrap()

        soul_path = temp_agent_dir / "SOUL.md"
        assert soul_path.exists()

    @pytest.mark.asyncio
    async def test_complete_bootstrap_returns_greeting(self, bootstrap_service, mock_llm):
        """Completing bootstrap should return a greeting."""
        mock_llm.responses = ["# SOUL.md content"]

        result = await bootstrap_service.complete_bootstrap()
        assert "Nice to meet you" in result or "ready" in result.lower()

    @pytest.mark.asyncio
    async def test_complete_bootstrap_with_avatar_description(self, bootstrap_service, mock_llm):
        """Completing with avatar description should mention avatar."""
        mock_llm.responses = ["# SOUL.md content"]

        result = await bootstrap_service.complete_bootstrap(avatar_description="a friendly owl")
        assert "avatar" in result.lower()
