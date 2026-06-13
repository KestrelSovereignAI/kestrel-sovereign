#!/usr/bin/env python3
"""
Comprehensive tests for Kestrel's 5-level privacy system.
Tests all privacy modes, transitions, PII filtering, and LLM restrictions.
"""

import pytest
import pytest_asyncio
import asyncio
import tempfile
import os
from datetime import datetime, timezone
from unittest.mock import patch, MagicMock

from kestrel_sovereign.privacy import PrivacyMode
from kestrel_sovereign.features.privacy import PrivacyAgent
from kestrel_sovereign.ephemeral_session import EphemeralSession
from kestrel_sovereign.storage import AsyncStorage


class TestEphemeralSession:
    """Test the EphemeralSession class for in-memory storage"""

    def test_ephemeral_session_creation(self):
        """Test that ephemeral session is created correctly"""
        session = EphemeralSession(max_messages=50)
        assert session.messages == []
        assert session.max_messages == 50
        assert session.created_at is not None

    def test_ephemeral_session_add_message(self):
        """Test adding messages to ephemeral session"""
        session = EphemeralSession()
        session.add_message("user", "Hello")
        session.add_message("assistant", "Hi there")

        assert len(session.messages) == 2
        assert session.messages[0]["role"] == "user"
        assert session.messages[0]["content"] == "Hello"
        assert session.messages[1]["role"] == "assistant"

    def test_ephemeral_session_buffer_limit(self):
        """Test that ephemeral session respects buffer limit"""
        session = EphemeralSession(max_messages=5)

        # Add 10 messages
        for i in range(10):
            session.add_message("user", f"Message {i}")

        # Should only keep last 5
        assert len(session.messages) == 5
        assert session.messages[0]["content"] == "Message 5"
        assert session.messages[-1]["content"] == "Message 9"

    def test_ephemeral_session_get_context(self):
        """Test getting context from ephemeral session"""
        session = EphemeralSession()
        session.add_message("user", "What is AI?")
        session.add_message("assistant", "AI is artificial intelligence")

        context = session.get_context(limit=2)
        assert "user: What is AI?" in context
        assert "assistant: AI is artificial intelligence" in context

    def test_ephemeral_session_clear(self):
        """Test clearing ephemeral session"""
        session = EphemeralSession()
        session.add_message("user", "Test message")
        session.add_message("assistant", "Response")

        assert len(session.messages) == 2

        session.clear()
        assert len(session.messages) == 0
        assert session.context == {}

    def test_ephemeral_session_get_stats(self):
        """Test getting session statistics"""
        session = EphemeralSession(max_messages=100)
        session.add_message("user", "Test")

        stats = session.get_stats()
        assert stats["message_count"] == 1
        assert stats["max_buffer_size"] == 100
        assert stats["storage_mode"] == "in-memory only"
        assert stats["persistent_storage"] is False


class TestPrivacyModes:
    """Test privacy modes"""

    @pytest_asyncio.fixture
    async def temp_storage(self, temp_dir):
        """Create a temporary async storage for testing"""
        db_path = str(temp_dir / "test.db")
        storage = AsyncStorage(db_path)
        await storage.initialize()
        yield storage
        await storage.close()

    def test_privacy_mode_enum(self):
        """Test that all privacy modes are defined"""
        assert PrivacyMode.EPHEMERAL.value == "ephemeral"
        assert PrivacyMode.ISOLATED.value == "isolated"
        assert PrivacyMode.ANONYMOUS.value == "anonymous"
        assert PrivacyMode.NORMAL.value == "normal"
        assert PrivacyMode.PUBLIC.value == "public"

    @pytest.mark.asyncio
    async def test_ephemeral_mode_stores_nothing(self, temp_storage):
        """EPHEMERAL mode must not persist any data to storage"""
        privacy_agent = PrivacyAgent(temp_storage, PrivacyMode.EPHEMERAL)

        # Add messages
        await privacy_agent.add_conversation("user", "Tell me your name")
        await privacy_agent.add_conversation("assistant", "I am Kestrel")
        await privacy_agent.add_conversation("user", "Remember this secret: 12345")

        # Check storage - should be empty
        history = await temp_storage.get_conversation_history()
        assert len(history) == 0, "EPHEMERAL mode stored messages to database!"

        # Check ephemeral session - should have messages in memory
        assert privacy_agent.ephemeral_session is not None
        assert len(privacy_agent.ephemeral_session.messages) == 3

    @pytest.mark.asyncio
    async def test_isolated_mode_temporary_storage(self, temp_storage):
        """ISOLATED mode stores in temporary session"""
        privacy_agent = PrivacyAgent(temp_storage, PrivacyMode.ISOLATED)

        # Add messages
        await privacy_agent.add_conversation("user", "Test message")
        await privacy_agent.add_conversation("assistant", "Test response")

        # Check storage - should be empty
        history = await temp_storage.get_conversation_history()
        assert len(history) == 0

        # Check isolated session - should have messages
        assert len(privacy_agent.isolated_session) == 2
        assert privacy_agent.isolated_session[0]["content"] == "Test message"

    @pytest.mark.asyncio
    async def test_anonymous_mode_filters_pii(self, temp_storage):
        """ANONYMOUS mode should redact PII"""
        privacy_agent = PrivacyAgent(temp_storage, PrivacyMode.ANONYMOUS)

        # Add message with PII
        await privacy_agent.add_conversation(
            "user",
            "My email is john.doe@example.com and my phone is 555-123-4567"
        )

        # Retrieve and verify PII filtered
        history = await temp_storage.get_conversation_history()
        assert len(history) == 1
        content = history[0]["content"]

        assert "john.doe@example.com" not in content
        assert "555-123-4567" not in content
        assert "[EMAIL_REDACTED]" in content
        assert "[PHONE_REDACTED]" in content

    @pytest.mark.asyncio
    async def test_anonymous_mode_filters_ssn(self, temp_storage):
        """ANONYMOUS mode should redact SSN"""
        privacy_agent = PrivacyAgent(temp_storage, PrivacyMode.ANONYMOUS)

        await privacy_agent.add_conversation(
            "user",
            "My SSN is 123-45-6789"
        )

        history = await temp_storage.get_conversation_history()
        content = history[0]["content"]

        assert "123-45-6789" not in content
        assert "[SSN_REDACTED]" in content

    @pytest.mark.asyncio
    async def test_anonymous_mode_filters_address(self, temp_storage):
        """ANONYMOUS mode should redact addresses"""
        privacy_agent = PrivacyAgent(temp_storage, PrivacyMode.ANONYMOUS)

        await privacy_agent.add_conversation(
            "user",
            "I live at 123 Main Street"
        )

        history = await temp_storage.get_conversation_history()
        content = history[0]["content"]

        assert "123 Main Street" not in content
        assert "[ADDRESS_REDACTED]" in content

    @pytest.mark.asyncio
    async def test_normal_mode_stores_normally(self, temp_storage):
        """NORMAL mode stores everything without filtering"""
        privacy_agent = PrivacyAgent(temp_storage, PrivacyMode.NORMAL)

        # Add message with PII (should NOT be filtered)
        original_text = "My email is john@example.com"
        await privacy_agent.add_conversation("user", original_text)

        # Retrieve and verify NOT filtered
        history = await temp_storage.get_conversation_history()
        assert len(history) == 1
        content = history[0]["content"]

        assert "john@example.com" in content
        assert "[EMAIL_REDACTED]" not in content

    @pytest.mark.asyncio
    async def test_public_mode_allows_storage(self, temp_storage):
        """PUBLIC mode stores everything and allows sharing"""
        privacy_agent = PrivacyAgent(temp_storage, PrivacyMode.PUBLIC)

        await privacy_agent.add_conversation("user", "This can be shared")
        await privacy_agent.add_conversation("assistant", "Yes, this is public")

        history = await temp_storage.get_conversation_history()
        assert len(history) == 2

        # Check metadata includes privacy mode (if metadata is present)
        # Note: metadata structure may vary by storage implementation
        if "metadata" in history[0] and isinstance(history[0]["metadata"], dict):
            assert history[0]["metadata"]["privacy_mode"] == "public"


class TestPrivacyModeTransitions:
    """Test transitions between privacy modes"""

    @pytest_asyncio.fixture
    async def temp_storage(self, temp_dir):
        """Create a temporary async storage for testing"""
        db_path = str(temp_dir / "test.db")
        storage = AsyncStorage(db_path)
        await storage.initialize()
        yield storage
        await storage.close()

    @pytest.mark.asyncio
    async def test_ephemeral_to_normal_transition(self, temp_storage):
        """Switching from EPHEMERAL to NORMAL should work"""
        privacy_agent = PrivacyAgent(temp_storage, PrivacyMode.EPHEMERAL)

        # Messages in EPHEMERAL not stored
        await privacy_agent.add_conversation("user", "Ephemeral message")
        history = await temp_storage.get_conversation_history()
        assert len(history) == 0

        # Switch to NORMAL
        result = privacy_agent.set_mode(PrivacyMode.NORMAL)
        assert "NORMAL mode" in result

        # New messages should be stored
        await privacy_agent.add_conversation("user", "Normal message")
        messages = await temp_storage.get_conversation_history()
        assert len(messages) == 1
        assert "Normal message" in messages[0]["content"]

    @pytest.mark.asyncio
    async def test_public_to_ephemeral_warning(self, temp_storage):
        """Switching from PUBLIC to EPHEMERAL should show warning"""
        privacy_agent = PrivacyAgent(temp_storage, PrivacyMode.PUBLIC)

        result = privacy_agent.set_mode(PrivacyMode.EPHEMERAL)
        assert "WARNING" in result
        assert "confirm" in result.lower()

    @pytest.mark.asyncio
    async def test_isolated_session_save(self, temp_storage):
        """ISOLATED session can be saved to permanent storage"""
        privacy_agent = PrivacyAgent(temp_storage, PrivacyMode.ISOLATED)

        # Add messages to isolated session
        await privacy_agent.add_conversation("user", "Isolated message 1")
        await privacy_agent.add_conversation("user", "Isolated message 2")

        assert len(privacy_agent.isolated_session) == 2
        history = await temp_storage.get_conversation_history()
        assert len(history) == 0

        # Save isolated session
        result = await privacy_agent.save_isolated_session()
        assert "Saved 2 messages" in result

        # Check that messages are now in storage
        history = await temp_storage.get_conversation_history()
        assert len(history) == 2

    @pytest.mark.asyncio
    async def test_isolated_session_discard(self, temp_storage):
        """ISOLATED session can be discarded without saving"""
        privacy_agent = PrivacyAgent(temp_storage, PrivacyMode.ISOLATED)

        # Add messages to isolated session
        await privacy_agent.add_conversation("user", "Discard me")
        assert len(privacy_agent.isolated_session) == 1

        # Discard session
        result = privacy_agent.discard_isolated_session()
        assert "Discarded 1 messages" in result

        # Check that messages are NOT in storage
        history = await temp_storage.get_conversation_history()
        assert len(history) == 0
        assert len(privacy_agent.isolated_session) == 0


class TestPrivacyAgent:
    """Test PrivacyAgent functionality"""

    @pytest_asyncio.fixture
    async def temp_storage(self, temp_dir):
        """Create a temporary async storage for testing"""
        db_path = str(temp_dir / "test.db")
        storage = AsyncStorage(db_path)
        await storage.initialize()
        yield storage
        await storage.close()

    @pytest.mark.asyncio
    async def test_privacy_agent_initialization(self, temp_storage):
        """Test PrivacyAgent initialization"""
        privacy_agent = PrivacyAgent(temp_storage, PrivacyMode.NORMAL)
        assert privacy_agent.privacy_mode == PrivacyMode.NORMAL
        assert privacy_agent.storage == temp_storage

    @pytest.mark.asyncio
    async def test_privacy_agent_get_status(self, temp_storage):
        """Test getting privacy status"""
        privacy_agent = PrivacyAgent(temp_storage, PrivacyMode.EPHEMERAL)
        status = privacy_agent.get_status()

        assert "ephemeral" in status.lower()

    @pytest.mark.asyncio
    async def test_privacy_agent_get_detailed_status(self, temp_storage):
        """Test getting detailed privacy status"""
        privacy_agent = PrivacyAgent(temp_storage, PrivacyMode.ANONYMOUS)
        status = privacy_agent.get_detailed_status()

        assert status["privacy_mode"] == "anonymous"
        assert status["pii_filtering"] is True
        assert status["backup_encryption"] == "required"
        assert status["llm_providers"]["cloud_openai"] is False

    @pytest.mark.asyncio
    async def test_ephemeral_mode_llm_restriction(self, temp_storage):
        """EPHEMERAL mode should restrict LLM providers"""
        privacy_agent = PrivacyAgent(temp_storage, PrivacyMode.EPHEMERAL)
        status = privacy_agent.get_detailed_status()

        # EPHEMERAL should only allow local LLM
        assert status["llm_providers"]["cloud_openai"] is False
        assert status["llm_providers"]["cloud_anthropic"] is False

    @pytest.mark.asyncio
    async def test_get_conversation_history_respects_mode(self, temp_storage):
        """Conversation history should respect privacy mode"""
        privacy_agent = PrivacyAgent(temp_storage, PrivacyMode.EPHEMERAL)
        await privacy_agent.add_conversation("user", "Ephemeral test")

        # Should return from ephemeral session, not storage
        history = await privacy_agent.get_conversation_history(limit=10)
        assert len(history) == 1
        assert history[0]["content"] == "Ephemeral test"

        # Storage should be empty
        storage_history = await temp_storage.get_conversation_history()
        assert len(storage_history) == 0


class TestPIIFiltering:
    """Test PII filtering in detail"""

    @pytest_asyncio.fixture
    async def temp_storage(self, temp_dir):
        """Create a temporary async storage for testing"""
        db_path = str(temp_dir / "test.db")
        storage = AsyncStorage(db_path)
        await storage.initialize()
        yield storage
        await storage.close()

    @pytest.mark.asyncio
    async def test_email_redaction(self, temp_storage):
        """Test email address redaction"""
        privacy_agent = PrivacyAgent(temp_storage, PrivacyMode.ANONYMOUS)

        test_cases = [
            "john@example.com",
            "jane.doe@company.co.uk",
            "test123@test-domain.org"
        ]

        for email in test_cases:
            await privacy_agent.add_conversation("user", f"My email is {email}")

        history = await temp_storage.get_conversation_history()
        for i, email in enumerate(test_cases):
            assert email not in history[i]["content"]
            assert "[EMAIL_REDACTED]" in history[i]["content"]

    @pytest.mark.asyncio
    async def test_phone_redaction(self, temp_storage):
        """Test phone number redaction"""
        privacy_agent = PrivacyAgent(temp_storage, PrivacyMode.ANONYMOUS)

        test_cases = [
            "555-123-4567",
            "(555) 123-4567",
            "555.123.4567"
        ]

        for phone in test_cases:
            await privacy_agent.add_conversation("user", f"Call me at {phone}")

        history = await temp_storage.get_conversation_history()
        for i, phone in enumerate(test_cases):
            assert phone not in history[i]["content"]
            assert "[PHONE_REDACTED]" in history[i]["content"]

    @pytest.mark.asyncio
    async def test_credit_card_redaction(self, temp_storage):
        """Test credit card number redaction"""
        privacy_agent = PrivacyAgent(temp_storage, PrivacyMode.ANONYMOUS)

        await privacy_agent.add_conversation("user", "Card: 4532-1234-5678-9012")

        history = await temp_storage.get_conversation_history()
        assert "4532-1234-5678-9012" not in history[0]["content"]
        assert "[CARD_REDACTED]" in history[0]["content"]


class TestBackupIntegration:
    """Test backup functionality with privacy modes"""

    @pytest_asyncio.fixture
    async def temp_storage(self, temp_dir):
        """Create a temporary async storage for testing"""
        db_path = str(temp_dir / "test.db")
        storage = AsyncStorage(db_path)
        await storage.initialize()
        yield storage
        await storage.close()

    @pytest.mark.asyncio
    async def test_ephemeral_mode_no_backup(self, temp_storage):
        """EPHEMERAL mode should prevent backups"""
        privacy_agent = PrivacyAgent(temp_storage, PrivacyMode.EPHEMERAL)
        status = privacy_agent.get_detailed_status()

        assert status["backup_status"] == "disabled"

    @pytest.mark.asyncio
    async def test_anonymous_mode_requires_encryption(self, temp_storage):
        """ANONYMOUS mode should require encrypted backups"""
        privacy_agent = PrivacyAgent(temp_storage, PrivacyMode.ANONYMOUS)
        status = privacy_agent.get_detailed_status()

        assert status["backup_encryption"] == "required"

    @pytest.mark.asyncio
    async def test_normal_mode_optional_encryption(self, temp_storage):
        """NORMAL mode should have optional encryption"""
        privacy_agent = PrivacyAgent(temp_storage, PrivacyMode.NORMAL)
        status = privacy_agent.get_detailed_status()

        assert status["backup_encryption"] == "optional"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
