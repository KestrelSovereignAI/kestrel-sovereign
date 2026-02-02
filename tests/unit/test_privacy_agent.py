import pytest
import os
from kestrel_sovereign.features.privacy import PrivacyAgent
from kestrel_sovereign.storage import AsyncStorage
from kestrel_sovereign.privacy import PrivacyMode


def test_initial_mode_sync():
    """Tests that the initial privacy mode is NORMAL (sync test with mock)."""
    from unittest.mock import Mock
    storage = Mock()
    agent = PrivacyAgent(storage)
    assert agent.privacy_mode == PrivacyMode.NORMAL


def test_set_mode_sync():
    """Tests setting the privacy mode (sync test with mock)."""
    from unittest.mock import Mock
    storage = Mock()
    agent = PrivacyAgent(storage)
    response = agent.set_mode(PrivacyMode.ISOLATED)
    assert agent.privacy_mode == PrivacyMode.ISOLATED
    assert "Privacy mode changed" in response


@pytest.mark.asyncio
async def test_isolated_mode_flow(tmp_path):
    """Tests the full flow of using ISOLATED mode."""
    db_path = tmp_path / "test_isolated.db"
    storage = AsyncStorage(str(db_path))
    await storage.initialize()
    privacy_agent = PrivacyAgent(storage)

    privacy_agent.set_mode(PrivacyMode.ISOLATED)

    # Add messages
    await privacy_agent.add_conversation("user", "This is a secret.")
    await privacy_agent.add_conversation("assistant", "I will keep it safe.")

    # Check that messages are in isolated session
    assert len(privacy_agent.isolated_session) == 2

    # get_conversation_history() should return isolated session contents in ISOLATED mode
    history = await privacy_agent.get_conversation_history()
    assert len(history) == 2

    # Verify underlying storage is still empty (nothing persisted yet)
    permanent_history = await storage.get_conversation_history()
    assert len(permanent_history) == 0

    # Save the session to permanent storage
    await privacy_agent.save_isolated_session()

    # Check that isolated session is cleared
    assert len(privacy_agent.isolated_session) == 0

    # Now permanent storage should have the messages
    permanent_history = await storage.get_conversation_history()
    assert len(permanent_history) == 2

    await storage.close()


@pytest.mark.asyncio
async def test_ephemeral_mode(tmp_path):
    """Tests that EPHEMERAL mode does not persist messages to storage."""
    db_path = tmp_path / "test_ephemeral.db"
    storage = AsyncStorage(str(db_path))
    await storage.initialize()
    privacy_agent = PrivacyAgent(storage)

    privacy_agent.set_mode(PrivacyMode.EPHEMERAL)
    await privacy_agent.add_conversation("user", "This should not be stored.")

    # get_conversation_history returns ephemeral session in EPHEMERAL mode
    history = await privacy_agent.get_conversation_history()
    assert len(history) == 1  # Message is in ephemeral session

    # But underlying storage should be empty (nothing persisted)
    permanent_history = await storage.get_conversation_history()
    assert len(permanent_history) == 0

    await storage.close()


@pytest.mark.asyncio
async def test_anonymous_mode(tmp_path):
    """Tests that ANONYMOUS mode redacts PII."""
    db_path = tmp_path / "test_anonymous.db"
    storage = AsyncStorage(str(db_path))
    await storage.initialize()
    privacy_agent = PrivacyAgent(storage)

    privacy_agent.set_mode(PrivacyMode.ANONYMOUS)
    pii_message = "My email is test@example.com and my phone is 555-123-4567."
    await privacy_agent.add_conversation("user", pii_message)

    history = await privacy_agent.get_conversation_history()
    assert len(history) == 1

    stored_message = history[0]['content']
    assert "test@example.com" not in stored_message
    assert "555-123-4567" not in stored_message
    assert "[EMAIL_REDACTED]" in stored_message
    assert "[PHONE_REDACTED]" in stored_message

    await storage.close() 