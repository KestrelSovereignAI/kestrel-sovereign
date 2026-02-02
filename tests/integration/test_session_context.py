"""
Tests for session-based conversation context loading.

When a user selects a conversation from history, the agent should
load that session's context (not the most recent messages globally).
"""
import pytest
import asyncio
from datetime import datetime, timedelta

from kestrel_sovereign.storage.async_storage import AsyncStorage


@pytest.fixture
async def storage_with_sessions(tmp_path):
    """Create a storage instance with multiple conversation sessions."""
    db_path = str(tmp_path / "test_sessions.db")
    storage = AsyncStorage(db_path=db_path, agent_id="test-agent")
    await storage.initialize()

    # Create Session 1: Messages from 2 hours ago
    session1_time = datetime.now() - timedelta(hours=2)
    await storage.db.execute_commit(
        "INSERT INTO conversation_history (agent_id, role, content, metadata, created_at) VALUES (?, ?, ?, ?, ?)",
        ("test-agent", "user", "Hello from session 1", None, session1_time.strftime('%Y-%m-%d %H:%M:%S'))
    )
    # Get the ID of first message - this is session 1's ID
    rows = await storage.db.fetchall(
        "SELECT id FROM conversation_history WHERE agent_id = ? ORDER BY id ASC LIMIT 1",
        ("test-agent",)
    )
    session1_id = str(rows[0][0])

    await storage.db.execute_commit(
        "INSERT INTO conversation_history (agent_id, role, content, metadata, created_at) VALUES (?, ?, ?, ?, ?)",
        ("test-agent", "assistant", "Hi there, session 1 response", None,
         (session1_time + timedelta(seconds=5)).strftime('%Y-%m-%d %H:%M:%S'))
    )
    await storage.db.execute_commit(
        "INSERT INTO conversation_history (agent_id, role, content, metadata, created_at) VALUES (?, ?, ?, ?, ?)",
        ("test-agent", "user", "Another message in session 1", None,
         (session1_time + timedelta(minutes=5)).strftime('%Y-%m-%d %H:%M:%S'))
    )

    # Create Session 2: Messages from 50 minutes ago
    # (gap from session 1 end at ~115 min ago to session 2 start at 50 min ago = 65 min gap)
    session2_time = datetime.now() - timedelta(minutes=50)
    await storage.db.execute_commit(
        "INSERT INTO conversation_history (agent_id, role, content, metadata, created_at) VALUES (?, ?, ?, ?, ?)",
        ("test-agent", "user", "Hello from session 2", None, session2_time.strftime('%Y-%m-%d %H:%M:%S'))
    )
    # Get the ID of session 2's first message
    rows = await storage.db.fetchall(
        "SELECT id FROM conversation_history WHERE agent_id = ? AND content = ?",
        ("test-agent", "Hello from session 2")
    )
    session2_id = str(rows[0][0])

    await storage.db.execute_commit(
        "INSERT INTO conversation_history (agent_id, role, content, metadata, created_at) VALUES (?, ?, ?, ?, ?)",
        ("test-agent", "assistant", "Hello session 2 response", None,
         (session2_time + timedelta(seconds=5)).strftime('%Y-%m-%d %H:%M:%S'))
    )

    # Create Session 3: Recent messages (current session)
    # (gap from session 2 end at ~50 min ago to session 3 at 5 min ago = 45 min gap)
    session3_time = datetime.now() - timedelta(minutes=5)
    await storage.db.execute_commit(
        "INSERT INTO conversation_history (agent_id, role, content, metadata, created_at) VALUES (?, ?, ?, ?, ?)",
        ("test-agent", "user", "Hello from session 3 (most recent)", None, session3_time.strftime('%Y-%m-%d %H:%M:%S'))
    )
    rows = await storage.db.fetchall(
        "SELECT id FROM conversation_history WHERE agent_id = ? AND content LIKE '%session 3%'",
        ("test-agent",)
    )
    session3_id = str(rows[0][0])

    await storage.db.execute_commit(
        "INSERT INTO conversation_history (agent_id, role, content, metadata, created_at) VALUES (?, ?, ?, ?, ?)",
        ("test-agent", "assistant", "Session 3 response", None,
         (session3_time + timedelta(seconds=5)).strftime('%Y-%m-%d %H:%M:%S'))
    )

    yield storage, session1_id, session2_id, session3_id

    await storage.close()


@pytest.mark.asyncio
async def test_get_history_without_session_returns_recent(storage_with_sessions):
    """Without session_id, get_conversation_history returns most recent messages."""
    storage, session1_id, session2_id, session3_id = storage_with_sessions

    # Get history without session_id
    history = await storage.get_conversation_history(limit=10)

    # Should include messages from multiple sessions (most recent first after reversal)
    assert len(history) >= 2
    # Most recent messages should be from session 3
    contents = [m['content'] for m in history]
    assert any('session 3' in c for c in contents)


@pytest.mark.asyncio
async def test_get_history_with_session_id_filters_to_session(storage_with_sessions):
    """With session_id, get_conversation_history returns only that session's messages."""
    storage, session1_id, session2_id, session3_id = storage_with_sessions

    # Get history for session 1
    history = await storage.get_conversation_history(limit=50, session_id=session1_id)

    # Should only contain session 1 messages
    contents = [m['content'] for m in history]
    assert all('session 1' in c.lower() for c in contents), f"Expected session 1 messages only, got: {contents}"
    assert len(history) == 3  # 2 user messages + 1 assistant message in session 1


@pytest.mark.asyncio
async def test_get_history_session_2(storage_with_sessions):
    """Verify session 2 can be loaded independently."""
    storage, session1_id, session2_id, session3_id = storage_with_sessions

    # Get history for session 2
    history = await storage.get_conversation_history(limit=50, session_id=session2_id)

    contents = [m['content'] for m in history]
    assert all('session 2' in c.lower() for c in contents), f"Expected session 2 messages only, got: {contents}"
    assert len(history) == 2  # 1 user + 1 assistant in session 2


@pytest.mark.asyncio
async def test_get_history_session_3(storage_with_sessions):
    """Verify most recent session can be loaded by ID."""
    storage, session1_id, session2_id, session3_id = storage_with_sessions

    # Get history for session 3
    history = await storage.get_conversation_history(limit=50, session_id=session3_id)

    contents = [m['content'] for m in history]
    assert all('session 3' in c.lower() for c in contents), f"Expected session 3 messages only, got: {contents}"
    assert len(history) == 2


@pytest.mark.asyncio
async def test_nonexistent_session_returns_empty(storage_with_sessions):
    """A non-existent session_id should return empty history."""
    storage, session1_id, session2_id, session3_id = storage_with_sessions

    history = await storage.get_conversation_history(limit=50, session_id="99999")

    assert history == []


@pytest.mark.asyncio
async def test_privacy_feature_passes_session_id(tmp_path):
    """Verify PrivacyAgent passes session_id through to storage."""
    from kestrel_sovereign.features.privacy.feature import PrivacyAgent
    from kestrel_sovereign.privacy import PrivacyMode

    db_path = str(tmp_path / "privacy_test.db")
    storage = AsyncStorage(db_path=db_path, agent_id="privacy-test")
    await storage.initialize()

    # Add some messages
    await storage.add_conversation("user", "Test message 1")
    rows = await storage.db.fetchall(
        "SELECT id FROM conversation_history ORDER BY id ASC LIMIT 1", ()
    )
    session_id = str(rows[0][0])
    await storage.add_conversation("assistant", "Response 1")

    # Create PrivacyAgent
    privacy_agent = PrivacyAgent(storage, PrivacyMode.NORMAL)

    # Get history with session_id
    history = await privacy_agent.get_conversation_history(limit=10, session_id=session_id)

    assert len(history) == 2
    await storage.close()
