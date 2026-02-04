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


@pytest.mark.asyncio
async def test_resumed_conversation_includes_messages_with_session_id_metadata(tmp_path):
    """
    Test that resuming an old conversation includes messages sent after a time gap.
    
    Scenario:
    1. Session 1 starts with messages at t=0
    2. Time gap of 2 hours (exceeds 30-min threshold)
    3. User loads session 1 and sends new message (with session_id in metadata)
    4. Loading session 1 again should include BOTH original AND new messages
    
    This tests the "resumed conversation" feature where session_id in metadata
    bridges the time gap that would normally split sessions.
    """
    import json
    
    db_path = str(tmp_path / "resumed_test.db")
    storage = AsyncStorage(db_path=db_path, agent_id="resume-test")
    await storage.initialize()
    
    # Original session 1 messages (2 hours ago)
    session1_time = datetime.now() - timedelta(hours=2)
    await storage.db.execute_commit(
        "INSERT INTO conversation_history (agent_id, role, content, metadata, created_at) VALUES (?, ?, ?, ?, ?)",
        ("resume-test", "user", "Original message in session 1", None, 
         session1_time.strftime('%Y-%m-%d %H:%M:%S'))
    )
    
    # Get session 1 ID
    rows = await storage.db.fetchall(
        "SELECT id FROM conversation_history WHERE agent_id = ? ORDER BY id ASC LIMIT 1",
        ("resume-test",)
    )
    session1_id = str(rows[0][0])
    
    await storage.db.execute_commit(
        "INSERT INTO conversation_history (agent_id, role, content, metadata, created_at) VALUES (?, ?, ?, ?, ?)",
        ("resume-test", "assistant", "Original response in session 1", None,
         (session1_time + timedelta(seconds=5)).strftime('%Y-%m-%d %H:%M:%S'))
    )
    
    # --- TIME GAP OF 2 HOURS ---
    
    # New message sent NOW while viewing session 1 (has session_id in metadata)
    resumed_time = datetime.now() - timedelta(minutes=1)
    resumed_metadata = json.dumps({"session_id": session1_id})
    await storage.db.execute_commit(
        "INSERT INTO conversation_history (agent_id, role, content, metadata, created_at) VALUES (?, ?, ?, ?, ?)",
        ("resume-test", "user", "Resumed message - should be included", resumed_metadata,
         resumed_time.strftime('%Y-%m-%d %H:%M:%S'))
    )
    await storage.db.execute_commit(
        "INSERT INTO conversation_history (agent_id, role, content, metadata, created_at) VALUES (?, ?, ?, ?, ?)",
        ("resume-test", "assistant", "Resumed response - should be included", resumed_metadata,
         (resumed_time + timedelta(seconds=5)).strftime('%Y-%m-%d %H:%M:%S'))
    )
    
    # Load session 1 - should include ALL 4 messages (2 original + 2 resumed)
    history = await storage.get_conversation_history(limit=50, session_id=session1_id)
    
    contents = [m['content'] for m in history]
    assert len(history) == 4, f"Expected 4 messages (2 original + 2 resumed), got {len(history)}: {contents}"
    assert "Original message" in contents[0], "First message should be original"
    assert "Resumed message" in contents[2] or "Resumed message" in contents[3], \
        f"Resumed messages should be included: {contents}"
    
    await storage.close()


@pytest.mark.asyncio
async def test_messages_without_session_id_metadata_are_excluded_after_gap(tmp_path):
    """
    Test that messages WITHOUT session_id in metadata are excluded after a time gap.
    
    This is the inverse of the above test - messages sent in a different context
    (not while viewing session 1) should NOT appear when loading session 1.
    """
    db_path = str(tmp_path / "excluded_test.db")
    storage = AsyncStorage(db_path=db_path, agent_id="exclude-test")
    await storage.initialize()
    
    # Session 1 messages (2 hours ago)
    session1_time = datetime.now() - timedelta(hours=2)
    await storage.db.execute_commit(
        "INSERT INTO conversation_history (agent_id, role, content, metadata, created_at) VALUES (?, ?, ?, ?, ?)",
        ("exclude-test", "user", "Session 1 message", None, 
         session1_time.strftime('%Y-%m-%d %H:%M:%S'))
    )
    
    rows = await storage.db.fetchall(
        "SELECT id FROM conversation_history WHERE agent_id = ? ORDER BY id ASC LIMIT 1",
        ("exclude-test",)
    )
    session1_id = str(rows[0][0])
    
    await storage.db.execute_commit(
        "INSERT INTO conversation_history (agent_id, role, content, metadata, created_at) VALUES (?, ?, ?, ?, ?)",
        ("exclude-test", "assistant", "Session 1 response", None,
         (session1_time + timedelta(seconds=5)).strftime('%Y-%m-%d %H:%M:%S'))
    )
    
    # --- TIME GAP ---
    
    # Session 2 messages (recent, NO session_id linking to session 1)
    session2_time = datetime.now() - timedelta(minutes=5)
    await storage.db.execute_commit(
        "INSERT INTO conversation_history (agent_id, role, content, metadata, created_at) VALUES (?, ?, ?, ?, ?)",
        ("exclude-test", "user", "Session 2 message - different context", None,
         session2_time.strftime('%Y-%m-%d %H:%M:%S'))
    )
    
    # Load session 1 - should only include 2 messages, NOT session 2
    history = await storage.get_conversation_history(limit=50, session_id=session1_id)
    
    contents = [m['content'] for m in history]
    assert len(history) == 2, f"Expected 2 messages from session 1 only, got {len(history)}: {contents}"
    assert all("Session 1" in c for c in contents), f"Should only have session 1 messages: {contents}"
    assert not any("Session 2" in c for c in contents), f"Session 2 should be excluded: {contents}"
    
    await storage.close()
