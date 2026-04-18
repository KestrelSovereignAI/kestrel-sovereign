"""
Tests for implicit session_id derivation in AsyncConversationStore.

Background: session_id was a parameter that flowed end-to-end through
the API → process_input → add_conversation → metadata, but no client
ever sent it. Result: zero messages in production had session_id in
metadata despite the entire infrastructure being wired.

Fix: derive an implicit session_id from the time-gap heuristic when
no explicit session_id is provided. Reuses the previous session_id
if within 30 minutes; mints a new UUID otherwise.

These tests verify the wiring is real, not aspirational.
"""

import asyncio
import json
import pytest
import tempfile
from datetime import datetime, timezone, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

from kestrel_sovereign.storage.async_conversation_store import AsyncConversationStore
from kestrel_sovereign.storage.async_database import AsyncDatabase


@pytest.fixture
async def store():
    """Real SQLite-backed conversation store for integration verification."""
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "test.db"
        db = await AsyncDatabase.sqlite(str(db_path))
        store = AsyncConversationStore(db, agent_id="test-agent")
        yield store
        await db.close()


class TestImplicitSessionDerivation:
    """Verify that messages without explicit session_id get implicit ones."""

    @pytest.mark.asyncio
    async def test_first_message_gets_session_id(self, store):
        """The very first message should get a fresh implicit session_id."""
        await store.add_conversation("user", "hello")

        rows = await store.db.fetchall(
            "SELECT metadata FROM conversation_history WHERE agent_id = ?",
            ("test-agent",),
        )
        assert len(rows) == 1
        meta = json.loads(rows[0][0])
        assert "session_id" in meta
        assert len(meta["session_id"]) == 36  # UUID4 length

    @pytest.mark.asyncio
    async def test_consecutive_messages_share_session(self, store):
        """Two messages in quick succession should share an implicit session."""
        await store.add_conversation("user", "first")
        await store.add_conversation("assistant", "response")
        await store.add_conversation("user", "second")

        rows = await store.db.fetchall(
            "SELECT metadata FROM conversation_history WHERE agent_id = ? ORDER BY id",
            ("test-agent",),
        )
        sessions = [json.loads(r[0])["session_id"] for r in rows]
        assert sessions[0] == sessions[1] == sessions[2]

    @pytest.mark.asyncio
    async def test_explicit_session_id_wins(self, store):
        """An explicit session_id should not be overridden by derivation."""
        await store.add_conversation("user", "hello", session_id="my-custom-session")

        rows = await store.db.fetchall(
            "SELECT metadata FROM conversation_history WHERE agent_id = ?",
            ("test-agent",),
        )
        meta = json.loads(rows[0][0])
        assert meta["session_id"] == "my-custom-session"

    @pytest.mark.asyncio
    async def test_explicit_session_followed_by_implicit_continues(self, store):
        """If client sends explicit session_id once, subsequent implicit
        messages within the gap should continue that session."""
        await store.add_conversation("user", "explicit", session_id="explicit-sid")
        await store.add_conversation("assistant", "implicit response")

        rows = await store.db.fetchall(
            "SELECT metadata FROM conversation_history WHERE agent_id = ? ORDER BY id",
            ("test-agent",),
        )
        sessions = [json.loads(r[0])["session_id"] for r in rows]
        assert sessions[0] == sessions[1] == "explicit-sid"


class TestSessionGapBoundary:
    """Verify the 30-minute gap actually creates a new session."""

    @pytest.mark.asyncio
    async def test_gap_over_30_min_starts_new_session(self, store):
        """Manually inject an old message and verify a new one starts fresh."""
        # Manually insert a message timestamped 31 minutes ago
        old_time = (datetime.now(timezone.utc) - timedelta(minutes=31)).isoformat()
        old_meta = {"session_id": "old-session"}
        await store.db.execute_commit(
            "INSERT INTO conversation_history (agent_id, role, content, metadata, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            ("test-agent", "user", "old message", json.dumps(old_meta), old_time),
        )

        # Now add a new message via add_conversation; it should NOT reuse old-session
        await store.add_conversation("user", "new message")

        rows = await store.db.fetchall(
            "SELECT metadata FROM conversation_history WHERE agent_id = ? ORDER BY id",
            ("test-agent",),
        )
        sessions = [json.loads(r[0])["session_id"] for r in rows]
        assert sessions[0] == "old-session"
        assert sessions[1] != "old-session"
        assert len(sessions[1]) == 36  # fresh UUID

    @pytest.mark.asyncio
    async def test_gap_under_30_min_reuses_session(self, store):
        """A gap of 5 minutes should still reuse the previous session."""
        recent_time = (datetime.now(timezone.utc) - timedelta(minutes=5)).isoformat()
        recent_meta = {"session_id": "recent-session"}
        await store.db.execute_commit(
            "INSERT INTO conversation_history (agent_id, role, content, metadata, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            ("test-agent", "user", "recent message", json.dumps(recent_meta), recent_time),
        )

        await store.add_conversation("user", "new message")

        rows = await store.db.fetchall(
            "SELECT metadata FROM conversation_history WHERE agent_id = ? ORDER BY id",
            ("test-agent",),
        )
        sessions = [json.loads(r[0])["session_id"] for r in rows]
        assert sessions[0] == sessions[1] == "recent-session"


class TestSessionIdRetrieval:
    """Verify that get_conversation_history(session_id=...) actually filters
    when implicit sessions are now populated."""

    @pytest.mark.asyncio
    async def test_can_filter_by_implicit_session_id(self, store):
        """Add messages, capture the implicit session_id, verify retrieval works."""
        await store.add_conversation("user", "msg1")
        await store.add_conversation("assistant", "msg2")

        # Read the session_id that was assigned
        rows = await store.db.fetchall(
            "SELECT metadata FROM conversation_history WHERE agent_id = ?",
            ("test-agent",),
        )
        sid = json.loads(rows[0][0])["session_id"]

        # Now query by that session_id — should return both messages
        history = await store.get_conversation_history(session_id=sid)
        assert len(history) == 2
        contents = {h["content"] for h in history}
        assert contents == {"msg1", "msg2"}


class TestErrorIsolation:
    """Implicit session derivation must never break writes."""

    @pytest.mark.asyncio
    async def test_derivation_failure_does_not_block_write(self):
        """Even if the previous-message lookup fails, the message must store."""
        db = MagicMock()
        # First call (lookup) fails; second call (insert) succeeds
        db.fetchone = AsyncMock(side_effect=RuntimeError("db hiccup"))
        db.execute_commit = AsyncMock()

        store = AsyncConversationStore(db, agent_id="test-agent")
        # Should not raise
        await store.add_conversation("user", "hello")

        # Insert was still attempted
        db.execute_commit.assert_called_once()


class TestSessionGapMinutesCentralized:
    """Verify all three subsystems read SESSION_GAP_MINUTES from the
    central SDK constant, not local copies.

    Previous state: 3 files independently defined SESSION_GAP_MINUTES = 30
    (storage/async_conversation_store.py, storage/memory_consolidator.py,
    features/wellness/metrics.py). If any one drifted, sessions would
    fragment differently across subsystems.
    """

    def test_sdk_constant_exists(self):
        from kestrel_sdk.config.constants import SESSION_GAP_MINUTES
        assert SESSION_GAP_MINUTES == 30

    def test_async_conversation_store_uses_sdk_constant(self, monkeypatch):
        """Patch the SDK constant; AsyncConversationStore must follow."""
        import kestrel_sdk.config.constants as sdk_constants
        from kestrel_sovereign.storage.async_conversation_store import AsyncConversationStore
        monkeypatch.setattr(sdk_constants, "SESSION_GAP_MINUTES", 99)

        # Use a minimal stub to access the property
        store = AsyncConversationStore.__new__(AsyncConversationStore)
        assert store._IMPLICIT_SESSION_GAP_MINUTES == 99

    def test_memory_consolidator_uses_sdk_constant(self, monkeypatch):
        import kestrel_sdk.config.constants as sdk_constants
        from kestrel_sovereign.storage.memory_consolidator import MemoryConsolidator
        monkeypatch.setattr(sdk_constants, "SESSION_GAP_MINUTES", 77)

        consolidator = MemoryConsolidator.__new__(MemoryConsolidator)
        assert consolidator.SESSION_GAP_MINUTES == 77

    def test_wellness_metrics_uses_sdk_constant(self, monkeypatch):
        import kestrel_sdk.config.constants as sdk_constants
        from kestrel_sovereign.features.wellness.metrics import SessionContinuityCalculator
        monkeypatch.setattr(sdk_constants, "SESSION_GAP_MINUTES", 55)

        calc = SessionContinuityCalculator.__new__(SessionContinuityCalculator)
        assert calc.SESSION_GAP_MINUTES == 55


class TestSearchHistorySessionScoping:
    """Verify search_history(session_id=...) actually scopes results."""

    @pytest.mark.asyncio
    async def test_search_filters_to_session(self, store):
        """A search with session_id should only return matches from that session."""
        # Add two messages in session A
        await store.add_conversation("user", "blue sky", session_id="session-A")
        await store.add_conversation("assistant", "blue ocean", session_id="session-A")
        # Add two messages in session B
        await store.add_conversation("user", "blue mountain", session_id="session-B")
        await store.add_conversation("assistant", "blue river", session_id="session-B")

        # Search in session A only
        results = await store.search_history("blue", session_id="session-A")
        assert len(results) == 2
        contents = {r["content"] for r in results}
        assert contents == {"blue sky", "blue ocean"}

    @pytest.mark.asyncio
    async def test_search_without_session_id_returns_all(self, store):
        """Backward compatibility: no session_id arg = search all history."""
        await store.add_conversation("user", "alpha", session_id="s1")
        await store.add_conversation("user", "alpha again", session_id="s2")

        results = await store.search_history("alpha")
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_search_unknown_session_returns_empty(self, store):
        await store.add_conversation("user", "hello")
        results = await store.search_history("hello", session_id="nonexistent-session")
        assert results == []
