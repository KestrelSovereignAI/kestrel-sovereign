"""Storage-layer tests for user-assigned conversation names (issue #716).

Exercises the real SQLite-backed ``AsyncConversationStore`` with the
``conversation_titles`` table so migrations, upsert semantics, and
trim/clear rules are all verified end-to-end without mocks.
"""

import tempfile
from pathlib import Path

import pytest

from kestrel_sovereign.storage.async_conversation_store import AsyncConversationStore
from kestrel_sovereign.storage.async_database import AsyncDatabase


@pytest.fixture
async def store():
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "test.db"
        db = await AsyncDatabase.sqlite(str(db_path))
        s = AsyncConversationStore(db, agent_id="test-agent")
        yield s
        await db.close()


@pytest.mark.asyncio
async def test_set_then_get_roundtrips(store):
    """Basic upsert: set a name, read it back verbatim."""
    stored = await store.set_conversation_name("sess-1", "My Thread")
    assert stored == "My Thread"
    fetched = await store.get_conversation_name("sess-1")
    assert fetched == "My Thread"


@pytest.mark.asyncio
async def test_set_trims_whitespace(store):
    """Leading/trailing whitespace is stripped at storage."""
    stored = await store.set_conversation_name("sess-1", "  Padded Name  ")
    assert stored == "Padded Name"
    assert await store.get_conversation_name("sess-1") == "Padded Name"


@pytest.mark.asyncio
async def test_set_caps_length(store):
    """Stored value is capped to ``MAX_CONVERSATION_NAME_LENGTH`` so a
    pathological UI (or bad client) can't wedge the sidebar with a
    multi-megabyte title.
    """
    cap = AsyncConversationStore.MAX_CONVERSATION_NAME_LENGTH
    very_long = "x" * (cap + 100)
    stored = await store.set_conversation_name("sess-1", very_long)
    assert stored is not None
    assert len(stored) == cap
    assert stored == "x" * cap


@pytest.mark.asyncio
async def test_empty_string_clears(store):
    """Empty string clears the row entirely (falls back to computed preview in UI)."""
    await store.set_conversation_name("sess-1", "Initial")
    cleared = await store.set_conversation_name("sess-1", "")
    assert cleared is None
    assert await store.get_conversation_name("sess-1") is None


@pytest.mark.asyncio
async def test_whitespace_only_clears(store):
    """Whitespace-only clears too — UI can commit a blanked field and
    get the same effect as explicit null.
    """
    await store.set_conversation_name("sess-1", "Initial")
    cleared = await store.set_conversation_name("sess-1", "   \t\n  ")
    assert cleared is None
    assert await store.get_conversation_name("sess-1") is None


@pytest.mark.asyncio
async def test_none_clears(store):
    """Passing Python None clears the row."""
    await store.set_conversation_name("sess-1", "Initial")
    cleared = await store.set_conversation_name("sess-1", None)
    assert cleared is None
    assert await store.get_conversation_name("sess-1") is None


@pytest.mark.asyncio
async def test_upsert_overwrites_previous_name(store):
    """Renaming twice replaces the previous name, doesn't create dup rows."""
    await store.set_conversation_name("sess-1", "First")
    await store.set_conversation_name("sess-1", "Second")
    assert await store.get_conversation_name("sess-1") == "Second"

    # Exactly one row exists for this (agent, session) pair.
    count = await store.db.fetchone(
        "SELECT COUNT(*) FROM conversation_titles "
        "WHERE agent_id = ? AND session_id = ?",
        ("test-agent", "sess-1"),
    )
    assert count[0] == 1


@pytest.mark.asyncio
async def test_names_are_per_agent_scoped(store):
    """Two agents naming the same session_id don't clobber each other."""
    await store.set_conversation_name("sess-1", "Agent A Title")

    store_b = AsyncConversationStore(store.db, agent_id="other-agent")
    await store_b.set_conversation_name("sess-1", "Agent B Title")

    assert await store.get_conversation_name("sess-1") == "Agent A Title"
    assert await store_b.get_conversation_name("sess-1") == "Agent B Title"


@pytest.mark.asyncio
async def test_get_conversation_names_bulk_only_returns_own_agent(store):
    """``get_conversation_names`` only sees this agent's rows, even when
    another agent has renames in the same table.
    """
    await store.set_conversation_name("sess-1", "Mine One")
    await store.set_conversation_name("sess-2", "Mine Two")

    store_b = AsyncConversationStore(store.db, agent_id="other-agent")
    await store_b.set_conversation_name("sess-1", "Theirs")

    names = await store.get_conversation_names()
    assert names == {"sess-1": "Mine One", "sess-2": "Mine Two"}

    names_b = await store_b.get_conversation_names()
    assert names_b == {"sess-1": "Theirs"}


@pytest.mark.asyncio
async def test_get_conversation_names_ignores_cleared_rows(store):
    """Cleared sessions (name = NULL or row deleted) don't leak into the
    bulk read — the client would otherwise see the cleared key and the
    UI would fall through to the preview.  Same visible behavior either
    way, but the bulk read staying tight saves a wire trip.
    """
    await store.set_conversation_name("sess-1", "Kept")
    await store.set_conversation_name("sess-2", "Temp")
    await store.set_conversation_name("sess-2", "")  # clear

    names = await store.get_conversation_names()
    assert names == {"sess-1": "Kept"}


@pytest.mark.asyncio
async def test_get_conversation_name_missing_session_is_none(store):
    """Never-renamed session → None (no row)."""
    assert await store.get_conversation_name("never-renamed") is None
