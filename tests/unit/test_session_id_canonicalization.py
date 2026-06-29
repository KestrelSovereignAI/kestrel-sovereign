"""End-to-end storage tests for canonical session identity (#2012).

The conversation-list endpoint historically keyed each session by the row-id
of its first message, so the web UI round-tripped a bare integer (e.g.
``"1314"``) back as the ``session_id`` on the next turn. Stamping that integer
onto continued messages split one conversation across two keys (the integer
vs. the UUID on the session's own ``new_session`` marker), so the message pane
loaded empty on a hard refresh.

These exercise the real SQLite-backed ``AsyncConversationStore``:

- ``_canonicalize_session_id`` maps an integer marker-row-id to the marker UUID
- ``add_conversation`` / ``resolve_session_id`` apply it, so a continued turn
  echoing the integer is filed under the canonical UUID
- the store's own dual-scheme resolver then loads the whole conversation under
  that UUID
"""

import json
import tempfile
from pathlib import Path

import pytest

from kestrel_sovereign.storage.async_conversation_store import AsyncConversationStore
from kestrel_sovereign.storage.async_database import AsyncDatabase


@pytest.fixture
async def store():
    with tempfile.TemporaryDirectory() as tmp:
        db = await AsyncDatabase.sqlite(str(Path(tmp) / "canonical.db"))
        yield AsyncConversationStore(db, agent_id="test-agent")
        await db.close()


async def _marker_with_uuid(store, uuid):
    """Persist a new_session marker carrying ``uuid`` and return its row-id."""
    await store.add_conversation(
        "system", "",
        metadata={"new_session": True, "type": "session_marker"},
        session_id=uuid,
    )
    row = await store.db.fetchone(
        "SELECT id FROM conversation_history "
        "WHERE agent_id = ? ORDER BY id DESC LIMIT 1",
        ("test-agent",),
    )
    return row[0]


async def _session_id_of(store, row_id):
    row = await store.db.fetchone(
        "SELECT metadata FROM conversation_history WHERE id = ?", (row_id,)
    )
    return json.loads(row[0])["session_id"]


@pytest.mark.asyncio
async def test_canonicalize_maps_integer_marker_rowid_to_uuid(store):
    uuid = "e1fd6fe5-885e-4d8b-9aaa-0000000000aa"
    marker_id = await _marker_with_uuid(store, uuid)

    assert await store._canonicalize_session_id(str(marker_id)) == uuid
    # Already-canonical UUID and None pass through.
    assert await store._canonicalize_session_id(uuid) == uuid
    assert await store._canonicalize_session_id(None) is None


@pytest.mark.asyncio
async def test_canonicalize_leaves_non_marker_integer_alone(store):
    # A plain user turn whose row-id is NOT a new_session marker — a
    # legitimate legacy time-gap key with no UUID to map to.
    await store.add_conversation("user", "first", session_id="some-uuid-x")
    row = await store.db.fetchone(
        "SELECT id FROM conversation_history ORDER BY id DESC LIMIT 1", ()
    )
    legacy_anchor = str(row[0])
    assert await store._canonicalize_session_id(legacy_anchor) == legacy_anchor


@pytest.mark.asyncio
async def test_add_conversation_relinks_integer_echo_to_uuid(store):
    uuid = "53519af8-0000-0000-0000-0000000000bb"
    marker_id = await _marker_with_uuid(store, uuid)

    # UI echoes the integer marker-row-id as the session_id on the next turn.
    await store.add_conversation("user", "hi", session_id=str(marker_id))
    await store.add_conversation("assistant", "hello", session_id=str(marker_id))

    rows = await store.db.fetchall(
        "SELECT id, role, metadata FROM conversation_history "
        "WHERE role IN ('user', 'assistant') ORDER BY id", ()
    )
    for _id, _role, meta_json in rows:
        assert json.loads(meta_json)["session_id"] == uuid


@pytest.mark.asyncio
async def test_resolve_session_id_canonicalizes_explicit_integer(store):
    uuid = "722e2ba0-0000-0000-0000-0000000000cc"
    marker_id = await _marker_with_uuid(store, uuid)
    assert await store.resolve_session_id(str(marker_id)) == uuid


@pytest.mark.asyncio
async def test_continued_turns_load_under_canonical_uuid(store):
    """The payoff: after canonicalization, the store's dual-scheme resolver
    loads the whole conversation under the marker UUID — the continued turns
    are no longer orphaned under the integer key."""
    uuid = "60402f43-0000-0000-0000-0000000000dd"
    marker_id = await _marker_with_uuid(store, uuid)
    await store.add_conversation("user", "hi", session_id=str(marker_id))
    await store.add_conversation("assistant", "hello", session_id=str(marker_id))

    rows = await store._get_session_messages(uuid, limit=50)
    roles = [r[1] for r in rows]
    # Marker stripped; both content turns present under the UUID.
    assert roles.count("user") == 1
    assert roles.count("assistant") == 1
