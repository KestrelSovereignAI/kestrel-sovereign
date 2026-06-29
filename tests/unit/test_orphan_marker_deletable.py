"""Orphaned new_session marker must not make a session undeletable (#2027).

Observed live: after partial cleanup (content deleted individually), a session's
``new_session`` marker is left live and orphaned. The listing still surfaces the
session (the live marker anchors it + time-adjacent rows attach), but
``delete_conversation_session`` resolved only *content* rows — all trashed — and
returned 0 → "No active conversation found." The fix: session lifecycle ops
include the marker (``include_markers=True``), so the anchor is cleared.
"""
import json
import tempfile
from pathlib import Path

import pytest

from kestrel_sovereign.storage.async_conversation_store import AsyncConversationStore
from kestrel_sovereign.storage.async_database import AsyncDatabase

SID = "phantom-2027-uuid-aaaa"  # non-digit → canonical session label


async def _insert(store, role, content, created, *, session_id=SID, marker=False):
    meta = {"session_id": session_id} if session_id is not None else {}
    if marker:
        meta.update({"type": "session_marker", "new_session": True})
    await store.db.execute_commit(
        "INSERT INTO conversation_history "
        "(agent_id, role, content, metadata, created_at) VALUES (?, ?, ?, ?, ?)",
        (store.agent_id, role, content, json.dumps(meta), created),
    )
    row = await store.db.fetchall(
        "SELECT id FROM conversation_history ORDER BY id DESC LIMIT 1", ()
    )
    return row[0][0]


async def _live_ids(store):
    rows = await store.db.fetchall(
        "SELECT id FROM conversation_history WHERE deleted_at IS NULL ORDER BY id", ()
    )
    return [r[0] for r in rows]


@pytest.fixture
async def store():
    with tempfile.TemporaryDirectory() as tmp:
        db = await AsyncDatabase.sqlite(str(Path(tmp) / "t.db"))
        s = AsyncConversationStore(db, agent_id="test-agent")
        yield s
        await db.close()


async def _orphan_setup(store):
    """Trashed content + a live orphan marker + a foreign live neighbour."""
    u = await _insert(store, "user", "test content", "2026-06-30 09:58:00")
    a = await _insert(store, "assistant", "test reply", "2026-06-30 09:59:00")
    marker = await _insert(store, "system", "", "2026-06-30 10:00:00", marker=True)
    # A foreign live row with no session_id, time-adjacent to the marker — this
    # is what gets misattributed to the phantom in the live grouping.
    foreign = await _insert(store, "user", "unrelated", "2026-06-30 10:05:00", session_id=None)
    await store.delete_message(u)
    await store.delete_message(a)  # content -> Trash, only the marker stays live
    return marker, foreign


@pytest.mark.asyncio
async def test_orphan_marker_session_lists_but_now_deletes(store):
    marker, foreign = await _orphan_setup(store)

    # The phantom shows in the active list (anchored by the live marker).
    listed = {s["session_id"] for s in await store.list_conversation_sessions()}
    assert SID in listed

    # Pre-fix this returned 0 ("No active conversation found"); now it resolves
    # the live orphan marker and trashes it.
    deleted = await store.delete_conversation_session(SID)
    assert deleted == 1

    # The phantom is gone from the active list...
    after = {s["session_id"] for s in await store.list_conversation_sessions()}
    assert SID not in after
    # ...the marker is trashed, and the foreign row was NEVER touched.
    assert marker not in await _live_ids(store)
    assert foreign in await _live_ids(store)


@pytest.mark.asyncio
async def test_delete_only_touches_tagged_rows(store):
    marker, foreign = await _orphan_setup(store)
    live_before = set(await _live_ids(store))
    await store.delete_conversation_session(SID)
    live_after = set(await _live_ids(store))
    # Exactly the marker left the live set — nothing else.
    assert live_before - live_after == {marker}


@pytest.mark.asyncio
async def test_delete_restore_round_trip_includes_marker(store):
    marker, _ = await _orphan_setup(store)
    assert await store.delete_conversation_session(SID) == 1
    assert marker not in await _live_ids(store)
    # Restore brings the whole conversation back from Trash — the marker plus
    # the two content rows that were trashed during cleanup (3 total) — and the
    # marker is live again (symmetric with delete including it).
    assert await store.restore_conversation_session(SID) == 3
    assert marker in await _live_ids(store)


@pytest.mark.asyncio
async def test_purge_destroys_marker_too(store):
    marker, foreign = await _orphan_setup(store)
    purged = await store.purge_conversation_session(SID)
    assert purged >= 1
    # Marker is gone for good; foreign row untouched.
    all_ids = [r[0] for r in await store.db.fetchall("SELECT id FROM conversation_history", ())]
    assert marker not in all_ids
    assert foreign in all_ids


@pytest.mark.asyncio
async def test_healthy_session_delete_unchanged(store):
    # A normal session (content + marker, nothing pre-trashed) still deletes
    # fully — content AND marker go to Trash.
    u = await _insert(store, "user", "hello", "2026-06-30 11:00:00")
    marker = await _insert(store, "system", "", "2026-06-30 11:00:01", marker=True)
    a = await _insert(store, "assistant", "hi", "2026-06-30 11:00:02")
    deleted = await store.delete_conversation_session(SID)
    assert deleted == 3  # user + assistant + marker
    assert await _live_ids(store) == []
