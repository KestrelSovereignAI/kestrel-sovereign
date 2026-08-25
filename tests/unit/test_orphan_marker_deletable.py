"""Orphaned new_session marker must not make a session undeletable (#2027).

Observed live: after partial cleanup (content deleted individually), a session's
``new_session`` marker is left live and orphaned. The listing still surfaces the
session (the live marker anchors it + time-adjacent rows attach), but
``delete_conversation_session`` resolved only *content* rows — all trashed — and
returned 0 → "No active conversation found." The fix: session lifecycle ops
include the marker (``include_markers=True``), so the anchor is cleared.

**Scope moved under #3120, and the neighbour below is why.** These cases were
written asserting that delete touched only rows TAGGED with the session, and
the fixture's own comment says what that cost: the live unlabeled row five
minutes after the marker "gets misattributed to the phantom in the live
grouping". It is not a misattribution the list corrects — the list reports this
session with ``message_count: 1``, and that one message IS the neighbour. So
the old scope was a session you could open to find nothing, delete to lose
nothing, and purge while the row it showed you stayed live under a session of
its own. Lifecycle now touches what the list shows. #2027's actual subject —
the marker must be resolvable and must go with its session — is unchanged.
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
    """Trashed content + a live orphan marker + the live row beside it."""
    u = await _insert(store, "user", "test content", "2026-06-30 09:58:00")
    a = await _insert(store, "assistant", "test reply", "2026-06-30 09:59:00")
    marker = await _insert(store, "system", "", "2026-06-30 10:00:00", marker=True)
    # A live row with no session_id, time-adjacent to the marker. An unlabeled
    # row names no session of its own, so the grouper gives it to the one it
    # fell after — this one — and the list reports the phantom with it as its
    # single message.
    neighbour = await _insert(store, "user", "unrelated", "2026-06-30 10:05:00", session_id=None)
    await store.delete_message(u)
    await store.delete_message(a)  # content -> Trash, only the marker stays live
    return marker, neighbour


@pytest.mark.asyncio
async def test_orphan_marker_session_lists_but_now_deletes(store):
    marker, neighbour = await _orphan_setup(store)

    # The phantom shows in the active list (anchored by the live marker), and
    # it shows the neighbour as its one message — which is the whole reason
    # the scope below is what it is.
    listed = {
        s["session_id"]: s["message_count"]
        for s in await store.list_conversation_sessions()
    }
    assert listed.get(SID) == 1
    assert [row[0] for row in await store._get_session_messages(SID, limit=50)] == [
        neighbour
    ]

    # Pre-#2027 this returned 0 ("No active conversation found"). It resolves
    # the live orphan marker AND the row the list shows under it.
    deleted = await store.delete_conversation_session(SID)
    assert deleted == 2

    # The phantom is gone from the active list, and so is everything it showed.
    after = {s["session_id"] for s in await store.list_conversation_sessions()}
    assert SID not in after
    live = await _live_ids(store)
    assert marker not in live
    assert neighbour not in live


@pytest.mark.asyncio
async def test_delete_touches_exactly_what_the_list_showed(store):
    marker, neighbour = await _orphan_setup(store)
    live_before = set(await _live_ids(store))
    await store.delete_conversation_session(SID)
    live_after = set(await _live_ids(store))
    # The marker and the row listed under it. Nothing else — the scope grew to
    # match the list, it did not stop being a scope.
    assert live_before - live_after == {marker, neighbour}


@pytest.mark.asyncio
async def test_delete_restore_round_trip_includes_marker(store):
    marker, neighbour = await _orphan_setup(store)
    assert await store.delete_conversation_session(SID) == 2
    assert marker not in await _live_ids(store)
    # Restore brings the whole conversation back from Trash — the marker, the
    # two content rows trashed during cleanup, and the neighbour that went with
    # them — and the marker is live again (symmetric with delete including it).
    assert await store.restore_conversation_session(SID) == 4
    live = await _live_ids(store)
    assert marker in live
    assert neighbour in live


@pytest.mark.asyncio
async def test_purge_destroys_marker_too(store):
    marker, neighbour = await _orphan_setup(store)
    purged = await store.purge_conversation_session(SID)
    assert purged >= 1
    # Both are gone for good. A purge that spared the neighbour would leave a
    # row the list had just shown inside this session live underneath it,
    # where it reappears as a session of its own.
    all_ids = [r[0] for r in await store.db.fetchall("SELECT id FROM conversation_history", ())]
    assert marker not in all_ids
    assert neighbour not in all_ids


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
