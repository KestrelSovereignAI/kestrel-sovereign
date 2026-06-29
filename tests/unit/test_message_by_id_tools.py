"""Single-message lifecycle by identity (#2022).

delete_messages matches by CONTENT; these tools address ONE row by its
message_id, with an optional session_id guard. Tests exercise the new store
membership check and the memory tools against a real SQLite DB + privacy facade
(NORMAL / EPHEMERAL / ISOLATED), so identity-vs-text, the guard, pin cleanup,
and the soft-delete-default / purge-harder contract are all verified end-to-end.
"""
import json
import tempfile
from pathlib import Path

import pytest

from kestrel_sovereign.storage.async_conversation_store import AsyncConversationStore
from kestrel_sovereign.storage.async_storage import AsyncStorage
from kestrel_sovereign.storage.privacy_wrapper import PrivacyEnforcingStorage
from kestrel_sovereign.privacy import PrivacyMode
from kestrel_sovereign.features.memory.feature import MemoryFeature

SID_A = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
SID_B = "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"


async def _insert(store, role, content, session_id, created):
    await store.db.execute_commit(
        "INSERT INTO conversation_history "
        "(agent_id, role, content, metadata, created_at) VALUES (?, ?, ?, ?, ?)",
        (store.agent_id, role, content, json.dumps({"session_id": session_id}), created),
    )
    row = await store.db.fetchall(
        "SELECT id FROM conversation_history ORDER BY id DESC LIMIT 1", ()
    )
    return row[0][0]


@pytest.fixture
async def store():
    """Real SQLite conv store + NORMAL privacy facade over the same DB."""
    with tempfile.TemporaryDirectory() as tmp:
        facade = await AsyncStorage.create_sqlite(str(Path(tmp) / "t.db"))
        facade.agent_id = "test-agent"
        conv = AsyncConversationStore(facade.db, agent_id="test-agent")
        facade.conversation = conv
        # A pin table so pin-cleanup is observable (production creates it via
        # the memory_agency feature; the wrapper tolerates its absence).
        await conv.db.execute_commit(
            "CREATE TABLE IF NOT EXISTS memory_pins "
            "(message_id INTEGER, agent_id TEXT)",
            (),
        )
        conv.privacy_storage = PrivacyEnforcingStorage(facade, PrivacyMode.NORMAL)
        yield conv
        await facade.close()


def _feature(store, privacy_storage=None):
    feature = MemoryFeature.__new__(MemoryFeature)
    feature.storage = privacy_storage or store.privacy_storage
    feature._get_conversation_store = lambda: store
    return feature


async def _live_contents(store):
    rows = await store.get_full_history_with_ids(include_excluded=True, include_stashed=True)
    return [r["content"] for r in rows]


# --------------------------------------------------------------------------
# Store membership resolver
# --------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_message_belongs_to_session_matches_by_identity(store):
    a_id = await _insert(store, "user", "x", SID_A, "2026-06-30 10:00:00")
    b_id = await _insert(store, "user", "x", SID_B, "2026-06-30 11:00:00")
    assert await store.message_belongs_to_session(a_id, SID_A) is True
    assert await store.message_belongs_to_session(a_id, SID_B) is False
    assert await store.message_belongs_to_session(b_id, SID_B) is True
    # Resolves trashed rows too (so a restore guard works).
    await store.delete_message(a_id)
    assert await store.message_belongs_to_session(a_id, SID_A) is True


# --------------------------------------------------------------------------
# Identity, not text
# --------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_delete_by_id_removes_exactly_one_of_duplicate_text(store):
    # Two messages with IDENTICAL text in different sessions. Deleting by id
    # removes precisely the addressed row — proving identity != text.
    a_id = await _insert(store, "user", "same text", SID_A, "2026-06-30 10:00:00")
    await _insert(store, "user", "same text", SID_B, "2026-06-30 11:00:00")

    result = await _feature(store).delete_message_by_id(a_id)
    assert result.status == "ok"
    assert result.data["message_id"] == a_id
    # The other identical-text message survives.
    assert await _live_contents(store) == ["same text"]


# --------------------------------------------------------------------------
# session_id guard
# --------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_wrong_session_guard_refuses_and_deletes_nothing(store):
    a_id = await _insert(store, "user", "alpha", SID_A, "2026-06-30 10:00:00")
    result = await _feature(store).delete_message_by_id(a_id, session_id=SID_B)
    assert result.status != "ok"
    # Untouched.
    assert await _live_contents(store) == ["alpha"]


@pytest.mark.asyncio
async def test_correct_session_guard_allows_delete(store):
    a_id = await _insert(store, "user", "alpha", SID_A, "2026-06-30 10:00:00")
    result = await _feature(store).delete_message_by_id(a_id, session_id=SID_A)
    assert result.status == "ok"
    assert await _live_contents(store) == []


# --------------------------------------------------------------------------
# Soft-delete default + restore round-trip
# --------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_delete_then_restore_round_trip(store):
    a_id = await _insert(store, "user", "alpha", SID_A, "2026-06-30 10:00:00")
    feature = _feature(store)

    assert (await feature.delete_message_by_id(a_id)).status == "ok"
    assert await _live_contents(store) == []  # in Trash, not live

    restored = await feature.restore_message_by_id(a_id)
    assert restored.status == "ok"
    assert await _live_contents(store) == ["alpha"]

    # Restoring a live (non-trashed) message is a no-op failure.
    assert (await feature.restore_message_by_id(a_id)).status != "ok"


# --------------------------------------------------------------------------
# Purge is intentionally harder
# --------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_purge_requires_confirm_and_is_permanent(store):
    a_id = await _insert(store, "user", "alpha", SID_A, "2026-06-30 10:00:00")
    feature = _feature(store)

    preview = await feature.purge_message_by_id(a_id)
    assert preview.status == "ok"
    assert preview.data["mode"] == "preview"
    # Nothing destroyed yet.
    assert await _live_contents(store) == ["alpha"]

    # Non-bool confirm refused.
    assert (await feature.purge_message_by_id(a_id, confirm="true")).status != "ok"

    done = await feature.purge_message_by_id(a_id, confirm=True)
    assert done.status == "ok"
    # Gone for good — not even in Trash.
    rows = await store.get_full_history_with_ids(
        include_excluded=True, include_stashed=True, include_deleted=True
    )
    assert all(r["id"] != a_id for r in rows)


# --------------------------------------------------------------------------
# Pin consistency
# --------------------------------------------------------------------------

async def _pin_count(store, message_id):
    rows = await store.db.fetchall(
        "SELECT 1 FROM memory_pins WHERE message_id = ? AND agent_id = ?",
        (message_id, store.agent_id),
    )
    return len(rows)


@pytest.mark.asyncio
async def test_delete_and_purge_drop_pins(store):
    a_id = await _insert(store, "user", "pinned", SID_A, "2026-06-30 10:00:00")
    await store.db.execute_commit(
        "INSERT INTO memory_pins (message_id, agent_id) VALUES (?, ?)",
        (a_id, store.agent_id),
    )
    assert await _pin_count(store, a_id) == 1

    # Soft-delete drops the pin (pins can't point into Trash).
    assert (await _feature(store).delete_message_by_id(a_id)).status == "ok"
    assert await _pin_count(store, a_id) == 0


# --------------------------------------------------------------------------
# Privacy modes
# --------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_ephemeral_refuses_and_guard_is_false(store):
    a_id = await _insert(store, "user", "alpha", SID_A, "2026-06-30 10:00:00")
    ephemeral = PrivacyEnforcingStorage(store.privacy_storage._storage, PrivacyMode.EPHEMERAL)
    feature = _feature(store, privacy_storage=ephemeral)

    assert (await feature.delete_message_by_id(a_id)).status != "ok"
    # Guard sees no persistent data.
    assert await ephemeral.message_belongs_to_session(a_id, SID_A) is False
    # Persistent row untouched.
    assert await _live_contents(store) == ["alpha"]


@pytest.mark.asyncio
async def test_isolated_deletes_from_in_memory_buffer(store):
    isolated = PrivacyEnforcingStorage(store.privacy_storage._storage, PrivacyMode.ISOLATED)
    await isolated.add_conversation("user", "iso one", session_id=SID_A)
    await isolated.add_conversation("user", "iso two", session_id=SID_A)
    feature = _feature(store, privacy_storage=isolated)

    # Index 0 belongs to SID_A; a wrong-session guard refuses.
    assert (await feature.delete_message_by_id(0, session_id=SID_B)).status != "ok"
    assert len(isolated._session_conversations) == 2

    # Correct id deletes that in-memory row.
    assert (await feature.delete_message_by_id(0, session_id=SID_A)).status == "ok"
    assert [c["content"] for c in isolated._session_conversations] == ["iso two"]
