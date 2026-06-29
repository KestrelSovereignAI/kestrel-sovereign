"""Message-level trash navigation (#2025).

list_conversations(include_trashed=True) lists whole trashed SESSIONS; this tool
lists individual trashed MESSAGES so the agent can discover ids to feed
restore_message_by_id / purge_message_by_id. Verified against a real SQLite DB +
privacy facade (NORMAL / EPHEMERAL / ISOLATED).
"""
import json
import tempfile
from pathlib import Path

import pytest

from kestrel_sovereign.storage.async_conversation_store import AsyncConversationStore
from kestrel_sovereign.storage.async_storage import AsyncStorage
from kestrel_sovereign.storage.privacy_wrapper import PrivacyEnforcingStorage
from kestrel_sovereign.privacy import PrivacyMode
from kestrel_sovereign.security.input_guardrails import wrap_user_input
from kestrel_sovereign.features.memory.feature import MemoryFeature

SID = "cccccccc-cccc-cccc-cccc-cccccccccccc"


async def _insert(store, role, content, created, *, sent_form=False):
    meta = {"session_id": SID}
    if sent_form:
        meta["sent_form"] = True
    await store.db.execute_commit(
        "INSERT INTO conversation_history "
        "(agent_id, role, content, metadata, created_at) VALUES (?, ?, ?, ?, ?)",
        (store.agent_id, role, content, json.dumps(meta), created),
    )
    row = await store.db.fetchall(
        "SELECT id FROM conversation_history ORDER BY id DESC LIMIT 1", ()
    )
    return row[0][0]


@pytest.fixture
async def store():
    with tempfile.TemporaryDirectory() as tmp:
        facade = await AsyncStorage.create_sqlite(str(Path(tmp) / "t.db"))
        facade.agent_id = "test-agent"
        conv = AsyncConversationStore(facade.db, agent_id="test-agent")
        facade.conversation = conv
        conv.privacy_storage = PrivacyEnforcingStorage(facade, PrivacyMode.NORMAL)
        yield conv
        await facade.close()


def _feature(store, privacy_storage=None):
    feature = MemoryFeature.__new__(MemoryFeature)
    feature.storage = privacy_storage or store.privacy_storage
    feature._get_conversation_store = lambda: store
    return feature


@pytest.mark.asyncio
async def test_lists_only_trashed_messages(store):
    live_id = await _insert(store, "user", "still here", "2026-06-30 10:00:00")
    gone_id = await _insert(store, "user", "trashed text", "2026-06-30 10:01:00")
    await store.delete_message(gone_id)

    result = await _feature(store).list_trashed_messages()
    assert result.status == "ok"
    ids = [m["message_id"] for m in result.data["messages"]]
    assert gone_id in ids
    assert live_id not in ids  # live messages never appear
    row = next(m for m in result.data["messages"] if m["message_id"] == gone_id)
    assert row["session_id"] == SID
    assert row["preview"] == "trashed text"
    assert row["deleted_at"] is not None


@pytest.mark.asyncio
async def test_discover_then_restore_round_trip(store):
    mid = await _insert(store, "user", "oops", "2026-06-30 10:00:00")
    await store.delete_message(mid)
    feature = _feature(store)

    listed = await feature.list_trashed_messages()
    assert mid in [m["message_id"] for m in listed.data["messages"]]

    restored = await feature.restore_message_by_id(mid)
    assert restored.status == "ok"

    after = await feature.list_trashed_messages()
    assert mid not in [m["message_id"] for m in after.data["messages"]]


@pytest.mark.asyncio
async def test_preview_unwraps_sent_form(store):
    # A user turn persisted in sent-form must preview as the raw user text.
    mid = await _insert(
        store, "user", wrap_user_input("what did we decide?"),
        "2026-06-30 10:00:00", sent_form=True,
    )
    await store.delete_message(mid)
    result = await _feature(store).list_trashed_messages()
    row = next(m for m in result.data["messages"] if m["message_id"] == mid)
    assert row["preview"] == "what did we decide?"


@pytest.mark.asyncio
async def test_trashed_session_markers_are_hidden(store):
    # Deleting a session trashes its new_session marker (#2027); that structural
    # row must NOT show in the message-level Trash listing or inflate the count.
    await _insert(store, "user", "real msg", "2026-06-30 10:00:00")
    await store.db.execute_commit(
        "INSERT INTO conversation_history "
        "(agent_id, role, content, metadata, created_at) VALUES (?, ?, ?, ?, ?)",
        (store.agent_id, "system", "",
         json.dumps({"session_id": SID, "type": "session_marker", "new_session": True}),
         "2026-06-30 10:00:01"),
    )
    await store.delete_conversation_session(SID)  # trashes content + marker

    result = await _feature(store).list_trashed_messages()
    assert "system" not in [m["role"] for m in result.data["messages"]]  # marker hidden
    assert result.data["count"] == 1  # only the real message
    assert result.data["messages"][0]["preview"] == "real msg"


@pytest.mark.asyncio
async def test_empty_trash_returns_zero(store):
    await _insert(store, "user", "live only", "2026-06-30 10:00:00")
    result = await _feature(store).list_trashed_messages()
    assert result.status == "ok"
    assert result.data["count"] == 0


@pytest.mark.asyncio
async def test_ephemeral_and_isolated_expose_no_trash(store):
    mid = await _insert(store, "user", "trashed", "2026-06-30 10:00:00")
    await store.delete_message(mid)

    ephemeral = PrivacyEnforcingStorage(store.privacy_storage._storage, PrivacyMode.EPHEMERAL)
    assert (await _feature(store, ephemeral).list_trashed_messages()).data["count"] == 0

    isolated = PrivacyEnforcingStorage(store.privacy_storage._storage, PrivacyMode.ISOLATED)
    assert (await _feature(store, isolated).list_trashed_messages()).data["count"] == 0
