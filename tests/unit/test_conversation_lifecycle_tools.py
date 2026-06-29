"""Session-grain navigation + lifecycle, store and memory-tool layers (#2019).

Covers the gap that motivated the work: the agent could only pattern-delete
individual messages, never see or operate on whole conversations. These tests
exercise the new store methods against a real SQLite DB and the new memory
tools against that same store (via a bypassed MemoryFeature), so the wiring is
verified end-to-end, not mocked.
"""
import json
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from kestrel_sovereign.storage.async_conversation_store import AsyncConversationStore
from kestrel_sovereign.storage.async_storage import AsyncStorage
from kestrel_sovereign.storage.privacy_wrapper import PrivacyEnforcingStorage
from kestrel_sovereign.privacy import PrivacyMode
from kestrel_sovereign.features.memory.feature import MemoryFeature

SID_A = "11111111-1111-1111-1111-111111111111"
SID_B = "22222222-2222-2222-2222-222222222222"
BASE = datetime(2026, 6, 29, 12, 0, 0)


async def _insert(store, role, content, session_id, minutes):
    created = (BASE + timedelta(minutes=minutes)).strftime("%Y-%m-%d %H:%M:%S")
    meta = json.dumps({"session_id": session_id})
    await store.db.execute_commit(
        "INSERT INTO conversation_history "
        "(agent_id, role, content, metadata, created_at) VALUES (?, ?, ?, ?, ?)",
        (store.agent_id, role, content, meta, created),
    )


@pytest.fixture
async def store():
    """Real SQLite conversation store seeded with two distinct sessions.

    Tools route through the privacy facade, so the fixture also exposes a
    privacy-wrapped AsyncStorage (NORMAL mode) via ``store.privacy_storage``
    so the same DB is reachable from both layers.
    """
    with tempfile.TemporaryDirectory() as tmp:
        facade = await AsyncStorage.create_sqlite(str(Path(tmp) / "t.db"))
        facade.agent_id = "test-agent"
        conv = AsyncConversationStore(facade.db, agent_id="test-agent")
        facade.conversation = conv
        # Session A
        await _insert(conv, "user", "alpha secret", SID_A, 0)
        await _insert(conv, "assistant", "reply A", SID_A, 1)
        # Session B — large gap so the grouper treats it as separate
        await _insert(conv, "user", "beta secret", SID_B, 100)
        await _insert(conv, "assistant", "reply B", SID_B, 101)
        # Attach a NORMAL-mode privacy facade over the same DB for tool tests.
        privacy = PrivacyEnforcingStorage(facade, PrivacyMode.NORMAL)
        conv.privacy_storage = privacy
        yield conv
        await facade.close()


def _feature(store, privacy_storage=None):
    """A MemoryFeature wired to the real storage without full agent bootstrap."""
    feature = MemoryFeature.__new__(MemoryFeature)
    feature.storage = privacy_storage or store.privacy_storage
    feature._get_conversation_store = lambda: store
    return feature


# --------------------------------------------------------------------------
# Store layer
# --------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_list_sessions_groups_and_orders_newest_first(store):
    sessions = await store.list_conversation_sessions()
    assert [s["session_id"] for s in sessions] == [SID_B, SID_A]
    assert all(s["message_count"] == 2 for s in sessions)
    assert all(s["user_message_count"] == 1 for s in sessions)
    assert sessions[0]["preview"] == "beta secret"
    assert sessions[0]["is_trashed"] is False


@pytest.mark.asyncio
async def test_resumed_session_lists_once_and_deletes_wholly(store):
    # Resume session A far past the gap (same UUID). It must surface as ONE
    # entry (not two), and deleting it must take all of its rows — the codex
    # P2 collision guard (#2019).
    await _insert(store, "user", "alpha resumed", SID_A, 300)
    sessions = await store.list_conversation_sessions()
    # A was resumed most recently, so it ranks first by last activity.
    assert [s["session_id"] for s in sessions] == [SID_A, SID_B]
    a = next(s for s in sessions if s["session_id"] == SID_A)
    assert a["message_count"] == 3  # 2 original + 1 resumed, coalesced
    # Deleting A removes every A row, including the resumed cluster.
    assert await store.delete_conversation_session(SID_A) == 3
    assert [s["session_id"] for s in await store.list_conversation_sessions()] == [SID_B]


@pytest.mark.asyncio
async def test_find_messages_matching_scopes_to_session(store):
    everywhere = await store.find_messages_matching("secret")
    assert len(everywhere) == 2
    scoped = await store.find_messages_matching("secret", session_id=SID_A)
    assert len(scoped) == 1
    assert scoped[0]["content"] == "alpha secret"


@pytest.mark.asyncio
async def test_scoped_delete_does_not_touch_other_sessions(store):
    deleted = await store.delete_messages_matching("secret", session_id=SID_A)
    assert deleted == 1
    # The beta "secret" message in session B must survive.
    remaining = await store.find_messages_matching("secret")
    assert [m["content"] for m in remaining] == ["beta secret"]


@pytest.mark.asyncio
async def test_delete_restore_purge_session_round_trip(store):
    assert await store.delete_conversation_session(SID_B) == 2
    active = await store.list_conversation_sessions()
    assert [s["session_id"] for s in active] == [SID_A]
    trashed = await store.list_conversation_sessions(include_trashed=True)
    assert [s["session_id"] for s in trashed] == [SID_B]
    assert trashed[0]["is_trashed"] is True

    assert await store.restore_conversation_session(SID_B) == 2
    assert len(await store.list_conversation_sessions()) == 2

    assert await store.purge_conversation_session(SID_B) == 2
    assert [s["session_id"] for s in await store.list_conversation_sessions()] == [SID_A]
    assert await store.list_conversation_sessions(include_trashed=True) == []


# --------------------------------------------------------------------------
# Memory tool layer
# --------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_list_conversations_tool_returns_sessions(store):
    result = await _feature(store).list_conversations()
    assert result.status == "ok"
    assert result.data["count"] == 2
    assert [s["session_id"] for s in result.data["sessions"]] == [SID_B, SID_A]


@pytest.mark.asyncio
async def test_delete_conversation_tool_preview_then_confirm(store):
    feature = _feature(store)

    preview = await feature.delete_conversation(SID_B)
    assert preview.status == "ok"
    assert preview.data["mode"] == "preview"
    assert preview.data["session"]["message_count"] == 2
    # Nothing deleted yet.
    assert len(await store.list_conversation_sessions()) == 2

    done = await feature.delete_conversation(SID_B, confirm=True)
    assert done.status == "ok"
    assert done.data["deleted"] == 2
    assert [s["session_id"] for s in await store.list_conversation_sessions()] == [SID_A]


@pytest.mark.asyncio
async def test_delete_conversation_tool_rejects_unknown_session(store):
    result = await _feature(store).delete_conversation("no-such-session", confirm=True)
    assert result.status != "ok"


@pytest.mark.asyncio
async def test_delete_conversation_tool_rejects_nonbool_confirm(store):
    result = await _feature(store).delete_conversation(SID_B, confirm="true")
    assert result.status != "ok"
    # Must not have deleted anything.
    assert len(await store.list_conversation_sessions()) == 2


@pytest.mark.asyncio
async def test_restore_conversation_tool(store):
    await store.delete_conversation_session(SID_B)
    result = await _feature(store).restore_conversation(SID_B)
    assert result.status == "ok"
    assert result.data["restored"] == 2
    assert len(await store.list_conversation_sessions()) == 2


@pytest.mark.asyncio
async def test_purge_conversation_tool_is_permanent(store):
    feature = _feature(store)
    preview = await feature.purge_conversation(SID_B)
    assert preview.status == "ok" and preview.data["mode"] == "preview"

    done = await feature.purge_conversation(SID_B, confirm=True)
    assert done.status == "ok"
    assert done.data["purged"] == 2
    # Gone from both active and trash — no recovery.
    assert await store.list_conversation_sessions(include_trashed=True) == []


@pytest.mark.asyncio
async def test_delete_messages_positional_confirm_still_deletes(store):
    # Back-compat: delete_messages(pattern, True) must perform a real delete,
    # not bind True to session_id and silently become a preview (codex P2).
    result = await _feature(store).delete_messages("alpha secret", True)
    assert result.status == "ok"
    assert result.data["mode"] == "delete"
    assert result.data["deleted"] == 1


@pytest.mark.asyncio
async def test_purge_preview_counts_live_and_trashed(store):
    # Trash one of session A's two messages, then preview a purge: it must
    # report BOTH the live and trashed rows it will destroy (codex P2).
    await store.delete_messages_matching("alpha secret", session_id=SID_A)  # 1 → trash
    feature = _feature(store)
    preview = await feature.purge_conversation(SID_A)
    assert preview.status == "ok"
    assert preview.data["would_destroy"] == 2
    assert preview.data["live"] == 1
    assert preview.data["trashed"] == 1


@pytest.mark.asyncio
async def test_purge_preview_counts_partially_trashed_legacy_session(store):
    # Legacy row-id session (no metadata session_id) with one row trashed: the
    # purge preview must count BOTH via the resolver, not via a mis-keyed
    # trashed summary (codex P2).
    created0 = "2026-06-30 09:00:00"
    created1 = "2026-06-30 09:01:00"
    await store.db.execute_commit(
        "INSERT INTO conversation_history (agent_id, role, content, metadata, created_at) "
        "VALUES (?, ?, ?, ?, ?)",
        (store.agent_id, "user", "legacy one", "{}", created0),
    )
    await store.db.execute_commit(
        "INSERT INTO conversation_history (agent_id, role, content, metadata, created_at) "
        "VALUES (?, ?, ?, ?, ?)",
        (store.agent_id, "assistant", "legacy two", "{}", created1),
    )
    rows = await store.db.fetchall(
        "SELECT id FROM conversation_history WHERE content LIKE 'legacy%' ORDER BY id",
        (),
    )
    anchor_id, second_id = str(rows[0][0]), rows[1][0]
    await store.delete_message(second_id)  # trash one of the two

    preview = await _feature(store).purge_conversation(anchor_id)
    assert preview.status == "ok"
    assert preview.data["would_destroy"] == 2  # live anchor + trashed sibling
    assert preview.data["live"] == 1
    assert preview.data["trashed"] == 1


@pytest.mark.asyncio
async def test_delete_messages_tool_scopes_to_session(store):
    feature = _feature(store)
    result = await feature.delete_messages("secret", session_id=SID_A, confirm=True)
    assert result.status == "ok"
    assert result.data["deleted"] == 1
    assert result.data["session_id"] == SID_A
    remaining = await store.find_messages_matching("secret")
    assert [m["content"] for m in remaining] == ["beta secret"]


# --------------------------------------------------------------------------
# Privacy enforcement — tools must honor the facade, not bypass it (codex P1)
# --------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_ephemeral_mode_hides_and_refuses_persistent_ops(store):
    # An EPHEMERAL agent has no persistent data: listing exposes nothing and
    # the destructive ops must refuse rather than reach the raw store.
    ephemeral = PrivacyEnforcingStorage(store.privacy_storage._storage, PrivacyMode.EPHEMERAL)
    feature = _feature(store, privacy_storage=ephemeral)

    listed = await feature.list_conversations()
    assert listed.status == "ok"
    assert listed.data["count"] == 0  # persistent rows not exposed

    deleted = await feature.delete_conversation(SID_A, confirm=True)
    assert deleted.status != "ok"  # refused under ephemeral mode

    purged = await feature.purge_conversation(SID_A, confirm=True)
    assert purged.status != "ok"

    # And the underlying persistent rows are untouched.
    assert len(await store.list_conversation_sessions()) == 2


@pytest.mark.asyncio
async def test_isolated_scoped_delete_respects_session(store):
    # In ISOLATED mode, a scoped delete_messages must not reach across
    # in-memory conversations that share the same text (codex P2).
    isolated = PrivacyEnforcingStorage(store.privacy_storage._storage, PrivacyMode.ISOLATED)
    isolated._session_conversations = [
        {"role": "user", "content": "shared text", "metadata": {"session_id": SID_A}},
        {"role": "user", "content": "shared text", "metadata": {"session_id": SID_B}},
    ]
    deleted = await isolated.delete_messages_matching("shared", session_id=SID_A)
    assert deleted == 1
    survivors = [c["metadata"]["session_id"] for c in isolated._session_conversations]
    assert survivors == [SID_B]


@pytest.mark.asyncio
async def test_isolated_list_and_delete_are_session_scoped(store):
    # ISOLATED listing must surface the real per-session ids, and deleting one
    # must leave the other intact (codex P2 follow-up).
    isolated = PrivacyEnforcingStorage(store.privacy_storage._storage, PrivacyMode.ISOLATED)
    # Two isolated sessions separated by a real time gap so the grouper splits
    # them (differing session_ids alone don't create a boundary).
    isolated._session_conversations = [
        {"role": "user", "content": "first", "session_id": SID_A,
         "metadata": {}, "created_at": "2026-06-29 12:00:00"},
        {"role": "user", "content": "second", "session_id": SID_B,
         "metadata": {}, "created_at": "2026-06-29 18:00:00"},
    ]
    feature = _feature(store, privacy_storage=isolated)

    listed = await feature.list_conversations()
    ids = {s["session_id"] for s in listed.data["sessions"]}
    assert ids == {SID_A, SID_B}

    done = await feature.delete_conversation(SID_A, confirm=True)
    assert done.status == "ok"
    remaining = {c["session_id"] for c in isolated._session_conversations}
    assert remaining == {SID_B}


@pytest.mark.asyncio
async def test_isolated_unlabeled_session_is_listable_and_deletable(store):
    # ISOLATED messages stored without a session_id must still be navigable:
    # they bucket under a stable sentinel id that delete can resolve, never a
    # synthetic index (codex P2).
    isolated = PrivacyEnforcingStorage(store.privacy_storage._storage, PrivacyMode.ISOLATED)
    await isolated.add_conversation("user", "no session id here")  # session_id=None
    feature = _feature(store, privacy_storage=isolated)

    listed = await feature.list_conversations()
    assert listed.data["count"] == 1
    sid = listed.data["sessions"][0]["session_id"]

    done = await feature.delete_conversation(sid, confirm=True)
    assert done.status == "ok"
    assert done.data["deleted"] == 1
    assert isolated._session_conversations == []


@pytest.mark.asyncio
async def test_resumed_session_ranks_by_latest_activity(store):
    # Session A is older but gets resumed after B — it must list FIRST (newest
    # activity), not be buried under B by first-cluster position (#2019).
    await _insert(store, "user", "alpha resumed", SID_A, 300)
    sessions = await store.list_conversation_sessions()
    assert sessions[0]["session_id"] == SID_A
    assert sessions[0]["message_count"] == 3
