"""Integration tests for archive + unarchive round-trip (#2149).

Mirror image of ``test_soft_delete_round_trip.py`` for the new archive
state: write → archive → active list drops it → archived list surfaces it
→ unarchive → active list returns it. Uses a real SQLite database so the
``archived_at`` migration, index, and SQL filters are all exercised.

Runs against ``AsyncConversationStore`` directly (storage layer) and
through ``PrivacyEnforcingStorage`` (the endpoint path) so both surfaces
honor the same contract.
"""
from __future__ import annotations

import pytest

from kestrel_sovereign.privacy import PrivacyMode
from kestrel_sovereign.storage import AsyncStorage
from kestrel_sovereign.storage.privacy_wrapper import PrivacyEnforcingStorage


AGENT_ID = "did:test:archive-round-trip"


@pytest.mark.asyncio
async def test_archive_filters_from_default_reads(tmp_path):
    """archive_conversation_session stamps archived_at; live reads still
    return the rows (archived rows stay live, deleted_at IS NULL) but the
    only_archived view surfaces them.
    """
    db_path = tmp_path / "kestrel.db"
    async with AsyncStorage(str(db_path), agent_id=AGENT_ID) as storage:
        session_id = "arch-session"
        for role, content in [("user", "one"), ("assistant", "two")]:
            await storage.conversation.add_conversation(
                role, content, session_id=session_id
            )

        archived = await storage.conversation.archive_conversation_session(
            session_id
        )
        assert archived == 2

        # Archived rows are still live (never soft-deleted).
        live = await storage.conversation.get_full_history_with_ids()
        assert len(live) == 2
        assert all(m["archived_at"] is not None for m in live)
        assert all(m["deleted_at"] is None for m in live)

        # only_archived surfaces them.
        arch = await storage.conversation.get_full_history_with_ids(
            only_archived=True
        )
        assert len(arch) == 2
        assert all(m["archived_at"] is not None for m in arch)


@pytest.mark.asyncio
async def test_archive_then_unarchive(tmp_path):
    """unarchive clears archived_at; only_archived no longer returns it."""
    db_path = tmp_path / "kestrel.db"
    async with AsyncStorage(str(db_path), agent_id=AGENT_ID) as storage:
        session_id = "arch-session-2"
        await storage.conversation.add_conversation(
            "user", "hello", session_id=session_id
        )

        assert await storage.conversation.archive_conversation_session(
            session_id
        ) == 1
        assert len(
            await storage.conversation.get_full_history_with_ids(only_archived=True)
        ) == 1

        unarchived = await storage.conversation.unarchive_conversation_session(
            session_id
        )
        assert unarchived == 1

        assert await storage.conversation.get_full_history_with_ids(
            only_archived=True
        ) == []
        live = await storage.conversation.get_full_history_with_ids()
        assert len(live) == 1
        assert live[0]["archived_at"] is None


@pytest.mark.asyncio
async def test_archive_re_stamping_is_a_noop(tmp_path):
    """Archiving an already-archived session archives nothing new."""
    db_path = tmp_path / "kestrel.db"
    async with AsyncStorage(str(db_path), agent_id=AGENT_ID) as storage:
        session_id = "arch-session-3"
        await storage.conversation.add_conversation(
            "user", "twice", session_id=session_id
        )

        first = await storage.conversation.archive_conversation_session(session_id)
        assert first == 1

        again = await storage.conversation.archive_conversation_session(session_id)
        assert again == 0


@pytest.mark.asyncio
async def test_archive_does_not_touch_deleted_rows(tmp_path):
    """Soft-deleted rows are excluded from archiving (deleted_at IS NULL)."""
    db_path = tmp_path / "kestrel.db"
    async with AsyncStorage(str(db_path), agent_id=AGENT_ID) as storage:
        session_id = "arch-session-4"
        await storage.conversation.add_conversation(
            "user", "gone", session_id=session_id
        )
        rows = await storage.conversation.get_full_history_with_ids()
        await storage.conversation.delete_message(rows[0]["id"])

        # Nothing live left in the session to archive.
        archived = await storage.conversation.archive_conversation_session(
            session_id
        )
        assert archived == 0
        assert await storage.conversation.get_full_history_with_ids(
            only_archived=True
        ) == []


@pytest.mark.asyncio
async def test_session_archive_round_trip_through_privacy_wrapper(tmp_path):
    """End-to-end via the privacy wrapper (the HTTP-endpoint path).

    Archive a session → the active view drops it and the archived view /
    list surfaces it → unarchive → active view returns it. NORMAL mode.
    """
    db_path = tmp_path / "kestrel.db"
    async with AsyncStorage(str(db_path), agent_id=AGENT_ID) as underlying:
        wrapper = PrivacyEnforcingStorage(underlying, PrivacyMode.NORMAL)

        session_id = "wrapper-arch-session"
        for role, content in [
            ("user", "first"),
            ("assistant", "second"),
            ("user", "third"),
        ]:
            await wrapper.add_conversation(role, content, session_id=session_id)

        # Active view has all three; archived view is empty.
        assert len(await wrapper.query_conversations(AGENT_ID, view="active")) == 3
        assert await wrapper.query_conversations(AGENT_ID, view="archived") == []
        assert await wrapper.list_archived_conversations() == []

        # Archive the whole session.
        archived = await wrapper.archive_conversation_session(session_id, AGENT_ID)
        assert archived == 3

        # Active view drops it; archived view + list surface it.
        assert await wrapper.query_conversations(AGENT_ID, view="active") == []
        assert len(await wrapper.query_conversations(AGENT_ID, view="archived")) == 3
        arch_list = await wrapper.list_archived_conversations()
        assert len(arch_list) == 3
        assert all(m["archived_at"] is not None for m in arch_list)

        # Unarchive brings it back to active.
        unarchived = await wrapper.unarchive_conversation_session(
            session_id, AGENT_ID
        )
        assert unarchived == 3

        assert len(await wrapper.query_conversations(AGENT_ID, view="active")) == 3
        assert await wrapper.query_conversations(AGENT_ID, view="archived") == []
        assert await wrapper.list_archived_conversations() == []


@pytest.mark.asyncio
async def test_archive_view_default_is_active(tmp_path):
    """An unrecognized view falls back to active (no leak of archived rows)."""
    db_path = tmp_path / "kestrel.db"
    async with AsyncStorage(str(db_path), agent_id=AGENT_ID) as underlying:
        wrapper = PrivacyEnforcingStorage(underlying, PrivacyMode.NORMAL)
        session_id = "view-fallback-session"
        await wrapper.add_conversation("user", "hi", session_id=session_id)
        await wrapper.archive_conversation_session(session_id, AGENT_ID)

        # Garbage view -> active semantics -> archived row hidden.
        assert await wrapper.query_conversations(AGENT_ID, view="bogus") == []
