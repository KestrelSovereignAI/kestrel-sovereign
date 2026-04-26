"""Integration tests for soft-delete + restore + purge round-trip (#763).

Exercises the new lifecycle without an LLM: write → delete → trash list →
restore → delete → purge. Uses a real SQLite database so the schema
migration, index, and SQL filters are all exercised.

These tests run against ``AsyncConversationStore`` directly (storage
layer) and through ``PrivacyEnforcingStorage`` (the path the HTTP
endpoints take), to make sure both surfaces honor the same contract.
"""
from __future__ import annotations

import pytest

from kestrel_sovereign.privacy import PrivacyMode
from kestrel_sovereign.storage import AsyncStorage
from kestrel_sovereign.storage.privacy_wrapper import PrivacyEnforcingStorage


AGENT_ID = "did:test:soft-delete-round-trip"


@pytest.mark.asyncio
async def test_soft_delete_filters_from_default_reads(tmp_path):
    """delete_message stamps deleted_at; live reads stop returning it."""
    db_path = tmp_path / "kestrel.db"
    async with AsyncStorage(str(db_path), agent_id=AGENT_ID) as storage:
        await storage.conversation.add_conversation("user", "first message")
        await storage.conversation.add_conversation("assistant", "second message")
        await storage.conversation.add_conversation("user", "third message")

        live = await storage.conversation.get_full_history_with_ids()
        assert len(live) == 3
        target_id = live[1]["id"]

        deleted = await storage.conversation.delete_message(target_id)
        assert deleted is True

        live_after = await storage.conversation.get_full_history_with_ids()
        assert len(live_after) == 2
        assert all(m["id"] != target_id for m in live_after)

        # Soft-deleted row is still in the table — only hidden by default
        trash = await storage.conversation.get_full_history_with_ids(
            only_deleted=True
        )
        assert len(trash) == 1
        assert trash[0]["id"] == target_id
        assert trash[0]["deleted_at"] is not None


@pytest.mark.asyncio
async def test_soft_delete_then_restore(tmp_path):
    """restore_message clears deleted_at and the row reappears in live reads."""
    db_path = tmp_path / "kestrel.db"
    async with AsyncStorage(str(db_path), agent_id=AGENT_ID) as storage:
        await storage.conversation.add_conversation("user", "hello")
        rows = await storage.conversation.get_full_history_with_ids()
        msg_id = rows[0]["id"]

        await storage.conversation.delete_message(msg_id)
        live = await storage.conversation.get_full_history_with_ids()
        assert live == []

        restored = await storage.conversation.restore_message(msg_id)
        assert restored is True

        live = await storage.conversation.get_full_history_with_ids()
        assert len(live) == 1
        assert live[0]["id"] == msg_id
        assert live[0]["deleted_at"] is None


@pytest.mark.asyncio
async def test_soft_delete_re_stamping_is_a_noop(tmp_path):
    """Soft-deleting an already-trashed row is a no-op (False / 0).

    This protects retention semantics — clicking delete twice doesn't
    extend the deleted_at timestamp and push the row's sweep date back.
    """
    db_path = tmp_path / "kestrel.db"
    async with AsyncStorage(str(db_path), agent_id=AGENT_ID) as storage:
        await storage.conversation.add_conversation("user", "twice")
        rows = await storage.conversation.get_full_history_with_ids()
        msg_id = rows[0]["id"]

        first = await storage.conversation.delete_message(msg_id)
        assert first is True

        again = await storage.conversation.delete_message(msg_id)
        assert again is False


@pytest.mark.asyncio
async def test_purge_message_destroys_row(tmp_path):
    """purge_message hard-deletes; restore can no longer recover it."""
    db_path = tmp_path / "kestrel.db"
    async with AsyncStorage(str(db_path), agent_id=AGENT_ID) as storage:
        await storage.conversation.add_conversation("user", "delete me")
        rows = await storage.conversation.get_full_history_with_ids()
        msg_id = rows[0]["id"]

        purged = await storage.conversation.purge_message(
            msg_id, reason="test"
        )
        assert purged is True

        # No live, no trash — the row is gone.
        live = await storage.conversation.get_full_history_with_ids()
        assert live == []
        trash = await storage.conversation.get_full_history_with_ids(
            only_deleted=True
        )
        assert trash == []

        # Restore reports failure: there's nothing to clear deleted_at on.
        restored = await storage.conversation.restore_message(msg_id)
        assert restored is False


@pytest.mark.asyncio
async def test_clear_history_is_soft(tmp_path):
    """clear_history stamps deleted_at on every live row but leaves data
    in the table for Trash recovery.
    """
    db_path = tmp_path / "kestrel.db"
    async with AsyncStorage(str(db_path), agent_id=AGENT_ID) as storage:
        for i in range(5):
            await storage.conversation.add_conversation("user", f"msg {i}")

        await storage.conversation.clear_history()

        live = await storage.conversation.get_full_history_with_ids()
        assert live == []

        trash = await storage.conversation.get_full_history_with_ids(
            only_deleted=True
        )
        assert len(trash) == 5


@pytest.mark.asyncio
async def test_purge_all_destroys_everything(tmp_path):
    """purge_all hard-deletes both live and soft-deleted rows.

    Sentinel for the EPHEMERAL hard-purge path (#767) and for the
    sovereign_adapter restore-from-CAR rebuild path that always runs a
    raw DELETE before re-inserting.
    """
    db_path = tmp_path / "kestrel.db"
    async with AsyncStorage(str(db_path), agent_id=AGENT_ID) as storage:
        await storage.conversation.add_conversation("user", "live")
        rows = await storage.conversation.get_full_history_with_ids()
        live_id = rows[0]["id"]
        await storage.conversation.add_conversation("user", "to be trashed")
        trash_rows = await storage.conversation.get_full_history_with_ids()
        trash_id = next(r["id"] for r in trash_rows if r["id"] != live_id)
        await storage.conversation.delete_message(trash_id)

        purged = await storage.conversation.purge_all(reason="test")
        assert purged == 2

        live = await storage.conversation.get_full_history_with_ids()
        trash = await storage.conversation.get_full_history_with_ids(
            only_deleted=True
        )
        assert live == []
        assert trash == []


@pytest.mark.asyncio
async def test_session_round_trip_through_privacy_wrapper(tmp_path):
    """End-to-end: privacy wrapper soft-deletes a session, lists trash,
    restores it, then permanently purges it.

    This is the path the HTTP endpoints take. NORMAL mode only — the
    other modes have separate semantics covered by the privacy_wrapper
    unit tests.
    """
    db_path = tmp_path / "kestrel.db"
    async with AsyncStorage(str(db_path), agent_id=AGENT_ID) as underlying:
        wrapper = PrivacyEnforcingStorage(underlying, PrivacyMode.NORMAL)

        # Build a session of three messages that all share an explicit
        # session_id in metadata. This exercises the metadata-based
        # session resolution path.
        session_id = "test-session-uuid"
        for role, content in [
            ("user", "first"),
            ("assistant", "second"),
            ("user", "third"),
        ]:
            await wrapper.add_conversation(
                role, content, session_id=session_id
            )

        # Soft-delete the whole session
        deleted = await wrapper.delete_conversation_session(
            session_id, AGENT_ID
        )
        assert deleted == 3

        # Live reads return nothing
        live = await wrapper.query_conversations(AGENT_ID)
        assert live == []

        # Trash listing finds all three rows
        trash = await wrapper.list_trashed_conversations()
        assert len(trash) == 3

        # Restore brings them back
        restored = await wrapper.restore_conversation_session(
            session_id, AGENT_ID
        )
        assert restored == 3

        live = await wrapper.query_conversations(AGENT_ID)
        assert len(live) == 3

        # Final hard purge — gone forever, audit reason recorded
        purged = await wrapper.purge_conversation_session(
            session_id, AGENT_ID, reason="integration-test"
        )
        assert purged == 3

        live = await wrapper.query_conversations(AGENT_ID)
        trash = await wrapper.list_trashed_conversations()
        assert live == []
        assert trash == []
