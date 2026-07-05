"""Regression (#2158 follow-up): metadata flag lookups must match BOTH the
space form (Python ``json.dumps`` — ``"key": value``) AND the minified form
that SQLite ``json_set()`` re-emits (``"key":value``).

#2158 switched ``decay_protected`` writes to an atomic SQL ``json_set`` UPDATE,
which rewrites the WHOLE metadata object minified. Any row that has ever been
pinned (or otherwise json_set-touched) is stored minified, so the store's
space-form ``LIKE '%"stashed": true%'`` queries silently stopped finding it —
stashed/excluded/audit rows vanished from get_stashed_messages / list_stashes /
get_excluded_messages / get_all_audit_failures. These run against a real SQLite
DB so the actual SQL is exercised.
"""
from __future__ import annotations

import pytest

from kestrel_sovereign.storage import AsyncStorage

AGENT_ID = "did:test:metadata-like-minified"


async def _minify_metadata(storage, msg_id):
    """Rewrite a row's metadata via json_set exactly like _set_decay_protected —
    SQLite json_set() re-emits the whole object minified (no space after ':')."""
    await storage.db.execute_commit(
        "UPDATE conversation_history "
        "SET metadata = json_set(COALESCE(metadata, '{}'), "
        "'$.decay_protected', json('true')) "
        "WHERE id = ?",
        (msg_id,),
    )
    row = await storage.db.fetchone(
        "SELECT metadata FROM conversation_history WHERE id = ?", (msg_id,)
    )
    # Precondition: the pin really did minify the JSON (no ": " left).
    assert '": ' not in row[0], row[0]
    return row[0]


@pytest.mark.asyncio
async def test_stashed_message_found_after_json_set_minifies_metadata(tmp_path):
    async with AsyncStorage(str(tmp_path / "k.db"), agent_id=AGENT_ID) as storage:
        conv = storage.conversation
        await conv.add_conversation(
            "user", "stash me",
            metadata={"stashed": True, "stash_id": "s-1", "stash_name": "n"},
        )
        msg_id = (await storage.db.fetchone(
            "SELECT id FROM conversation_history WHERE agent_id = ?", (AGENT_ID,)
        ))[0]
        await _minify_metadata(storage, msg_id)

        # All three stash lookups must still find the minified row.
        assert [m["id"] for m in await conv.get_stashed_messages()] == [msg_id]
        assert [m["id"] for m in await conv.get_stashed_messages(stash_id="s-1")] == [msg_id]
        stashes = await conv.list_stashes()
        assert any(s["stash_id"] == "s-1" for s in stashes)


@pytest.mark.asyncio
async def test_excluded_message_found_after_json_set_minifies_metadata(tmp_path):
    async with AsyncStorage(str(tmp_path / "k.db"), agent_id=AGENT_ID) as storage:
        conv = storage.conversation
        await conv.add_conversation(
            "assistant", "old turn",
            metadata={"excluded_from_context": True},
        )
        msg_id = (await storage.db.fetchone(
            "SELECT id FROM conversation_history WHERE agent_id = ?", (AGENT_ID,)
        ))[0]
        await _minify_metadata(storage, msg_id)

        assert [m["id"] for m in await conv.get_excluded_messages()] == [msg_id]


@pytest.mark.asyncio
async def test_audit_failure_found_after_json_set_minifies_metadata(tmp_path):
    async with AsyncStorage(str(tmp_path / "k.db"), agent_id=AGENT_ID) as storage:
        conv = storage.conversation
        await conv.add_conversation(
            "assistant", "flagged", metadata={"audit_failure": True},
        )
        msg_id = (await storage.db.fetchone(
            "SELECT id FROM conversation_history WHERE agent_id = ?", (AGENT_ID,)
        ))[0]
        await _minify_metadata(storage, msg_id)

        # get_all_audit_failures returns {role, content, metadata} (no id).
        failures = await conv.get_all_audit_failures()
        assert len(failures) == 1
        assert failures[0]["content"] == "flagged"


@pytest.mark.asyncio
async def test_space_format_rows_still_found(tmp_path):
    """The dual-form match must not regress the ordinary (never-pinned,
    space-format) rows."""
    async with AsyncStorage(str(tmp_path / "k.db"), agent_id=AGENT_ID) as storage:
        conv = storage.conversation
        await conv.add_conversation(
            "user", "plain stash", metadata={"stashed": True, "stash_id": "s-2"},
        )
        assert len(await conv.get_stashed_messages()) == 1
        assert len(await conv.get_stashed_messages(stash_id="s-2")) == 1
