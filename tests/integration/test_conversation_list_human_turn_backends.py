"""#3653 on both backends: a person's turn the session cache has not seen counts.

The conversation list reads the #2959 session projection, which is only as
current as its last repair, and a continuation page does not repair at all.
Whether a session has a human turn is what the chat console's auto-open trusts,
so a session the projection records without one is looked up again in history:
in the rows above the frontier it accounted for while nothing below has moved,
and in the session's whole live history once something has. The endpoint-level
cases run on SQLite; these pin the store's reads of those rows on PostgreSQL as
well.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pytest

from kestrel_sovereign.storage import AsyncStorage
from kestrel_sovereign.storage.session_id_column import column_session_id
from kestrel_sovereign.storage.turn_provenance import turn_opener_metadata

WAKE = turn_opener_metadata({"source": "talon.job_complete", "mode": "cognition"})
PERSON = turn_opener_metadata(None)


def _minutes_ago(minutes: int) -> datetime:
    now = datetime.now(timezone.utc).replace(tzinfo=None, microsecond=0)
    return now - timedelta(minutes=minutes)


async def _row(db, agent_id, role, content, session_id, metadata, *, minutes_ago):
    document = json.dumps({**metadata, "session_id": session_id})
    await db.execute(
        "INSERT INTO conversation_history "
        "(agent_id, role, content, metadata, session_id, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (
            agent_id, role, content, document,
            column_session_id(document), _minutes_ago(minutes_ago),
        ),
    )


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_a_human_turn_above_the_projection_frontier_counts(db_backend):
    agent_id = f"did:test:3653:{uuid4()}"
    storage = AsyncStorage(backend=db_backend, agent_id=agent_id)
    await storage.initialize()
    try:
        db = storage.db
        await _row(db, agent_id, "user", "[wake]", "older-chain", WAKE, minutes_ago=180)
        await _row(db, agent_id, "assistant", "noted", "older-chain", {}, minutes_ago=179)
        await _row(db, agent_id, "user", "[wake]", "newer-chain", WAKE, minutes_ago=60)
        await _row(db, agent_id, "assistant", "noted", "newer-chain", {}, minutes_ago=59)

        # Page one repairs the projection over those rows.
        first = await storage.list_session_page(agent_id=agent_id, limit=1)
        [newest] = first["sessions"]
        assert newest["session_id"] == "newer-chain"
        assert newest["preview_content"] is None, "a wake-only chain has no human turn"
        assert first["next_cursor"]

        await _row(
            db, agent_id, "user", "hold that merge", "older-chain", PERSON,
            minutes_ago=10,
        )

        # A continuation page does not repair, so it serves the cached counts.
        second = await storage.list_session_page(
            agent_id=agent_id, limit=1, cursor=first["next_cursor"]
        )
        [session] = second["sessions"]
        assert session["session_id"] == "older-chain"
        assert session["message_count"] == 2, "the projection has not seen the turn"
        assert session["preview_content"] == "hold that merge"
    finally:
        await storage.close()


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_trashing_the_first_human_turn_leaves_the_others_counted(db_backend):
    agent_id = f"did:test:3653:{uuid4()}"
    storage = AsyncStorage(backend=db_backend, agent_id=agent_id)
    await storage.initialize()
    try:
        db = storage.db
        await _row(db, agent_id, "user", "a question", "operator", PERSON, minutes_ago=180)
        await _row(db, agent_id, "assistant", "an answer", "operator", {}, minutes_ago=179)
        await _row(db, agent_id, "user", "a follow-up", "operator", PERSON, minutes_ago=178)
        await _row(db, agent_id, "user", "[wake]", "chain", WAKE, minutes_ago=60)
        await _row(db, agent_id, "assistant", "noted", "chain", {}, minutes_ago=59)

        first = await storage.list_session_page(agent_id=agent_id, limit=1)
        assert first["next_cursor"]

        # Below the projection's frontier, and not an append.
        await db.execute(
            "UPDATE conversation_history SET deleted_at = ? "
            "WHERE agent_id = ? AND content = ?",
            (_minutes_ago(5), agent_id, "a question"),
        )

        second = await storage.list_session_page(
            agent_id=agent_id, limit=1, cursor=first["next_cursor"]
        )
        [session] = second["sessions"]
        assert session["session_id"] == "operator"
        assert session["message_count"] == 3, "the projection has not been repaired"
        assert session["preview_content"] == "a follow-up"
    finally:
        await storage.close()
