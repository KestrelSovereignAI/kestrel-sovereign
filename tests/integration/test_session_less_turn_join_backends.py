"""#3654 on both backends: a session-less turn is refused an autonomous chain.

The unit cases in ``tests/unit/test_session_less_turn_join.py`` cover the rule
in detail on SQLite. This pins the two cases that matter on PostgreSQL as well:
the store reads each row's role and metadata the same way on both, so the reply
to a wake keeps the turn out of the chain, the reply to a person's turn still
lets it in, and an opener stamped as neither (a legacy row) keeps it out.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pytest

from kestrel_sovereign.storage import AsyncStorage
from kestrel_sovereign.storage.session_id_column import is_stampable_session_id
from kestrel_sovereign.storage.turn_provenance import turn_opener_metadata

WAKE = {
    "sent_form": True,
    **turn_opener_metadata({"source": "restart.completed", "mode": "cognition"}),
}
PERSON = {"sent_form": True, **turn_opener_metadata(None)}
LEGACY = {"sent_form": True}
NOTICE = {"sent_form": True, "operator_signal": True}


@pytest.fixture
async def unexpected_model_services(monkeypatch):
    """Session provenance needs only the native store, not default LLM owners."""
    from kestrel_sovereign.llm.service import LLMService

    # These tests verify row/session semantics. Embeddings have their own
    # native-service fixtures; never start an ambient provider here.
    monkeypatch.setenv("KESTREL_DISABLE_CONVERSATION_EMBEDDINGS", "true")
    original = LLMService.__init__
    services = []

    def observe(service, *args, **kwargs):
        original(service, *args, **kwargs)
        services.append(service)

    monkeypatch.setattr(LLMService, "__init__", observe)
    try:
        yield services
    finally:
        for service in services:
            await service.close()
            await service.wait_for_close_completion()


def _minutes_ago(minutes: int) -> datetime:
    now = datetime.now(timezone.utc).replace(tzinfo=None, microsecond=0)
    return now - timedelta(minutes=minutes)


async def _turn(db, agent_id, session_id, opener, *, minutes_ago):
    for role, metadata, age in (
        ("user", opener, minutes_ago + 1),
        ("user", NOTICE, minutes_ago + 1),
        ("assistant", {}, minutes_ago),
    ):
        await db.execute(
            "INSERT INTO conversation_history "
            "(agent_id, role, content, metadata, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                agent_id,
                role,
                f"{role} row",
                json.dumps({**metadata, "session_id": session_id}),
                _minutes_ago(age),
            ),
        )


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_a_session_less_turn_skips_the_chain_and_joins_a_person(db_backend):
    agent_id = f"did:test:3654:{uuid4()}"
    storage = AsyncStorage(backend=db_backend, agent_id=agent_id)
    await storage.initialize()
    try:
        await _turn(storage.db, agent_id, "chain-S", WAKE, minutes_ago=4)

        resolved = await storage.resolve_session_id(None)
        assert resolved != "chain-S"
        assert is_stampable_session_id(resolved)

        await _turn(storage.db, agent_id, "human-H", PERSON, minutes_ago=2)

        assert await storage.resolve_session_id(None) == "human-H"

        await _turn(storage.db, agent_id, "legacy-L", LEGACY, minutes_ago=1)

        resolved = await storage.resolve_session_id(None)
        assert resolved not in ("legacy-L", "human-H", "chain-S")
        assert is_stampable_session_id(resolved)
    finally:
        await storage.close()


@pytest.mark.asyncio
@pytest.mark.dual_backend
@pytest.mark.parametrize("raw", ["{not json", "[]", '"human-H"', "42"])
async def test_an_unreadable_newest_row_files_the_next_turn_whole(db_backend, raw, unexpected_model_services):
    """The newest row's metadata names no session: a fresh one, never none.

    The session-less turn that follows (its opening row and the reply) lands
    whole in that fresh session, on either backend.
    """
    agent_id = f"did:test:3654:{uuid4()}"
    storage = AsyncStorage(backend=db_backend, agent_id=agent_id)
    await storage.initialize()
    try:
        await _turn(storage.db, agent_id, "human-H", PERSON, minutes_ago=3)
        await storage.db.execute(
            "INSERT INTO conversation_history "
            "(agent_id, role, content, metadata, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (agent_id, "assistant", "unreadable row", raw, _minutes_ago(1)),
        )
        [(unreadable,)] = await storage.db.fetchall(
            "SELECT MAX(id) FROM conversation_history WHERE agent_id = ?",
            (agent_id,),
        )

        resolved = await storage.resolve_session_id(None)
        assert resolved != "human-H"
        assert is_stampable_session_id(resolved)

        await storage.add_conversation("user", "operator turn", metadata=PERSON)
        await storage.add_conversation("assistant", "reply")

        rows = await storage.db.fetchall(
            "SELECT metadata FROM conversation_history "
            "WHERE agent_id = ? AND id > ? ORDER BY id",
            (agent_id, unreadable),
        )
        sessions = {json.loads(metadata).get("session_id") for (metadata,) in rows}
        assert len(rows) == 2 and len(sessions) == 1
        [session] = sessions
        assert session != "human-H" and is_stampable_session_id(session)
        assert not unexpected_model_services, (
            "session provenance created unrelated default LLM services/native writers"
        )
    finally:
        await storage.close()
