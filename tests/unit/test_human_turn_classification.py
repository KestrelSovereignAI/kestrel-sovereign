"""#3653: one definition of a human turn, shared by every reader.

The chat console auto-opens the newest conversation a person took part in. What
counts as a person's turn is decided once, by
:func:`~kestrel_sovereign.storage.session_grouping.is_human_turn`: the session
preview and its "Autonomous wake" label (#2947), the list's ``has_human_turn``,
and the session-less join rule of #3654 all read it. These pin the definition
and that the readers cannot drift from it.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from kestrel_sovereign.signals.sources.a2a_task_submitted import (
    SOURCE_NAME as A2A_TASK_SUBMITTED,
)
from kestrel_sovereign.signals.sources.channels import SOURCE_NAME as CHANNEL_MESSAGE
from kestrel_sovereign.storage.async_conversation_store import AsyncConversationStore
from kestrel_sovereign.storage.conversation_created_at import canonical_created_at
from kestrel_sovereign.storage.session_id_column import column_session_id
from kestrel_sovereign.storage.session_grouping import (
    PERSON_MESSAGE_WAKE_SOURCES,
    group_messages_into_sessions,
    inbound_message_preview,
    is_human_turn,
    summarize_sessions,
)
from kestrel_sovereign.storage.turn_provenance import (
    latest_turn_opened_by_person,
    opened_by_person,
    turn_opener_metadata,
)


def _wake(source):
    return {"signal_wake": {"source": source, "mode": "cognition"}}


#: ``(role, metadata, is a human turn, has positive evidence of a person)``.
CASES = [
    pytest.param("user", turn_opener_metadata(None), True, True, id="typed-person"),
    pytest.param("user", {"sent_form": True}, True, False, id="legacy-untagged"),
    pytest.param("user", {}, True, False, id="no-metadata"),
    pytest.param("user", _wake(CHANNEL_MESSAGE), True, True, id="channel-message"),
    pytest.param("user", _wake("talon.job_complete"), False, False, id="talon-wake"),
    pytest.param("user", _wake("restart.completed"), False, False, id="restart-wake"),
    pytest.param("user", _wake(A2A_TASK_SUBMITTED), False, False, id="a2a-inbound"),
    pytest.param("user", {"signal_wake": True}, False, False, id="sourceless-wake"),
    pytest.param(
        "user", {**turn_opener_metadata(None), **_wake("heartbeat")}, False, False,
        id="wake-over-person-stamp",
    ),
    pytest.param(
        "user", {"operator_signal": True}, False, False, id="operator-notice",
    ),
    pytest.param(
        "user", {"operator_signal": True, **turn_opener_metadata(None)}, False, False,
        id="operator-notice-stamped",
    ),
    pytest.param("assistant", turn_opener_metadata(None), False, False, id="assistant"),
    pytest.param("system", {}, False, False, id="system"),
]


@pytest.mark.parametrize("role, metadata, human, evidenced", CASES)
def test_what_counts_as_a_human_turn(role, metadata, human, evidenced):
    assert is_human_turn(role, metadata) is human


@pytest.mark.parametrize("role, metadata, human, evidenced", CASES)
def test_the_join_rule_demands_evidence_on_top_of_the_same_definition(
    role, metadata, human, evidenced,
):
    """#3654 joins only on positive evidence, but never on a row the list
    would not call a human turn: one definition, with a stricter reader."""
    assert opened_by_person(role, metadata) is evidenced
    if evidenced:
        assert human


def test_the_person_message_sources_are_the_channel_source():
    """Named in storage, which may not import the dispatcher; pinned here to
    the source that actually dispatches channel messages."""
    assert CHANNEL_MESSAGE in PERSON_MESSAGE_WAKE_SOURCES
    assert A2A_TASK_SUBMITTED not in PERSON_MESSAGE_WAKE_SOURCES


BASE = datetime(2026, 10, 15, 9, 0, 0)


def _row(i, role, content, metadata, *, minutes):
    return {
        "id": i,
        "role": role,
        "content": content,
        "metadata": {"session_id": "s1", **metadata},
        "created_at": BASE + timedelta(minutes=minutes),
    }


def test_the_preview_row_is_the_first_human_turn():
    """The preview is what the list's flag is read from, so it must be chosen
    by the same definition: a channel message is previewed, a wake is not."""
    sessions = group_messages_into_sessions([
        _row(1, "user", "[TALON_JOB_COMPLETE] ...", _wake("talon.job_complete"), minutes=0),
        _row(2, "user", "[budget notice]", {"operator_signal": True}, minutes=1),
        _row(3, "user", "[CHANNEL_MESSAGE] ...", _wake(CHANNEL_MESSAGE), minutes=2),
        _row(4, "user", "typed later", turn_opener_metadata(None), minutes=3),
    ])
    [session] = sessions
    assert session["preview_content"] == "[CHANNEL_MESSAGE] ..."
    assert session["preview_wake_source"] == "talon.job_complete"


def test_the_agent_facing_list_names_a_channel_message_for_how_it_arrived():
    """The channel row holds the wake template, not what the person wrote; the
    agent's own list titles it like the UI does instead of quoting it."""
    [summary] = summarize_sessions([
        _row(1, "user", "[CHANNEL_MESSAGE] A message arrived ...", _wake(CHANNEL_MESSAGE), minutes=0),
        _row(2, "assistant", "replied", {}, minutes=1),
    ])
    assert summary["preview"] == inbound_message_preview(CHANNEL_MESSAGE)
    assert summary["preview"] == "Inbound message — channel.message"


def test_the_agent_facing_list_still_labels_an_a2a_session_autonomous():
    [summary] = summarize_sessions([
        _row(1, "user", "[A2A_TASK_SUBMITTED] ...", _wake(A2A_TASK_SUBMITTED), minutes=0),
    ])
    assert summary["preview"] == "Autonomous wake — a2a.task_submitted"


# ── The session-less join reads the same definition ───────────────────────


AGENT = "did:test:3653"


@pytest.fixture
async def store(tmp_path, sqlite_database_factory):
    db = await sqlite_database_factory(tmp_path / "agent.db")
    return AsyncConversationStore(db, agent_id=AGENT)


async def _insert(store, role, session_id, metadata, *, minutes_ago):
    """A row as a writer files it, ``session_id`` column included."""
    created_at = canonical_created_at(
        datetime.now(timezone.utc) - timedelta(minutes=minutes_ago)
    )
    document = json.dumps({**metadata, "session_id": session_id})
    await store.db.execute_commit(
        "INSERT INTO conversation_history "
        "(agent_id, role, content, metadata, session_id, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (
            store.agent_id, role, f"{role} row", document,
            column_session_id(document), created_at,
        ),
    )


@pytest.mark.asyncio
async def test_a_session_less_turn_joins_a_conversation_a_channel_message_opened(store):
    """A person's channel message is a human turn, so the operator continuing
    within the gap from a pane with no session joins it, as #3654 keeps for any
    conversation a person is having. The wake tag is the evidence."""
    await _insert(store, "user", "telegram", _wake(CHANNEL_MESSAGE), minutes_ago=5)
    await _insert(store, "assistant", "telegram", {}, minutes_ago=4)

    assert await store.resolve_session_id(None) == "telegram"


@pytest.mark.asyncio
async def test_a_session_less_turn_still_never_joins_an_a2a_chain(store):
    await _insert(store, "user", "peer", _wake(A2A_TASK_SUBMITTED), minutes_ago=5)
    await _insert(store, "assistant", "peer", {}, minutes_ago=4)

    assert await store.resolve_session_id(None) != "peer"


def test_the_join_classifier_accepts_a_channel_message_as_the_latest_turn():
    rows = [
        ("assistant", {"session_id": "telegram"}),
        ("user", {"session_id": "telegram", **_wake(CHANNEL_MESSAGE)}),
    ]
    assert latest_turn_opened_by_person(rows, "telegram") is True


# ── A projection built under an older definition ──────────────────────────


def test_the_definition_is_part_of_what_the_projection_is_named_for():
    """The session projection stores which row its preview pick chose, so a
    projection built under an older definition of a human turn holds an answer
    this revision would not give. Its change triggers are named for the source
    of ``session_grouping`` (``_grouping_source``); the definition has to live
    there for a change to it to retire every stored projection."""
    import inspect

    from kestrel_sovereign.storage import conversation_sessions, session_grouping

    assert is_human_turn.__module__ == session_grouping.__name__
    source = conversation_sessions._grouping_source()
    assert inspect.getsource(is_human_turn) in source
    assert f"PERSON_MESSAGE_WAKE_SOURCES={PERSON_MESSAGE_WAKE_SOURCES!r}" in source


@pytest.mark.asyncio
async def test_an_upgraded_projection_counts_a_channel_message(tmp_path, monkeypatch):
    """Before #3653 no wake counted as a person's turn, so a projection built
    then records a channel-only session with no human turn. The upgraded
    release must not serve that answer: opening the database under the new
    definition retires the projection, and page one rebuilds it."""
    from kestrel_sovereign.storage import conversation_sessions, session_grouping
    from kestrel_sovereign.storage.async_database import AsyncDatabase
    from kestrel_sovereign.storage.conversation_sessions import (
        ConversationSessionProjection,
    )

    path = str(tmp_path / "upgrade.db")
    current = session_grouping.PERSON_MESSAGE_WAKE_SOURCES
    conversation_sessions._grouping_source.cache_clear()
    try:
        # The older release: no wake source carried a person's message.
        monkeypatch.setattr(session_grouping, "PERSON_MESSAGE_WAKE_SOURCES", frozenset())
        conversation_sessions._grouping_source.cache_clear()
        db = await AsyncDatabase.sqlite(path)
        try:
            store = AsyncConversationStore(db, agent_id=AGENT)
            await _insert(store, "user", "telegram", _wake(CHANNEL_MESSAGE), minutes_ago=5)
            await _insert(store, "assistant", "telegram", {}, minutes_ago=4)
            [built] = (await store.list_session_page(limit=10))["sessions"]
            assert built["preview_content"] is None, (
                "the older definition did not count the channel message, so "
                "this would pass without the upgrade it is about"
            )
            assert not await ConversationSessionProjection(db, AGENT).is_stale()
        finally:
            await db.close()

        # The upgraded release opens the same database.
        monkeypatch.setattr(session_grouping, "PERSON_MESSAGE_WAKE_SOURCES", current)
        conversation_sessions._grouping_source.cache_clear()
        db = await AsyncDatabase.sqlite(path)
        try:
            assert await ConversationSessionProjection(db, AGENT).is_stale(), (
                "a projection built under the older definition reported itself current"
            )
            store = AsyncConversationStore(db, agent_id=AGENT)
            [listed] = (await store.list_session_page(limit=10))["sessions"]
            assert listed["session_id"] == "telegram"
            assert listed["preview_content"] == "user row"
            assert listed["preview_metadata"]["signal_wake"]["source"] == CHANNEL_MESSAGE
        finally:
            await db.close()
    finally:
        monkeypatch.setattr(session_grouping, "PERSON_MESSAGE_WAKE_SOURCES", current)
        conversation_sessions._grouping_source.cache_clear()


# ── Looking for a human turn the session cache does not record ────────────


@pytest.mark.asyncio
async def test_the_history_search_reads_in_chunks_and_keeps_the_first_turn(
    store, monkeypatch,
):
    """Bounded reads, continued past a chunk, and the FIRST human turn of each
    session wins, as it does in the projection's own pick."""
    from kestrel_sovereign.storage import conversation_sessions

    monkeypatch.setattr(conversation_sessions, "CHUNK_ROWS", 2)
    for minutes_ago in (60, 59, 58):
        await _insert(store, "user", "late", _wake("talon.job_complete"), minutes_ago=minutes_ago)
    await _insert(store, "user", "early", turn_opener_metadata(None), minutes_ago=57)
    await _insert(store, "user", "early", turn_opener_metadata(None), minutes_ago=56)
    await _insert(store, "user", "late", _wake(A2A_TASK_SUBMITTED), minutes_ago=55)
    await _insert(store, "user", "late", _wake(CHANNEL_MESSAGE), minutes_ago=54)
    await _insert(store, "user", "late", turn_opener_metadata(None), minutes_ago=53)
    await _insert(store, "user", "never", _wake("heartbeat"), minutes_ago=52)
    ids = [
        row[0] for row in await store.db.fetchall(
            "SELECT id FROM conversation_history WHERE agent_id = ? ORDER BY id",
            (AGENT,),
        )
    ]

    found = await store._first_human_turns(["early", "late", "never"], above=0)

    assert found == {"early": ids[3], "late": ids[6]}


@pytest.mark.asyncio
async def test_the_search_reads_only_what_was_appended_while_the_cache_is_exact(
    store, monkeypatch,
):
    """A wake-only session has no human turn to point at, on every page. While
    nothing below the projection's frontier has moved, the cache is exact
    about those rows, so the search reads only the rows appended since rather
    than the session's whole history; once something there moves, it reads
    the whole of it."""
    await _insert(store, "user", "older", _wake("talon.job_complete"), minutes_ago=120)
    await _insert(store, "assistant", "older", {}, minutes_ago=119)
    await _insert(store, "user", "newer", _wake("talon.job_complete"), minutes_ago=60)
    await _insert(store, "assistant", "newer", {}, minutes_ago=59)
    first = await store.list_session_page(limit=1)
    frontier = (
        await store.db.fetchone(
            "SELECT MAX(id) FROM conversation_history WHERE agent_id = ?", (AGENT,)
        )
    )[0]

    searched = []
    original = store._first_human_turns

    async def recording(session_ids, *, above, agent_id=None):
        searched.append((sorted(session_ids), above))
        return await original(session_ids, above=above, agent_id=agent_id)

    monkeypatch.setattr(store, "_first_human_turns", recording)

    await store.list_session_page(limit=1, cursor=first["next_cursor"])
    assert searched == [(["older"], frontier)]

    await store.db.execute_commit(
        "UPDATE conversation_history SET deleted_at = ? "
        "WHERE agent_id = ? AND role = 'assistant' AND session_id = 'older'",
        (canonical_created_at(datetime.now(timezone.utc)), AGENT),
    )
    searched.clear()
    await store.list_session_page(limit=1, cursor=first["next_cursor"])
    assert searched == [(["older"], 0)], "a movement below the frontier"
