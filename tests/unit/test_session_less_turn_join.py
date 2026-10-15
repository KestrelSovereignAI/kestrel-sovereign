"""#3654: a session-less turn never joins the agent's autonomous chain.

A turn that arrives without a session (a chat pane before it holds one, an API
caller that sends none) used to be filed by the 30-minute rule alone: the
session of the agent's newest row, if that row was under 30 minutes old. An
agent that chains its own work writes wake rows into its chain session every
few minutes (#3429: one session per autonomous chain), so the operator's turn
landed in the chain and every later turn followed it there.

The rule now looks at what the newest session's latest TURN was. The newest
row of a chain is usually the agent's reply to a wake, so the walk goes back to
the row that opened the turn and reads its stored metadata:

* a ``signal_wake`` tag: an autonomous chain, never joined;
* ``turn_opener: "person"``, which the agent stamps on every turn that is not
  a wake: a person's turn, joined as before;
* anything else (a legacy row carrying neither mark, unreadable metadata, no
  opener at all): no evidence, so a fresh session.

The agent's own writers are covered too: a real agent stamps the turns it
opens, including the ones bootstrap persists while discovery is running.

Every row here is inserted with an explicit ``created_at`` a few minutes in the
past, so the 30-minute window is open without any test waiting on a clock.
"""

from __future__ import annotations

import json
import logging
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock

import httpx
import pytest

from kestrel_sovereign.bootstrap import BootstrapState
from kestrel_sovereign.llm.adapter import LLMResponse
from kestrel_sovereign.llm.output_ceiling import attach_stop_reason
from kestrel_sovereign.storage.async_conversation_store import AsyncConversationStore
from kestrel_sovereign.storage.conversation_created_at import canonical_created_at
from kestrel_sovereign.storage.session_id_column import (
    is_stampable_session_id,
    new_session_id,
)
from kestrel_sovereign.storage.turn_provenance import (
    LATEST_TURN_SCAN_ROWS,
    latest_turn_opened_by_person,
    opens_turn,
    turn_opener_metadata,
)
from tests.shared.booted_agent import booted_agent

AGENT = "did:test:3654"
CHAIN = "chain-session-S"
HUMAN = "human-session-H"
OTHER = "other-session-X"

WAKE_TAG = {"source": "talon.job_complete", "mode": "cognition"}
WAKE = {"sent_form": True, **turn_opener_metadata(WAKE_TAG)}
PERSON = {"sent_form": True, **turn_opener_metadata(None)}
NOTICE = {
    "sent_form": True,
    "operator_signal": True,
    "operator_signal_sources": ["budget"],
    "operator_signal_fallback": True,
}


@pytest.fixture
async def store(tmp_path, sqlite_database_factory):
    db = await sqlite_database_factory(tmp_path / "agent.db")
    return AsyncConversationStore(db, agent_id=AGENT)


async def _insert(store, role, session_id, metadata=None, *, minutes_ago, raw=None):
    """Insert one row as stored, dated ``minutes_ago`` before now."""
    if raw is None:
        document = dict(metadata or {})
        if session_id is not None:
            document["session_id"] = session_id
        raw = json.dumps(document)
    created_at = canonical_created_at(
        datetime.now(timezone.utc) - timedelta(minutes=minutes_ago)
    )
    await store.db.execute_commit(
        "INSERT INTO conversation_history "
        "(agent_id, role, content, metadata, created_at) VALUES (?, ?, ?, ?, ?)",
        (store.agent_id, role, f"{role} row", raw, created_at),
    )


async def _wake_turn(store, session_id, *, minutes_ago):
    """A wake turn as KestrelAgent persists it: prompt, notice, reply."""
    await _insert(store, "user", session_id, WAKE, minutes_ago=minutes_ago + 1)
    await _insert(store, "user", session_id, NOTICE, minutes_ago=minutes_ago + 1)
    await _insert(store, "assistant", session_id, minutes_ago=minutes_ago)


async def _person_turn(store, session_id, *, minutes_ago):
    """A person's turn: what they sent, an injected notice, the reply."""
    await _insert(store, "user", session_id, PERSON, minutes_ago=minutes_ago + 1)
    await _insert(store, "user", session_id, NOTICE, minutes_ago=minutes_ago + 1)
    await _insert(store, "assistant", session_id, minutes_ago=minutes_ago)


async def _session_by_row(store):
    rows = await store.db.fetchall(
        "SELECT role, metadata FROM conversation_history "
        "WHERE agent_id = ? ORDER BY id",
        (AGENT,),
    )
    return [(role, json.loads(metadata).get("session_id")) for role, metadata in rows]


async def _assert_fresh(store, refused):
    resolved = await store.resolve_session_id(None)
    assert resolved != refused
    assert is_stampable_session_id(resolved)
    # Positive evidence, not a coin flip: another ask mints another session.
    assert await store.resolve_session_id(None) not in (refused, resolved)
    return resolved


# ── Autonomous chains are never joined ─────────────────────────────────────


@pytest.mark.asyncio
async def test_latest_row_a_wake_prompt_is_not_joined(store):
    """A wake prompt under 30 minutes old: the turn does NOT resolve to S."""
    await _insert(store, "user", CHAIN, WAKE, minutes_ago=5)

    await _assert_fresh(store, CHAIN)


@pytest.mark.asyncio
async def test_latest_row_the_reply_to_a_wake_is_not_joined(store):
    """The usual shape of a chain: its newest row is the agent's own reply.

    Classifying the literal newest row would see an assistant row and let the
    bug through; the turn it answers was a wake.
    """
    await _wake_turn(store, CHAIN, minutes_ago=4)
    [(newest_role, newest_session)] = (await _session_by_row(store))[-1:]
    assert (newest_role, newest_session) == ("assistant", CHAIN)

    await _assert_fresh(store, CHAIN)


@pytest.mark.asyncio
async def test_a_long_chain_with_a_person_far_back_is_not_joined(store):
    """Only the latest turn decides: an old human turn does not vouch for S."""
    await _person_turn(store, CHAIN, minutes_ago=20)
    await _wake_turn(store, CHAIN, minutes_ago=10)
    await _wake_turn(store, CHAIN, minutes_ago=3)

    await _assert_fresh(store, CHAIN)


@pytest.mark.asyncio
async def test_a_notice_newer_than_the_chain_does_not_vouch_for_it(store):
    """An operator notice opens nothing; the wake it was injected into does."""
    await _insert(store, "user", CHAIN, WAKE, minutes_ago=5)
    await _insert(store, "user", CHAIN, NOTICE, minutes_ago=4)

    await _assert_fresh(store, CHAIN)


# ── A person's conversation is joined exactly as before ────────────────────


@pytest.mark.asyncio
async def test_latest_row_a_person_turn_is_joined(store):
    """A human turn in H under 30 minutes old: the turn resolves to H."""
    await _insert(store, "user", HUMAN, PERSON, minutes_ago=5)

    assert await store.resolve_session_id(None) == HUMAN


@pytest.mark.asyncio
@pytest.mark.parametrize("trailing_role", ["assistant", "tool", "system"])
async def test_reply_tool_or_system_row_in_a_person_conversation_is_joined(
    store, trailing_role,
):
    """The newest row of H is not the person's, but the turn it belongs to is."""
    await _person_turn(store, HUMAN, minutes_ago=6)
    await _insert(store, trailing_role, HUMAN, minutes_ago=5)

    assert await store.resolve_session_id(None) == HUMAN


@pytest.mark.asyncio
async def test_a_person_turn_after_a_wake_in_the_same_session_is_joined(store):
    """A person who answered inside the chain owns its latest turn."""
    await _wake_turn(store, HUMAN, minutes_ago=10)
    await _person_turn(store, HUMAN, minutes_ago=3)

    assert await store.resolve_session_id(None) == HUMAN


@pytest.mark.asyncio
async def test_other_sessions_rows_do_not_decide(store):
    """Only the newest session's own rows are read for its latest turn."""
    await _insert(store, "user", HUMAN, PERSON, minutes_ago=10)
    await _wake_turn(store, CHAIN, minutes_ago=7)
    await _insert(store, "system", HUMAN, minutes_ago=5)

    assert await store.resolve_session_id(None) == HUMAN

    await _insert(store, "system", OTHER, minutes_ago=4)
    await _insert(store, "assistant", CHAIN, minutes_ago=3)

    await _assert_fresh(store, CHAIN)


@pytest.mark.asyncio
async def test_the_gap_still_ends_a_person_conversation(store):
    """Positive evidence adds a refusal; it never extends the 30 minutes."""
    await _insert(store, "user", HUMAN, PERSON, minutes_ago=31)

    await _assert_fresh(store, HUMAN)


# ── No evidence means a fresh session, not a guess from timing ─────────────


@pytest.mark.asyncio
async def test_an_unreadable_opener_is_not_joined(store):
    """A metadata document no parser reads might be H's latest opener.

    An older person's turn in H must not vouch for the turn the unreadable
    row may have opened.
    """
    await _insert(store, "user", HUMAN, PERSON, minutes_ago=7)
    await _insert(store, "user", None, raw="{not json", minutes_ago=6)
    await _insert(store, "assistant", HUMAN, minutes_ago=5)

    await _assert_fresh(store, HUMAN)


@pytest.mark.asyncio
async def test_a_session_with_no_opener_is_not_joined(store):
    """Replies and notices alone say nothing about who opened the turn."""
    await _insert(store, "user", HUMAN, NOTICE, minutes_ago=6)
    await _insert(store, "assistant", HUMAN, minutes_ago=5)

    await _assert_fresh(store, HUMAN)


#: Openers as they were stored before #3654, when nothing marked a person's
#: turn. Each is also exactly what a wake looked like when its tag was not
#: written: bootstrap persisted the wakes it handled with no metadata of its
#: own, and the store then added only the session.
LEGACY_OPENERS = {
    "bootstrap_row": {},
    "turn_body_row": {"sent_form": True},
    "privacy_stamped_row": {
        "sent_form": True,
        "privacy_mode": "normal",
        "timestamp": "2026-10-01T09:00:00+00:00",
    },
}


@pytest.mark.asyncio
@pytest.mark.parametrize("opener", LEGACY_OPENERS.values(), ids=LEGACY_OPENERS.keys())
async def test_a_legacy_untagged_opener_is_not_joined(store, opener):
    """An opener that records neither a wake nor a person proves nothing.

    It may be a person's turn written before the stamp existed, or a wake whose
    tag was never written. Reading the missing tag as "a person" would file an
    operator's turn into a chain whenever that tag was lost.
    """
    await _insert(store, "user", CHAIN, opener, minutes_ago=6)
    await _insert(store, "assistant", CHAIN, minutes_ago=5)

    await _assert_fresh(store, CHAIN)


@pytest.mark.asyncio
async def test_a_session_less_turn_after_a_legacy_opener_starts_fresh(store):
    """The write path refuses the same evidence ``resolve_session_id`` does."""
    await _insert(store, "user", CHAIN, LEGACY_OPENERS["turn_body_row"], minutes_ago=6)
    await _insert(store, "assistant", CHAIN, minutes_ago=5)

    await store.add_conversation("user", "operator turn", metadata=PERSON)

    [(_, opened)] = (await _session_by_row(store))[-1:]
    assert opened not in (CHAIN, None)


@pytest.mark.asyncio
async def test_an_opener_outside_the_scan_window_is_not_joined(store):
    """The walk is bounded; past the bound there is no evidence."""
    await _insert(store, "user", HUMAN, PERSON, minutes_ago=10)
    for _ in range(LATEST_TURN_SCAN_ROWS - 1):
        await _insert(store, "assistant", HUMAN, minutes_ago=5)

    assert await store.resolve_session_id(None) == HUMAN

    await _insert(store, "assistant", HUMAN, minutes_ago=5)

    await _assert_fresh(store, HUMAN)


# ── An unreadable newest row starts a fresh session ────────────────────────
#
# The newest row is read before any classification: it names the session a
# turn might join. A document that is not an object names none, so the turn
# starts fresh, exactly as after a legacy row with no session (#3654 review).
# It used to raise out of the derivation, which answered ``None``: the endpoint
# then ran the turn with no session, its opening row was filed under none, and
# its reply minted a session of its own.

#: Newest-row metadata, as stored, from which no session can be read.
UNREADABLE_NEWEST = {
    "malformed_json": "{not json",
    "list": json.dumps([{"session_id": HUMAN, **PERSON}]),
    "string": json.dumps(HUMAN),
    "number": "42",
}


def _is_minted(session_id):
    """A session the store minted: a UUID4, in its canonical spelling."""
    return (
        isinstance(session_id, str)
        and str(uuid.UUID(session_id)) == session_id
        and uuid.UUID(session_id).version == 4
    )


async def _sessions_after(db, row_id):
    """``(role, session_id)`` of every row written after ``row_id``."""
    rows = await db.fetchall(
        "SELECT role, metadata FROM conversation_history WHERE id > ? ORDER BY id",
        (row_id,),
    )
    return [(role, json.loads(metadata).get("session_id")) for role, metadata in rows]


def _derivation_failures(caplog):
    """WARNINGs from the derivation's exception handler."""
    return [
        record for record in caplog.records
        if record.levelno == logging.WARNING
        and "Implicit session derivation failed" in record.getMessage()
    ]


async def _newest_row_id(db):
    [(row_id,)] = await db.fetchall(
        "SELECT id FROM conversation_history ORDER BY id DESC LIMIT 1"
    )
    return row_id


@pytest.mark.asyncio
@pytest.mark.parametrize("role", ["user", "assistant"])
@pytest.mark.parametrize(
    "raw", UNREADABLE_NEWEST.values(), ids=UNREADABLE_NEWEST.keys()
)
async def test_an_unreadable_newest_row_resolves_to_a_fresh_session(
    store, raw, role, caplog,
):
    """Under 30 minutes old and over a person's turn, it still proves nothing.

    It is read as naming no session, not left to fail into the derivation's
    exception handler, which is there for failures nobody anticipated.
    """
    await _person_turn(store, HUMAN, minutes_ago=7)
    await _insert(store, role, None, raw=raw, minutes_ago=5)

    with caplog.at_level(logging.WARNING):
        assert _is_minted(await _assert_fresh(store, HUMAN))
    assert not _derivation_failures(caplog)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "raw", UNREADABLE_NEWEST.values(), ids=UNREADABLE_NEWEST.keys()
)
async def test_a_session_less_turn_after_an_unreadable_newest_row_is_one_session(
    store, raw, caplog,
):
    """The opening row starts a session and the reply follows it there."""
    await _person_turn(store, HUMAN, minutes_ago=7)
    await _insert(store, "assistant", None, raw=raw, minutes_ago=5)
    unreadable = await _newest_row_id(store.db)

    with caplog.at_level(logging.WARNING):
        await store.add_conversation("user", "operator turn", metadata=PERSON)
        await store.add_conversation("user", "notice", metadata=NOTICE)
        await store.add_conversation("assistant", "reply")
    assert not _derivation_failures(caplog)

    turn = await _sessions_after(store.db, unreadable)
    assert [role for role, _ in turn] == ["user", "user", "assistant"]
    [session] = {sid for _, sid in turn}
    assert _is_minted(session) and session != HUMAN


@pytest.mark.asyncio
@pytest.mark.parametrize("raw", UNREADABLE_NEWEST.values(), ids=UNREADABLE_NEWEST.keys())
async def test_a_row_outside_any_turn_after_an_unreadable_row_gets_a_session(store, raw):
    """The plain rule reads the same newest row, and is never left with none."""
    await _insert(store, "assistant", None, raw=raw, minutes_ago=5)
    unreadable = await _newest_row_id(store.db)

    await store.add_conversation("system", "notice", metadata={"event": "safe_mode"})

    [(role, session)] = await _sessions_after(store.db, unreadable)
    assert role == "system" and _is_minted(session)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "value", [7, ["s"], {"id": "s"}, True, "", "a\x00b", "x" * 4096],
    ids=["int", "list", "object", "bool", "empty", "nul", "oversized"],
)
async def test_a_newest_row_naming_no_usable_session_is_not_joined(store, value):
    """A ``session_id`` that cannot key a session is no session to join."""
    await _insert(
        store, "user", None, raw=json.dumps({"session_id": value, **PERSON}),
        minutes_ago=5,
    )

    resolved = await store.resolve_session_id(None)
    assert _is_minted(resolved)


@pytest.mark.asyncio
async def test_a_failed_lookup_still_files_the_turn_whole(store, monkeypatch, caplog):
    """A derivation that fails starts a fresh session rather than none.

    The opening row's lookup fails; the reply's succeeds and finds that row,
    so the two share the session the failure minted.
    """
    await _person_turn(store, HUMAN, minutes_ago=6)
    before = await _newest_row_id(store.db)
    fetchall = store.db.fetchall
    calls = {"n": 0}

    async def fail_once(sql, *args, **kwargs):
        if "ORDER BY id DESC LIMIT ?" in sql and calls["n"] == 0:
            calls["n"] += 1
            raise RuntimeError("db hiccup")
        return await fetchall(sql, *args, **kwargs)

    monkeypatch.setattr(store.db, "fetchall", fail_once)
    with caplog.at_level(logging.WARNING):
        await store.add_conversation("user", "operator turn", metadata=PERSON)
    monkeypatch.setattr(store.db, "fetchall", fetchall)
    await store.add_conversation("assistant", "reply")

    assert calls["n"] == 1
    turn = await _sessions_after(store.db, before)
    [session] = {sid for _, sid in turn}
    assert len(turn) == 2 and _is_minted(session) and session != HUMAN
    assert len(_derivation_failures(caplog)) == 1


# ── The other metadata reads on the resolution path ────────────────────────
#
# ``resolve_session_id`` with an explicit integer-shaped id looks the id up as
# a row and reads that row's metadata, then every row naming the marker's
# session. A document there that is not an object is no marker.


@pytest.mark.asyncio
@pytest.mark.parametrize("raw", UNREADABLE_NEWEST.values(), ids=UNREADABLE_NEWEST.keys())
async def test_an_integer_id_naming_an_unreadable_row_is_kept(store, raw):
    await _insert(store, "user", None, raw=raw, minutes_ago=5)
    row_id = await _newest_row_id(store.db)

    assert await store.resolve_session_id(str(row_id)) == str(row_id)


@pytest.mark.asyncio
async def test_a_non_object_row_naming_the_marker_session_declines_it(store, caplog):
    """The marker is not canonicalized onto when its ownership cannot be read.

    The row matches the ownership query's text pattern but is not an object.
    It is refused as unreadable, not left to an exception handler.
    """
    marker = new_session_id()
    await _insert(
        store, "user", marker, {"new_session": True}, minutes_ago=6,
    )
    marker_row = await _newest_row_id(store.db)
    await _insert(
        store, "user", None,
        raw=json.dumps([{"session_id": marker, "new_session": True}]),
        minutes_ago=5,
    )

    with caplog.at_level(logging.WARNING):
        assert await store.resolve_session_id(str(marker_row)) == str(marker_row)
    assert not [r for r in caplog.records if "Marker ownership" in r.getMessage()]

    await store.db.execute_commit(
        "DELETE FROM conversation_history WHERE id > ?", (marker_row,)
    )
    assert await store.resolve_session_id(str(marker_row)) == marker


# ── Session-less writes go through the same rule ───────────────────────────


@pytest.mark.asyncio
async def test_a_session_less_turn_after_a_wake_lands_whole_in_a_fresh_session(store):
    """``add_conversation`` with no session uses the same derivation.

    The prompt opens the turn and is refused the chain; the notice and the
    reply complete it and follow the prompt into its fresh session.
    """
    await _wake_turn(store, CHAIN, minutes_ago=4)

    await store.add_conversation("user", "from /v1/chat/completions", metadata=PERSON)
    await store.add_conversation("user", "notice", metadata=NOTICE)
    await store.add_conversation("assistant", "reply")

    rows = await _session_by_row(store)
    assert [sid for _, sid in rows[:3]] == [CHAIN] * 3
    turn = {sid for _, sid in rows[3:]}
    assert len(turn) == 1
    assert turn.pop() not in (CHAIN, None)


@pytest.mark.asyncio
async def test_a_session_less_turn_continues_a_person_conversation(store):
    """The unchanged half: within the gap a person's conversation is joined."""
    await _person_turn(store, HUMAN, minutes_ago=4)

    await store.add_conversation("user", "next question", metadata=PERSON)
    await store.add_conversation("assistant", "answer")

    assert {sid for _, sid in await _session_by_row(store)} == {HUMAN}


@pytest.mark.asyncio
async def test_a_row_outside_any_turn_keeps_the_plain_rule(store):
    """A session-less system notice is not a turn and still follows the newest row."""
    await _wake_turn(store, CHAIN, minutes_ago=4)

    await store.add_conversation(
        "system", "SAFE MODE ACTIVATED", metadata={"event": "safe_mode"},
    )

    assert (await _session_by_row(store))[-1] == ("system", CHAIN)


# ── The classifier itself ──────────────────────────────────────────────────


def test_classifier_reads_dict_and_text_metadata_alike():
    rows = [
        ("assistant", {"session_id": HUMAN}),
        ("user", json.dumps({"session_id": HUMAN, **PERSON})),
    ]
    assert latest_turn_opened_by_person(rows, HUMAN) is True
    assert latest_turn_opened_by_person(reversed(rows), HUMAN) is True


@pytest.mark.parametrize("raw", ["[]", '"text"', "null", "{broken"])
def test_classifier_refuses_metadata_that_is_not_an_object(raw):
    rows = [("user", raw), ("user", {"session_id": HUMAN, **PERSON})]
    assert latest_turn_opened_by_person(rows, HUMAN) is False


def test_classifier_skips_rows_with_no_metadata():
    rows = [("user", None), ("user", ""), ("user", {"session_id": HUMAN, **PERSON})]
    assert latest_turn_opened_by_person(rows, HUMAN) is True


def test_classifier_reads_a_wake_with_no_source_as_a_wake():
    rows = [("user", {"session_id": CHAIN, "signal_wake": True})]
    assert latest_turn_opened_by_person(rows, CHAIN) is False


def test_classifier_reads_a_wake_tag_over_a_person_stamp():
    """No writer sets both; if one did, the autonomous mark is the safe reading."""
    rows = [("user", {"session_id": CHAIN, **PERSON, "signal_wake": WAKE_TAG})]
    assert latest_turn_opened_by_person(rows, CHAIN) is False


@pytest.mark.parametrize("value", [True, "Person", "agent", None, 1])
def test_classifier_accepts_only_the_person_stamp(value):
    rows = [("user", {"session_id": HUMAN, "turn_opener": value})]
    assert latest_turn_opened_by_person(rows, HUMAN) is False


def test_opener_metadata_is_a_new_dict_each_time():
    """A writer adds its own keys to it; no two turns may share one."""
    first = turn_opener_metadata(None)
    first["session_id"] = HUMAN
    assert turn_opener_metadata(None) == {"turn_opener": "person"}
    assert turn_opener_metadata(WAKE_TAG) == {"signal_wake": WAKE_TAG}


# ── The agent stamps every turn it opens ───────────────────────────────────
#
# The store can only refuse what the writers mark. These run real turns on a
# booted agent, so a writer that stops stamping (or stamps a wake as a person)
# shows up as a session-less turn landing in the wrong session.


async def _scripted_reply(**_kwargs):
    return attach_stop_reason(LLMResponse(content="ack"), "stop")


async def _scripted_stream(**_kwargs):
    yield "ack"


async def _openers(agent):
    """The metadata of every row that opened a turn, oldest first."""
    rows = await agent._raw_storage.db.fetchall(
        "SELECT role, metadata FROM conversation_history ORDER BY id"
    )
    return [
        json.loads(metadata)
        for role, metadata in rows
        if opens_turn(role, json.loads(metadata))
    ]


@pytest.mark.asyncio
async def test_the_agent_stamps_wakes_and_persons_on_both_entry_points(
    tmp_path, monkeypatch,
):
    async with booted_agent(
        tmp_path, name="Kite", provenance="test:3654", features=frozenset()
    ) as agent:
        monkeypatch.setattr(agent.llm_service, "generate_with_messages", _scripted_reply)
        monkeypatch.setattr(
            agent.llm_service, "stream_with_tool_detection", _scripted_stream
        )

        # A wake, bound the way the dispatcher binds an unbound one (#3429).
        chain = new_session_id()
        await agent.process_input("wake prompt", session_id=chain, signal_wake=WAKE_TAG)
        operator = await _assert_fresh(agent.storage, chain)

        await agent.process_input("hello", session_id=operator)
        assert await agent.storage.resolve_session_id(None) == operator

        streamed = new_session_id()
        async for _chunk in agent.process_input_streaming(
            "and again", session_id=streamed
        ):
            pass
        assert await agent.storage.resolve_session_id(None) == streamed

        wake, person, streamed_person = await _openers(agent)

    assert wake["signal_wake"] == WAKE_TAG and "turn_opener" not in wake
    for row in (person, streamed_person):
        assert row["turn_opener"] == "person" and "signal_wake" not in row


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "state", [BootstrapState.PENDING, BootstrapState.DISCOVERY], ids=lambda s: s.value
)
async def test_bootstrap_stamps_the_turns_it_persists(tmp_path, monkeypatch, state):
    """A wake handled while discovery runs is still a wake (#3654 review).

    ``process_input`` hands a turn to ``_handle_bootstrap`` until onboarding
    completes, wakes included, and bootstrap persists that turn's user row
    itself. It must carry the wake tag through, or the chain's latest turn
    reads as a person's and the next session-less operator turn joins it.
    """
    async with booted_agent(
        tmp_path, name="Kite", provenance="test:3654", features=frozenset()
    ) as agent:
        monkeypatch.setattr(agent.llm_service, "generate_with_messages", _scripted_reply)
        bootstrap = agent.bootstrap_service
        monkeypatch.setattr(
            bootstrap,
            "generate_wake_up_message",
            AsyncMock(return_value="What should I call you?"),
        )
        monkeypatch.setattr(
            bootstrap,
            "process_discovery_message",
            AsyncMock(return_value=("Tell me more.", False, False)),
        )
        reply = {
            BootstrapState.PENDING: "What should I call you?",
            BootstrapState.DISCOVERY: "Tell me more.",
        }[state]

        await bootstrap.set_bootstrap_state(state)
        chain = new_session_id()
        assert await agent.process_input(
            "wake prompt", session_id=chain, signal_wake=WAKE_TAG
        ) == reply
        await _assert_fresh(agent.storage, chain)

        # The person's session-less turn through the same branch is refused
        # the chain, and its own session is then joined.
        await bootstrap.set_bootstrap_state(state)
        assert await agent.process_input("I'm Sam") == reply
        *_, wake, person = await _openers(agent)
        assert await agent.storage.resolve_session_id(None) == person["session_id"]

    assert wake["signal_wake"] == WAKE_TAG and wake["session_id"] == chain
    assert person["turn_opener"] == "person" and "signal_wake" not in person
    assert person["session_id"] != chain


@asynccontextmanager
async def _agent_doors(agent):
    """``/api/agent/invoke`` and ``/api/agent/stream`` for ``agent``, served
    on the test's own loop."""
    from fastapi import FastAPI

    from kestrel_sovereign.api_errors import register_api_error_handlers
    from kestrel_sovereign.endpoints.agent import router
    from kestrel_sovereign.rate_limit import limiter

    app = FastAPI()
    app.state.limiter = limiter
    app.state.agent = agent
    app.include_router(router)
    register_api_error_handlers(app)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test",
    ) as client:
        yield client


@pytest.mark.asyncio
@pytest.mark.usefixtures("isolated_process_rate_limiter")
@pytest.mark.parametrize("door", ["invoke", "stream"])
@pytest.mark.parametrize(
    "raw", UNREADABLE_NEWEST.values(), ids=UNREADABLE_NEWEST.keys()
)
async def test_a_door_files_the_turn_whole_under_the_session_it_returns(
    tmp_path, monkeypatch, door, raw,
):
    """The session a door answers with is the one the whole turn lands in.

    The newest row is unreadable, right after a person's turn. The door must
    still answer with a session, and the turn it runs (the operator's row and
    the agent's reply) must be filed whole under it.
    """
    async with booted_agent(
        tmp_path, name="Kite", provenance="test:3654", features=frozenset()
    ) as agent:
        monkeypatch.setattr(agent.llm_service, "generate_with_messages", _scripted_reply)
        monkeypatch.setattr(
            agent.llm_service, "stream_with_tool_detection", _scripted_stream
        )
        human = new_session_id()
        await agent.process_input("hello", session_id=human)
        conversation = agent._raw_storage.conversation
        await _insert(conversation, "assistant", None, raw=raw, minutes_ago=0)
        unreadable = await _newest_row_id(conversation.db)

        async with _agent_doors(agent) as client:
            response = await client.post(
                f"/api/agent/{door}", json={"input": "next question"},
            )
        assert response.status_code == 200, response.text
        if door == "invoke":
            returned = response.json()["session_id"]
        else:
            assert "ack" in response.text
            returned = response.headers.get("X-Session-Id")
        turn = await _sessions_after(conversation.db, unreadable)

    assert _is_minted(returned) and returned != human
    assert {"user", "assistant"} <= {role for role, _ in turn}
    assert {session for _, session in turn} == {returned}
