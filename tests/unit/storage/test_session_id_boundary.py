"""#3098: where a session ends, read from both sides.

A session's boundary is decided twice. ``group_messages_into_sessions`` walks a
transcript forward and says where one conversation stops and the next begins;
``AsyncConversationStore._filter_session_rows`` takes one of the ids that walk
produced and resolves it back to its rows, for display and for delete/purge.
The two are separate implementations of one rule, and this file is about the
place they disagreed.

The resolver's forward walk did not stop when a row said it belonged somewhere
else. So the grouper could not stop either: had it split a stamped row out of
the legacy cluster beside it, deleting the legacy session the list showed would
have reached through the split and destroyed the stamped session too (#2019).
The grouping was made to lie about where the session ended so that delete would
stay scoped to what the list showed.

The cost of that lie was paid by Phase A's ``session_id`` column, which is
derived per row from that row's own metadata and cannot see a neighbour. It
filed the absorbed row under the id the grouping denied it, ``project_transcript``
found a stamped row grouped elsewhere, refused to guess — and the whole
conversation vanished from the list.

So the fix is on the side the premise came from: the walk stops at a row filed
under a different canonical id. Both halves of this file matter. The first says
the conversation is listed; the second says delete is still scoped to it, which
is the property the absorption was protecting and the reason it cannot simply
be dropped from the grouper alone.
"""

from __future__ import annotations

import json
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from kestrel_sovereign.storage.async_conversation_store import AsyncConversationStore
from kestrel_sovereign.storage.async_database import AsyncDatabase
from kestrel_sovereign.storage.session_grouping import (
    canonical_session_id,
    group_messages_into_sessions,
)
from kestrel_sovereign.storage.session_id_column import column_session_id

AGENT = "did:test:session-boundary"
BASE = datetime(2026, 6, 1, 9, 0, 0)
UUID_A = "8f1d1c62-9b0e-4b2c-9a1d-000000000001"
UUID_B = "8f1d1c62-9b0e-4b2c-9a1d-000000000002"


def _stamp(minute: int) -> str:
    return (BASE + timedelta(minutes=minute)).strftime("%Y-%m-%d %H:%M:%S")


@pytest.fixture
async def store():
    with tempfile.TemporaryDirectory() as tmp:
        db = await AsyncDatabase.sqlite(str(Path(tmp) / "boundary.db"))
        yield AsyncConversationStore(db, agent_id=AGENT)
        await db.close()


async def _insert(store, minute: int, session_id=None, **extra):
    """One live row, written the way the store writes one.

    The column is stamped through ``column_session_id`` rather than by hand so
    the fixture cannot manufacture the disagreement the tests are about.
    """
    metadata = dict(extra)
    if session_id is not None:
        metadata["session_id"] = session_id
    metadata_json = json.dumps(metadata)
    await store.db.execute(
        "INSERT INTO conversation_history "
        "(agent_id, role, content, metadata, session_id, created_at) "
        "VALUES (?,?,?,?,?,?)",
        (
            AGENT,
            "user",
            f"turn at {minute}",
            metadata_json,
            column_session_id(metadata_json),
            _stamp(minute),
        ),
    )
    row = await store.db.fetchone(
        "SELECT id FROM conversation_history WHERE agent_id = ? "
        "ORDER BY id DESC LIMIT 1",
        (AGENT,),
    )
    return row[0]


def _msg(row_id, minute, session_id=None, **extra):
    metadata = dict(extra)
    if session_id is not None:
        metadata["session_id"] = session_id
    return {
        "id": row_id,
        "role": "user",
        "content": f"turn at {minute}",
        "metadata": metadata,
        "created_at": _stamp(minute),
    }


class TestTheGroupingSide:
    def test_a_stamped_row_beside_a_legacy_cluster_is_its_own_session(self):
        """The shape from the ticket: two rows five minutes apart, one stamped.

        This used to be one session keyed ``"1"`` holding both rows, which put
        the grouping in contradiction with row 2's own column.
        """
        sessions = group_messages_into_sessions(
            [_msg(1, 0), _msg(2, 5, session_id=UUID_A)]
        )
        assert [(s["session_id"], s["message_count"]) for s in sessions] == [
            ("1", 1),
            (UUID_A, 1),
        ]

    def test_an_unlabeled_row_still_inherits_the_session_before_it(self):
        """Splitting on an id CHANGE is not splitting on the absence of one.

        A row carrying no id names no session of its own, so it belongs to
        whichever one it fell next to — including a stamped one. Only a
        different canonical id is a boundary.
        """
        sessions = group_messages_into_sessions(
            [_msg(1, 0, session_id=UUID_A), _msg(2, 5)]
        )
        assert [(s["session_id"], s["message_count"]) for s in sessions] == [
            (UUID_A, 2)
        ]

    def test_a_bare_integer_id_is_not_a_boundary(self):
        """Emma's 473 unstamped rows carry one of these, and they must not split.

        A bare integer in ``session_id`` names a ROW, not a session (#2012) —
        the list used to hand the UI a row id and the UI echoed it back. The
        column refuses such a value too, so treating it as a boundary here
        would re-create the very disagreement this ticket closes.
        """
        assert canonical_session_id({"session_id": "1314"}) is None
        sessions = group_messages_into_sessions(
            [_msg(1, 0), _msg(2, 5, session_id="1314")]
        )
        assert [(s["session_id"], s["message_count"]) for s in sessions] == [
            ("1", 2)
        ]


class TestTheResolverSide:
    """The half that makes the split above safe rather than merely tidy."""

    @pytest.mark.asyncio
    async def test_the_legacy_cluster_resolves_without_the_stamped_row(self, store):
        legacy = await _insert(store, 0)
        stamped = await _insert(store, 5, session_id=UUID_A)

        rows = await store._get_session_messages(str(legacy), limit=50)
        assert [r[0] for r in rows] == [legacy]
        assert stamped not in [r[0] for r in rows]

    @pytest.mark.asyncio
    async def test_delete_scope_matches_what_the_list_shows(self, store):
        """The #2019 property, asserted against the exact resolver.

        ``_get_complete_session_message_ids`` is what hard purge locks its row
        set from. If it still reached through the boundary, the split list
        would be an invitation to destroy the session beside the one deleted —
        which is precisely why the absorption existed.
        """
        legacy = await _insert(store, 0)
        stamped = await _insert(store, 5, session_id=UUID_A)

        assert await store._get_complete_session_message_ids(str(legacy)) == [legacy]
        assert await store._get_complete_session_message_ids(UUID_A) == [stamped]

    @pytest.mark.asyncio
    async def test_an_unlabeled_row_past_the_boundary_does_not_rejoin(self, store):
        """Ending the run is not the same as skipping one row.

        Row 3 carries no id and falls inside the gap from row 1, so a walk that
        merely skipped row 2 would take it back into the legacy cluster. The
        grouper gives it to row 2's session, and the two have to agree.
        """
        legacy = await _insert(store, 0)
        stamped = await _insert(store, 5, session_id=UUID_A)
        trailing = await _insert(store, 10)

        rows = await store._get_session_messages(str(legacy), limit=50)
        assert [r[0] for r in rows] == [legacy]

        sessions = group_messages_into_sessions(
            [_msg(legacy, 0), _msg(stamped, 5, session_id=UUID_A), _msg(trailing, 10)]
        )
        assert [(s["session_id"], s["message_count"]) for s in sessions] == [
            ("1", 1),
            (UUID_A, 2),
        ]

    @pytest.mark.asyncio
    async def test_an_explicit_resumption_reopens_the_run(self, store):
        """A boundary ends the implicit run; it does not end the session.

        A later row that names this session outright is a member however far
        away it sits and whatever lies between — that is the metadata half of
        the dual-scheme resolver, and the grouper coalesces such clusters back
        together by id.
        """
        first = await _insert(store, 0, session_id=UUID_A)
        await _insert(store, 5, session_id=UUID_B)
        resumed = await _insert(store, 10, session_id=UUID_A)

        rows = await store._get_session_messages(UUID_A, limit=50)
        assert sorted(r[0] for r in rows) == [first, resumed]
        assert await store._get_complete_session_message_ids(UUID_A) == [
            first, resumed,
        ]

    @pytest.mark.asyncio
    async def test_a_bare_integer_id_does_not_reopen_a_legacy_run(self, store):
        """A resumption is judged by the grouper's rule, not by string equality.

        A bare integer in ``session_id`` names a ROW (#2012), and the grouper
        reads such a row as unlabeled — so it belongs to whatever session it
        fell into, here the stamped one. Admitting it as a "resumption" of the
        legacy key filed it under a session the list showed it nowhere near,
        and deleting that legacy session would have taken a row displayed under
        another conversation with it.

        It also means a numeric session cannot be resumed at all, which is the
        truth rather than a restriction: its key IS its first row's id, and no
        second cluster can begin at that row.
        """
        legacy = await _insert(store, 0)
        stamped = await _insert(store, 5, session_id=UUID_A)
        echoed = await _insert(store, 10, session_id=str(legacy))

        assert await store._get_complete_session_message_ids(str(legacy)) == [legacy]
        sessions = group_messages_into_sessions(
            [
                _msg(legacy, 0),
                _msg(stamped, 5, session_id=UUID_A),
                _msg(echoed, 10, session_id=str(legacy)),
            ]
        )
        assert [(s["session_id"], s["message_count"]) for s in sessions] == [
            ("1", 1),
            (UUID_A, 2),
        ]

    @pytest.mark.asyncio
    async def test_a_canonical_session_owns_the_unlabeled_run_after_it(self, store):
        """Both resolvers, because only one of them can destroy anything.

        An unlabeled row names no session, so the grouper gives it to the one
        it fell after — including a stamped one. Resolving a canonical id by
        metadata alone therefore answered with a strict subset of what the list
        showed: a short count and transcript, and a hard purge that left the
        inherited row live to reappear under whatever session the reader then
        put it in.
        """
        stamped = await _insert(store, 0, session_id=UUID_A)
        inherited = await _insert(store, 5)

        rows = await store._get_session_messages(UUID_A, limit=50)
        assert sorted(r[0] for r in rows) == [stamped, inherited]
        assert await store._get_complete_session_message_ids(UUID_A) == [
            stamped, inherited,
        ]
        sessions = group_messages_into_sessions(
            [_msg(stamped, 0, session_id=UUID_A), _msg(inherited, 5)]
        )
        assert [(s["session_id"], s["message_count"]) for s in sessions] == [
            (UUID_A, 2)
        ]

    @pytest.mark.asyncio
    async def test_a_trashed_legacy_anchor_does_not_reach_the_session_after_it(
        self, store
    ):
        """The boundary applies to the FIRST candidate too, and this is why.

        A legacy anchor is looked up regardless of its state — its timestamp is
        needed to restore the session it owned — but a live read filters the
        row itself out. The next conversation is then the first candidate the
        walk sees, and exempting the first candidate from the boundary made a
        retry of the legacy session's DELETE soft-delete an unrelated
        conversation instead of returning zero.
        """
        legacy = await _insert(store, 0)
        stamped = await _insert(store, 5, session_id=UUID_A)
        await store.db.execute(
            "UPDATE conversation_history SET deleted_at = ? WHERE id = ?",
            ("2026-06-01 10:00:00", legacy),
        )

        rows = await store._get_session_messages(str(legacy), limit=50)
        assert [r[0] for r in rows] == []
        assert await store.delete_conversation_session(str(legacy)) == 0
        still_live = await store.db.fetchone(
            "SELECT deleted_at FROM conversation_history WHERE id = ?", (stamped,)
        )
        assert still_live[0] is None, "deleting the legacy session took another one"

    @pytest.mark.asyncio
    async def test_a_metadata_document_that_is_not_an_object_is_empty(self, store):
        """One legacy row must not take a whole conversation down with it.

        ``metadata`` is free text and legacy rows hold documents no parser can
        read — ``parse_message_metadata`` exists because three call sites had
        grown their own copy and disagreed about a document that parses to
        something other than an OBJECT. ``"[]"`` is valid JSON and is not
        metadata. This walk had a fourth copy, which handed a list to code that
        calls ``.get``, so every read, count and purge of the session that row
        fell in raised AttributeError.
        """
        stamped = await _insert(store, 0, session_id=UUID_A)
        await store.db.execute(
            "INSERT INTO conversation_history "
            "(agent_id, role, content, metadata, session_id, created_at) "
            "VALUES (?,?,?,?,?,?)",
            (AGENT, "user", "legacy blob", "[]", None, _stamp(5)),
        )
        inherited = (
            await store.db.fetchone(
                "SELECT id FROM conversation_history WHERE agent_id = ? "
                "ORDER BY id DESC LIMIT 1",
                (AGENT,),
            )
        )[0]

        rows = await store._get_session_messages(UUID_A, limit=50)
        assert sorted(r[0] for r in rows) == [stamped, inherited]
        assert await store._get_complete_session_message_ids(UUID_A) == [
            stamped, inherited,
        ]

    @pytest.mark.asyncio
    async def test_a_resumption_past_the_window_brings_its_tail(self, store):
        """A cap on the ANSWER is not a cap on where the answer may be found.

        The forward walk reads a window of rows, and a session resumed further
        away than one window used to lose the unlabeled replies that followed
        the resumption — the naming row itself came back from the metadata
        query, but the walk needs a contiguous run to decide where a session
        stops, and those rows were simply absent. The count beside it and the
        purge behind it are uncapped, so the transcript was the only one of the
        three that was short.

        Twelve rows of another conversation sit between, which is three times
        the window this ``limit`` opens.
        """
        first = await _insert(store, 0, session_id=UUID_A)
        for minute in range(1, 13):
            await _insert(store, minute, session_id=UUID_B)
        resumed = await _insert(store, 500, session_id=UUID_A)
        tail = await _insert(store, 501)

        rows = await store._get_session_messages(UUID_A, limit=5)
        assert sorted(r[0] for r in rows) == [first, resumed, tail]
        # The uncapped resolver is the authority the capped one must match.
        assert await store._get_complete_session_message_ids(UUID_A) == [
            first, resumed, tail,
        ]

    @pytest.mark.asyncio
    async def test_two_stamped_sessions_stay_separate(self, store):
        """Unchanged behaviour, pinned because the walk now decides it too."""
        first = await _insert(store, 0, session_id=UUID_A)
        second = await _insert(store, 5, session_id=UUID_B)

        assert await store._get_complete_session_message_ids(UUID_A) == [first]
        assert await store._get_complete_session_message_ids(UUID_B) == [second]
