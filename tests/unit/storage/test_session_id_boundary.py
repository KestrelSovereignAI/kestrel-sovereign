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

        A later row that names the legacy key outright is a member however far
        away it sits and whatever lies between — that is the metadata half of
        the dual-scheme resolver, and the grouper coalesces such clusters back
        together by id.
        """
        legacy = await _insert(store, 0)
        await _insert(store, 5, session_id=UUID_A)
        resumed = await _insert(store, 10, session_id=str(legacy))

        rows = await store._get_session_messages(str(legacy), limit=50)
        assert sorted(r[0] for r in rows) == [legacy, resumed]

    @pytest.mark.asyncio
    async def test_two_stamped_sessions_stay_separate(self, store):
        """Unchanged behaviour, pinned because the walk now decides it too."""
        first = await _insert(store, 0, session_id=UUID_A)
        second = await _insert(store, 5, session_id=UUID_B)

        assert await store._get_complete_session_message_ids(UUID_A) == [first]
        assert await store._get_complete_session_message_ids(UUID_B) == [second]
