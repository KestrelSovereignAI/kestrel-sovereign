"""#3061: a session's fold is trusted per session, not per agent.

``ConversationSessionProjection`` folds an append into the sessions it touched
— O(rows appended) — by reading each session's rows through the ``session_id``
column. A row filed under no session id cannot be read that way: it belongs to
whichever cluster it falls next to, and only the transcript says which.

The guard for that used to be ``_has_unstamped_rows()`` — does this AGENT hold
any such row — and if so the whole step re-derived the agent's entire live
history. That is a global answer to a local hazard, and legacy rows are not a
transient upgrade state: they are what old history looks like for ever, so one
of them made every repair pay for all of it, on the read path #2960 put it on.

The guard is the CHUNK now, and it asks two things. A chunk holding a row it
cannot file refuses to fold at all — so a walk that reaches its frontier without
escalating has accounted for every live row at or below it, and a fold only ever
ADDS a chunk's rows to what is already stored. And a chunk whose rows land at or
before the newest unstamped row refuses too, because a row arriving BESIDE one
can take it and no column read can see that happen.

An ordinary append satisfies both and folds; the legacy rows below keep the
answer the transcript already gave them.

Measured, SQLite, one session in three legacy, ``list_session_page(50)`` timed
straight after one appended row:

=========  ==========  ==========
live rows  before      after
=========  ==========  ==========
1,500      18.0 ms     3.4 ms
15,000     133.0 ms    6.3 ms
=========  ==========  ==========

The tests below are about the two ways a local guard can be wrong: trusting a
fold that should not be trusted, and — the one that actually happened while
writing it — silently dropping the rows it cannot file.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime, timedelta

import pytest

from kestrel_sovereign.kestrel_config.constants import SESSION_GAP_MINUTES
from kestrel_sovereign.storage.async_database import AsyncDatabase
from kestrel_sovereign.storage.conversation_sessions import (
    ConversationSessionProjection,
)

AGENT = "did:test:legacy-fold"
BASE = datetime(2026, 6, 1, 9, 0, 0)

PRE = """
CREATE TABLE conversation_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    agent_id TEXT NOT NULL DEFAULT '',
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    metadata TEXT,
    created_at TIMESTAMP,
    deleted_at TIMESTAMP DEFAULT NULL,
    archived_at TIMESTAMP DEFAULT NULL
);
"""


def _stamp(minute):
    return (BASE + timedelta(minutes=minute)).strftime("%Y-%m-%d %H:%M:%S")


def _seed(path, rows):
    """A pre-#2958 database: no ``session_id`` column, so the boot ALTER
    backfills it from metadata exactly as it does for a real upgrade."""
    conn = sqlite3.connect(str(path))
    try:
        conn.executescript(PRE)
        for session_id, minute in rows:
            metadata = "{}" if session_id is None else json.dumps(
                {"session_id": session_id}
            )
            conn.execute(
                "INSERT INTO conversation_history "
                "(agent_id, role, content, metadata, created_at) VALUES (?,?,?,?,?)",
                (AGENT, "user", f"turn at {minute}", metadata, _stamp(minute)),
            )
        conn.commit()
    finally:
        conn.close()


def _counting_transcript_passes(monkeypatch):
    passes = []
    original = ConversationSessionProjection._rebuild_from_transcript

    async def counting(self):
        passes.append(self.agent_id)
        return await original(self)

    monkeypatch.setattr(
        ConversationSessionProjection, "_rebuild_from_transcript", counting
    )
    return passes


async def _append_unstamped(db, minute):
    """A row that names no session, written after the database was opened.

    Which is how one actually arrives once #3120's pass has run over a
    history: the write path derives an implicit session id and returns ``None``
    if that derivation raises, so a row can still land unlabeled beside a
    session. The frontier these cases are about is what covers it.
    """
    await db.execute(
        "INSERT INTO conversation_history "
        "(agent_id, role, content, metadata, created_at, session_id) "
        "VALUES (?, 'user', 'new turn', '{}', ?, NULL)",
        (AGENT, _stamp(minute)),
    )


async def _append(db, session_id, minute):
    await db.execute(
        "INSERT INTO conversation_history "
        "(agent_id, role, content, metadata, created_at, session_id) "
        "VALUES (?, 'user', 'new turn', ?, ?, ?)",
        (AGENT, json.dumps({"session_id": session_id}), _stamp(minute), session_id),
    )


class TestTheAcceptance:
    @pytest.mark.asyncio
    async def test_an_append_beside_legacy_rows_does_not_re_derive_them(
        self, tmp_path, monkeypatch
    ):
        """The ticket, counted on the thing that costs.

        ``RepairOutcome.kind`` is the wrong gauge — a generation rotation
        reports ``rebuilt`` from the CHUNKED path, which is bounded — so this
        counts whole-transcript passes instead.
        """
        path = tmp_path / "append.db"
        _seed(path, [(None, minute) for minute in range(6)])

        passes = _counting_transcript_passes(monkeypatch)
        db = await AsyncDatabase.sqlite(str(path))
        try:
            projection = ConversationSessionProjection(db, AGENT)
            await projection.repair()
            assert [r["session_id"] for r in await projection.list()] == ["1"]

            # The first append rotates the ledger generation, which is a rebuild
            # by any design; the ones after it are the case the ticket is about.
            await _append(db, "sess-new", 500)
            await projection.repair()
            settled = len(passes)

            for turn in range(3):
                await _append(db, "sess-new", 600 + turn)
                await projection.repair()
            assert len(passes) == settled, (
                f"{len(passes) - settled} appends re-derived the whole "
                "transcript beside untouched legacy rows"
            )
            assert {r["session_id"] for r in await projection.list()} == {
                "1", "sess-new",
            }
        finally:
            await db.close()

    @pytest.mark.asyncio
    async def test_a_wholly_unstamped_session_is_still_listed(
        self, tmp_path, monkeypatch, caplog
    ):
        """The defect the first version of this fix had.

        A fold reads rows by their ``session_id`` column, so a row without one
        is a row it cannot file. Skipping it was safe only while the step had
        already refused to reach the fold with any unstamped row anywhere; with
        the guard per session, skipping means a conversation that exists in
        history and in no projection. Measured while writing this: six legacy
        rows and an empty list.
        """
        path = tmp_path / "wholly.db"
        _seed(path, [(None, minute) for minute in range(4)])

        passes = _counting_transcript_passes(monkeypatch)
        db = await AsyncDatabase.sqlite(str(path))
        try:
            projection = ConversationSessionProjection(db, AGENT)
            with caplog.at_level(logging.ERROR):
                await projection.repair()
            listed = {r["session_id"]: r["message_count"] for r in await projection.list()}
            assert listed == {"1": 4}
            assert passes == [AGENT], (
                "the chunk folded rows it could not file rather than escalating"
            )
            # ...and escalated on purpose rather than by falling into the
            # violation branch. Filing a NULL column under the string "None"
            # also reaches the transcript, by way of `project_transcript`
            # refusing a session the grouping does not show — the same outcome
            # logged as a Phase A violation that has not happened.
            assert not [
                record for record in caplog.records
                if record.levelno >= logging.ERROR
            ], "an expected state was reported as a contract violation"
        finally:
            await db.close()


class TestWhatTheInvariantBuys:
    @pytest.mark.asyncio
    async def test_an_append_landing_beside_an_unstamped_row_escalates(
        self, tmp_path, monkeypatch
    ):
        """A row can arrive BESIDE an unstamped one and take it (codex R2 P1).

        Ordinary appends land after everything and cannot. A row whose id is
        higher but whose stamp is earlier can, and that is not hypothetical:
        PostgreSQL's ``NOW()`` is transaction-start time, so an overlapping
        writer commits a later id carrying an earlier timestamp, and an import
        or a restore can write anything.

        The unstamped row is not in this chunk and no column read can see it
        move. Measured without the frontier: stored ``sess-a=2, sess-b=1`` under
        a watermark reporting itself current, where the reader says
        ``sess-a=1, sess-b=2``.

        The projection must be settled first — the first append after a build
        rotates the ledger generation and walks from the floor, which escalates
        for a different reason and would make this pass without testing
        anything. A per-session guard was tried and survived its own mutant
        precisely because no test here could construct an inversion.
        """
        path = tmp_path / "inversion.db"
        _seed(path, [("sess-a", 0)])

        passes = _counting_transcript_passes(monkeypatch)
        db = await AsyncDatabase.sqlite(str(path))
        try:
            await _append_unstamped(db, 2)
            projection = ConversationSessionProjection(db, AGENT)
            await projection.repair()
            await _append(db, "sess-z", 2000)
            await projection.repair()
            assert {
                r["session_id"]: r["message_count"] for r in await projection.list()
            } == {"sess-a": 2, "sess-z": 1}
            settled = len(passes)

            # Higher id, earlier stamp: it lands between sess-a and the row only
            # the transcript can attribute.
            await _append(db, "sess-b", 1)
            await projection.repair()
            assert len(passes) > settled, "the chunk folded beside an unstamped row"
            listed = {
                r["session_id"]: r["message_count"] for r in await projection.list()
            }
            assert listed == {"sess-a": 1, "sess-b": 2, "sess-z": 1}, listed
        finally:
            await db.close()

    @pytest.mark.asyncio
    async def test_a_fold_beside_an_unstamped_row_keeps_the_count_it_was_given(
        self, tmp_path, monkeypatch
    ):
        """A fold ADDS; it never recounts what is already stored.

        This session has an id the column holds AND an unstamped row inside its
        own span — the row carries no id of its own, so grouping attributes it
        here, and no column read can see it. The transcript derives it once;
        every later append folds on top of that answer rather than replacing it
        with what the column alone can see.

        Which is why the escalation lives at the CHUNK and not per session: a
        per-session guard was tried here and had nothing left to catch.
        """
        path = tmp_path / "straddle.db"
        _seed(path, [("sess-a", 0), ("sess-a", 1)])

        passes = _counting_transcript_passes(monkeypatch)
        db = await AsyncDatabase.sqlite(str(path))
        try:
            await _append_unstamped(db, 2)
            projection = ConversationSessionProjection(db, AGENT)
            await projection.repair()
            assert passes == [AGENT], "the unstamped row was folded, not escalated"
            assert {
                r["session_id"]: r["message_count"] for r in await projection.list()
            } == {"sess-a": 3}

            # Settle first. The first append after a build rotates the ledger
            # generation, which discards and walks from the floor — over the
            # unstamped row, so it escalates. Asserting on THAT step would be
            # asserting on the transcript pass, not on a fold; the mutant that
            # made a fold overwrite instead of adding survived exactly that.
            # Well past the grouping gap: a row inside it is absorbed by the
            # cluster beside it, so the frontier carries the gap and such a
            # chunk escalates (see TestTheFrontiersEdges).
            await _append(db, "sess-a", 2000)
            await projection.repair()
            settled = len(passes)

            await _append(db, "sess-a", 2001)
            await projection.repair()
            assert len(passes) == settled, "this step escalated, so it folded nothing"
            listed = {
                r["session_id"]: r["message_count"] for r in await projection.list()
            }
            assert listed == {"sess-a": 5}, (
                f"{listed}: the fold recounted the session from its column and "
                "lost the row only the transcript can attribute"
            )
        finally:
            await db.close()


@pytest.mark.asyncio
async def test_a_stamped_row_beside_a_legacy_cluster_still_lists(tmp_path):
    """#3098, which this ticket's frontier hands over to.

    Two live rows five minutes apart, one stamped. ``project_transcript`` used
    to find the stamped row grouped under the legacy cluster's numeric anchor
    while its own column said otherwise, refuse to guess, and drop BOTH — the
    conversation list came back empty, reproduced on main.

    The disagreement was the grouper's, and the grouper's reason was the
    resolver's: a legacy cluster absorbed the stamped row because
    ``_filter_session_rows`` walked straight through it, so splitting the list
    would have let deleting the legacy session destroy the stamped one too
    (#2019). #3098 stopped the walk; both sessions are now listed as
    themselves. See ``test_session_id_boundary.py`` for the pair of them.
    """
    path = tmp_path / "absorbed.db"
    _seed(path, [(None, 0), ("8f1d1c62-9b0e-4b2c-9a1d-000000000001", 5)])

    db = await AsyncDatabase.sqlite(str(path))
    try:
        projection = ConversationSessionProjection(db, AGENT)
        await projection.repair()
        listed = {r["session_id"]: r["message_count"] for r in await projection.list()}
        assert listed == {"1": 1, "8f1d1c62-9b0e-4b2c-9a1d-000000000001": 1}
    finally:
        await db.close()


class TestTheFrontierIsExact:
    """The frontier is one stamp, and #3098 is why it can be.

    It used to run a grouping gap past the newest unstamped row and then
    further still, along a walk of the cluster that row sat in, because a
    legacy cluster ABSORBED following stamped rows and so extended
    transitively. That absorption is gone. Every live row above the newest
    unstamped one carries a column, a column is a canonical id, and a canonical
    id now starts its own session — so the cluster ends AT that row and the
    walk could only ever have handed it back.

    What remains is the claim itself: a chunk landing after the newest
    unstamped row is one the column can answer alone. The grouper is a
    left-to-right fold, so an unstamped row's session is decided by what
    precedes it and a later row cannot move it; the one thing a later row can
    do is resume a session by naming it, and coalescing only ADDS, which is
    what a fold does.
    """

    @pytest.mark.asyncio
    async def test_a_chunk_after_the_newest_unstamped_row_folds(
        self, tmp_path, monkeypatch
    ):
        """The tightening, stated as the case that used to escalate.

        A stamped row five minutes after the legacy one no longer needs the
        transcript — and the answer it folds to is the reader's, which is the
        half that makes the fold legitimate rather than merely cheap.
        """
        path = tmp_path / "gap.db"
        _seed(path, [(None, 0)])

        passes = _counting_transcript_passes(monkeypatch)
        db = await AsyncDatabase.sqlite(str(path))
        try:
            projection = ConversationSessionProjection(db, AGENT)
            await projection.repair()
            await _append(db, "sess-z", 43200)
            await projection.repair()
            settled = len(passes)

            await _append(db, "sess-b", 5)  # inside the old gap window
            await projection.repair()
            assert len(passes) == settled, "this step escalated, so it folded nothing"
            listed = {
                r["session_id"]: r["message_count"] for r in await projection.list()
            }
            assert listed == {"1": 1, "sess-b": 1, "sess-z": 1}, listed
        finally:
            await db.close()

    @pytest.mark.asyncio
    async def test_a_chunk_sharing_the_frontier_stamp_escalates(
        self, tmp_path, monkeypatch
    ):
        """``<=``, not ``<``, and the tie is the reason.

        ``created_at`` is stored to the second, so a row landing in the same
        second as the newest unstamped one is ordinary rather than exotic — and
        canonical order breaks that tie by id, which an import or an
        overlapping PostgreSQL writer may set either way. A row that sorts
        BEFORE the unstamped one takes it, and no column read can see that, so
        the boundary stamp itself is inside the fence.
        """
        path = tmp_path / "tie.db"
        _seed(path, [("sess-a", 0)])

        passes = _counting_transcript_passes(monkeypatch)
        db = await AsyncDatabase.sqlite(str(path))
        try:
            await _append_unstamped(db, 2)
            projection = ConversationSessionProjection(db, AGENT)
            await projection.repair()
            await _append(db, "sess-z", 2000)
            await projection.repair()
            settled = len(passes)

            await _append(db, "sess-b", 2)  # the same second as the unstamped row
            await projection.repair()
            assert len(passes) > settled, "the chunk folded on the frontier itself"
        finally:
            await db.close()


@pytest.mark.asyncio
async def test_a_stamped_chain_no_longer_extends_the_fence(tmp_path):
    """Unstamped at 0, stamped at 20, stamped at 40 is three sessions, not one.

    This was the case the reach walk existed for: each adjacent gap is under
    thirty minutes, and a legacy cluster used to absorb stamped rows, so the
    reader called all three one conversation and a frontier of "newest
    unstamped plus one gap" would have folded minute 40 under its own id.

    #3098 made each stamped row its own session, so there is no chain to
    follow. The assertion is the whole listing rather than an absence: the
    projection and the reader now agree outright, where this test previously
    could only say the fold had not invented a session and xfail the rest.
    """
    path = tmp_path / "chain.db"
    _seed(path, [(None, 0)])

    db = await AsyncDatabase.sqlite(str(path))
    try:
        projection = ConversationSessionProjection(db, AGENT)
        await projection.repair()
        await _append(db, "sess-x", 20)
        await projection.repair()
        await _append(db, "sess-z", 43200)
        await projection.repair()
        await _append(db, "sess-y", 40)
        await projection.repair()

        listed = {r["session_id"]: r["message_count"] for r in await projection.list()}
        assert listed == {"1": 1, "sess-x": 1, "sess-y": 1, "sess-z": 1}, listed
    finally:
        await db.close()
