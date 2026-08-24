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
        _seed(path, [("sess-a", 0), (None, 2)])

        passes = _counting_transcript_passes(monkeypatch)
        db = await AsyncDatabase.sqlite(str(path))
        try:
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
        _seed(path, [("sess-a", 0), ("sess-a", 1), (None, 2)])

        passes = _counting_transcript_passes(monkeypatch)
        db = await AsyncDatabase.sqlite(str(path))
        try:
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
async def test_a_stamped_row_absorbed_into_a_legacy_cluster_still_lists(tmp_path):
    """#3098, pinned here because #3061 is what would have widened its reach.

    ``group_messages_into_sessions`` deliberately does not split a row carrying
    an explicit id out of a legacy numeric cluster (#2019): the resolver walks
    that cluster forward in time and does not stop on id changes, so a split
    list would let deleting the legacy session destroy the other one too. The
    column, derived per row from that row's own metadata, files it under the
    explicit id anyway — and ``project_transcript`` then drops the session as
    the Phase A violation it is.

    Reproduced on main: two live rows five minutes apart, one stamped, and the
    conversation list is EMPTY.

    This is xfail rather than absent because #3061 wanted the column's charset
    widened — `rasa_shim` files every SMS turn under `sms:{sender}`, which the
    column cannot hold, so those rows are unstamped for ever — and that widening
    turns this from unreachable into reachable for any agent with legacy history
    that starts receiving SMS. The test says which order the two must land in.
    """
    path = tmp_path / "absorbed.db"
    _seed(path, [(None, 0), ("8f1d1c62-9b0e-4b2c-9a1d-000000000001", 5)])

    db = await AsyncDatabase.sqlite(str(path))
    try:
        projection = ConversationSessionProjection(db, AGENT)
        await projection.repair()
        listed = {r["session_id"]: r["message_count"] for r in await projection.list()}
        if not listed:
            pytest.xfail("#3098: the absorbed row's column contradicts grouping")
        assert listed == {"1": 2}
    finally:
        await db.close()


class TestTheFrontiersEdges:
    @pytest.mark.asyncio
    async def test_a_chunk_inside_the_gap_window_escalates(self, tmp_path, monkeypatch):
        """The frontier carries the grouping gap, and that is not padding.

        A stamped row arriving within ``SESSION_GAP_MINUTES`` of the newest
        unstamped one is absorbed into that row's cluster by the grouper —
        deliberately (#2019), because the resolver walks a legacy numeric
        cluster forward in time and does not stop on id changes. A fold files it
        under its own column id instead. Measured without the extension: the
        reader saw one session of two rows and the projection stored two
        sessions of one, calling itself current.

        What the escalation hands over to is #3098: the transcript pass then
        drops that cluster, because the absorbed row's column contradicts the
        grouping. That is pre-existing and identical on main — asserted here as
        "the fold did not invent a different answer", which is this ticket's
        part of it.
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

            await _append(db, "sess-b", 5)  # above the frontier, inside the gap
            await projection.repair()
            assert len(passes) > settled, "a chunk inside the gap window folded"
            listed = {r["session_id"] for r in await projection.list()}
            assert "sess-b" not in listed, (
                f"{listed}: the fold listed a session the reader absorbs into "
                "the legacy cluster beside it"
            )
        finally:
            await db.close()


@pytest.mark.asyncio
async def test_a_legacy_cluster_reaching_past_the_gap_is_fenced(tmp_path):
    """A legacy cluster extends transitively, and the fence follows it.

    Unstamped at minute 0, stamped at 20, stamped at 40 is ONE cluster to the
    reader: each adjacent gap is under thirty minutes, and a legacy cluster
    absorbs stamped rows (#2019). A frontier of "newest unstamped row plus one
    gap" stops at minute 30 and would fold minute 40 under its own id.

    The reach is walked over history rather than read from the projection —
    which is where a cluster's end is written down, and is exactly what #3098
    makes unreadable, since the absorbed row at minute 20 causes the cluster to
    be refused and never stored.

    So the assertion is in two parts. The fold's part holds now: it invents no
    session the reader does not show. Full agreement is still #3098's, and the
    xfail below is that ticket's, not this one's — this branch and main produce
    the identical (wrong) answer for it.
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

        listed = {r["session_id"] for r in await projection.list()}
        assert "sess-y" not in listed, (
            f"{listed}: the fold listed a session the reader absorbs into the "
            "legacy cluster reaching past it"
        )
        if listed != {"1", "sess-z"}:
            pytest.xfail(f"#3098: the reader groups these as 1 + sess-z, not {listed}")
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_a_chain_longer_than_the_walk_refuses_rather_than_guesses(
    tmp_path, monkeypatch
):
    """The reach walk is bounded, and running out is an answer.

    A legacy cluster extends for as long as consecutive rows stay within the
    grouping gap, which nothing in the data bounds — so the walk does, and a
    chain it cannot see the end of returns a frontier nothing can be after
    rather than the last row it happened to reach. Guessing there would fold a
    row the reader puts inside the cluster.
    """
    monkeypatch.setattr(
        "kestrel_sovereign.storage.conversation_sessions.CLUSTER_REACH_ROWS", 2
    )
    path = tmp_path / "long-chain.db"
    _seed(path, [(None, 0)] + [(f"sess-{n}", n * 10) for n in range(1, 6)])

    passes = _counting_transcript_passes(monkeypatch)
    db = await AsyncDatabase.sqlite(str(path))
    try:
        projection = ConversationSessionProjection(db, AGENT)
        await projection.repair()
        await _append(db, "sess-far", 43200)
        await projection.repair()
        settled = len(passes)

        # Far past any real cluster, but the walk cannot prove where the chain
        # ends within two rows, so nothing may be folded.
        await _append(db, "sess-later", 43300)
        await projection.repair()
        assert len(passes) > settled, (
            "a chunk folded while the walk had not reached the end of the chain"
        )
    finally:
        await db.close()
