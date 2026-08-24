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

The guard is now a frontier. A session cannot hold a row standing after its own
last message, so a session that STARTS after every unstamped row provably holds
none and folds soundly. Everything else escalates, exactly as before.

Measured, SQLite, one session in three legacy, ``list_session_page(50)`` timed
straight after one appended row:

=========  =========  ==========  ==========
live rows  stamped    legacy old  legacy new
=========  =========  ==========  ==========
1,500      3.7 ms     18.0 ms     3.4 ms
15,000     4.9 ms     133.0 ms    6.3 ms
=========  =========  ==========  ==========

The tests below are about the two ways a local guard can be wrong: trusting a
fold that should not be trusted, and — the one that actually happened while
writing it — silently dropping the rows it cannot file.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta

import pytest

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
        self, tmp_path, monkeypatch
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
            await projection.repair()
            listed = {r["session_id"]: r["message_count"] for r in await projection.list()}
            assert listed == {"1": 4}
            assert passes == [AGENT], (
                "the chunk folded rows it could not file rather than escalating"
            )
        finally:
            await db.close()


class TestWhatTheFrontierRefuses:
    @pytest.mark.asyncio
    async def test_a_session_that_may_hold_an_unstamped_row_escalates(
        self, tmp_path, monkeypatch
    ):
        """The frontier is a bound on where an unstamped row can be, not a flag.

        This session has an id the column holds AND an unstamped row inside its
        own span — the row carries no id of its own, so grouping attributes it
        here. Folding by column would count three rows and store that under a
        watermark saying current. It starts before the frontier, so it is not
        provably whole, and the transcript is derived instead.
        """
        path = tmp_path / "straddle.db"
        _seed(path, [("sess-a", 0), ("sess-a", 1), (None, 2)])

        passes = _counting_transcript_passes(monkeypatch)
        db = await AsyncDatabase.sqlite(str(path))
        try:
            projection = ConversationSessionProjection(db, AGENT)
            await projection.repair()
            await _append(db, "sess-a", 3)
            await projection.repair()
            listed = {r["session_id"]: r["message_count"] for r in await projection.list()}
            assert listed == {"sess-a": 4}, (
                f"{listed}: the fold counted only the rows its column could see"
            )
            assert passes, "the straddling session was folded rather than escalated"
        finally:
            await db.close()

    @pytest.mark.asyncio
    async def test_an_unstamped_row_that_cannot_be_dated_stops_every_fold(
        self, tmp_path, monkeypatch
    ):
        """Nothing can be proved to stand after a row that stands nowhere.

        ``created_at`` is NOT NULL with a CHECK since #3009, so this state is
        only reachable from a database that predates it — which is exactly the
        database this ticket is about.
        """
        path = tmp_path / "undatable.db"
        _seed(path, [("sess-a", 0)])
        conn = sqlite3.connect(str(path))
        conn.execute(
            "INSERT INTO conversation_history "
            "(agent_id, role, content, metadata, created_at) VALUES (?,?,?,?,NULL)",
            (AGENT, "user", "undatable", "{}"),
        )
        conn.commit()
        conn.close()

        passes = _counting_transcript_passes(monkeypatch)
        db = await AsyncDatabase.sqlite(str(path))
        try:
            projection = ConversationSessionProjection(db, AGENT)
            await projection.repair()
            before = len(passes)
            await _append(db, "sess-b", 900)
            await projection.repair()
            assert len(passes) > before, (
                "a session was proved clear of a row that cannot be placed in time"
            )
        finally:
            await db.close()


class TestTheWidenedColumn:
    @pytest.mark.asyncio
    async def test_an_sms_conversation_folds_like_any_other(
        self, tmp_path, monkeypatch
    ):
        """``rasa_shim`` files every SMS turn under ``sms:{sender}``.

        A colon was outside the column's charset, so those rows were unstamped
        for ever — and with a per-agent guard that meant an SMS agent re-derived
        its whole history on every turn. The charset is printable ASCII now, so
        the rows stamp on write and the fold reads them like any other session.
        """
        path = tmp_path / "sms.db"
        _seed(path, [("sms:+15551234567", 0), ("sms:+15551234567", 1)])

        passes = _counting_transcript_passes(monkeypatch)
        db = await AsyncDatabase.sqlite(str(path))
        try:
            columns = await db.fetchall(
                "SELECT session_id FROM conversation_history WHERE agent_id = ?",
                (AGENT,),
            )
            assert [row[0] for row in columns] == ["sms:+15551234567"] * 2, (
                "the backfill left the SMS rows unstamped, so this agent is "
                "still on the transcript derivation"
            )
            projection = ConversationSessionProjection(db, AGENT)
            await projection.repair()
            settled = len(passes)
            await _append(db, "sms:+15551234567", 2)
            await projection.repair()
            await _append(db, "sms:+15551234567", 3)
            await projection.repair()
            assert len(passes) == settled
            assert [r["message_count"] for r in await projection.list()] == [4]
        finally:
            await db.close()
