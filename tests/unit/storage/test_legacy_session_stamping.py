"""#3061: every live row gets a session id its column can hold.

``ConversationSessionProjection`` folds an append into the sessions it touched
— O(rows appended) — but only while every live row of that agent carries a
``session_id``. One row that does not sends every repair down the
whole-transcript path instead, and #2960 put that repair on the conversation
list's read path. The column is linear in history size there and flat when it
is not, and it never improves on its own: the NULL is not an upgrade state, it
is what old history looks like for ever.

Two things leave a row NULL, and this file is built around both.

**A metadata id the column's contract excluded.** Widened to printable ASCII
here, which fixes the writers — ``rasa_shim`` files every SMS turn under
``sms:{sender}`` — but not the rows already stored, because Phase A's backfill
runs once inside the ALTER's own transaction and every existing database is
long past it.

**A cluster carrying no usable id at all.** The grouper keys those by
``str(first row id)``, which is all digits and therefore refused on purpose
(#2012). Those are re-keyed, and the tests below are mostly about the limits of
that: what may be renamed, what may not, and why a cluster is re-keyed whole or
not at all.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta

import pytest

from kestrel_sovereign.storage.async_database import AsyncDatabase
from kestrel_sovereign.storage.conversation_sessions import (
    ConversationSessionProjection,


    legacy_session_assignments,
)

AGENT = "did:test:legacy-stamping"

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
CREATE TABLE conversation_titles (
    agent_id TEXT NOT NULL DEFAULT '',
    session_id TEXT NOT NULL,
    name TEXT,
    updated_at TIMESTAMP,
    PRIMARY KEY (agent_id, session_id)
);
"""


def _seed(path, rows, titles=()):
    """Write a pre-#3061 database: history, and any names the user gave it."""
    conn = sqlite3.connect(str(path))
    try:
        conn.executescript(PRE)
        for role, content, metadata, created_at in rows:
            conn.execute(
                "INSERT INTO conversation_history "
                "(agent_id, role, content, metadata, created_at) VALUES (?,?,?,?,?)",
                (AGENT, role, content, metadata, created_at),
            )
        for session_id, name in titles:
            conn.execute(
                "INSERT INTO conversation_titles (agent_id, session_id, name) "
                "VALUES (?,?,?)",
                (AGENT, session_id, name),
            )
        conn.commit()
    finally:
        conn.close()


BASE = datetime(2026, 6, 1, 9, 0, 0)


def _stamp(minute):
    return (BASE + timedelta(minutes=minute)).strftime("%Y-%m-%d %H:%M:%S")


def _legacy(*, at=0, count=2, metadata="{}"):
    return [("user", f"legacy turn {i}", metadata, _stamp(at + i)) for i in range(count)]


async def _rows(db):
    return await db.fetchall(
        "SELECT id, metadata, session_id FROM conversation_history ORDER BY id", ()
    )


class TestTheAcceptance:
    @pytest.mark.asyncio
    async def test_a_legacy_history_stops_re_deriving_itself(
        self, tmp_path, monkeypatch
    ):
        """The ticket, stated as a differential.

        Counted rather than inferred, and counted on the thing that costs: how
        many times a repair reads and groups the agent's WHOLE live history.
        ``RepairOutcome.kind`` is the wrong gauge — a generation rotation
        reports ``rebuilt`` from the CHUNKED path, which is bounded, so a test
        watching the kind would call a fixed history broken and a broken one
        fixed.
        """
        path = tmp_path / "legacy.db"
        _seed(path, _legacy(count=6))

        passes = []
        original = ConversationSessionProjection._rebuild_from_transcript

        async def counting(self):
            passes.append(self.agent_id)
            return await original(self)

        monkeypatch.setattr(
            ConversationSessionProjection, "_rebuild_from_transcript", counting
        )

        db = await AsyncDatabase.sqlite(str(path))
        try:
            projection = ConversationSessionProjection(db, AGENT)
            assert not await projection._has_unstamped_rows()
            await projection.repair()
            for turn in range(3):
                await db.execute(
                    "INSERT INTO conversation_history "
                    "(agent_id, role, content, metadata, created_at, session_id) "
                    "VALUES (?, 'user', 'new turn', ?, ?, 'sess-new')",
                    (AGENT, json.dumps({"session_id": "sess-new"}), _stamp(500 + turn * 90)),
                )
                await projection.repair()
            assert passes == [], (
                f"{len(passes)} repairs re-derived the whole transcript after "
                "the stamping pass had run"
            )

            # ...and one unstamped row is all it takes to go back, which is what
            # makes the assertion above about the stamping rather than about
            # this fixture being small.
            await db.execute(
                "INSERT INTO conversation_history "
                "(agent_id, role, content, metadata, created_at) "
                "VALUES (?, 'user', 'unstamped', '{}', ?)",
                (AGENT, _stamp(900)),
            )
            await projection.repair()
            assert passes == [AGENT]
        finally:
            await db.close()

    @pytest.mark.asyncio
    async def test_the_cluster_keeps_its_rows(self, tmp_path):
        """Re-keying writes down the reader's answer; it does not re-decide it.

        The rows a session is made of are the grouper's, before and after — the
        only thing that changes is the spelling of the id they are filed under.
        """
        path = tmp_path / "membership.db"
        _seed(path, _legacy(count=4) + _legacy(at=300, count=3))

        db = await AsyncDatabase.sqlite(str(path))
        try:
            projection = ConversationSessionProjection(db, AGENT)
            await projection.repair()
            listed = {
                row["session_id"]: row["message_count"] for row in await projection.list()
            }
            assert listed == {"legacy-1": 4, "legacy-5": 3}
        finally:
            await db.close()

    @pytest.mark.asyncio
    async def test_a_second_boot_changes_nothing(self, tmp_path):
        """Idempotent, and guarded by the state rather than by a marker."""
        path = tmp_path / "twice.db"
        _seed(path, _legacy(count=3))

        db = await AsyncDatabase.sqlite(str(path))
        first = await _rows(db)
        await db.close()

        db = await AsyncDatabase.sqlite(str(path))
        try:
            assert await _rows(db) == first
        finally:
            await db.close()


class TestWhatMayBeRenamed:
    @pytest.mark.asyncio
    async def test_an_id_a_writer_chose_is_never_renamed(self, tmp_path):
        """The rule: only a value session grouping IGNORES may be overwritten.

        ``sms:{sender}`` is the case that matters — it was outside the column's
        contract until this ticket widened it, and a pass that "fixed" it by
        renaming would break every later SMS turn, which arrives under the same
        id.
        """
        path = tmp_path / "chosen.db"
        _seed(path, [
            ("user", "sms turn", json.dumps({"session_id": "sms:+15551234567"}), _stamp(0)),
        ])

        db = await AsyncDatabase.sqlite(str(path))
        try:
            (row,) = await _rows(db)
            assert json.loads(row[1])["session_id"] == "sms:+15551234567"
            assert row[2] == "sms:+15551234567", (
                "the id is printable ASCII and the column can hold it now; "
                "leaving it NULL is what kept this agent on the transcript pass"
            )
        finally:
            await db.close()

    @pytest.mark.asyncio
    async def test_a_bare_integer_is_replaced_because_every_reader_ignores_it(
        self, tmp_path
    ):
        """#2012's mis-filed keys. Grouping ignores them, so nothing is lost."""
        path = tmp_path / "integer.db"
        _seed(path, [
            ("user", "a", json.dumps({"session_id": "197"}), _stamp(0)),
            ("user", "b", json.dumps({"session_id": "197"}), _stamp(1)),
        ])

        db = await AsyncDatabase.sqlite(str(path))
        try:
            rows = await _rows(db)
            assert [row[2] for row in rows] == ["legacy-1", "legacy-1"]
            assert json.loads(rows[0][1])["session_id"] == "legacy-1"
        finally:
            await db.close()

    @pytest.mark.asyncio
    async def test_a_non_string_id_is_left_alone(self, tmp_path):
        """The reader-disagreement case Phase A's contract exists for.

        ``{"session_id": true}`` extracts as ``1`` in SQLite, ``'true'`` in
        PostgreSQL and ``'True'`` under Python's ``str()``. Grouping files the
        row under Python's rendering, so a pass that "wrote down the reader's
        answer" here would be choosing one reader over the others and calling
        it a migration. The column stays silent instead.
        """
        path = tmp_path / "boolean.db"
        _seed(path, [("user", "a", json.dumps({"session_id": True}), _stamp(0))])

        db = await AsyncDatabase.sqlite(str(path))
        try:
            (row,) = await _rows(db)
            assert row[2] is None
            assert json.loads(row[1])["session_id"] is True
        finally:
            await db.close()


class TestWholeOrNotAtAll:
    def test_one_unwritable_row_withholds_its_whole_cluster(self):
        """Half a re-keyed cluster never converges.

        The rewritten rows group under the new id on the next pass while the
        rest group by time gap under another, so the conversation stays split
        for ever. Measured against exactly this transcript: only the third row
        holds a document a key can be added to.
        """
        rows = [
            (1, "user", "{not json", "2026-06-01 09:00:00", None, None),
            (2, "user", "42", "2026-06-01 09:01:00", None, None),
            (3, "user", "{}", "2026-06-01 09:02:00", None, None),
        ]
        stamp, rekey, titles, left_alone = legacy_session_assignments(rows)
        assert rekey == {}, "part of a cluster was re-keyed, which splits it"
        assert stamp == {}
        assert titles == {}
        assert left_alone == ["1"]

    def test_a_wholly_writable_cluster_is_taken(self):
        rows = [
            (1, "user", "{}", "2026-06-01 09:00:00", None, None),
            (2, "user", json.dumps({"session_id": "9"}), "2026-06-01 09:01:00", None, None),
        ]
        stamp, rekey, titles, left_alone = legacy_session_assignments(rows)
        assert rekey == {1: "legacy-1", 2: "legacy-1"}
        assert titles == {"1": "legacy-1"}
        assert left_alone == []


class TestTheNameFollowsTheConversation:
    @pytest.mark.asyncio
    async def test_a_user_assigned_name_is_carried_to_the_new_id(self, tmp_path):
        """``conversation_titles`` is keyed on the id being replaced.

        A rename is a fact the user authored, unlike the preview beside it, so
        it travels with the conversation rather than being dropped as a cost of
        the migration.
        """
        path = tmp_path / "titled.db"
        _seed(path, _legacy(count=2), titles=[("1", "Penguin plans")])

        db = await AsyncDatabase.sqlite(str(path))
        try:
            names = await db.fetchall(
                "SELECT session_id, name FROM conversation_titles WHERE agent_id = ?",
                (AGENT,),
            )
            assert names == [("legacy-1", "Penguin plans")]
        finally:
            await db.close()

    @pytest.mark.asyncio
    async def test_a_name_already_on_the_new_id_is_not_clobbered(self, tmp_path):
        """The table is keyed ``(agent_id, session_id)``, so a blind UPDATE raises.

        Learning that inside a migration is the worst place to learn it, so the
        carry is guarded: the existing name wins and the old row is left where
        it is rather than destroying either.
        """
        path = tmp_path / "collision.db"
        _seed(
            path,
            _legacy(count=2),
            titles=[("1", "the old name"), ("legacy-1", "the standing name")],
        )

        db = await AsyncDatabase.sqlite(str(path))
        try:
            names = dict(await db.fetchall(
                "SELECT session_id, name FROM conversation_titles WHERE agent_id = ?",
                (AGENT,),
            ))
            assert names["legacy-1"] == "the standing name"
        finally:
            await db.close()

    def test_a_minted_key_does_not_collide_with_one_a_writer_chose(self):
        """A literal ``legacy-1`` in the transcript is unlikely and cheap to check."""
        rows = [
            (1, "user", "{}", "2026-06-01 09:00:00", None, None),
            (2, "user", json.dumps({"session_id": "legacy-1"}), "2026-06-01 12:00:00", None, None),
        ]
        _stamped, rekey, titles, _left = legacy_session_assignments(rows)
        assert rekey == {1: "legacy-1-2"}
        assert titles == {"1": "legacy-1-2"}
