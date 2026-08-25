"""#3120: a legacy row writes down which session it is in.

The migration exists because membership was never recorded. The grouper derives
it from what a row fell next to, and that evidence is destroyed the moment rows
move between the list's three views — which is why two codex rounds on the
resolver found four P1s and no rule over the rows could separate a session's
own trashed tail from an unrelated one.

These are about what it will and will not touch. The refusals are the subject
as much as the writes: a claim that conflicts with the grouping is a question
this migration is not entitled to answer.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta

import pytest

from kestrel_sovereign.storage.async_database import AsyncDatabase
from kestrel_sovereign.storage.conversation_sessions import live_transcript_sql
from kestrel_sovereign.storage.legacy_session_stamp import (
    plan_stamps,
    stamp_legacy_sessions,
)

AGENT = "did:test:legacy-stamp"
BASE = datetime(2026, 6, 1, 9, 0, 0)
UUID_A = "8f1d1c62-9b0e-4b2c-9a1d-000000000001"
UUID_B = "8f1d1c62-9b0e-4b2c-9a1d-000000000002"

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


def _stamp(minute: int) -> str:
    return (BASE + timedelta(minutes=minute)).strftime("%Y-%m-%d %H:%M:%S")


def _seed(path, rows):
    """A pre-#2958 database, so the boot ALTER backfills the column as it does
    for a real upgrade. ``rows`` are ``(metadata dict or None, minute)``."""
    conn = sqlite3.connect(str(path))
    try:
        conn.executescript(PRE)
        for metadata, minute in rows:
            conn.execute(
                "INSERT INTO conversation_history "
                "(agent_id, role, content, metadata, created_at) VALUES (?,?,?,?,?)",
                (
                    AGENT,
                    "user",
                    f"turn at {minute}",
                    None if metadata is None else json.dumps(metadata),
                    _stamp(minute),
                ),
            )
        conn.commit()
    finally:
        conn.close()


def _derived(rows):
    """The shape :func:`live_transcript_sql` selects, built without a database.

    ``(id, role, metadata, created_at, session_id column, canonical key)`` —
    the projection's own read, so the planner is exercised against exactly what
    it sees in production. Built here rather than by booting, because booting
    RUNS the migration and there would be nothing left to plan.
    """
    from kestrel_sovereign.storage.session_id_column import column_session_id

    derived = []
    for row_id, metadata, minute in rows:
        blob = None if metadata is None else json.dumps(metadata)
        derived.append(
            (row_id, "user", blob, _stamp(minute), column_session_id(blob), _stamp(minute))
        )
    return derived


async def _metadata(db):
    rows = await db.fetchall(
        "SELECT id, metadata, session_id FROM conversation_history "
        "WHERE agent_id = ? ORDER BY id",
        (AGENT,),
    )
    return {
        row[0]: (json.loads(row[1] or "{}").get("session_id"), row[2])
        for row in rows
    }


class TestWhatItWritesDown:
    def test_a_claim_that_agrees_with_the_grouping_is_respelled(self):
        """The common case, and the safest one: the bare integer names a row in
        the very session the grouper assigns, so writing the grouping down
        moves no message anywhere."""
        plan = plan_stamps(
            _derived([(1, {"session_id": UUID_A}, 0), (2, {"session_id": "1"}, 5)])
        )
        assert [(s.row_id, s.session_id) for s in plan.stamps] == [(2, UUID_A)]
        assert plan.conflicts == []

    def test_a_claim_naming_no_live_row_is_written_down(self):
        """Nothing live disagrees, so the grouping is the only answer standing.

        The referenced row is archived, or deleted, or gone: it names a session
        in another view and this row is displayed in this one.
        """
        plan = plan_stamps(
            _derived([(1, {"session_id": UUID_A}, 0), (2, {"session_id": "9999"}, 5)])
        )
        assert [(s.row_id, s.session_id) for s in plan.stamps] == [(2, UUID_A)]

    def test_a_claim_naming_another_live_session_is_refused(self):
        """The row says one thing and the transcript says another.

        Which is right is not a question a migration may answer: the claim is
        what a writer recorded, the grouping is what the list shows, and
        overwriting either loses a record. Reported and left alone.
        """
        plan = plan_stamps(
            _derived(
                [
                    (1, {"session_id": UUID_B}, 0),
                    (2, {"session_id": UUID_A}, 100),
                    (3, {"session_id": "1"}, 105),
                ]
            )
        )
        assert plan.stamps == []
        assert plan.conflicts == [(3, "1", UUID_A)]


class TestWhatItLeavesAlone:
    def test_a_legacy_cluster_keeps_its_key(self):
        """A cluster keyed by a row id is a different operation.

        The ``session_id`` column may not hold that key (#2958), and re-keying
        such a cluster is what #3061 abandoned — ``conversation_titles`` is
        keyed on the legacy id, so every rename would be lost. Nothing here
        moves a session's key.
        """
        plan = plan_stamps(_derived([(1, None, 0), (2, None, 5)]))
        assert plan.stamps == []
        assert plan.conflicts == []

    def test_a_row_id_key_is_never_written_however_it_is_spelled(self):
        """The key must come from a row's METADATA, not the row-id fallback.

        Asking instead whether the key LOOKS like a row id answers by a digit
        test, and a negative id has a sign in front of it: it passes, the
        column's charset admits the hyphen, and ``-5`` would be written into
        rows as though it were a session someone had named.
        """
        plan = plan_stamps(_derived([(-5, None, 0), (-4, None, 5)]))
        assert plan.stamps == []

    def test_a_legacy_cluster_reports_no_conflicts_either(self):
        """Silence, not just inaction.

        The per-row eligibility test would refuse these anyway — a digit key is
        exactly what the column declines. What skipping the cluster adds is
        that its rows are never ASKED about their claims, so a claim naming
        another session is not reported as a conflict the operator can do
        nothing about on a cluster nothing was going to touch.
        """
        plan = plan_stamps(
            _derived(
                [
                    (1, {"session_id": UUID_A}, 0),
                    (2, None, 100),
                    (3, {"session_id": "1"}, 105),
                ]
            )
        )
        assert plan.stamps == []
        assert plan.conflicts == []

    def test_a_document_that_is_not_an_object_is_not_replaced(self):
        """A message with an unreadable blob is still a message.

        ``"[]"`` is valid JSON and is not metadata. Overwriting it with a
        document of our own would destroy whatever it holds.
        """
        rows = _derived([(1, {"session_id": UUID_A}, 0)])
        rows.append((2, "user", "[]", _stamp(5), None, _stamp(5)))
        assert plan_stamps(rows).stamps == []

    def test_a_key_the_column_cannot_hold_is_not_written(self):
        """Metadata the column may not follow puts the two back into the
        disagreement this exists to end.

        ``rasa_shim`` files every SMS turn under ``sms:{sender}``, which the
        column's charset refuses (#3061).
        """
        plan = plan_stamps(
            _derived([(1, {"session_id": "sms:+15551234567"}, 0), (2, None, 5)])
        )
        assert plan.stamps == []


    def test_an_empty_document_is_not_replaced(self):
        """Only SQL NULL is absent.

        An empty string is a value someone stored, and ``json.loads`` refuses
        it — treating it as missing would replace it with a document of our
        own, which is the one thing this promises not to do.
        """
        rows = _derived([(1, {"session_id": UUID_A}, 0)])
        rows.append((2, "user", "", _stamp(5), None, _stamp(5)))
        assert plan_stamps(rows).stamps == []

    def test_a_duplicated_key_is_not_resolved(self):
        """A document with the key twice has no single reading.

        SQLite's ``json_extract`` takes the first, PostgreSQL's ``jsonb`` and
        Python's ``json.loads`` take the last — so whichever is "right", two of
        the three readers disagree, and rewriting it would silently pick one.
        """
        rows = _derived([(1, {"session_id": UUID_A}, 0)])
        rows.append(
            (2, "user", '{"session_id": "9999", "session_id": "1"}',
             _stamp(5), None, _stamp(5))
        )
        assert plan_stamps(rows).stamps == []

    def test_a_claim_naming_a_live_marker_is_refused(self):
        """A marker is structural, and the grouper leaves it out of the
        messages it collects — so a claim naming one is placed by asking the
        marker itself, not by looking it up among rows that never contain it.
        Reading "not placed" as "names nothing live" would overwrite the claim
        instead of reporting the conflict it is.
        """
        rows = _derived(
            [
                (1, {"session_id": UUID_B, "new_session": True,
                     "type": "session_marker"}, 0),
                (2, {"session_id": UUID_B}, 1),
                (3, {"session_id": UUID_A}, 100),
                (4, {"session_id": "1"}, 105),
            ]
        )
        plan = plan_stamps(rows)
        assert plan.stamps == []
        assert plan.conflicts == [(4, "1", UUID_A)]

    def test_a_claim_naming_this_session_s_own_marker_agrees(self):
        """And placing the marker is what makes that a claim rather than a
        conflict: the marker STARTS this session, so a later row naming it is
        saying where it already is. Refusing here would leave a whole session's
        continuation unstamped for naming the row that opened it — which is
        precisely what #2012 says the UI round-tripped.
        """
        rows = _derived(
            [
                (1, {"session_id": UUID_A, "new_session": True,
                     "type": "session_marker"}, 0),
                (2, {"session_id": "1"}, 1),
            ]
        )
        plan = plan_stamps(rows)
        assert [(s.row_id, s.session_id) for s in plan.stamps] == [(2, UUID_A)]
        assert plan.conflicts == []


class TestThePass:
    @pytest.mark.asyncio
    async def test_it_writes_the_rows_it_planned(self, tmp_path):
        path = tmp_path / "pass.db"
        _seed(path, [({"session_id": UUID_A}, 0), ({"session_id": "1"}, 5)])

        db = await AsyncDatabase.sqlite(str(path))
        try:
            assert (await _metadata(db))[2] == ("1", None)
            assert await stamp_legacy_sessions(db) == {"stamped": 1, "refused": 0, "skipped": 0, "incomplete": 0}
            # Metadata AND the column, because a reader that consults one and
            # not the other is the shape #3098 was about.
            assert (await _metadata(db))[2] == (UUID_A, UUID_A)
        finally:
            await db.close()

    @pytest.mark.asyncio
    async def test_a_second_pass_writes_nothing(self, tmp_path):
        """Idempotent by construction, which is why it needs no marker.

        A row that names its session is not a candidate, and a row this refuses
        is refused again for the same reason — so "has this run" is a question
        it never has to ask. Three separate review findings were that question's
        rather than the work's: an agent DID left in a durable table after a
        privacy purge, a completion record that could not be published
        atomically against a concurrent restore, and an anti-join over the whole
        transcript on every ``from_pool()``.
        """
        path = tmp_path / "twice.db"
        _seed(path, [({"session_id": UUID_A}, 0), ({"session_id": "1"}, 5)])

        db = await AsyncDatabase.sqlite(str(path))
        try:
            assert await stamp_legacy_sessions(db) == {
                "stamped": 1, "refused": 0, "skipped": 0, "incomplete": 0,
            }
            # A second pass has nothing to plan, so it is not incomplete — it
            # is finished. "Incomplete" means a row this pass MEANT to write
            # moved out from under it.
            assert await stamp_legacy_sessions(db) == {
                "stamped": 0, "refused": 0, "skipped": 0, "incomplete": 0,
            }
        finally:
            await db.close()

    @pytest.mark.asyncio
    async def test_every_view_is_planned(self, tmp_path):
        """Each of the list's three views groups its OWN rows, so each is
        planned; a claim is resolved across all of them, because a claim names
        a ROW and a row lives in exactly one."""
        path = tmp_path / "views.db"
        _seed(
            path,
            [
                ({"session_id": UUID_A}, 0),
                ({"session_id": "1"}, 5),
                ({"session_id": UUID_B}, 100),
                ({"session_id": "3"}, 105),
            ],
        )
        conn = sqlite3.connect(str(path))
        conn.execute(
            "UPDATE conversation_history SET archived_at = ? WHERE id IN (3, 4)",
            ("2026-06-01 12:00:00",),
        )
        conn.commit()
        conn.close()

        db = await AsyncDatabase.sqlite(str(path))
        try:
            await stamp_legacy_sessions(db)
            observed = await _metadata(db)
            assert observed[2] == (UUID_A, UUID_A), "the active view was not stamped"
            assert observed[4] == (UUID_B, UUID_B), "the archived view was not stamped"
        finally:
            await db.close()

    @pytest.mark.asyncio
    async def test_history_moving_under_the_pass_stops_it(self, tmp_path):
        """The compare-and-set asks whether the ROW moved. That is not the
        whole question: a candidate's metadata can sit still while the rows
        that decide which session it is in do not. A purge removing a canonical
        anchor and missing its unlabeled tail would otherwise have this stamp
        that tail back INTO the purged session, and a privacy purge that leaves
        a row naming what it destroyed is not one.
        """
        path = tmp_path / "moved.db"
        _seed(path, [({"session_id": UUID_A}, 0), ({"session_id": "1"}, 5)])
        db = await AsyncDatabase.sqlite(str(path))

        from kestrel_sovereign.storage import legacy_session_stamp

        real_stamp = legacy_session_stamp._change_stamp
        calls = {"n": 0}

        async def moving(database, agent):
            calls["n"] += 1
            value = await real_stamp(database, agent)
            # The second read is the one taken before the writes: report a row
            # event nothing in this pass produced.
            return None if value is None else value + (7 if calls["n"] > 1 else 0)

        try:
            legacy_session_stamp._change_stamp = moving
            assert await stamp_legacy_sessions(db) == {"stamped": 0, "refused": 0, "skipped": 0, "incomplete": 1}
            assert (await _metadata(db))[2] == ("1", None)
        finally:
            legacy_session_stamp._change_stamp = real_stamp
            await db.close()

    @pytest.mark.asyncio
    async def test_a_row_rewritten_under_the_plan_is_not_clobbered(
        self, tmp_path, monkeypatch
    ):
        """The plan is read outside the write transactions, deliberately — it
        is unbounded. So every write re-checks the document it was made from,
        and a row that moved is skipped rather than overwritten."""
        path = tmp_path / "raced.db"
        _seed(path, [({"session_id": UUID_A}, 0), ({"session_id": "1"}, 5)])
        db = await AsyncDatabase.sqlite(str(path))

        from kestrel_sovereign.storage import legacy_session_stamp

        real_plan = legacy_session_stamp.plan_stamps

        def stale(rows, placements=None):
            plan = real_plan(rows, placements)
            return legacy_session_stamp.StampPlan(
                [
                    stamp._replace(previous_metadata='{"session_id": "stale"}')
                    for stamp in plan.stamps
                ],
                plan.conflicts,
            )

        monkeypatch.setattr(legacy_session_stamp, "plan_stamps", stale)
        try:
            assert await stamp_legacy_sessions(db) == {"stamped": 0, "refused": 0, "skipped": 0, "incomplete": 1}
            assert (await _metadata(db))[2] == ("1", None), "the row was clobbered"
        finally:
            await db.close()
