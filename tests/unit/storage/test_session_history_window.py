"""#3431: a session-scoped read with a limit keeps the session's NEWEST rows.

A session-bound turn reads its history through
``get_conversation_history(limit=CONTEXT_HISTORY_LIMIT, session_id=...)``. The
resolver used to cut the session at its head: its candidate queries were
``ORDER BY created_at ASC ... LIMIT`` and its walk stopped at the ``limit``-th
member. Once a conversation passed fifty messages the agent never saw anything
said since — Emma's 69-message conversation was frozen at its 50th-oldest row,
three days before the turn reading it.

The unscoped read has always returned the newest ``limit``. These pin the
scoped read to the same window, and pin that only the END the limit cuts
changed: membership (anchor, legacy clusters, markers, other sessions) is
decided over the whole session first.
"""

from __future__ import annotations

import json
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from kestrel_sovereign.agent.context_manager import CONTEXT_HISTORY_LIMIT
from kestrel_sovereign.privacy import PrivacyMode
from kestrel_sovereign.storage.async_conversation_store import AsyncConversationStore
from kestrel_sovereign.storage.async_database import AsyncDatabase
from kestrel_sovereign.storage.privacy_wrapper import PrivacyEnforcingStorage
from kestrel_sovereign.storage.session_id_column import column_session_id

AGENT = "did:test:session-history-window"
BASE = datetime(2026, 8, 24, 9, 0, 0)
UUID_A = "7d7f1ecf-0000-4000-8000-000000000001"
UUID_B = "7d7f1ecf-0000-4000-8000-000000000002"


def _stamp(minute: int) -> str:
    return (BASE + timedelta(minutes=minute)).strftime("%Y-%m-%d %H:%M:%S")


@pytest.fixture
async def store():
    with tempfile.TemporaryDirectory() as tmp:
        db = await AsyncDatabase.sqlite(str(Path(tmp) / "window.db"))
        yield AsyncConversationStore(db, agent_id=AGENT)
        await db.close()


async def _insert(store, minute: int, session_id=None, **extra) -> int:
    """One live row, written the way the store writes one."""
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


async def _insert_run(store, minutes, session_id=None) -> list[int]:
    return [await _insert(store, minute, session_id) for minute in minutes]


async def _history_ids(store, session_id, limit) -> list[int]:
    history = await store.get_conversation_history(limit, session_id=session_id)
    return [entry["id"] for entry in history]


class TestALongSessionYieldsItsNewestRows:
    @pytest.mark.asyncio
    async def test_the_live_shape_keeps_the_turn_three_minutes_ago(self, store):
        """Emma's conversation: 69 messages, read with the production limit."""
        ids = await _insert_run(store, range(69), UUID_A)

        assert await _history_ids(store, UUID_A, CONTEXT_HISTORY_LIMIT) == ids[
            -CONTEXT_HISTORY_LIMIT:
        ]

    @pytest.mark.asyncio
    async def test_the_raw_resolver_keeps_its_newest_first_contract(self, store):
        ids = await _insert_run(store, range(12), UUID_A)

        rows = await store._get_session_messages(UUID_A, limit=5)
        assert [r[0] for r in rows] == list(reversed(ids[-5:]))

    @pytest.mark.asyncio
    async def test_a_legacy_cluster_longer_than_the_window(self, store):
        """The forward walk is a run, and the window is its tail.

        Twenty unlabeled rows a minute apart, then a gap no session survives,
        then a later conversation. The window is the run's last rows — not the
        rows after the gap, which belong to a different session however new
        they are.
        """
        ids = await _insert_run(store, range(20))
        await _insert_run(store, range(200, 210))

        assert await _history_ids(store, str(ids[0]), 4) == ids[-4:]


class TestAShortSessionIsUnchanged:
    @pytest.mark.asyncio
    async def test_under_the_limit(self, store):
        ids = await _insert_run(store, range(7), UUID_A)
        assert await _history_ids(store, UUID_A, 50) == ids

    @pytest.mark.asyncio
    async def test_exactly_at_the_limit(self, store):
        ids = await _insert_run(store, range(50), UUID_A)
        assert await _history_ids(store, UUID_A, 50) == ids

    @pytest.mark.asyncio
    async def test_a_zero_limit_returns_nothing(self, store):
        await _insert_run(store, range(3), UUID_A)
        assert await store._get_session_messages(UUID_A, limit=0) == []


class TestOtherSessionsNeverFillTheWindow:
    @pytest.mark.asyncio
    async def test_a_newer_session_does_not_top_up_a_short_one(self, store):
        """Three rows of A, then a busier, newer B. A's window is A's three."""
        mine = await _insert_run(store, range(3), UUID_A)
        await _insert_run(store, range(3, 40), UUID_B)

        assert await _history_ids(store, UUID_A, 10) == mine

    @pytest.mark.asyncio
    async def test_interleaved_sessions_keep_their_own_newest(self, store):
        """A and B alternate. A's window is A's newest five, never a B row."""
        mine = []
        for minute in range(0, 40, 2):
            mine.append(await _insert(store, minute, UUID_A))
            await _insert(store, minute + 1, UUID_B)

        assert await _history_ids(store, UUID_A, 5) == mine[-5:]

    @pytest.mark.asyncio
    async def test_a_legacy_run_closed_by_another_session(self, store):
        """The run ends at a row filed elsewhere; its successors are not ours.

        The unlabeled rows after the stamped one inherit THAT session (#3098),
        so they are newer than the legacy run and inside the gap, and still
        not members. The window is the legacy run's tail.
        """
        legacy = await _insert_run(store, range(6))
        await _insert(store, 6, UUID_B)
        await _insert_run(store, range(7, 13))

        assert await _history_ids(store, str(legacy[0]), 4) == legacy[-4:]

    @pytest.mark.asyncio
    async def test_markers_do_not_take_a_slot(self, store):
        """Markers are stripped from a read, so they cannot occupy the window."""
        await _insert(
            store, 0, UUID_A, new_session=True, type="session_marker"
        )
        turns = await _insert_run(store, range(1, 9), UUID_A)

        assert await _history_ids(store, UUID_A, 3) == turns[-3:]


class TestTheTranscriptPathStillReturnsAWholeSession:
    @pytest.mark.asyncio
    async def test_query_session_rows_at_the_transcript_limit(self, store):
        """``query_session_rows(limit=1000)`` is the UI transcript read."""
        ids = await _insert_run(store, range(120), UUID_A)
        await _insert_run(store, range(120, 130), UUID_B)

        wrapper = PrivacyEnforcingStorage(store, PrivacyMode.NORMAL)
        rows = await wrapper.query_session_rows(UUID_A, limit=1000)
        assert [r[0] for r in rows] == ids

    @pytest.mark.asyncio
    async def test_a_session_wider_than_one_id_batch_resolves_whole(self, store):
        """Full rows are fetched by id in bounded batches; none may be lost."""
        ids = await _insert_run(store, range(620), UUID_A)

        rows = await store._get_session_messages(UUID_A, limit=10_000)
        assert [r[0] for r in rows] == list(reversed(ids))
        assert await store.delete_conversation_session(UUID_A) == len(ids)

    @pytest.mark.asyncio
    async def test_isolated_transcript_keeps_its_newest_rows(self, store):
        """The in-memory ISOLATED path cuts at the same end as the store."""
        wrapper = PrivacyEnforcingStorage(store, PrivacyMode.ISOLATED)
        for turn in range(6):
            await wrapper.add_conversation("user", f"a{turn}", session_id="iso-a")
            await wrapper.add_conversation("user", f"b{turn}", session_id="iso-b")

        rows = await wrapper.query_session_rows("iso-a", limit=3)
        assert [r[2] for r in rows] == ["a3", "a4", "a5"]
        whole = await wrapper.query_session_rows("iso-a", limit=1000)
        assert [r[2] for r in whole] == [f"a{turn}" for turn in range(6)]


async def _insert_bulk(store, count: int, session_id=None, start_second: int = 0):
    """``count`` live rows a second apart in one statement batch.

    A second apart keeps a legacy (unlabeled) run inside one session however
    long it is; ids come back in insertion order.
    """
    metadata_json = json.dumps({"session_id": session_id} if session_id else {})
    await store.db.execute_many(
        "INSERT INTO conversation_history "
        "(agent_id, role, content, metadata, session_id, created_at) "
        "VALUES (?,?,?,?,?,?)",
        [
            (
                AGENT,
                "user",
                f"turn {offset}",
                metadata_json,
                column_session_id(metadata_json),
                (BASE + timedelta(seconds=start_second + offset)).strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),
            )
            for offset in range(count)
        ],
    )
    rows = await store.db.fetchall(
        "SELECT id FROM conversation_history WHERE agent_id = ? "
        "ORDER BY id DESC LIMIT ?",
        (AGENT, count),
    )
    return sorted(row[0] for row in rows)


async def _live_ids(store) -> set[int]:
    rows = await store.db.fetchall(
        "SELECT id FROM conversation_history "
        "WHERE agent_id = ? AND deleted_at IS NULL AND archived_at IS NULL",
        (AGENT,),
    )
    return {row[0] for row in rows}


# Past the 10,000 the lifecycle operations used to pass as a limit, so a
# resolver that keeps any window — head or tail — leaves a member behind.
_LONGER_THAN_ANY_WINDOW = 10_000


class TestLifecycleActsOnTheWholeSession:
    """A history window is not a lifecycle scope.

    Delete, archive, unarchive and restore resolve a session to act on ALL of
    it. Once a limit keeps a session's newest rows (#3431), a lifecycle op
    passing one skips the session's head — and its anchor marker with it,
    which leaves a deleted or archived session in the active list (#2027).
    """

    @pytest.fixture
    async def long_session(self, store):
        marker = await _insert(
            store, 0, UUID_A, new_session=True, type="session_marker"
        )
        turns = await _insert_bulk(
            store, _LONGER_THAN_ANY_WINDOW, UUID_A, start_second=120
        )
        neighbour = await _insert(store, 20_000, UUID_B)
        return marker, turns, neighbour

    @pytest.mark.asyncio
    async def test_delete_trashes_the_anchor_marker(self, store, long_session):
        marker, turns, neighbour = long_session

        assert await store.delete_conversation_session(UUID_A) == len(turns) + 1
        assert await _live_ids(store) == {neighbour}

        assert await store.restore_conversation_session(UUID_A) == len(turns) + 1
        assert await _live_ids(store) == {marker, *turns, neighbour}

    @pytest.mark.asyncio
    async def test_archive_stamps_the_anchor_marker(self, store, long_session):
        marker, turns, neighbour = long_session

        assert await store.archive_conversation_session(UUID_A) == len(turns) + 1
        assert await _live_ids(store) == {neighbour}

        assert (
            await store.unarchive_conversation_session(UUID_A) == len(turns) + 1
        )
        assert await _live_ids(store) == {marker, *turns, neighbour}

    @pytest.mark.asyncio
    async def test_a_legacy_cluster_loses_its_anchor_row_too(self, store):
        """A numeric session is keyed by its FIRST row — the one a tail drops."""
        cluster = await _insert_bulk(store, _LONGER_THAN_ANY_WINDOW + 1)

        assert await store.delete_conversation_session(str(cluster[0])) == len(
            cluster
        )
        assert await _live_ids(store) == set()

    @pytest.mark.asyncio
    async def test_a_failed_batch_leaves_the_session_as_it_was(
        self, store, long_session, monkeypatch
    ):
        """The batches are one transaction: no half-deleted session."""
        marker, turns, neighbour = long_session
        real_execute = store.db.execute
        updates = 0

        async def failing_execute(sql, params=()):
            nonlocal updates
            if sql.startswith("UPDATE conversation_history SET deleted_at"):
                updates += 1
                if updates == 2:
                    raise RuntimeError("injected failure on the second batch")
            return await real_execute(sql, params)

        monkeypatch.setattr(store.db, "execute", failing_execute)
        with pytest.raises(Exception, match="injected failure"):
            await store.delete_conversation_session(UUID_A)
        monkeypatch.setattr(store.db, "execute", real_execute)

        assert updates == 2
        assert await _live_ids(store) == {marker, *turns, neighbour}

    @pytest.mark.asyncio
    async def test_a_pattern_delete_reaches_both_ends(self, store):
        """Session scope for a pattern delete is the whole session as well.

        A window cut at either end loses one of the two matches.
        """
        oldest = await _insert(store, 0, UUID_A)
        await _insert_bulk(store, _LONGER_THAN_ANY_WINDOW, UUID_A, start_second=120)
        newest = await _insert(store, 20_000, UUID_A)

        matches = await store.find_messages_matching("turn at", session_id=UUID_A)
        assert sorted(match["id"] for match in matches) == [oldest, newest]
