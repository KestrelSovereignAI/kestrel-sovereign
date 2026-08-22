"""#2960: paging the ``conversation_sessions`` projection by key.

The list this replaces read a fixed window of ``conversation_history`` and
grouped whatever fell inside it, so a session whose rows had all aged past the
window was absent from every page at every ``limit`` — measured on Emma's live
database, 49 of 144 conversations. The read here is bounded by the PAGE and
continued by a KEY, which is what makes "ask again" reach the rest.

Three things can go wrong with a keyset page, and this file is built around all
three.

**The order and its continuation can disagree.** A cursor is a claim about where
one page stopped, so the predicate that resumes it has to be the exact inverse
of the ``ORDER BY`` that produced it — NULL placement included. The seam is
where that shows: a tie spanning two pages is served twice or not at all when
the two disagree about the tie-break.

**Two renderings of one rule can drift.** The active view pages in SQL and the
archived view (and ISOLATED's in-memory buffer) page in Python, over the same
declaration. The differential below runs a corpus through both and requires
them to produce the same sequence at every cursor.

**The page can stop using its index.** An ``ORDER BY`` the index does not serve
makes the engine sort the agent's whole session table before applying ``LIMIT``
— O(sessions) per page, which is the cost this table was measured into existence
to remove, arriving back with every test still green.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Dict, List

import pytest

from kestrel_sovereign.storage.async_database import AsyncDatabase
from kestrel_sovereign.storage.conversation_sessions import (
    ConversationSessionProjection,
    SessionCursorError,
    decode_session_cursor,
    encode_session_cursor,
)
from kestrel_sovereign.storage.session_grouping import (
    SESSION_ORDER,
    session_cursor_after,
    session_order_index_columns,
    session_order_sql,
    sort_sessions,
)

AGENT = "did:test:session-page"
START = datetime(2026, 5, 1, 9, 0, 0)


def _rows() -> List[Dict[str, Any]]:
    """A corpus whose shape is what makes the seam interesting.

    Deliberately full of ties: ``last_message_at`` repeats every third session,
    so the tie-break decides several page boundaries rather than none. Ties are
    ordinary in real data — SQLite stores history to the second, and a wake and
    the turn it triggers are written in one transaction.
    """
    rows = []
    for index in range(37):
        stamp = START + timedelta(minutes=(index // 3) * 5)
        rows.append({
            "session_id": f"sess-{index:03d}",
            "started_at": stamp.strftime("%Y-%m-%d %H:%M:%S"),
            "last_message_at": stamp.strftime("%Y-%m-%d %H:%M:%S"),
            "message_count": 2,
            "user_message_count": 1,
            "first_user_message_id": index + 1,
            "wake_source": None,
        })
    return rows


async def _seeded(tmp_path, name: str, rows: List[Dict[str, Any]]):
    db = await AsyncDatabase.sqlite(str(tmp_path / name))
    await db.execute_many(
        "INSERT INTO conversation_sessions "
        "(agent_id, session_id, started_at, last_message_at, message_count, "
        "user_message_count, first_user_message_id, wake_source) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        [
            (AGENT, r["session_id"], r["started_at"], r["last_message_at"],
             r["message_count"], r["user_message_count"],
             r["first_user_message_id"], r["wake_source"])
            for r in rows
        ],
    )
    return db, ConversationSessionProjection(db, AGENT)


async def _page_all_sql(projection, limit: int) -> List[str]:
    """Every session id the SQL page reaches, in order, by asking again."""
    seen: List[str] = []
    after = None
    for _ in range(500):
        page = await projection.page(limit=limit, after=after)
        if not page:
            return seen
        seen.extend(row["session_id"] for row in page)
        if len(page) < limit:
            return seen
        token = encode_session_cursor(page[-1], "active")
        after = decode_session_cursor(token, "active")
    raise AssertionError("paging did not terminate")


def _page_all_python(rows: List[Dict[str, Any]], limit: int) -> List[str]:
    """The same walk over the same rows, through the Python rendering."""
    ordered = sort_sessions(list(rows))
    seen: List[str] = []
    after = None
    for _ in range(500):
        remaining = session_cursor_after(ordered, after)
        page = remaining[:limit]
        if not page:
            return seen
        seen.extend(row["session_id"] for row in page)
        if len(page) < limit:
            return seen
        token = encode_session_cursor(page[-1], "active")
        after = decode_session_cursor(token, "active")
    raise AssertionError("paging did not terminate")


@pytest.mark.asyncio
@pytest.mark.parametrize("limit", [1, 2, 5, 37, 100])
async def test_paging_reaches_every_session_exactly_once(tmp_path, limit):
    """At any page size, and at page sizes that land exactly on the end."""
    rows = _rows()
    db, projection = await _seeded(tmp_path, f"page-{limit}.db", rows)
    try:
        seen = await _page_all_sql(projection, limit)
    finally:
        await db.close()
    assert len(seen) == len(set(seen)), "a session was served on two pages"
    assert set(seen) == {row["session_id"] for row in rows}
    # ...and in the one order, across every seam.
    expected = [row["session_id"] for row in sort_sessions(list(rows))]
    assert seen == expected


@pytest.mark.asyncio
@pytest.mark.parametrize("limit", [1, 2, 5, 37, 100])
async def test_the_sql_page_and_the_python_page_are_the_same_walk(tmp_path, limit):
    """The differential. Two renderings of :data:`SESSION_ORDER`, one answer.

    The active view pages in the engine and the archived view pages in Python;
    a caller cannot tell which served it, so they must not be able to tell.
    """
    rows = _rows()
    db, projection = await _seeded(tmp_path, f"diff-{limit}.db", rows)
    try:
        assert await _page_all_sql(projection, limit) == _page_all_python(rows, limit)
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_a_tie_spanning_a_page_seam_is_neither_repeated_nor_dropped(tmp_path):
    """The seam case the tie-break exists for.

    Three sessions share one ``last_message_at`` and the page ends in the middle
    of them. Resolved by the tie-break, the next page resumes at the third;
    resolved by ``last_message_at`` alone it either re-serves the first two or
    steps over the third.
    """
    stamp = START.strftime("%Y-%m-%d %H:%M:%S")
    rows = [
        {"session_id": sid, "started_at": stamp, "last_message_at": stamp,
         "message_count": 1, "user_message_count": 1,
         "first_user_message_id": index + 1, "wake_source": None}
        for index, sid in enumerate(["aaa", "bbb", "ccc"])
    ]
    db, projection = await _seeded(tmp_path, "tie.db", rows)
    try:
        first = await projection.page(limit=2)
        assert [r["session_id"] for r in first] == ["aaa", "bbb"]
        after = decode_session_cursor(encode_session_cursor(first[-1], "active"), "active")
        second = await projection.page(limit=2, after=after)
        assert [r["session_id"] for r in second] == ["ccc"]
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_a_session_with_no_readable_activity_is_still_reachable(tmp_path):
    """A NULL ``last_message_at`` sorts LAST and is reached, not lost.

    ``last_message_at < ?`` is NULL for such a row, so a keyset predicate that
    did not say otherwise would exclude it from every page — a conversation
    unreachable at any depth, which is the defect this ticket removes. Nothing
    in the codebase writes a NULL here today; the column permits one, and a
    session the list can never show is not a state to discover from production.
    """
    rows = _rows()[:4]
    rows.append({
        "session_id": "undated", "started_at": None, "last_message_at": None,
        "message_count": 1, "user_message_count": 1,
        "first_user_message_id": 99, "wake_source": None,
    })
    db, projection = await _seeded(tmp_path, "undated.db", rows)
    try:
        seen = await _page_all_sql(projection, 2)
    finally:
        await db.close()
    assert "undated" in seen, "an undatable session was unreachable by paging"
    assert seen[-1] == "undated", "an undatable session is the EARLIEST, so it is last"


@pytest.mark.asyncio
async def test_the_page_is_served_by_its_index_not_by_a_sort(tmp_path):
    """The bound is the page, and it stays the page.

    An ``ORDER BY`` the index does not serve makes SQLite sort the agent's whole
    session table before applying ``LIMIT``. That is O(sessions) per request —
    the cost this table exists to remove — and every other test here still
    passes when it comes back.
    """
    db, projection = await _seeded(tmp_path, "plan.db", _rows())
    try:
        plan = await db.fetchall(
            "EXPLAIN QUERY PLAN SELECT session_id FROM conversation_sessions "
            f"WHERE agent_id = ? {session_order_sql(db.backend_type)} LIMIT ?",
            (AGENT, 10),
        )
    finally:
        await db.close()
    text = " ".join(str(row) for row in plan).upper()
    assert "IDX_CONVERSATION_SESSIONS_RECENT" in text, text
    assert "USE TEMP B-TREE" not in text, text


def test_the_ordering_and_the_index_are_one_expression():
    """An index that does not carry every key the page orders by is no index.

    Compared as text, in both dialects, because the failure is silent: the two
    can drift into orderings that agree on the corpus a test happens to use.
    """
    for backend in ("sqlite", "postgres"):
        assert session_order_sql(backend) == (
            "ORDER BY " + session_order_index_columns(backend)
        )


def test_a_cursor_carries_this_orderings_keys_and_nothing_else():
    row = {"last_message_at": "2026-05-01 09:00:00", "session_id": "sess-001",
           "message_count": 7}
    token = encode_session_cursor(row, "active")
    assert decode_session_cursor(token, "active") == (
        "2026-05-01 09:00:00", "sess-001",
    )
    assert len(SESSION_ORDER) == 2, "the cursor's shape follows SESSION_ORDER"


@pytest.mark.parametrize("token", ["", "not-base64!!", "YWJj", "e30="])
def test_an_unreadable_cursor_is_refused_rather_than_ignored(token):
    """Restarting at page one for an unreadable cursor answers a request for
    page nine with page one, which reads as a list that forgot where it was."""
    with pytest.raises(SessionCursorError):
        decode_session_cursor(token, "active")


def test_a_cursor_minted_for_another_view_is_refused():
    """The views are served by different machinery over different memberships,
    so a token from one names a place the other never ordered by."""
    token = encode_session_cursor(
        {"last_message_at": "2026-05-01 09:00:00", "session_id": "s"}, "active"
    )
    with pytest.raises(SessionCursorError):
        decode_session_cursor(token, "archived")


@pytest.mark.asyncio
async def test_the_page_is_bounded_in_sql_not_trimmed_in_python(tmp_path):
    """``LIMIT`` is what makes page one's cost independent of history's size.

    Reading more and slicing would pass every ordering test here while making
    the read O(sessions) again — so the statement itself is inspected.
    """
    seen = {}
    db, projection = await _seeded(tmp_path, "bounded.db", _rows())
    original = db.fetchall

    async def _recording(sql, params=None):
        seen["sql"], seen["params"] = sql, params
        return await original(sql, params)

    db.fetchall = _recording
    try:
        await projection.page(limit=3)
    finally:
        db.fetchall = original
        await db.close()
    assert "LIMIT ?" in seen["sql"], seen["sql"]
    assert seen["params"][-1] == 3
