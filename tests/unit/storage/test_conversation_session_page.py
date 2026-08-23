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

import base64
import json
import os
from uuid import uuid4
from datetime import datetime, timedelta
from typing import Any, Dict, List

import pytest

from kestrel_sovereign.storage.async_database import AsyncDatabase
from kestrel_sovereign.storage.conversation_sessions import (
    SESSION_CURSOR_MAX_LENGTH,
    ConversationSessionProjection,
    SessionCursorError,
    decode_session_cursor,
    encode_session_cursor,
)
from kestrel_sovereign.storage.session_id_column import SESSION_ID_MAX_LENGTH
from kestrel_sovereign.storage.session_grouping import (
    SESSION_ORDER,
    session_order_index_columns,
    session_order_sql,
    sort_sessions,
)

AGENT = "did:test:session-page"


@pytest.fixture(params=["sqlite", "postgres"])
async def sqlite_or_postgres(request, tmp_path):
    """``(db, agent)`` on both engines, because this claim is engine-dependent.

    The watermark epoch is written by a statement whose INSERT path runs on one
    backend and whose conflict path runs on the other, so a single-engine
    fixture proves it for whichever engine happens to work.

    The agent id is unique per run, and that is not tidiness. A PostgreSQL
    database is reused between runs, so a fixed id inherits the previous run's
    watermark — the repair then finds nothing stale, writes nothing, and the
    test reads back an epoch some earlier build wrote. Measured: a mutant that
    disabled the epoch entirely still passed here, on state it had not created.
    """
    agent = f"{AGENT}-{uuid4()}"
    if request.param == "postgres":
        url = os.environ.get("TEST_POSTGRES_URL")
        if not url:
            pytest.skip("TEST_POSTGRES_URL is not set")
        db = await AsyncDatabase.postgres(url)
    else:
        db = await AsyncDatabase.sqlite(str(tmp_path / "either.db"))
    try:
        yield db, agent
    finally:
        try:
            await db.execute(
                "DELETE FROM conversation_history WHERE agent_id = ?", (agent,)
            )
        finally:
            await db.close()
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
    """The same sessions, ordered and sliced rather than walked by key.

    This is what the grouped paths do — ISOLATED's buffer and the archived view
    materialize the whole set, order it, and take a window — so it is the
    answer the keyset walk has to agree with. Two paging MODELS over one
    ordering, which is the disagreement worth catching: a keyset that resumes
    even slightly wrong lands somewhere a slice never would.
    """
    ordered = sort_sessions(list(rows))
    seen: List[str] = []
    offset = 0
    for _ in range(500):
        page = ordered[offset:offset + limit]
        if not page:
            return seen
        seen.extend(row["session_id"] for row in page)
        offset += len(page)
        if len(page) < limit:
            return seen
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
    """The differential. Two paging models over one ordering, one answer.

    The active view walks the projection by key in the engine; the archived
    view and ISOLATED materialize their sessions and slice. A caller cannot
    tell which served it, so they must not be able to tell.
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
async def test_the_stamps_a_page_orders_by_cannot_be_null(tmp_path):
    """The invariant is in the schema, not only in the writers.

    A NULL ``last_message_at`` would be a session the page cannot reach:
    ``last_message_at < ?`` is NULL for such a row, so it falls out of every
    page at every depth — a conversation that has vanished, which is the defect
    this ticket exists to remove. Admitting it in the predicate instead costs
    the index seek (measured, 17.7 ms against 0.11 ms on 200,000 sessions), so
    the column states what every writer already guarantees and the predicate
    stays seekable.
    """
    db, _projection = await _seeded(tmp_path, "not-null.db", _rows()[:1])
    try:
        for column in ("started_at", "last_message_at"):
            with pytest.raises(Exception) as raised:
                await db.execute(
                    "INSERT INTO conversation_sessions "
                    "(agent_id, session_id, started_at, last_message_at, "
                    "message_count, user_message_count) VALUES (?, ?, ?, ?, ?, ?)",
                    (AGENT, f"null-{column}",
                     None if column == "started_at" else "2026-05-01 09:00:00",
                     None if column == "last_message_at" else "2026-05-01 09:00:00",
                     1, 1),
                )
            assert "NOT NULL" in str(raised.value).upper(), raised.value
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_a_table_predating_the_not_null_stamps_is_replaced_and_rederived(
    tmp_path,
):
    """The upgrade path. This table is a cache, so the migration is to drop it.

    A database created before #2960 permits a NULL in the columns the page
    orders by, and there is no ``ALTER`` for that on SQLite. Rewriting it in
    place would be more expensive than deriving it again — nothing here is a
    record, and a repair reproduces every row.

    The watermarks are the other half: left alone they would describe rows that
    no longer exist and report the projection current over an empty table. The
    generation is rotated for exactly that, which is the mechanism already
    written for "the mechanism's shape moved".
    """
    from kestrel_sovereign.storage.async_conversation_store import AsyncConversationStore

    path = str(tmp_path / "legacy-shape.db")
    db = await AsyncDatabase.sqlite(path)
    try:
        # Put the pre-#2960 shape back, and fill it the way a shipped build
        # would have: rows, and a watermark saying they are accounted for.
        await db.execute("DROP TABLE conversation_sessions")
        await db.execute(
            "CREATE TABLE conversation_sessions ("
            " agent_id TEXT NOT NULL, session_id TEXT NOT NULL,"
            " started_at TIMESTAMP, last_message_at TIMESTAMP,"
            " message_count INTEGER NOT NULL DEFAULT 0,"
            " user_message_count INTEGER NOT NULL DEFAULT 0,"
            " first_user_message_id INTEGER, wake_source TEXT,"
            " PRIMARY KEY (agent_id, session_id))"
        )
        await db.execute(
            "INSERT INTO conversation_history "
            "(agent_id, role, content, metadata, created_at) "
            "VALUES (?, 'user', 'hello', '{}', '2026-05-01 09:00:00')",
            (AGENT,),
        )
        projection = ConversationSessionProjection(db, AGENT)
        await projection.repair()
        assert not await projection.is_stale(), "the fixture must start current"
        # A row only the old shape could hold, added AFTER the repair so it is
        # not discarded by it — and writing this table moves no change stamp,
        # so the watermark goes on saying the projection is current.
        await db.execute(
            "INSERT INTO conversation_sessions (agent_id, session_id, started_at,"
            " last_message_at, message_count, user_message_count)"
            " VALUES (?, 'ghost', NULL, NULL, 3, 2)",
            (AGENT,),
        )
        assert not await projection.is_stale()
        assert any(r["session_id"] == "ghost" for r in await projection.list())
    finally:
        await db.close()

    # ...and now a build that wants the stamps NOT NULL opens it.
    db = await AsyncDatabase.sqlite(path)
    try:
        assert not await db._column_accepts_null(
            "conversation_sessions", "last_message_at"
        ), "the migration left the old shape standing"
        projection = ConversationSessionProjection(db, AGENT)
        assert await projection.is_stale(), (
            "the table was replaced but the watermark still claimed it was "
            "current — a projection reporting itself true over nothing"
        )
        await projection.repair()
        listed = {row["session_id"] for row in await projection.list()}
        assert "ghost" not in listed, "a row from the retired table survived"
        assert listed == {"1"}, listed
        # ...and the page is still bounded. ``DROP TABLE`` takes the table's
        # indexes with it, so a migration that only restored the ROWS would
        # leave every page sorting the whole table — the cost this epic
        # removed, gone again on the one boot nobody watches.
        plan = await db.fetchall(
            "EXPLAIN QUERY PLAN SELECT session_id FROM conversation_sessions "
            f"WHERE agent_id = ? {session_order_sql(db.backend_type)} LIMIT ?",
            (AGENT, 10),
        )
        text = " ".join(str(row) for row in plan).upper()
        assert "IDX_CONVERSATION_SESSIONS_RECENT" in text, text
        assert "USE TEMP B-TREE" not in text, text
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_a_recreated_empty_cache_does_not_report_itself_current(tmp_path):
    """The sessions table can go WITHOUT its watermarks, and then it lies.

    A restore that carried the watermark tables but not the cache, or a hand
    recovery that dropped it, leaves the numbers matching: the ledger has not
    moved, the watermark still equals it, and ``is_stale()`` answers False over
    a table that is now empty. Before #2960 that cost nothing, because nothing
    read the table. Now it is a conversation list that serves nothing, for
    ever, beside intact history.

    Recreating it is therefore the same event as replacing it, and both rotate
    the generation.
    """
    path = str(tmp_path / "recreated.db")
    db = await AsyncDatabase.sqlite(path)
    try:
        await db.execute(
            "INSERT INTO conversation_history "
            "(agent_id, role, content, metadata, created_at) "
            "VALUES (?, 'user', 'hello', '{}', '2026-05-01 09:00:00')",
            (AGENT,),
        )
        projection = ConversationSessionProjection(db, AGENT)
        await projection.repair()
        assert [r["session_id"] for r in await projection.list()] == ["1"]
        assert not await projection.is_stale()
        # ...and the cache alone goes.
        await db.execute("DROP TABLE conversation_sessions")
    finally:
        await db.close()

    db = await AsyncDatabase.sqlite(path)
    try:
        projection = ConversationSessionProjection(db, AGENT)
        assert await projection.is_stale(), (
            "an empty cache reported itself current: the watermark still "
            "matched a ledger that had not moved, so nothing would ever "
            "rebuild it and the list would serve nothing for ever"
        )
        await projection.repair()
        assert [r["session_id"] for r in await projection.list()] == ["1"]
    finally:
        await db.close()


@pytest.mark.parametrize("stamp", ["not-a-date", "", "2026-13-45 99:99:99", "0"])
def test_a_cursor_whose_timestamp_is_not_one_is_refused(stamp):
    """A cursor is client-supplied text, and its keys are typed.

    Checking only that a key is a *string* lets a tampered token through, and
    what happens then is worse than an error on one backend and worse than that
    on the other: on PostgreSQL the string reaches asyncpg as a ``TIMESTAMP``
    parameter and raises out of the query, past the handler that turns a bad
    cursor into a 400; on SQLite it compares as text against canonical stamps
    and quietly selects the wrong page.
    """
    token = base64.urlsafe_b64encode(
        json.dumps({"v": 1, "kind": "keyset", "view": "active",
                    "k": [stamp, "sess-001"]}).encode()
    ).decode().rstrip("=")
    with pytest.raises(SessionCursorError):
        decode_session_cursor(token, "active")


def test_a_cursor_carrying_a_real_timestamp_is_accepted():
    """...and the refusal above is not simply refusing everything."""
    token = base64.urlsafe_b64encode(
        json.dumps(
            {"v": 1, "kind": "keyset", "view": "active",
             "k": ["2026-05-01 09:00:00", "sess-001"]}
        ).encode()
    ).decode().rstrip("=")
    assert decode_session_cursor(token, "active") == (
        "2026-05-01 09:00:00", "sess-001",
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "session_id",
    ["x" * (SESSION_ID_MAX_LENGTH + 1), "a\x00b", "\ud800", "\U0001F642" * 200],
    ids=["too-long", "nul", "lone-surrogate", "too-long-in-bytes"],
)
async def test_a_key_the_primary_key_cannot_hold_is_refused(tmp_path, session_id):
    """Openable is not the same question as storable, and both must hold.

    #2960 widened the list to key on whatever the grouper produced, so an id no
    longer has to fit Phase A's column charset — that is what keeps
    ``rasa_shim``'s ``sms:{sender}`` sessions listed. It does still have to fit
    this table's own primary key: PostgreSQL cannot hold a NUL in TEXT, cannot
    encode a lone surrogate, and refuses a composite B-tree entry past ~2704
    bytes.

    The consequence of getting that wrong is not a lost session. ``_store``
    raises inside the repair that runs on the first page of every conversation
    list, so ONE such row would fail the whole list for that agent until
    somebody edited the database by hand.
    """
    db = await AsyncDatabase.sqlite(str(tmp_path / "unstorable.db"))
    try:
        for index, text in enumerate(("first", "second")):
            await db.execute(
                "INSERT INTO conversation_history "
                "(agent_id, role, content, metadata, created_at) "
                "VALUES (?, 'user', ?, ?, ?)",
                (AGENT, text, json.dumps({"session_id": session_id}),
                 f"2026-05-01 09:0{index}:00"),
            )
        projection = ConversationSessionProjection(db, AGENT)
        # The point is that this does not RAISE, and that the list still works.
        await projection.repair()
        assert [r["session_id"] for r in await projection.page(limit=10)] == []
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_a_key_the_column_forbids_but_the_key_can_hold_is_listed(tmp_path):
    """...and the bound is not simply the column contract wearing a new name."""
    db = await AsyncDatabase.sqlite(str(tmp_path / "storable.db"))
    try:
        for index, text in enumerate(("first", "second")):
            await db.execute(
                "INSERT INTO conversation_history "
                "(agent_id, role, content, metadata, created_at) "
                "VALUES (?, 'user', ?, ?, ?)",
                (AGENT, text, json.dumps({"session_id": "sms:+15551234567"}),
                 f"2026-05-01 09:0{index}:00"),
            )
        projection = ConversationSessionProjection(db, AGENT)
        await projection.repair()
        assert [r["session_id"] for r in await projection.page(limit=10)] == [
            "sms:+15551234567"
        ]
    finally:
        await db.close()


@pytest.mark.parametrize(
    "session_id",
    ["\x01" * SESSION_ID_MAX_LENGTH, '"' * SESSION_ID_MAX_LENGTH,
     "\U0001F642" * (SESSION_ID_MAX_LENGTH // 4), "x" * SESSION_ID_MAX_LENGTH],
    ids=["control-chars", "quotes", "emoji", "plain"],
)
def test_every_token_the_server_can_mint_fits_the_parameter_that_takes_it(session_id):
    """A `next_cursor` the endpoint's own parameter refuses is a page boundary
    nothing can cross — this ticket's bug, wearing a 422.

    The worst case is not the longest id but the most heavily ESCAPED one: JSON
    turns an ASCII control character into a six-byte ``u`` escape, which
    is where the bound's multiplier comes from.
    """
    token = encode_session_cursor(
        {"last_message_at": "2026-01-01 00:00:00.123456", "session_id": session_id},
        "archived",
    )
    assert len(token) <= SESSION_CURSOR_MAX_LENGTH, (
        f"a {len(session_id)}-character id minted a {len(token)}-character "
        f"token, past the {SESSION_CURSOR_MAX_LENGTH} the endpoint accepts"
    )
    assert decode_session_cursor(token, "archived")[1] == session_id


@pytest.mark.parametrize("keys", [[None, "s"], ["2026-05-01 09:00:00", None], [None, None]])
def test_a_cursor_carrying_a_null_key_is_refused(keys):
    """No server mints one — both ordering keys are NOT NULL — and accepting it
    fails differently on each path: the Python continuation compares a string
    with ``None`` and raises (a 500), the SQL one compares against NULL and
    quietly serves an empty page."""
    token = base64.urlsafe_b64encode(
        json.dumps({"v": 1, "kind": "keyset", "view": "active", "k": keys}).encode()
    ).decode().rstrip("=")
    with pytest.raises(SessionCursorError):
        decode_session_cursor(token, "active")


async def _store_with(db):
    from kestrel_sovereign.storage.async_conversation_store import AsyncConversationStore

    return AsyncConversationStore(db, agent_id=AGENT)


@pytest.mark.asyncio
async def test_a_page_rebuilt_underneath_the_read_is_not_returned(tmp_path):
    """The readiness check and the read are ONE observation, or they are two.

    Another request's repair clears the table and commits its chunks as it
    goes, so a page read in that window is a partial one — and its last row
    comes back with ``next_cursor: null``, which is a truncated list wearing the
    shape of a complete one. The watermark is therefore read again afterwards
    and required to be identical; it moves only when a repair writes it.
    """
    from kestrel_sovereign.storage.async_conversation_store import ProjectionNotReady

    db = await AsyncDatabase.sqlite(str(tmp_path / "rebuilt-under.db"))
    try:
        await db.execute(
            "INSERT INTO conversation_history "
            "(agent_id, role, content, metadata, created_at) "
            "VALUES (?, 'user', 'hello', '{}', '2026-05-01 09:00:00')",
            (AGENT,),
        )
        store = await _store_with(db)
        projection = ConversationSessionProjection(db, AGENT)
        await projection.repair()

        # A repair lands between the check and the read, every time.
        real_page = projection.page
        rebuilt = {"count": 0}

        async def page_after_a_rebuild(*args, **kwargs):
            rebuilt["count"] += 1
            # What any real repair does to this table: the write counter moves.
            await db.execute(
                "UPDATE conversation_session_watermarks "
                "SET accounted_revision = accounted_revision + 1 WHERE agent_id = ?",
                (AGENT,),
            )
            return await real_page(*args, **kwargs)

        projection.page = page_after_a_rebuild
        with pytest.raises(ProjectionNotReady):
            await store._page_a_whole_projection(
                projection, limit=10, after=None, refresh=False
            )
        assert rebuilt["count"] > 1, "the read must be retried, not abandoned"
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_a_rebuild_that_lands_where_it_started_still_invalidates_the_page(
    tmp_path,
):
    """The ABA the write counter exists for, driven through the real rebuild.

    ``rebuild()`` invalidates the watermark, discards every projection row and
    walks again. Over a history that did not change in the meantime it arrives
    at a watermark IDENTICAL to the one it replaced — same generation, stamp,
    appends, through and target. A page read during the walk is partial, and a
    before/after comparison of those fields certifies it and returns the
    truncated page with ``next_cursor: null``.

    Nothing here is simulated: the concurrent pass is
    ``ConversationSessionProjection.rebuild`` itself.
    """
    from kestrel_sovereign.storage.async_conversation_store import ProjectionNotReady

    db = await AsyncDatabase.sqlite(str(tmp_path / "aba.db"))
    try:
        for index in range(4):
            await db.execute(
                "INSERT INTO conversation_history "
                "(agent_id, role, content, metadata, created_at) "
                "VALUES (?, 'user', ?, '{}', ?)",
                (AGENT, f"m{index}", f"2026-05-0{index + 1} 09:00:00"),
            )
        store = await _store_with(db)
        projection = ConversationSessionProjection(db, AGENT)
        await projection.repair()
        before = await projection.accounted()

        # A rebuild lands entirely between the fence's two reads.
        real_page = projection.page
        rebuilds = {"count": 0}

        async def page_then_rebuild(*args, **kwargs):
            rows = await real_page(*args, **kwargs)
            rebuilds["count"] += 1
            await ConversationSessionProjection(db, AGENT).rebuild()
            return rows

        projection.page = page_then_rebuild
        with pytest.raises(ProjectionNotReady):
            await store._page_a_whole_projection(
                projection, limit=10, after=None, refresh=False
            )

        after = await ConversationSessionProjection(db, AGENT).accounted()
        assert (after.generation, after.valid, after.stamp, after.appends,
                after.through, after.target) == (
            before.generation, before.valid, before.stamp, before.appends,
            before.through, before.target
        ), "the case only means something while every FIELD comes back identical"
        assert after.revision > before.revision, "the write counter must have moved"
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_a_watermark_deleted_and_recreated_under_the_read_invalidates_it(
    tmp_path,
):
    """The write counter is monotonic only while its ROW lives.

    ``purge_session_projection`` deletes that row, and so does the
    empty-history branch of the transcript pass. The next repair INSERTs a
    fresh one from the default, so two incarnations show the same count with
    the ledger generation unchanged — measured: revision 0, delete, repair,
    revision 0. A page read straddling that would compare equal and be
    certified over a cache that had been emptied and rebuilt underneath it.

    The epoch is what makes the pair unable to repeat.
    """
    from kestrel_sovereign.storage.async_conversation_store import ProjectionNotReady

    db = await AsyncDatabase.sqlite(str(tmp_path / "recreated-watermark.db"))
    try:
        await db.execute(
            "INSERT INTO conversation_history "
            "(agent_id, role, content, metadata, created_at) "
            "VALUES (?, 'user', 'hello', '{}', '2026-05-01 09:00:00')",
            (AGENT,),
        )
        store = await _store_with(db)
        projection = ConversationSessionProjection(db, AGENT)
        await projection.repair()
        before = await projection.accounted()

        real_page = projection.page
        cycles = {"count": 0}

        async def page_then_recreate(*args, **kwargs):
            rows = await real_page(*args, **kwargs)
            cycles["count"] += 1
            await db.execute(
                "DELETE FROM conversation_session_watermarks WHERE agent_id = ?",
                (AGENT,),
            )
            await ConversationSessionProjection(db, AGENT).repair()
            return rows

        projection.page = page_then_recreate
        with pytest.raises(ProjectionNotReady):
            await store._page_a_whole_projection(
                projection, limit=10, after=None, refresh=False
            )

        after = await ConversationSessionProjection(db, AGENT).accounted()
        assert after.revision == before.revision, (
            "the case only means something while the COUNTER comes back the same"
        )
        assert after.generation == before.generation, (
            "...and while the ledger generation is untouched, which it is: the "
            "watermark row went, the ledger did not"
        )
        assert after.epoch != before.epoch
        assert cycles["count"] > 1, "the read must be retried, not abandoned"
    finally:
        await db.close()


@pytest.mark.parametrize("session_id", ["a\x00b", "\ud800"])
def test_a_cursor_key_the_store_cannot_hold_is_refused(session_id):
    """A cursor key is client-supplied text, and text is not enough.

    A NUL or a lone surrogate is a string Python holds happily and the drivers
    refuse — and the refusal comes out of the QUERY, past the handler that
    turns a bad cursor into a 400 and into the one that reports a server fault.
    """
    token = base64.urlsafe_b64encode(
        json.dumps(
            {"v": 1, "kind": "keyset", "view": "active",
             "k": ["2026-05-01 09:00:00", session_id]},
            ensure_ascii=False,
        ).encode("utf-8", errors="surrogatepass")
    ).decode().rstrip("=")
    with pytest.raises(SessionCursorError):
        decode_session_cursor(token, "active")


@pytest.mark.asyncio
async def test_the_watermark_epoch_is_set_on_every_backend(tmp_path, sqlite_or_postgres):
    """The epoch has to be written where the ROW is actually created.

    On PostgreSQL it is not this statement that creates it: ``_claim()`` inserts
    the row first, as a thing to lock, so ``_record``'s INSERT always takes the
    conflict path. An insert-only epoch was therefore empty for every agent on
    PostgreSQL — the fence degenerated to the revision alone on the backend that
    matters most — while a SQLite check said it worked, because ``_claim()`` is
    a no-op there.
    """
    db, agent = sqlite_or_postgres
    await db.execute(
        "INSERT INTO conversation_history "
        "(agent_id, role, content, metadata, created_at) "
        "VALUES (?, 'user', 'hello', '{}', '2026-05-01 09:00:00')",
        (agent,),
    )
    projection = ConversationSessionProjection(db, agent)
    await projection.repair()

    first = await projection.accounted()
    assert first.epoch, "the watermark was written with no epoch at all"

    # Set ONCE: a later write must not move it, or it would be a second
    # revision counter answering the question the first one answers.
    await projection.repair()
    await db.execute(
        "INSERT INTO conversation_history "
        "(agent_id, role, content, metadata, created_at) "
        "VALUES (?, 'user', 'again', '{}', '2026-05-01 11:00:00')",
        (agent,),
    )
    await projection.repair()
    later = await projection.accounted()
    assert later.epoch == first.epoch, "the epoch moved while its row lived"
    assert later.revision > first.revision, "...and the counter did not move"


@pytest.mark.asyncio
async def test_an_emptied_cache_is_invalidated_even_with_no_ledger_row(tmp_path):
    """Rotating the generation can match no rows, and then it claims nothing.

    An agent whose projection was built before the triggers existed, or restored
    with one table and not the other, has a watermark and no slot-0 ledger row.
    Its generation is ``''`` and its stamp is 0 — exactly what a MISSING ledger
    reads back as — so the numbers agree, ``is_stale()`` answers false, and a
    freshly emptied cache is served for ever beside intact history.
    """
    path = str(tmp_path / "no-ledger.db")
    db = await AsyncDatabase.sqlite(path)
    try:
        await db.execute(
            "INSERT INTO conversation_history "
            "(agent_id, role, content, metadata, created_at) "
            "VALUES (?, 'user', 'hello', '{}', '2026-05-01 09:00:00')",
            (AGENT,),
        )
        projection = ConversationSessionProjection(db, AGENT)
        await projection.repair()
        # The state this is about, built exactly: a watermark recorded when
        # there was no ledger to compare against — generation '' and stamp 0,
        # which is precisely what a MISSING ledger row reads back as. That
        # coincidence is the bug; a watermark with a non-zero stamp would be
        # detected as stale by ordinary arithmetic and prove nothing here.
        await db.execute(
            "DELETE FROM conversation_history_changes WHERE agent_id = ?", (AGENT,)
        )
        await db.execute(
            "UPDATE conversation_session_watermarks "
            "SET accounted_generation = '', accounted_stamp = 0, "
            "accounted_appends = 0 WHERE agent_id = ?",
            (AGENT,),
        )
        assert not await projection.is_stale(), (
            "the case only means something while the projection reports itself "
            "CURRENT — that is the coincidence being defended against"
        )
        await db.execute("DROP TABLE conversation_sessions")
    finally:
        await db.close()

    db = await AsyncDatabase.sqlite(path)
    try:
        projection = ConversationSessionProjection(db, AGENT)
        assert await projection.is_stale(), (
            "an emptied cache reported itself current with no ledger to rotate"
        )
        await projection.repair()
        assert [r["session_id"] for r in await projection.list()] == ["1"]
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_a_generation_rotated_under_the_read_invalidates_the_page(tmp_path):
    """The fence asks the same question at both ends, generation included.

    Rotating the generation leaves the watermark ROW untouched, so a fence that
    compared only that row would pass — and return a page read from a cache
    that has since been retired, ending with ``next_cursor: null``. The two
    ends of a boundary have to ask one thing.
    """
    from kestrel_sovereign.storage.async_conversation_store import ProjectionNotReady

    db = await AsyncDatabase.sqlite(str(tmp_path / "rotated-under.db"))
    try:
        await db.execute(
            "INSERT INTO conversation_history "
            "(agent_id, role, content, metadata, created_at) "
            "VALUES (?, 'user', 'hello', '{}', '2026-05-01 09:00:00')",
            (AGENT,),
        )
        store = await _store_with(db)
        projection = ConversationSessionProjection(db, AGENT)
        await projection.repair()

        real_page = projection.page
        rotations = {"count": 0}

        async def page_then_rotate(*args, **kwargs):
            rows = await real_page(*args, **kwargs)
            rotations["count"] += 1
            # The watermark row is deliberately NOT touched.
            await db.execute(
                "UPDATE conversation_history_changes "
                "SET generation = ? WHERE agent_id = ? AND slot = 0",
                (f"incarnation-{rotations['count']}", AGENT),
            )
            return rows

        projection.page = page_then_rotate
        with pytest.raises(ProjectionNotReady):
            await store._page_a_whole_projection(
                projection, limit=10, after=None, refresh=False
            )
        assert rotations["count"] > 1, "the read must be retried, not abandoned"
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_a_watermark_from_a_retired_generation_does_not_serve_a_page(tmp_path):
    """``valid`` and ``complete`` can both hold over a cache that is gone.

    Recreating the sessions table rotates the change-stamp generation and
    leaves the watermark, which then describes rows that no longer exist. Both
    numbers still line up, so a check that asked only those two would page an
    empty cache and report the end of the list.
    """
    from kestrel_sovereign.storage.async_conversation_store import ProjectionNotReady

    db = await AsyncDatabase.sqlite(str(tmp_path / "old-generation.db"))
    try:
        await db.execute(
            "INSERT INTO conversation_history "
            "(agent_id, role, content, metadata, created_at) "
            "VALUES (?, 'user', 'hello', '{}', '2026-05-01 09:00:00')",
            (AGENT,),
        )
        store = await _store_with(db)
        projection = ConversationSessionProjection(db, AGENT)
        await projection.repair()
        accounted = await projection.accounted()
        assert accounted.valid and accounted.complete

        # What recreating the cache does: the rows go, the generation rotates,
        # the watermark stays.
        await db.execute("DELETE FROM conversation_sessions WHERE agent_id = ?", (AGENT,))
        await db.execute(
            "UPDATE conversation_history_changes SET generation = 'a-new-incarnation' "
            "WHERE agent_id = ? AND slot = 0",
            (AGENT,),
        )
        still = await projection.accounted()
        assert still.valid and still.complete, (
            "the case only means something while the FLAGS still say current"
        )

        assert await store._whole_watermark(projection) is None
        # A continuation repairs rather than refusing for ever, so it recovers.
        # ``(rows, fence)`` since #2961: search walks more than one page and
        # has to require they all came from the same projection revision.
        rows, fence = await store._page_a_whole_projection(
            projection, limit=10, after=None, refresh=False
        )
        assert [r["session_id"] for r in rows] == ["1"]
        assert fence == (await store._whole_watermark(projection)).fence
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_a_continuation_page_seeks_rather_than_rewalking(tmp_path):
    """Page nine must not cost what pages one through eight already paid.

    The disjunction a keyset expands to is not a range, so the engine cannot
    seek on it — it plans ``(agent_id=?)`` and walks every entry above the
    cursor. Measured on 200,000 sessions: 17.7 ms a page that way against
    0.11 ms with the redundant leading bound, which is ``O(rows already paged)``
    reappearing on the continuation after the first page was made bounded.
    """
    db, projection = await _seeded(tmp_path, "seek.db", _rows())
    try:
        first = await projection.page(limit=5)
        after = projection._bound_cursor(
            decode_session_cursor(encode_session_cursor(first[-1], "active"), "active")
        )
        from kestrel_sovereign.storage.session_grouping import session_cursor_clause

        clause, params = session_cursor_clause(db.backend_type, after)
        plan = await db.fetchall(
            "EXPLAIN QUERY PLAN SELECT session_id FROM conversation_sessions "
            f"WHERE agent_id = ? AND {clause} "
            f"{session_order_sql(db.backend_type)} LIMIT ?",
            (AGENT, *params, 5),
        )
    finally:
        await db.close()
    text = " ".join(str(row) for row in plan).upper()
    assert "LAST_MESSAGE_AT<" in text.replace(" ", ""), (
        f"the continuation did not seek on the ordering key: {text}"
    )
    assert "USE TEMP B-TREE" not in text, text


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
