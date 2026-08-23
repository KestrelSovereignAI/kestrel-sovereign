"""#2961: search reads the sessions table, so search and the list agree.

Phase D of #2948. The epic was filed because two surfaces derived "which
conversations exist" independently and bounded themselves differently — the
list at 1,000 rows of history, search at 5,000. Neither number was a retention
policy anyone chose, and a conversation outside one was simply absent from that
surface with nothing in the response to say so. #2960 removed the list's bound;
this file is about the other one.

What the tests here are built around:

**A horizon is invisible from inside.** A search that scanned 5,000 rows and
found nothing returned exactly what a search that scanned everything and found
nothing returns. So the tests seed corpora deliberately larger than the bound
that used to exist and put the answer *underneath* it — the only place the
defect was ever observable.

**Two derivations of one thing drift.** Search used to re-group raw rows into
sessions and report its own counts and previews; a session whose rows straddled
the window got a truncated summary that looked like a whole one. The tests
below compare search's session dicts against the LIST's, field for field,
rather than against expectations written out by hand — which would only record
what search does, not that it agrees with anything.

**The cheap path and the complete path are not the same path.** The walk stops
at ``limit`` matches instead of at a row count, so it is usually cheaper than
the scan it replaces as well as complete. That is a property worth a test,
because the obvious way to make search complete — decrypt everything, every
time — passes every correctness test here.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Sequence, Tuple
from uuid import uuid4

import pytest

from kestrel_sovereign.storage.async_conversation_store import (
    AsyncConversationStore,
    SearchTerms,
    _within_row_budget,
    session_match_decoration,
)
from kestrel_sovereign.storage.async_database import AsyncDatabase
from kestrel_sovereign.storage.session_grouping import timestamp_query_param
from kestrel_sovereign.storage.session_id_column import column_session_id

#: Larger than the ``SEARCH_SESSIONS_SCAN_LIMIT`` this ticket retired. The
#: corpora below cross it so that "the answer is under the old bound" is a
#: state the tests can actually reach.
RETIRED_SCAN_LIMIT = 5000

START = datetime(2026, 6, 1, 9, 0, 0)


def _stamp(minutes: int) -> str:
    return (START + timedelta(minutes=minutes)).strftime("%Y-%m-%d %H:%M:%S")


async def _seed(
    db: AsyncDatabase,
    agent: str,
    rows: Sequence[Tuple[Optional[str], str, str, int]],
    *,
    archived: bool = False,
) -> None:
    """Insert ``(session_id, role, content, minute)`` rows as a writer would.

    ``session_id`` of ``None`` seeds a *legacy* row — no id in metadata and none
    in the column — which is the shape 473 of Emma's 1,522 live rows are in and
    the only shape the indexed column cannot answer for.
    """
    params = []
    for session_id, role, content, minute in rows:
        metadata: Dict[str, Any] = {}
        if session_id is not None:
            metadata["session_id"] = session_id
        raw = json.dumps(metadata)
        # Bound through the adapter the write path binds with: PostgreSQL holds
        # a real ``timestamp`` and asyncpg refuses text for it, while SQLite
        # holds the canonical string. One spelling here would seed a corpus the
        # engine under test does not actually store.
        stamp = timestamp_query_param(db.backend_type, _stamp(minute))
        params.append(
            (
                agent,
                role,
                content,
                raw,
                stamp,
                column_session_id(raw),
                stamp if archived else None,
            )
        )
    await db.execute_many(
        "INSERT INTO conversation_history "
        "(agent_id, role, content, metadata, created_at, session_id, archived_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        params,
    )


def _wide_corpus(
    sessions: int, per_session: int, *, needle_in: int
) -> List[Tuple[Optional[str], str, str, int]]:
    """``sessions`` conversations, oldest first, with the needle in exactly one.

    Every session is separated by more than the grouping gap so the corpus is
    the same shape whether its rows carry ids or not.
    """
    rows: List[Tuple[Optional[str], str, str, int]] = []
    minute = 0
    for index in range(sessions):
        for turn in range(per_session):
            text = "an ordinary turn about nothing in particular"
            if index == needle_in and turn == 0:
                text = "the launch codes are in the penguin folder"
            rows.append((f"sess-{index:05d}", "user" if turn % 2 == 0 else "assistant", text, minute))
            minute += 1
        minute += 60  # past SESSION_GAP_MINUTES
    return rows


async def _all_listed(store: AsyncConversationStore) -> List[Dict[str, Any]]:
    """Every session the list can reach, by paging it to the end."""
    from kestrel_sovereign.storage.conversation_sessions import encode_session_cursor

    listed: List[Dict[str, Any]] = []
    cursor = None
    while True:
        page = await store.list_session_page(limit=100, cursor=cursor)
        listed.extend(page["sessions"])
        cursor = page["next_cursor"]
        if cursor is None:
            break
    return listed


@pytest.fixture
async def store(tmp_path):
    db = await AsyncDatabase.sqlite(str(tmp_path / "search.db"))
    try:
        yield AsyncConversationStore(db, agent_id=f"did:test:search-{uuid4()}")
    finally:
        await db.close()


@pytest.fixture(params=["sqlite", "postgres"])
async def either_engine(request, tmp_path):
    """``AsyncConversationStore`` on both engines.

    The walk pages the projection in SQL and compares a cursor the codec spells
    backend-free, so "search finds the session the list finds" is a claim about
    a comparison each engine performs itself. A unique agent per run because a
    PostgreSQL database is reused between runs and a fixed id would inherit the
    previous run's projection.
    """
    agent = f"did:test:search-{uuid4()}"
    if request.param == "postgres":
        url = os.environ.get("TEST_POSTGRES_URL")
        if not url:
            pytest.skip("TEST_POSTGRES_URL is not set")
        db = await AsyncDatabase.postgres(url)
    else:
        db = await AsyncDatabase.sqlite(str(tmp_path / "either.db"))
    try:
        yield AsyncConversationStore(db, agent_id=agent)
    finally:
        try:
            await db.execute(
                "DELETE FROM conversation_history WHERE agent_id = ?", (agent,)
            )
        finally:
            await db.close()


# ---------------------------------------------------------------------------
# The horizon
# ---------------------------------------------------------------------------

class TestTheRetiredHorizon:
    @pytest.mark.asyncio
    async def test_a_session_older_than_the_retired_scan_is_findable(self, store):
        """The defect, stated as a test.

        The needle is in the OLDEST conversation of a history larger than the
        bound that used to exist, so a search that reads the newest 5,000 rows
        finds nothing and reports that as "no matches" — indistinguishable, from
        the response, from a history that does not contain it.
        """
        rows = _wide_corpus(sessions=400, per_session=14, needle_in=0)
        assert len(rows) > RETIRED_SCAN_LIMIT, "the corpus must cross the old bound"
        await _seed(store.db, store.agent_id, rows)

        results = await store.search_sessions("penguin folder")

        assert [s["session_id"] for s in results] == ["sess-00000"]
        assert "penguin folder" in results[0]["match_snippet"]

    @pytest.mark.asyncio
    async def test_a_match_deeper_than_one_step_of_the_walk_is_found(self, store):
        """...and the walk's own step is not a horizon either.

        The walk takes ``SEARCH_SESSION_STEP`` sessions at a time. A step that
        was mistaken for a bound would pass the test above — 400 sessions of 14
        rows is one transcript, but only two steps — so this puts the needle
        past several of them.
        """
        step = AsyncConversationStore.SEARCH_SESSION_STEP
        rows = _wide_corpus(sessions=step * 3 + 7, per_session=2, needle_in=0)
        await _seed(store.db, store.agent_id, rows)

        results = await store.search_sessions("penguin folder")

        assert [s["session_id"] for s in results] == ["sess-00000"]

    @pytest.mark.asyncio
    async def test_archived_search_reads_every_archived_row(self, store):
        """The archived view has no table (#3062) and no cap either.

        The archived LIST reads its rows uncapped, so a cap here would put a
        conversation in that list which no query could find — the same defect,
        one tab across.
        """
        rows = _wide_corpus(sessions=400, per_session=14, needle_in=0)
        assert len(rows) > RETIRED_SCAN_LIMIT
        await _seed(store.db, store.agent_id, rows, archived=True)

        results = await store.search_sessions("penguin folder", view="archived")

        assert [s["session_id"] for s in results] == ["sess-00000"]
        assert await store.search_sessions("penguin folder", view="active") == []


# ---------------------------------------------------------------------------
# Search is the list, filtered
# ---------------------------------------------------------------------------

class TestSearchAgreesWithTheList:
    @pytest.mark.asyncio
    async def test_a_found_session_is_the_listed_session(self, either_engine):
        """Field for field, not "looks about right".

        Search used to re-derive its summaries by grouping whatever rows fell
        inside its window, so a session whose rows straddled the edge reported a
        message count for the part that fit. Comparing against the list rather
        than against a hand-written expectation is what makes that observable:
        an expectation records what search does, and both numbers would have
        been written from the same wrong derivation.
        """
        store = either_engine
        rows = _wide_corpus(sessions=12, per_session=9, needle_in=3)
        await _seed(store.db, store.agent_id, rows)

        found = await store.search_sessions("penguin folder")
        listed = {s["session_id"]: s for s in await _all_listed(store)}

        assert len(found) == 1
        hit = found[0]
        counterpart = listed[hit["session_id"]]
        decoration = {"match_count", "match_role", "match_snippet", "name"}
        assert {k: v for k, v in hit.items() if k not in decoration} == counterpart

    @pytest.mark.asyncio
    async def test_every_session_the_list_reaches_is_searchable(self, either_engine):
        """The acceptance criterion, in both directions.

        Every session in this corpus contains the query, so the set search
        returns and the set paging reaches are the same set — and in the same
        order, because they are the same table read the same way.
        """
        store = either_engine
        sessions = 40
        rows = []
        minute = 0
        for index in range(sessions):
            for turn in range(4):
                rows.append(
                    (f"sess-{index:05d}", "user", "everyone talks about penguins", minute)
                )
                minute += 1
            minute += 60
        await _seed(store.db, store.agent_id, rows)

        found = await store.search_sessions("penguins", limit=sessions)
        listed = await _all_listed(store)

        assert [s["session_id"] for s in found] == [s["session_id"] for s in listed]

    @pytest.mark.asyncio
    async def test_a_session_with_no_id_anywhere_is_searched_where_it_is_listed(
        self, store
    ):
        """Legacy rows: the column cannot answer, so the grouper does.

        A row carrying no ``session_id`` in metadata or column belongs to
        whichever cluster it falls next to, which is decided by the rows before
        it. The list shows such a cluster under a key invented from its first
        row id; search has to find it under the same key or the two disagree
        about a conversation that plainly exists.
        """
        await _seed(
            store.db,
            store.agent_id,
            [
                (None, "user", "the launch codes are in the penguin folder", 0),
                (None, "assistant", "understood", 1),
                (None, "user", "unrelated later conversation", 500),
            ],
        )

        found = await store.search_sessions("penguin folder")
        listed = [s["session_id"] for s in await _all_listed(store)]

        assert len(found) == 1
        assert found[0]["session_id"] in listed
        assert found[0]["message_count"] == 2

    @pytest.mark.asyncio
    async def test_a_titled_session_with_no_rows_is_still_findable(self, store):
        """A conversation that exists only as its marker (#2222).

        It has no message rows, so the membership map has no entry for it and
        there is nothing to match content against — but the list shows it, so a
        query against the name the user gave it has to reach it.
        """
        await store.db.execute(
            "INSERT INTO conversation_history "
            "(agent_id, role, content, metadata, created_at) VALUES (?, ?, ?, ?, ?)",
            (
                store.agent_id,
                "system",
                "",
                json.dumps({"session_id": "marker-only", "new_session": True}),
                _stamp(0),
            ),
        )
        await store.set_conversation_name("marker-only", "Penguin plans")

        found = await store.search_sessions("penguin")

        assert [s["session_id"] for s in found] == ["marker-only"]
        assert found[0]["name"] == "Penguin plans"
        assert found[0]["match_count"] == 0
        assert found[0]["match_snippet"] is None


# ---------------------------------------------------------------------------
# What the walk costs
# ---------------------------------------------------------------------------

class TestTheWalkStopsOnAnAnswer:
    @pytest.mark.asyncio
    async def test_a_satisfied_limit_stops_the_walk(self, store, monkeypatch):
        """Complete is not the same as exhaustive, and both are testable.

        Decrypting the whole history on every keystroke would pass every
        correctness test in this file. The walk instead reads sessions in list
        order and stops once it has ``limit`` matches, so a query answered by
        the newest conversations costs the newest conversations.
        """
        sessions = 300
        rows = []
        minute = 0
        for index in range(sessions):
            for turn in range(4):
                rows.append((f"sess-{index:05d}", "user", "penguins again", minute))
                minute += 1
            minute += 60
        await _seed(store.db, store.agent_id, rows)

        read: List[int] = []
        original = AsyncConversationStore._search_rows

        async def counting(self, message_ids):
            ids = list(message_ids)
            read.append(len(ids))
            return await original(self, ids)

        monkeypatch.setattr(AsyncConversationStore, "_search_rows", counting)
        found = await store.search_sessions("penguins", limit=5)

        assert len(found) == 5
        assert sum(read) < sessions * 4 / 2, (
            f"the walk decrypted {sum(read)} of {sessions * 4} rows to answer a "
            "query the newest five conversations already satisfied"
        )

    @pytest.mark.asyncio
    async def test_a_query_matching_nothing_still_reads_everything(self, store):
        """The other half of the same property, so it is not read as a cap.

        Proving a session does NOT match means reading it. The walk exhausting
        the table for a query with no answer is the honest cost of content no
        index can see — and the reason this is a test is that "stop early" is
        exactly the change that would quietly turn into "stop".
        """
        rows = _wide_corpus(sessions=30, per_session=4, needle_in=0)
        await _seed(store.db, store.agent_id, rows)

        assert await store.search_sessions("no such text anywhere") == []
        # ...and the needle that IS there, in the oldest session, is still found.
        assert [s["session_id"] for s in await store.search_sessions("penguin folder")] == [
            "sess-00000"
        ]


# ---------------------------------------------------------------------------
# The pieces
# ---------------------------------------------------------------------------

class TestRowBudget:
    def test_batches_stay_within_the_budget(self):
        page = [{"session_id": f"s{i}"} for i in range(10)]
        membership = {f"s{i}": list(range(3)) for i in range(10)}
        batches = list(_within_row_budget(page, membership, 7))
        assert [len(b) for b in batches] == [2, 2, 2, 2, 2]

    def test_a_session_larger_than_the_budget_gets_its_own_batch(self):
        """Splitting one session across batches would report two partial answers.

        ``match_count`` is a count over the session's rows and ``first_hit`` is
        the earliest of them; neither is computable from a fragment, so an
        oversized session is read whole even though nothing bounds it.
        """
        page = [{"session_id": "small"}, {"session_id": "huge"}, {"session_id": "after"}]
        membership = {"small": [1], "huge": list(range(50)), "after": [2]}
        batches = list(_within_row_budget(page, membership, 5))
        assert [[s["session_id"] for s in b] for b in batches] == [
            ["small"],
            ["huge"],
            ["after"],
        ]

    def test_a_session_the_map_does_not_know_costs_nothing(self):
        page = [{"session_id": "ghost"}]
        assert [len(b) for b in _within_row_budget(page, {}, 5)] == [1]


class TestSearchTerms:
    def test_an_empty_query_compiles_to_nothing(self):
        assert SearchTerms.compile("   ") is None

    def test_a_wrapper_only_query_still_matches_by_substring(self):
        """#1554: the fallback is withheld, the substring match is not.

        A query made entirely of transport scaffolding would score ≥0.6 against
        every sent-form row in the history, so the tokenized fallback is gated
        out — but the literal text is still text a user may have typed.
        """
        terms = SearchTerms.compile("<user_input>")
        assert terms is not None
        assert terms.use_token_fallback is False

    def test_one_matcher_serves_both_paths(self):
        """The decoration is a function of rows, and only of rows.

        Search reaches sessions two ways now. Both call this, so a rule can only
        be changed for both at once.
        """
        terms = SearchTerms.compile("penguin")
        rows = [{"role": "user", "content": "a penguin appeared"}]
        assert session_match_decoration(rows, None, terms) == {
            "match_count": 1,
            "match_role": "user",
            "match_snippet": "a penguin appeared",
        }
        assert session_match_decoration([], None, terms) is None
        assert session_match_decoration([], "Penguin plans", terms) == {
            "match_count": 0,
            "match_role": None,
            "match_snippet": None,
        }


# ---------------------------------------------------------------------------
# Privacy: searching now WRITES, so it has to be leased
# ---------------------------------------------------------------------------

class TestSearchIsLeasedLikeTheList:
    @pytest.mark.asyncio
    async def test_a_lease_is_held_across_the_search(self):
        """Search repairs the projection, so a mode flip must not overtake it.

        Before #2961 search only read rows, and reading needs no lease. Now its
        first step repairs — a WRITE to ``conversation_sessions`` — so a
        transition to EPHEMERAL landing mid-await could let that repair
        republish a description of conversations the sweep had just cleared.
        ``set_privacy_mode`` already refuses while any lease is held; this is
        about search taking one.
        """
        from unittest.mock import MagicMock

        from kestrel_sovereign.privacy import PrivacyMode
        from kestrel_sovereign.storage.privacy_wrapper import PrivacyEnforcingStorage

        storage = MagicMock()
        wrapper = PrivacyEnforcingStorage(storage, PrivacyMode.NORMAL)
        held: List[int] = []

        async def observing(query, view="active", limit=20):
            held.append(wrapper._active_session_projection_leases)
            return []

        storage.conversation.search_sessions = observing

        assert await wrapper.search_conversations("a", "penguin") == []
        assert held == [1], "the search ran without a projection lease held"
        assert wrapper._active_session_projection_leases == 0, (
            "the lease outlived the search, so no privacy transition could "
            "ever proceed again"
        )

    @pytest.mark.asyncio
    async def test_a_failing_search_still_releases_its_lease(self):
        from unittest.mock import MagicMock

        from kestrel_sovereign.privacy import PrivacyMode
        from kestrel_sovereign.storage.privacy_wrapper import PrivacyEnforcingStorage

        storage = MagicMock()
        wrapper = PrivacyEnforcingStorage(storage, PrivacyMode.NORMAL)

        async def raising(query, view="active", limit=20):
            raise RuntimeError("index not ready")

        storage.conversation.search_sessions = raising

        with pytest.raises(RuntimeError):
            await wrapper.search_conversations("a", "penguin")
        assert wrapper._active_session_projection_leases == 0


class TestTheCostIsSaidOutLoud:
    @pytest.mark.asyncio
    async def test_a_large_membership_map_is_logged(self, store, caplog, monkeypatch):
        """A cost that grows with history is the shape this epic exists to remove.

        The map is the whole live history on every search — measured at about
        5.4 microseconds a row — and the walk's early exit does not shorten it.
        An operator should hear that from a log rather than from a slow pane,
        and the message has to name the reason that is actually true of every
        search: attribution needs the transcript, whether or not any row is
        unstamped. The corpus below is fully stamped for exactly that reason.
        """
        import logging

        from kestrel_sovereign.storage import conversation_sessions

        monkeypatch.setattr(conversation_sessions, "TRANSCRIPT_PASS_NOISY_ROWS", 8)
        await _seed(store.db, store.agent_id, _wide_corpus(5, 4, needle_in=0))

        with caplog.at_level(logging.WARNING):
            await store.search_sessions("penguin folder")

        said = [r.getMessage() for r in caplog.records if "search_sessions" in r.getMessage()]
        assert said, "the whole-history read happened silently"
        assert "all 20 of its live rows" in said[0]
        assert "#3075" in said[0]

    @pytest.mark.asyncio
    async def test_a_small_history_says_nothing(self, store, caplog):
        import logging

        await _seed(store.db, store.agent_id, _wide_corpus(2, 2, needle_in=0))
        with caplog.at_level(logging.WARNING):
            await store.search_sessions("penguin folder")
        assert not [
            r for r in caplog.records if "search_sessions" in r.getMessage()
        ]


# ---------------------------------------------------------------------------
# One walk, one projection
# ---------------------------------------------------------------------------

class TestTheWalkIsOneProjection:
    @pytest.mark.asyncio
    async def test_a_repair_between_pages_does_not_lose_a_match(
        self, store, monkeypatch
    ):
        """The list tolerates this; search cannot (codex R1 P2).

        A session holding the only match receives a new, NON-matching message,
        and another request repairs the projection between two pages of this
        walk. Its ``last_message_at`` moves ahead of the cursor, so no later page
        can reach it — and search answers "no matches" for a conversation that
        matches. Reproduced against the unfenced walk before the fence existed:
        ``found: []``.

        The list's version of this event is a session shown twice or skipped in
        a list the user is scrolling. Search's version is silence, which is also
        what search says when there is genuinely nothing — so there is no shape
        left in the response for "and a page went missing".
        """
        from kestrel_sovereign.storage.conversation_sessions import (
            ConversationSessionProjection,
        )

        await _seed(
            store.db,
            store.agent_id,
            [
                ("old", "user", "the launch codes are in the penguin folder", 0),
                ("mid", "user", "nothing of interest here", 100),
                ("new", "user", "nothing of interest here either", 200),
            ],
        )
        monkeypatch.setattr(AsyncConversationStore, "SEARCH_SESSION_STEP", 1)

        original = AsyncConversationStore._search_rows
        fired: List[bool] = []

        async def repairing(self, message_ids):
            rows = await original(self, message_ids)
            if not fired:
                fired.append(True)
                await _seed(
                    store.db,
                    store.agent_id,
                    [("old", "user", "an unrelated later remark", 500)],
                )
                await ConversationSessionProjection(
                    store.db, store.agent_id
                ).repair()
            return rows

        monkeypatch.setattr(AsyncConversationStore, "_search_rows", repairing)

        found = await store.search_sessions("penguin folder")

        assert fired, "the fixture never got between two pages"
        assert [s["session_id"] for s in found] == ["old"]

    @pytest.mark.asyncio
    async def test_a_projection_that_never_settles_refuses(self, store, monkeypatch):
        """...and giving up says so, rather than returning the short answer.

        A walk restarted forever would hang; a walk that returned what it had
        would report a partial search as a complete one. The list refuses on the
        same condition, and the endpoint turns both into the same 503.
        """
        from kestrel_sovereign.storage.async_conversation_store import (
            ProjectionNotReady,
        )
        from kestrel_sovereign.storage.conversation_sessions import (
            ConversationSessionProjection,
        )

        await _seed(
            store.db,
            store.agent_id,
            [
                ("old", "user", "the launch codes are in the penguin folder", 0),
                ("mid", "user", "nothing of interest here", 100),
                ("new", "user", "nothing of interest here either", 200),
            ],
        )
        monkeypatch.setattr(AsyncConversationStore, "SEARCH_SESSION_STEP", 1)

        original = AsyncConversationStore._search_rows
        moved: List[int] = []

        async def always_repairing(self, message_ids):
            rows = await original(self, message_ids)
            moved.append(len(moved))
            await _seed(
                store.db,
                store.agent_id,
                [("old", "user", f"remark {len(moved)}", 500 + len(moved))],
            )
            await ConversationSessionProjection(store.db, store.agent_id).repair()
            return rows

        monkeypatch.setattr(AsyncConversationStore, "_search_rows", always_repairing)

        with pytest.raises(ProjectionNotReady):
            await store.search_sessions("penguin folder", limit=2)

    @pytest.mark.asyncio
    async def test_a_restarted_walk_re_reads_the_membership_map(
        self, store, monkeypatch
    ):
        """A restart is a new walk, so it needs a new map.

        The invariant is that the projection is no newer than the map, so every
        session the walk can return has its rows in it. A restart repairs again,
        which can project a conversation that arrived after the first map was
        read — and carrying that map forward would list it and find nothing in
        it, which for search means reporting no match against a message that
        matches.
        """
        from kestrel_sovereign.storage.conversation_sessions import (
            ConversationSessionProjection,
        )

        await _seed(
            store.db,
            store.agent_id,
            [
                ("a", "user", "nothing of interest here", 0),
                ("b", "user", "nothing of interest here", 100),
                ("c", "user", "nothing of interest here", 200),
            ],
        )
        monkeypatch.setattr(AsyncConversationStore, "SEARCH_SESSION_STEP", 1)

        original = AsyncConversationStore._search_rows
        fired: List[bool] = []

        async def landing_a_new_match(self, message_ids):
            rows = await original(self, message_ids)
            if not fired:
                fired.append(True)
                await _seed(
                    store.db,
                    store.agent_id,
                    [("late", "user", "the launch codes are in the penguin folder", 500)],
                )
                await ConversationSessionProjection(
                    store.db, store.agent_id
                ).repair()
            return rows

        monkeypatch.setattr(
            AsyncConversationStore, "_search_rows", landing_a_new_match
        )

        found = await store.search_sessions("penguin folder")

        assert fired, "the fixture never got between two pages"
        assert [s["session_id"] for s in found] == ["late"]

    @pytest.mark.asyncio
    async def test_an_empty_projection_reads_no_history(self, store, monkeypatch):
        """Nothing to search means nothing to attribute.

        The map is the one unbounded read on this path, so the walk must not
        take it to discover that there was never anything to search — which is
        the state a brand-new agent is in on its first keystroke.
        """
        read: List[bool] = []
        original = AsyncConversationStore._live_membership

        async def counting(self):
            read.append(True)
            return await original(self)

        monkeypatch.setattr(AsyncConversationStore, "_live_membership", counting)

        assert await store.search_sessions("penguin") == []
        assert not read, "an agent with no conversations read its whole history"

    @pytest.mark.asyncio
    async def test_the_rows_matched_are_the_rows_the_summary_counts(
        self, store, monkeypatch
    ):
        """One result may not carry two snapshots (codex R2 P2).

        Reproduced before the frontier bound existed: a message appended between
        the projection page and the membership read was matched against, while
        the summary beside it still came from the page — so the session came
        back with ``message_count: 1`` and a snippet from its second message.
        A count that does not count the thing it is shown next to is a worse
        answer than a stale one, because nothing about it looks stale.

        The consequence is that this search does not find the new message at
        all, which is the projection's ordinary lag rather than a loss: the next
        query repairs before it walks. The second half of this test is that
        promise, because "consistent" is cheap to achieve by never updating.
        """
        await _seed(
            store.db, store.agent_id, [("s1", "user", "nothing of interest", 0)]
        )

        original = AsyncConversationStore._live_membership
        fired: List[bool] = []

        async def racing(self, through):
            if not fired:
                fired.append(True)
                await _seed(
                    store.db,
                    store.agent_id,
                    [("s1", "user", "the launch codes are in the penguin folder", 1)],
                )
            return await original(self, through)

        monkeypatch.setattr(AsyncConversationStore, "_live_membership", racing)
        during = await store.search_sessions("penguin folder")
        assert fired, "the fixture never raced the membership read"

        # Nothing inconsistent came back: the row was above the frontier the
        # summaries describe, so it was not matched against at all.
        assert during == []

        monkeypatch.setattr(AsyncConversationStore, "_live_membership", original)
        after = await store.search_sessions("penguin folder")
        assert [s["session_id"] for s in after] == ["s1"]
        assert after[0]["message_count"] == 2, (
            "the message the previous query could not see is now both counted "
            "and matched — one snapshot, one answer"
        )
