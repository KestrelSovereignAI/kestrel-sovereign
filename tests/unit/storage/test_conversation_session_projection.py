"""#2959: the ``conversation_sessions`` projection, and the stamp that polices it.

The projection records which session each row was filed under so a reader need
not re-derive it. Three things can go wrong with a record like that, and this
file is built around all three.

**It can disagree with the algorithm it replaces.** The differential test runs
:func:`group_messages_into_sessions` + :func:`coalesce_sessions_by_session_id`
over a whole transcript and requires the projection to have reached the same
place, field by field. It is asked of **two** corpora, because the projection has
two derivations and only one of them is exercised by each:

* ``MODERN_CORPUS`` — every row carries a session id, which is the state Phase
  A's backfill leaves and the only state current write paths produce. The
  projection reads each session's own rows, and the differential is the strong
  claim: the projection groups ONE session while the reference groups the WHOLE
  transcript, where clusters split on gaps and are re-merged by id.
* ``CORPUS`` — the same shapes plus rows carrying no session id at all. Those
  belong to whichever cluster they fall next to, so a stamped user turn followed
  by an unstamped reply is *two* messages in that session. Reading only the rows
  that carry the id reports one, which is the disagreement this corpus exists to
  catch.

**It can go stale without saying so.** No write path maintains this table, so
*every* mutation leaves it stale — and staleness must be detectable with no
cooperation from the code that caused it. Each of insert / soft-delete / restore
/ archive / purge gets its own case, driven through the real store method. So do
the two mutations no aggregate can see: re-homing a row from one session to
another, and flipping the metadata flags that decide the preview. Both arrive
through ``update_message_metadata``, whose key set is the caller's, and both are
seen because the *database* counts row events rather than any write path
remembering to.

**Its watermark can run ahead of what it accounted for.** A watermark that has
run ahead is a silent gap: the projection claims to be current over rows it
never read. The crash and concurrency cases below are the ones that check the
direction, and each carries a bounded timeout — a lock or interleaving test that
can hang cannot report the bug it exists to catch.

The concurrency cases are about the fence rather than about detection, and the
distinction is what the earlier design got wrong. A superseded repair used to
write its rows first and discover it had lost afterwards, so between those two
moments — and forever, if it died in between — a published watermark stood over
rows a stale pass had overwritten. So the cases below park a repair on both
sides of its write: after ``_refresh``, and after deriving a row but *before*
``_store``, which is the ordering only a fence survives.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List

import pytest

from kestrel_sovereign.storage.async_conversation_store import AsyncConversationStore
from kestrel_sovereign.storage.async_database import AsyncDatabase
from kestrel_sovereign.storage.conversation_sessions import (
    CURRENT,
    DEFERRED,
    INCREMENTAL,
    PROJECTION_INPUT_COLUMNS,
    REBUILT,
    ConversationSessionProjection,
    SessionWatermark,
)
from kestrel_sovereign.storage.session_grouping import (
    coalesce_sessions_by_session_id,
    coerce_session_timestamp,
    group_messages_into_sessions,
)
from kestrel_sovereign.storage.session_id_column import is_stampable_session_id

AGENT = "did:test:session-projection"
UUID_A = "5b2e7d10-1c3f-4a55-8c0d-000000000001"
UUID_B = "5b2e7d10-1c3f-4a55-8c0d-000000000002"
UUID_C = "5b2e7d10-1c3f-4a55-8c0d-000000000003"
UUID_WAKE = "5b2e7d10-1c3f-4a55-8c0d-000000000004"
UUID_MARKER_ONLY = "5b2e7d10-1c3f-4a55-8c0d-000000000005"
OUTSIDE_THE_CONTRACT = "did:x:1"

BASE = datetime(2026, 3, 1, 9, 0, 0, tzinfo=timezone.utc)

#: How long an interleaved repair may be parked before the test rescues itself.
#: Bounded because a hung interleaving test is a timed-out CI job with no failing
#: assertion — indistinguishable from flake, which is how a 23-hour advisory-lock
#: wedge went unreported during Phase A.
INTERLEAVE_BUDGET = 10.0


def _accounting(watermark: SessionWatermark) -> tuple:
    """What a watermark *claims*, without its epoch.

    The epoch is an ownership token whose value is nobody's contract — it counts
    repairs, so pinning it in an assertion would make every case that adds one
    fail for a reason unrelated to what it tests. What the projection promises is
    the other three fields.
    """
    return (watermark.valid, watermark.frontier, watermark.changes)


def _at(minutes: int) -> str:
    """A stored timestamp, in the ISO form SQLite history actually holds."""
    return (BASE + timedelta(minutes=minutes)).replace(tzinfo=None).isoformat()


async def _seed(
    db: AsyncDatabase,
    rows: List[Dict[str, Any]],
    agent_id: str = AGENT,
) -> None:
    """Write rows exactly as they would be stored, column included.

    Hand-written rather than driven through ``add_conversation`` because these
    corpora need controlled timestamps (the gap rule is the whole point) and
    shapes no live writer produces any more — legacy rows with no session id,
    ids outside the column's contract. The column is stamped the same way every
    write path stamps it, so nothing here is a shape the store could not hold.
    """
    from kestrel_sovereign.storage.session_id_column import column_session_id

    for row in rows:
        metadata = json.dumps(row["metadata"]) if row.get("metadata") else None
        await db.execute(
            "INSERT INTO conversation_history "
            "(agent_id, role, content, metadata, session_id, created_at, "
            "deleted_at, archived_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                agent_id,
                row.get("role", "user"),
                row.get("content", "text"),
                metadata,
                column_session_id(metadata),
                row["created_at"],
                row.get("deleted_at"),
                row.get("archived_at"),
            ),
        )


async def _live_history(db: AsyncDatabase, agent_id: str = AGENT) -> List[Dict[str, Any]]:
    """The corpus the read path would hand the grouper: live rows, id order."""
    rows = await db.fetchall(
        "SELECT id, role, content, metadata, created_at FROM conversation_history "
        "WHERE agent_id = ? AND deleted_at IS NULL AND archived_at IS NULL "
        "ORDER BY id ASC",
        (agent_id,),
    )
    return [
        {
            "id": row[0],
            "role": row[1],
            "content": row[2],
            "metadata": json.loads(row[3]) if row[3] else {},
            "created_at": row[4],
        }
        for row in rows
    ]


def _reference_sessions(history: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """What the grouper says about a whole transcript, keyed by session id.

    ``first_user_message_id`` is recovered from the preview independently of
    how the projection finds it: the grouper reports the previewed TEXT, and
    each row in these corpora carries unique text, so the row it came from is
    unambiguous. That independence is the point — a reference that reused the
    projection's own trick could not catch the projection getting it wrong.
    """
    grouped = coalesce_sessions_by_session_id(
        group_messages_into_sessions(history, keep_empty_markers=True)
    )
    by_content: Dict[Any, Any] = {}
    for message in history:
        by_content.setdefault(message["content"], message["id"])
    reference = {}
    for session in grouped:
        preview = session["preview_content"]
        reference[str(session["session_id"])] = {
            # Compared as instants, not spellings: the stored column is a
            # TIMESTAMP, and PostgreSQL hands it back as a ``datetime`` where
            # SQLite hands back the ISO text it was given. Comparing the text
            # would make this file's claim true of one engine only.
            "started_at": coerce_session_timestamp(session["started_at"]),
            "last_message_at": coerce_session_timestamp(session["last_message_at"]),
            "message_count": session["message_count"],
            "user_message_count": session["user_message_count"],
            "first_user_message_id": (
                None if preview is None else by_content[preview]
            ),
            "wake_source": session["preview_wake_source"],
        }
    return reference


def _stored(row: Dict[str, Any]) -> Dict[str, Any]:
    """A stored projection row in the reference's shape."""
    return {
        "started_at": coerce_session_timestamp(row["started_at"]),
        "last_message_at": coerce_session_timestamp(row["last_message_at"]),
        "message_count": row["message_count"],
        "user_message_count": row["user_message_count"],
        "first_user_message_id": row["first_user_message_id"],
        "wake_source": row["wake_source"],
    }


async def _assert_agrees_with_the_grouper(
    db: AsyncDatabase,
    projection: ConversationSessionProjection,
    agent_id: str = AGENT,
) -> Dict[str, Dict[str, Any]]:
    """The differential claim, in the two directions it can be broken.

    * Every session the projection claims, the grouper also finds, with the
      same boundaries, counts, pointer and wake source.
    * Every session the grouper finds whose id the column may hold, the
      projection claims. Absence is only ever an id outside that contract —
      the legacy row-id keys and the ``did:x:1`` shapes — which is Phase A's
      invariant carried forward: silent where it must be, never wrong.
    """
    history = await _live_history(db, agent_id)
    reference = _reference_sessions(history)
    stored = {row["session_id"]: _stored(row) for row in await projection.list()}

    for session_id, projected in stored.items():
        assert session_id in reference, (
            f"projection claims session {session_id!r}, which the grouper does "
            f"not report at all: {sorted(reference)}"
        )
        assert projected == reference[session_id], session_id

    expected = {
        session_id for session_id in reference if is_stampable_session_id(session_id)
    }
    assert set(stored) == expected
    return stored


async def _assert_repaired_projection_is_true(
    db: AsyncDatabase,
    projection: ConversationSessionProjection,
    agent_id: str = AGENT,
) -> None:
    """Repair, then hold the repaired table to the rows it describes.

    Every mutation case ends here. Under this contract the projection is allowed
    to be stale *until* something repairs it, so the claim is about the state
    after a repair: the stamp agrees, every pointer names a live user row, and
    the whole table still agrees with the grouper. A pointer at a soft-deleted,
    archived or purged row is checked against ``conversation_history`` itself —
    a claim about a row is only worth what the row says.
    """
    await projection.repair()
    assert not await projection.is_stale(), (
        "a repair left the projection still reporting itself stale"
    )

    for row in await projection.list():
        pointer = row["first_user_message_id"]
        if pointer is not None:
            live = await db.fetchone(
                "SELECT role FROM conversation_history WHERE id = ? "
                "AND agent_id = ? AND deleted_at IS NULL AND archived_at IS NULL",
                (pointer, agent_id),
            )
            assert live is not None, (
                f"session {row['session_id']!r} points at message {pointer}, "
                "which is not a live row"
            )
            assert live[0] == "user"
        counted = await db.fetchval(
            "SELECT COUNT(*) FROM conversation_history WHERE agent_id = ? "
            "AND session_id = ? AND deleted_at IS NULL AND archived_at IS NULL",
            (agent_id, row["session_id"]),
        )
        assert counted > 0, (
            f"session {row['session_id']!r} has a projection row but no live rows"
        )
    # ...and the differential, so no case can pass by having quietly stopped
    # projecting anything at all.
    await _assert_agrees_with_the_grouper(db, projection, agent_id)


async def _message_ids(db: AsyncDatabase, agent_id: str = AGENT) -> Dict[str, int]:
    """``{content: id}``. Every corpus row's text is unique, so this is 1:1."""
    return {
        row[0]: row[1]
        for row in await db.fetchall(
            "SELECT content, id FROM conversation_history WHERE agent_id = ?",
            (agent_id,),
        )
    }


async def _frontier(db: AsyncDatabase, agent_id: str = AGENT) -> int:
    """The highest row id this agent has, asked without the projection's help."""
    return int(
        await db.fetchval(
            "SELECT COALESCE(MAX(id), 0) FROM conversation_history WHERE agent_id = ?",
            (agent_id,),
        )
        or 0
    )


# ── the corpora ──────────────────────────────────────────────────────────
#
# Every shape the ticket names, because the shapes interact: a resumed session
# straddles another, an operator-signal notice and an autonomous wake precede a
# human turn, and the deleted rows are interleaved rather than trailing.
#
# MODERN_CORPUS is the state Phase A's backfill leaves — every row carries a
# session id — and is where the per-session derivation and the incremental
# repair live. CORPUS is MODERN_CORPUS plus the rows that carry none, which is
# what a legacy database still holds and what forces the transcript derivation.

MODERN_CORPUS: List[Dict[str, Any]] = [
    # A: an operator-signal notice and an autonomous wake BEFORE the human
    # turn. Both count as user messages and neither may become the pointer.
    {"content": "A budget notice", "role": "user", "created_at": _at(180),
     "metadata": {"session_id": UUID_A, "operator_signal": True}},
    {"content": "A wake", "role": "user", "created_at": _at(181),
     "metadata": {"session_id": UUID_A,
                  "signal_wake": {"source": "heartbeat", "mode": "cognition"}}},
    {"content": "A human turn", "role": "user", "created_at": _at(182),
     "metadata": {"session_id": UUID_A}},
    {"content": "A reply", "role": "assistant", "created_at": _at(183),
     "metadata": {"session_id": UUID_A}},
    # B: a marker-started session inside A's window, then its own turns.
    {"content": "B marker", "role": "user", "created_at": _at(184),
     "metadata": {"session_id": UUID_B, "new_session": True}},
    {"content": "B human turn", "role": "user", "created_at": _at(185),
     "metadata": {"session_id": UUID_B}},
    # A wake-only session: no human turn at all, so no pointer — but a wake
    # source, which is what #2947 titles it by.
    {"content": "W wake", "role": "user", "created_at": _at(300),
     "metadata": {"session_id": UUID_WAKE,
                  "signal_wake": {"source": "talon.job_complete"}}},
    {"content": "W reply", "role": "assistant", "created_at": _at(301),
     "metadata": {"session_id": UUID_WAKE}},
    # A conversation that exists only as its new_session marker: the user
    # opened a chat and said nothing (#2222). It is a session — the UI shows a
    # tile for it — and it is the shape ``keep_empty_markers`` exists for, so
    # without it here that flag would be a choice no case could contradict.
    {"content": "M marker", "role": "user", "created_at": _at(400),
     "metadata": {"session_id": UUID_MARKER_ONLY, "new_session": True}},
    # A resumed past the gap: same id, hours later. The grouper splits this
    # into a second cluster and coalescing merges it back; the projection
    # never split it in the first place, and they must still agree.
    {"content": "A resumed", "role": "user", "created_at": _at(500),
     "metadata": {"session_id": UUID_A}},
    # C, carrying soft-deleted and archived rows. The pointer must skip both.
    {"content": "C deleted first turn", "role": "user", "created_at": _at(600),
     "metadata": {"session_id": UUID_C}, "deleted_at": _at(601)},
    {"content": "C archived turn", "role": "user", "created_at": _at(602),
     "metadata": {"session_id": UUID_C}, "archived_at": _at(603)},
    {"content": "C live turn", "role": "user", "created_at": _at(604),
     "metadata": {"session_id": UUID_C}},
    {"content": "C reply", "role": "assistant", "created_at": _at(605),
     "metadata": {"session_id": UUID_C}},
]

#: The rows that carry no session id the column may hold. Ordered into the
#: transcript by timestamp below.
UNSTAMPED_ROWS: List[Dict[str, Any]] = [
    # Legacy rows with no session id at all. They group under a row-id key,
    # which the column may not hold — the projection must stay silent, not
    # invent one.
    {"content": "legacy user", "role": "user", "created_at": _at(0), "metadata": {}},
    {"content": "legacy reply", "role": "assistant", "created_at": _at(1),
     "metadata": {}},
    # An id outside the column's contract: grouping honours it, the column
    # cannot hold it, so the projection is absent here too.
    {"content": "uncolumned user", "role": "user", "created_at": _at(90),
     "metadata": {"session_id": OUTSIDE_THE_CONTRACT}},
    # The shape a projection that reads only its own rows gets WRONG: an
    # unstamped reply immediately after a stamped turn. The grouper attributes
    # it to B — two messages, ending at :186 — because a row filed under
    # nothing stays with the session it fell next to.
    {"content": "B unstamped reply", "role": "assistant", "created_at": _at(186),
     "metadata": {}},
]

CORPUS: List[Dict[str, Any]] = sorted(
    MODERN_CORPUS + UNSTAMPED_ROWS, key=lambda row: row["created_at"]
)


async def _open(tmp_path, name: str, corpus) -> tuple:
    db = await AsyncDatabase.sqlite(str(tmp_path / name))
    await _seed(db, corpus)
    store = AsyncConversationStore(db, agent_id=AGENT)
    projection = ConversationSessionProjection(db, AGENT)
    await projection.repair()
    return db, store, projection


@pytest.fixture
async def seeded(tmp_path):
    """The mixed corpus, repaired up to date and saying so."""
    db, store, projection = await _open(tmp_path, "projection.db", CORPUS)
    try:
        yield db, store, projection
    finally:
        await db.close()


@pytest.fixture
async def modern(tmp_path):
    """The all-stamped corpus — the per-session derivation's own state."""
    db, store, projection = await _open(tmp_path, "modern.db", MODERN_CORPUS)
    try:
        yield db, store, projection
    finally:
        await db.close()


@pytest.fixture(params=["mixed", "all-stamped"])
async def either(request, tmp_path):
    """Both corpora, for claims that are about each derivation separately.

    Parametrized at the fixture rather than resolved inside the test, because
    an async fixture cannot be fetched by name from a running event loop.
    """
    corpus = CORPUS if request.param == "mixed" else MODERN_CORPUS
    db, store, projection = await _open(tmp_path, f"{request.param}.db", corpus)
    try:
        yield db, store, projection
    finally:
        await db.close()


# ── the differential ─────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_the_projection_says_what_the_grouper_says(seeded):
    """The acceptance gate for this phase, over every shape the ticket names.

    Spot-checks accompany the differential rather than replace it: the
    differential proves agreement, and these prove the corpus really contains
    the shapes it claims to — a corpus that had silently stopped exercising
    wake-only sessions would agree perfectly and mean nothing.
    """
    db, _store, projection = seeded
    stored = await _assert_agrees_with_the_grouper(db, projection)
    ids = await _message_ids(db)

    # A: the operator-signal notice and the wake are counted as user messages
    # but neither is the pointer — the #2947 skip, applied once, here.
    assert stored[UUID_A]["user_message_count"] == 4
    assert stored[UUID_A]["first_user_message_id"] == ids["A human turn"]
    assert stored[UUID_A]["wake_source"] == "heartbeat"
    # ...and the resumption past the gap is inside the same row, not a second.
    assert stored[UUID_A]["last_message_at"] == coerce_session_timestamp(_at(500))
    assert stored[UUID_A]["message_count"] == 5

    # B: a marker is structural, so it sets the start but is not a message —
    # and the unstamped reply that followed IS one, because the grouper
    # attributes a row filed under nothing to the session it fell next to.
    assert stored[UUID_B]["message_count"] == 2
    assert stored[UUID_B]["started_at"] == coerce_session_timestamp(_at(184))
    assert stored[UUID_B]["last_message_at"] == coerce_session_timestamp(_at(186))
    assert stored[UUID_B]["first_user_message_id"] == ids["B human turn"]

    # W: a session with no human turn has no pointer, and is titled by its wake.
    assert stored[UUID_WAKE]["first_user_message_id"] is None
    assert stored[UUID_WAKE]["wake_source"] == "talon.job_complete"

    # M: an opened-but-unused conversation is a session with no messages in it.
    # Projecting it is what lets a reader show the tile; a reader that wants
    # only sessions with traffic filters on message_count.
    assert stored[UUID_MARKER_ONLY]["message_count"] == 0
    assert stored[UUID_MARKER_ONLY]["first_user_message_id"] is None

    # C: the pointer skips the trashed and the archived rows.
    assert stored[UUID_C]["message_count"] == 2
    assert stored[UUID_C]["first_user_message_id"] == ids["C live turn"]

    # The legacy and outside-the-contract clusters are absent, never invented.
    assert OUTSIDE_THE_CONTRACT not in stored


@pytest.mark.asyncio
async def test_an_unstamped_neighbour_belongs_to_the_session_it_fell_next_to(
    tmp_path,
):
    """The smallest form of the shape a per-session read gets wrong.

    Two rows: a stamped user turn and an unstamped reply a minute later. The
    grouper reports one session of two messages ending at the reply; a
    projection built from ``WHERE session_id = ?`` reports one message ending a
    minute earlier. Isolated from the big corpus because there it is one
    assertion among many, and this is the whole claim.
    """
    db = await AsyncDatabase.sqlite(str(tmp_path / "neighbour.db"))
    try:
        await _seed(db, [
            {"content": "stamped turn", "role": "user", "created_at": _at(0),
             "metadata": {"session_id": UUID_A}},
            {"content": "unstamped reply", "role": "assistant",
             "created_at": _at(1), "metadata": {}},
        ])
        projection = ConversationSessionProjection(db, AGENT)
        await projection.repair()

        row = await projection.get(UUID_A)
        assert row["message_count"] == 2, (
            "the unstamped reply was dropped, so the projection disagrees with "
            "the grouper about the session it is caching"
        )
        assert coerce_session_timestamp(row["last_message_at"]) == (
            coerce_session_timestamp(_at(1))
        )
        await _assert_agrees_with_the_grouper(db, projection)
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_the_all_stamped_derivation_says_what_the_grouper_says(modern):
    """The same differential where the projection reads one session at a time.

    This is the strong form of the claim, and the reason the two corpora are
    both here: with no unstamped rows the projection never groups the transcript,
    so agreeing with a whole-transcript grouping is a claim about the ALGORITHM
    rather than about having run the same function twice.
    """
    db, _store, projection = modern
    stored = await _assert_agrees_with_the_grouper(db, projection)
    ids = await _message_ids(db)

    assert stored[UUID_A]["message_count"] == 5
    assert stored[UUID_A]["first_user_message_id"] == ids["A human turn"]
    assert stored[UUID_B]["message_count"] == 1
    assert stored[UUID_MARKER_ONLY]["message_count"] == 0
    assert stored[UUID_WAKE]["wake_source"] == "talon.job_complete"


@pytest.mark.asyncio
async def test_a_rebuild_from_nothing_equals_the_repairs_that_got_there(either):
    """The recovery path is the same table as the incremental one.

    A cache whose only correct state is the one it happened to accumulate cannot
    be recovered, and recovery is what this contract trades the write-path
    obligation for. So the projection is driven through a sequence of mutations
    and repairs, snapshotted, then dropped entirely and rebuilt from
    ``conversation_history`` — the only source of truth — and the two tables must
    be identical row for row.

    Asked of both corpora because the two derivations reach the table by
    different code, and "rebuild equals repair" is a claim about each of them.
    """
    db, store, projection = either
    ids = await _message_ids(db)

    # A sequence of incremental repairs, interleaved with real mutations.
    await store.add_conversation("user", "a new turn", session_id=UUID_B)
    await projection.repair()
    assert await store.delete_message(ids["A human turn"])
    await projection.repair()
    assert await store.restore_message(ids["A human turn"])
    await projection.repair()
    assert await store.archive_conversation_session(UUID_WAKE) > 0
    await projection.repair()

    incremental = await projection.list()
    assert incremental, "the corpus projected nothing, so this proves nothing"

    await db.execute("DELETE FROM conversation_sessions WHERE agent_id = ?", (AGENT,))
    assert await projection.list() == []
    assert await projection.rebuild() == len(incremental)

    assert await projection.list() == incremental
    assert not await projection.is_stale()


# ── staleness, per mutation, with no write-path cooperation ───────────────


@pytest.mark.asyncio
async def test_an_insert_is_detected_and_repaired_incrementally(modern):
    """A new row is the common case, and the only one that can stay cheap.

    The incremental branch is entered only when the change stamp's movement is
    entirely explained by rows standing above the frontier — which is what a
    plain append is, and nothing else is.
    """
    db, store, projection = modern
    assert not await projection.is_stale()

    await store.add_conversation("user", "brand new turn", session_id=UUID_B)

    assert await projection.is_stale()
    outcome = await projection.repair()
    assert outcome.kind == INCREMENTAL, outcome
    assert outcome.advanced
    assert (await projection.get(UUID_B))["message_count"] == 2
    await _assert_repaired_projection_is_true(db, projection)


@pytest.mark.asyncio
async def test_an_append_to_a_history_holding_unstamped_rows_is_not_incremental(
    seeded,
):
    """The exclusion that keeps the cheap branch honest.

    An appended row carrying no session id belongs to whichever session it fell
    next to, and the incremental branch finds the sessions to refresh by asking
    which ids the new rows are FILED under — which for such a row is none. So an
    agent whose history still holds unstamped rows is rebuilt rather than caught
    up, and the session that grew is right afterwards either way.
    """
    db, store, projection = seeded

    await store.add_conversation("user", "a stamped append", session_id=UUID_B)

    outcome = await projection.repair()
    assert outcome.kind == REBUILT, outcome
    assert (await projection.get(UUID_B))["message_count"] == 3
    await _assert_repaired_projection_is_true(db, projection)


@pytest.mark.asyncio
async def test_a_soft_delete_is_detected_although_it_appends_no_row(either):
    """The mutation a watermark on ``id`` alone cannot see.

    Nothing is appended, so the frontier is unmoved — and the pointer this
    session had is now at a trashed row, which is exactly the state the old
    contract needed every write path to prevent. Here it is merely detected.
    """
    db, store, projection = either
    ids = await _message_ids(db)
    assert (await projection.get(UUID_A))["first_user_message_id"] == ids[
        "A human turn"
    ]
    before = await _frontier(db)

    assert await store.delete_message(ids["A human turn"])

    assert await _frontier(db) == before, (
        "a soft-delete moved the frontier — then this case is not testing what "
        "it says"
    )
    assert await projection.is_stale()
    outcome = await projection.repair()
    assert outcome.kind == REBUILT, outcome
    assert (await projection.get(UUID_A))["first_user_message_id"] != ids[
        "A human turn"
    ]
    await _assert_repaired_projection_is_true(db, projection)


@pytest.mark.asyncio
async def test_a_restore_is_detected(seeded):
    """A row rejoining the live set is a change in the other direction.

    The corpus ships ``C deleted first turn`` already trashed, so restoring it
    both raises the live count and moves the pointer back to the earliest turn.
    """
    db, store, projection = seeded
    ids = await _message_ids(db)
    assert (await projection.get(UUID_C))["first_user_message_id"] == ids[
        "C live turn"
    ]

    assert await store.restore_message(ids["C deleted first turn"])

    assert await projection.is_stale()
    await _assert_repaired_projection_is_true(db, projection)
    assert (await projection.get(UUID_C))["first_user_message_id"] == ids[
        "C deleted first turn"
    ]


@pytest.mark.asyncio
async def test_an_archive_is_detected_through_its_own_column(seeded):
    """Archive uses ``archived_at``, not ``deleted_at``. Both leave the live set.

    Given its own case because it is separate code touching a separate column:
    a stamp that only watched ``deleted_at`` would pass every test above and
    let a whole archived session go on being listed.
    """
    db, store, projection = seeded
    assert await projection.get(UUID_WAKE) is not None

    assert await store.archive_conversation_session(UUID_WAKE) == 2

    assert await projection.is_stale()
    await _assert_repaired_projection_is_true(db, projection)
    assert await projection.get(UUID_WAKE) is None, (
        "a session with no live rows left keeps a projection row describing "
        "nothing"
    )


@pytest.mark.asyncio
async def test_a_purge_is_detected(seeded):
    """A hard delete removes rows the projection counted."""
    db, store, projection = seeded

    assert await store.purge_conversation_session(UUID_C) == 4

    assert await projection.is_stale()
    await _assert_repaired_projection_is_true(db, projection)
    assert await projection.get(UUID_C) is None


@pytest.mark.asyncio
async def test_purging_rows_already_in_the_trash_is_still_detected(seeded):
    """The case no count of LIVE rows can pass.

    Retention (#2509/#2567) empties the trash: rows that were *already* out of
    the live set are physically removed. Nothing about the live set moves, and
    if the newest row was not among them the frontier does not move either. The
    DELETE itself is the event, and it must be seen — because
    ``first_user_message_id`` can be pointing at one of those rows.

    Written against the store's own retention method rather than a hand-rolled
    DELETE, so it is the shipped path that is proved detectable.
    """
    db, store, projection = seeded
    ids = await _message_ids(db)
    before_frontier = await _frontier(db)
    live_before = await db.fetchval(
        "SELECT COUNT(*) FROM conversation_history WHERE agent_id = ? "
        "AND deleted_at IS NULL AND archived_at IS NULL",
        (AGENT,),
    )

    purged = await store.purge_trash_older_than(_at(9999))
    assert purged == 1, "the corpus should offer exactly one trashed row to purge"
    assert await db.fetchone(
        "SELECT id FROM conversation_history WHERE id = ?",
        (ids["C deleted first turn"],),
    ) is None

    assert await _frontier(db) == before_frontier
    assert await db.fetchval(
        "SELECT COUNT(*) FROM conversation_history WHERE agent_id = ? "
        "AND deleted_at IS NULL AND archived_at IS NULL",
        (AGENT,),
    ) == live_before, (
        "this case only means something while the LIVE set is unchanged"
    )
    assert await projection.is_stale()
    await _assert_repaired_projection_is_true(db, projection)


@pytest.mark.asyncio
async def test_an_exchange_that_preserves_every_aggregate_is_still_detected(seeded):
    """Trash one row and restore another: every count lands where it started.

    Soft-delete and restore are perfect inverses in count-space, so no vector of
    aggregates can tell the pair apart from nothing having happened. That is not
    an exotic interleaving — it is two ordinary clicks in the trash UI — and
    both sessions' rows would be silently wrong.

    Counting row EVENTS sees it, because two rows changed.
    """
    db, store, projection = seeded
    ids = await _message_ids(db)
    live_before = await db.fetchval(
        "SELECT COUNT(*) FROM conversation_history WHERE agent_id = ? "
        "AND deleted_at IS NULL AND archived_at IS NULL",
        (AGENT,),
    )

    assert await store.delete_message(ids["A human turn"])
    assert await store.restore_message(ids["C deleted first turn"])

    assert await db.fetchval(
        "SELECT COUNT(*) FROM conversation_history WHERE agent_id = ? "
        "AND deleted_at IS NULL AND archived_at IS NULL",
        (AGENT,),
    ) == live_before, (
        "the exchange changed the live COUNT, so this case is not exercising "
        "what it says it is"
    )
    assert await projection.is_stale()
    await _assert_repaired_projection_is_true(db, projection)


@pytest.mark.asyncio
async def test_re_homing_a_row_to_another_session_is_detected(either):
    """The mutation that moves nothing an aggregate can see.

    ``update_message_metadata``'s key set is the CALLER's, so it is the one
    supported door through which a row changes session after insertion. Doing so
    leaves the row count, the live count, the id-sum and the frontier exactly
    where they were: only *which* session the row belongs to has changed. A
    summary built from aggregates reports the projection current while one
    session over-counts and another under-counts.

    The engine's own row-event count sees it, and it sees it because the engine —
    not the caller — decides that a row changed.
    """
    db, store, projection = either
    ids = await _message_ids(db)
    assert (await projection.get(UUID_A))["message_count"] == 5

    assert await store.update_message_metadata(
        ids["A resumed"], {"session_id": UUID_C}
    )

    assert await projection.is_stale(), (
        "a row moved from one session to another and the projection still "
        "reports itself current"
    )
    await _assert_repaired_projection_is_true(db, projection)
    assert (await projection.get(UUID_A))["message_count"] == 4
    assert ids["A resumed"] is not None


@pytest.mark.asyncio
async def test_flipping_a_preview_flag_is_detected(modern):
    """The other invisible metadata rewrite: the same rows, a different preview.

    ``operator_signal`` and ``signal_wake`` decide whether a user row may become
    the session's previewed turn (#2947). Setting one moves
    ``first_user_message_id`` without moving a single count, so it is the same
    class of change as re-homing and needs its own case: a stamp watching only
    ``session_id`` would pass the case above and miss this one.
    """
    db, store, projection = modern
    ids = await _message_ids(db)
    assert (await projection.get(UUID_A))["first_user_message_id"] == ids[
        "A human turn"
    ]

    assert await store.update_message_metadata(
        ids["A human turn"], {"operator_signal": True}
    )

    assert await projection.is_stale()
    await _assert_repaired_projection_is_true(db, projection)
    assert (await projection.get(UUID_A))["first_user_message_id"] == ids[
        "A resumed"
    ], "the notice the picker must skip is still the session's preview"


@pytest.mark.asyncio
async def test_a_write_the_projection_cannot_see_does_not_force_a_rebuild(modern):
    """The narrowing, asserted rather than trusted.

    ``content``, ``rendered_content``, ``embedding_vec`` and ``model`` are all
    rewritten in place by ordinary paths — the #1402 canonical/transport split,
    the embedding co-write, the encryption backfill — and no field of this table
    is derived from any of them. Stamping those writes would make every one of
    them cost a full rebuild.

    The direction that matters is the other one, and it is covered by every case
    above: a column the projection DOES read must move the stamp. This is the
    case that keeps the narrowing from being free money.
    """
    db, _store, projection = modern
    ids = await _message_ids(db)
    before = await projection.observed_changes()

    await db.execute(
        "UPDATE conversation_history SET content = ?, model = ? WHERE id = ?",
        ("rewritten by a path this table cannot see", "some-model",
         ids["A human turn"]),
    )

    assert await projection.observed_changes() == before
    assert not await projection.is_stale()
    assert (await projection.repair()).kind == CURRENT


def test_the_triggers_watch_exactly_what_the_derivation_reads():
    """One column list, compiled into the SELECTs and into the triggers.

    Two lists is how a projection starts lying: a column the derivation reads
    but the trigger does not watch makes a rewrite of it invisible, and the
    symptom is a table that quietly disagrees with the rows it describes. So the
    trigger DDL is generated from the same constant the derivation is written
    against, and this holds the generated SQL to it on both dialects.
    """
    from kestrel_sovereign.storage.conversation_sessions import (
        _LIVE_ROWS,
        _OWN_ROWS,
        mutation_triggers,
    )

    for backend in ("sqlite", "postgres"):
        update = dict(mutation_triggers(backend))["conversation_history_change_update"]
        for column in PROJECTION_INPUT_COLUMNS:
            assert f"OLD.{column}" in update and f"NEW.{column}" in update, (
                f"{backend}: the update trigger does not watch {column}"
            )

    # ...and nothing the derivation reads is missing from that list. ``id`` is
    # the primary key on both engines and cannot be updated in place, which is
    # why it is not watched.
    read = {"id", *PROJECTION_INPUT_COLUMNS}
    for statement in (_OWN_ROWS, _LIVE_ROWS):
        selected = statement.split("SELECT ")[1].split(" FROM ")[0]
        for column in (name.strip() for name in selected.split(",")):
            assert column in read, f"{column} is read but never watched"


@pytest.mark.asyncio
async def test_a_fresh_agent_is_rebuilt_once_and_then_reports_itself_current(
    tmp_path,
):
    """An agent with no history has a watermark that is INVALID, not zero.

    The two are the same numbers and different states. Invalid must rebuild;
    "validly accounted for nothing" must not. Getting that backwards in either
    direction is a bug — a fresh agent rebuilt on every read, or a projection
    that never builds at all.
    """
    db = await AsyncDatabase.sqlite(str(tmp_path / "empty.db"))
    try:
        projection = ConversationSessionProjection(db, AGENT)
        assert _accounting(await projection.accounted()) == (False, 0, 0)
        assert await projection.observed_changes() == 0
        assert await projection.is_stale(), (
            "an unbuilt projection reported itself current"
        )

        assert (await projection.repair()).kind == REBUILT
        assert _accounting(await projection.accounted()) == (True, 0, 0)
        assert not await projection.is_stale()
        assert (await projection.repair()).kind == CURRENT
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_a_database_that_gains_the_ledger_after_its_history_is_rebuilt(
    tmp_path,
):
    """The upgrade path, and the case the validity flag exists for.

    On an existing database the change ledger and the watermark are created
    empty beside a full ``conversation_history``: the stamp reads zero, the
    watermark reads zero, and by the numbers alone the projection is current.
    It is not — it has never been built, and any rows left in it describe
    nothing.

    Expressed as the state rather than as an upgrade because that is what an
    upgrade leaves, and because an orphan row planted here is exactly what a
    projection restored from a backup without its watermark would carry.
    """
    db = await AsyncDatabase.sqlite(str(tmp_path / "upgraded.db"))
    try:
        await _seed(db, MODERN_CORPUS)
        projection = ConversationSessionProjection(db, AGENT)
        await db.execute(
            "INSERT INTO conversation_sessions "
            "(agent_id, session_id, message_count) VALUES (?, ?, ?)",
            (AGENT, "a-session-that-is-gone", 7),
        )
        await db.execute("DELETE FROM conversation_history_changes", ())
        await db.execute("DELETE FROM conversation_session_watermarks", ())

        assert await projection.observed_changes() == 0
        assert _accounting(await projection.accounted()) == (False, 0, 0), (
            "this case only means anything while the NUMBERS say current"
        )
        assert await projection.is_stale()

        assert (await projection.repair()).kind == REBUILT
        assert await projection.get("a-session-that-is-gone") is None, (
            "a projection row describing no history survived a repair that "
            "then recorded itself current"
        )
        assert (await projection.get(UUID_A))["message_count"] == 5
        assert not await projection.is_stale()
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_initializing_a_database_wires_the_projections_whole_schema(tmp_path):
    """The schema is declared AND called — separate claims, separately checked.

    ``tests/integration/test_session_id_column_backend_parity.py`` proves the
    declarations create what they say on both engines. This proves
    ``_init_schema`` actually invokes them: a constant that is never called reads
    exactly like an index, and the only symptom is a query plan — or, for a
    trigger, a projection that reports itself current forever. There is also no
    completion marker for a projection backfill, because there is no backfill —
    the watermark is the marker.
    """
    from kestrel_sovereign.storage.async_database import (
        _SESSION_FRONTIER_INDEX,
        _SESSION_PROJECTION_INDEX,
    )
    from kestrel_sovereign.storage.conversation_sessions import (
        mutation_triggers,
        projection_tables,
    )

    db = await AsyncDatabase.sqlite(str(tmp_path / "boot.db"))
    try:
        for name, table, _columns in (
            _SESSION_PROJECTION_INDEX,
            _SESSION_FRONTIER_INDEX,
        ):
            assert await db._index_exists(name, table) is True, name
        for table, _ddl in projection_tables():
            assert await db.table_exists(table), table
        for trigger, _ddl in mutation_triggers(db.backend_type):
            assert await db._trigger_exists(trigger, "conversation_history"), trigger
        assert await db.fetchval(
            "SELECT COUNT(*) FROM schema_backfills WHERE name LIKE ?",
            ("%conversation_sessions%",),
        ) == 0
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_creating_the_projection_schema_twice_changes_nothing(tmp_path):
    """``_init_schema`` runs on every ``from_pool()``, so this runs constantly.

    The second call must find everything present and take neither the migration
    lock nor a ``CREATE`` — and, more to the point, must not recreate a trigger
    in a way that would double-count the next row event and break the
    incremental branch's arithmetic.
    """
    db = await AsyncDatabase.sqlite(str(tmp_path / "twice.db"))
    try:
        await db.ensure_session_projection_schema()
        await db.ensure_session_projection_schema()

        projection = ConversationSessionProjection(db, AGENT)
        await _seed(db, MODERN_CORPUS[:1])
        assert await projection.observed_changes() == 1, (
            "one INSERT moved the change stamp by more than one, so the "
            "incremental branch's arithmetic no longer holds"
        )
    finally:
        await db.close()


# ── the watermark's direction ─────────────────────────────────────────────


@pytest.mark.asyncio
async def test_an_incremental_repair_leaves_rows_below_the_frontier_alone(modern):
    """Incremental means incremental — proved by what it does NOT fix.

    A projection row is corrupted by hand to stand in for one that a previous
    pass left wrong. An insert then makes the projection stale in a way the
    stamp's arithmetic attributes entirely to rows above the frontier, so the
    repair is entitled to skip everything below it — and the corrupt row
    survives.

    This is the one case where being able to see the optimization matters: if
    this test passed with the corrupt row repaired, "incremental" would just be
    a label on a full rebuild. The second half is the safety net that makes the
    first half acceptable: a change below the frontier forces a rebuild, and the
    rebuild fixes it.
    """
    db, store, projection = modern
    await db.execute(
        "UPDATE conversation_sessions SET message_count = 999 "
        "WHERE agent_id = ? AND session_id = ?",
        (AGENT, UUID_C),
    )

    await store.add_conversation("user", "a turn in B", session_id=UUID_B)
    outcome = await projection.repair()
    assert outcome.kind == INCREMENTAL, outcome
    assert (await projection.get(UUID_C))["message_count"] == 999, (
        "the incremental repair rewrote a session the stamp had already "
        "accounted for"
    )

    # ...and the exact recovery path still exists for exactly this.
    assert await projection.rebuild() > 0
    assert (await projection.get(UUID_C))["message_count"] == 2
    await _assert_repaired_projection_is_true(db, projection)


@pytest.mark.asyncio
async def test_an_append_beside_an_edit_is_never_repaired_incrementally(modern):
    """The arithmetic, at the boundary where a looser test would break.

    One append and one soft-delete between two repairs: the stamp moves by two
    while exactly one row stands above the frontier. A test of "did anything
    arrive?" would take the cheap branch and never revisit the deleted row's
    session. Equality is what refuses, and it refuses because each row event
    counts once.
    """
    db, store, projection = modern
    ids = await _message_ids(db)

    await store.add_conversation("user", "a fresh turn", session_id=UUID_B)
    assert await store.delete_message(ids["A human turn"])

    outcome = await projection.repair()
    assert outcome.kind == REBUILT, outcome
    assert (await projection.get(UUID_A))["first_user_message_id"] != ids[
        "A human turn"
    ]
    await _assert_repaired_projection_is_true(db, projection)


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_a_crash_mid_repair_leaves_the_watermark_behind_never_ahead(modern):
    """A checkpoint may only advance when what it records is beyond loss.

    The watermark is published last, after every session's row, so a pass that
    dies part-way has written some rows and claimed none of them. The next repair
    redoes the work. The forbidden direction is the other one: a watermark
    recorded before the rows would claim to have accounted for rows it never
    read, and nothing would ever revisit them.

    "Behind" is asserted as the two numbers plus the flag rather than as equality
    with the state before the crash, because taking the epoch invalidates: a dead
    pass leaves the projection accounting for NOTHING, which is further behind
    than where it started and is the direction that costs a rebuild rather than
    trusting rows a dead pass half-wrote.
    """
    db, store, projection = modern
    ids = await _message_ids(db)
    assert await store.delete_message(ids["A human turn"])
    before = await projection.accounted()
    assert before.valid, "the case needs a valid watermark to fall behind FROM"

    class _Crashes(ConversationSessionProjection):
        """Fails on the second session it writes, mid-pass."""

        written = 0

        async def _store(self, session, epoch):
            if self.written == 1:
                raise RuntimeError("crash mid-repair")
            self.written += 1
            return await super()._store(session, epoch)

    with pytest.raises(RuntimeError, match="crash mid-repair"):
        await _Crashes(db, AGENT).repair()

    after = await projection.accounted()
    assert not after.valid, (
        "a pass that died part-way left the projection claiming to account "
        "for something"
    )
    assert after.frontier <= before.frontier and after.changes <= before.changes, (
        "the watermark moved forward despite the pass not finishing"
    )
    assert await projection.is_stale()

    # And the redo costs nothing but the redo.
    await _assert_repaired_projection_is_true(db, projection)


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_a_row_arriving_mid_pass_leaves_the_watermark_behind(modern):
    """The stamp and the frontier are read before the pass, not after it.

    A row that arrives while a repair is running may or may not be picked up by
    the session read that follows it — the pass cannot know which. Recording the
    state it *started* from means the answer is always "behind", and behind is
    the recoverable direction. Re-reading at the end would record a frontier
    covering a row this pass may never have looked at.
    """
    db, store, projection = modern
    ids = await _message_ids(db)
    assert await store.delete_message(ids["A human turn"])

    class _WritesMidPass(ConversationSessionProjection):
        async def _refresh(self, session_ids, epoch):
            written = await super()._refresh(session_ids, epoch)
            if not getattr(self, "_done", False):
                self._done = True
                await store.add_conversation(
                    "user", "arrived mid-pass", session_id=UUID_B
                )
            return written

    await _WritesMidPass(db, AGENT).repair()

    accounted = await projection.accounted()
    assert accounted.frontier < await _frontier(db), (
        "the watermark accounted for a row that arrived after it was read"
    )
    assert await projection.is_stale()
    await _assert_repaired_projection_is_true(db, projection)


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_a_superseded_repair_may_not_publish_what_it_accounted_for(
    modern,
):
    """Two repairs racing: the superseded one may not claim a state it cannot
    vouch for.

    This is the whole of the concurrency story, and it replaces the
    serialization that failed review twice. Two passes can each derive a
    projection row from rows the other has already changed, and a single upsert
    does not order them — so ownership is settled by an epoch taken before either
    derives anything, and the pass whose epoch was taken away publishes nothing.

    Bounded on both sides: the parked pass rescues itself on
    ``INTERLEAVE_BUDGET`` and the case carries a timeout, so a boundary that
    stopped working reports a failure instead of a hung job.
    """
    db, store, projection = modern
    ids = await _message_ids(db)
    assert await store.delete_message(ids["A human turn"])

    parked = asyncio.Event()
    resume = asyncio.Event()

    class _Parks(ConversationSessionProjection):
        """Does its whole pass, then waits before publishing its watermark."""

        async def _refresh(self, session_ids, epoch):
            written = await super()._refresh(session_ids, epoch)
            parked.set()
            await asyncio.wait_for(resume.wait(), INTERLEAVE_BUDGET)
            return written

    slow = asyncio.create_task(_Parks(db, AGENT).repair())
    try:
        async with asyncio.timeout(INTERLEAVE_BUDGET):
            await parked.wait()
        winner = await ConversationSessionProjection(db, AGENT).repair()
        assert winner.advanced
    finally:
        resume.set()
    loser = await slow

    assert not loser.advanced, (
        "both passes published a watermark, so one of them claimed a state "
        "the other had already superseded"
    )
    # The winner rebuilt from history under its own epoch, so its watermark is
    # the one standing and it describes the rows that are there. The superseded
    # pass neither advanced it nor knocked it down.
    assert not await projection.is_stale(), (
        "the superseded pass invalidated the winner's watermark, so a correct "
        "projection now reports itself stale"
    )
    assert (await projection.get(UUID_A))["message_count"] == len(
        [
            row
            for row in await _live_history(db)
            if row["metadata"].get("session_id") == UUID_A
        ]
    )
    await _assert_repaired_projection_is_true(db, projection)


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_a_superseded_repair_cannot_overwrite_a_published_row(modern):
    """The fence, at the only moment detection could not have saved it.

    Every repair writes its rows before it can know whether it won, so a pass
    parked between deriving a row and storing it is holding a *stale* answer. Let
    a newer repair publish a correct row and its watermark in the meantime, and
    then let the parked pass go: if its write is merely regretted afterwards, the
    published row is wrong for the interval between the write and the discovery —
    and permanently if the pass dies in between, since a watermark cannot
    un-write a row. So the write itself has to be refused.

    Reproduced at exactly that ordering: park after ``project_transcript`` has
    produced the row and before ``_store`` puts it anywhere.
    """
    db, store, projection = modern
    ids = await _message_ids(db)
    # Something for the slow pass to do. Session A is untouched by it, so the
    # row that pass is about to derive for A is the one already stored.
    assert await store.delete_message(ids["C live turn"])
    stale_count = (await projection.get(UUID_A))["message_count"]

    parked = asyncio.Event()
    resume = asyncio.Event()

    class _ParksBeforeStoring(ConversationSessionProjection):
        """Derives a row, then waits — holding an answer, having written none."""

        async def _store(self, session, epoch):
            if not parked.is_set():
                parked.set()
                await asyncio.wait_for(resume.wait(), INTERLEAVE_BUDGET)
            return await super()._store(session, epoch)

    # The slow pass derives session A from history as it stands now...
    slow = asyncio.create_task(_ParksBeforeStoring(db, AGENT).repair())
    try:
        async with asyncio.timeout(INTERLEAVE_BUDGET):
            await parked.wait()
        # ...and history moves under it, so what it is holding is now wrong.
        assert await store.delete_message(ids["A human turn"])
        winner = await ConversationSessionProjection(db, AGENT).repair()
        assert winner.advanced
        published = await projection.list()
        assert not await projection.is_stale()
        assert (await projection.get(UUID_A))["message_count"] != stale_count, (
            "the parked pass is holding the same answer the winner published, "
            "so letting its write through would be undetectable and this case "
            "would pass without the fence"
        )
    finally:
        resume.set()
    loser = await slow

    assert not loser.advanced
    # The claim, asserted before anything about what the pass *reported*: the
    # rows are the winner's, byte for byte.
    assert await projection.list() == published, (
        "a superseded repair overwrote rows the winning repair had published, "
        "while the watermark went on saying the projection was current"
    )
    assert loser.sessions == 0, (
        "a superseded pass reported writing rows, so the fence let its writes "
        "through"
    )
    assert not await projection.is_stale(), (
        "the projection is reporting itself current, which is only honest "
        "because the rows above are still the winner's"
    )
    # ...and the table really is what the grouper says, deletion included.
    await _assert_repaired_projection_is_true(db, projection)


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_a_superseded_repair_cannot_delete_a_published_row(modern):
    """The other half of the fence: a repair removes rows as well as writing them.

    A pass that found a session empty goes on to drop its row, and that decision
    ages exactly as badly as a derived count — restore the rows and the session
    is alive again. A fence on the upsert alone would leave a superseded pass
    able to delete a session the winner had just published, and "absent while
    claiming to be current" is the same silent disagreement as "wrong": no reader
    can tell either from the truth.

    Given its own case rather than folded into the upsert's, because it is a
    separate statement and a fence can be forgotten on one without the other.
    """
    db, store, projection = modern
    ids = await _message_ids(db)
    # Empty session B out, so a repair reaching it will drop its row.
    for content in ("B marker", "B human turn"):
        assert await store.delete_message(ids[content])
    assert await projection.get(UUID_B) is not None, (
        "the projection has not yet caught up, which is what makes the parked "
        "pass's decision to drop this row a stale one"
    )

    parked = asyncio.Event()
    resume = asyncio.Event()
    dropping: List[str] = []

    class _ParksBeforeForgetting(ConversationSessionProjection):
        """Decides a session is gone, then waits before dropping its row."""

        async def _forget(self, session_id, epoch):
            if not parked.is_set():
                dropping.append(session_id)
                parked.set()
                await asyncio.wait_for(resume.wait(), INTERLEAVE_BUDGET)
            return await super()._forget(session_id, epoch)

    slow = asyncio.create_task(_ParksBeforeForgetting(db, AGENT).repair())
    try:
        async with asyncio.timeout(INTERLEAVE_BUDGET):
            await parked.wait()
        assert dropping == [UUID_B]
        # B comes back before the drop lands, so the decision is now wrong.
        for content in ("B marker", "B human turn"):
            assert await store.restore_message(ids[content])
        winner = await ConversationSessionProjection(db, AGENT).repair()
        assert winner.advanced
        published = await projection.list()
        assert await projection.get(UUID_B) is not None
    finally:
        resume.set()
    loser = await slow

    assert not loser.advanced
    assert await projection.list() == published, (
        "a superseded repair dropped a row the winning repair had published, "
        "while the watermark went on saying the projection was current"
    )
    assert not await projection.is_stale()
    await _assert_repaired_projection_is_true(db, projection)


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_a_repair_that_never_takes_the_epoch_writes_nothing(modern):
    """Losing the claim is bounded, and costs nothing but the attempts.

    A claim fails when another pass took the epoch first, and the answer is to
    re-read and try again — but retrying forever is a spin, so after
    ``CLAIM_ATTEMPTS`` the pass reports DEFERRED and stops. What it must not do
    on the way out is leave a mark: it never owned the epoch, so it may not have
    written a row, and it may not have moved the watermark of whichever pass
    does.

    Simulated by reading a watermark that has always already moved, which is
    what unbounded contention looks like from inside one pass, and is the only
    way to reach the exhausted branch deterministically. ``rebuild`` is asked
    the same question because it answers differently on purpose: it is the verb
    reached for when the table is known to be wrong, so it raises rather than
    quietly doing nothing.
    """
    db, store, projection = modern
    ids = await _message_ids(db)
    assert await store.delete_message(ids["A human turn"])
    before_rows = await projection.list()
    before_watermark = await projection.accounted()

    class _AlwaysBeaten(ConversationSessionProjection):
        """Every epoch it reads is one another pass has already superseded."""

        async def accounted(self):
            watermark = await super().accounted()
            return replace(watermark, epoch=watermark.epoch - 1)

    outcome = await _AlwaysBeaten(db, AGENT).repair()

    assert outcome.kind == DEFERRED
    assert not outcome.advanced
    assert outcome.sessions == 0
    assert await projection.list() == before_rows, (
        "a pass that never owned the epoch wrote a projection row anyway"
    )
    assert await projection.accounted() == before_watermark, (
        "a pass that never owned the epoch moved the watermark anyway"
    )

    with pytest.raises(RuntimeError, match="could not take"):
        await _AlwaysBeaten(db, AGENT).rebuild()
    assert await projection.list() == before_rows

    # ...and a pass that CAN take it still repairs normally afterwards.
    await _assert_repaired_projection_is_true(db, projection)


# ── absent where it must be ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_an_id_outside_the_column_contract_is_never_projected(seeded):
    """Phase A's invariant, carried forward: absent is allowed, wrong is not.

    ``did:x:1`` is a session the grouper honours and the indexed column may not
    hold, so no projection row may claim it — and adding more of its rows must
    not change that.
    """
    db, _store, projection = seeded
    await _seed(
        db,
        [{"content": "more uncolumned", "role": "user", "created_at": _at(91),
          "metadata": {"session_id": OUTSIDE_THE_CONTRACT}}],
    )

    await projection.repair()
    assert await projection.get(OUTSIDE_THE_CONTRACT) is None
    assert not any(
        row["session_id"] == OUTSIDE_THE_CONTRACT for row in await projection.list()
    )
    await _assert_repaired_projection_is_true(db, projection)


@pytest.mark.asyncio
async def test_a_column_value_the_contract_forbids_is_still_not_keyed(modern):
    """The guard that only a Phase A violation can reach — defended, not assumed.

    ``_sessions`` filters the ids it found through
    :func:`is_stampable_session_id`, and the column contract means no shipped
    write path can produce one it rejects. That makes the filter a clause no
    ordinary test can exercise, which Phase A's Finding named as the worst kind:
    it reads as protection while a mutation removing it goes unnoticed.

    So the violation is written directly into the column, bypassing
    ``column_session_id`` the way only a future bug could. The projection's
    primary key is ``(agent_id, session_id)`` and Phase C will read it, so a row
    keyed by a value the contract forbids is a key no reader can round-trip.
    Absent is the permitted direction.

    On the all-stamped corpus deliberately: that is the derivation ``_sessions``
    feeds, so this is the path where the filter is load-bearing.
    """
    db, _store, projection = modern
    await db.execute(
        "INSERT INTO conversation_history "
        "(agent_id, role, content, metadata, session_id, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (
            AGENT, "user", "smuggled row",
            json.dumps({"session_id": OUTSIDE_THE_CONTRACT}),
            OUTSIDE_THE_CONTRACT,
            _at(700),
        ),
    )

    await projection.repair()
    assert await projection.get(OUTSIDE_THE_CONTRACT) is None
    assert not any(
        row["session_id"] == OUTSIDE_THE_CONTRACT for row in await projection.list()
    )


@pytest.mark.asyncio
async def test_a_row_whose_column_and_metadata_disagree_is_refused_not_guessed(
    modern, caplog
):
    """A Phase A violation: the column says one session, metadata another.

    Every field is derived by handing rows to the grouper, which reads
    ``metadata.session_id``. If a row's column and metadata name different
    sessions, the rows selected by the column do not group under the column's
    id, and there is no answer that is not a guess — so the session is refused
    and logged rather than stored under one of the two candidates.

    Reachable only by writing the divergence by hand, which is exactly why it
    needs a case: it is otherwise a branch no test defends.
    """
    db, _store, projection = modern
    await db.execute(
        "INSERT INTO conversation_history "
        "(agent_id, role, content, metadata, session_id, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (
            AGENT, "user", "divergent row",
            json.dumps({"session_id": UUID_B}),
            UUID_C,
            _at(800),
        ),
    )

    with caplog.at_level("ERROR"):
        await projection.repair()

    assert any(
        "refusing to project a session the transcript does not show" in record.message
        for record in caplog.records
    ), [record.message for record in caplog.records]
    assert await projection.get(UUID_C) is None, (
        "a session whose rows the grouper files elsewhere was stored anyway"
    )


@pytest.mark.asyncio
async def test_a_divergent_row_taints_the_session_the_grouper_files_it_under(
    seeded, caplog
):
    """The same violation seen from the transcript derivation.

    Here the column's claim is not what selects the rows — the whole live
    transcript is grouped — so the refusal has to be made on the *membership*:
    a row the grouper attributed to B while its indexed column says C. Filing it
    under B would store a session containing a row that, by the column Phase C
    will query, is not in it.

    Without this check the projection and the reference grouper would agree with
    each other and both be wrong about where the row lives, which is why the
    differential cannot stand in for this case.
    """
    db, _store, projection = seeded
    assert await projection.get(UUID_B) is not None
    await db.execute(
        "INSERT INTO conversation_history "
        "(agent_id, role, content, metadata, session_id, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (
            AGENT, "user", "divergent row",
            json.dumps({"session_id": UUID_B}),
            UUID_C,
            _at(800),
        ),
    )

    with caplog.at_level("ERROR"):
        await projection.repair()

    assert any(
        "refusing to project a session the transcript does not show" in record.message
        for record in caplog.records
    ), [record.message for record in caplog.records]
    assert await projection.get(UUID_B) is None, (
        "a session was stored holding a row whose own column says it is "
        "somewhere else"
    )


@pytest.mark.asyncio
async def test_a_projection_row_is_dropped_when_its_session_stops_existing(seeded):
    """An orphan row is a claim about rows that are gone.

    Purging every row of a session leaves nothing for the projection to describe,
    so the row must go rather than linger as the newest "session" in a list
    ordered by ``last_message_at``.
    """
    db, store, projection = seeded
    assert await projection.get(UUID_B) is not None

    assert await store.purge_conversation_session(UUID_B) == 2

    await projection.repair()
    assert await projection.get(UUID_B) is None
    await _assert_repaired_projection_is_true(db, projection)


@pytest.mark.asyncio
async def test_a_projection_for_one_agent_never_answers_for_another(tmp_path):
    """The change stamp is per agent, and so is every repair.

    Two agents share the table, so a stamp that forgot its ``agent_id`` would
    make each one permanently stale in the other's presence — and a repair that
    forgot it would hand one agent's sessions to the other.
    """
    db = await AsyncDatabase.sqlite(str(tmp_path / "two-agents.db"))
    other = "did:test:session-projection-other"
    try:
        await _seed(db, MODERN_CORPUS)
        await _seed(
            db,
            [{"content": "other agent turn", "role": "user", "created_at": _at(0),
              "metadata": {"session_id": UUID_A}}],
            agent_id=other,
        )
        mine = ConversationSessionProjection(db, AGENT)
        theirs = ConversationSessionProjection(db, other)

        await mine.repair()
        assert not await mine.is_stale()
        assert await theirs.is_stale(), (
            "repairing one agent marked another agent's projection current"
        )
        assert await theirs.get(UUID_A) is None

        await theirs.repair()
        assert (await theirs.get(UUID_A))["message_count"] == 1
        assert (await mine.get(UUID_A))["message_count"] == 5
        assert not await mine.is_stale()

        # The sharp end of "per agent": another agent's traffic must not make
        # this one stale. A stamp that forgot its ``agent_id`` filter still
        # passes everything above — both agents would simply share one global
        # counter, each repairing its own sessions — and only shows up as every
        # agent in a multi-agent host rebuilding on every other agent's turn.
        await _seed(
            db,
            [{"content": "their second turn", "role": "user",
              "created_at": _at(5), "metadata": {"session_id": UUID_B}}],
            agent_id=other,
        )
        assert await theirs.is_stale()
        assert not await mine.is_stale(), (
            "another agent's new row made this agent's projection stale"
        )
    finally:
        await db.close()
