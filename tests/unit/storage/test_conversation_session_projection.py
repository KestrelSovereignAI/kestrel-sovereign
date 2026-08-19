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

The concurrency and crash cases are about that direction rather than about
detection, and they are what the chunk contract has to earn. A repair walks
history in bounded chunks, each one a short transaction that writes its rows and
records what it accounted for together — so a chunk that dies leaves neither, a
pass that dies resumes from the last chunk that committed, and three passes
racing merely do overlapping idempotent work. Each carries a bounded timeout: a
concurrency case that can hang cannot report the bug it exists to catch, and in
CI it is a timed-out job with no failing assertion.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List

import pytest

from kestrel_sovereign.storage.async_conversation_store import AsyncConversationStore
from kestrel_sovereign.storage.async_database import AsyncDatabase
from kestrel_sovereign.storage.conversation_sessions import (
    CURRENT,
    INCREMENTAL,
    PROJECTION_INPUT_COLUMNS,
    REBUILT,
    ConversationSessionProjection,
    SessionWatermark,
    project_transcript,
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

def _accounting(watermark: SessionWatermark) -> tuple:
    """What a watermark claims, as a tuple a case can pin exactly."""
    return (
        watermark.valid,
        watermark.stamp,
        watermark.through,
        watermark.target,
    )


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
    """The corpus the read path would hand the grouper: live rows, its order.

    ``canonical_order()`` rather than ``id ASC``, because that is what
    ``/api/conversations`` actually feeds the grouper — it selects newest-first
    by ``created_at`` and reverses. An earlier version of this helper said
    ``ORDER BY id ASC``, which made every differential test below compare the
    projection against a read path that does not exist: both sides were in id
    order, so the one case where the two orders differ could not fail. Round 6
    of review found the divergence that this hid.
    """
    from kestrel_sovereign.storage.conversation_sessions import canonical_order

    rows = await db.fetchall(
        "SELECT id, role, content, metadata, created_at FROM conversation_history "
        "WHERE agent_id = ? AND deleted_at IS NULL AND archived_at IS NULL "
        f"{canonical_order(db.backend_type)}",
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


async def _assert_watermark_is_not_ahead(
    db: AsyncDatabase,
    projection: ConversationSessionProjection,
    agent_id: str = AGENT,
) -> None:
    """Everything the watermark claims to have accounted for really is stored.

    The forbidden direction, asserted directly rather than inferred from a repair
    that went on to succeed. A watermark is a claim about rows, and a precise
    one: **every session's stored row describes that session's live rows at or
    below ``through``, and nothing above it.** So the claim is checked against
    the grouper's answer over exactly that prefix of history — the same reference
    the differential uses, restricted to the part the projection says it has
    read.

    The restriction is what makes this a real check rather than a weaker one. A
    repair folds each chunk's rows in as it walks, so a session straddling
    ``through`` is genuinely partial part-way through a walk, and comparing it
    against the WHOLE transcript's answer would fail for a projection that is
    behaving exactly as designed. Comparing against the prefix instead still
    fails for a fold that double-counted, dropped a chunk, or ran ahead of its
    watermark — which are the mistakes this exists to catch.

    Sessions with no row at or below ``through`` are deliberately not checked: an
    unfinished walk is *allowed* to be silent about them, and a case that
    required otherwise would be asserting the opposite of what chunking is for.

    The claim is also *as of the recorded stamp*, so when the stamp has moved
    since, there is no content claim left to check — only that the projection
    says as much. Comparing content there would be asserting that a projection
    permitted to be stale is not stale, which is the opposite of this contract.
    """
    accounted = await projection.accounted()
    if not accounted.valid:
        return
    if await projection.observed_changes() != accounted.stamp:
        assert await projection.is_stale(), (
            "the change stamp moved and the projection still reports itself "
            "current"
        )
        return
    prefix = [
        message
        for message in await _live_history(db, agent_id)
        if message["id"] <= accounted.through
    ]
    reference = _reference_sessions(prefix)
    stored = {row["session_id"]: _stored(row) for row in await projection.list()}
    claimed = {
        str(row[0])
        for row in await db.fetchall(
            "SELECT DISTINCT session_id FROM conversation_history "
            "WHERE agent_id = ? AND id <= ? AND session_id IS NOT NULL "
            "AND deleted_at IS NULL AND archived_at IS NULL",
            (agent_id, accounted.through),
        )
        if is_stampable_session_id(row[0])
    }
    for session_id in claimed & set(reference):
        assert session_id in stored, (
            f"the watermark accounts for history through {accounted.through}, "
            f"which includes session {session_id!r} — and it is not stored"
        )
        assert stored[session_id] == reference[session_id], session_id


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
    assert outcome.current
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


@pytest.mark.asyncio
async def test_recalling_a_memory_does_not_invalidate_the_projection(modern):
    """The narrowing that matters most, because the write is on the READ path.

    ``MemoryRetriever.update_access`` bumps ``access_count`` and stamps
    ``last_accessed`` through ``atomic_increment_metadata_counter`` every time a
    memory is retrieved, and ``update_applied`` does the same for
    ``applied_count``. Those keys share the document the session id lives in, so
    a trigger comparing ``metadata`` WHOLE moved the stamp on every recall —
    appending no row, which is exactly the movement ``_plan`` cannot attribute.
    Recall therefore rebuilt the entire projection, and recall is not a rare
    event; it is what the agent does.

    Driven through the real store method rather than a hand-written UPDATE,
    because the claim is about the statement production actually issues — its
    PostgreSQL form reserializes the whole document through ``jsonb_set``, so a
    comparison of raw text would differ even where no value did.
    """
    db, store, projection = modern
    ids = await _message_ids(db)
    before = await projection.observed_changes()

    assert await store.atomic_increment_metadata_counter(
        ids["A human turn"], "access_count", "last_accessed"
    )

    assert await projection.observed_changes() == before, (
        "bookkeeping on the read path moved the change stamp, so every memory "
        "retrieval invalidates the projection"
    )
    assert not await projection.is_stale()
    assert (await projection.repair()).kind == CURRENT

    # ...and the write really landed, so this is not passing because nothing
    # happened. A no-op UPDATE would move no stamp for the wrong reason.
    stored = await db.fetchone(
        "SELECT metadata FROM conversation_history WHERE id = ?",
        (ids["A human turn"],),
    )
    written = json.loads(stored[0])
    assert written["access_count"] == 1
    assert written["last_accessed"]
    assert written["session_id"] == UUID_A, "the merge dropped the session id"


@pytest.mark.asyncio
async def test_a_metadata_document_neither_reader_can_trust_is_compared_whole(
    modern,
):
    """The fallback, in both of the directions it has to be right.

    ``metadata`` is free text and legacy rows hold documents that will not
    parse. The extraction must therefore *never raise* — an UPDATE of an
    unrelated column on such a row is an ordinary write (the encryption
    backfill, the #1402 canonical/transport split), and a trigger that raised
    would fail it. And it must never *hide* a change: two malformed documents
    are different documents, so they are compared as the raw text they are.

    A document carrying a watched key twice takes the same branch on SQLite, for
    the reason Phase A measured: ``json_extract`` reads the first occurrence and
    ``json.loads`` the last, so an extraction there would compare a value the
    derivation never sees.
    """
    db, _store, projection = modern
    ids = await _message_ids(db)
    target = ids["A reply"]

    async def _set(metadata):
        await db.execute(
            "UPDATE conversation_history SET metadata = ? WHERE id = ?",
            (metadata, target),
        )

    before = await projection.observed_changes()
    await _set("not json at all")
    assert await projection.observed_changes() == before + 1

    # An ordinary write to a column this table cannot see, on a row whose
    # metadata cannot be parsed. The trigger runs; it must not raise.
    after_malformed = await projection.observed_changes()
    await db.execute(
        "UPDATE conversation_history SET content = ? WHERE id = ?",
        ("rewritten while the metadata is unreadable", target),
    )
    assert await projection.observed_changes() == after_malformed, (
        "an unwatched write moved the stamp for a row whose metadata cannot "
        "be parsed"
    )

    # Two malformed documents are two documents. Collapsing them to one value
    # would make this change invisible.
    await _set("also not json, but different")
    assert await projection.observed_changes() == after_malformed + 1, (
        "a change between two unreadable documents was not detected"
    )

    # A duplicated watched key: the readers disagree about which occurrence
    # wins, so the whole document is compared instead of a value only one of
    # them would read.
    duplicated = await projection.observed_changes()
    await _set('{"session_id": "%s", "session_id": "%s"}' % (UUID_A, UUID_B))
    assert await projection.observed_changes() == duplicated + 1
    await _set('{"session_id": "%s", "session_id": "%s"}' % (UUID_A, UUID_C))
    assert await projection.observed_changes() == duplicated + 2, (
        "a duplicated key hid a change from the reader that takes the last one"
    )


def test_the_triggers_watch_exactly_what_the_derivation_reads():
    """One column list, compiled into the SELECTs and into the triggers.

    Two lists is how a projection starts lying: a column the derivation reads
    but the trigger does not watch makes a rewrite of it invisible, and the
    symptom is a table that quietly disagrees with the rows it describes. So the
    trigger DDL is generated from the same constant the derivation is written
    against, and this holds the generated SQL to it on both dialects.
    """
    from kestrel_sovereign.storage.conversation_sessions import (
        _DERIVED_FROM,
        _chunk_sql,
        _live_rows_through,
        _own_rows_through,
        mutation_triggers,
    )

    for backend in ("sqlite", "postgres"):
        update = dict(mutation_triggers(backend))["conversation_history_change_update"]
        for column in PROJECTION_INPUT_COLUMNS:
            assert f"OLD.{column}" in update and f"NEW.{column}" in update, (
                f"{backend}: the update trigger does not watch {column}"
            )

    # ...and nothing the derivation reads is missing from that list. ``id`` is
    # in it: a primary key IS writable on both engines (measured), and the
    # projection stores one.
    read = set(PROJECTION_INPUT_COLUMNS)
    for column in (name.strip() for name in _DERIVED_FROM.split(",")):
        assert column in read, f"{column} is read but never watched"
    # ...and every statement really does select that list, so the check above is
    # about the SQL rather than about a constant nothing uses.
    for backend in ("sqlite", "postgres"):
        for statement in (
            _chunk_sql(backend),
            _own_rows_through(backend),
            _live_rows_through(backend),
        ):
            selected = statement.split("SELECT ")[1].split(" FROM ")[0]
            # Every statement selects the derivation's columns AND the key its
            # ordering sorts on, so the fold never re-derives that key itself.
            assert selected.startswith(_DERIVED_FROM), selected
            assert len(selected.split(", ")) == len(
                _DERIVED_FROM.split(", ")
            ) + 1, selected


def test_the_watched_metadata_keys_are_the_ones_the_grouper_consults():
    """The narrowing that keeps ordinary recall from rebuilding the projection.

    ``metadata`` is watched by KEY rather than whole, because ``access_count``
    and ``last_accessed`` share that document and are rewritten every time a
    memory is retrieved. Narrowing is only safe while the list really is every
    key the grouper reads, so the list is checked against
    ``session_grouping.py``'s own source rather than against a comment: a key
    consulted there and missing here is a change the stamp cannot see, and the
    symptom is a projection that quietly disagrees with the transcript.

    Read from the AST rather than by importing and introspecting, because what
    is being asked is "which literal keys does this module look up", and that is
    a property of the text — a runtime probe would only see the keys the corpus
    it was run against happened to contain.
    """
    import ast
    import inspect

    from kestrel_sovereign.storage import session_grouping
    from kestrel_sovereign.storage.conversation_sessions import (
        PROJECTION_METADATA_KEYS,
    )

    tree = ast.parse(inspect.getsource(session_grouping))
    consulted = {
        node.args[0].value
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "get"
        # The two names the metadata dict travels under in that module.
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id in {"meta", "metadata"}
        and node.args
        and isinstance(node.args[0], ast.Constant)
        and isinstance(node.args[0].value, str)
    }
    assert consulted, "found no metadata lookups at all — the scan is broken"
    assert consulted == set(PROJECTION_METADATA_KEYS), (
        "the grouper consults metadata keys the change stamp does not watch "
        f"(or the reverse): grouper={sorted(consulted)}, "
        f"watched={sorted(PROJECTION_METADATA_KEYS)}"
    )


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
        assert _accounting(await projection.accounted()) == (False, 0, 0, 0)
        assert await projection.observed_changes() == 0
        assert await projection.is_stale(), (
            "an unbuilt projection reported itself current"
        )

        assert (await projection.repair()).kind == REBUILT
        assert _accounting(await projection.accounted()) == (True, 0, 0, 0)
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
        assert _accounting(await projection.accounted()) == (False, 0, 0, 0), (
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
        # `_SESSION_PROJECTION_INDEX` carries name and table only — its columns
        # are generated per dialect at ensure time — so take the first two of
        # each rather than assuming both tuples are the same width.
        for entry in (_SESSION_PROJECTION_INDEX, _SESSION_FRONTIER_INDEX):
            name, table = entry[0], entry[1]
            assert await db._index_exists(name, table) is True, name
        assert await db._index_exists(
            "idx_conversation_agent_canonical", "conversation_history"
        ) is True, "the index the conversation list's ordering needs is absent"
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


class _CountsHistoryRows:
    """A database that also reports how many history rows were handed back.

    Wrapping the database rather than the projection because the quantity under
    test is I/O, and only the thing issuing the I/O can count it. Counting the
    rows a *frontier query selects* would be the mistake the measurement exists
    to catch: the old derivation selected a bounded chunk of ids and then read
    every row of every session those ids named.
    """

    def __init__(self, db):
        self._db = db
        self.rows_read = 0

    def __getattr__(self, name):
        return getattr(self._db, name)

    async def fetchall(self, sql, params=()):
        rows = await self._db.fetchall(sql, params)
        if "FROM conversation_history" in sql:
            self.rows_read += len(rows)
        return rows


@pytest.mark.asyncio
@pytest.mark.timeout(300)
async def test_a_whale_session_is_read_once_per_walk_not_once_per_chunk(tmp_path):
    """The bound this table exists for, measured in rows READ.

    #2877 binds completion wakes to their originating chat session, so a
    long-lived autonomous agent accumulates very large message counts in
    comparatively few sessions — the "whale" the epic measured a 4.9-second
    skip-scan against. A repair that recomputed each session named by a chunk
    from all of that session's live rows would read the whale once *per chunk*:
    quadratic in history, inside transactions this design advertises as short,
    and completely invisible to a test that only checked the frontier query's
    LIMIT.

    So the measurement is total rows read, and the walk is forced to be many
    chunks before it is taken. The bound is deliberately loose — twice the
    corpus, where the honest answer is once — because what it has to separate is
    ``O(history)`` from ``O(history × chunks)``, and at these numbers those are
    400 and 16,000.
    """
    rows = 400
    chunk = 10
    db = await AsyncDatabase.sqlite(str(tmp_path / "whale.db"))
    try:
        await _seed(
            db,
            [
                {
                    "content": f"whale turn {index}",
                    "role": "user" if index % 2 == 0 else "assistant",
                    # Inside the gap window, so this really is one cluster and
                    # not many that coalescing happens to merge.
                    "created_at": _at(index),
                    "metadata": {"session_id": UUID_A},
                }
                for index in range(rows)
            ],
        )
        counting = _CountsHistoryRows(db)

        class _CountsChunks(ConversationSessionProjection):
            chunks = 0

            async def _chunk(self, plan):
                self.chunks += 1
                return await super()._chunk(plan)

        projection = _CountsChunks(counting, AGENT, chunk_rows=chunk)
        assert (await projection.repair()).current

        assert projection.chunks >= rows // chunk, (
            "the walk was not chunked, so the bound below proves nothing"
        )
        assert counting.rows_read <= rows * 2, (
            f"the walk read {counting.rows_read} history rows for a corpus of "
            f"{rows} across {projection.chunks} chunks — the derivation is "
            "re-reading whole sessions rather than folding the rows it selected"
        )

        # ...and it got the right answer, so the bound was not bought by
        # reading less than the projection needed.
        stored = await ConversationSessionProjection(db, AGENT).get(UUID_A)
        assert stored["message_count"] == rows
        assert stored["user_message_count"] == rows // 2
        await _assert_agrees_with_the_grouper(
            db, ConversationSessionProjection(db, AGENT)
        )
    finally:
        await db.close()


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_a_session_whose_timestamps_go_backwards_escalates_to_the_transcript(
    tmp_path,
):
    """The precondition the fold rests on, checked instead of assumed.

    Folding combines a stored row with a chunk's slice through
    ``coalesce_sessions_by_session_id``, which merges boundaries by min/max. A
    single cluster's boundaries are *positional* — the first and last row as the
    walk sees them — and the walk is bounded by ids while the read path derives
    in ``canonical_order()``. The two agree only while ``created_at`` does not
    decrease as ``id`` increases. Every writer produces both from one INSERT, so
    it holds; but "holds in practice" is the reasoning that stops being true
    later, and the disagreement would be silent.

    ``minutes`` below inverts, so the last row by id (09:25) is not the last row
    in time (09:40). Escalating is what makes the stored answer the second one —
    which is what the conversation list shows, and what it orders by. Until
    round 6 this asserted the FIRST, because the helper that models the read
    path claimed id order; both sides were wrong together and agreed.

    Every timestamp here stays inside the gap window, so this really is one
    cluster the grouper reads positionally, and not several that coalescing would
    merge by min/max anyway — which is the shape that would make the case pass
    without testing anything.

    The walk is stopped mid-way first. That is not decoration: the exact
    derivation is bounded to rows at or below the chunk's end, and a derivation
    that read the session's *whole* history would still arrive at the right
    answer by the end of the walk while claiming, in the middle of it, to have
    accounted for rows the watermark says it has not reached — and the next chunk
    would then fold those rows in a second time.
    """
    minutes = [10, 30, 20, 15, 40, 25]
    db = await AsyncDatabase.sqlite(str(tmp_path / "backwards.db"))
    try:
        await _seed(
            db,
            [
                {
                    "content": f"turn at {minute}",
                    "role": "user" if index % 2 == 0 else "assistant",
                    "created_at": _at(minute),
                    "metadata": {"session_id": UUID_A},
                }
                for index, minute in enumerate(minutes)
            ],
        )
        # Two chunks of two would stop the walk with four of the six rows
        # accounted for — except that crossing the third row's backdating now
        # escalates to the whole-transcript pass, which finishes. A slice
        # grouped in isolation reads its boundaries positionally, and that is
        # only the grouper's answer while the slice is not one the grouper
        # would have split; deciding which it is needs the neighbours, so the
        # fold hands over rather than guessing.
        #
        # The bounded-chunk property is genuinely given up on this path, and
        # that is the trade: an inversion needs two overlapping writers and a
        # transaction-start clock, so it is rare, and paying a full pass when it
        # happens is cheaper than storing a last_message_at that precedes its
        # own started_at.
        partial = ConversationSessionProjection(
            db, AGENT, chunk_rows=2, step_budget=2
        )
        assert (await partial.repair()).kind == REBUILT, (
            "an inversion must escalate to the transcript pass, not be folded "
            "or derived from the session's own rows"
        )
        accounted = await partial.accounted()
        assert accounted.valid and accounted.complete
        await _assert_watermark_is_not_ahead(db, partial)

        projection = ConversationSessionProjection(db, AGENT, chunk_rows=2)
        assert (await projection.repair()).current

        stored = await projection.get(UUID_A)
        assert stored["message_count"] == len(minutes)
        assert coerce_session_timestamp(stored["last_message_at"]) == (
            coerce_session_timestamp(_at(max(minutes)))
        ), (
            "the session's last message is the latest one, and the transcript "
            "pass must report it"
        )
        await _assert_agrees_with_the_grouper(db, projection)
    finally:
        await db.close()


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_a_backdated_row_split_by_another_session_matches_the_grouper(
    tmp_path,
):
    """The case an isolated re-derivation gets wrong, and gets wrong silently.

    A session's cluster boundaries are not a property of its own rows. Another
    session's row between two of them ENDS a cluster, and
    ``coalesce_sessions_by_session_id`` then merges the pieces by min/max. Group
    the same session's rows alone and there is no split to find: one cluster,
    boundaries read positionally as the first and last row in id order.

    Those two readings agree only while ``created_at`` rises with ``id``.
    PostgreSQL does not promise that — ``NOW()`` is transaction-start time, so
    two overlapping writers can commit a later id carrying an earlier stamp —
    and the disagreement is not a near miss. Here the grouper reads
    ``started_at`` 09:00 / ``last_message_at`` 10:00 while the isolated reading
    reports them the other way round: a stored row whose last message precedes
    its own start, on the column the conversation list is ordered by, under a
    watermark claiming to be current.

    The separator is what makes this differ from
    ``test_a_session_whose_timestamps_go_backwards_escalates_to_the_transcript``:
    there the inversion sits inside one uninterrupted run, where the isolated
    reading happens to agree. Remove the ``UUID_B`` row below and the case stops
    testing anything.
    """
    db = await AsyncDatabase.sqlite(str(tmp_path / "backdated_split.db"))
    try:
        await _seed(
            db,
            [
                {
                    "content": "first of A",
                    "role": "user",
                    "created_at": _at(60),
                    "metadata": {"session_id": UUID_A},
                },
                {
                    "content": "B interleaves, ending A's cluster",
                    "role": "user",
                    "created_at": _at(65),
                    "metadata": {"session_id": UUID_B},
                },
                {
                    "content": "second of A, backdated before the first",
                    "role": "user",
                    "created_at": _at(0),
                    "metadata": {"session_id": UUID_A},
                },
            ],
        )
        projection = ConversationSessionProjection(db, AGENT, chunk_rows=10)
        for _ in range(10):
            if (await projection.repair()).current:
                break

        stored = await projection.get(UUID_A)
        started = coerce_session_timestamp(stored["started_at"])
        last = coerce_session_timestamp(stored["last_message_at"])
        assert started <= last, (
            "stored last_message_at precedes its own started_at: the session "
            "was derived from its own rows, which cannot see the split"
        )
        assert started == coerce_session_timestamp(_at(0))
        assert last == coerce_session_timestamp(_at(60))
        await _assert_agrees_with_the_grouper(db, projection)
    finally:
        await db.close()


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_a_backdated_row_in_iso_spelling_is_still_seen_as_backdated(
    tmp_path,
):
    """The order check must parse, not compare strings.

    SQLite history legitimately holds both spellings: the SQL one written by
    ``datetime('now')`` (``2020-01-01 10:00``) and the ISO one written by
    ``datetime.isoformat()`` (``2020-01-01T09:00``). Lexically ``T`` (0x54)
    sorts after a space (0x20), so the ISO row compares GREATER than the
    SQL-spelled row an hour after it, and a genuine decrease reads as an
    increase.

    That is worse than not checking at all. The guard's whole purpose is to
    refuse a fold it cannot prove correct; one that silently passes the case it
    was written for stores an inverted row and marks it current. This is the
    same 'T'-versus-space trap ``purge_channel_messages_since`` documents, which
    is why it is pinned here rather than left to reading.
    """
    db = await AsyncDatabase.sqlite(str(tmp_path / "iso_backdated.db"))
    try:
        await _seed(
            db,
            [
                {
                    "content": "first of A, SQL spelling",
                    "role": "user",
                    "created_at": "2020-01-01 10:00:00",
                    "metadata": {"session_id": UUID_A},
                },
                {
                    "content": "B ends A's cluster",
                    "role": "user",
                    "created_at": "2020-01-01 10:05:00",
                    "metadata": {"session_id": UUID_B},
                },
                {
                    "content": "second of A, ISO spelling, an hour EARLIER",
                    "role": "user",
                    "created_at": "2020-01-01T09:00:00",
                    "metadata": {"session_id": UUID_A},
                },
            ],
        )
        projection = ConversationSessionProjection(db, AGENT, chunk_rows=10)
        for _ in range(10):
            if (await projection.repair()).current:
                break

        stored = await projection.get(UUID_A)
        started = coerce_session_timestamp(stored["started_at"])
        last = coerce_session_timestamp(stored["last_message_at"])
        assert started <= last, (
            "the ISO-spelled backdating was compared as a raw string, so the "
            "decrease was missed and an inverted row was stored"
        )
        await _assert_agrees_with_the_grouper(db, projection)
    finally:
        await db.close()


def test_a_row_with_no_usable_timestamp_derives_the_same_answer_twice():
    """Derivation must be a function of the rows, not of when it ran.

    ``created_at`` is nullable and legacy SQLite rows can hold shapes that will
    not parse. Given one, ``group_messages_into_sessions`` dates it from the
    WALL CLOCK unless told otherwise — so the projection would persist whatever
    instant the repair happened to run at and mark itself current, while the
    same unchanged transcript re-derived a moment later produces different
    boundaries. Nothing notices: no row moved, so the change stamp never
    advances, and the cache disagrees with its source permanently.

    Two derivations of identical input must therefore be identical. Any clock
    reading in between is the defect, which is why this compares runs rather
    than checking a particular value.
    """
    rows = [
        (1, "user", '{"session_id": "%s"}' % UUID_A, "2020-01-01 10:00:00", UUID_A),
        (2, "user", '{"session_id": "%s"}' % UUID_A, None, UUID_A),
    ]
    first = project_transcript(rows)
    second = project_transcript(rows)

    # And the harder case: NOTHING parses, so there is no stamp in hand to pin
    # ``now`` to. Falling back to the default there reads the wall clock just
    # as surely as having no fallback at all.
    unusable = [
        (1, "user", '{"session_id": "%s"}' % UUID_B, None, UUID_B),
        (2, "user", '{"session_id": "%s"}' % UUID_B, "not-a-timestamp", UUID_B),
    ]
    assert [
        (p.session_id, p.started_at, p.last_message_at)
        for p in project_transcript(unusable)
    ] == [
        (p.session_id, p.started_at, p.last_message_at)
        for p in project_transcript(unusable)
    ], (
        "with no parseable timestamp anywhere the derivation fell back to the "
        "wall clock, so the same rows produce a different answer each run"
    )

    assert [(p.session_id, p.started_at, p.last_message_at) for p in first] == [
        (p.session_id, p.started_at, p.last_message_at) for p in second
    ], (
        "the same rows derived two different answers, so a clock was read: the "
        "unparseable timestamp was filled from wall time rather than from the "
        "rows in hand"
    )


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_a_row_arriving_mid_transcript_pass_is_not_counted_twice(tmp_path):
    """A snapshot must describe one frontier, and the watermark must be it.

    ``_rebuild_from_transcript`` reads the target id, then reads the rows. A row
    committed between those two reads lands INSIDE the derivation while the
    watermark recorded for it stops short — so a LATER repair sees that row as
    an append above the target and folds it into the session a second time,
    doubling its count and marking the result current.

    The fixture has to earn that. Reaching the transcript path with *unstamped*
    rows keeps every later repair on the same path, which re-derives from
    scratch and is idempotent — the double-fold never happens and the case
    passes while testing nothing. So the history here is fully stamped and
    reaches the transcript pass the other way: a backdated row split by another
    session, which the fold refuses because it cannot see the cluster split. The
    pass runs once; every repair after it folds.
    """
    db = await AsyncDatabase.sqlite(str(tmp_path / "frontier.db"))
    try:
        await _seed(
            db,
            [
                {"content": "A first", "role": "user", "created_at": _at(60),
                 "metadata": {"session_id": UUID_A}},
                {"content": "B splits A's cluster", "role": "user", "created_at": _at(65),
                 "metadata": {"session_id": UUID_B}},
                {"content": "A backdated", "role": "user", "created_at": _at(0),
                 "metadata": {"session_id": UUID_A}},
            ],
        )
        projection = ConversationSessionProjection(db, AGENT, chunk_rows=10)

        original = projection._max_id
        fired = False

        async def append_at_the_frontier():
            nonlocal fired
            target = await original()
            if not fired:
                fired = True
                # Commits after the frontier is read, before the rows are — so
                # it is derived, but the watermark about to be written does not
                # cover it.
                # Into a SEPARATE, inversion-free session. Appending to A
                # would be folded by nothing: A's backdating escalates every
                # later fold to the transcript pass too, which re-derives from
                # scratch and is idempotent — hiding the very double-count this
                # is here to catch.
                await _seed(
                    db,
                    [{"content": "arrived mid-pass", "role": "user",
                      "created_at": _at(120),
                      "metadata": {"session_id": UUID_C}}],
                )
            return target

        projection._max_id = append_at_the_frontier
        for _ in range(20):
            if (await projection.repair()).current:
                break
        assert fired, "the fixture never raced the frontier, so it tests nothing"

        projection._max_id = original
        for _ in range(20):
            if (await projection.repair()).current:
                break
        await _assert_agrees_with_the_grouper(db, projection)
    finally:
        await db.close()


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_a_crash_mid_repair_leaves_the_watermark_behind_never_ahead(modern):
    """A checkpoint may only advance when what it records is beyond loss.

    Each chunk records what it accounted for in the *same transaction* that
    writes the rows, so a pass that dies part-way leaves the watermark wherever
    the last chunk to COMMIT left it — never at one that did not finish. The
    forbidden direction is the other one: a watermark claiming rows nobody read,
    because nothing would ever revisit them.

    Run at ``chunk_rows=1`` so the walk is many chunks and the crash lands inside
    one rather than tidily between passes.
    """
    db, store, projection = modern
    ids = await _message_ids(db)
    assert await store.delete_message(ids["A human turn"])
    assert (await projection.repair()).current

    class _Crashes(ConversationSessionProjection):
        """Dies while storing the third session of the walk."""

        written = 0

        async def _store(self, session):
            if self.written == 2:
                raise RuntimeError("crash mid-repair")
            self.written += 1
            return await super()._store(session)

    await ConversationSessionProjection(db, AGENT).rebuild()
    await db.execute(
        "UPDATE conversation_session_watermarks SET accounted_valid = 0 "
        "WHERE agent_id = ?",
        (AGENT,),
    )
    with pytest.raises(Exception, match="crash mid-repair"):
        await _Crashes(db, AGENT, chunk_rows=1).repair()

    after = await projection.accounted()
    assert not after.complete, (
        "a pass that died part-way recorded a walk that had reached its target"
    )
    assert after.through < await _frontier(db)
    assert await projection.is_stale()
    await _assert_watermark_is_not_ahead(db, projection)

    # And the redo costs nothing but the redo.
    await _assert_repaired_projection_is_true(db, projection)


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_a_chunk_that_dies_writes_neither_its_rows_nor_its_watermark(modern):
    """Atomic by construction — the property the epoch machinery had to build.

    A chunk stores its sessions and then records what it accounted for. If the
    recording fails, the rows must go with it: rows without their watermark is
    the harmless direction only because it never happens *within* a chunk, and
    rows the next pass would find already stored while the watermark says it
    never got there is how a projection starts disagreeing with itself.

    The mutation this defends against is moving the watermark write out of the
    step's transaction, which no assertion about a *successful* repair can catch,
    because a successful repair looks identical either way.
    """
    db, store, projection = modern
    await store.add_conversation("user", "a turn to account for", session_id=UUID_B)
    before_rows = await projection.list()
    before = await projection.accounted()

    class _FailsToRecord(ConversationSessionProjection):
        async def _record(self, watermark):
            await super()._record(watermark)
            raise RuntimeError("the chunk died after recording")

    with pytest.raises(Exception, match="the chunk died after recording"):
        await _FailsToRecord(db, AGENT).repair()

    assert await projection.list() == before_rows, (
        "a chunk that could not finish left its rows behind, so the projection "
        "holds work no watermark accounts for"
    )
    assert await projection.accounted() == before
    await _assert_repaired_projection_is_true(db, projection)


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_a_walk_resumes_from_the_chunk_that_last_committed(modern):
    """A crash costs one chunk's redo, not the whole walk.

    That is the difference between a watermark that records *progress* and one
    that only records completion, and it is why the stored state carries both how
    far the walk has got and where it ends. A pass arriving after a crash finds a
    part-walked projection and continues it, rather than discarding what previous
    chunks paid for.
    """
    db, _store, projection = modern

    class _Crashes(ConversationSessionProjection):
        written = 0

        async def _store(self, session):
            if self.written == 2:
                raise RuntimeError("crash mid-repair")
            self.written += 1
            return await super()._store(session)

    with pytest.raises(Exception, match="crash mid-repair"):
        await _Crashes(db, AGENT, chunk_rows=1).rebuild()
    interrupted = await projection.accounted()
    assert interrupted.valid and not interrupted.complete
    assert interrupted.through > 0, (
        "no chunk committed, so 'resumes' has nothing to prove"
    )

    resumed = []

    class _Records(ConversationSessionProjection):
        async def _chunk(self, plan):
            resumed.append(plan)
            return await super()._chunk(plan)

    outcome = await _Records(db, AGENT, chunk_rows=1).repair()

    assert outcome.kind == INCREMENTAL, (
        "the walk started again from nothing rather than resuming"
    )
    assert resumed[0].through == interrupted.through
    assert resumed[0].target == interrupted.target
    assert not await projection.is_stale()
    await _assert_agrees_with_the_grouper(db, projection)


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_a_row_arriving_during_a_chunk_leaves_the_watermark_behind(modern):
    """The stamp is read before the derivation, never after it.

    A row that lands while a chunk is running may or may not be in the rows that
    chunk read — it cannot know which. Recording the stamp it *started* from
    makes the answer always "behind", and behind is the recoverable direction.

    The write is injected inside the chunk, which on SQLite means it joins that
    chunk's own transaction: the row and its ledger bump commit *with* the
    watermark that must nonetheless exclude them. That is the hardest form of the
    claim — reading the stamp after the derivation would record a state covering
    a row this pass may never have looked at, and the two writes committing
    together would make it look deliberate.
    """
    db, store, projection = modern
    ids = await _message_ids(db)
    assert await store.delete_message(ids["A human turn"])

    class _WritesMidChunk(ConversationSessionProjection):
        arrived = False

        async def _fold(self, rows, through):
            written = await super()._fold(rows, through)
            if not self.arrived:
                self.arrived = True
                await store.add_conversation(
                    "user", "arrived mid-chunk", session_id=UUID_B
                )
            return written

    # One step, so what is asserted is one step's own record rather than the
    # loop's: a later step would read the higher stamp and legitimately catch up,
    # which is correct behaviour and would hide the ordering this case is about.
    outcome = await _WritesMidChunk(db, AGENT, step_budget=1).repair()

    assert not outcome.current
    accounted = await projection.accounted()
    assert accounted.stamp < await projection.observed_changes(), (
        "the watermark accounted for a row event that happened after it read "
        "the stamp"
    )
    assert accounted.target < await _frontier(db)
    assert await projection.is_stale()
    await _assert_watermark_is_not_ahead(db, projection)
    await _assert_repaired_projection_is_true(db, projection)


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_a_repair_that_runs_out_of_chunks_says_it_is_still_behind(modern):
    """A budget, not a loop until current — and the difference is reported.

    An agent written to faster than a walk can cross it would keep a "repair
    until nothing is left" call running forever. This contract permits a
    projection to be behind and requires it to say so, so the pass stops and
    reports, and what it did commit is kept.
    """
    db, _store, projection = modern

    outcome = await ConversationSessionProjection(
        db, AGENT, chunk_rows=1, step_budget=2
    ).rebuild()

    assert outcome == 1, (
        "the count must be sessions the projection HOLDS, not chunk writes. "
        "The budget bought two chunks of one ROW each, and both rows belong to "
        "the same session — which is upserted once per chunk. Summing the "
        "writes reports two sessions for a projection containing one, which is "
        "the number a caller would then reconcile against list()"
    )
    assert len(await projection.list()) == outcome, (
        "the returned count and the table must agree; they are the same claim"
    )
    accounted = await projection.accounted()
    assert accounted.valid and not accounted.complete
    assert await projection.is_stale()
    # What the budget DID buy is committed: chunks commit one at a time, so a
    # pass that stops early leaves its finished chunks standing.
    assert await projection.list(), (
        "a pass that stopped early left nothing behind, so its chunks were not "
        "committing as they went"
    )
    await _assert_watermark_is_not_ahead(db, projection)

    assert (await projection.repair()).current
    await _assert_agrees_with_the_grouper(db, projection)


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_two_connections_repairing_one_file_take_turns(tmp_path):
    """SQLite's half of the serialization, which is the transaction itself.

    On PostgreSQL a repair step is held apart from its peers by a row lock. On
    SQLite there is no second mechanism, because a write transaction is already
    exclusive — but only if it *is* a write transaction from the start. A step
    reads its state and then writes it, so under the default deferred ``BEGIN``
    the second connection reads a snapshot, discovers on its first write that the
    database has moved, and cannot upgrade: SQLite answers ``BUSY_SNAPSHOT``,
    which ``busy_timeout`` does not retry because waiting cannot make a stale
    snapshot fresh. Asking for the writer slot at ``BEGIN`` turns that into
    taking turns.

    Two genuinely separate connections to one file, because that is the only
    shape that can tell the difference. Within one connection the backend's own
    write lock already serializes transactions, so a same-connection case would
    pass either way — which is exactly what makes this worth a case of its own.
    """
    path = str(tmp_path / "two-connections.db")
    first = await AsyncDatabase.sqlite(path)
    second = await AsyncDatabase.sqlite(path)
    try:
        await _seed(first, MODERN_CORPUS)
        outcomes = await asyncio.gather(
            ConversationSessionProjection(first, AGENT, chunk_rows=1).repair(),
            ConversationSessionProjection(second, AGENT, chunk_rows=1).repair(),
        )
        assert all(outcome.sessions >= 0 for outcome in outcomes)

        projection = ConversationSessionProjection(first, AGENT)
        assert (await projection.repair()).current
        assert not await projection.is_stale()
        await _assert_agrees_with_the_grouper(first, projection)
    finally:
        await first.close()
        await second.close()


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_two_repairs_racing_leave_a_projection_that_is_true(modern):
    """The whole concurrency story: overlapping idempotent work.

    No epoch, no fence, no compare-and-swap — three passes are simply started at
    once over the same agent. Each chunk recomputes from the rows rather than
    incrementing, so duplicated work is harmless; each writes its rows and its
    watermark together, so no pass can leave the other's rows under its own
    claim. What must hold at the end is what holds after one pass: the table
    agrees with the grouper, and it does not call itself current unless it is.

    Bounded by a timeout because a concurrency case that can hang cannot report
    the bug it exists to catch — in CI it is a timed-out job with no failing
    assertion, indistinguishable from flake.
    """
    db, store, projection = modern
    ids = await _message_ids(db)
    assert await store.delete_message(ids["A human turn"])

    await asyncio.gather(
        *(
            ConversationSessionProjection(db, AGENT, chunk_rows=1).repair()
            for _ in range(3)
        )
    )

    assert not await projection.is_stale(), (
        "racing repairs left the projection stale with nobody left to repair it"
    )
    await _assert_watermark_is_not_ahead(db, projection)
    await _assert_agrees_with_the_grouper(db, projection)
    assert ids["A human turn"] is not None


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

``project_transcript`` drops any session whose id
    :func:`is_stampable_session_id` rejects, and the column contract means no
    shipped write path can produce one. That makes the filter a clause no
    ordinary test can exercise, which Phase A's Finding named as the worst kind:
    it reads as protection while a mutation removing it goes unnoticed.

    So the violation is written directly into the column, bypassing
    ``column_session_id`` the way only a future bug could. The projection's
    primary key is ``(agent_id, session_id)`` and Phase C will read it, so a row
    keyed by a value the contract forbids is a key no reader can round-trip.
    Absent is the permitted direction.

    On the all-stamped corpus deliberately: that is the derivation a chunk
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
    id, and there is no answer that is not a guess — so the session the grouper
    files that row under is refused and logged rather than stored.

    **Which session is tainted is the point** (round-8 review). The damage is to
    B, the session the reader attributes the row to — its stored count would
    otherwise stand as though the row were not there. C, whose column the row
    claims, keeps its own legitimate rows: they really are session C, and
    dropping them would lose a real conversation over an unrelated row's
    corruption.

    This used to come out the other way round. The fold returned "nothing" for
    C, the caller took that as "forget C", and B — never recomputed, because an
    incremental chunk only touches the sessions its own rows name — kept a stale
    count under a watermark recorded as current. Escalating to the transcript is
    what makes the refusal land on the session that is actually wrong.

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
    assert await projection.get(UUID_B) is None, (
        "the session the grouper files the divergent row under kept a count "
        "that does not include it, under a watermark claiming to be current"
    )
    surviving = await projection.get(UUID_C)
    assert surviving is not None, (
        "C's own legitimate rows were dropped because an unrelated row claimed "
        "its id — a real conversation lost to another row's corruption"
    )
    assert surviving["message_count"] == 2, surviving
    assert not await projection.is_stale(), (
        "the projection refused a session and then reported itself stale "
        "forever; a refusal is an answer, not an incomplete repair"
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


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_the_preview_pointer_follows_the_order_the_list_reads_in(tmp_path):
    """The projection must pick the preview the conversation list would show.

    ``first_user_message_id`` is what the card shows, and the grouper picks the
    first eligible user row *in the order it is handed the transcript*.
    ``/api/conversations`` hands it rows in ``canonical_order()`` — it selects
    ``created_at DESC`` and reverses — so wherever id order and time order
    disagree, deriving in id order names a different row and the card shows a
    different message.

    PostgreSQL makes that disagreement reachable rather than theoretical:
    ``NOW()`` is transaction-start time, so a writer that began earlier and
    committed later carries a lower timestamp on a higher id. Here the second
    row by id is the first in time, so the two orders name different previews
    and only one of them is the one the user sees.

    Seeded through :func:`_seed` with explicit stamps, because reproducing the
    inversion through two live overlapping writers would be timing-dependent
    while the property under test is not.
    """
    db = await AsyncDatabase.sqlite(str(tmp_path / "inverted.db"))
    try:
        await _seed(
            db,
            [
                {"content": "later id, earlier clock", "role": "user",
                 "created_at": _at(20), "metadata": {"session_id": UUID_A}},
                {"content": "earlier id, later clock", "role": "user",
                 "created_at": _at(10), "metadata": {"session_id": UUID_A}},
                {"content": "reply", "role": "assistant",
                 "created_at": _at(25), "metadata": {"session_id": UUID_A}},
            ],
        )
        ids = {row["content"]: row["id"] for row in await _live_history(db)}
        chronologically_first = ids["earlier id, later clock"]
        assert chronologically_first == max(
            ids["later id, earlier clock"], ids["earlier id, later clock"]
        ), "the corpus stopped inverting, so this proves nothing"

        projection = ConversationSessionProjection(db, AGENT)
        await projection.repair()

        stored = await projection.get(UUID_A)
        assert stored["first_user_message_id"] == chronologically_first, (
            "the projection previews the lowest-id user row, but the list "
            "previews the earliest one — the card would change on the day the "
            "projection replaced the derivation"
        )
        await _assert_agrees_with_the_grouper(db, projection)
    finally:
        await db.close()


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_the_list_orders_ties_deterministically(tmp_path):
    """Equal timestamps must not be resolved by whatever the backend feels like.

    ``/api/conversations`` used to select ``ORDER BY created_at DESC`` with no
    tie-break. Rows sharing a timestamp are common — a wake and the turn it
    triggers are written in the same transaction, and SQLite history is stored
    to the second — so the same history could be handed to the grouper in two
    different orders on two calls, and a session boundary or a preview could
    move without anything having changed.

    Asserted through a real database rather than a double: the ordering is
    decided by SQL, so a mock storage would only prove that the test knows what
    it wrote.
    """
    from kestrel_sovereign.privacy import PrivacyMode
    from kestrel_sovereign.storage import AsyncStorage
    from kestrel_sovereign.storage.privacy_wrapper import PrivacyEnforcingStorage

    async with AsyncStorage(str(tmp_path / "ties.db"), agent_id=AGENT) as storage:
        for content in ("first", "second", "third"):
            await storage.db.execute(
                "INSERT INTO conversation_history "
                "(agent_id, role, content, metadata, created_at) "
                "VALUES (?, 'user', ?, NULL, ?)",
                (AGENT, content, _at(0)),
            )
        wrapper = PrivacyEnforcingStorage(storage, PrivacyMode.NORMAL)

        rows = await wrapper.query_conversations(AGENT, limit=10)
        ids = [row[0] for row in rows]

        assert len(ids) == 3, "the corpus did not land, so the order proves nothing"
        assert ids == sorted(ids, reverse=True), (
            "rows sharing a timestamp came back in an order the backend chose; "
            f"got {ids}. The list must break ties on id so the same history "
            "always groups the same way."
        )


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_rewriting_a_message_id_is_detected(seeded):
    """A primary key is not immutable, on either engine.

    The watched-column list once omitted ``id`` on the grounds that a primary
    key "cannot be updated in place". It can: ``UPDATE ... SET id = ?`` was
    measured to succeed against both SQLite and PostgreSQL. The projection
    leans on ``id`` for the canonical tie-break, the chunk frontier, and
    ``first_user_message_id`` — which it *stores* — so an unwatched rewrite
    leaves a stored pointer to a row that no longer carries that id while the
    watermark still claims to be current.

    Nothing in this codebase rewrites an id. Maintenance and import SQL is the
    traffic this exists for, and it is precisely the traffic that never goes
    through a write path that could have remembered to invalidate.
    """
    db, _store, projection = seeded
    assert not await projection.is_stale(), "the fixture must start current"

    moved = int(await db.fetchval(
        "SELECT MAX(id) FROM conversation_history WHERE agent_id = ?", (AGENT,)
    ))
    await db.execute(
        "UPDATE conversation_history SET id = ? WHERE id = ?",
        (moved + 1000, moved),
    )

    assert await projection.is_stale(), (
        "a message id was rewritten under the projection and it still reports "
        "itself current"
    )


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_tied_sessions_order_the_same_way_in_both_paths(tmp_path):
    """A page of sessions must not depend on which path produced it.

    Both used to sort on ``last_message_at`` alone. The list did it in Python,
    where a stable sort left ties in grouping order; the projection did it in
    SQL, where ties came back by ``session_id``. Ties are ordinary — SQLite
    stores history to the second — so with a limit applied the two paths could
    put a different session on the page, and swapping one for the other in
    Phase C would have silently reordered the sidebar.

    Here the later-appearing session sorts FIRST lexically, so grouping order
    and the canonical order genuinely differ and an agreeing answer cannot be
    an accident of the corpus.
    """
    from kestrel_sovereign.storage.session_grouping import (
        coalesce_sessions_by_session_id,
        group_messages_into_sessions,
        sort_sessions,
    )

    db = await AsyncDatabase.sqlite(str(tmp_path / "ties.db"))
    try:
        await _seed(
            db,
            [
                {"content": "B first in the transcript", "role": "user",
                 "created_at": _at(0), "metadata": {"session_id": UUID_B}},
                {"content": "A second", "role": "user",
                 "created_at": _at(0), "metadata": {"session_id": UUID_A}},
            ],
        )
        projection = ConversationSessionProjection(db, AGENT)
        await projection.repair()

        grouped = sort_sessions(coalesce_sessions_by_session_id(
            group_messages_into_sessions(await _live_history(db),
                                         keep_empty_markers=True)
        ))
        reader = [str(session["session_id"]) for session in grouped]
        projected = [row["session_id"] for row in await projection.list()]

        assert len({s["last_message_at"] for s in grouped}) == 1, (
            "the sessions stopped tying, so nothing about ties is under test"
        )
        # Asserted against the RULE, not against each other. Comparing the two
        # paths alone is satisfied by them being wrong together: drop the
        # tie-break and the reader falls back to grouping order while SQLite
        # returns the projection rows in insertion order — which is the same
        # grouping order, so they still agree and the mutant lives. The
        # canonical answer is named instead.
        canonical = [UUID_A, UUID_B]
        assert reader == canonical, (
            f"the list ordered tied sessions {reader}; ties must fall back to "
            "session_id, not to whatever order grouping emitted"
        )
        assert projected == canonical, (
            f"the projection ordered tied sessions {projected}"
        )
    finally:
        await db.close()


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_mixed_timestamp_spellings_are_ordered_chronologically(tmp_path):
    """SQLite history mixes two spellings, and raw text order inverts them.

    ``created_at`` is TEXT on SQLite and legacy rows legitimately hold both the
    ISO form (``T`` separator) and the SQL form (space). ``"T"`` is 0x54 and a
    space is 0x20, so ``'2026-03-01T09:00:00'`` compares GREATER than
    ``'2026-03-01 10:00:00'`` — an hour-earlier row sorts last. Measured: those
    two rows come back reversed under a raw ``ORDER BY created_at``, and
    ``julianday()`` restores chronology.

    Asked of ``query_conversations``, because that is where the order is
    actually consumed: the conversation list pages by it and reverses it before
    grouping, so getting it wrong reorders the transcript the grouper sees. The
    projection's ordinary chunk path walks ids and never reaches this clause,
    which is why an earlier version of this test passed with the fix removed.
    """
    from kestrel_sovereign.privacy import PrivacyMode
    from kestrel_sovereign.storage import AsyncStorage
    from kestrel_sovereign.storage.privacy_wrapper import PrivacyEnforcingStorage

    async with AsyncStorage(str(tmp_path / "spellings.db"), agent_id=AGENT) as storage:
        for content, stamp in (
            ("earlier, ISO spelling", "2026-03-01T09:00:00"),
            ("later, SQL spelling", "2026-03-01 10:00:00"),
        ):
            await storage.db.execute(
                "INSERT INTO conversation_history "
                "(agent_id, role, content, metadata, created_at) "
                "VALUES (?, 'user', ?, NULL, ?)",
                (AGENT, content, stamp),
            )
        wrapper = PrivacyEnforcingStorage(storage, PrivacyMode.NORMAL)

        rows = await wrapper.query_conversations(AGENT, limit=10)
        newest_first = [row[2] for row in rows]

        assert newest_first == ["later, SQL spelling", "earlier, ISO spelling"], (
            "the list paged these newest-first by TEXT, not by time: the "
            f"space-spelled 10:00 row sorted below the T-spelled 09:00 one. "
            f"Got {newest_first}"
        )


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_appending_elsewhere_does_not_restale_an_undatable_session(tmp_path):
    """An undatable row's substitute stamp must not depend on other sessions.

    The substitute is the stamp of the row before it — LOCAL. A previous fix
    used the transcript's MAXIMUM stamp, which is deterministic but global: a
    row appended to session B re-dated the undatable row in session A, so A's
    derived activity time moved while an incremental repair (which only
    recomputes the sessions its own chunk names) never touched A. The watermark
    then recorded the chunk as accounted for over a stale A.

    So this appends to B and asks A whether it changed.
    """
    db = await AsyncDatabase.sqlite(str(tmp_path / "undatable.db"))
    try:
        await _seed(
            db,
            [
                {"content": "A datable", "role": "user", "created_at": _at(0),
                 "metadata": {"session_id": UUID_A}},
                {"content": "A undatable", "role": "assistant",
                 "created_at": "not-a-date", "metadata": {"session_id": UUID_A}},
                {"content": "B datable", "role": "user", "created_at": _at(100),
                 "metadata": {"session_id": UUID_B}},
            ],
        )
        projection = ConversationSessionProjection(db, AGENT)
        await projection.repair()
        before = await projection.get(UUID_A)

        await _seed(db, [{"content": "B later", "role": "user",
                          "created_at": _at(10_000),
                          "metadata": {"session_id": UUID_B}}])
        assert await projection.is_stale()
        await projection.repair()

        assert await projection.get(UUID_A) == before, (
            "appending to another session moved session A's stored row; an "
            "incremental repair would leave that unnoticed"
        )
        await _assert_agrees_with_the_grouper(db, projection)
    finally:
        await db.close()


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_a_renumbered_row_is_not_counted_twice(tmp_path):
    """Moving an existing row's id must not read as a new message arriving.

    The catch-up test is "the change counter moved by exactly the number of
    rows now standing above the bookmark". An ``UPDATE ... SET id = ?`` that
    lifts an already-counted row above the bookmark satisfies it exactly — one
    change, one row above — so the row is folded into its own session a second
    time and the inflated count is recorded as current.
    """
    db = await AsyncDatabase.sqlite(str(tmp_path / "renumber.db"))
    try:
        await _seed(
            db,
            [
                {"content": "one", "role": "user", "created_at": _at(0),
                 "metadata": {"session_id": UUID_A}},
                {"content": "two", "role": "assistant", "created_at": _at(1),
                 "metadata": {"session_id": UUID_A}},
            ],
        )
        projection = ConversationSessionProjection(db, AGENT)
        assert (await projection.repair()).current
        assert (await projection.get(UUID_A))["message_count"] == 2

        # The LAST row by id, which is also the latest in time. Moving an
        # earlier one is caught for free by the fold's monotonicity guard — the
        # slice would start before the stored row's last message and escalate —
        # so it would prove nothing about this test's subject.
        top = int(await db.fetchval(
            "SELECT MAX(id) FROM conversation_history WHERE agent_id = ?", (AGENT,)
        ))
        await db.execute(
            "UPDATE conversation_history SET id = ? WHERE id = ?",
            (top + 100, top),
        )

        await projection.repair()

        assert (await projection.get(UUID_A))["message_count"] == 2, (
            "the renumbered row was counted a second time"
        )
        await _assert_agrees_with_the_grouper(db, projection)
    finally:
        await db.close()


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_a_nonpositive_id_is_still_walked(tmp_path):
    """A row numbered zero or below must not be invisible to the walk.

    Every walk selects ``id > through`` and a rebuild starts at ``through = 0``,
    so a row imported with a nonpositive id is never folded — while the
    watermark is still recorded complete and ``is_stale()`` answers false. The
    session it belongs to then carries a permanently understated count.

    Only maintenance or import SQL produces such an id; ``AUTOINCREMENT`` and
    ``bigserial`` do not. That is why it needs a case rather than a schema
    assumption.
    """
    db = await AsyncDatabase.sqlite(str(tmp_path / "nonpositive.db"))
    try:
        await _seed(
            db,
            [{"content": "ordinary", "role": "user", "created_at": _at(5),
              "metadata": {"session_id": UUID_A}}],
        )
        await db.execute(
            "INSERT INTO conversation_history "
            "(id, agent_id, role, content, metadata, session_id, created_at) "
            "VALUES (?, ?, 'user', 'imported below zero', ?, ?, ?)",
            (-7, AGENT, json.dumps({"session_id": UUID_A}), UUID_A, _at(0)),
        )

        projection = ConversationSessionProjection(db, AGENT, chunk_rows=1)
        await projection.repair()

        assert (await projection.get(UUID_A))["message_count"] == 2, (
            "the row numbered -7 was never walked, so its session undercounts"
        )
        await _assert_agrees_with_the_grouper(db, projection)
    finally:
        await db.close()


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_a_purge_is_not_undone_by_a_repair_that_was_already_running(tmp_path):
    """"Leave no trace" has to survive a repair that started before the purge.

    A transcript rebuild derives OUTSIDE the repair lock — it is an unbounded
    pass and holding a lock across it is the wedge this design refuses. So it
    can read history, have that history purged underneath it, and then publish
    rows describing messages that no longer exist.

    The currency half is worse than the privacy half. A first post-upgrade
    repair reads a stamp of 0, because the ledger is created empty beside a
    history that is already full. The purge then erases the ledger, which also
    reads as 0. So the resurrected projection sits under a valid watermark whose
    stamp matches the ledger exactly, and ``is_stale()`` answers **false**: the
    projection reports itself a faithful cache of a history that is empty.

    Staged rather than raced: the rebuild is paused between deriving and
    publishing, which is the window, and a real race would only reach it
    sometimes.
    """
    from kestrel_sovereign.privacy import PrivacyMode
    from kestrel_sovereign.storage import AsyncStorage
    from kestrel_sovereign.storage.privacy_wrapper import PrivacyEnforcingStorage

    async with AsyncStorage(str(tmp_path / "resurrect.db"), agent_id=AGENT) as storage:
        db = storage.db
        # History first, projection schema second: the ledger is then empty
        # beside a full history, which is the post-upgrade shape that makes the
        # published stamp 0.
        await _seed(db, [
            {"content": "secret turn", "role": "user", "created_at": _at(0),
             "metadata": {"session_id": UUID_A}},
            # Unstamped, which is what routes the repair through the transcript
            # derivation rather than the chunked walk. Load-bearing: without it
            # there is no derive-then-publish window to pause in.
            {"content": "unstamped reply", "role": "assistant",
             "created_at": _at(1), "metadata": {}},
        ])
        await db.ensure_session_projection_schema()

        derived = asyncio.Event()
        release = asyncio.Event()

        # Pause between the derivation's READ of history and the transaction
        # that publishes it — the only window there is. Pausing inside the
        # transaction would hold SQLite's writer slot and deadlock the purge,
        # proving only that the two cannot interleave once publishing has
        # begun. Held at `fetchall` so the production `_rebuild_from_transcript`
        # runs unaltered: a staged copy of it here would be a test asserting
        # against its own reimplementation.
        import kestrel_sovereign.storage.conversation_sessions as module

        real_fetchall = db.fetchall
        transcript_sql = module._live_rows_through(db.backend_type)

        async def _pausing_fetchall(query, params=()):
            rows = await real_fetchall(query, params)
            if query == transcript_sql and not derived.is_set():
                derived.set()
                await asyncio.wait_for(release.wait(), 30)
            return rows

        db.fetchall = _pausing_fetchall
        projection = ConversationSessionProjection(db, AGENT)
        slow = asyncio.create_task(projection.repair())
        try:
            await asyncio.wait_for(derived.wait(), 30)
            wrapper = PrivacyEnforcingStorage(storage, PrivacyMode.EPHEMERAL)
            await storage.conversation.purge_all_since("1970-01-01", reason="test")
            await wrapper.purge_ephemeral_session(reason="test")
        finally:
            release.set()
        await asyncio.wait_for(asyncio.shield(slow), 30)
        db.fetchall = real_fetchall

        projection = ConversationSessionProjection(db, AGENT)
        assert await projection.list() == [] or await projection.is_stale(), (
            "a repair that started before the purge republished the purged "
            "session, and the projection reports itself current over it"
        )
        assert await projection.list() == [], (
            "purged conversation content is standing in the projection after "
            "an EPHEMERAL exit"
        )
        assert await db.fetchval(
            "SELECT COUNT(*) FROM conversation_session_watermarks "
            "WHERE agent_id = ?", (AGENT,)
        ) == 0, (
            "the repair left a watermark row naming this agent in a table the "
            "purge had emptied"
        )


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_a_purge_is_not_undone_when_the_stamp_cannot_witness_it(tmp_path):
    """The same race, in the state where the change stamp says nothing.

    The stamp catches an ordinary concurrent mutation, and in the sibling test
    above it is what does. It cannot catch this one. An upgrade creates the
    ledger EMPTY beside a ``conversation_history`` that is already full — the
    module docstring names that state — so a first repair derives at stamp 0.
    The sweep then erases the ledger, and a missing ledger also reads 0.
    Unchanged, by a route that changed everything: history is gone and the
    projection about to be published describes it.

    The empty ledger is arranged by deleting the row, which is precisely the
    post-upgrade shape rather than a state invented for this test — every row
    of history is still there, and only the counter that has never seen them is
    absent.
    """
    from kestrel_sovereign.privacy import PrivacyMode
    from kestrel_sovereign.storage import AsyncStorage
    from kestrel_sovereign.storage.privacy_wrapper import PrivacyEnforcingStorage
    import kestrel_sovereign.storage.conversation_sessions as module

    async with AsyncStorage(str(tmp_path / "upgrade.db"), agent_id=AGENT) as storage:
        db = storage.db
        await _seed(db, [
            {"content": "secret turn", "role": "user", "created_at": _at(0),
             "metadata": {"session_id": UUID_A}},
            {"content": "unstamped reply", "role": "assistant",
             "created_at": _at(1), "metadata": {}},
        ])
        # The post-upgrade shape: full history, ledger that has never counted it.
        await db.execute(
            "DELETE FROM conversation_history_changes WHERE agent_id = ?", (AGENT,)
        )
        projection = ConversationSessionProjection(db, AGENT)
        assert await projection.observed_changes() == 0, "the fixture is not post-upgrade"

        derived = asyncio.Event()
        release = asyncio.Event()
        real_fetchall = db.fetchall
        transcript_sql = module._live_rows_through(db.backend_type)

        async def _pausing_fetchall(query, params=()):
            rows = await real_fetchall(query, params)
            if query == transcript_sql and not derived.is_set():
                derived.set()
                await asyncio.wait_for(release.wait(), 30)
            return rows

        db.fetchall = _pausing_fetchall
        slow = asyncio.create_task(projection.repair())
        try:
            await asyncio.wait_for(derived.wait(), 30)
            wrapper = PrivacyEnforcingStorage(storage, PrivacyMode.EPHEMERAL)
            await storage.conversation.purge_all_since("1970-01-01", reason="test")
            await wrapper.purge_ephemeral_session(reason="test")
        finally:
            release.set()
        await asyncio.wait_for(asyncio.shield(slow), 30)
        db.fetchall = real_fetchall

        fresh = ConversationSessionProjection(db, AGENT)
        assert await fresh.observed_changes() == 0, (
            "the ledger was not erased, so the stamp CAN witness the purge and "
            "this case proves nothing the sibling test does not"
        )
        assert await fresh.list() == [], (
            "purged history was republished under a stamp that matches the "
            "erased ledger exactly"
        )


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_a_scoped_purge_is_not_undone_by_a_running_repair(tmp_path):
    """The case the emptiness check cannot see: history SURVIVES the purge.

    An EPHEMERAL stint that leaks one turn beside legitimate pre-entry history
    is swept by ``created_at``, so the leak goes and the rest stays. History is
    therefore not empty when a repair that started earlier publishes, and the
    "is there any live row" question answers yes — while the snapshot in hand
    still contains the leaked session.

    What the projection would put back is not message text (it stores a pointer
    and counts, never content) but a row naming a session that existed, when,
    and how long it was. "Leave no trace" is about the trace, so the stamp is
    asked as well: every purge is a row event, so an erased leak moves the
    ledger, and a snapshot derived before it is refused.
    """
    from kestrel_sovereign.privacy import PrivacyMode
    from kestrel_sovereign.storage import AsyncStorage
    from kestrel_sovereign.storage.privacy_wrapper import PrivacyEnforcingStorage
    import kestrel_sovereign.storage.conversation_sessions as module

    async with AsyncStorage(str(tmp_path / "scoped.db"), agent_id=AGENT) as storage:
        db = storage.db
        await storage.conversation.add_conversation(
            "user", "legitimate, pre-EPHEMERAL", session_id=UUID_A
        )
        # Past the per-second watermark boundary before entering, so the row
        # above is strictly older than entry and the scoped sweep must keep it.
        await asyncio.sleep(1.05)
        wrapper = PrivacyEnforcingStorage(storage, PrivacyMode.EPHEMERAL)
        await asyncio.sleep(1.05)
        await storage.conversation.add_conversation(
            "user", "leaked turn", session_id=UUID_B
        )
        # Unstamped, to route the repair through the transcript derivation.
        await _seed(db, [{"content": "unstamped", "role": "assistant",
                          "created_at": _at(900), "metadata": {}}])

        projection = ConversationSessionProjection(db, AGENT)
        derived = asyncio.Event()
        release = asyncio.Event()
        real_fetchall = db.fetchall
        transcript_sql = module._live_rows_through(db.backend_type)

        async def _pausing_fetchall(query, params=()):
            rows = await real_fetchall(query, params)
            if query == transcript_sql and not derived.is_set():
                derived.set()
                await asyncio.wait_for(release.wait(), 30)
            return rows

        db.fetchall = _pausing_fetchall
        slow = asyncio.create_task(projection.repair())
        try:
            await asyncio.wait_for(derived.wait(), 30)
            await wrapper.purge_ephemeral_session(reason="test")
        finally:
            release.set()
        await asyncio.wait_for(asyncio.shield(slow), 30)
        db.fetchall = real_fetchall

        fresh = ConversationSessionProjection(db, AGENT)
        assert await fresh._any_live_row(), (
            "history was emptied, so the sibling test's check would catch this "
            "and nothing here is about the stamp"
        )
        assert await fresh.get(UUID_B) is None, (
            "the repair republished a row naming the session that leaked during "
            "the EPHEMERAL stint"
        )


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_a_purge_and_refill_does_not_let_a_stale_snapshot_publish(tmp_path):
    """The stamp can come back to where it was, so it is not enough on its own.

    A repair derives at stamp N. The sweep empties history and erases the
    ledger, which restarts at 1. N new turns then arrive, and the ledger reads N
    again — the value the repair is holding. History is not empty, so the
    emptiness check passes; the stamp matches, so that check passes; and the
    pre-purge snapshot is published under the OLD target, leaving every newer
    row unaccounted while ``is_stale()`` answers false.

    This is the same collision that made the sweep erase the watermark
    alongside the ledger: a counter that can restart says nothing across the
    restart. The frontier is asked as well, because it does not restart —
    ``MAX(id)`` after a purge and refill is not the ``MAX(id)`` the snapshot was
    taken at.
    """
    from kestrel_sovereign.privacy import PrivacyMode
    from kestrel_sovereign.storage import AsyncStorage
    from kestrel_sovereign.storage.privacy_wrapper import PrivacyEnforcingStorage
    import kestrel_sovereign.storage.conversation_sessions as module

    async with AsyncStorage(str(tmp_path / "refill.db"), agent_id=AGENT) as storage:
        db = storage.db
        await _seed(db, [
            {"content": "pre-purge turn", "role": "user", "created_at": _at(0),
             "metadata": {"session_id": UUID_A}},
            {"content": "unstamped", "role": "assistant", "created_at": _at(1),
             "metadata": {}},
        ])
        projection = ConversationSessionProjection(db, AGENT)
        stamp_at_derivation = await projection.observed_changes()

        derived = asyncio.Event()
        release = asyncio.Event()
        real_fetchall = db.fetchall
        transcript_sql = module._live_rows_through(db.backend_type)

        async def _pausing_fetchall(query, params=()):
            rows = await real_fetchall(query, params)
            if query == transcript_sql and not derived.is_set():
                derived.set()
                await asyncio.wait_for(release.wait(), 30)
            return rows

        db.fetchall = _pausing_fetchall
        slow = asyncio.create_task(projection.repair())
        try:
            await asyncio.wait_for(derived.wait(), 30)
            wrapper = PrivacyEnforcingStorage(storage, PrivacyMode.EPHEMERAL)
            await storage.conversation.purge_all_since("1970-01-01", reason="test")
            await wrapper.purge_ephemeral_session(reason="test")
            assert await projection.observed_changes() == 0, "the ledger was not erased"
            # Refill until the restarted ledger reads exactly what the paused
            # repair is holding.
            while await projection.observed_changes() < stamp_at_derivation:
                await _seed(db, [{
                    "content": "post-purge turn", "role": "user",
                    "created_at": _at(500), "metadata": {"session_id": UUID_C},
                }])
            assert await projection.observed_changes() == stamp_at_derivation, (
                "the collision this guards against did not occur"
            )
        finally:
            release.set()
        await asyncio.wait_for(asyncio.shield(slow), 30)
        db.fetchall = real_fetchall

        fresh = ConversationSessionProjection(db, AGENT)
        assert await fresh.get(UUID_A) is None or await fresh.is_stale(), (
            "a snapshot taken before the purge was published as current over "
            "history that replaced it"
        )


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_a_timestamp_sql_cannot_read_is_not_read_by_the_fold_either(tmp_path):
    """The ordering key and the fold's guard must accept the same values.

    ``canonical_order()`` orders SQLite by ``julianday(created_at)``, and the
    fold's monotonicity guard parses the same column with
    ``coerce_session_timestamp``. Those are two different domains: basic ISO
    (``20260101T110000``) parses in Python and returns NULL from ``julianday``.

    So the canonical read sorts that row FIRST (NULLs lead), while the guard
    parses both, sees 10:00 then 11:00 rising with id, and folds happily —
    producing different boundaries and a different preview from the transcript
    the reader would see, under a watermark that says current.

    Only an import writes a stamp in that form; every writer here uses
    ``isoformat()`` or SQLite's ``datetime('now')``. That is exactly why it
    needs a case: it is a domain mismatch no ordinary write can reach.
    """
    db = await AsyncDatabase.sqlite(str(tmp_path / "domains.db"))
    try:
        await _seed(db, [
            {"content": "ordinary 10:00", "role": "user",
             "created_at": "2026-01-01T10:00:00",
             "metadata": {"session_id": UUID_A}},
            {"content": "basic ISO 11:00", "role": "user",
             "created_at": "20260101T110000",
             "metadata": {"session_id": UUID_A}},
        ])
        assert await db.fetchval(
            "SELECT julianday(created_at) FROM conversation_history "
            "WHERE content = ?", ("basic ISO 11:00",)
        ) is None, "julianday now reads this form, so the domains no longer differ"

        projection = ConversationSessionProjection(db, AGENT)
        await projection.repair()

        await _assert_agrees_with_the_grouper(db, projection)
    finally:
        await db.close()


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_a_negative_row_id_fallback_is_not_projected(tmp_path):
    """A session keyed by a row id must never be stored, sign included.

    An unstamped row has no ``session_id``, so the grouper keys its cluster by
    ``str(id)``. Such a key belongs to no session the column can hold, and the
    projection is meant to stay silent about it — Phase A's invariant, enforced
    by ``is_stampable_session_id`` rejecting all-digit keys.

    That rejection is a test on the key's SHAPE, and shape is a proxy for the
    thing that matters: whether the key came from the ``session_id`` column at
    all. The proxy breaks the moment an id is negative — ``"-1"`` contains a
    hyphen, so it is not all digits and passes — and #3001 made negative ids a
    supported shape. The projection would then list a session that
    ``coerce_persistent_message_id`` refuses, so opening or deleting it fails.

    Asked of the column, not of the key.
    """
    db = await AsyncDatabase.sqlite(str(tmp_path / "negative.db"))
    try:
        await db.execute(
            "INSERT INTO conversation_history "
            "(id, agent_id, role, content, metadata, session_id, created_at) "
            "VALUES (?, ?, 'user', 'imported, unstamped', NULL, NULL, ?)",
            (-7, AGENT, _at(0)),
        )
        projection = ConversationSessionProjection(db, AGENT)
        await projection.repair()

        assert await projection.get("-7") is None, (
            "a session keyed by a negative row id was projected; nothing can "
            "resolve it, so the list would show a conversation that cannot be "
            "opened or deleted"
        )
        assert await projection.list() == [], (
            f"projected {await projection.list()}"
        )
        assert not await projection.is_stale(), (
            "staying silent about a session it cannot key is the contract; "
            "reporting stale forever is not"
        )
    finally:
        await db.close()


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_a_replaced_history_with_reused_ids_is_not_reported_current(tmp_path):
    """The counters can return to any earlier value; the generation cannot.

    A full purge erases the ledger, and an import that replaces history with the
    SAME ids and the same number of row events brings back both the stamp and
    ``MAX(id)``. Every numeric witness agrees, and a snapshot taken before the
    purge would publish as current over history it does not describe.

    That is the third mechanism this class has broken — the sweep, the
    publication fence, and the fence's own revalidation — so it is closed at the
    source: the ledger row carries a value set once when it is created, and a
    stamp is comparable only to the incarnation it was read from.
    """
    db = await AsyncDatabase.sqlite(str(tmp_path / "reused.db"))
    try:
        await _seed(db, [
            {"content": "original", "role": "user", "created_at": _at(0),
             "metadata": {"session_id": UUID_A}},
        ])
        projection = ConversationSessionProjection(db, AGENT)
        await projection.repair()
        assert not await projection.is_stale()
        first = await projection.accounted()
        assert first.generation, "the ledger did not record a generation"

        row_id = int(await db.fetchval(
            "SELECT MAX(id) FROM conversation_history WHERE agent_id = ?", (AGENT,)
        ))
        stamp, appends = first.stamp, first.appends

        # Erase history and the ledger exactly as the sweep does, then put back
        # the same number of rows under the same ids.
        await db.execute("DELETE FROM conversation_history WHERE agent_id = ?", (AGENT,))
        await db.execute(
            "DELETE FROM conversation_history_changes WHERE agent_id = ?", (AGENT,)
        )
        await db.execute(
            "INSERT INTO conversation_history "
            "(id, agent_id, role, content, metadata, session_id, created_at) "
            "VALUES (?, ?, 'user', 'replacement', ?, ?, ?)",
            (row_id, AGENT, json.dumps({"session_id": UUID_C}), UUID_C, _at(0)),
        )
        while await projection.observed_changes() < stamp:
            await _seed(db, [{"content": "filler", "role": "user",
                              "created_at": _at(0),
                              "metadata": {"session_id": UUID_C}}])

        assert await projection.observed_changes() == stamp, "stamp did not return"
        assert int(await db.fetchval(
            "SELECT MAX(id) FROM conversation_history WHERE agent_id = ?", (AGENT,)
        )) == row_id, "MAX(id) did not return, so the numeric witnesses differ"

        assert await projection.is_stale(), (
            "every number matched the pre-purge world, so the projection "
            "reported itself current over history that replaced it"
        )
    finally:
        await db.close()
