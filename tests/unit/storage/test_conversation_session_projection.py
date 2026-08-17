"""#2959: the ``conversation_sessions`` projection agrees with the grouper.

The projection records the decision the write path already made — which session
a row was filed under — so a reader need not re-derive it. Two things can go
wrong with a record like that, and this file is built around both.

**It can disagree with the algorithm it replaces.** The differential test
below runs :func:`group_messages_into_sessions` +
:func:`coalesce_sessions_by_session_id` over a whole transcript and requires
the projection to have reached the same place, field by field. That is not a
tautology even though the projection also calls those functions: the projection
groups ONE session's rows, the reference groups the WHOLE transcript — where
clusters split on gaps, absorb their neighbours and are re-merged by id. The
claim under test is that those two arrive at the same answer.

**It can outlive what it describes.** ``first_user_message_id`` names a row,
and a name is only true while the row is live. Every mutation path gets its own
case here rather than one shared "mutations work" test, because they are
different code with different failure modes: a soft-delete stamps, a purge
DELETEs, an archive uses a third column, and a metadata merge moves a row
between two sessions at once. The invariant is asserted the same way for all of
them — the pointer is live, and the counts equal the live rows — by
:func:`_assert_projection_is_true`.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import pytest

from kestrel_sovereign.storage.async_conversation_store import AsyncConversationStore
from kestrel_sovereign.storage.async_database import (
    _SESSION_PROJECTION_BACKFILL,
    AsyncDatabase,
)
from kestrel_sovereign.storage.conversation_sessions import (
    ConversationSessionProjection,
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
OUTSIDE_THE_CONTRACT = "did:x:1"

BASE = datetime(2026, 3, 1, 9, 0, 0, tzinfo=timezone.utc)


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
    by_content = {}
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
    db: AsyncDatabase, store: AsyncConversationStore, agent_id: str = AGENT
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
    stored = {row["session_id"]: _stored(row) for row in await store.session_projection.list()}

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


async def _assert_projection_is_true(
    db: AsyncDatabase, store: AsyncConversationStore, agent_id: str = AGENT
) -> None:
    """The pointer invariant, asked of the database rather than of the code.

    Every mutation case ends here. A pointer at a soft-deleted, archived or
    purged row is the failure this projection exists not to have, so it is
    checked against ``conversation_history`` itself — a claim about a row is
    only worth what the row says.
    """
    for row in await store.session_projection.list():
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
    await _assert_agrees_with_the_grouper(db, store, agent_id)


# ── the corpus ───────────────────────────────────────────────────────────
#
# Every shape the ticket names, in one transcript, because the shapes interact:
# a legacy cluster sits next to a modern one, a resumed session straddles
# another, and the deleted rows are interleaved rather than trailing.

CORPUS: List[Dict[str, Any]] = [
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


@pytest.fixture
async def seeded(tmp_path):
    """A database holding the corpus, with the projection already built."""
    db = await AsyncDatabase.sqlite(str(tmp_path / "projection.db"))
    try:
        await _seed(db, CORPUS)
        store = AsyncConversationStore(db, agent_id=AGENT)
        await store.session_projection.rebuild()
        yield db, store
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_the_projection_says_what_the_grouper_says(seeded):
    """The acceptance gate for this phase, over every shape the ticket names.

    Spot-checks accompany the differential rather than replace it: the
    differential proves agreement, and these prove the corpus really contains
    the shapes it claims to — a corpus that had silently stopped exercising
    wake-only sessions would agree perfectly and mean nothing.
    """
    db, store = seeded
    stored = await _assert_agrees_with_the_grouper(db, store)

    ids = {
        row[0]: row[1]
        for row in await db.fetchall(
            "SELECT content, id FROM conversation_history WHERE agent_id = ?",
            (AGENT,),
        )
    }

    # A: the operator-signal notice and the wake are counted as user messages
    # but neither is the pointer — the #2947 skip, applied once, here.
    assert stored[UUID_A]["user_message_count"] == 4
    assert stored[UUID_A]["first_user_message_id"] == ids["A human turn"]
    assert stored[UUID_A]["wake_source"] == "heartbeat"
    # ...and the resumption past the gap is inside the same row, not a second.
    assert stored[UUID_A]["last_message_at"] == coerce_session_timestamp(_at(500))
    assert stored[UUID_A]["message_count"] == 5

    # B: a marker is structural. It sets the start but is not a message.
    assert stored[UUID_B]["message_count"] == 1
    assert stored[UUID_B]["started_at"] == coerce_session_timestamp(_at(184))
    assert stored[UUID_B]["last_message_at"] == coerce_session_timestamp(_at(185))

    # A wake-only session has no human turn to point at, and says why.
    assert stored[UUID_WAKE]["first_user_message_id"] is None
    assert stored[UUID_WAKE]["wake_source"] == "talon.job_complete"

    # C: the pointer skipped the deleted row AND the archived one.
    assert stored[UUID_C]["first_user_message_id"] == ids["C live turn"]
    assert stored[UUID_C]["message_count"] == 2

    # The two shapes the column may not hold are absent, not guessed at.
    assert OUTSIDE_THE_CONTRACT not in stored
    assert not [key for key in stored if str(key).isdigit()]


@pytest.mark.asyncio
async def test_a_session_id_the_column_cannot_hold_is_never_claimed(seeded):
    """Absence is the invariant, so it gets its own case.

    The grouper reports both the legacy row-id cluster and the ``did:x:1``
    session; the Phase A column contract admits neither. A projection that
    started claiming them would still "agree" with the grouper on every field
    — what it would break is the rule that this table is silent where it
    cannot be sure.

    What enforces that is upstream and worth naming, because it is not a
    filter in this ticket's code: membership is read from
    ``conversation_history.session_id``, and an id outside the contract is on
    no row's column, so no query can find rows for it. Asked point-blank for
    one, the projection still has nothing to write. This is an end-to-end
    property rather than a guard, and it would fail the day membership started
    being resolved from metadata again.
    """
    db, store = seeded
    history = await _live_history(db)
    reference = _reference_sessions(history)

    unclaimable = [
        session_id
        for session_id in reference
        if not is_stampable_session_id(session_id)
    ]
    assert OUTSIDE_THE_CONTRACT in unclaimable
    assert any(str(session_id).isdigit() for session_id in unclaimable)

    stored = {row["session_id"] for row in await store.session_projection.list()}
    assert stored.isdisjoint(unclaimable)

    # Asked for them directly — the shape a future caller holding metadata
    # rather than the column would produce.
    await store.refresh_session_projection(unclaimable)
    stored_after = {row["session_id"] for row in await store.session_projection.list()}
    assert stored_after == stored


@pytest.mark.asyncio
async def test_the_pointer_is_the_turn_the_picker_chose_even_with_no_text(tmp_path):
    """The pointer follows the picker's CHOICE, not the presence of text.

    A user turn with an empty body is still the turn the preview picker
    settles on — it assigns that text and is thereafter done, so a later turn
    with words in it does not displace it (the list then shows a blank title,
    which is session grouping's behaviour and not this table's to change).

    The projection hands the picker row ids in place of text, and this is the
    case that catches that substitution acquiring an opinion of its own:
    "skip the empty one" would look like an improvement and would be the
    projection naming a different row than the transcript previews.
    """
    db = await AsyncDatabase.sqlite(str(tmp_path / "empty.db"))
    try:
        await _seed(db, [
            {"content": "", "role": "user", "created_at": _at(0),
             "metadata": {"session_id": UUID_A}},
            {"content": "the turn with words in it", "role": "user",
             "created_at": _at(1), "metadata": {"session_id": UUID_A}},
        ])
        store = AsyncConversationStore(db, agent_id=AGENT)
        await store.session_projection.rebuild()

        blank = await db.fetchval(
            "SELECT MIN(id) FROM conversation_history WHERE agent_id = ?", (AGENT,)
        )
        assert (await store.session_projection.get(UUID_A))[
            "first_user_message_id"
        ] == blank
        # ...which is the row the grouper previewed, asked independently.
        grouped = coalesce_sessions_by_session_id(
            group_messages_into_sessions(await _live_history(db))
        )
        assert grouped[0]["preview_content"] == ""
        await _assert_projection_is_true(db, store)
    finally:
        await db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("legacy_first", [True, False])
async def test_a_legacy_row_beside_a_session_is_where_the_two_answers_part(
    tmp_path, legacy_first
):
    """The one shape where the projection and the display grouper differ.

    A row carrying no session id at all is filed under nothing, so the grouper
    attributes it by proximity — and within the gap window that pulls a
    neighbouring session around:

    * ``legacy_first``: the unlabeled row anchors a cluster keyed by its own
      ROW ID, and the following labeled row is absorbed into it, so the grouper
      reports no session under that id at all.
    * otherwise: the unlabeled row is absorbed INTO the labeled session, whose
      grouped count is then one higher than the number of rows filed under it.

    The projection sides with membership, not with proximity, and this asserts
    that it is the side with the consequences: ``_get_complete_session_message_ids``
    — the resolver behind delete, archive, restore, purge and
    ``count_session_messages`` — resolves exactly the rows the projection
    counts, in both directions. The grouper's own comment says as much about
    the legacy fallback: it merges there deliberately, to match what deleting
    the *row-id* session touches.

    So this is a difference, it is named, and it is confined to rows the write
    path has not produced since #2012 stamped every turn. Which of the two
    answers a list should show is a read-path question, and the read path is
    Phase C (#2960).
    """
    db = await AsyncDatabase.sqlite(str(tmp_path / f"legacy-{legacy_first}.db"))
    try:
        legacy = {"content": "legacy", "role": "user", "created_at": _at(0),
                  "metadata": {}}
        labeled = {"content": "labeled", "role": "user", "created_at": _at(1),
                   "metadata": {"session_id": UUID_A}}
        await _seed(db, [legacy, labeled] if legacy_first else [labeled, legacy])
        store = AsyncConversationStore(db, agent_id=AGENT)
        await store.session_projection.rebuild()

        reference = _reference_sessions(await _live_history(db))
        stored = {row["session_id"]: _stored(row)
                  for row in await store.session_projection.list()}

        # Whatever the grouper did, the projection counts the one row that is
        # filed under the id...
        assert stored[UUID_A]["message_count"] == 1
        # ...and that is exactly what every lifecycle operation would touch.
        resolved = await store._get_complete_session_message_ids(
            UUID_A, deleted_filter="live"
        )
        assert len(resolved) == stored[UUID_A]["message_count"]
        assert await store.count_session_messages(UUID_A, deleted_filter="live") == 1

        if legacy_first:
            # The grouper reports one cluster, keyed by the legacy row's id,
            # holding both rows — no session under UUID_A at all.
            assert UUID_A not in reference
            assert [session["message_count"] for session in reference.values()] == [2]
        else:
            # The grouper reports UUID_A holding both rows; the projection
            # holds the one filed under it.
            assert reference[UUID_A]["message_count"] == 2
    finally:
        await db.close()


# ── one case per mutation path ───────────────────────────────────────────


@pytest.mark.asyncio
async def test_soft_deleting_the_pointer_advances_it_and_restoring_brings_it_back(
    seeded,
):
    """Delete and restore, on the single-message door.

    The pointer is the row most likely to be deleted — it is the first thing
    in the conversation — so this is the case the invariant was written for.
    Both directions are asserted: an advance that never came back would leave
    the projection permanently disagreeing with a transcript that has the turn.
    """
    db, store = seeded
    pointer = (await store.session_projection.get(UUID_A))["first_user_message_id"]
    successor = await db.fetchval(
        "SELECT id FROM conversation_history WHERE agent_id = ? AND session_id = ? "
        "AND role = 'user' AND id > ? ORDER BY id ASC LIMIT 1",
        (AGENT, UUID_A, pointer),
    )

    assert await store.delete_message(pointer)
    moved = await store.session_projection.get(UUID_A)
    assert moved["first_user_message_id"] == successor
    assert moved["message_count"] == 4
    await _assert_projection_is_true(db, store)

    assert await store.restore_message(pointer)
    restored = await store.session_projection.get(UUID_A)
    assert restored["first_user_message_id"] == pointer
    assert restored["message_count"] == 5
    await _assert_projection_is_true(db, store)


@pytest.mark.asyncio
async def test_deleting_a_whole_session_removes_its_row_and_restoring_rebuilds_it(
    seeded,
):
    """The session-scoped door, which resolves membership before it stamps.

    A session with no live rows has nothing to describe, so its row goes —
    rather than lingering as a count of zero pointing at a trashed turn.
    """
    db, store = seeded
    assert await store.delete_conversation_session(UUID_WAKE) == 2
    assert await store.session_projection.get(UUID_WAKE) is None
    await _assert_projection_is_true(db, store)

    assert await store.restore_conversation_session(UUID_WAKE) == 2
    restored = await store.session_projection.get(UUID_WAKE)
    assert restored is not None
    assert restored["message_count"] == 2
    assert restored["wake_source"] == "talon.job_complete"
    await _assert_projection_is_true(db, store)


@pytest.mark.asyncio
async def test_archiving_takes_a_session_out_of_the_projection_and_unarchiving_returns_it(
    seeded,
):
    """Archive is a third column, and the pointer has to respect it too.

    An archived row is not deleted — it is intact and restorable — but it is
    out of the live set, which is what the projection describes. The bug this
    guards is the easy one: a maintenance path that learned ``deleted_at`` and
    never learned ``archived_at``.
    """
    db, store = seeded
    assert await store.archive_conversation_session(UUID_B) == 2
    assert await store.session_projection.get(UUID_B) is None
    await _assert_projection_is_true(db, store)

    assert await store.unarchive_conversation_session(UUID_B) == 2
    assert (await store.session_projection.get(UUID_B))["message_count"] == 1
    await _assert_projection_is_true(db, store)


@pytest.mark.asyncio
async def test_purging_a_session_leaves_no_row_pointing_at_a_destroyed_message(seeded):
    """Purge is the one that cannot be repaired afterwards.

    A soft-delete leaves the row to reconcile against; a purge does not. So the
    projection is recomputed inside the destructive transaction, and this
    asserts the observable that follows from it: nothing in the table names a
    message the database no longer has.
    """
    db, store = seeded
    assert await store.purge_conversation_session(UUID_C) > 0
    assert await store.session_projection.get(UUID_C) is None

    for row in await store.session_projection.list():
        pointer = row["first_user_message_id"]
        if pointer is not None:
            assert await db.fetchval(
                "SELECT COUNT(*) FROM conversation_history WHERE id = ?",
                (pointer,),
            ) == 1
    await _assert_projection_is_true(db, store)


@pytest.mark.asyncio
async def test_purging_one_message_moves_the_pointer_off_it(seeded):
    """The single-row purge, which resolves nothing and audits by id."""
    db, store = seeded
    pointer = (await store.session_projection.get(UUID_A))["first_user_message_id"]

    assert await store.purge_message(pointer)
    moved = await store.session_projection.get(UUID_A)
    assert moved["first_user_message_id"] != pointer
    assert await db.fetchval(
        "SELECT COUNT(*) FROM conversation_history WHERE id = ?", (pointer,)
    ) == 0
    await _assert_projection_is_true(db, store)


@pytest.mark.asyncio
async def test_ageing_out_trash_leaves_the_live_projection_exactly_as_it_was(seeded):
    """The janitor's target set is disjoint from what the projection describes.

    ``purge_trash_older_than`` destroys by ``deleted_at`` across every session
    at once, with no session named by the caller — which makes it look like the
    path most able to strand a pointer. It is not, and the reason is worth
    holding to account rather than assuming: its predicate is
    ``deleted_at IS NOT NULL``, so every row it can reach had already left the
    live set, and the projection never mentioned it.

    So this asserts the projection is UNCHANGED, not merely still true. The
    mutation it exists to catch is not a missing refresh — a refresh here
    recomputes the same values — it is the janitor's scope widening to reach a
    live row, which this would report as a projection that moved.
    """
    db, store = seeded
    pointer = (await store.session_projection.get(UUID_A))["first_user_message_id"]
    assert await store.delete_message(pointer)
    before = await store.session_projection.list()
    assert before

    purged = await store.purge_trash_older_than(
        (datetime.now(timezone.utc) + timedelta(days=1)).replace(tzinfo=None).isoformat()
    )
    assert purged > 0
    assert await db.fetchval(
        "SELECT COUNT(*) FROM conversation_history WHERE id = ?", (pointer,)
    ) == 0
    assert await store.session_projection.list() == before
    await _assert_projection_is_true(db, store)


@pytest.mark.asyncio
async def test_an_ephemeral_leak_purge_cannot_strand_a_pointer(seeded):
    """``purge_all_since`` destroys by timestamp, again with no session named."""
    db, store = seeded
    assert await store.purge_all_since(_at(500)) > 0
    await _assert_projection_is_true(db, store)
    assert await store.session_projection.get(UUID_C) is None
    # A survives — with only the rows written before the cutoff.
    assert (await store.session_projection.get(UUID_A))["message_count"] == 4


@pytest.mark.asyncio
async def test_clearing_the_history_leaves_no_sessions_behind(seeded):
    """Nothing is live, so nothing may be claimed."""
    db, store = seeded
    await store.clear_history()
    assert await store.session_projection.list() == []
    await _assert_projection_is_true(db, store)


@pytest.mark.asyncio
async def test_moving_a_row_between_sessions_updates_both_ends(seeded):
    """The metadata door, which changes two sessions in one statement.

    ``update_message_metadata`` is the only way a row's session changes after
    insertion, and a maintenance pass that refreshed only the destination would
    leave the source counting a row it no longer has. Both ends are asserted,
    and then the move outside the column's contract — where the destination
    cannot be claimed at all and the source must still be corrected.
    """
    db, store = seeded
    pointer = (await store.session_projection.get(UUID_A))["first_user_message_id"]

    assert await store.update_message_metadata(pointer, {"session_id": UUID_B})
    assert (await store.session_projection.get(UUID_A))["message_count"] == 4
    assert (await store.session_projection.get(UUID_A))["first_user_message_id"] != pointer
    moved_into = await store.session_projection.get(UUID_B)
    assert moved_into["message_count"] == 2
    await _assert_projection_is_true(db, store)

    # Out of the contract entirely: B loses the row, and nothing claims it.
    assert await store.update_message_metadata(
        pointer, {"session_id": OUTSIDE_THE_CONTRACT}
    )
    assert (await store.session_projection.get(UUID_B))["message_count"] == 1
    assert await store.session_projection.get(OUTSIDE_THE_CONTRACT) is None
    await _assert_projection_is_true(db, store)


@pytest.mark.asyncio
async def test_an_ordinary_turn_extends_its_session(tmp_path):
    """The loud path: ``add_conversation`` maintains what it writes.

    Driven through the public API with no seeding, so it also covers the
    session the store mints for itself — the write-time decision this whole
    table exists to record.
    """
    db = await AsyncDatabase.sqlite(str(tmp_path / "add.db"))
    try:
        store = AsyncConversationStore(db, agent_id=AGENT)
        await store.add_conversation("user", "first", session_id=UUID_A)
        await store.add_conversation("assistant", "reply", session_id=UUID_A)

        row = await store.session_projection.get(UUID_A)
        assert row["message_count"] == 2
        assert row["user_message_count"] == 1
        assert row["first_user_message_id"] == await db.fetchval(
            "SELECT MIN(id) FROM conversation_history WHERE agent_id = ?", (AGENT,)
        )
        await _assert_projection_is_true(db, store)

        # An id the column may not hold stays unclaimed on the write path too.
        await store.add_conversation("user", "elsewhere", session_id=OUTSIDE_THE_CONTRACT)
        assert await store.session_projection.get(OUTSIDE_THE_CONTRACT) is None
        await _assert_projection_is_true(db, store)
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_a_salvage_marker_is_counted_in_the_session_it_salvaged(tmp_path):
    """A writer outside the store still keeps the projection true.

    The salvage marker is hand-written SQL inside its own transaction, so it
    picks up nothing for free — exactly the kind of quiet writer that leaves a
    projection one row short with no visible symptom.
    """
    from kestrel_sovereign.agent.salvage import SalvageReason, salvage_messages

    db = await AsyncDatabase.sqlite(str(tmp_path / "salvage.db"))
    try:
        store = AsyncConversationStore(db, agent_id=AGENT)
        await store.add_conversation("user", "long turn", session_id=UUID_A)
        original = int(await db.fetchval(
            "SELECT MIN(id) FROM conversation_history WHERE agent_id = ?", (AGENT,)
        ))

        await salvage_messages(
            conv_store=store,
            original_messages=[{"id": original}],
            reason=SalvageReason.AUTO_PRUNE_PRETRIM,
            model="test-model",
            session_id=UUID_A,
            token_estimate=42,
        )

        assert (await store.session_projection.get(UUID_A))["message_count"] == 2
        await _assert_projection_is_true(db, store)
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_nightly_consolidation_archives_without_stranding_the_pointer(tmp_path):
    """The forgetting curve is a live-set mutation, so it maintains this table.

    ``MemoryConsolidator._archive_decayed`` is the nightly sweep that stamps
    ``archived_at`` on faded turns. It writes ``conversation_history`` with its
    own SQL — it does not go through the store's archive methods — so it picks up
    nothing for free, and archiving a session's FIRST user turn is exactly the
    shape that leaves a pointer naming a row no reader can see.

    Both of its archival branches are exercised: the decay branch (a turn old and
    unimportant enough to fade) and the legacy-``metadata.archived``
    canonicalization beside it, which is the one most likely to be forgotten
    because it looks like a migration rather than a mutation.

    ``_archive_decayed`` is driven directly rather than through
    ``run_consolidation``: the phases around it want an LLM service and a graph
    store, and archival is the mutation under test.
    """
    from kestrel_sovereign.storage.memory_consolidator import MemoryConsolidator

    def ago(**delta) -> str:
        """A stored timestamp relative to now, so the decay maths cannot rot."""
        return (
            (datetime.now(timezone.utc) - timedelta(**delta))
            .replace(tzinfo=None)
            .isoformat()
        )

    db = await AsyncDatabase.sqlite(str(tmp_path / "decay.db"))
    try:
        # 800 days at importance 0.0 is a half-life of 30 days — strength ~1e-8,
        # far below DECAY_ARCHIVE_THRESHOLD. The survivor is minutes old.
        await _seed(db, [
            {"content": "the faded first turn", "role": "user",
             "created_at": ago(days=800),
             "metadata": {"session_id": UUID_A, "importance": 0.0}},
            {"content": "the turn that survives", "role": "user",
             "created_at": ago(minutes=5),
             "metadata": {"session_id": UUID_A, "importance": 1.0}},
            {"content": "already archived the legacy way", "role": "user",
             "created_at": ago(minutes=4),
             "metadata": {"session_id": UUID_A, "importance": 1.0,
                          "archived": True}},
        ])
        store = AsyncConversationStore(db, agent_id=AGENT)
        await store.session_projection.rebuild()
        ids = {
            row[0]: row[1]
            for row in await db.fetchall(
                "SELECT content, id FROM conversation_history WHERE agent_id = ?",
                (AGENT,),
            )
        }
        before = await store.session_projection.get(UUID_A)
        assert before["message_count"] == 3
        assert before["first_user_message_id"] == ids["the faded first turn"]

        consolidator = MemoryConsolidator(db=db, agent_id=AGENT)
        assert await consolidator._archive_decayed() == 2

        after = await store.session_projection.get(UUID_A)
        assert after["first_user_message_id"] == ids["the turn that survives"]
        assert after["message_count"] == 1
        await _assert_projection_is_true(db, store)
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_sessions_are_refreshed_in_one_deterministic_order(seeded, monkeypatch):
    """The acquisition ORDER is the deadlock guard, so it is asserted directly.

    A refresh routinely names several sessions — a session-scoped delete resolves
    a legacy row-id session together with the UUID sessions inside its window —
    and each takes a lock held until the enclosing mutation commits. Two callers
    naming an overlapping set in opposite orders is then a cycle: on PostgreSQL a
    detected deadlock (one transaction aborted mid-mutation), which is not a
    failure any test can provoke on demand.

    So the guard is asserted where it is deterministic: whatever order a caller
    hands them in, the sessions are locked in one global order.
    """
    db, store = seeded
    original = ConversationSessionProjection._serialized
    order: List[str] = []

    def recording(self, session_id):
        order.append(session_id)
        return original(self, session_id)

    monkeypatch.setattr(ConversationSessionProjection, "_serialized", recording)
    await store.refresh_session_projection([UUID_C, UUID_A, UUID_WAKE, UUID_B, UUID_A])

    assert order == sorted({UUID_A, UUID_B, UUID_C, UUID_WAKE})
    await _assert_projection_is_true(db, store)


# ── the backfill ─────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_history_that_predates_the_table_is_projected_at_boot(tmp_path):
    """The backfill, driven the way a real upgrade drives it.

    The rows are written by a first ``AsyncDatabase.sqlite`` call, then the
    marker and the table are dropped so the next open is indistinguishable from
    an upgrade. What comes back must be the same table the live write paths
    would have maintained — which is checked against the grouper, not against a
    hand-written expectation.
    """
    path = str(tmp_path / "upgrade.db")
    db = await AsyncDatabase.sqlite(path)
    try:
        await _seed(db, CORPUS)
        await db.execute("DELETE FROM conversation_sessions", ())
        await db.execute(
            "DELETE FROM schema_backfills WHERE name = ?",
            (_SESSION_PROJECTION_BACKFILL,),
        )
    finally:
        await db.close()

    db = await AsyncDatabase.sqlite(path)
    try:
        store = AsyncConversationStore(db, agent_id=AGENT)
        stored = await _assert_agrees_with_the_grouper(db, store)
        assert stored, "the backfill projected nothing at all"
        assert await db.fetchval(
            "SELECT COUNT(*) FROM schema_backfills WHERE name = ?",
            (_SESSION_PROJECTION_BACKFILL,),
        ) == 1
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_the_backfill_sees_the_session_ids_the_2012_relink_leaves(tmp_path):
    """Ordering inside ``_init_schema``, pinned by its consequence.

    The #2012 migration RELINKS a message whose ``session_id`` was stored as
    the bare row-id of its ``new_session`` marker onto that marker's canonical
    UUID — rewriting both the metadata and the column. The projection reads
    that column, so building it first would file this history under a key the
    relink then replaces, and the completion marker would stop the projection
    from ever reconsidering. Absent-for-good, on the exact rows the relink
    exists to rescue.

    So the assertion is about the finished table on the boot that migrates: the
    relinked turn is counted under the UUID and is the pointer. Move the
    backfill back above the relink and this fails.
    """
    import sqlite3

    path = str(tmp_path / "relink.db")
    connection = sqlite3.connect(path)
    try:
        connection.executescript(
            "CREATE TABLE conversation_history ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, "
            "agent_id TEXT NOT NULL DEFAULT '', role TEXT NOT NULL, "
            "content TEXT NOT NULL, metadata TEXT, "
            "created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, "
            "deleted_at TIMESTAMP DEFAULT NULL);"
        )
        # The marker owns the UUID; the turn after it was filed under the
        # marker's ROW ID, which is the #2012 defect.
        connection.execute(
            "INSERT INTO conversation_history "
            "(id, agent_id, role, content, metadata, created_at) "
            "VALUES (1, ?, 'user', 'marker', ?, ?)",
            (AGENT, json.dumps({"session_id": UUID_A, "new_session": True}), _at(0)),
        )
        connection.execute(
            "INSERT INTO conversation_history "
            "(id, agent_id, role, content, metadata, created_at) "
            "VALUES (2, ?, 'user', 'the relinked turn', ?, ?)",
            (AGENT, json.dumps({"session_id": "1"}), _at(1)),
        )
        connection.commit()
    finally:
        connection.close()

    db = await AsyncDatabase.sqlite(path)
    try:
        # The relink really did run — otherwise this test would be asserting
        # the projection agrees with a rewrite that never happened.
        assert await db.fetchval(
            "SELECT session_id FROM conversation_history WHERE id = 2", ()
        ) == UUID_A

        store = AsyncConversationStore(db, agent_id=AGENT)
        projected = await store.session_projection.get(UUID_A)
        assert projected is not None, (
            "the relinked session has no projection row — the backfill ran "
            "before the #2012 relink and will never revisit it"
        )
        assert projected["message_count"] == 1
        assert projected["first_user_message_id"] == 2
        await _assert_projection_is_true(db, store)
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_a_relink_after_the_backfill_marker_still_moves_the_projection(tmp_path):
    """The relink runs on EVERY boot; the projection's backfill runs once.

    That asymmetry is the bug this pins. ``migrate_canonical_session_ids`` has no
    completion sentinel — it is idempotent by construction and re-examines the
    table on every ``_init_schema`` — so a numeric session key written after the
    upgrade (an old client echoing a row-id back) is relinked on some later
    restart, long after the projection's one-time backfill has recorded "done".
    Nothing then rebuilds the projection, so it keeps describing a membership the
    relink has already changed: not absent, which is recoverable, but WRONG.

    The pre-state is deliberately a projection row that EXISTS and says zero
    messages, rather than no row at all. Absent would be within the invariant;
    the failure is a row that stays at zero while the session has a turn.
    """
    path = str(tmp_path / "relink-after-marker.db")
    db = await AsyncDatabase.sqlite(path)
    try:
        # The marker owns the UUID and is the only stampable row...
        await _seed(db, [
            {"content": "marker", "role": "user", "created_at": _at(0),
             "metadata": {"session_id": UUID_A, "new_session": True}},
        ])
        marker_id = int(await db.fetchval(
            "SELECT MIN(id) FROM conversation_history WHERE agent_id = ?", (AGENT,)
        ))
        # ...and the turn after it is filed under the marker's ROW ID, which the
        # column contract declines, so it starts with session_id NULL.
        await _seed(db, [
            {"content": "the late relinked turn", "role": "user",
             "created_at": _at(1), "metadata": {"session_id": str(marker_id)}},
        ])
        store = AsyncConversationStore(db, agent_id=AGENT)
        await store.session_projection.rebuild()

        # The pre-state: the projection exists, and the relink has not happened.
        assert (await store.session_projection.get(UUID_A))["message_count"] == 0
        # ...and this database has already completed the one-time backfill, so
        # nothing here will be revisited by it.
        assert await db.fetchval(
            "SELECT COUNT(*) FROM schema_backfills WHERE name = ?",
            (_SESSION_PROJECTION_BACKFILL,),
        ) == 1
    finally:
        await db.close()

    db = await AsyncDatabase.sqlite(path)
    try:
        relinked = int(await db.fetchval(
            "SELECT id FROM conversation_history WHERE agent_id = ? "
            "AND content = ?", (AGENT, "the late relinked turn"),
        ))
        # The relink really ran on this boot — otherwise the assertions below
        # would be about a rewrite that never happened.
        assert await db.fetchval(
            "SELECT session_id FROM conversation_history WHERE id = ?", (relinked,)
        ) == UUID_A

        store = AsyncConversationStore(db, agent_id=AGENT)
        projected = await store.session_projection.get(UUID_A)
        assert projected["message_count"] == 1, (
            "the projection still describes the membership the relink replaced"
        )
        assert projected["first_user_message_id"] == relinked
        await _assert_projection_is_true(db, store)
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_the_backfill_pass_does_not_run_inside_the_migration_lock(tmp_path):
    """Its placement is the ticket's own instruction, so it is asserted.

    ``migration_lock`` is ``BEGIN IMMEDIATE`` for its whole block, and this
    backfill is a Python iteration pass that re-enters the database per
    session. Nesting the two is the ABBA the lock's docstring warns about, and
    the failure it produces is a boot that hangs rather than one that reports.

    Proved by holding SQLite's writer slot while the pass runs: if the pass
    were inside the lock it could not proceed, and this test would time out
    instead of passing. The timeout is what makes that a reportable failure
    rather than a wedged worker — see #2958's 23-hour hang.
    """
    import asyncio

    path = str(tmp_path / "lockfree.db")
    db = await AsyncDatabase.sqlite(path)
    try:
        await _seed(db, CORPUS)
        await db.execute("DELETE FROM conversation_sessions", ())
        await db.execute(
            "DELETE FROM schema_backfills WHERE name = ?",
            (_SESSION_PROJECTION_BACKFILL,),
        )

        projection = ConversationSessionProjection(db, AGENT)
        async with asyncio.timeout(30):
            assert await projection.rebuild() > 0

        # The marker is the part that IS serialized, so it is taken here and
        # not by the pass above.
        assert await db.fetchval(
            "SELECT COUNT(*) FROM schema_backfills WHERE name = ?",
            (_SESSION_PROJECTION_BACKFILL,),
        ) == 0
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_rebuilding_twice_changes_nothing_and_forgets_what_is_gone(seeded):
    """Idempotent, and self-correcting in both directions.

    The pass is allowed to run more than once — that is the price of keeping it
    outside the lock — so running it twice must reach the same table. And a
    stale row planted by hand must be dropped: the rebuild is also the repair,
    and a repair that only ever adds cannot fix the state that needs fixing.
    """
    db, store = seeded
    before = await store.session_projection.list()
    await store.session_projection.rebuild()
    assert await store.session_projection.list() == before

    await db.execute(
        "INSERT INTO conversation_sessions "
        "(agent_id, session_id, started_at, last_message_at, message_count, "
        "user_message_count, first_user_message_id, wake_source) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (AGENT, "9c2a1f77-0000-4000-8000-00000000dead", _at(0), _at(1), 3, 2,
         999999, None),
    )
    await store.session_projection.rebuild()
    assert await store.session_projection.list() == before
    await _assert_projection_is_true(db, store)


@pytest.mark.asyncio
async def test_one_agents_mutations_leave_another_agents_projection_alone(tmp_path):
    """The table is keyed by agent, and every statement here says so."""
    db = await AsyncDatabase.sqlite(str(tmp_path / "tenants.db"))
    other = "did:test:session-projection-other"
    try:
        await _seed(db, CORPUS)
        await _seed(db, CORPUS, agent_id=other)
        mine = AsyncConversationStore(db, agent_id=AGENT)
        theirs = AsyncConversationStore(db, agent_id=other)
        await mine.session_projection.rebuild()
        await theirs.session_projection.rebuild()
        expected = await theirs.session_projection.list()

        await mine.clear_history()
        assert await mine.session_projection.list() == []
        assert await theirs.session_projection.list() == expected
        await _assert_projection_is_true(db, theirs, agent_id=other)
    finally:
        await db.close()
