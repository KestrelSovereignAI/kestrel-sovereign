"""The ``conversation_sessions`` projection (#2959).

The 30-minute session rule is applied at **write** time:
:meth:`AsyncConversationStore._derive_implicit_session_id` reuses the previous
row's session when the gap is under the window and mints a UUID otherwise, then
stamps the answer into the row. Every read then throws that answer away and
re-derives it — :func:`group_messages_into_sessions` splits the transcript on
gap *or* id change, and :func:`coalesce_sessions_by_session_id` merges the
same-id clusters back together, arriving where the writer started.

This table records the writer's answer instead, so a reader can ask for it.

What a row claims
=================

One row per ``(agent_id, session_id)``, describing the **live** rows filed
under that id: the ones ``deleted_at IS NULL AND archived_at IS NULL``. That is
the same membership every lifecycle operation acts on — ``_get_session_messages``
and ``_get_complete_session_message_ids`` resolve an explicit id by metadata
match, so delete / archive / restore / purge / ``count_session_messages`` all
touch exactly the rows this row counts.

Membership, not proximity — and the two part company in exactly one place. A
row carrying no session id is filed under nothing, so the display grouper
attributes it to whichever cluster it falls next to: within the gap window an
unlabeled row is absorbed into the session before it, and an unlabeled row that
comes *first* anchors a cluster keyed by its own row id which then swallows the
labeled rows after it. This table follows the id, so its answer differs from
the grouper's for those neighbours — and it is the answer with consequences,
since it is the set delete and purge act on (session grouping's own comment
says it merges the legacy case deliberately, to match what deleting the
*row-id* session touches). The shape is confined to rows the write path has not
produced since #2012, it is asserted rather than left implicit in
``tests/unit/storage/test_conversation_session_projection.py``, and which
answer a LIST should show is a read-path question that belongs to Phase C
(#2960).

``first_user_message_id`` is a **pointer**, never a copy. ``content`` is KSAv2
ciphertext, so a preview column here would be a plaintext copy of encrypted
text sitting beside the ciphertext, and a record that outlives what it
describes. See #2948. The pointer applies the #2947 skip — operator-signal
notices and autonomous signal wakes are not human turns — once, here, instead
of on every request.

The invariants
==============

1. **A claimed ``first_user_message_id`` is live.** A pointer at a deleted,
   archived or purged row is the "record that claims what nobody observed"
   class. The projection is therefore recomputed from the surviving rows in the
   same write unit as every mutation, rather than adjusted by deltas: a count
   that is *derived* cannot drift, and a pointer that is *derived* cannot
   outlive its row.
2. **The projection may be absent where the grouper finds a session; it may
   never disagree.** Absent is recoverable, wrong is not. It is absent for a
   session whose id is outside the Phase A column contract (see
   :mod:`kestrel_sovereign.storage.session_id_column`) and for one whose rows
   are all gone.

Recomputation is not enough on its own
======================================

"Read the survivors, write what they say" is only true of the rows the reader
could see, and two concurrent refreshes of one session can each be describing a
world the other has already changed. Both engines allow it, in different ways:

* **PostgreSQL** gives each transaction its own snapshot, so two transactions
  soft-deleting different rows of one session each observe the *other's* row as
  still live. Each writes a truthful-looking projection; the second one to
  commit leaves a pointer at a row that is deleted by the time both are done.
  This is the same MVCC hazard
  :meth:`~kestrel_sovereign.storage.lexical_memory_index.ConversationLexicalIndex.serialized_token_cleanup`
  exists for, and it is closed the same way.
* **SQLite** has one writer, but a refresh outside a transaction is two
  statements. A second turn can land between one refresh's read and its write,
  so the older read's count overwrites the newer one's — the projection then
  claims one message while two are live.

So every refresh takes a **transaction-scoped boundary on ``(agent_id,
session_id)``** and holds it across both the read and the write. The lock is
keyed on the id rather than taken on the projection row, because a session
whose row does not exist yet is exactly the case that needs serializing and
``SELECT ... FOR UPDATE`` cannot lock a row that is absent.

Sessions are locked in sorted order, so a refresh naming several of them (a
session-scoped delete resolves more than one) cannot form a cycle with a
concurrent refresh naming an overlapping set. Where a caller has already opened
a transaction — every destructive path does — the boundary is inherited by it
and released only when the mutation commits, which is what makes "the last
refresh sees every committed mutation" true rather than hopeful. Its place in
this repository's global lock order is between the history rows and the lexical
token keys, which is where the purge path already acquires it.

Correctness under PostgreSQL rests on ``READ COMMITTED`` — the configured
default — where the ``SELECT`` taken after the boundary is granted sees
everything committed by then. Under ``REPEATABLE READ`` a refresh would keep
its pre-lock snapshot and the serialization would buy nothing.

What it costs
=============

A refresh re-reads the session's live rows — id, role, metadata, timestamp; no
bodies — so a turn in a long-running conversation pays a scan of that
conversation. That is deliberate, and it is the price of invariant 1: an
incremental ``message_count = message_count + 1`` is a number nothing can
check, and a pointer maintained by "only move it if the deleted row was the
pointer" is a special case per mutation path. Recomputation has one rule and no
state to be wrong about. If profiling ever says otherwise, the answer is a
narrower recompute — not a delta.

Why the derivation calls the grouper
====================================

The fields here are the grouper's fields — ``started_at``,
``last_message_at``, the two counts, the previewed turn, the wake source — and
every one of them is a rule that could be restated slightly differently. Phase
A's Finding 4 was exactly that: one rule written twice, drifting. So this does
not restate them. It hands one session's rows to the very functions the
differential test compares against, and reads the answer off.

That is not circular, and the differential test is not tautological: the
projection groups **one session's rows**, while the test groups the **whole
transcript** — where clusters split on gaps, absorb unlabeled legacy rows, and
are re-merged by id. The claim under test is that the two arrive at the same
place, which is a claim about the algorithm and not about this function.

The one adaptation is ``first_user_message_id``. The grouper reports the
previewed *text*, not which row it came from, and this table stores the row.
Rather than restate the picker's skip rule to find that row again, each row is
handed its own id in place of its text, and what comes back as the "preview" is
then the id of the row the picker chose.

That substitution is faithful because the picker never *inspects* ``content``:
it assigns the first eligible user turn's text and is thereafter settled,
whatever that text was. (A ``None`` would be the exception — it is assigned and
leaves the picker unsettled — but ``conversation_history.content`` is
``NOT NULL`` on both engines, so no row can present one. A branch for it would
be a guard no test could defend.) It also keeps message bodies out of this path
entirely: a projection refresh never reads ciphertext, at any size of session.
"""
from __future__ import annotations

import hashlib
import json
import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import (
    Any,
    AsyncIterator,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
)

from .session_grouping import (
    coalesce_sessions_by_session_id,
    group_messages_into_sessions,
    timestamp_query_param,
)
from .session_id_column import is_stampable_session_id

logger = logging.getLogger(__name__)

_LOCK_DOMAIN = b"kestrel:conversation-session-projection-lock:v1\0"

#: Columns a projection row carries, in the order the upsert below binds them.
#: Named once so the INSERT, the UPDATE and the tests cannot disagree about the
#: shape of a row.
PROJECTION_COLUMNS: Tuple[str, ...] = (
    "started_at",
    "last_message_at",
    "message_count",
    "user_message_count",
    "first_user_message_id",
    "wake_source",
)

#: What a refresh selects per row. ``content`` is deliberately absent — the
#: derivation never needs it (see the module docstring), and reading ciphertext
#: bodies to maintain an index would be a cost paid on every single turn.
_ROW_SELECT = (
    "SELECT id, role, metadata, created_at "
    "FROM conversation_history "
    "WHERE agent_id = ? AND session_id = ? "
    "AND deleted_at IS NULL AND archived_at IS NULL "
    "ORDER BY id ASC"
)


def _lock_id(agent_id: str, session_id: str) -> int:
    """Stable signed-64-bit PostgreSQL advisory key for one session's row.

    Length-prefixed so no two ``(agent, session)`` pairs can share a hash
    input — ``("ab", "c")`` and ``("a", "bc")`` would otherwise be the same
    boundary. A hash collision only over-serializes two unrelated sessions; it
    cannot make a projection wrong.

    Its own domain prefix, so this key can never coincide with a migration's
    (:func:`~kestrel_sovereign.storage.async_database._backfill_lock_id`) or the
    lexical cleanup's. Two data keys colliding costs a little concurrency; a
    data key colliding with a migration's would let a single turn block a boot.
    """
    agent_bytes = agent_id.encode("utf-8")
    session_bytes = session_id.encode("utf-8")
    payload = b"".join(
        (
            _LOCK_DOMAIN,
            len(agent_bytes).to_bytes(4, "big"),
            agent_bytes,
            len(session_bytes).to_bytes(4, "big"),
            session_bytes,
        )
    )
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big", signed=True)


@dataclass(frozen=True, slots=True)
class SessionProjection:
    """One session as the write path decided it, ready to be stored."""

    session_id: str
    started_at: str
    last_message_at: str
    message_count: int
    user_message_count: int
    first_user_message_id: Optional[int]
    wake_source: Optional[str]


def _parse_metadata(raw: Any) -> Dict[str, Any]:
    """Metadata as the grouper wants it: a dict, or an empty one."""
    if isinstance(raw, dict):
        return raw
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except (TypeError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def project_session(session_id: str, rows: Sequence[Sequence[Any]]) -> Optional[SessionProjection]:
    """Derive one session's projection from its live rows.

    ``rows`` are ``(id, role, metadata, created_at)`` ordered by id ascending —
    the order the read path feeds the grouper, so gap arithmetic sees the same
    sequence it would there.

    Returns ``None`` when the session has no live rows left; the caller deletes
    the projection row rather than leaving one that describes nothing.
    """
    if not rows:
        return None

    messages: List[Dict[str, Any]] = []
    for row_id, role, metadata, created_at in rows:
        messages.append(
            {
                "id": row_id,
                "role": role,
                # The row's identity standing in for its text, so the preview
                # picker answers WHICH row rather than what it said. See the
                # module docstring for why that is faithful.
                "content": str(row_id),
                "metadata": _parse_metadata(metadata),
                "created_at": created_at,
            }
        )

    # ``keep_empty_markers`` so a conversation that exists only as its
    # ``new_session`` marker is still a session (#2222) — the UI prepends a tile
    # for it the moment the user starts typing, and a projection that dropped it
    # could not serve that reader. A reader that wants only sessions with
    # traffic filters on ``message_count``.
    grouped = coalesce_sessions_by_session_id(
        group_messages_into_sessions(messages, keep_empty_markers=True)
    )
    if not grouped:
        return None
    if len(grouped) != 1 or str(grouped[0]["session_id"]) != str(session_id):
        # Every row here carries this id in its metadata — Phase A only stamps
        # the column from the metadata the same INSERT stores — so the grouper
        # must file them all under it. Anything else means the column and the
        # metadata have parted company, which is a Phase A violation and not
        # something to paper over with a row that claims one of the answers.
        logger.error(
            "conversation_sessions: rows stamped session_id=%r grouped into %r; "
            "refusing to project a session the transcript does not show",
            session_id,
            [session["session_id"] for session in grouped],
        )
        return None

    session = grouped[0]
    # The picker was handed ids in place of text, so this IS the pointer.
    # ``is None`` rather than a truth test: only "the picker settled on
    # nothing" means no pointer, and a row id is never an empty string.
    preview = session.get("preview_content")
    return SessionProjection(
        session_id=str(session_id),
        started_at=session["started_at"],
        last_message_at=session["last_message_at"],
        message_count=session["message_count"],
        user_message_count=session["user_message_count"],
        first_user_message_id=None if preview is None else int(preview),
        wake_source=session.get("preview_wake_source"),
    )


class ConversationSessionProjection:
    """Keeps ``conversation_sessions`` true for one agent.

    Every method here is a *recompute*, never an adjustment. Callers name the
    sessions their mutation could have changed; this reads what survived and
    writes that. There is no path by which a count is incremented, so there is
    no path by which one drifts.
    """

    def __init__(self, db, agent_id: str) -> None:
        self.db = db
        self.agent_id = agent_id

    # ── maintenance ──────────────────────────────────────────────────────

    async def refresh(self, session_ids: Iterable[Optional[str]]) -> None:
        """Recompute the named sessions from the rows that are live now.

        Each session's read and write happen inside one serialized boundary —
        see :meth:`_serialized` and the module docstring — so a refresh
        describing an older state cannot land on top of a newer one.

        Sorted, because the order locks are taken in is the whole difference
        between a bounded wait and a cycle: a caller naming ``{A, B}`` and one
        naming ``{B, A}`` would otherwise deadlock, and destructive paths
        routinely name more than one session (a legacy row-id session overlaps
        the UUID sessions inside its window).

        Ids outside the Phase A column contract are skipped, and that is a
        shortcut rather than a gate: the COLUMN is the gate. No row can carry
        such an id there, so a refresh for one would find no rows, conclude
        "gone", and delete a projection row that was never written — the same
        outcome, one query later. Whoever reads this looking for the thing that
        keeps an unclaimable id out of the table: it is not this line, it is
        :func:`~kestrel_sovereign.storage.session_id_column.column_session_id`.
        """
        for session_id in sorted(_claimable(session_ids)):
            async with self._serialized(session_id):
                rows = await self.db.fetchall(
                    _ROW_SELECT, (self.agent_id, session_id)
                )
                projection = project_session(session_id, rows)
                if projection is None:
                    await self.forget(session_id)
                else:
                    await self._store(projection)

    @asynccontextmanager
    async def _serialized(self, session_id: str) -> AsyncIterator[None]:
        """Own one session's projection for a read and the write that follows.

        The boundary is transaction-scoped, so it is released by the commit
        rather than by anything here remembering to. When a mutation has already
        opened a transaction the boundary joins it and outlives this block,
        which is the point: the rows and the row that describes them become
        visible to other refreshers together.

        Neither backend can serialize this for free:

        * PostgreSQL takes an advisory lock on the ``(agent, session)`` key. Not
          a row lock on the projection row — the row is legitimately absent for
          a session being written for the first time, and that is precisely the
          case two concurrent first turns need serialized.
        * SQLite begins deferred, so a standalone refresh would read before
          reserving anything and could be overtaken between its two statements.
          A write that touches no row promotes the transaction to the one writer
          slot up front. Same device as ``serialized_token_cleanup`` and the
          exact purge's snapshot read, for the same reason.

        An unrecognized backend raises rather than running unserialized. Losing
        the boundary silently is how the projection acquires a wrong row, and
        wrong is the state it may never be in.
        """
        async with self.db.transaction():
            backend = getattr(self.db, "backend_type", "")
            if backend == "postgres":
                await self.db.fetchval(
                    "SELECT pg_advisory_xact_lock(?)",
                    (_lock_id(self.agent_id, session_id),),
                )
            elif backend == "sqlite":
                await self.db.execute(
                    "UPDATE conversation_sessions SET agent_id = agent_id WHERE 0"
                )
            else:  # pragma: no cover - AsyncDatabase exposes only these two
                raise RuntimeError(
                    "conversation_sessions cannot serialize a refresh on "
                    f"backend {backend!r}"
                )
            yield

    async def forget(self, session_id: str) -> None:
        """Drop a session's row. Used when nothing live is filed under it."""
        await self.db.execute(
            "DELETE FROM conversation_sessions "
            "WHERE agent_id = ? AND session_id = ?",
            (self.agent_id, session_id),
        )

    async def forget_all(self) -> None:
        """Drop every row for this agent — the whole history left the live set."""
        await self.db.execute(
            "DELETE FROM conversation_sessions WHERE agent_id = ?",
            (self.agent_id,),
        )

    async def rebuild(self) -> int:
        """Recompute every session this agent has, and return how many.

        The backfill and the repair are the same pass, deliberately: a
        projection that can only be built by a migration is one that cannot be
        fixed after a partial write.

        A Python iteration pass, one session at a time — which is why it must
        run OUTSIDE ``migration_lock``. That lock is ``BEGIN IMMEDIATE`` for its
        whole block, and a pass that re-enters the database inside it is the
        ABBA the lock's own docstring warns about.

        **Call this outside a transaction of your own, not inside one.** Each
        session takes its own boundary and releases it at the end of the
        enclosing transaction — so alone, this holds one key at a time, while
        nested inside a caller's transaction it would hold one per session until
        that caller commits. On PostgreSQL advisory locks occupy the same
        cluster-wide lock table as row locks, so a bulk restore of a few thousand
        sessions could exhaust it and fail unrelated work with "out of shared
        memory". The bulk importers therefore ``forget_all`` inside their
        transaction and rebuild after it commits: absent for the interval, which
        invariant 2 permits, rather than one transaction holding a key per
        session.
        """
        session_ids = [
            row[0]
            for row in await self.db.fetchall(
                "SELECT DISTINCT session_id FROM conversation_history "
                "WHERE agent_id = ? AND session_id IS NOT NULL",
                (self.agent_id,),
            )
            if row[0]
        ]
        # A session whose rows have all left the live set keeps no projection
        # row, and one whose id is no longer stamped anywhere keeps none either.
        # Both are found by diffing against what is stored rather than by
        # trusting the previous pass.
        stored = {
            row[0]
            for row in await self.db.fetchall(
                "SELECT session_id FROM conversation_sessions WHERE agent_id = ?",
                (self.agent_id,),
            )
        }
        await self.refresh(session_ids)
        # Sorted and inside the same boundary a refresh uses: dropping a row is
        # a write about that session like any other, and an id no live row
        # stamps is one nothing else is refreshing — but "nothing else" is a
        # claim about timing, and the boundary is what makes it one this code
        # does not have to make.
        for orphan in sorted(stored - set(_claimable(session_ids))):
            async with self._serialized(orphan):
                await self.forget(orphan)
        return len(_claimable(session_ids))

    # ── reads (for tests and Phase C) ────────────────────────────────────

    async def get(self, session_id: str) -> Optional[Dict[str, Any]]:
        """One stored projection row as a dict, or ``None``."""
        row = await self.db.fetchone(
            f"SELECT session_id, {', '.join(PROJECTION_COLUMNS)} "
            "FROM conversation_sessions WHERE agent_id = ? AND session_id = ?",
            (self.agent_id, session_id),
        )
        return None if row is None else _as_dict(row)

    async def list(self) -> List[Dict[str, Any]]:
        """Every stored projection row for this agent, newest activity first."""
        rows = await self.db.fetchall(
            f"SELECT session_id, {', '.join(PROJECTION_COLUMNS)} "
            "FROM conversation_sessions WHERE agent_id = ? "
            "ORDER BY last_message_at DESC, session_id ASC",
            (self.agent_id,),
        )
        return [_as_dict(row) for row in rows]

    # ── internals ────────────────────────────────────────────────────────

    async def _store(self, projection: SessionProjection) -> None:
        """Upsert one row. Both dialects accept this ON CONFLICT spelling."""
        assignments = ", ".join(
            f"{column} = excluded.{column}" for column in PROJECTION_COLUMNS
        )
        await self.db.execute(
            "INSERT INTO conversation_sessions "
            f"(agent_id, session_id, {', '.join(PROJECTION_COLUMNS)}) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT (agent_id, session_id) DO UPDATE SET "
            f"{assignments}",
            (
                self.agent_id,
                projection.session_id,
                # The grouper returns ISO text; asyncpg wants a datetime for a
                # TIMESTAMP column and SQLite wants the text. One shared
                # adapter, the same one the session queries bind through.
                self._timestamp(projection.started_at),
                self._timestamp(projection.last_message_at),
                projection.message_count,
                projection.user_message_count,
                projection.first_user_message_id,
                projection.wake_source,
            ),
        )

    def _timestamp(self, value: str) -> Any:
        return timestamp_query_param(getattr(self.db, "backend_type", ""), value)


def _claimable(session_ids: Iterable[Optional[str]]) -> List[str]:
    """The ids worth asking about, de-duplicated, order preserved.

    See :meth:`ConversationSessionProjection.refresh` — dropping the
    unstampable ones changes no outcome, only the number of queries.
    """
    seen: Dict[str, None] = {}
    for session_id in session_ids:
        if is_stampable_session_id(session_id):
            seen.setdefault(str(session_id), None)
    return list(seen)


def _as_dict(row: Sequence[Any]) -> Dict[str, Any]:
    return {
        "session_id": row[0],
        **{column: row[index + 1] for index, column in enumerate(PROJECTION_COLUMNS)},
    }
