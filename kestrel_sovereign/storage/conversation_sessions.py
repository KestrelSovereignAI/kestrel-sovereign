"""The ``conversation_sessions`` projection (#2959).

``conversation_history`` is the only source of truth about sessions. This table
is a **convenience copy that knows how far behind it is**.

Why a copy at all
=================

The conversations list shows the newest N sessions, so for an agent expected to
run indefinitely its cost must track the size of that page — history only grows.
Measured on PostgreSQL 16 with Phase A's index, a page of 50:

====================================  =========  ==========  ===========
history                               aggregate  skip-scan   projection
====================================  =========  ==========  ===========
1,000,000 rows                            62 ms      0.6 ms  O(sessions)
5,000,000 rows                           417 ms      0.5 ms  O(sessions)
5,000,000 rows, one dominant session     417 ms    4,913 ms  O(sessions)
====================================  =========  ==========  ===========

Deriving on the fly loses in one direction or the other. ``GROUP BY`` scans all
history every time. The skip-scan is excellent until sessions are few and large
— ``session_id <> previous`` is not indexable, so it walks — and that is the
*designed* shape here: #2877 binds completion wakes to their originating
session, so long-lived autonomous agents accumulate large message counts in
comparatively few sessions. Only the projection is bounded in every
distribution.

Left for Phase C (#2960), stated here so it is not rediscovered
--------------------------------------------------------------

Nothing in production calls :meth:`repair` yet, so ``conversation_sessions``
and ``conversation_session_watermarks`` hold nothing outside tests. That is
what makes the EPHEMERAL sweep in ``privacy_wrapper`` as narrow as it is: the
change ledger is the only table a trigger can fill on its own.

The moment Phase C maintains the projection, two things become live that are
dormant here, and both were found by review against this design rather than
guessed at:

* **A repair can republish purged state.** :meth:`_rebuild_from_transcript`
  reads history OUTSIDE its transaction, deliberately, because that read is
  unbounded. A pass that began before an EPHEMERAL purge can therefore take the
  lock afterwards and write a pre-purge snapshot — restoring leaked counts,
  timestamps and pointers after the purge reported success. Publishing needs to
  revalidate under the lock that the change stamp it read still stands.
* **The sweep grows back to three tables**, and with it the questions this
  revision could answer by narrowing: what evidence a retry uses, and whether
  the deletes need to be atomic with each other.

The cleanest answer to both is probably not to purge harder but to stop
projecting: a projection not maintained while EPHEMERAL is in force has nothing
to erase. That is a cross-cutting privacy change touching the trigger layer,
which is why it is named here rather than attempted in this phase.

The contract: a rebuildable cache, repaired in bounded chunks
=============================================================

The first attempt at this table maintained it as a **correctness invariant** —
every write path refreshed it, and the refreshes were serialized against each
other. Two review rounds raised two P1s, both inside that mechanism: concurrent
refreshes overwriting the projection with stale state, and then a bulk deletion
bypassing the serialization added to fix the first. Repeated P1s in one
mechanism, with clean fixes, is the signature of the mechanism being wrong
rather than the fixes being sloppy. So the obligation is removed instead:

1. **No write path maintains this table.** Inserts, soft-delete, restore,
   archive and purge (#2509/#2567) may all leave it stale. That is legal. Note
   the public surface below — there is deliberately no method a mutation could
   call to "keep it in step", because such a method is the thing that gets
   forgotten.
2. **Staleness is detected exactly, and cheaply.** The *database itself* counts
   every row event on ``conversation_history`` into a per-agent change stamp
   (see :func:`mutation_triggers`), and the projection records the stamp it
   worked from. Two primary-key reads answer "is this current", at any size of
   history, and nothing a write path can forget is involved.
3. **Repair walks history in bounded chunks**, each chunk one short transaction
   that both accounts for rows and records that it did.
4. **A full rebuild is always available.**
   :meth:`ConversationSessionProjection.rebuild` discards what is stored and
   derives every session from ``conversation_history`` again, arriving at the
   same table as any sequence of incremental repairs.

Why a database trigger and not a call in the write paths
========================================================

Because a call in the write paths is the mechanism that failed review twice. A
change stamp bumped from Python is a thing each mutation must remember, and
``AsyncConversationStore.update_message_metadata`` is proof that "each mutation"
is not a closed set: its key set is the *caller's*, so it can re-home a row from
one session to another, or flip ``operator_signal`` / ``signal_wake`` /
``new_session``, without touching liveness, counts or ``MAX(id)``.

That last part is also why the watermark cannot be an id alone. A mutation that
appends no row — delete, archive, restore, re-home — does not raise ``MAX(id)``,
so an id watermark cannot see it; nor can a summary built from aggregates, since
re-homing a row between two sessions leaves every total where it was. A counter
the engine maintains sees all of them, because the engine, not the caller,
decides when a row changed.

:data:`PROJECTION_INPUT_COLUMNS` is the column list the triggers watch, and it
is the same list this module reads. Adding a column to one without the other is
what makes a projection lie, so they are one constant.

``metadata`` is the one entry in that list the trigger does not compare whole,
and the reason is that comparing it whole made ordinary reading rebuild the
projection. ``MemoryRetriever.update_access`` and ``update_applied`` bump
``access_count`` / ``applied_count`` and stamp ``last_accessed`` through
``AsyncConversationStore.atomic_increment_metadata_counter`` on *retrieval*, so
recalling a memory rewrote the column, moved the stamp, appended no row — and
:meth:`ConversationSessionProjection._plan` therefore read a movement appends
cannot explain and derived every session again. Recall is not a rare event; it
is what the agent does. So the trigger compares
:data:`PROJECTION_METADATA_KEYS` — the keys the grouper actually consults —
extracted by :func:`watched_metadata_sql` in each dialect's own JSON functions,
and a document neither dialect can read reliably falls back to the raw text so
the direction of the mistake is always "rebuild needlessly", never "miss a
change".

Why each chunk is a transaction
===============================

Each step is::

    BEGIN
      take this agent's repair lock
      fold this agent's next CHUNK_ROWS live rows into the sessions they belong to
      record that the projection now accounts for history through their last id
    COMMIT

which buys four properties that would otherwise have to be reconstructed by
hand. An earlier draft of this module reconstructed them with an epoch counter
beside the watermark, every row write carrying ``WHERE the epoch is still
mine``, publication by compare-and-swap, and a newer repair fencing an older one
off mid-flight; that machinery is what this replaces:

* **Atomic by construction.** The rows and the watermark move together or
  neither does. There is nothing for a losing writer to half-publish, and no
  interval in which a recorded watermark stands over rows that were overwritten
  after it.
* **Crash-safe.** A crash costs at most one chunk's redo, and the watermark is
  never ahead of what was accounted for.
* **Exactly once, rather than idempotent.** A chunk *folds* its rows into the
  sessions they belong to; it does not recompute those sessions from scratch.
  That is the difference between a walk that reads each of the agent's live rows
  once and one that reads a session's whole history again for every chunk that
  mentions it — which for the whale session #2877 makes the designed shape is a
  full scan per chunk, inside a transaction advertised as short. A fold is only
  right if each row is folded once, so the fold and the watermark advance are one
  transaction (a crash redoes both or neither) and repairs of one agent are
  serialized (see below).
* **Bounded hold.** Nothing is held across an unbounded pass. That is the actual
  lesson of the wedge Phase A hit — a test sat for 23 hours 26 minutes holding
  ``pg_advisory_xact_lock``, idle in transaction, no waiter — and it is a lesson
  about holding a lock across an unbounded Python iteration, not about
  transactions. A bounded transaction is the ordinary tool for making two writes
  atomic, and reaching past it for optimistic concurrency trades a solved
  problem for an unsolved one.

Concurrency: one repair step per agent at a time
===============================================

A step takes the agent's own watermark row before it plans anything, and holds
it until the step commits. That is the whole mechanism — no epoch, no fence, no
compare-and-swap — and it costs one indexed statement on a row every step writes
anyway. Two things need it, and they are one requirement seen from two sides:

* **A fold must see each row once.** Two passes that both read ``through`` and
  both fold ``(through, through + CHUNK_ROWS]`` would each add those rows'
  counts, and the projection would be quietly too large. Recomputing instead of
  folding is what used to make that safe, and it is what made a whale session
  cost a scan per chunk.
* **A rebuild clears the ground before it walks it**, and under PostgreSQL's
  MVCC a ``DELETE`` cannot see rows another transaction has inserted and not yet
  committed. So an older rebuild's row for a session the newer rebuild no longer
  finds survives the newer rebuild's ``DELETE`` and lands after it — and the
  newer one then commits a watermark that says *current* standing over an orphan
  nothing will revisit. Nothing detects it: the stamp is the same, so
  :meth:`ConversationSessionProjection.is_stale` answers ``False`` truthfully
  about the stamp and falsely about the table. That is the one forbidden state
  in this whole contract, and it is not reachable through the atomic-pair
  argument, because the pair each transaction writes is internally consistent
  while the *table* is the union of two of them.

What the lock is **not** is a hold across the pass. It lives and dies inside one
chunk's transaction, whose work is bounded by :data:`CHUNK_ROWS`, and PostgreSQL
is given an explicit :data:`REPAIR_LOCK_WAIT_MS` so a pathological holder
surfaces as an error naming this contract rather than as the silent wedge Phase
A spent a day of wall-clock on. SQLite needs no second mechanism: its write
transactions are already exclusive, which is the same guarantee arrived at for
free.

Steps still carry nothing between them. Each replans under the lock from the
state it reads there, so two passes over one agent do not race — they take turns
advancing the same walk, and neither can act on a conclusion the other has
invalidated.

The one place a watermark still moves **backwards** is the transcript
derivation, which must read the agent's whole live history *outside* any
transaction (that read is the unbounded work nothing may be held across) and can
therefore reach the lock holding an older stamp than the one stored. Backwards
is the safe direction: it costs a redo. Forwards past what was accounted for is
what may never happen, and no pass can write it, because no pass records a stamp
it did not read before deriving.

What a row claims
=================

One row per ``(agent_id, session_id)``, describing the rows the *display
grouper* attributes to that session and which are **live** — the ones
``deleted_at IS NULL AND archived_at IS NULL``. That is the membership every
lifecycle operation acts on, so delete / archive / restore / purge /
``count_session_messages`` all touch the rows a row counts.

Attribution, not just membership, and the distinction is the whole of
:func:`project_transcript`. A row carrying no session id is filed under nothing,
so the grouper attributes it to whichever cluster it falls next to — a stamped
user turn followed by an unstamped assistant reply is *two* messages in that
session, not one. Reading only the rows that carry the id would report one, and
a projection that disagrees with the algorithm it caches is worse than no
projection. So an agent whose history still contains unstamped live rows is
projected by grouping its transcript, exactly as a reader would; an agent whose
rows all carry an id (the state Phase A's backfill leaves, and the only state
new writes produce) is projected by folding its rows forward, which is where the
``O(sessions)`` claim above comes from. The two paths are chosen by one indexed
probe, and no chunked pass is taken while an unstamped row could move an answer.

What a stored row means while a walk is in flight is therefore precise rather
than approximate: **it describes that session's live rows at or below the
watermark's ``through``**, and nothing above it. A part-walked session is
genuinely partial, and the watermark says exactly how partial. That is what
makes folding legitimate — a fold adds the rows between the old ``through`` and
the new one, which are by construction the rows the stored value does not yet
include.

``first_user_message_id`` is a **pointer**, never a copy. ``content`` is
ciphertext, so a preview column here would be a plaintext copy of encrypted text
sitting beside the ciphertext, and a record that outlives what it describes (see
#2948). Under this contract a pointer at a since-deleted row is a *detectable
stale state* rather than corruption: the row's departure moves the stamp.

What the change stamp can and cannot see
========================================

It counts row events on ``conversation_history``: one per INSERT, one per DELETE
and one per UPDATE that touches any of :data:`PROJECTION_INPUT_COLUMNS`. It is
monotonic and per agent, so:

* every insert, soft-delete, restore, archive, unarchive, purge and metadata
  rewrite moves it — including the ones that leave every aggregate unchanged,
  such as re-homing a row from one session to another;
* it cannot be moved *back*, so a projection can never mistake a changed history
  for the one it accounted for;
* what it deliberately does **not** describe is *which* rows moved. That is why a
  repair that cannot attribute the movement to appends derives everything again.

An UPDATE that touches only columns outside that list — ``content``,
``rendered_content``, ``embedding_vec``, ``model`` — does not move the stamp,
because nothing this module reads can have changed. That exclusion is checked by
a test rather than trusted, since the failure it would cause is silent.

Why the derivation calls the grouper
====================================

The fields here are the grouper's fields — ``started_at``, ``last_message_at``,
the two counts, the previewed turn, the wake source — and every one of them is a
rule that could be restated slightly differently. Phase A's Finding 4 was
exactly that: one rule written twice, drifting. So this does not restate them.
It hands rows to the very functions the differential test compares against, and
reads the answer off.

That is not circular, and the differential test is not tautological: on the
folded path the projection groups **one chunk's slice of one session** while the
test groups the **whole transcript** — where clusters split on gaps, absorb
unlabeled legacy rows, and are re-merged by id. The claim under test is that the
two arrive at the same place, which is a claim about the algorithm and not about
this function.

The *merge* of a slice into what is already stored is the grouper's too. Folding
needs a rule for combining two partial views of one session — earliest start,
latest activity, summed counts, the first preview to appear wins — and that rule
already exists, because a session resumed past the gap produces several clusters
that :func:`coalesce_sessions_by_session_id` combines by exactly those rules. So
a fold reconstitutes the stored row as a grouper session, hands it and the
slice's session to that function, and reads the answer off. Writing the four
rules again here is how they would come to differ from the read path's.

Coalescing merges by min/max where a single cluster's boundaries are *positional*
— the first and last row in id order — so the two agree only while ``created_at``
does not decrease as ``id`` increases within a session. Every writer produces
both from the same INSERT, so it holds; but a fold that quietly assumed it would
be wrong without saying so, and the assumption is cheap to check. A fold whose
slice starts before what the stored row already claims is therefore not folded at
all: that session is derived exactly, from its live rows up to the chunk's end.
The expensive branch is the *definition*, and the fold is the optimization of it
that is only taken where it provably agrees.

The one adaptation is ``first_user_message_id``. The grouper reports the
previewed *text*, not which row it came from, and this table stores the row.
Rather than restate the picker's skip rule (#2947: operator-signal notices and
autonomous wakes are not human turns) to find that row again, each row is handed
its own id in place of its text, and what comes back as the "preview" is then
the id of the row the picker chose.

That substitution is faithful because the picker never *inspects* ``content``:
it assigns the first eligible user turn's text and is thereafter settled,
whatever that text was. (A ``None`` would be the exception — it is assigned and
leaves the picker unsettled — but ``conversation_history.content`` is
``NOT NULL`` on both engines, so no row can present one. A branch for it would
be a guard no test could defend.) It also keeps message bodies out of this path
entirely: a repair never reads ciphertext, at any size of session.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .session_grouping import (
    canonical_timestamp_sql,
    coalesce_sessions_by_session_id,
    coerce_session_timestamp,
    group_messages_into_sessions,
    session_order_sql,
    timestamp_query_param,
)
from .session_id_column import SESSION_ID_KEY, is_stampable_session_id

logger = logging.getLogger(__name__)

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

#: What a watermark claims, in :class:`SessionWatermark` field order. All four
#: are written by one statement, in the same transaction as the rows they
#: describe — which is this module's entire concurrency mechanism.
WATERMARK_COLUMNS: Tuple[str, ...] = (
    "accounted_valid",
    "accounted_stamp",
    "accounted_through",
    "accounted_target",
)

#: How many of an agent's live rows one chunk folds.
#:
#: It bounds the rows a step *reads*, not merely the rows it selects, and the
#: distinction is the difference between a bounded repair and an unbounded one. A
#: step folds exactly the rows it selected into the sessions they belong to, so
#: one walk reads each of the agent's live rows once. An earlier draft selected a
#: chunk of ids and then recomputed each session those ids named from ALL of its
#: live rows: for the whale session #2877 makes the designed shape — five million
#: rows under one id — that is five thousand chunks each scanning five million
#: rows, all inside transactions this module advertises as short.
CHUNK_ROWS = 1000

#: How many chunks one :meth:`ConversationSessionProjection.repair` will run.
#:
#: A budget rather than "loop until current", because those differ for an agent
#: being written to faster than a walk can cross it. The honest report there is a
#: projection that is still behind — which this contract permits and makes
#: visible — rather than a call that never returns.
STEP_BUDGET = 4096

#: How long PostgreSQL may wait for another repair of the same agent, in ms.
#:
#: The lock is held for one chunk's transaction, so a wait longer than this is
#: not contention — it is a holder that has stopped making progress, which is the
#: shape of the wedge Phase A lost a day to. An error naming the contract is what
#: makes that visible; waiting forever is what made it invisible. SQLite needs no
#: counterpart: its write transactions are already exclusive, and its own
#: ``busy_timeout`` bounds the wait for them.
REPAIR_LOCK_WAIT_MS = 30_000

#: Every ``conversation_history`` column this module's answer depends on, and
#: therefore every column whose UPDATE must move the change stamp.
#:
#: One constant, read by the derivation's SELECTs *and* compiled into the
#: triggers, because two lists is how a projection starts lying: add a column
#: here that the trigger does not watch and a rewrite of it is invisible; watch
#: one the derivation never reads and every write of it forces a needless
#: rebuild.
#:
#: ``id`` is watched. An earlier revision left it out on the grounds that a
#: primary key "cannot be updated in place" — which is simply false on both
#: engines, as ``UPDATE ... SET id = ?`` was measured to succeed against SQLite
#: and PostgreSQL alike. The projection leans on ``id`` in three places (the
#: canonical tie-break, the chunk frontier, and ``first_user_message_id``, which
#: is stored), so a rewrite that moved one would leave the ledger untouched and
#: ``is_stale()`` answering false over a pointer to a row that no longer carries
#: that id. It costs nothing in practice: nothing in this codebase rewrites an
#: id, so the added comparison fires for maintenance and import SQL only —
#: which is exactly the traffic that would otherwise go unseen.
PROJECTION_INPUT_COLUMNS: Tuple[str, ...] = (
    "id",
    "agent_id",
    "session_id",
    "role",
    "metadata",
    "created_at",
    "deleted_at",
    "archived_at",
)

#: Every ``metadata`` key this module's answer depends on.
#:
#: ``metadata`` is in :data:`PROJECTION_INPUT_COLUMNS` because the derivation
#: reads it, but it is the one column the trigger must not compare *whole*:
#: ``access_count``, ``applied_count`` and ``last_accessed`` live in the same
#: document and are rewritten on every memory retrieval
#: (``MemoryRetriever.update_access`` → ``atomic_increment_metadata_counter``).
#: Those writes append no row, so a stamp they move is a stamp movement appends
#: cannot explain — a full rebuild, caused by reading.
#:
#: These four are what the grouper consults:
#: :func:`~kestrel_sovereign.storage.session_grouping.group_messages_into_sessions`
#: files a row by ``session_id``, splits on ``new_session``, and skips
#: ``operator_signal`` / ``signal_wake`` rows when picking the preview. The list
#: is held to that by a test that reads the grouper's source rather than this
#: comment, because a key added there and not here is invisible in the worst
#: direction.
PROJECTION_METADATA_KEYS: Tuple[str, ...] = (
    SESSION_ID_KEY,
    "new_session",
    "operator_signal",
    "signal_wake",
)


def watched_metadata_sql(backend_type: str, reference: str) -> str:
    """The part of ``reference``'s metadata this module's answer depends on.

    ``reference`` is a row expression such as ``OLD.metadata``. The result is a
    scalar SQL expression the trigger compares between OLD and NEW, so that a
    write touching only unwatched keys moves nothing.

    Three properties, in the order they matter:

    1. **It cannot raise.** ``metadata`` is free text and legacy rows hold
       malformed documents, so an extraction that raised would abort the
       ordinary UPDATE it rode along with — the encryption backfill, the #1402
       canonical/transport split — turning a projection optimization into a
       write path that fails. Both dialects are asked whether the parse will
       succeed (``json_valid`` / ``pg_input_is_valid``) before it is attempted,
       and both were measured short-circuiting a ``CASE`` on that answer for a
       document that would otherwise raise: sqlite 3.50.4 and PostgreSQL 16.14.
    2. **A document it cannot read reliably falls back to the raw text.** Two
       different malformed documents then compare unequal and force a rebuild,
       which is the harmless direction; collapsing them to one value would make
       a change between them invisible, which is not. The two branches cannot
       be confused for one another, and that is provable rather than lucky: the
       THEN branch yields a valid JSON document with no duplicated key, so any
       raw text equal to one would itself have taken the THEN branch.
    3. **It resolves duplicate keys the way Python does, or declines.** JSON
       permits a key twice and the readers disagree — SQLite's ``json_extract``
       takes the first, ``jsonb`` and ``json.loads`` take the last (Phase A
       measured this). PostgreSQL therefore agrees with the derivation for
       free; SQLite does not, so a document carrying a *watched* key more than
       once takes the raw-text branch rather than a value the derivation would
       not have read.
    """
    if backend_type == "postgres":
        built = ", ".join(
            f"'{key}', {reference}::jsonb -> '{key}'"
            for key in PROJECTION_METADATA_KEYS
        )
        return (
            f"CASE WHEN pg_input_is_valid({reference}, 'jsonb') "
            f"THEN jsonb_build_object({built})::text ELSE {reference} END"
        )
    paths = ", ".join(f"'$.\"{key}\"'" for key in PROJECTION_METADATA_KEYS)
    watched = ", ".join(f"'{key}'" for key in PROJECTION_METADATA_KEYS)
    # Nested rather than one conjunction: ``json_each`` raises on the same
    # documents ``json_extract`` does, so the validity test has to be a CASE the
    # duplicate test sits INSIDE, not a neighbour it might be evaluated beside.
    return (
        f"CASE WHEN json_valid({reference}) = 1 THEN "
        f"CASE WHEN (SELECT COUNT(*) - COUNT(DISTINCT key) "
        f"FROM json_each({reference}) WHERE key IN ({watched})) = 0 "
        f"THEN json_extract({reference}, {paths}) "
        f"ELSE {reference} END "
        f"ELSE {reference} END"
    )


def _watched_changed(backend_type: str, distinct: str) -> str:
    """"Did this UPDATE touch anything the projection reads?", in SQL.

    ``distinct`` is the dialect's null-safe inequality — ``IS DISTINCT FROM`` on
    PostgreSQL, ``IS NOT`` on SQLite. Every column in
    :data:`PROJECTION_INPUT_COLUMNS` is compared, and ``metadata`` is compared
    through :func:`watched_metadata_sql` so the keys that ride along in that
    document without being read do not count as a change.
    """
    terms = []
    for column in PROJECTION_INPUT_COLUMNS:
        if column == "metadata":
            terms.append(
                f"{watched_metadata_sql(backend_type, f'OLD.{column}')} "
                f"{distinct} "
                f"{watched_metadata_sql(backend_type, f'NEW.{column}')}"
            )
        else:
            terms.append(f"OLD.{column} {distinct} NEW.{column}")
    return " OR ".join(terms)


#: What "live" means, authored once. The projection describes these rows and the
#: probes below count these rows; two spellings of one membership rule is the
#: shape that drifts (Phase A's Finding 4).
_LIVE = "deleted_at IS NULL AND archived_at IS NULL"

#: The columns a derivation reads. ``content`` is deliberately absent — the
#: derivation never needs it (see the module docstring), and reading ciphertext
#: bodies to maintain an index would be a cost paid on every repair.
_DERIVED_FROM = "id, role, metadata, created_at, session_id"

#: **Ordered as the conversation list orders.** ``/api/conversations`` selects
#: ``ORDER BY created_at DESC`` and reverses, so the grouper sees chronological
#: order; deriving in id order instead would let the two disagree about session
#: boundaries wherever the two disagree about sequence. PostgreSQL's ``NOW()``
#: is transaction-start time, so an overlapping writer commits a later id
#: carrying an earlier timestamp — this is a real case, not a hypothetical one.
#: ``id`` breaks ties, which the endpoint's own ``ORDER BY`` did not, so equal
#: timestamps stop being resolved arbitrarily by the backend.
#: Spelled as SQL rather than sorted in Python so it is the SAME comparison
#: the endpoint makes, character for character. A backend that orders these
#: timestamp strings oddly must order both the same odd way; fidelity to the
#: list is the contract, and a "better" order here would be a disagreement.
#: The columns that define it, most significant first. Both directions are
#: built from this one tuple so the list and the projection cannot drift into
#: ordering by different things — the defect this constant was added for was
#: exactly two call sites spelling "in order" differently.
CANONICAL_ORDER_COLUMNS = ("created_at", "id")


def canonical_order(backend_type: str, *, descending: bool = False) -> str:
    """``ORDER BY`` for the one order sessions are derived in.

    ``descending`` is for a newest-first page that will be reversed before
    grouping, which is what ``/api/conversations`` does.

    ``created_at`` is compared through :func:`canonical_timestamp_sql`, never
    raw. SQLite stores this column as TEXT and its history legitimately mixes
    the ISO spelling with the SQL one — and ``"T"`` (0x54) sorts after a space
    (0x20), so ``'2026-03-01T09:00:00'`` compares GREATER than
    ``'2026-03-01 10:00:00'`` and an hour-earlier row sorts last. Measured: the
    two orders genuinely invert for that pair, and ``julianday()`` restores
    chronology. The module already warned about this trap for the fold's
    monotonicity guard; the first spelling of this function walked into it one
    definition away, which is what a rule written twice does.

    On PostgreSQL the column is a real timestamp and compares correctly on its
    own, so :func:`canonical_timestamp_sql` returns it unchanged there.
    """
    direction = "DESC" if descending else "ASC"
    # NULLs are placed explicitly, and the two directions are exact reverses of
    # each other so the reader's newest-first page reverses into this order.
    # The engines disagree by default — measured, `created_at ASC` puts NULL
    # LAST on PostgreSQL and FIRST on SQLite — which alone would make one
    # "canonical" order two. It also decides where an undatable row SITS, and
    # the grouper dates such a row from the row before it: ordered last, an
    # ordinary append to any session becomes its new predecessor and silently
    # re-dates it, which an incremental repair (touching only the appended
    # session) then records as current. Ordered first it has no predecessor and
    # never can acquire one, so its stamp is the epoch and nothing appended
    # later can move it. Undatable means "earliest", on both engines, always.
    nulls = "NULLS LAST" if descending else "NULLS FIRST"
    return "ORDER BY " + ", ".join(
        f"{_canonical_key(backend_type, column)} {direction}"
        + (f" {nulls}" if column == "created_at" else "")
        for column in CANONICAL_ORDER_COLUMNS
    )


def _canonical_key(backend_type: str, column: str) -> str:
    """One ordering key, spelled so both engines compare it the same way.

    Only ``created_at`` needs the treatment. ``id`` is an integer on both
    engines, so it compares numerically and identically — and PostgreSQL
    rejects ``COLLATE`` on a non-text type outright, which is how an earlier
    attempt to apply the bytewise rule uniformly here announced itself.
    Collation belongs on ``session_id``, in :func:`session_order_sql`.
    """
    if column == "created_at":
        return canonical_timestamp_sql(backend_type, column)
    return column




#: What one chunk selects: this agent's next live rows, and only those. The rows
#: a step reads and the rows it folds are the same rows, which is what makes the
#: chunk a bound on work rather than only on ids. Seeded by
#: ``idx_conversation_agent_row_id``.
_CHUNK = (
    f"SELECT {_DERIVED_FROM} FROM conversation_history "
    "WHERE agent_id = ? AND id > ? AND id <= ? "
    f"AND {_LIVE} "
    "ORDER BY id ASC LIMIT ?"
)

def _own_rows_through(backend_type: str) -> str:
    """One session's live rows up to the point a walk has reached.

    Only reached where a fold would not provably agree (see the module
    docstring) and by :meth:`_forget`'s caller, so its cost is the session's
    size rather than a per-chunk toll.

    A function of the backend rather than a constant, because the canonical
    order is: SQLite needs its TEXT timestamps compared through ``julianday``
    and PostgreSQL does not.
    """
    return (
        f"SELECT {_DERIVED_FROM} FROM conversation_history "
        "WHERE agent_id = ? AND session_id = ? AND id <= ? "
        f"AND {_LIVE} "
        f"{canonical_order(backend_type)}"
    )


def _live_rows_through(backend_type: str) -> str:
    """Every live row of this agent's up to one frontier, for the transcript
    pass — the same columns, in the order a reader would see them, so unstamped
    rows are attributed the way the grouper attributes them."""
    return (
        f"SELECT {_DERIVED_FROM} FROM conversation_history "
        f"WHERE agent_id = ? AND {_LIVE} "
        "AND id <= ? "
        f"{canonical_order(backend_type)}"
    )

#: :meth:`ConversationSessionProjection.repair` did nothing: the projection had
#: accounted for every row and the change stamp had not moved.
CURRENT = "current"

#: The walk continued — either it had not yet reached its target, or every change
#: since was a row appended above it — so only the rows it walked over were
#: folded into the sessions they belong to.
INCREMENTAL = "incremental"

#: The change stamp moved in a way appends cannot explain — a soft-delete,
#: restore, archive, purge or metadata rewrite — or the watermark accounted for
#: nothing at all. Which rows moved is not recoverable from a counter, so what
#: was stored was discarded and every session derived again.
REBUILT = "rebuilt"


# ── the schema this module owns ──────────────────────────────────────────
#
# Declared here rather than in ``CORE_SCHEMA`` because the projection's tables,
# its change ledger and the triggers that fill it are one design: a table
# created without its trigger is a projection that reports itself current
# forever. ``AsyncDatabase.ensure_session_projection_schema`` creates them
# together, behind one probe → migration lock → re-probe, which is what
# ``CREATE TABLE IF NOT EXISTS`` in a schema loop does not give: that spelling
# is idempotent in SEQUENCE and unsafe in PARALLEL, and ``_init_schema`` runs on
# every ``from_pool()`` — frinz calls it per request — so a post-upgrade request
# burst is exactly the parallel case.

_CONVERSATION_SESSIONS_DDL = """
CREATE TABLE IF NOT EXISTS conversation_sessions (
    agent_id              TEXT NOT NULL,
    session_id            TEXT NOT NULL,
    started_at            TIMESTAMP,
    last_message_at       TIMESTAMP,
    message_count         INTEGER NOT NULL DEFAULT 0,
    user_message_count    INTEGER NOT NULL DEFAULT 0,
    first_user_message_id INTEGER,
    wake_source           TEXT,
    PRIMARY KEY (agent_id, session_id)
)
"""

_WATERMARKS_DDL = """
CREATE TABLE IF NOT EXISTS conversation_session_watermarks (
    agent_id          TEXT PRIMARY KEY,
    accounted_valid   INTEGER NOT NULL DEFAULT 0,
    accounted_stamp   BIGINT NOT NULL DEFAULT 0,
    accounted_through BIGINT NOT NULL DEFAULT 0,
    accounted_target  BIGINT NOT NULL DEFAULT 0
)
"""

_CHANGES_DDL = """
CREATE TABLE IF NOT EXISTS conversation_history_changes (
    agent_id TEXT PRIMARY KEY,
    changes  BIGINT NOT NULL DEFAULT 0
)
"""

#: The upsert both dialects' triggers perform, parameterized only by which
#: row's ``agent_id`` is being stamped. Written once so the three SQLite
#: triggers and the one PostgreSQL function cannot count differently.
_BUMP = (
    "INSERT INTO conversation_history_changes (agent_id, changes) "
    "VALUES ({row}.agent_id, 1) "
    "ON CONFLICT (agent_id) DO UPDATE "
    "SET changes = conversation_history_changes.changes + 1"
)


def projection_tables() -> Tuple[Tuple[str, str], ...]:
    """``(table, DDL)`` for every table the projection owns, in creation order.

    The ledger comes last because the triggers created after it reference it; a
    trigger whose target table does not yet exist fails at creation on
    PostgreSQL and at *fire* time on SQLite, and the second of those is a boot
    that looks fine until the first message is written.

    DDL is written in the SQLite spelling and normalized per backend by the
    caller, the same treatment ``CORE_SCHEMA`` gets, so there is one declaration
    of each column's type rather than two that can drift.

    ``accounted_valid`` is what distinguishes *nothing has ever been accounted
    for* from *validly accounted for an empty history*. Those look identical as
    numbers — both are all-zero — and conflating them is reachable rather than
    theoretical: an upgrade creates this table and the ledger empty beside a
    ``conversation_history`` that is already full, so by the numbers alone a
    projection that has never been built would report itself current. The flag is
    checked before any equality.

    ``accounted_through`` is how far a walk has got, and ``accounted_target`` is
    where it ends. They are separate because a part-walked projection is a real
    state that must be *resumable* — the alternative is a crash costing a whole
    rebuild rather than one chunk — and because the target is also the reference
    point the append arithmetic in
    :meth:`ConversationSessionProjection._plan` needs.

    ``BIGINT`` on the counters, and ``accounted_stamp`` is why: it counts every
    row event over the agent's lifetime, which for an agent expected to run
    indefinitely is precisely the number this table exists to stop being bounded
    by. ``normalize_schema`` maps a bare ``INTEGER`` to PostgreSQL's ``int4``, so
    overflowing it there would not produce a wrong number but "value out of int32
    range" raised out of every subsequent repair — an agent whose projection can
    never advance again. SQLite reads ``BIGINT`` as INTEGER affinity, so both
    engines hold the range.
    """
    return (
        ("conversation_sessions", _CONVERSATION_SESSIONS_DDL.strip()),
        ("conversation_session_watermarks", _WATERMARKS_DDL.strip()),
        ("conversation_history_changes", _CHANGES_DDL.strip()),
    )


def mutation_trigger_function(backend_type: str) -> Optional[Tuple[str, str]]:
    """``(name, DDL)`` of the PL/pgSQL function the PostgreSQL triggers call.

    ``None`` on SQLite, which has no separate function object — its trigger
    bodies are the statements themselves.

    ``CREATE OR REPLACE`` rather than a probe: it is run under the same
    migration lock as the triggers, and an unconditional replace cannot leave a
    trigger pointing at an older body. What it must not do is run *unlocked* —
    two concurrent initializers replacing one function collide on
    ``pg_proc``'s unique index, and the loser's whole ``from_pool()`` raises.
    """
    if backend_type != "postgres":
        return None
    return (
        "kestrel_conversation_history_change",
        "CREATE OR REPLACE FUNCTION kestrel_conversation_history_change() "
        "RETURNS trigger AS $kestrel$ "
        "BEGIN "
        "  IF (TG_OP = 'DELETE') THEN "
        f"    {_BUMP.format(row='OLD')}; "
        "    RETURN OLD; "
        "  END IF; "
        "  IF (TG_OP = 'UPDATE' AND OLD.agent_id IS DISTINCT FROM NEW.agent_id) THEN "
        f"    {_BUMP.format(row='OLD')}; "
        "  END IF; "
        f"  {_BUMP.format(row='NEW')}; "
        "  RETURN NEW; "
        "END; "
        "$kestrel$ LANGUAGE plpgsql"
    )


def mutation_triggers(backend_type: str) -> Tuple[Tuple[str, str], ...]:
    """``(trigger, DDL)`` for the change stamp, in this engine's dialect.

    Three triggers, not one, and the split is the arithmetic in
    :meth:`ConversationSessionProjection._plan`: **each row event moves the
    stamp by exactly one**. That is what lets a repair prove a movement was
    entirely appends by comparing the stamp's delta against the number of rows
    now standing above what it accounted for — a claim that would be meaningless
    if some events counted twice.

    So an UPDATE stamps ``NEW.agent_id`` once, and ``OLD.agent_id`` only when a
    row actually changes hands. Re-homing between agents is not a thing any
    shipped path does, but "in practice nobody does that" is the reasoning that
    stops being true later, and the alternative here costs one comparison.

    The UPDATE trigger is narrowed to :data:`PROJECTION_INPUT_COLUMNS`. Not for
    tidiness: ``content``, ``rendered_content``, ``embedding_vec`` and ``model``
    are all rewritten in place by ordinary paths (the #1402 canonical/transport
    split, the embedding co-write, the encryption backfill), and stamping those
    would force a full rebuild for a change no field of this table can see.

    ``metadata`` is narrowed *within* the column, to
    :data:`PROJECTION_METADATA_KEYS`, and that is the same argument one level
    down rather than a refinement of it. ``access_count``, ``applied_count`` and
    ``last_accessed`` share that document and are rewritten every time a memory
    is retrieved, through ``atomic_increment_metadata_counter``. Comparing the
    column whole therefore made *reading* move the stamp — and a stamp movement
    with no row appended is precisely the movement
    :meth:`ConversationSessionProjection._plan` cannot attribute, so ordinary
    recall rebuilt the whole projection. :func:`watched_metadata_sql` is what
    each side is compared through.

    Row-level on both engines rather than PostgreSQL's statement-level
    transition tables, because one shape that is provably identical on the two
    engines is worth more here than a bulk-delete optimization on one of them:
    the price is N updates of a single row inside a statement that is already
    updating N rows, all in the same transaction.
    """
    watched_is_distinct = _watched_changed(backend_type, "IS DISTINCT FROM")
    if backend_type == "postgres":
        return (
            (
                "conversation_history_change_insert",
                "CREATE TRIGGER conversation_history_change_insert "
                "AFTER INSERT ON conversation_history FOR EACH ROW "
                "EXECUTE FUNCTION kestrel_conversation_history_change()",
            ),
            (
                "conversation_history_change_update",
                "CREATE TRIGGER conversation_history_change_update "
                "AFTER UPDATE ON conversation_history FOR EACH ROW "
                f"WHEN ({watched_is_distinct}) "
                "EXECUTE FUNCTION kestrel_conversation_history_change()",
            ),
            (
                "conversation_history_change_delete",
                "CREATE TRIGGER conversation_history_change_delete "
                "AFTER DELETE ON conversation_history FOR EACH ROW "
                "EXECUTE FUNCTION kestrel_conversation_history_change()",
            ),
        )
    # SQLite spells null-safe inequality ``IS NOT`` and has no trigger
    # functions, so each body carries the upsert directly. The conditional
    # second stamp is an ``INSERT ... SELECT ... WHERE``: SQLite's own answer to
    # "an upsert I want to skip", and the form that keeps the ON CONFLICT clause
    # unambiguous to its parser.
    watched_is_not = _watched_changed(backend_type, "IS NOT")
    rehomed = (
        "INSERT INTO conversation_history_changes (agent_id, changes) "
        "SELECT OLD.agent_id, 1 WHERE OLD.agent_id IS NOT NEW.agent_id "
        "ON CONFLICT (agent_id) DO UPDATE "
        "SET changes = conversation_history_changes.changes + 1"
    )
    return (
        (
            "conversation_history_change_insert",
            "CREATE TRIGGER conversation_history_change_insert "
            "AFTER INSERT ON conversation_history FOR EACH ROW BEGIN "
            f"{_BUMP.format(row='NEW')}; END",
        ),
        (
            "conversation_history_change_update",
            "CREATE TRIGGER conversation_history_change_update "
            "AFTER UPDATE ON conversation_history FOR EACH ROW "
            f"WHEN ({watched_is_not}) BEGIN "
            f"{_BUMP.format(row='NEW')}; {rehomed}; END",
        ),
        (
            "conversation_history_change_delete",
            "CREATE TRIGGER conversation_history_change_delete "
            "AFTER DELETE ON conversation_history FOR EACH ROW BEGIN "
            f"{_BUMP.format(row='OLD')}; END",
        ),
    )


@dataclass(frozen=True, slots=True)
class SessionWatermark:
    """What the projection records having accounted for.

    ``valid`` is not decoration and is checked before either number. All-zero is
    the state of a projection that has never been built, and for an agent with no
    history it is *also* the truth — so without a flag those two are one value,
    and an upgrade (which creates this table and the ledger empty beside a
    history that is already full) would call an empty projection current.

    ``stamp`` is the per-agent change stamp the walk that wrote this was working
    from. ``through`` is how far up ``conversation_history.id`` that walk has
    accounted for; ``target`` is where it ends — **the agent's ``MAX(id)`` at the
    instant ``stamp`` was read**. That pairing is not incidental: it is what makes
    the append arithmetic in :meth:`ConversationSessionProjection._plan` exact,
    and it is why the two are always written by the same statement.

    A projection is current exactly when it is valid, its walk reached its
    target, and the stamp has not moved since.
    """

    valid: bool = False
    stamp: int = 0
    through: int = 0
    target: int = 0

    @property
    def complete(self) -> bool:
        """Whether the walk that recorded this reached its target."""
        return self.through >= self.target

    def as_params(self) -> Tuple[int, ...]:
        """This watermark in ``WATERMARK_COLUMNS`` order.

        ``valid`` is bound as an ``int`` because the column is ``INTEGER`` on
        both engines; asyncpg refuses a Python ``bool`` for one.
        """
        return (int(self.valid), self.stamp, self.through, self.target)


#: "This projection accounts for nothing." The state before a first repair, and
#: the state :meth:`ConversationSessionProjection.rebuild` installs. A step
#: reading this derives every session again, whatever the numbers beside it say.
INVALID = SessionWatermark()


@dataclass(frozen=True, slots=True)
class RepairOutcome:
    """What a repair did, so a caller can tell "nothing to do" from "rebuilt".

    ``kind`` is the most expensive branch the pass took, so a repair that
    discarded the table and started again reports :data:`REBUILT` even if it
    finished with cheap chunks.

    ``current`` is whether the projection was current when the pass stopped.
    ``False`` is not an error: an agent written to faster than a walk can cross
    it leaves a projection that is legitimately behind, and saying so is what
    this contract is for.
    """

    kind: str
    sessions: int
    current: bool


class _NeedsTranscript(Exception):
    """A fold cannot be completed from this session's own rows.

    Where a session's boundaries stop being derivable in isolation. Clusters are
    split by *neighbouring* rows — another session's row between two of this
    one's ends a cluster — and ``coalesce_sessions_by_session_id`` then merges
    the pieces by min/max. Re-deriving the session alone sees no such split, so
    it produces one cluster whose boundaries are positional: the first and last
    row in id order.

    Those agree only while ``created_at`` rises with ``id``. On PostgreSQL it
    need not: ``NOW()`` is transaction-start time, so two overlapping writers
    can commit a later id carrying an earlier timestamp. The isolated derivation
    then stores a ``last_message_at`` EARLIER than its ``started_at`` — not
    merely a different answer from the grouper's but an incoherent row, under a
    watermark claiming to be current, on the column the conversation list is
    ordered by.

    So the fold gives up rather than guessing, and :meth:`_step` answers with the
    whole-transcript pass, which reads rows in order and can see the splits.
    Expensive, and the rarity is the point: it costs a full pass only when the
    cheap path cannot prove it agrees.
    """



def _caused_by(exc: BaseException, wanted: type) -> bool:
    """Whether *wanted* appears anywhere in *exc*'s cause/context chain."""
    seen = set()
    cur: Optional[BaseException] = exc
    while cur is not None and id(cur) not in seen:
        if isinstance(cur, wanted):
            return True
        seen.add(id(cur))
        cur = cur.__cause__ or cur.__context__
    return False


@dataclass(frozen=True, slots=True)
class _Plan:
    """What one step decided to do, computed inside that step's transaction.

    Never carried across steps. A step reasoning from a previous step's
    conclusion would be reasoning from a state another repair may have replaced,
    which is the class of bug that made the first design reach for a fence.
    """

    kind: str
    stamp: int
    through: int
    target: int
    discard: bool


@dataclass(frozen=True, slots=True)
class _Step:
    """What one chunk accomplished."""

    kind: str
    sessions: int
    done: bool


@dataclass(frozen=True, slots=True)
class SessionProjection:
    """One session as the rows attributed to it describe it, ready to store."""

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


def project_transcript(
    rows: Sequence[Sequence[Any]], expect: Optional[str] = None
) -> List[SessionProjection]:
    """Project every session the grouper finds in ``rows`` and the column may key.

    ``rows`` are ``(id, role, metadata, created_at, session_id)`` ordered by id
    ascending — the order the read path feeds the grouper, so gap arithmetic and
    the attribution of unstamped rows see the same sequence they would there.

    Given a whole transcript this is the reader's own answer, unstamped
    neighbours included. Given one session's rows, ``expect`` names the session
    those rows were selected by, and the grouping is required to produce that
    one session and nothing else — see the module docstring for how the two
    callers are chosen between.

    Sessions the column cannot key are dropped rather than stored under a key no
    reader could round-trip: the legacy row-id fallbacks of #2012, and any id
    outside Phase A's portable contract. Absence is the permitted direction.

    A session is also dropped — loudly — when a row the grouper attributed to it
    carries a *different* id in its indexed column. The column is derived from
    the metadata the same INSERT stores (#2958), so the two disagreeing is a
    Phase A violation, and there is no answer that is not a guess: filing it
    under either candidate would put a row in the projection that the transcript
    does not show there. A row whose column is NULL is not a disagreement — that
    is the ordinary unstamped row this function exists to attribute.

    ``expect`` refuses the same violation from the other side, and it is not the
    same check. Rows selected *by the column* that group into more than one
    session mean some of them are not where the column says, and the session the
    column named would otherwise be stored describing a strict subset of the
    rows a reader querying that column would find. Refuse the whole thing rather
    than store a count nobody else would compute.
    """
    messages: List[Dict[str, Any]] = []
    stamped: Dict[Any, Any] = {}
    for row_id, role, metadata, created_at, column in rows:
        stamped[row_id] = column
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
    # could not serve that reader. A reader wanting only sessions with traffic
    # filters on ``message_count``.
    # ``now`` pinned to the newest parseable stamp rather than left to default.
    # Unset, the grouper dates a row whose ``created_at`` is NULL or malformed
    # — which the nullable column and legacy SQLite rows both permit — from the
    # WALL CLOCK. The projection would persist that instant and mark itself
    # current, while grouping the same unchanged transcript a minute later
    # produces different boundaries, with nothing to notice because no row
    # moved and the change stamp never advanced. A cache that disagrees with
    # its source on re-derivation, permanently, and silently.
    #
    # The newest stamp present is deterministic, derived only from the rows in
    # hand, and orders such a row last — which is where a row with no time of
    # its own belongs in a sequence read in id order.
    # ``now`` is deliberately NOT passed. The deterministic substitute for an
    # undatable row is the grouper's own rule now, so the read path and this
    # projection cannot disagree about a legacy row with a malformed
    # ``created_at`` — which they did while this computed the fallback itself
    # and `/api/conversations` let the grouper reach for the wall clock (round-7
    # review). A projection that has to be reproducible from the rows cannot own
    # half of a rule the reader owns the other half of.
    grouped = coalesce_sessions_by_session_id(
        group_messages_into_sessions(
            messages,
            keep_empty_markers=True,
            collect_messages=True,
        )
    )
    if (
        expect is not None
        and messages
        and (len(grouped) != 1 or str(grouped[0]["session_id"]) != str(expect))
    ):
        logger.error(
            "conversation_sessions: rows stamped session_id=%r grouped into %r; "
            "refusing to project a session the transcript does not show",
            expect,
            [session["session_id"] for session in grouped],
        )
        return []

    projections: List[SessionProjection] = []
    for session in grouped:
        session_id = str(session["session_id"])
        if not is_stampable_session_id(session_id):
            continue
        divergent = [
            message["id"]
            for message in session.get("messages", ())
            if stamped.get(message["id"]) is not None
            and str(stamped[message["id"]]) != session_id
        ]
        if divergent:
            logger.error(
                "conversation_sessions: rows %s are stamped with a session_id "
                "their metadata groups under %r; refusing to project a session "
                "the transcript does not show",
                divergent,
                session_id,
            )
            continue
        projections.append(_as_projection(session))
    return projections


def _as_projection(session: Dict[str, Any]) -> SessionProjection:
    """One of the grouper's sessions as a row this table can hold."""
    # The picker was handed ids in place of text, so this IS the pointer.
    # ``is None`` rather than a truth test: only "the picker settled on nothing"
    # means no pointer, and a row id is never an empty string.
    preview = session.get("preview_content")
    return SessionProjection(
        session_id=str(session["session_id"]),
        started_at=session["started_at"],
        last_message_at=session["last_message_at"],
        message_count=session["message_count"],
        user_message_count=session["user_message_count"],
        first_user_message_id=None if preview is None else int(preview),
        wake_source=session.get("preview_wake_source"),
    )


def _as_grouped(projection: SessionProjection) -> Dict[str, Any]:
    """...and back, so a stored row can be merged by the grouper's own rules.

    The exact inverse of :func:`_as_projection` over the fields
    :func:`coalesce_sessions_by_session_id` consults, which is what lets a fold
    combine "what is stored" with "what this chunk saw" without restating how
    two views of one session combine. ``preview_metadata`` is carried as
    ``None`` because the merge only ever moves it alongside a preview it is
    taking, and nothing here reads it back.
    """
    pointer = projection.first_user_message_id
    return {
        "session_id": projection.session_id,
        "started_at": projection.started_at,
        "last_message_at": projection.last_message_at,
        "message_count": projection.message_count,
        "user_message_count": projection.user_message_count,
        "preview_content": None if pointer is None else str(pointer),
        "preview_metadata": None,
        "preview_wake_source": projection.wake_source,
    }


class ConversationSessionProjection:
    """Reads ``conversation_sessions`` for one agent, and repairs it on request.

    The public surface is deliberately narrow: ask whether it is stale, repair
    it, rebuild it, read it. There is **no** "refresh these sessions" method for
    a mutation to call — see the module docstring. A repair folds rows forward
    from the point the watermark records, and a movement it cannot attribute to
    appends discards what is stored and derives it all again, so no count can
    drift past the next repair.

    ``chunk_rows`` and ``step_budget`` are constructor arguments so a test can
    make a small corpus take many steps. Their defaults are the shipped values,
    so a caller passing neither gets exactly what production runs.
    """

    def __init__(
        self,
        db,
        agent_id: str,
        *,
        chunk_rows: int = CHUNK_ROWS,
        step_budget: int = STEP_BUDGET,
    ) -> None:
        self.db = db
        self.agent_id = agent_id
        self.chunk_rows = max(1, int(chunk_rows))
        self.step_budget = max(1, int(step_budget))

    # ── staleness ────────────────────────────────────────────────────────

    async def observed_changes(self) -> int:
        """This agent's change stamp as the database keeps it.

        One primary-key read. A missing row is zero rather than an error: an
        agent whose history has never been touched has had no row events, which
        is what zero says.

        Within one incarnation of the row the triggers only ever write it
        upward, which is what lets :meth:`is_stale` compare it for equality
        rather than for order. The row is not immortal, though: the EPHEMERAL
        sweep erases it when no history survives, and the trigger's next write
        is an INSERT of ``1``. So a stamp read from one incarnation says nothing
        about a later one, and equality across that boundary would be an
        accident rather than evidence. The sweep is what upholds this — it
        erases every table this projection owns together, so no stamp outlives
        the ledger it was read from (``purge_session_projection``).
        """
        value = await self.db.fetchval(
            "SELECT changes FROM conversation_history_changes WHERE agent_id = ?",
            (self.agent_id,),
        )
        return int(value or 0)

    async def accounted(self) -> SessionWatermark:
        """The state the projection records having accounted for.

        A missing row reads as :data:`INVALID` rather than as an error: an agent
        whose projection has never been repaired has accounted for nothing, and
        *invalid* is the honest name for that — it must derive, not compare.
        """
        row = await self.db.fetchone(
            f"SELECT {', '.join(WATERMARK_COLUMNS)} "
            "FROM conversation_session_watermarks WHERE agent_id = ?",
            (self.agent_id,),
        )
        if row is None:
            return INVALID
        return SessionWatermark(
            bool(row[0]), int(row[1]), int(row[2]), int(row[3])
        )

    async def is_stale(self) -> bool:
        """Whether the projection disagrees with the rows it describes.

        Two primary-key reads, and no scan of history at any size — which is the
        whole reason the stamp is maintained by the engine. ``False`` is the
        strong claim and it rests on three things: the watermark is valid, the
        walk that wrote it reached its target, and the stamp has not moved since.
        Every row event moves the stamp, so "has not moved" means "no row this
        projection could describe has changed" — including the changes that leave
        every aggregate where it was, such as a re-homed row or a flipped preview
        flag, and including the ones an id watermark cannot see because they
        append nothing.
        """
        accounted = await self.accounted()
        if not accounted.valid or not accounted.complete:
            return True
        return await self.observed_changes() != accounted.stamp

    # ── repair ───────────────────────────────────────────────────────────

    async def repair(self) -> RepairOutcome:
        """Bring the projection up to date, one bounded chunk at a time.

        Each step is a short transaction that recomputes the sessions named by
        the next chunk of this agent's history and records that it did — both
        halves together, so the rows and the watermark move as one. Steps carry
        nothing between them: every decision is made inside a step's own
        transaction from the state it reads there, which is what makes two
        repairs racing harmless (module docstring, "what two racing repairs do").

        Three kinds, in increasing cost:

        * :data:`CURRENT` — valid, walked to its target, and the stamp has not
          moved. Two reads; no rows read and none written.
        * :data:`INCREMENTAL` — the walk continued, either because it had not
          finished or because every change since was a row appended above its
          target. Only the sessions those rows belong to are recomputed.
        * :data:`REBUILT` — anything else. What was stored is discarded and every
          session derived again.

        **Why the append test is exact.** Each row event moves the stamp by one,
        so over any interval the delta is ``inserts + updates + deletes`` — while
        the number of rows now standing above the recorded ``target`` is
        ``inserts_above - deletes_above``, no row having stood above it when the
        stamp was read (``target`` was ``MAX(id)`` at that instant). The second is
        never larger than the first, and they are equal only when ``updates``,
        ``deletes`` and ``inserts_below`` are all zero. So the equality is not a
        heuristic that usually holds: it is false whenever anything at or below
        the target moved, which is exactly when continuing the walk would be
        unsound.

        **Call this outside a transaction of your own.** It is a Python iteration
        pass that re-enters the database per session, so wrapping the whole of it
        in one transaction is the ABBA shape ``migration_lock``'s own docstring
        warns about — and on SQLite it would hold the single writer slot for the
        entire walk, which is the unbounded hold this design exists to avoid.
        """
        kind = CURRENT
        sessions = 0
        for _ in range(self.step_budget):
            step = await self._step()
            sessions += step.sessions
            if step.kind == REBUILT or (kind == CURRENT and step.kind != CURRENT):
                kind = step.kind
            if step.done:
                # ``done`` is about the STEP: it finished the plan it was given.
                # ``current`` is about the PROJECTION, and the two part company
                # whenever a row arrives while the pass is running — the plan
                # was computed from a stamp read before the derivation, so the
                # pass can complete correctly and still leave the projection
                # behind. Reporting the step's completion as the projection's
                # currency is how a caller ends up told it is up to date while
                # a newer row sits unaccounted. Asked rather than assumed.
                return RepairOutcome(kind, sessions, not await self.is_stale())
        logger.info(
            "conversation_sessions: %s's projection was still behind after %d "
            "chunks; it stays legitimately stale until the next repair",
            self.agent_id,
            self.step_budget,
        )
        return RepairOutcome(kind, sessions, False)

    async def rebuild(self) -> int:
        """Discard what is stored and derive every session from history again.

        The recovery path for a projection that has been corrupted by hand or
        dropped outright, and what "a rebuild equals the repairs that got there"
        is asked of. Unlike :meth:`repair` it does not short-circuit on a
        matching stamp: a caller reaching for this wants the table derived again,
        not to be told the watermark looks fine.

        Invalidating first is what makes that unconditional, and it is done by
        writing the watermark rather than by deleting rows. An invalid watermark
        is the one state every step already treats as "derive everything", so a
        rebuild runs through exactly the machinery a repair does — including its
        chunking — instead of being a second path that can drift from it.

        Returns how many sessions the projection HOLDS afterwards, counted from
        the table rather than summed from the chunks that wrote it: a session
        spanning several chunks is upserted once per chunk, so the write count
        exceeds the session count and contradicts what this promises.

        **A rebuild is budgeted like any repair and may stop part-way.** A
        history longer than ``step_budget * chunk_rows`` leaves the projection
        legitimately behind, with the chunks it did finish committed. That is
        the contract, not a failure — but it does mean this count alone must
        not be read as "and now it is complete". Ask :meth:`accounted` or
        :meth:`is_stale`, which is where completion has always been answered.

        Zero is a real answer for an agent with no sessions.
        """
        await self._record(INVALID)
        await self.repair()
        return len(await self.list())

    # ── reads (for tests, diagnostics, and Phase C) ──────────────────────

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
            f"{session_order_sql(self.db.backend_type)}",
            (self.agent_id,),
        )
        return [_as_dict(row) for row in rows]

    # ── one chunk ────────────────────────────────────────────────────────

    async def _step(self) -> _Step:
        """One chunk: decide, derive, write and record, in one short transaction.

        Everything the step decides from is read inside the transaction, under
        this agent's repair lock, and everything it concludes is written there.
        So a step is atomic in the only sense that matters here — nobody can
        observe its rows without its watermark, or its watermark without its rows
        — and it is alone in the only sense that matters: no second repair of
        this agent can be between its read and its write.

        The staleness probe ahead of the transaction is an optimization and is
        allowed to be wrong. It costs two primary-key reads and saves a
        write-mode transaction on the overwhelmingly common "nothing has changed"
        call; being told to enter and then finding nothing to do is handled by
        :meth:`_plan` returning ``None`` under the lock, which is where the
        authoritative answer has always been.

        The transcript derivation is the one thing that cannot live inside a
        chunk, because attribution is a property of the whole sequence — a chunk
        boundary would split a cluster and file an unstamped row under the wrong
        session. So when an unstamped live row is present the step hands over to
        :meth:`_rebuild_from_transcript`, which reads outside any transaction.
        """
        if not await self.is_stale():
            return _Step(CURRENT, 0, True)
        # ``immediate`` so SQLite takes its writer slot at BEGIN rather than on
        # first write: a deferred transaction that has already read cannot
        # upgrade, so cross-process repairs would fail here rather than take
        # turns. On PostgreSQL the lock below is the mechanism and this is a
        # no-op.
        try:
            async with self.db.transaction(immediate=True):
                await self._claim()
                accounted = await self.accounted()
                # The stamp is read BEFORE anything is derived, never after. A row
                # arriving during the step then falls outside what the step records,
                # so the next repair sees it. Behind is the safe direction; ahead is
                # the silent gap this whole contract exists to make impossible.
                observed = await self.observed_changes()
                plan = await self._plan(accounted, observed)
                if plan is None:
                    return _Step(CURRENT, 0, True)
                if not await self._has_unstamped_rows():
                    return await self._chunk(plan)
        except Exception as exc:
            # The signal is raised inside the step's transaction so that the
            # rollback is the abort: nothing the chunk wrote stands, and the
            # watermark still describes the state before it. The backend wraps
            # whatever leaves a transaction in its own error type, so the cause
            # chain is walked rather than the type matched — catching only
            # ``_NeedsTranscript`` here silently never fires, which is how this
            # was first found.
            if not _caused_by(exc, _NeedsTranscript):
                raise
            # Falls through to the transcript pass below, the only reading that
            # can see the cluster splits a session's own rows do not show.

        # Outside the transaction above, deliberately: the transcript pass reads
        # the agent's whole live history, which is the unbounded work nothing may
        # be held across.
        return await self._rebuild_from_transcript()

    async def _claim(self) -> None:
        """Hold this agent's repair for the rest of the caller's transaction.

        On PostgreSQL, two statements, and the first is not redundant.
        ``FOR UPDATE`` locks the
        rows it *finds*, so an agent whose watermark row does not exist yet — the
        first repair, which is also the discard-everything repair — would be
        locked by nobody, and two first repairs would proceed together. Inserting
        the row first makes it exist to be locked, and the insert itself
        serializes the pair that raced to create it: on PostgreSQL the loser
        waits on the primary key's unique index and then finds the winner's row.
        The inserted row is all zeros, which is :data:`INVALID` — the same state
        a missing row reads as, so nothing downstream can tell the difference.

        Postgres is told how long it may wait. The lock is held for one chunk, so
        a longer wait is a holder that has stopped, and Phase A's 23-hour wedge
        is what an unbounded wait looks like when that happens. ``SET LOCAL``
        scopes the limit to this transaction, so no connection carries it back to
        the pool.

        **This does nothing at all on SQLite**, and the emptiness is the design.
        A SQLite write transaction is already exclusive, so the exclusion is the
        step's ``BEGIN IMMEDIATE`` (see :meth:`_step`) and there is nothing here
        to add. Issuing the insert on both engines would be worse than useless:
        it would make the step's first statement a write, which is a *second*
        way of getting the writer slot at BEGIN — so either mechanism could be
        deleted with every test still passing. Measured, on this file's
        two-connection case. One mechanism per engine, each with a case that
        fails when it goes, is the only arrangement where that cannot happen.
        """
        if getattr(self.db, "backend_type", "") != "postgres":
            return
        # BEFORE the insert, not after: the insert is itself a statement that
        # can block. Two first repairs race on a watermark row that does not
        # exist yet, so the loser waits on the winner's uncommitted unique-key
        # entry — and a limit set afterwards does not apply to the wait that
        # already happened. A stalled winner would then hold the loser
        # indefinitely, which is the unbounded wait this limit exists to
        # forbid, arriving one statement too late to forbid it.
        await self.db.execute(
            f"SET LOCAL lock_timeout = {int(REPAIR_LOCK_WAIT_MS)}"
        )
        await self.db.execute(
            "INSERT INTO conversation_session_watermarks (agent_id) VALUES (?) "
            "ON CONFLICT (agent_id) DO NOTHING",
            (self.agent_id,),
        )
        await self.db.fetchval(
            "SELECT accounted_valid FROM conversation_session_watermarks "
            "WHERE agent_id = ? FOR UPDATE",
            (self.agent_id,),
        )

    async def _plan(
        self, accounted: SessionWatermark, observed: int
    ) -> Optional[_Plan]:
        """What this step should do, or ``None`` when there is nothing to do.

        Three answers, and the order they are asked in is load-bearing:

        1. **An invalid watermark derives everything**, before any arithmetic is
           reached. Zero is a number two different states share — "never built"
           and "built, and nothing has happened since" — and only the flag tells
           them apart. The state where that bites is the upgrade itself: the
           ledger and the watermark are created empty beside a
           ``conversation_history`` that is already full.
        2. **An unmoved stamp continues the walk**, with the same stamp and the
           same target. Nothing anywhere has changed since that stamp was read,
           so everything the walk has already written is still true and the only
           work left is the rows it has not reached.
        3. **A moved stamp is arithmetic.** If the movement is entirely rows
           appended above the recorded target, what the walk has written is
           untouched and it simply gets a further target; otherwise something at
           or below the target moved, which a counter cannot localize, so
           everything is derived again.

        ``MAX(id)`` is read *after* ``observed``, never before, so the pair this
        writes keeps the property the arithmetic depends on: no row stood above
        ``target`` at the instant ``stamp`` was read. Reading it first would admit
        a row that landed in between — counted by the stamp, yet at or below the
        target — and the next step's equality would be comparing two numbers that
        are no longer about the same instant.
        """
        # LIMIT (#3001): everything below assumes row ids are POSITIVE and only
        # ever APPEND. Both hold for every writer here — ``AUTOINCREMENT`` and
        # ``bigserial`` issue increasing positive ids — and neither is enforced
        # by the schema, so maintenance or import SQL can break them. Two shapes
        # this misreads, both reported by round-9 review:
        #
        #   * an id rewritten from at-or-below the target to above it bumps the
        #     stamp once and leaves one row above the target, which is exactly
        #     an append's signature, so the row is folded into its already
        #     counted session a second time;
        #   * a row with an id of zero or less is never selected by the walk,
        #     which starts at ``through = 0`` and takes ``id > through``.
        #
        # Both end with a watermark recorded as current over a projection that
        # is not. Telling them apart needs the watermark to carry more than
        # ``max(id)`` — a live row count would separate "one appended" from "one
        # moved" — which is a change to what a watermark IS, so it is #3001
        # rather than a guard bolted on here.
        if not accounted.valid:
            return _Plan(REBUILT, observed, 0, await self._max_id(), True)
        if observed == accounted.stamp:
            if accounted.complete:
                return None
            return _Plan(
                INCREMENTAL,
                accounted.stamp,
                accounted.through,
                accounted.target,
                False,
            )
        delta = observed - accounted.stamp
        if delta > 0 and delta == await self._rows_above(accounted.target):
            return _Plan(
                INCREMENTAL,
                observed,
                accounted.through,
                await self._max_id(),
                False,
            )
        return _Plan(REBUILT, observed, 0, await self._max_id(), True)

    async def _chunk(self, plan: _Plan) -> _Step:
        """Fold the next chunk of history in, inside the caller's transaction.

        ``discard`` empties the table first, which is what makes a rebuild a
        rebuild: a session whose every row has been purged leaves nothing behind
        to find it by, so no derivation can remove its stored row and only
        clearing the ground can. It costs nothing asymptotically — the walk that
        follows visits every session anyway — and it is one statement rather than
        a second spelling of "which sessions still exist", which is the kind of
        restatement Phase A's Finding 4 was filed about.

        The projection is then absent for the duration of that walk. Absent is
        the permitted direction: the watermark says so throughout, and the epic's
        invariant is that this table may be silent but may never disagree. That
        the DELETE cannot be undone by a *concurrent* rebuild's uncommitted
        inserts is what :meth:`_claim` is for.

        The chunk is measured in **this agent's live rows**, not in a span of
        ids, so a sparse agent in a shared history does not pay a step per empty
        range, and the rows a step reads are exactly the rows it folds.
        """
        if plan.discard:
            await self.db.execute(
                "DELETE FROM conversation_sessions WHERE agent_id = ?",
                (self.agent_id,),
            )
        rows = await self.db.fetchall(
            _CHUNK,
            (self.agent_id, plan.through, plan.target, self.chunk_rows),
        )
        # Short of a full chunk means no further LIVE row of this agent's stands
        # at or below the target — and a row that is not live contributes nothing
        # to fold, so there is nothing between here and the target to come back
        # for. The two branches are genuinely different values rather than a
        # tidier spelling of one: the last row's id is where the NEXT chunk must
        # resume from, while the target is what having finished means.
        through = (
            plan.target if len(rows) < self.chunk_rows else int(rows[-1][0])
        )
        written = await self._fold(rows, through)
        await self._record(
            SessionWatermark(True, plan.stamp, through, plan.target)
        )
        # Reaching the target IS being finished, and saying so is not cosmetic:
        # a repair whose last permitted step lands exactly on the target would
        # otherwise burn its whole budget and then report ``current=False``
        # while :meth:`is_stale` already answers false. A caller that trusts the
        # outcome would call again forever; one that trusts ``is_stale`` would
        # disagree with the value it was just handed. ``through`` is what was
        # accounted for, so comparing it to the target is the same question the
        # next :meth:`_plan` would ask.
        return _Step(plan.kind, written, through >= plan.target)

    async def _rebuild_from_transcript(self) -> _Step:
        """Derive every session by grouping the agent's whole live transcript.

        The path for a history that still holds rows carrying no session id.
        Those belong to whichever cluster they fall next to, so they can only be
        attributed by reading the transcript in order — which is why this is not
        chunked by id, and why the reading happens outside any transaction.

        The writing is one transaction, and that is not a relaxation of the
        bounded-hold rule but the same rule applied to what is actually
        unbounded. The *read* is the unbounded work and it is outside; the write
        is one upsert per session, which is the size of the table this pass
        exists to produce. Splitting it into batches was worse than it looked:
        only the first batch cleared the ground, so a second pass interleaving
        with it left the union of two derivations standing under whichever
        watermark committed last — the orphan state :meth:`_claim` exists to make
        impossible, reintroduced one level up.

        Progress is deliberately not recorded mid-pass, and that is a real
        difference from the chunked path rather than an oversight: a resumable
        transcript walk would have to name the id below which attribution is
        already settled, and that id is not a batch boundary — one session's rows
        can straddle any number of them. So this pass is all-or-nothing, and a
        crash costs the pass rather than the projection.
        """
        observed = await self.observed_changes()
        target = await self._max_id()
        # NOT COVERED BY A TEST. Four attempts failed to build one that can
        # observe the defect: an unstamped-row fixture keeps every later repair
        # on this same path, which re-derives from scratch and is idempotent,
        # and a fixture that reaches here via an inversion makes every later
        # fold escalate here too, for the same reason. Both pass with the bound
        # removed. The reasoning below stands on its own and the change is one
        # clause, but treat it as unverified: if it is ever suspected, the
        # missing test is the first thing to write.
        #
        # Bounded by the frontier the target was read at, not left open. A row
        # appended between those two reads would otherwise be derived INTO the
        # projection while the watermark recorded below still stops short of
        # it — so the next repair sees it as an append above the target and
        # folds it a second time, doubling that session's count and marking the
        # result current. The read is the unbounded work, but "unbounded" is
        # about how MANY rows, never about which: a snapshot must describe one
        # frontier and say which one.
        projections = project_transcript(
            await self.db.fetchall(
                _live_rows_through(self.db.backend_type),
                (self.agent_id, target),
            )
        )

        written = 0
        async with self.db.transaction(immediate=True):
            await self._claim()
            await self.db.execute(
                "DELETE FROM conversation_sessions WHERE agent_id = ?",
                (self.agent_id,),
            )
            for projection in projections:
                written += await self._store(projection)
            await self._record(SessionWatermark(True, observed, target, target))
        # Accounted through == target by construction: this pass derived every
        # live row, not a chunk of them. Reporting otherwise sends the caller
        # back for a step that has nothing left to do.
        return _Step(REBUILT, written, True)

    # ── internals ────────────────────────────────────────────────────────

    async def _max_id(self) -> int:
        """The highest row id this agent has. One backward index step.

        ``idx_conversation_agent_row_id`` is what makes that true on PostgreSQL;
        SQLite gets the plan free because ``id`` is its rowid and rides in every
        index as a trailing key column.

        Worth stating rather than assuming: a row at or below this can still
        *appear* afterwards, because a sequence hands out ids before the
        transaction holding one commits. That is not a hole. Such a row's INSERT
        moves the stamp and does not stand above the target, so the next step's
        arithmetic finds a movement appends cannot explain and derives everything
        again — the conservative branch, reached without anyone having to notice
        the case.
        """
        return int(
            await self.db.fetchval(
                "SELECT COALESCE(MAX(id), 0) FROM conversation_history "
                "WHERE agent_id = ?",
                (self.agent_id,),
            )
            or 0
        )

    async def _rows_above(self, target: int) -> int:
        """How many of this agent's rows stand above ``target``.

        Counted over ALL rows, live or not, because the stamp counts row events
        regardless of liveness and the two are compared against each other.
        Bounded by the rows that appeared since the last pass — which are the
        rows the walk is about to read anyway.
        """
        return int(
            await self.db.fetchval(
                "SELECT COUNT(*) FROM conversation_history "
                "WHERE agent_id = ? AND id > ?",
                (self.agent_id, target),
            )
            or 0
        )

    async def _has_unstamped_rows(self) -> bool:
        """Whether any live row of this agent's is filed under no session id.

        Seeded by Phase A's ``(agent_id, session_id)`` index and stopped at the
        first hit, so the ordinary answer costs one seek. It is not free in the
        pathological case — an agent holding many unstamped rows that are all
        soft-deleted or archived pays a heap visit per candidate — but that is an
        agent already on the transcript derivation, whose repair reads its whole
        live history anyway.

        This chooses between the two derivations: with no unstamped rows a
        session's own rows are the whole of its story, and with any of them
        attribution has to be read off the transcript (see the module docstring).
        """
        row = await self.db.fetchone(
            "SELECT 1 FROM conversation_history "
            f"WHERE agent_id = ? AND session_id IS NULL AND {_LIVE} "
            "LIMIT 1",
            (self.agent_id,),
        )
        return row is not None

    async def _fold(
        self, rows: Sequence[Sequence[Any]], through: int
    ) -> int:
        """Fold one chunk's rows into the sessions they belong to.

        Only ever reached with no unstamped live rows in play (see
        :meth:`_step`), which is what makes a row's column the whole story of
        where it belongs — so the chunk can be partitioned by that column without
        consulting anything else, and each partition grouped on its own.

        The rows handed here are the rows the chunk read. Nothing re-reads a
        session's history: that is the difference between a walk costing one pass
        over the agent's live rows and one costing a pass per chunk, which for a
        session of millions of rows is the difference the projection exists to
        buy in the first place.

        Sorted so that two passes folding the same chunk touch its rows in the
        same order. They should not overlap at all — :meth:`_claim` sees to that
        — but a deterministic order costs nothing and means an unforeseen overlap
        is contention rather than a lock cycle.
        """
        by_session: Dict[str, List[Sequence[Any]]] = {}
        for row in rows:
            session_id = row[4]
            if session_id is None:
                continue
            by_session.setdefault(str(session_id), []).append(row)

        written = 0
        for session_id in sorted(by_session):
            projection = await self._folded(
                session_id, by_session[session_id], through
            )
            if projection is None:
                # The rows the column selected are not the ones the transcript
                # files under it — a Phase A violation, logged by
                # ``project_transcript``. Absent rather than a guess.
                await self._forget(session_id)
            else:
                written += await self._store(projection)
        return written

    async def _folded(
        self,
        session_id: str,
        rows: Sequence[Sequence[Any]],
        through: int,
    ) -> Optional[SessionProjection]:
        """What one session looks like once this chunk's rows are counted in.

        The stored row describes that session's live rows up to the previous
        watermark; ``rows`` are the ones between there and ``through``. Combining
        the two is :func:`coalesce_sessions_by_session_id`'s job — it already
        combines the several clusters a session resumed past the gap produces,
        by exactly the rules a fold needs — so the rule is used rather than
        restated.

        Coalescing merges boundaries by min/max, while a single cluster's
        boundaries are positional: the first and last row in id order. Those
        agree while ``created_at`` does not decrease as ``id`` increases within a
        session, which is true of everything that writes history (both come from
        one INSERT) and is not true by construction. So it is checked, not
        assumed: a slice that starts before what the stored row already claims is
        not folded at all — that session is derived exactly, from its live rows
        up to this chunk's end. The same branch catches a stored timestamp that
        cannot be read back as one, since a value that will not parse cannot be
        merged with anything.
        """
        partial = project_transcript(rows, expect=session_id)
        if not partial:
            # Empty means the slice did not group into the session its
            # ``session_id`` column names — the row's metadata put it somewhere
            # else. Returning None here would only make the caller FORGET this
            # session, leaving whichever session the reader actually attributes
            # the row to standing at its old count, under a watermark recording
            # the chunk as accounted for. What is wrong is not confined to this
            # session, so neither is the repair: hand the whole transcript over,
            # which is the same escalation the monotonicity check below uses and
            # for the same reason (round-8 review).
            raise _NeedsTranscript(session_id)
        # The rows handed here are this session's slice of the chunk, with the
        # neighbours that split its clusters already removed by :meth:`_fold`.
        # A slice grouped in isolation yields ONE cluster whose boundaries are
        # positional — first and last in id order — while the grouper would have
        # split it on the intervening row and coalesced the pieces by min/max.
        # Those agree only while created_at rises with id, which PostgreSQL does
        # not guarantee: NOW() is transaction-start time, so an overlapping
        # writer can commit a later id carrying an earlier timestamp. Left alone
        # it stores last_message_at EARLIER than started_at — an incoherent row,
        # on the column the conversation list is ordered by.
        #
        # Checked on the rows rather than inferred from the stored row, because
        # the very first build has no stored row to compare against and is
        # exactly where this was first seen.
        # Through the canonical parser, never as raw strings. SQLite history
        # legitimately mixes the SQL spelling with the ISO one, and "T" sorts
        # AFTER a space — so `2020-01-01T09:00` compares greater than
        # `2020-01-01 10:00` and a genuine decrease reads as an increase. A
        # guard that misses the case it exists for is worse than none, because
        # the row it then stores claims to be current. A stamp that will not
        # parse at all, or is absent, is not evidence of order either: escalate
        # rather than guess (and rather than raise TypeError on None).
        stamps = []
        for row in rows:
            stamp = coerce_session_timestamp(row[3])
            if stamp is None:
                raise _NeedsTranscript(session_id)
            stamps.append(stamp)
        if any(later < earlier for earlier, later in zip(stamps, stamps[1:])):
            raise _NeedsTranscript(session_id)
        stored = await self.get(session_id)
        if stored is None:
            return partial[0]
        previous = _stored_projection(stored)
        if previous is None or previous.last_message_at > partial[0].started_at:
            raise _NeedsTranscript(session_id)
        merged = coalesce_sessions_by_session_id(
            [_as_grouped(previous), _as_grouped(partial[0])]
        )
        return _as_projection(merged[0])

    async def _derive(
        self, session_id: str, through: int
    ) -> Optional[SessionProjection]:
        """One session as its own live rows up to ``through`` describe it.

        The definition a fold is an optimization of, and the branch taken where
        the optimization would not provably agree with it. Bounded by the
        session's size rather than by the chunk, which is why it is not the
        ordinary path.
        """
        projections = project_transcript(
            await self.db.fetchall(
                _own_rows_through(self.db.backend_type),
                (self.agent_id, session_id, through),
            ),
            expect=session_id,
        )
        return projections[0] if projections else None

    async def _forget(self, session_id: str) -> int:
        """Drop one session's row."""
        return await self.db.execute(
            "DELETE FROM conversation_sessions "
            "WHERE agent_id = ? AND session_id = ?",
            (self.agent_id, session_id),
        )

    async def _store(self, projection: SessionProjection) -> int:
        """Upsert one session's row.

        PostgreSQL takes the parameters' types from the INSERT's target columns,
        so the ``TIMESTAMP`` pair needs no cast.
        """
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
                # TIMESTAMP column and SQLite wants the text. One shared adapter,
                # the same one the session queries bind through.
                self._timestamp(projection.started_at),
                self._timestamp(projection.last_message_at),
                projection.message_count,
                projection.user_message_count,
                projection.first_user_message_id,
                projection.wake_source,
            ),
        )
        return 1

    async def _record(self, watermark: SessionWatermark) -> None:
        """Record what the projection now accounts for.

        Unconditional, and that is the design rather than an oversight. Every
        caller holds this agent's repair lock and wrote its rows in the same
        transaction, so what stands is always a consistent pair. The numbers can
        still move *backwards*, because :meth:`_rebuild_from_transcript` reads
        its stamp before the unbounded derivation it cannot hold a lock across,
        and may reach the lock after somebody else has advanced past it.
        Backwards costs a redo and is the safe direction; forwards past what was
        accounted for may never happen, and no step can write it, because every
        number here was computed from state that pass read before it derived.

        A guard forbidding the backwards move was considered and refused: it
        would have to refuse that step's *rows* as well, or leave them standing
        under somebody else's advanced watermark — which is the exact silent gap
        this contract exists to close.
        """
        await self.db.execute(
            "INSERT INTO conversation_session_watermarks "
            f"(agent_id, {', '.join(WATERMARK_COLUMNS)}) "
            "VALUES (?, ?, ?, ?, ?) "
            "ON CONFLICT (agent_id) DO UPDATE SET "
            + ", ".join(
                f"{column} = excluded.{column}" for column in WATERMARK_COLUMNS
            ),
            (self.agent_id, *watermark.as_params()),
        )

    def _timestamp(self, value: str) -> Any:
        return timestamp_query_param(getattr(self.db, "backend_type", ""), value)


def _as_dict(row: Sequence[Any]) -> Dict[str, Any]:
    return {
        "session_id": row[0],
        **{column: row[index + 1] for index, column in enumerate(PROJECTION_COLUMNS)},
    }


def _stored_projection(row: Dict[str, Any]) -> Optional[SessionProjection]:
    """A stored row read back as a projection, or ``None`` if it cannot be.

    The timestamps come back as whatever the engine holds — a ``datetime`` from
    PostgreSQL, the ISO text it was given from SQLite — and a fold has to compare
    them against the grouper's ISO output, so both sides are normalized to one
    spelling of one instant here rather than at each comparison.

    ``None`` means the stored pair cannot be read as instants at all, which is a
    row nothing can safely be merged into. :meth:`_folded` answers that by
    deriving the session exactly instead, so the unreadable value is replaced
    rather than propagated.
    """
    started = coerce_session_timestamp(row["started_at"])
    last = coerce_session_timestamp(row["last_message_at"])
    if started is None or last is None:
        return None
    pointer = row["first_user_message_id"]
    return SessionProjection(
        session_id=str(row["session_id"]),
        started_at=started.isoformat(),
        last_message_at=last.isoformat(),
        message_count=int(row["message_count"]),
        user_message_count=int(row["user_message_count"]),
        first_user_message_id=None if pointer is None else int(pointer),
        wake_source=row["wake_source"],
    )
