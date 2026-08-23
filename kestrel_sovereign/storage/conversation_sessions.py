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

What Phase C (#2960) settled
----------------------------

:meth:`repair` now runs in production: ``list_session_page`` calls it before
serving the first page of the active conversation list. Two things this design
named as dormant became live with it, and the answer to both was the one this
section originally guessed at — **stop projecting**, rather than purge harder.

* **A repair can republish purged state.** :meth:`_rebuild_from_transcript`
  reads history OUTSIDE its transaction, deliberately, because that read is
  unbounded. A pass that began before an EPHEMERAL purge could therefore take
  the lock afterwards and write a pre-purge snapshot, restoring leaked counts,
  timestamps and pointers after the purge reported success.
* **The sweep would grow back to three tables**, with the questions that come
  with it: what evidence a retry uses, and whether the deletes must be atomic
  with each other.

Neither is reachable now, because ``PrivacyEnforcingStorage.list_session_page``
returns before any repair while EPHEMERAL is in force. A projection that is
never maintained under that mode has nothing to erase and nothing to republish,
so the sweep stays as narrow as it is — the change ledger, which is the only
table a trigger can fill on its own.

The mode is read per request rather than latched at boot, so a mode that is
turned on mid-life stops the projection at the next read; what an earlier,
non-EPHEMERAL life projected is what the sweep already erases.

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
all: the whole transcript is derived again, because which cluster a row belongs
to is not a property of its own session's rows — a neighbour splits them, and
coalescing merges the pieces. The transcript pass is the *definition*, and the
fold is the optimization of it that is only taken where it provably agrees.

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

import base64
import hashlib
import json
import logging
import re
from datetime import datetime
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .session_grouping import (
    SESSION_ORDER,
    SESSION_ORDER_TEXT_COLUMNS,
    canonical_timestamp_sql,
    coalesce_sessions_by_session_id,
    coerce_session_timestamp,
    group_messages_into_sessions,
    session_cursor_clause,
    parse_message_metadata,
    session_cursor_values,
    session_order_sql,
    timestamp_query_param,
)
from .conversation_ids import coerce_persistent_message_id
from .session_id_column import (
    SESSION_ID_KEY,
    SESSION_ID_MAX_LENGTH,
    is_stampable_session_id,
    is_storable_session_id,
)

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
#: How many times this agent's watermark has been written, ever.
#:
#: Not one of :data:`WATERMARK_COLUMNS`, because those are what a WALK decided
#: and this is how many decisions there have been. It is the one column
#: ``_record`` does not take from the value it is given.
#:
#: It exists because the watermark's own fields can return to a previous value.
#: ``rebuild()`` invalidates, discards every row, and walks again — and over an
#: unchanged history it arrives at a watermark identical to the one it replaced,
#: field for field. A page read during that walk sees a partial table, and a
#: before/after comparison of the watermark alone then certifies it: same
#: generation, same stamp, same target, so the truncated page is returned with
#: ``next_cursor: null``. ABA, in a cache whose whole contract is "may be silent,
#: may never disagree".
#:
#: Monotonic per agent and bumped by the same statement that writes the
#: watermark, so no pass can move one without the other.
WATERMARK_REVISION_COLUMN = "accounted_revision"

#: Which INCARNATION of this agent's watermark row the counter belongs to.
#:
#: The counter alone is monotonic only while the ROW lives, and the row does not
#: always live: ``purge_session_projection`` deletes it, and so does the
#: empty-history branch of the transcript pass. The next repair then INSERTs a
#: fresh one starting from the default, so two different incarnations can show
#: the same count — measured: revision 0, delete, repair, revision 0, with the
#: ledger generation unchanged because the ledger was never touched. A page read
#: straddling that would compare equal and be certified.
#:
#: Set when the row is created and never updated, exactly as the ledger's own
#: ``generation`` is, and for the same reason: counters restart, so something
#: has to say which run of them you are looking at.
WATERMARK_EPOCH_COLUMN = "accounted_epoch"

WATERMARK_COLUMNS: Tuple[str, ...] = (
    "accounted_generation",
    "accounted_valid",
    "accounted_stamp",
    "accounted_appends",
    "accounted_through",
    "accounted_target",
)

#: How large a whole-transcript derivation has to get before it is worth saying.
#:
#: Below this the pass is a few milliseconds and happens on databases that have
#: barely any history; above it the cost is user-visible on a read path, and
#: the operator should hear about it from a log rather than from a slow pane.
#: Measured at roughly 8.5 microseconds a live row, so this is the ~100 ms mark.
TRANSCRIPT_PASS_NOISY_ROWS = 10_000

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


def active_history_predicate() -> str:
    """What the conversation list's DEFAULT view selects — live, not archived.

    Identical text to :func:`live_history_predicate` today and separate on
    purpose: one names what a PROJECTION walk describes, the other what a READ
    filters to, and a partial index is only used when its predicate matches its
    query's. Collapsing them would tie the two together at the first change.
    """
    return _LIVE


def archived_history_predicate() -> str:
    """...and what its archived view selects (#2149), which is not a subset."""
    return "deleted_at IS NULL AND archived_at IS NOT NULL"


def live_history_predicate() -> str:
    """:data:`_LIVE`, for callers that must index exactly what the walk selects.

    A partial index is only used when its predicate matches the query's, so the
    two are the same string rather than two that agree today.
    """
    return _LIVE

#: The columns a derivation reads. ``content`` is deliberately absent — the
#: derivation never needs it (see the module docstring), and reading ciphertext
#: bodies to maintain an index would be a cost paid on every repair.
_DERIVED_FROM = "id, role, metadata, created_at, session_id"

#: Index of the ordering key every derivation selects alongside its columns.
_ORDER_KEY = 5

#: Index of ``created_at`` within :data:`_DERIVED_FROM`. Named because the fold
#: has to ask the GROUPER's parser about the same stamp the ordering key was
#: built from, and a bare ``row[3]`` beside a named ``row[_ORDER_KEY]`` reads
#: like the two are unrelated.
_CREATED_AT = 3


def _derived_from(backend_type: str) -> str:
    """The columns a derivation reads, plus the key its ORDER BY sorted on.

    The key travels WITH the row on purpose. The fold has to decide whether id
    order agrees with canonical order, and re-deriving the answer in Python
    means two implementations of "what is this timestamp" — which are not the
    same function. Measured: ``coerce_session_timestamp`` reads basic ISO
    (``20260101T110000``) and SQLite's ``julianday`` returns NULL for it, so the
    canonical read sorted that row FIRST while the guard parsed it, saw the
    stamps rising with id, and folded — storing boundaries and a preview the
    reader would never show, under a watermark saying current.

    Selecting the key removes the second implementation rather than trying to
    keep the two in step. Whatever SQL ordered by is what the fold reasons
    about, including its NULLs.
    """
    return f"{_DERIVED_FROM}, {_canonical_key(backend_type, 'created_at')}"

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


def canonical_order_index_columns(backend_type: str) -> str:
    """The index that makes :func:`canonical_order` a bounded traversal.

    Generated from the same key expressions the ``ORDER BY`` is, because an
    index that does not match the ordering it exists for is not a slow index —
    it is no index. Measured on 200,000 rows, ordering by the canonical keys
    without it:

    ==========  ==================================  =========
    backend     plan                                page of 1000
    ==========  ==================================  =========
    SQLite      ``USE TEMP B-TREE FOR ORDER BY``    227 ms
    SQLite      matching expression index            0.8 ms
    PostgreSQL  ``Sort`` (top-N heapsort)            22 ms
    PostgreSQL  ``Index Scan Backward``              0.2 ms
    ==========  ==================================  =========

    Both are O(history): the engine reads and sorts the agent's whole live
    history before applying ``LIMIT``. That is the cost this epic exists to
    remove, so reintroducing it on the list's own read path would have undone
    the point of the work while every test still passed.

    **One index serves both directions.** ``canonical_order()``'s ascending and
    descending forms are exact reverses — that is why the NULLs are placed
    explicitly — so the reader's newest-first page is a backward scan of the
    same index the derivation walks forward. Ascending order is therefore what
    is stored, with NULLs first, matching the ascending form exactly.
    """
    keys = ", ".join(
        _canonical_key(backend_type, column) for column in CANONICAL_ORDER_COLUMNS
    )
    nulls = " NULLS FIRST" if backend_type == "postgres" else ""
    first, _, rest = keys.partition(", ")
    return f"agent_id, {first} ASC{nulls}, {rest} ASC"


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




#: The longest token :func:`encode_session_cursor` can produce, in characters.
#:
#: Derived, not chosen. The endpoint has to accept every token it hands out —
#: a `next_cursor` its own parameter then refuses makes every session after
#: that page boundary unreachable, which is this ticket's bug wearing a 422 —
#: so the bound is computed from what a token can contain rather than set to a
#: round number that looked large enough.
#:
#: Worst case, in bytes of JSON before base64:
#:
#: * the fixed envelope and the timestamp key: under 96 bytes, generously.
#: * the session id: up to ``SESSION_ID_MAX_LENGTH`` bytes, and JSON escaping
#:   is what makes that not the answer. A quote or a backslash doubles, and an
#:   ASCII control character becomes ``\u00XX`` — six bytes for one. Six is
#:   therefore the multiplier, and it is reachable: the storability rule admits
#:   any text PostgreSQL can hold, which includes control characters.
#:
#: base64 is 4 characters per 3 bytes, and the padding is stripped.
_CURSOR_ENVELOPE_BYTES = 96
_JSON_WORST_ESCAPE = 6
SESSION_CURSOR_MAX_LENGTH = (
    ((_CURSOR_ENVELOPE_BYTES + _JSON_WORST_ESCAPE * SESSION_ID_MAX_LENGTH) + 2)
    // 3
) * 4

#: The wire format of a page cursor. Stamped into every token and checked on
#: the way back in, so a token minted by an older shape is refused rather than
#: read as though its fields meant what they mean now.
_CURSOR_VERSION = 1


class SessionCursorError(ValueError):
    """A page cursor that this build cannot read.

    Its own type rather than a bare ``ValueError`` because the caller has to
    tell "the client sent nonsense" (answerable with a 400) from "the database
    failed" (a 500). A cursor is client-supplied text, so this is reachable by
    anyone with the endpoint, and the two must not be confused.
    """


#: How a cursor names a position, and there are two because there are two ways
#: a page is produced.
#:
#: ``keyset`` carries the ordering's keys and is what the projection pages by:
#: it has an index and a stable key, and an offset there would count rows that
#: move between two requests.
#:
#: ``offset`` carries a count, and is what the grouped paths page by — the
#: archived view and ISOLATED's in-memory buffer. Those have no table: they
#: derive and order the WHOLE set inside one request and then slice it, so a
#: position in that slice is exactly what an offset is. A keyset there would
#: have to carry the boundary session's id, which nothing bounds — a 4,000
#: character id mints a 5,408 character token, past the length the endpoint's
#: own parameter accepts, so the server would hand back a `next_cursor` it then
#: refuses with 422 and every later session would be unreachable. Measured.
#:
#: An offset re-derives against a set that may have changed between requests,
#: which a keyset would not. That is the honest trade for these paths and no
#: worse than what they already are: the set is re-derived per request either
#: way, so it has always been a snapshot.
_CURSOR_KEYSET = "keyset"
_CURSOR_OFFSET = "offset"


def encode_offset_cursor(offset: int, view: str) -> str:
    """A position in a materialized, re-derived page sequence (#2960)."""
    return _encode_cursor(_CURSOR_OFFSET, view, int(offset))


def decode_offset_cursor(token: str, view: str) -> int:
    """An offset token back into a count, refusing anything else."""
    value = _decode_cursor(_CURSOR_OFFSET, token, view)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise SessionCursorError("cursor's offset is not a count")
    return value


def _encode_cursor(kind: str, view: str, keys: Any) -> str:
    return base64.urlsafe_b64encode(
        json.dumps(
            {"v": _CURSOR_VERSION, "kind": kind, "view": view, "k": keys},
            separators=(",", ":"),
            # UTF-8 rather than ASCII escapes. A size choice, not a guard —
            # what keeps a token inside the parameter that takes it is
            # `SESSION_CURSOR_MAX_LENGTH`, which is derived from the escaped
            # worst case and holds either way. This just stops an ordinary
            # non-ASCII session id from costing twelve characters per emoji.
            ensure_ascii=False,
        ).encode("utf-8")
    ).decode("ascii").rstrip("=")


def _decode_cursor(kind: str, token: str, view: str) -> Any:
    """The shared envelope checks: readable, this version, this kind, this view.

    ``kind`` is checked rather than dispatched on, so a token minted for one
    paging model can never be read by the other — the two mean different things
    by the same field, and a caller silently accepting the wrong one resumes at
    a position that was never its own.
    """
    if not isinstance(token, str) or not token:
        raise SessionCursorError("cursor is empty")
    padded = token + "=" * (-len(token) % 4)
    try:
        payload = json.loads(base64.urlsafe_b64decode(padded.encode("ascii")))
    except Exception as exc:
        raise SessionCursorError("cursor is not readable") from exc
    if not isinstance(payload, dict) or payload.get("v") != _CURSOR_VERSION:
        raise SessionCursorError("cursor was minted by a different version")
    if payload.get("kind") != kind:
        raise SessionCursorError(
            f"cursor names a {payload.get('kind')!r} position, not {kind!r}"
        )
    if payload.get("view") != view:
        raise SessionCursorError(
            f"cursor belongs to the {payload.get('view')!r} view, not {view!r}"
        )
    return payload.get("k")


def encode_session_cursor(session: Dict[str, Any], view: str) -> str:
    """One page's last row, spelled so the next request can resume from it.

    Opaque on purpose: the fields inside are :data:`SESSION_ORDER`'s keys, which
    is an ordering decision this epic has already changed twice. A client that
    could read the token would come to depend on that shape, and the tie-break
    is exactly the part a caller must not pin.

    ``view`` travels with it because the views are served by different
    machinery — the active list pages the #2959 projection in SQL, the archived
    one pages the grouper's output in Python — and a token minted by one and
    replayed against the other would resume from a key the other never ordered
    by. Refusing that is a 400; silently serving page one for it is a list that
    lies about where the user was.

    The keys go through :func:`session_cursor_values`, which is the same
    spelling the Python-side continuation compares in, so a cursor means one
    thing on every path that can honour it.
    """
    return _encode_cursor(
        _CURSOR_KEYSET, view, list(session_cursor_values(session))
    )


def decode_session_cursor(token: str, view: str) -> Tuple[Any, ...]:
    """A token back into :data:`SESSION_ORDER` values, comparably spelled.

    Every failure is one exception type. A client can put anything in this
    parameter, and the two alternatives are the ones that hurt: returning
    ``None`` for an unreadable token silently serves page one to a caller that
    asked for page nine, and letting a ``binascii`` error escape reports a
    client's typo as a server fault.

    Backend-free by construction. What comes back is text (or ``None``), and the
    caller that runs a SQL page binds it for its own engine — so this codec
    stays the one thing both the SQL and the Python continuation can share.
    """
    values = _decode_cursor(_CURSOR_KEYSET, token, view)
    if not isinstance(values, list) or len(values) != len(SESSION_ORDER):
        raise SessionCursorError("cursor does not carry this ordering's keys")
    for (column, _), value in zip(SESSION_ORDER, values):
        # NULL is not a value this ordering can hold. Both keys are NOT NULL in
        # the projection's schema and the grouper substitutes a stamp rather
        # than omitting one, so a null key is a token no server minted — and
        # letting it through is worse than refusing on every path: the Python
        # continuation compares a string against ``None`` and raises
        # ``TypeError`` (a 500), while the SQL one compares against NULL and
        # quietly serves an empty page.
        # Each key against what its COLUMN is, and that is the whole check:
        # both of these refuse every non-string, ``None`` included, so a
        # separate "is it text" guard ahead of them decided nothing. Verified
        # rather than assumed — a mutation removing it survived, which is what
        # that always means.
        #
        # Text alone would not be enough for the id either: a NUL or a lone
        # surrogate is a string Python holds happily and the drivers refuse,
        # and the refusal comes out of the QUERY rather than out of the cursor
        # check — a 500 for client-supplied input. So it is asked the same
        # bound the projection stores keys under.
        if column in SESSION_ORDER_TEXT_COLUMNS and not is_storable_session_id(value):
            raise SessionCursorError(
                f"cursor's {column} is not a value this store can hold"
            )
        # Each key is checked against what its COLUMN is, not merely against
        # being a string. A token is client-supplied, and a timestamp key that
        # cannot be read is not a cursor this build cannot honour — it is one
        # that reaches asyncpg as a ``TIMESTAMP`` parameter and raises out of
        # the query, past the handler that turns a bad cursor into a 400 and
        # into the one that reports a server fault. On SQLite it is worse than
        # an error: ``'not-a-date'`` compares as text against canonical stamps
        # and simply selects the wrong page.
        if (
            column not in SESSION_ORDER_TEXT_COLUMNS
            and coerce_session_timestamp(value) is None
        ):
            raise SessionCursorError(f"cursor's {column} is not a timestamp")
    return tuple(values)


#: What one chunk selects: this agent's next live rows, and only those. The rows
#: a step reads and the rows it folds are the same rows, which is what makes the
#: chunk a bound on work rather than only on ids. Seeded by
#: ``idx_conversation_agent_row_id``.
def _chunk_sql(backend_type: str) -> str:
    """One chunk: this agent's next live rows, and only those."""
    return (
        f"SELECT {_derived_from(backend_type)} FROM conversation_history "
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
        f"SELECT {_derived_from(backend_type)} FROM conversation_history "
        "WHERE agent_id = ? AND session_id = ? AND id <= ? "
        f"AND {_LIVE} "
        f"{canonical_order(backend_type)}"
    )


def _live_rows_through(backend_type: str) -> str:
    """Every live row of this agent's up to one frontier, for the transcript
    pass — the same columns, in the order a reader would see them, so unstamped
    rows are attributed the way the grouper attributes them."""
    return (
        f"SELECT {_derived_from(backend_type)} FROM conversation_history "
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

#: The projection columns that may never hold NULL, and the reason is the
#: PAGE (#2960) rather than tidiness.
#:
#: Every writer already guarantees it: the grouper produces ``isoformat()`` text
#: for both stamps — substituting the preceding row's instant for an undatable
#: row rather than leaving one out — and ``timestamp_query_param`` never returns
#: ``None``. The column merely failed to say so, and what that cost was
#: measured: a keyset predicate has to admit NULL to keep a NULL-stamped session
#: reachable at all, and that ``OR ... IS NULL`` is what stops the engine
#: seeking. On 200,000 sessions, SQLite: 0.11 ms with a seekable predicate,
#: 20.7 ms without — the same ``O(rows above the cursor)`` walk this epic exists
#: to remove, arriving back on the continuation instead of on the first page.
#:
#: So the choice was between a slow page and a column that states the invariant.
#: An invariant a writer keeps and a schema does not is one an ``UPDATE`` run by
#: hand can break, and the failure is silent: the session simply stops appearing.
NON_NULL_PROJECTION_COLUMNS: Tuple[str, ...] = ("started_at", "last_message_at")

_CONVERSATION_SESSIONS_DDL = """
CREATE TABLE IF NOT EXISTS conversation_sessions (
    agent_id              TEXT NOT NULL,
    session_id            TEXT NOT NULL,
    started_at            TIMESTAMP NOT NULL,
    last_message_at       TIMESTAMP NOT NULL,
    message_count         INTEGER NOT NULL DEFAULT 0,
    user_message_count    INTEGER NOT NULL DEFAULT 0,
    first_user_message_id INTEGER,
    wake_source           TEXT,
    PRIMARY KEY (agent_id, session_id)
)
"""

_WATERMARKS_DDL = """
CREATE TABLE IF NOT EXISTS conversation_session_watermarks (
    agent_id             TEXT PRIMARY KEY,
    accounted_epoch      TEXT NOT NULL DEFAULT '',
    accounted_revision   BIGINT NOT NULL DEFAULT 0,
    accounted_generation TEXT NOT NULL DEFAULT '',
    accounted_valid   INTEGER NOT NULL DEFAULT 0,
    accounted_stamp   BIGINT NOT NULL DEFAULT 0,
    accounted_appends BIGINT NOT NULL DEFAULT 0,
    accounted_through BIGINT NOT NULL DEFAULT 0,
    accounted_target  BIGINT NOT NULL DEFAULT 0
)
"""

_CHANGES_DDL = """
CREATE TABLE IF NOT EXISTS conversation_history_changes (
    agent_id   TEXT NOT NULL,
    slot       BIGINT NOT NULL DEFAULT 0,
    changes    BIGINT NOT NULL DEFAULT 0,
    appends    BIGINT NOT NULL DEFAULT 0,
    generation TEXT NOT NULL DEFAULT '',
    PRIMARY KEY (agent_id, slot)
)
"""

#: Which row of the ledger a writer adds to (#3005).
#:
#: One row per agent made every write transaction touching that agent's history
#: queue behind every other, for the length of the longest — the upsert takes a
#: row lock held to COMMIT, so a purge or a restore stalled ordinary appends.
#: Measured: two connections upserting one row block; two upserting different
#: rows do not.
#:
#: ``pg_backend_pid()`` makes non-collision a GUARANTEE rather than a
#: probability, which a modulus would not. A backend can only be inside one
#: transaction at a time, so no two CONCURRENT transactions can ever address
#: the same slot, and a lock on one can never be waited on by another. Slots
#: accumulate as pool connections churn; they are tiny, and folding them is
#: :meth:`repair`'s business once #2960 gives the projection a reader.
#:
#: SQLite has one writer by construction, so it has nothing to shard and stays
#: on slot 0.
def _slot(backend_type: str) -> str:
    if backend_type == "postgres":
        return "pg_backend_pid()"
    return "0"


#: Slot 0 is not a writer's slot; it is where the agent's ``generation`` lives.
#:
#: The generation identifies an INCARNATION of the ledger and so must be one
#: value per agent, which a sharded counter has nowhere to put. Reserving a
#: slot no ``pg_backend_pid()`` can return (pids start at 1) gives it a home
#: without a second table.
#:
#: ``DO NOTHING`` is what keeps that home from becoming the contention the
#: sharding just removed: measured, an ``ON CONFLICT DO NOTHING`` against an
#: already-committed row takes no lock a second writer waits on. It DOES wait
#: when the row is being created in-flight by another transaction — so an
#: agent's very first two concurrent writes can still serialize, once, ever.
#: That is measured rather than assumed, and it is the honest bound.
_ANCHOR_COLUMNS = "(agent_id, slot, changes, appends, generation)"

#: The upsert a PER-ROW trigger performs, parameterized by which row's
#: ``agent_id`` is being stamped and by whether the event APPENDED a row.
#: Written once so SQLite's three triggers and PostgreSQL's UPDATE function
#: cannot count differently. :func:`_statement_bump` is its per-statement
#: counterpart and must add the same total.
#:
#: Two counters, because one cannot answer the question :meth:`_plan` asks.
#: ``changes`` counts every row event; ``appends`` counts only inserts. A walk
#: may continue from where it stopped exactly when every change since was a row
#: arriving above its target — and "a row arrived" is indistinguishable from "an
#: existing row was renumbered above the target" if the only evidence is that
#: the counter moved by one and one row now stands up there. Both are one event
#: and one row. They differ in that the renumbering is not an append, so
#: counting appends separately is what separates them (#3001).

#: Every trigger and function installed by this module is named
#: ``<prefix><role>_<fingerprint>``, and these are the two prefixes. They are
#: what lets an initializer enumerate the mechanism a database is *currently*
#: carrying — not merely ask whether one expected name is present — so that a
#: superseded shape is found and retired rather than left running (#2998).
TRIGGER_NAME_PREFIX = "conversation_history_change_"
FUNCTION_NAME_PREFIX = "kestrel_conversation_history_"

#: What a name may contain once assembled. These are interpolated into DDL,
#: which no parameter can carry, so the guarantee is asserted rather than
#: assumed — the parts are this module's own constants and a hex digest, and
#: this is the check that keeps that true if a role is ever renamed.
_SAFE_NAME = re.compile(r"\A[a-z][a-z0-9_]*\Z")


def _new_generation(backend_type: str) -> str:
    """SQL for a value no other incarnation of this ledger row will hold.

    Set once, when the row is created, and never touched by the upserts that
    follow. It is what makes a stamp comparable to the ledger it was read from
    and to no other — the counters restart at 1 when the privacy sweep erases
    the row, so on their own they say nothing across that boundary, which has
    now produced the same defect in three separate mechanisms (the sweep, the
    publication fence, and the fence's own revalidation).

    Randomness rather than a sequence, because a sequence would have to live
    somewhere that survives the deletion — which is the thing privacy forbids.
    """
    if backend_type == "postgres":
        return "gen_random_uuid()::text"
    return "hex(randomblob(16))"

_BUMP = (
    "INSERT INTO conversation_history_changes "
    "(agent_id, slot, changes, appends) "
    "VALUES ({row}.agent_id, {slot}, 1, {appends}) "
    "ON CONFLICT (agent_id, slot) DO UPDATE "
    "SET changes = conversation_history_changes.changes + 1, "
    "appends = conversation_history_changes.appends + {appends}"
)

#: Creates the agent's slot-0 row and its generation, once, and never touches
#: either again. Runs BEFORE the bump in every trigger body: on SQLite the bump
#: also addresses slot 0, so a bump landing first would create the row with an
#: empty generation that this could no longer fill in.
def _anchor(generation: str, *, row: str = "", agents: str = "", where: str = "") -> str:
    """Create the agent's slot-0 row, generation and all, if it is not there.

    ``row`` names a per-row trigger's ``NEW``/``OLD``; ``agents`` is a subquery
    yielding one ``agent_id`` per agent for a per-statement trigger. Exactly one
    is given.

    The de-duplication in ``agents`` has to happen in a SUBQUERY rather than as
    ``SELECT DISTINCT`` beside the constants: the generation expression is
    volatile, so every projected row would differ and ``DISTINCT`` would dedupe
    nothing — one row per row written, and one uuid generated per row of a bulk
    insert.
    """
    values = (
        f"VALUES ({row}.agent_id, 0, 0, 0, {generation})" if row
        else f"SELECT agent_id, 0, 0, 0, {generation} FROM ({agents}) AS kestrel_agents"
    )
    if where:
        # The conditional form has to be a SELECT: SQLite has no way to put a
        # predicate on VALUES, and this is its own answer to "an upsert I want
        # to skip" (the shape the rehoming stamp already uses).
        values = f"SELECT {row}.agent_id, 0, 0, 0, {generation} {where}"
    return (
        "INSERT INTO conversation_history_changes "
        + _ANCHOR_COLUMNS
        + " "
        + values
        + " ON CONFLICT (agent_id, slot) DO NOTHING"
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


def _placeholder(kind: str, role: str) -> str:
    """The brace name a template uses to refer to one object of the mechanism."""
    return ("fn_" if kind == "function" else "trg_") + role


def _fingerprint(backend_type: str, templates: Sequence[str]) -> str:
    """A short digest of the whole mechanism's DDL, names excluded.

    Names are excluded because they are derived FROM this — the templates carry
    ``{fn_appended}``-style placeholders and are resolved afterwards, which is
    what stops the definition from depending on its own digest.
    """
    material = "\n".join((backend_type, *templates)).encode("utf-8")
    return hashlib.blake2s(material, digest_size=4).hexdigest()


def _statement_bump(
    source: str, appends: str, generation: str, slot: str
) -> str:
    """One ledger row per agent for a whole statement, counting rows touched.

    ``source`` is a transition table. ``count(*)`` per ``agent_id`` is exactly
    what N row-level bumps would have added, so the arithmetic
    :meth:`ConversationSessionProjection._plan` performs is unchanged — this is
    the same number arrived at in one statement instead of N.

    ``GROUP BY`` is not only for the multi-agent case: it is what makes the
    ``ON CONFLICT`` legal. PostgreSQL refuses an upsert whose source offers the
    same conflict key twice ("cannot affect row a second time"), and a bulk
    insert for one agent offers it once per row.
    """
    return (
        _anchor(generation, agents=f"SELECT DISTINCT agent_id FROM {source}")
        + "; "
        + "INSERT INTO conversation_history_changes "
        "(agent_id, slot, changes, appends) "
        f"SELECT agent_id, {slot}, count(*), {appends} "
        f"FROM {source} GROUP BY agent_id "
        "ON CONFLICT (agent_id, slot) DO UPDATE "
        "SET changes = conversation_history_changes.changes + EXCLUDED.changes, "
        "appends = conversation_history_changes.appends + EXCLUDED.appends"
    )


def _watched_projection(backend_type: str) -> str:
    """:data:`PROJECTION_INPUT_COLUMNS` as a select list, metadata narrowed.

    Built from the constant rather than spelled out, so a column added to what
    the projection reads is a column this compares — the alternative is two
    lists that agree until one is edited.

    Unqualified, because each side of the comparison selects from one
    transition table at a time.
    """
    return ", ".join(
        watched_metadata_sql(backend_type, column) if column == "metadata"
        else column
        for column in PROJECTION_INPUT_COLUMNS
    )


def _update_statement_bump(
    before: str, after: str, backend_type: str, generation: str
) -> str:
    """One ledger update for a whole UPDATE statement, counting as rows do.

    A transition table does not PAIR its rows — ``OLD`` and ``NEW`` arrive as
    two unordered tuplestores — so "which rows changed a watched column" cannot
    be asked of them the way a per-row trigger asks it, and ``id``, the only
    candidate join key, is itself watched precisely because it can be rewritten.
    Two comparisons get there without a pairing, and together they add exactly
    what the per-row form adds:

    1. **What changed**, credited to the agent that now holds it: the multiset
       difference ``after EXCEPT ALL before`` over the watched tuple. One row
       per row whose watched columns moved, and — this is the property that
       keeps ordinary recall free — empty when they did not.
       ``atomic_increment_metadata_counter`` rewrites the metadata document on
       every retrieval, and comparing only the watched keys inside it is what
       makes that write invisible here.

    2. **What departed**, credited to the agent that lost it: per agent, how
       many rows it held before minus how many it holds now, when positive. A
       same-agent update leaves those counts equal and contributes nothing; a
       row changing hands leaves the old agent one short.

    Worked against the per-row form, which stamps ``NEW.agent_id`` once per
    changed row and ``OLD.agent_id`` once more only when the row changed hands.
    Two rows of agent A edited in place and a third moved from A to B: (1)
    gives A two and B one, (2) gives A one; total A three, B one — which is
    what three per-row stamps on A and one on B come to. The invariant that
    each row event moves the stamp by exactly one survives the conversion,
    which matters because it is what lets a repair prove a movement was
    entirely appends.

    ``EXCEPT ALL`` compares NULLs as equal, the same null-safe semantics the
    per-row form spells ``IS DISTINCT FROM``.

    The outer ``GROUP BY`` is not decoration: the two comparisons can each name
    the same agent, and PostgreSQL refuses an upsert whose source offers one
    conflict key twice. Summing them first is also what makes the pair behave
    as one addition rather than two.
    """
    watched = _watched_projection(backend_type)
    changed = (
        f"SELECT agent_id, count(*) AS kestrel_n FROM "
        f"(SELECT {watched} FROM {after} "
        f"EXCEPT ALL SELECT {watched} FROM {before}) AS kestrel_changed "
        "GROUP BY agent_id"
    )
    departed = (
        "SELECT agent_id, kestrel_lost AS kestrel_n FROM ("
        "SELECT agent_id, kestrel_was - COALESCE(kestrel_is, 0) AS kestrel_lost "
        f"FROM (SELECT agent_id, count(*) AS kestrel_was FROM {before} "
        "GROUP BY agent_id) AS kestrel_old "
        f"LEFT JOIN (SELECT agent_id, count(*) AS kestrel_is FROM {after} "
        "GROUP BY agent_id) AS kestrel_new USING (agent_id)"
        ") AS kestrel_net WHERE kestrel_lost > 0"
    )
    return (
        _anchor(
            generation,
            agents=(
                f"SELECT agent_id FROM {before} "
                f"UNION SELECT agent_id FROM {after}"
            ),
        )
        + "; "
        + "INSERT INTO conversation_history_changes "
        "(agent_id, slot, changes, appends) "
        f"SELECT agent_id, {_slot(backend_type)}, SUM(kestrel_n)::bigint, 0 "
        f"FROM ({changed} UNION ALL {departed}) AS kestrel_moved "
        "GROUP BY agent_id "
        "ON CONFLICT (agent_id, slot) DO UPDATE "
        "SET changes = conversation_history_changes.changes + EXCLUDED.changes, "
        "appends = conversation_history_changes.appends + EXCLUDED.appends"
    )


def _pg_function(placeholder: str, body: str) -> str:
    """A trigger function around ``body``, named by ``placeholder``."""
    return (
        "CREATE OR REPLACE FUNCTION {"
        + placeholder
        + "}() RETURNS trigger AS $kestrel$ BEGIN "
        + body
        + "; RETURN NULL; END; $kestrel$ LANGUAGE plpgsql"
    )


def _mechanism_templates(backend_type: str) -> Tuple[Tuple[str, str, str], ...]:
    """``(kind, role, DDL-with-name-placeholders)``, functions before triggers.

    **INSERT and DELETE are statement-level on PostgreSQL; UPDATE is not.**
    That split is measured rather than stylistic. A row-level trigger that
    upserts a single counter row is a read-modify-write per affected row,
    holding that row's lock until COMMIT, and on PostgreSQL 16 with 50,000 rows
    for one agent it cost:

    ==============  ============  ===========
    statement       triggers off  triggers on
    ==============  ============  ===========
    bulk ``INSERT``       793 ms    11,334 ms
    bulk ``DELETE``        17 ms    10,769 ms
    ==============  ============  ===========

    — 14x and 633x, plus 50,617 dead tuples in a table that holds one row per
    agent. Those are live write paths the moment this projection ships:
    ``purge_all_since``, the full-history restore in ``sovereign_adapter``, the
    #2012 relink, the encryption backfill.

    INSERT and DELETE convert exactly, because ``count(*)`` over the transition
    table is arithmetically the same number N row-level bumps produce. UPDATE
    converts too, by the two comparisons :func:`_update_statement_bump`
    describes — a transition table does not PAIR its rows, but the pairing was
    never what the arithmetic needed.

    **UPDATE was left per-row at first, on a premise that was wrong.** The
    claim was that the bulk updates this repo runs touch nothing the projection
    reads, so the ``WHEN`` clause a per-row trigger carries would keep them
    free. That is true of the REWRITE paths it was checked against — the
    encryption backfill, the #1402 canonical/transport split,
    ``atomic_increment_metadata_counter`` on every recall — and false of the
    LIFECYCLE ones, which were not:

    * ``clear_history`` sets ``deleted_at`` on **every live message in one
      unbounded statement**;
    * ``delete_conversation_session``, ``archive_conversation_session`` and
      their restores set ``deleted_at``/``archived_at`` across a whole session.

    Those are precisely the watched columns. Measured on 50,000 rows, the
    ``clear_history`` shape cost **12,655 ms** per-row against **906 ms**
    per-statement, with 50,000 dead tuples against none — the same 14x, on a
    path a user reaches from a button.

    What the conversion costs in exchange is that a statement-level trigger
    cannot carry a ``WHEN`` clause, so the transition tables are built even for
    an update touching nothing watched. Measured on the same 50,000 rows, a
    bulk rewrite of ``content`` went from 1,005 ms to 1,289 ms and the counter
    correctly did not move. That is 28% on the rare case to save 93% on the
    common one.

    SQLite keeps all three row-level: it has no transition tables, so there is
    nothing to convert to. Its ``WHEN`` clause therefore still does the
    narrowing, and its bulk lifecycle updates still pay per row — cheaply, on
    an engine with one writer and no dead tuples to leave behind. **The two engines therefore no longer share one
    trigger shape**, and that is worth saying plainly rather than leaving to be
    discovered. What they do still share is the arithmetic — the ledger moves by
    one per row event on both — and that, not the DDL, is what
    :meth:`ConversationSessionProjection._plan` depends on.

    What this costs instead
    -----------------------

    A transition table is a tuplestore of WHOLE tuples, built for the duration
    of the statement and spilling past ``work_mem``. Measured on the same
    PostgreSQL 16, 200,000 rows of roughly 1 KB each inserted and then deleted
    in one statement apiece: **412 MB across 3 temp files**, and the delete
    itself 316 ms. PostgreSQL offers no way to narrow it to the one column this
    reads.

    That is a real price and it is the right one to pay here. It is transient —
    released when the statement ends — where the row-level form's 50,617 dead
    tuples were not, and it is disk where the row-level form's cost was a row
    lock held to COMMIT that every other writer for that agent queued behind.

    It is also bounded in practice by the callers. ``_purge_conversation_rows``
    already batches its deletes at 500 ids per statement, so every purge path
    is bounded by construction. The exception is the full-history restore in
    ``sovereign_adapter``, which deletes an agent's whole history in one
    statement — and that path already holds the entire history in Python memory
    to reinsert it, so this adds a second O(history) allocation to a path that
    had one, in the cheaper medium, while making the statement itself ~35x
    faster.

    What the ledger used to cost, and why it is sharded
    ---------------------------------------------------

    One row per agent, upserted inside the writing transaction, held that row's
    lock to COMMIT — so every write transaction touching one agent's history
    was serialized against every other for the length of the longest. Short
    autocommit appends never noticed. The long ones did:
    ``_purge_conversation_rows`` wraps its snapshot, its deletes and its
    lexical cleanup in one transaction, and the restore wraps a whole history,
    so an ordinary append for that agent waited on either.

    That was never inherent to the per-statement conversion — the per-row form
    took the same lock earlier and held it longer — but it WAS inherent to a
    single counter row, which is why the row is not single any more.
    :func:`_slot` gives each writer its own, keyed on ``pg_backend_pid()``, and
    the choice of key is the whole argument: a backend can only be inside one
    transaction at a time, so two CONCURRENT transactions can never address the
    same slot. A modulus would have made collision unlikely instead of
    impossible, and an intermittent stall is worse to diagnose than a reliable
    one.

    Measured, two connections, the second appending while the first holds a
    transaction that wrote the same agent's history: **blocked** before,
    **not blocked** after. One exception, measured rather than reasoned about:
    an agent's first-ever write creates its slot-0 anchor, and a second writer
    arriving while that INSERT is still in flight does wait for it. Once per
    agent per incarnation of the ledger.

    None of it was ever a deadlock, and that rests on lock ORDER rather than
    luck: every path writes history before it takes the lexical advisory lock,
    never the reverse, so there was no cycle to detect.
    ``test_concurrent_final_shared_key_purges_reclaim_tokens`` is where this
    lives — it found the serialization by deadlocking its own harness against
    it, and now asserts that both parties reach the key boundary, so it fails
    again if anything upstream starts serializing them.
    """
    generation = _new_generation(backend_type)
    if backend_type == "postgres":
        return (
            (
                "function",
                "appended",
                _pg_function(
                    "fn_appended",
                    _statement_bump(
                        "kestrel_appended", "count(*)", generation,
                        _slot(backend_type),
                    ),
                ),
            ),
            (
                "function",
                "updated",
                _pg_function(
                    "fn_updated",
                    _update_statement_bump(
                        "kestrel_before", "kestrel_after",
                        backend_type, generation,
                    ),
                ),
            ),
            (
                "function",
                "removed",
                _pg_function(
                    "fn_removed",
                    _statement_bump(
                        "kestrel_removed", "0", generation,
                        _slot(backend_type),
                    ),
                ),
            ),
            (
                "trigger",
                "insert",
                "CREATE TRIGGER {trg_insert} "
                "AFTER INSERT ON conversation_history "
                "REFERENCING NEW TABLE AS kestrel_appended "
                "FOR EACH STATEMENT EXECUTE FUNCTION {fn_appended}()",
            ),
            (
                "trigger",
                "update",
                "CREATE TRIGGER {trg_update} "
                "AFTER UPDATE ON conversation_history "
                "REFERENCING OLD TABLE AS kestrel_before "
                "NEW TABLE AS kestrel_after "
                "FOR EACH STATEMENT EXECUTE FUNCTION {fn_updated}()",
            ),
            (
                "trigger",
                "delete",
                "CREATE TRIGGER {trg_delete} "
                "AFTER DELETE ON conversation_history "
                "REFERENCING OLD TABLE AS kestrel_removed "
                "FOR EACH STATEMENT EXECUTE FUNCTION {fn_removed}()",
            ),
        )
    # SQLite spells null-safe inequality ``IS NOT`` and has no trigger
    # functions, so each body carries the upsert directly. The conditional
    # second stamp is an ``INSERT ... SELECT ... WHERE``: SQLite's own answer to
    # "an upsert I want to skip", and the form that keeps the ON CONFLICT clause
    # unambiguous to its parser.
    watched_is_not = _watched_changed(backend_type, "IS NOT")
    slot = _slot(backend_type)

    def anchor(row: str, only_when: str = "") -> str:
        return _anchor(generation, row=row, where=only_when)

    rehomed = (
        "INSERT INTO conversation_history_changes "
        "(agent_id, slot, changes, appends) "
        f"SELECT OLD.agent_id, {slot}, 1, 0 "
        "WHERE OLD.agent_id IS NOT NEW.agent_id "
        "ON CONFLICT (agent_id, slot) DO UPDATE "
        "SET changes = conversation_history_changes.changes + 1"
    )
    departed = " WHERE OLD.agent_id IS NOT NEW.agent_id"
    return (
        (
            "trigger",
            "insert",
            "CREATE TRIGGER {trg_insert} "
            "AFTER INSERT ON conversation_history FOR EACH ROW BEGIN "
            f"{anchor('NEW')}; "
            f"{_BUMP.format(row='NEW', appends=1, slot=slot)}; END",
        ),
        (
            "trigger",
            "update",
            "CREATE TRIGGER {trg_update} "
            "AFTER UPDATE ON conversation_history FOR EACH ROW "
            f"WHEN ({watched_is_not}) BEGIN "
            f"{anchor('NEW')}; "
            f"{_BUMP.format(row='NEW', appends=0, slot=slot)}; "
            f"{anchor('OLD', departed)}; {rehomed}; END",
        ),
        (
            "trigger",
            "delete",
            "CREATE TRIGGER {trg_delete} "
            "AFTER DELETE ON conversation_history FOR EACH ROW BEGIN "
            f"{anchor('OLD')}; "
            f"{_BUMP.format(row='OLD', appends=0, slot=slot)}; END",
        ),
    )


def _mechanism(backend_type: str) -> Tuple[Tuple[str, str, str, str], ...]:
    """``(kind, role, name, DDL)`` for the whole mechanism, names resolved.

    Every name ends in :func:`_fingerprint` of the mechanism it belongs to, so
    the objects installed in a database ARE their definition and a name probe
    answers the question it was always meant to ask (#2998).

    The fingerprint covers the functions as well as the triggers, and covers all
    of them together. A PostgreSQL trigger's behaviour is its function's body:
    editing :func:`watched_metadata_sql` leaves every ``CREATE TRIGGER``
    character-for-character identical, and something has to notice. Fingerprinting
    the mechanism as a whole makes one edit rename all six, which is coarser than
    strictly necessary and is the point — the objects are installed and retired
    as a set, so there is no state in which half of one shape is live beside half
    of another.
    """
    templates = _mechanism_templates(backend_type)
    stamp = _fingerprint(backend_type, [ddl for _kind, _role, ddl in templates])
    names = {
        _placeholder(kind, role): (
            (FUNCTION_NAME_PREFIX if kind == "function" else TRIGGER_NAME_PREFIX)
            + role
            + "_"
            + stamp
        )
        for kind, role, _ddl in templates
    }
    for name in names.values():
        if not _SAFE_NAME.match(name):
            raise ValueError(f"unusable SQL identifier for the change stamp: {name!r}")
    return tuple(
        (kind, role, names[_placeholder(kind, role)], ddl.format(**names))
        for kind, role, ddl in templates
    )


def mutation_trigger_functions(backend_type: str) -> Tuple[Tuple[str, str], ...]:
    """``(name, DDL)`` of every PL/pgSQL function the triggers call.

    Empty on SQLite, which has no separate function object — its trigger bodies
    are the statements themselves.

    ``CREATE OR REPLACE`` rather than a probe, and the replace cannot surprise a
    live trigger: the name carries the shape, so a changed body is a changed
    name and the statement targets a function nothing points at yet. Retiring
    the superseded one is the sweep's job, and the sweep runs after the triggers
    that used it are gone. What this must not do is run *unlocked* — two
    concurrent initializers creating one function collide on ``pg_proc``'s
    unique index, and the loser's whole ``from_pool()`` raises.
    """
    return tuple(
        (name, ddl)
        for kind, _role, name, ddl in _mechanism(backend_type)
        if kind == "function"
    )


def mutation_triggers(backend_type: str) -> Tuple[Tuple[str, str], ...]:
    """``(name, DDL)`` for the change stamp, in this engine's dialect.

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

    Which of these are per-row and which are per-statement is
    :func:`_mechanism_templates`' subject.
    """
    return tuple(
        (name, ddl)
        for kind, _role, name, ddl in _mechanism(backend_type)
        if kind == "trigger"
    )


def shape_change_invalidation(backend_type: str) -> str:
    """SQL retiring every ledger counter, for when the TRIGGERS change.

    The trigger shape is the definition of "a change". Widen it and every event
    that happened under the narrower one was, by the new definition, never
    recorded — the counter did not move for it, and the watermark that matched
    the counter still matches. So the projection reports itself current while
    holding an answer the grouper would not give, which is the single outcome
    this design exists to make impossible. Reachable on one host with one
    revision: it is what an ordinary upgrade does.

    Rotating the generation is not a new mechanism; it is the one already
    written for exactly this. :meth:`ConversationSessionProjection.is_stale` and
    :meth:`_plan` both treat a changed generation as "these counters belong to a
    different incarnation, so equality across them would be an accident", and
    both answer REBUILD. Invalidating the WATERMARKS instead would have been the
    more obvious edit and a worse one: a repair already in flight across the
    migration would go on to publish, having decided what to write under the old
    definition. The publish fence re-reads the generation, so rotating it stops
    that repair too.

    Every row, in one statement. Both engines evaluate their generator per row —
    ``gen_random_uuid()`` and ``randomblob`` are volatile — so the agents do not
    all land on one value and become comparable to each other.
    """
    return (
        "UPDATE conversation_history_changes SET generation = "
        + _new_generation(backend_type)
        # Slot 0 only: it is the one row that carries a generation, and the
        # writers' slots hold the empty string by construction (#3005).
        + " WHERE slot = 0"
    )


def emptied_cache_invalidation() -> str:
    """SQL retiring every watermark, for when the CACHE has been emptied.

    The companion to :func:`shape_change_invalidation`, and it is needed because
    that one can update nothing. Rotating the generation retires the counters a
    watermark was compared against — but only if a counter row exists. An agent
    whose projection was built before the triggers were installed, or restored
    without its ledger, has a watermark and no slot-0 row: its generation is the
    empty string and its stamp is zero, which is exactly what a MISSING ledger
    reads back as. The numbers agree, ``is_stale()`` answers false, and the
    freshly emptied cache is served for ever beside intact history.

    So the claim is made where it is true. The cache is empty, therefore nothing
    is accounted for, therefore no watermark is valid — which is a statement
    about the watermarks and is written to them.

    Both are run, not one: this catches the missing-ledger case, and the
    generation rotation catches a repair already IN FLIGHT, whose publish fence
    re-reads the generation and would otherwise commit a walk it decided on
    before the table went.
    """
    return "UPDATE conversation_session_watermarks SET accounted_valid = 0"


def mutation_trigger(backend_type: str, role: str) -> Tuple[str, str]:
    """One trigger's ``(name, DDL)`` by role — ``insert``/``update``/``delete``.

    Callers that want a particular trigger ask for it by the job it does, since
    the name is a derived value now and spelling it out would embed a
    fingerprint that changes whenever the mechanism does.
    """
    for kind, this_role, name, ddl in _mechanism(backend_type):
        if kind == "trigger" and this_role == role:
            return (name, ddl)
    raise KeyError(f"no {role} trigger for backend {backend_type}")


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

    generation: str = ""
    valid: bool = False
    stamp: int = 0
    appends: int = 0
    through: int = 0
    target: int = 0
    #: See :data:`WATERMARK_REVISION_COLUMN`. Read back, never supplied: the
    #: writer increments it, so a value constructed in Python carries 0 and
    #: only what came from the database carries the real count.
    revision: int = 0
    #: See :data:`WATERMARK_EPOCH_COLUMN`. Read back, never supplied.
    epoch: str = ""

    @property
    def fence(self) -> Tuple[str, int]:
        """What a reader compares before and after to know nothing moved.

        The pair, never either half. The revision cannot repeat while the row
        lives; the epoch cannot repeat across rows. Alone, each has a case it
        cannot see — a rebuild landing where it started, and a watermark
        deleted and recreated — and both of those return a page read from a
        half-built cache as the end of the list.
        """
        return (self.epoch, self.revision)

    @property
    def complete(self) -> bool:
        """Whether the walk that recorded this reached its target."""
        return self.through >= self.target

    def as_params(self) -> Tuple[int, ...]:
        """This watermark in ``WATERMARK_COLUMNS`` order.

        ``valid`` is bound as an ``int`` because the column is ``INTEGER`` on
        both engines; asyncpg refuses a Python ``bool`` for one.
        """
        return (
            self.generation, int(self.valid), self.stamp, self.appends,
            self.through, self.target,
        )


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
    generation: str
    stamp: int
    appends: int
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


def project_transcript(
    rows: Sequence[Sequence[Any]], expect: Optional[str] = None
) -> List[SessionProjection]:
    """Project every session the grouper finds in ``rows`` and the column may key.

    ``rows`` are ``(id, role, metadata, created_at, session_id, ...)`` ordered by id
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
    # Trailing columns are ignored here: the derivation also selects the key
    # its ORDER BY sorted on (see :func:`_derived_from`), which the fold
    # reads and the grouper has no use for.
    for row_id, role, metadata, created_at, column in (row[:5] for row in rows):
        stamped[row_id] = column
        messages.append(
            {
                "id": row_id,
                "role": role,
                # The row's identity standing in for its text, so the preview
                # picker answers WHICH row rather than what it said. See the
                # module docstring for why that is faithful.
                "content": str(row_id),
                "metadata": parse_message_metadata(metadata),
                "created_at": created_at,
            }
        )

    # ``keep_empty_markers`` so a conversation that exists only as its
    # ``new_session`` marker is still a session (#2222) — the UI prepends a tile
    # for it the moment the user starts typing, and a projection that dropped it
    # could not serve that reader. A reader wanting only sessions with traffic
    # filters on ``message_count``.
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

    # Every value this transcript's METADATA files a row under, which is what
    # decides whether a reader can open a session — not the indexed column.
    # ``AsyncConversationStore._get_session_messages`` resolves a session two
    # ways and neither of them is the column: as a row id, or by matching
    # ``metadata LIKE '%"session_id": "<value>"%'``. Phase A's column is a
    # derived duplicate for INDEXING, and asking it "can this be opened" was a
    # proxy for a question it does not answer.
    #
    # Collected with the grouper's own acceptance rule, because this has to be
    # the set of keys the grouper could have produced from metadata: it files a
    # row under ``metadata.session_id`` when that value is truthy and not
    # ``str(...).isdigit()``, and a value it would ignore is not a key it can
    # have chosen.
    #
    # Taken across all rows rather than per session, because a session
    # established by a `new_session` marker alone (#2222) carries no messages —
    # the marker is structural and excluded from them — while its own row does
    # carry the id. A per-session test would have silently stopped projecting
    # exactly the just-started conversations `keep_empty_markers` exists to keep.
    metadata_keys_seen = set()
    for message in messages:
        candidate = message["metadata"].get(SESSION_ID_KEY)
        if candidate and not str(candidate).isdigit():
            metadata_keys_seen.add(str(candidate))

    projections: List[SessionProjection] = []
    for session in grouped:
        session_id = str(session["session_id"])
        # The requirement is that a READER CAN OPEN this key. The grouper mints
        # a key exactly two ways, and each is opened by a different half of the
        # resolver:
        #
        # 1. From ``metadata.session_id``, which the resolver matches on
        #    directly. Charset does not enter into it — ``rasa_shim`` files
        #    every SMS turn under ``sms:{sender}``, which Phase A's column may
        #    not hold and the resolver opens without difficulty.
        # 2. Invented from the first row's id, for a legacy cluster carrying no
        #    session id anywhere (#2012), which the row-id half opens — and
        #    only if it is a usable row id, which is what #3001's ``"-1"`` case
        #    was about: SQLite allows a negative rowid, and a key of ``"-1"``
        #    is refused by ``coerce_persistent_message_id``, so neither half of
        #    the resolver would find it.
        #
        # Phase B asked ``is_stampable_session_id`` here instead, and that was a
        # proxy twice over: it is a question about the COLUMN, and the column
        # resolves nothing. It dropped both the row-id keys and every metadata
        # key outside the column's charset. That cost nothing while nothing read
        # this table; #2960 makes it THE conversation list, so a dropped session
        # is a conversation that has vanished — this ticket's own bug, arriving
        # by another route. Measured: an `sms:` session listed as absent, and
        # 473 of Emma's 1,522 live rows carry no session id at all.
        #
        # Storing them is sound because the two derivations cannot meet over
        # one. A key outside the column's charset is a key no row's column can
        # hold, so every row of that session is unstamped, so
        # `_has_unstamped_rows()` is true and every step takes the transcript
        # pass — the chunked fold, which is keyed on the column and could not
        # maintain such a row, never runs while one can exist. Leaving that
        # state cannot be reached by appends alone (removing an unstamped row is
        # a delete, an archive or a rewrite), so `_plan` answers REBUILT and the
        # discard clears anything left behind.
        if (
            session_id not in metadata_keys_seen
            and coerce_persistent_message_id(session_id) is None
        ):
            continue
        # ...and this table's own primary key has to be able to hold it, which
        # is a second question and not the same one. Openable says a reader can
        # find the session; storable says PostgreSQL can put the key in a
        # composite B-tree — no NUL, encodable, within the length bound. A key
        # past one of those does not lose a session quietly: `_store` raises,
        # inside the repair that runs on the first page of every conversation
        # list, so the WHOLE list fails for that agent until the row is edited
        # by hand. Refusing one session and saying so is the smaller harm, and
        # it is said out loud because a dropped session is now a conversation
        # missing from the UI.
        if not is_storable_session_id(session_id):
            logger.warning(
                "conversation_sessions: session %r cannot be stored as a "
                "projection key (%d bytes, NUL or unencodable); it will not "
                "appear in the conversation list",
                session_id[:64],
                len(session_id.encode("utf-8", errors="replace")),
            )
            continue
        # Asked of every session, whichever way its key was arrived at: a row
        # whose column names a different session is in the wrong place under
        # either derivation, and for an invented key ANY non-NULL column is that.
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
            "SELECT COALESCE(SUM(changes), 0) FROM conversation_history_changes "
            "WHERE agent_id = ?",
            (self.agent_id,),
        )
        return int(value or 0)

    async def observed_appends(self) -> int:
        """How many of this agent's row events were INSERTs.

        The other half of the pair :meth:`_plan` needs. Read separately rather
        than folded into :meth:`observed_changes` so that method keeps meaning
        exactly one thing — "has anything moved" — which is all
        :meth:`is_stale` asks.
        """
        value = await self.db.fetchval(
            "SELECT COALESCE(SUM(appends), 0) FROM conversation_history_changes "
            "WHERE agent_id = ?",
            (self.agent_id,),
        )
        return int(value or 0)

    async def observed_generation(self) -> str:
        """Which incarnation of the ledger row the counters belong to.

        Empty when there is no row. A stamp read from one incarnation is not
        comparable to another's, and the counters cannot say so themselves —
        they restart at 1, so they can arrive back at any earlier value.
        """
        value = await self.db.fetchval(
            "SELECT generation FROM conversation_history_changes "
            "WHERE agent_id = ? AND slot = 0",
            (self.agent_id,),
        )
        return str(value or "")

    async def accounted(self) -> SessionWatermark:
        """The state the projection records having accounted for.

        A missing row reads as :data:`INVALID` rather than as an error: an agent
        whose projection has never been repaired has accounted for nothing, and
        *invalid* is the honest name for that — it must derive, not compare.
        """
        row = await self.db.fetchone(
            f"SELECT {', '.join(WATERMARK_COLUMNS)}, {WATERMARK_REVISION_COLUMN}, "
            f"{WATERMARK_EPOCH_COLUMN} "
            "FROM conversation_session_watermarks WHERE agent_id = ?",
            (self.agent_id,),
        )
        if row is None:
            return INVALID
        return SessionWatermark(
            str(row[0]), bool(row[1]), int(row[2]), int(row[3]), int(row[4]),
            int(row[5]), int(row[6]), str(row[7] or ""),
        )

    async def is_stale(self) -> bool:
        """Whether the projection disagrees with the rows it describes.

        Three primary-key reads, and no scan of history at any size — which is
        the whole reason the stamp is maintained by the engine. ``False`` is the
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
        if accounted.generation != await self.observed_generation():
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
        ``inserts_above - deletes_above - moves_out + moves_in``, no row having
        stood above it when the stamp was read (``target`` was ``MAX(id)`` at
        that instant), where a *move* is an ``UPDATE`` that rewrote a row's
        ``id``.

        Two numbers cannot separate those. One append and one move are each one
        change leaving one row above the target, and folding the second counts a
        row this projection has already counted. So a third is read: ``appends``,
        which only the INSERT trigger raises. Requiring
        ``delta == appended == rows_above`` makes the test false whenever
        anything at or below the target moved, whenever anything was rewritten
        rather than added, and whenever an append landed below the target —
        which together are exactly the cases where continuing the walk would be
        unsound (#3001).

        Nothing in this codebase rewrites an id; maintenance and import SQL is
        the traffic this defends against, and it is precisely the traffic that
        never passes through a write path that could have invalidated anything
        on its way.

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
        # Claimed, so the wait is bounded. `_record` writes the watermark row,
        # which another PostgreSQL repair may be holding — and waiting on that
        # lock BEFORE `_claim()` has set `lock_timeout` is an unbounded wait,
        # which is the shape of Phase A's 23-hour wedge and the one thing this
        # class promises never to do.
        async with self.db.transaction(immediate=True):
            await self._claim()
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

    async def page(
        self, *, limit: int, after: Optional[Sequence[Any]] = None
    ) -> List[Dict[str, Any]]:
        """One page of this agent's sessions, newest activity first (#2960).

        ``after`` is the previous page's last row in :data:`SESSION_ORDER`, as
        :func:`decode_session_cursor` returns it; ``None`` starts at the top.

        This is the read the epic exists for. The list it replaces fetched a
        fixed window of ``conversation_history`` and hoped the sessions it
        wanted fell inside — measured on Emma, 34% of her conversations did not,
        and no ``limit`` could reach them because the window was a constant. Here
        the bound is the page, the continuation is a key rather than an offset,
        and every session is reachable by asking again.

        ``limit`` is applied in SQL, not in Python. Trimming a longer read would
        make the cost of page one a function of how much history exists, which
        is the property this table was measured into existence to remove.
        """
        where = ["agent_id = ?"]
        params: List[Any] = [self.agent_id]
        if after is not None:
            clause, cursor_params = session_cursor_clause(
                self.db.backend_type, self._bound_cursor(after)
            )
            where.append(clause)
            params.extend(cursor_params)
        rows = await self.db.fetchall(
            f"SELECT session_id, {', '.join(PROJECTION_COLUMNS)} "
            "FROM conversation_sessions "
            f"WHERE {' AND '.join(where)} "
            f"{session_order_sql(self.db.backend_type)} LIMIT ?",
            (*params, max(1, int(limit))),
        )
        return [_as_dict(row) for row in rows]

    def _bound_cursor(self, after: Sequence[Any]) -> Tuple[Any, ...]:
        """A decoded cursor's keys, typed for the engine holding the columns.

        The codec is backend-free — it hands back text — and the timestamp
        column is a real ``timestamp`` on PostgreSQL and canonical text on
        SQLite. Binding through the same adapter :meth:`_store` wrote the column
        with is what makes the comparison one the index can use, rather than a
        text-to-timestamp coercion the planner performs per row.
        """
        return tuple(
            value
            if column in SESSION_ORDER_TEXT_COLUMNS or value is None
            else timestamp_query_param(self.db.backend_type, value)
            for (column, _), value in zip(SESSION_ORDER, after)
        )

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

    async def claim_exclusion(self) -> bool:
        """Take this agent's repair exclusion, for a caller that is not a repair.

        The EPHEMERAL sweep needs it. Ordering its deletes to match a repair's
        writes is not enough on PostgreSQL: a `DELETE` matching no rows locks
        nothing, so an agent with a ledger row but no watermark yet — the first
        repair, which is also the rebuild-everything one — can have that repair
        insert a watermark AFTER the sweep's delete has passed over it, and the
        sweep reports success with a row naming the agent still standing
        (round-18 review).

        :meth:`_claim` already solves exactly that for repairs racing each
        other, by inserting the row before locking it so there is something to
        lock. Sharing it is the fix; a second mechanism would be the thing this
        module keeps having to undo.

        Returns whether the claim CREATED the watermark row, which the sweep
        needs: it deletes that row a statement later and reports how many it
        removed, and a row this claim made to have something to lock is not a
        leak.
        """
        self._claim_created_the_row = False
        await self._claim()
        return self._claim_created_the_row

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
        # ONE statement that locks in both branches, and whose result is
        # checked. The previous pair — `INSERT ... DO NOTHING` then
        # `SELECT ... FOR UPDATE` — locks nothing when the row already exists
        # (the insert is a no-op) and can then find nothing to lock: under READ
        # COMMITTED the waiter re-evaluates after the holder commits, and two
        # shipped call sites DELETE this row while holding it — the empty
        # history branch of `_rebuild_from_transcript` and the EPHEMERAL sweep.
        # The waiter woke with zero rows and no lock, silently, because nothing
        # inspected the result. Measured: `INSERT 0 0` then `FOR UPDATE`
        # returning 0 rows.
        #
        # `DO UPDATE` takes the row lock on the conflict path, and PostgreSQL
        # retries the insert when the conflicting row is deleted underneath it,
        # so this acquires in both orders. Measured against the same race: the
        # upsert returns its row.
        # ``xmax = 0`` is true only on the INSERT path of an upsert, so this
        # says whether the row was CREATED here — which the caller needs when
        # it is about to delete it and count what it removed. A row this claim
        # made to have something to lock is not a leak.
        acquired = await self.db.fetchval(
            "INSERT INTO conversation_session_watermarks (agent_id) VALUES (?) "
            "ON CONFLICT (agent_id) DO UPDATE SET agent_id = EXCLUDED.agent_id "
            "RETURNING (xmax = 0)",
            (self.agent_id,),
        )
        self._claim_created_the_row = bool(acquired)
        if acquired is None:
            # Never observed, and not swallowed if it happens: a claim that did
            # not acquire lets a sweep and a repair write the same tables at
            # once, which is the one forbidden state this module names.
            raise RuntimeError(
                "conversation_sessions: could not claim the repair exclusion "
                f"for {self.agent_id}; refusing to proceed unserialized"
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
        appends = await self.observed_appends()
        generation = await self.observed_generation()
        if not accounted.valid or accounted.generation != generation:
            return _Plan(
                REBUILT, generation, observed, appends, await self._id_floor(),
                await self._max_id(), True
            )
        if observed == accounted.stamp:
            if accounted.complete:
                return None
            return _Plan(
                INCREMENTAL,
                generation,
                accounted.stamp,
                accounted.appends,
                accounted.through,
                accounted.target,
                False,
            )
        delta = observed - accounted.stamp
        appended = appends - accounted.appends
        # Every change since must have been an append, AND each of those appends
        # must be standing above the target. Three numbers rather than two,
        # because two cannot tell a row ARRIVING above the target from an
        # existing row RENUMBERED above it: both are one change and one row up
        # there, and folding the second one counts a row this projection has
        # already counted (#3001). Only the first is also an append.
        if (
            delta > 0
            and delta == appended
            and delta == await self._rows_above(accounted.target)
        ):
            return _Plan(
                INCREMENTAL,
                generation,
                observed,
                appends,
                accounted.through,
                await self._max_id(),
                False,
            )
        return _Plan(
            REBUILT, generation, observed, appends, await self._id_floor(),
                await self._max_id(), True
        )

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
            _chunk_sql(self.db.backend_type),
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
            SessionWatermark(
                plan.generation, True, plan.stamp, plan.appends, through,
                plan.target,
            )
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
        appends = await self.observed_appends()
        generation = await self.observed_generation()
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
        transcript = await self.db.fetchall(
            _live_rows_through(self.db.backend_type),
            (self.agent_id, target),
        )
        # Said out loud, because #2960 put this pass on the conversation list's
        # read path and its cost is the agent's whole live history. One live row
        # with a NULL `session_id` is enough to choose it, and legacy history is
        # full of them (Emma: 473 of 1,522), so for such an agent EVERY repair
        # after a turn re-derives everything. Measured, SQLite, one session in
        # three legacy: 143 ms at 15,000 live rows, 524 ms at 60,000, 1,020 ms
        # at 120,000 — linear, about 8.5 microseconds a row.
        #
        # A silent cost that grows with history is the shape this epic exists to
        # remove, so it is logged rather than left to be discovered from a slow
        # pane. #3061 is the fix: stamp the legacy rows once and this pass
        # stops being chosen at all.
        if len(transcript) >= TRANSCRIPT_PASS_NOISY_ROWS:
            logger.warning(
                "conversation_sessions: %s has live rows carrying no session id, "
                "so this repair re-derived all %d of its live rows rather than "
                "folding a chunk (#3061). List latency for this agent grows "
                "with its history.",
                self.agent_id,
                len(transcript),
            )
        projections = project_transcript(transcript)

        written = 0
        async with self.db.transaction(immediate=True):
            await self._claim()
            # Revalidate before publishing. The derivation above ran OUTSIDE
            # this transaction — it has to, it is the unbounded pass — so
            # history can have been erased underneath it, and publishing then
            # puts back a description of messages that no longer exist. The
            # EPHEMERAL sweep is exactly that erasure, so "leave no trace" would
            # be undone by a repair that merely started first.
            #
            # Two questions, because one is not enough. The stamp catches every
            # ordinary mutation, but it cannot catch this one: a first
            # post-upgrade repair reads a stamp of 0 (the ledger is created
            # empty beside a history already full), the sweep then erases the
            # ledger, and a missing ledger also reads 0. Unchanged, by a route
            # that changed everything. So the second question is asked of
            # history itself, whose emptiness is the sweep's own precondition.
            if not await self._any_live_row():
                # No live history means the correct projection is the empty one,
                # and the ground is already cleared below. Publishing nothing
                # and recording nothing leaves the watermark invalid, so the
                # next repair derives again — from what is actually there.
                await self.db.execute(
                    "DELETE FROM conversation_sessions WHERE agent_id = ?",
                    (self.agent_id,),
                )
                # ...and the watermark row, which `_claim()` INSERTed a moment
                # ago on PostgreSQL to have something to lock.
                #
                # This is NOT "an empty history leaves no watermark". The
                # chunked path records a VALID one over an empty history on
                # purpose — `accounted_valid` exists to separate "nothing has
                # ever been accounted for" from "accounted for, and there was
                # nothing", and collapsing them makes an empty agent re-derive
                # for ever. The difference is what the two passes KNOW: the
                # chunked walk looked and found nothing, while this pass derived
                # a history that was erased underneath it and can vouch for
                # neither. Recording nothing is the honest answer to that, and
                # not recording the agent's id is a bonus rather than the rule. Leaving it
                # commits this agent's id into a table the purge had just
                # emptied — a no-trace exit undone by the repair that noticed
                # the purge (round-17 review). Nothing is lost: an absent
                # watermark is INVALID, which is what this branch means.
                await self.db.execute(
                    "DELETE FROM conversation_session_watermarks WHERE agent_id = ?",
                    (self.agent_id,),
                )
                return _Step(REBUILT, 0, True)
            delta = await self.observed_changes() - observed
            appended = await self.observed_appends() - appends
            if (
                await self.observed_generation() != generation
                or delta < 0
                or delta != appended
                or delta != await self._rows_above(target)
            ):
                # The SAME test `_plan` applies, for the same reason: every
                # change since the derivation must have been a row APPENDED
                # above the target. The snapshot is bounded at ``id <= target``
                # (above), so a row arriving beyond it cannot have changed what
                # was derived — those rows are simply left to the next fold.
                #
                # An earlier version refused whenever ``MAX(id)`` moved at all,
                # which is stricter than the invariant needs and does not
                # terminate: this pass is reached on EVERY repair for an agent
                # holding rows whose ``session_id`` the column cannot store, so
                # one message arriving during a whole-history read aborted the
                # publish, and `repair()` immediately tried again — up to the
                # step budget, each attempt another full scan holding SQLite's
                # writer slot.
                #
                # The generation is what covers a purge, which is why the
                # frontier no longer has to: a ledger erased and rebuilt gets a
                # new one, so a stamp from before it is not comparable at all.
                #
                # NOT COVERED BY A FAILING TEST, and said plainly rather than
                # left to look verified. The arithmetic above subtracts two
                # readings of the same counters, and after the sweep erases the
                # ledger the second reading belongs to a row that restarted at
                # zero — so the difference is across two unrelated sequences.
                # In the post-upgrade shape they line up exactly: measured,
                # `delta 3 == appended 3 == rows_above 3` describing a history
                # this snapshot never saw, admitted by every check but this one.
                #
                # What could not be observed is a DIFFERENT outcome. `repair()`
                # takes the mismatch through `_plan` on its next step and
                # rebuilds from current history, and `is_stale()` reports the
                # mismatch too, so the table converges either way — including
                # with `step_budget=1`. The claim this defends is narrower than
                # the end state: that rows derived from purged history are never
                # COMMITTED, not merely corrected afterwards. Relying on the
                # correction would make a privacy property depend on a later
                # step running at all.
                return _Step(REBUILT, 0, False)
            await self.db.execute(
                "DELETE FROM conversation_sessions WHERE agent_id = ?",
                (self.agent_id,),
            )
            for projection in projections:
                written += await self._store(projection)
            await self._record(
                SessionWatermark(
                    generation, True, observed, appends, target, target
                )
            )
        # Accounted through == target by construction: this pass derived every
        # live row, not a chunk of them. Reporting otherwise sends the caller
        # back for a step that has nothing left to do.
        return _Step(REBUILT, written, True)

    # ── internals ────────────────────────────────────────────────────────

    async def _any_live_row(self) -> bool:
        """Whether this agent has any live history at all. One index probe."""
        return bool(await self.db.fetchval(
            "SELECT 1 FROM conversation_history "
            f"WHERE agent_id = ? AND {_LIVE} LIMIT 1",
            (self.agent_id,),
        ))

    async def _id_floor(self) -> int:
        """The frontier a walk that has accounted for NOTHING starts from.

        Not zero. A walk selects ``id > through``, and while every writer here
        issues positive ids — ``AUTOINCREMENT`` and ``bigserial`` both do — the
        schema does not require it, so an imported or rewritten row numbered
        zero or less would sit below a zero frontier and never be folded at all,
        under a watermark recorded as complete (#3001).

        Read from the data rather than set to a type's minimum. A constant would
        have to know how wide the column is, and it is not the same width on
        both engines — ``conversation_history.id`` is ``int4`` on PostgreSQL, so
        the smallest 64-bit integer is not a value it can even be compared
        against. One backward index seek on the same index ``_max_id`` uses, and
        only on the rebuild path.

        An id of exactly the column's minimum would make this underflow and the
        repair would raise. That is the safe direction: a repair that fails
        loudly is a repair that has not recorded a false "current".
        """
        smallest = await self.db.fetchval(
            "SELECT MIN(id) FROM conversation_history WHERE agent_id = ?",
            (self.agent_id,),
        )
        return 0 if smallest is None else int(smallest) - 1

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
            # `_folded` returns a projection or raises `_NeedsTranscript` — it
            # has no "nothing to store" answer. It had one until the fold began
            # escalating a session whose rows the transcript files elsewhere,
            # and the branch that handled it (forgetting the session) outlived
            # the case by several rounds, looking like live handling of a
            # Phase A violation that in fact reached the transcript pass.
            written += await self._store(
                await self._folded(session_id, by_session[session_id], through)
            )
        return written

    async def _folded(
        self,
        session_id: str,
        rows: Sequence[Sequence[Any]],
        through: int,
    ) -> SessionProjection:
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
        not folded at all — it escalates, and the whole transcript is derived
        again. Deriving just this session would not do: another session's row
        between two of its own ENDS a cluster, so its boundaries are not a
        property of its own rows. The same branch catches a stored timestamp
        that cannot be read back as one, since a value that will not parse
        cannot be merged with anything.
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
        # The key SQL ordered by, selected with the row — not re-derived here.
        # Python and SQL do not read the same set of timestamps (basic ISO
        # parses in one and is NULL in the other), and a guard that answers
        # "does id order match canonical order" from the wrong one answers a
        # different question than the one that matters. A NULL key is a value
        # the canonical order sorts first and cannot compare: not evidence of
        # order, so escalate rather than guess.
        keys = [row[_ORDER_KEY] for row in rows]
        if any(key is None for key in keys):
            raise _NeedsTranscript(session_id)
        # ...and the other direction, which a non-NULL key cannot show.
        # ``julianday`` NORMALIZES a day-of-month overflow rather than
        # rejecting it: ``2023-02-29T12:00:00`` yields a real key pointing at
        # March 1, while ``fromisoformat`` — which is what the grouper reads
        # with — refuses it outright. Measured on sqlite 3.50.4, and
        # ``2023-04-31`` behaves the same; an impossible MONTH is refused by
        # both, so the gap is specific to the day.
        #
        # Such a row has an ordering key and no parsed stamp, so the fold takes
        # its undatable fallback — which looks at the predecessor in THIS
        # SESSION'S SLICE, where the grouper looks at the predecessor in the
        # whole history. Those differ exactly when another session's row falls
        # between, and the result is a stored ``last_message_at`` the grouper
        # would not produce, under a watermark reporting itself current.
        #
        # :data:`_JULIANDAY_READABLE` closed the direction where the PARSER was
        # wider than the key. This is the direction where the KEY is wider than
        # the parser, and one gate cannot answer both: the regex is a question
        # about syntax, and whether February has a 29th is a question about the
        # year.
        if any(coerce_session_timestamp(row[_CREATED_AT]) is None for row in rows):
            raise _NeedsTranscript(session_id)
        if any(later < earlier for earlier, later in zip(keys, keys[1:])):
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
            # Counted from the column list for the same reason `_record` does:
            # a literal run of "?" is a second statement of how many columns a
            # projection row has, and it breaks at runtime rather than being
            # carried along.
            f"VALUES ({', '.join('?' * (len(PROJECTION_COLUMNS) + 2))}) "
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
            f"(agent_id, {', '.join(WATERMARK_COLUMNS)}, {WATERMARK_EPOCH_COLUMN}) "
            # Placeholders counted from the column list, not written out. A
            # literal run of "?" is a second statement of how many columns a
            # watermark has, and adding one to WATERMARK_COLUMNS then fails at
            # runtime rather than being carried along — which is how adding
            # `accounted_appends` announced itself.
            f"VALUES ({', '.join('?' * (len(WATERMARK_COLUMNS) + 1))}, "
            f"{_new_generation(self.db.backend_type)}) "
            "ON CONFLICT (agent_id) DO UPDATE SET "
            + ", ".join(
                f"{column} = excluded.{column}" for column in WATERMARK_COLUMNS
            )
            # ...and the revision, which is the one column not taken from the
            # value being written. Its own clause because it counts WRITES
            # rather than recording a decision, and it is incremented by this
            # statement so nothing can move the watermark without moving it.
            + f", {WATERMARK_REVISION_COLUMN} = "
            f"conversation_session_watermarks.{WATERMARK_REVISION_COLUMN} + 1"
            # ...and the epoch, SET ONCE. Written on the conflict path as well
            # as the insert path, because on PostgreSQL the insert path is not
            # the one that runs: `_claim()` has already created the row as a
            # thing to lock, so this statement always conflicts and an
            # insert-only epoch stayed empty for every agent — the fence then
            # quietly degenerated to the revision alone, on the backend that
            # matters most. Measured, after a SQLite-only check said otherwise.
            #
            # A CASE rather than a plain assignment: the epoch names an
            # INCARNATION, so it must not change while the row lives, and
            # rewriting it on every write would make it a second revision
            # counter answering the question the first one already answers.
            # Set-once also gives the migrated rows their first value — they
            # arrive from the ALTER with the empty default.
            + f", {WATERMARK_EPOCH_COLUMN} = CASE WHEN "
            f"conversation_session_watermarks.{WATERMARK_EPOCH_COLUMN} = '' "
            f"THEN {_new_generation(self.db.backend_type)} ELSE "
            f"conversation_session_watermarks.{WATERMARK_EPOCH_COLUMN} END",
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
