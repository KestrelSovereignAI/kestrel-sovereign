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

The contract: a rebuildable cache with a change stamp
=====================================================

The first attempt at this table maintained it as a **correctness invariant** —
every write path refreshed it, and the refreshes were serialized against each
other. Two review rounds raised two P1s, both inside that mechanism: concurrent
refreshes overwriting the projection with stale state, and then a bulk deletion
bypassing the serialization added to fix the first. Repeated P1s in one
mechanism, with clean fixes, is the signature of the mechanism being wrong
rather than the fixes being sloppy. It obliged every write path that exists
*and every one added later* to remember, and when one forgot, the projection
lied silently.

So the obligation is removed instead:

1. **No write path maintains this table.** Inserts, soft-delete, restore,
   archive and purge (#2509/#2567) may all leave it stale. That is legal. Note
   the public surface below — there is deliberately no method a mutation could
   call to "keep it in step", because such a method is the thing that gets
   forgotten.
2. **Staleness is detected exactly, and cheaply.** The *database itself* counts
   every row event on ``conversation_history`` into a per-agent change stamp
   (see :func:`mutation_triggers`), and each repair records the stamp it worked
   from. The projection is current exactly when the stamp has not moved: two
   primary-key reads, no scan of history, and nothing a write path can forget
   because no write path is involved.
3. **Repair is incremental where it can be.** Appends are the common case
   (someone said something), and a repair may restrict itself to rows above the
   frontier exactly when the change stamp's movement is *fully explained* by the
   rows that appeared above it — see :meth:`ConversationSessionProjection.repair`
   for why that arithmetic is exact rather than hopeful.
4. **A full rebuild is always available.** Dropping every row and rebuilding
   from ``conversation_history`` yields the same table as any sequence of
   incremental repairs.

Why a database trigger and not a call in the write paths
========================================================

Because a call in the write paths is the mechanism that failed review twice. A
change stamp bumped from Python is a thing each mutation must remember, and
``AsyncConversationStore.update_message_metadata`` is proof that "each mutation"
is not a closed set: its key set is the *caller's*, so it can re-home a row from
one session to another, or flip ``operator_signal`` / ``signal_wake`` /
``new_session``, without touching liveness, counts or ``MAX(id)``. A summary
built from aggregates cannot see that; a counter the engine maintains sees it
because the engine, not the caller, decides when a row changed.

:data:`PROJECTION_INPUT_COLUMNS` is the column list the triggers watch, and it
is the same list this module reads. Adding a column to one without the other is
what makes a projection lie, so they are one constant.

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
new writes produce) is projected session by session, which is where the
``O(sessions)`` claim above comes from. The two paths are chosen by one indexed
probe, and :meth:`ConversationSessionProjection.repair` never takes the cheap
one while an unstamped row could move an answer.

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
  repair that cannot attribute the movement to appends recomputes everything.

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
per-session path the projection groups **one session's rows** while the test
groups the **whole transcript** — where clusters split on gaps, absorb unlabeled
legacy rows, and are re-merged by id. The claim under test is that the two
arrive at the same place, which is a claim about the algorithm and not about
this function.

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

Concurrency: a fence, not a lock
===============================

No lock is taken here. A test from Phase A's first attempt sat wedged **23 hours
26 minutes** holding ``pg_advisory_xact_lock`` — idle in transaction, no waiter
— and surfaced only when it blocked an unrelated run days later. Nor is a
transaction held across a Python iteration pass, which is why creating the
schema and maintaining the projection are separate concerns: the schema is
created through ``AsyncDatabase.ensure_session_projection_schema``, which owns
the probe → lock → re-probe a concurrent first-post-upgrade boot needs, and
nothing in this module enters that lock.

What replaces the lock is an **epoch**: a counter beside the watermark that says
whose writes count. A repair claims it before deriving anything, every row it
writes carries ``WHERE the epoch is still mine``, and it publishes its watermark
by compare-and-swap on that same epoch. A newer repair claims by bumping the
epoch, which fences the older pass off mid-flight — its remaining writes match
no rows and its publish fails.

Detecting the loss afterwards is **not** enough, and that is the whole reason
the epoch exists. A repair writes its rows before it can possibly know whether
it won, so a slow pass could derive a session, a newer pass could publish the
correct row and advance the watermark, and the slow pass could then overwrite
that row and only afterwards discover it had lost. For the interval in between —
and permanently, if it died before reaching its swap — the watermark would say
current over rows that are not. A watermark cannot un-write a row. So the write
itself is refused rather than regretted.

Two consequences worth stating, because both are deliberate:

* **Only one pass writes rows at a time**, without any pass ever waiting. A pass
  that arrives while another holds the epoch does not block; it takes the epoch
  and the previous holder becomes a no-op.
* **A claim invalidates the watermark for the duration of the pass.** A repair
  that dies half-way therefore costs a full rebuild rather than an incremental
  catch-up — the expensive direction, chosen because the cheap one would require
  trusting rows a dead pass had partly written.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from .session_grouping import (
    coalesce_sessions_by_session_id,
    group_messages_into_sessions,
    timestamp_query_param,
)
from .session_id_column import is_stampable_session_id

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

#: What a watermark *claims*, in :class:`SessionWatermark` field order. The
#: epoch is deliberately not in here: these three are written together by the
#: pass that publishes, and :data:`EPOCH_COLUMN` is the token that decides which
#: pass that may be.
WATERMARK_COLUMNS: Tuple[str, ...] = (
    "accounted_valid",
    "accounted_frontier",
    "accounted_changes",
)

#: Who is allowed to write — to the columns above and to the projection rows
#: themselves. See :meth:`ConversationSessionProjection._claim`.
EPOCH_COLUMN = "epoch"

#: Every write of a projection row carries this, which is what makes a
#: superseded repair harmless rather than merely detectable. Bound with
#: ``(agent_id, epoch)``.
_FENCE = (
    "EXISTS (SELECT 1 FROM conversation_session_watermarks "
    f"WHERE agent_id = ? AND {EPOCH_COLUMN} = ?)"
)

#: How many times a repair will re-read and retry its claim before giving up.
#:
#: A claim fails only when another pass claimed between this one's read and its
#: compare-and-swap, and the answer to that is to re-read and try again — but
#: retrying without a bound is a spin, and under contention the honest report is
#: :data:`DEFERRED` rather than a pass that never returns.
CLAIM_ATTEMPTS = 4

#: Every ``conversation_history`` column this module's answer depends on, and
#: therefore every column whose UPDATE must move the change stamp.
#:
#: One constant, read by the derivation's SELECTs *and* compiled into the
#: triggers, because two lists is how a projection starts lying: add a column
#: here that the trigger does not watch and a rewrite of it is invisible; watch
#: one the derivation never reads and every write of it forces a needless
#: rebuild. ``id`` is absent because it is the primary key on both engines and
#: cannot be updated in place.
PROJECTION_INPUT_COLUMNS: Tuple[str, ...] = (
    "agent_id",
    "session_id",
    "role",
    "metadata",
    "created_at",
    "deleted_at",
    "archived_at",
)

#: What "live" means, authored once. The projection describes these rows and the
#: probes below count these rows; two spellings of one membership rule is the
#: shape that drifts (Phase A's Finding 4).
_LIVE = "deleted_at IS NULL AND archived_at IS NULL"

#: What a per-session repair selects. ``content`` is deliberately absent — the
#: derivation never needs it (see the module docstring), and reading ciphertext
#: bodies to maintain an index would be a cost paid on every repair.
_OWN_ROWS = (
    "SELECT id, role, metadata, created_at, session_id "
    "FROM conversation_history "
    "WHERE agent_id = ? AND session_id = ? "
    f"AND {_LIVE} "
    "ORDER BY id ASC"
)

#: ...and what a transcript repair selects: the same columns over every live row,
#: in the order a reader would see them, so unstamped rows can be attributed the
#: way the grouper attributes them.
_LIVE_ROWS = (
    "SELECT id, role, metadata, created_at, session_id "
    "FROM conversation_history "
    f"WHERE agent_id = ? AND {_LIVE} "
    "ORDER BY id ASC"
)

#: :meth:`ConversationSessionProjection.repair` did nothing: the change stamp it
#: read equals the one the projection recorded, and the record is valid.
CURRENT = "current"

#: Every change since the last repair was a row appearing above the frontier, so
#: only the sessions those rows belong to were recomputed.
INCREMENTAL = "incremental"

#: The change stamp moved in a way appends cannot explain — a soft-delete,
#: restore, archive, purge or metadata rewrite — or the watermark was invalid.
#: Which session was affected is not recoverable from a counter, so every
#: session was recomputed and orphan rows swept.
REBUILT = "rebuilt"

#: Other repairs took the epoch out from under this one :data:`CLAIM_ATTEMPTS`
#: times running, so it wrote nothing at all rather than writing rows the
#: database would refuse. The projection is whatever those passes left it; a
#: caller that needs it current asks again.
DEFERRED = "deferred"


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
    agent_id           TEXT PRIMARY KEY,
    accounted_valid    INTEGER NOT NULL DEFAULT 0,
    accounted_frontier BIGINT NOT NULL DEFAULT 0,
    accounted_changes  BIGINT NOT NULL DEFAULT 0,
    epoch              BIGINT NOT NULL DEFAULT 0
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

    ``accounted_valid`` is what distinguishes *uninitialized or invalidated*
    from *validly accounted for an empty history*. Those look identical as
    numbers — both are all-zero — and conflating them is reachable rather than
    theoretical: claiming the epoch invalidates for the duration of a pass and a
    pass that dies leaves it invalid, so if the agent's history is then purged
    outright, a zero-versus-zero comparison would call an orphaned projection
    current. The flag is checked before any equality, and an invalid watermark
    always rebuilds and sweeps.

    ``epoch`` is the write fence. It is not part of the accounting and says
    nothing about how current the projection is; it says which repair's writes
    the database will accept, so that a superseded pass cannot overwrite a
    published row (see the module docstring).

    ``BIGINT`` on the counters, and ``accounted_changes`` is why: it counts
    every row event over the agent's lifetime, which for an agent expected to
    run indefinitely is precisely the number this table exists to stop being
    bounded by. Overflowing PostgreSQL's ``int4`` there would not produce a
    wrong number but "value out of int32 range" raised out of the
    compare-and-swap — an agent whose projection can never advance again.
    SQLite reads ``BIGINT`` as INTEGER affinity, so both engines hold the range.
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
    :meth:`ConversationSessionProjection.repair`: **each row event moves the
    stamp by exactly one**. That is what lets a repair prove a movement was
    entirely appends by comparing the stamp's delta against the number of rows
    now standing above the frontier — a claim that would be meaningless if some
    events counted twice.

    So an UPDATE stamps ``NEW.agent_id`` once, and ``OLD.agent_id`` only when a
    row actually changes hands. Re-homing between agents is not a thing any
    shipped path does, but "in practice nobody does that" is the reasoning that
    stops being true later, and the alternative here costs one comparison.

    The UPDATE trigger is narrowed to :data:`PROJECTION_INPUT_COLUMNS`. Not for
    tidiness: ``content``, ``rendered_content``, ``embedding_vec`` and ``model``
    are all rewritten in place by ordinary paths (the #1402 canonical/transport
    split, the embedding co-write, the encryption backfill), and stamping those
    would force a full rebuild for a change no field of this table can see.

    Row-level on both engines rather than PostgreSQL's statement-level
    transition tables, because one shape that is provably identical on the two
    engines is worth more here than a bulk-delete optimization on one of them:
    the price is N updates of a single row inside a statement that is already
    updating N rows, all in the same transaction.
    """
    watched_is_distinct = " OR ".join(
        f"OLD.{column} IS DISTINCT FROM NEW.{column}"
        for column in PROJECTION_INPUT_COLUMNS
    )
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
    watched_is_not = " OR ".join(
        f"OLD.{column} IS NOT NEW.{column}"
        for column in PROJECTION_INPUT_COLUMNS
    )
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
            f"WHEN {watched_is_not} BEGIN "
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
    """What a projection last recorded accounting for.

    ``valid`` is not decoration and is checked before either number. All-zero is
    the initial state, and what a claim leaves while a pass is running or after
    one has died, and for an agent with no history it is *also* the truth — so
    without a flag those states are one value, and the projection would call an
    orphaned table current whenever the history behind it had gone away.

    ``frontier`` is the highest ``conversation_history.id`` this pass had looked
    at; ``changes`` is the per-agent change stamp it worked from.

    ``epoch`` is of a different kind from the other three and is carried here
    only because it must be read *with* them: a claim compare-and-swaps on the
    epoch it read beside the state it is about to work from, and reading the two
    separately would let them come from different instants. It is an ownership
    token, not an accounting figure — no caller should read meaning into its
    value, and nothing about staleness depends on it.
    """

    valid: bool = False
    frontier: int = 0
    changes: int = 0
    epoch: int = 0

    def as_params(self) -> Tuple[int, ...]:
        """What this watermark claims, in ``WATERMARK_COLUMNS`` order.

        The epoch is not among them — it is bound separately by whichever
        statement is fencing on it.

        ``valid`` is bound as an ``int`` because the column is ``INTEGER`` on
        both engines; asyncpg refuses a Python ``bool`` for one.
        """
        return (int(self.valid), self.frontier, self.changes)


#: "This projection accounts for nothing." The state before a first repair, and
#: the state a claim installs for the duration of a pass. A repair reading this
#: always rebuilds and sweeps, whatever the numbers beside it say.
INVALID = SessionWatermark()


@dataclass(frozen=True, slots=True)
class RepairOutcome:
    """What a repair did, so a caller can tell "nothing to do" from "rebuilt".

    ``advanced`` is ``False`` when this pass was superseded — another repair took
    the epoch, either before this one could claim it (:data:`DEFERRED`) or while
    it was mid-pass. Neither case leaves anything behind: the fence refuses a
    superseded pass's row writes, so the projection holds only what the pass that
    owns the epoch put there.

    ``sessions`` counts rows the database actually accepted, not rows derived, so
    a superseded pass reports zero rather than the work it did in vain.
    """

    kind: str
    sessions: int
    advanced: bool


@dataclass(frozen=True, slots=True)
class _Claim:
    """One repair's right to write, and the two states that bracket its pass.

    ``accounted`` is what the projection reflected when the epoch was taken —
    the compare-and-swap that took it is also what proves that state was live,
    so the incremental decision may be made from it. ``target`` is what to
    publish once the pass has written everything, and carries the epoch every
    one of those writes is fenced on.
    """

    accounted: SessionWatermark
    target: SessionWatermark


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
    grouped = coalesce_sessions_by_session_id(
        group_messages_into_sessions(
            messages, keep_empty_markers=True, collect_messages=True
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
        # The picker was handed ids in place of text, so this IS the pointer.
        # ``is None`` rather than a truth test: only "the picker settled on
        # nothing" means no pointer, and a row id is never an empty string.
        preview = session.get("preview_content")
        projections.append(
            SessionProjection(
                session_id=session_id,
                started_at=session["started_at"],
                last_message_at=session["last_message_at"],
                message_count=session["message_count"],
                user_message_count=session["user_message_count"],
                first_user_message_id=None if preview is None else int(preview),
                wake_source=session.get("preview_wake_source"),
            )
        )
    return projections


class ConversationSessionProjection:
    """Reads ``conversation_sessions`` for one agent, and repairs it on request.

    The public surface is deliberately narrow: ask whether it is stale, repair
    it, rebuild it, read it. There is **no** "refresh these sessions" method for
    a mutation to call — see the module docstring. Every repair recomputes from
    the rows; nothing is ever incremented, so no count can drift.
    """

    def __init__(self, db, agent_id: str) -> None:
        self.db = db
        self.agent_id = agent_id

    # ── staleness ────────────────────────────────────────────────────────

    async def observed_changes(self) -> int:
        """This agent's change stamp as the database keeps it.

        One primary-key read. A missing row is zero rather than an error: an
        agent whose history has never been touched has had no row events, which
        is what zero says. The ledger is only ever written by the triggers, and
        only ever upward, so this can be compared for equality without a window
        in which it could have gone backwards.
        """
        value = await self.db.fetchval(
            "SELECT changes FROM conversation_history_changes WHERE agent_id = ?",
            (self.agent_id,),
        )
        return int(value or 0)

    async def accounted(self) -> SessionWatermark:
        """The state the projection last recorded accounting for.

        A missing row reads as :data:`INVALID` rather than as an error: an agent
        whose projection has never been repaired has accounted for nothing, and
        *invalid* is the honest name for that — it must rebuild, not compare.

        The epoch is read in the same statement, because a claim swaps on it
        against the state it read beside it; two reads could be two instants.
        """
        row = await self.db.fetchone(
            f"SELECT {', '.join((*WATERMARK_COLUMNS, EPOCH_COLUMN))} "
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
        strong claim and it rests on two things: the watermark says it is valid,
        and the stamp has not moved since the pass that wrote it. Every row event
        moves the stamp, so "has not moved" means "no row this projection could
        describe has changed", including the ones that leave every aggregate
        where it was.
        """
        accounted = await self.accounted()
        if not accounted.valid:
            return True
        return await self.observed_changes() != accounted.changes

    # ── repair ───────────────────────────────────────────────────────────

    async def repair(self) -> RepairOutcome:
        """Bring the projection up to date, incrementally where possible.

        Four outcomes, three of them in increasing cost:

        * ``CURRENT`` — the watermark is valid and the stamp has not moved. Two
          reads; no rows read and none written.
        * ``INCREMENTAL`` — every change since the last pass was a row appearing
          above the frontier, so only the sessions those rows belong to are
          recomputed.
        * ``REBUILT`` — anything else. Every session is recomputed and rows for
          sessions that no longer exist are swept.
        * ``DEFERRED`` — other repairs held the epoch through every claim
          attempt, so this pass wrote nothing. Not a cost, and not an error: the
          passes that displaced it are doing the work.

        Everything but ``CURRENT`` claims the epoch first, and claims it *after*
        reading the state it will work from, so the compare-and-swap that takes
        ownership is also what proves that state was live. From then until the
        pass publishes, no other pass's writes are accepted — which is why a
        superseded repair here is harmless rather than merely detectable (module
        docstring, "a fence, not a lock").

        **Why the incremental test is exact.** Each row event moves the stamp by
        one, so over any interval ``delta`` is
        ``inserts + updates + deletes`` — while the number of rows now standing
        above the old frontier is ``inserts_above - deletes_above``. The second
        is never larger than the first, and they are equal only when
        ``updates``, ``deletes`` and ``inserts_below`` are all zero. So the
        equality is not a heuristic that usually holds: it is false whenever
        anything at or below the frontier moved, which is exactly when the
        incremental branch would be unsound.

        An *invalid* watermark takes the rebuild branch before any of that
        arithmetic is reached, and that ordering is load-bearing rather than
        tidy. Zero is a number two different states share — "never built" and
        "built, and nothing has happened since" — and only the flag tells them
        apart. The state where that bites is the upgrade itself: the ledger and
        the watermark are created empty beside a ``conversation_history`` that
        is already full, so by the numbers alone the projection is current when
        it has never been built at all. The rebuild branch is also the only one
        that sweeps orphans, so calling that state "current" would additionally
        leave rows describing sessions no history row mentions, while reporting
        no staleness.

        What IS load-bearing about the order is that the stamp and the frontier
        are both read **before** any session's rows — which is why :meth:`_claim`
        reads them, rather than this method reading them afterwards. A row
        arriving during the pass is then outside the recorded stamp, so the next
        repair sees a delta appends cannot explain and rebuilds — costly, and
        safe. Recording either value *after* the pass would claim rows this pass
        may never have looked at, which is the silent gap this whole contract
        exists to make impossible.

        The order of the two reads *relative to each other* is not load-bearing,
        which is said here so it does not read as a rule someone later "restores"
        after finding it undefended. Either way the pass reads its sessions last,
        so a row landing between the two reads is seen by the pass; and either
        way the equality above stays strict — a row inside the frontier but
        outside the stamp contributes an event that is not an insert above the
        frontier, so the next repair cannot take the cheap branch. The frontier
        is only ever a lower bound for what the next incremental pass re-reads;
        the stamp alone answers "is this current".

        **Call this outside a transaction of your own.** It is a Python
        iteration pass that re-enters the database per session; wrapping it in
        one transaction is the ABBA shape ``migration_lock``'s own docstring
        warns about, and on SQLite it would hold the single writer slot for the
        whole sweep.
        """
        await self._ensure_watermark_row()
        accounted = await self.accounted()
        observed = await self.observed_changes()
        if accounted.valid and observed == accounted.changes:
            return RepairOutcome(CURRENT, 0, True)

        claim = await self._claim()
        if claim is None:
            return RepairOutcome(DEFERRED, 0, False)
        accounted, target = claim.accounted, claim.target

        if await self._only_appends_since(accounted, target.changes):
            kind = INCREMENTAL
            # No orphan sweep on this branch, and that is an argument rather
            # than an omission: nothing at or below the frontier moved — the
            # arithmetic above is what proves it — so no session below it can
            # have lost its last row. The only sessions that can have changed
            # are the ones named above.
            refreshed = await self._refresh(
                await self._sessions(after=accounted.frontier), target.epoch
            )
        else:
            kind = REBUILT
            refreshed = await self._rebuild_every_session(target.epoch)

        return RepairOutcome(kind, refreshed, await self._publish(target))

    async def rebuild(self) -> int:
        """Recompute every session from ``conversation_history``. Always valid.

        The exact answer, and the recovery path for a projection that has been
        dropped, corrupted by hand, or left behind by a superseded pass. Unlike
        :meth:`repair` it does not short-circuit on a matching stamp — a caller
        who has just dropped the table wants the rows back, not to be told the
        watermark looks fine.

        Returns how many sessions it wrote, which is rows the database accepted
        rather than rows derived. It raises instead of returning zero when it
        cannot take the epoch at all: this is the verb reached for when the table
        is known to be wrong, and a recovery call that quietly did nothing would
        be read as a recovery that happened. Zero is still a possible answer for
        an agent with no sessions, and for a pass whose epoch was taken away
        mid-rebuild — in that second case the rows standing are the newer pass's,
        which is the outcome this one was asking for anyway.
        """
        await self._ensure_watermark_row()
        claim = await self._claim()
        if claim is None:
            raise RuntimeError(
                f"conversation_sessions: could not take {self.agent_id}'s "
                f"projection from concurrent repairs in {CLAIM_ATTEMPTS} "
                "attempts, so nothing was rebuilt"
            )
        rebuilt = await self._rebuild_every_session(claim.target.epoch)
        await self._publish(claim.target)
        return rebuilt

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
            "ORDER BY last_message_at DESC, session_id ASC",
            (self.agent_id,),
        )
        return [_as_dict(row) for row in rows]

    # ── internals ────────────────────────────────────────────────────────

    async def _frontier(self) -> int:
        """The highest row id this agent has. One backward index step.

        ``idx_conversation_agent_row_id`` is what makes that true on PostgreSQL;
        SQLite gets the plan free because ``id`` is its rowid and rides in every
        index as a trailing key column.
        """
        return int(
            await self.db.fetchval(
                "SELECT COALESCE(MAX(id), 0) FROM conversation_history "
                "WHERE agent_id = ?",
                (self.agent_id,),
            )
            or 0
        )

    async def _rows_above(self, frontier: int) -> int:
        """How many of this agent's rows stand above ``frontier``.

        Counted over ALL rows, live or not, because the stamp counts row events
        regardless of liveness and the two are compared against each other.
        Bounded by the rows that appeared since the last repair — which are the
        rows an incremental pass is about to read anyway.
        """
        return int(
            await self.db.fetchval(
                "SELECT COUNT(*) FROM conversation_history "
                "WHERE agent_id = ? AND id > ?",
                (self.agent_id, frontier),
            )
            or 0
        )

    async def _only_appends_since(
        self, accounted: SessionWatermark, observed: int
    ) -> bool:
        """Whether the incremental branch is sound. See :meth:`repair`."""
        if not accounted.valid or observed <= accounted.changes:
            return False
        if await self._has_unstamped_rows():
            # An appended row that carries no session id belongs to whichever
            # session it fell next to, and ``_sessions(after=...)`` cannot name
            # that session because the row is filed under nothing. Attribution
            # is a transcript question, so an agent with unstamped rows is
            # rebuilt rather than caught up.
            return False
        return observed - accounted.changes == await self._rows_above(
            accounted.frontier
        )

    async def _has_unstamped_rows(self) -> bool:
        """Whether any live row of this agent's is filed under no session id.

        Seeded by Phase A's ``(agent_id, session_id)`` index and stopped at the
        first hit, so the ordinary answer costs one seek. It is not free in the
        pathological case — an agent holding many unstamped rows that are all
        soft-deleted or archived pays a heap visit per candidate — but that is
        an agent already on the transcript derivation, whose repair reads its
        whole live history anyway.

        This chooses between the two derivations: with no unstamped rows a
        session's own rows are the whole of its story, and with any of them,
        attribution has to be read off the transcript (see the module
        docstring).
        """
        row = await self.db.fetchone(
            "SELECT 1 FROM conversation_history "
            f"WHERE agent_id = ? AND session_id IS NULL AND {_LIVE} "
            "LIMIT 1",
            (self.agent_id,),
        )
        return row is not None

    async def _sessions(self, after: Optional[int] = None) -> List[str]:
        """The session ids this agent's rows are filed under.

        ``after`` restricts to rows the frontier has not accounted for. Ids
        outside the Phase A column contract are dropped, and that is a shortcut
        rather than a gate: the COLUMN is the gate, so no row can carry such an
        id and a query for one would find nothing. Whoever reads this looking
        for the thing that keeps an unclaimable id out of the table — it is
        :func:`~kestrel_sovereign.storage.session_id_column.column_session_id`,
        not this line.
        """
        clause = "" if after is None else "AND id > ? "
        params: tuple = (
            (self.agent_id,) if after is None else (self.agent_id, after)
        )
        rows = await self.db.fetchall(
            "SELECT DISTINCT session_id FROM conversation_history "
            f"WHERE agent_id = ? {clause}AND session_id IS NOT NULL",
            params,
        )
        seen: Dict[str, None] = {}
        for (session_id,) in rows:
            if is_stampable_session_id(session_id):
                seen.setdefault(str(session_id), None)
        return list(seen)

    async def _rebuild_every_session(self, epoch: int) -> int:
        """Recompute every session and drop the rows nothing describes any more.

        Two derivations, one answer. With unstamped live rows present the whole
        live transcript is grouped exactly as a reader would group it, so rows
        carrying no session id are attributed to the session they fall next to.
        Without them, each session's own rows are its whole story and the pass
        stays per-session — which is the ``O(sessions)`` shape the projection
        exists for, and the shape every agent is in once Phase A's backfill has
        run and only current write paths have added rows since.

        Sweeping orphans is what makes this branch self-correcting, and that is
        load-bearing for the fence: a pass that claimed the epoch and then died
        may have written some sessions and not others, and the next pass reads an
        invalid watermark and lands here. Recomputing every session and dropping
        every row no live row is filed under leaves nothing of a half-finished
        predecessor.
        """
        if await self._has_unstamped_rows():
            projections = project_transcript(
                await self.db.fetchall(_LIVE_ROWS, (self.agent_id,))
            )
            written = 0
            for projection in projections:
                written += await self._store(projection, epoch)
            await self._forget_orphans(
                [p.session_id for p in projections], epoch
            )
            return written

        sessions = await self._sessions()
        refreshed = await self._refresh(sessions, epoch)
        await self._forget_orphans(sessions, epoch)
        return refreshed

    async def _refresh(self, session_ids: Iterable[str], epoch: int) -> int:
        """Recompute the named sessions from the rows that are live now.

        Only ever reached with no unstamped live rows in play (see
        :meth:`_rebuild_every_session` and :meth:`_only_appends_since`), which is
        what makes reading one session's own rows the same answer the grouper
        would give for the whole transcript.

        Counts rows the database accepted. A pass whose epoch has been taken
        derives exactly as much and stores none of it, so it returns zero rather
        than a number that describes work no reader can see.
        """
        written = 0
        for session_id in sorted(session_ids):
            rows = await self.db.fetchall(_OWN_ROWS, (self.agent_id, session_id))
            projections = project_transcript(rows, expect=session_id)
            if projections:
                written += await self._store(projections[0], epoch)
            else:
                # No live rows left, or the rows the column selected are not
                # the ones the transcript files under it. Either way this
                # session has no row to store; the second case is logged by
                # ``project_transcript``.
                await self._forget(session_id, epoch)
        return written

    async def _forget_orphans(self, keep: Sequence[str], epoch: int) -> None:
        """Drop rows for sessions no live row is filed under any more.

        Diffed in Python against what is stored rather than expressed as a
        ``NOT IN`` over the session list: the list is unbounded, and a statement
        whose length grows with history is a different failure mode from a
        projection that is merely behind.
        """
        keeping: Set[str] = {str(session_id) for session_id in keep}
        stored = {
            str(row[0])
            for row in await self.db.fetchall(
                "SELECT session_id FROM conversation_sessions WHERE agent_id = ?",
                (self.agent_id,),
            )
        }
        for orphan in sorted(stored - keeping):
            await self._forget(orphan, epoch)

    async def _forget(self, session_id: str, epoch: int) -> int:
        """Drop one row, if this pass still owns the projection."""
        return await self.db.execute(
            "DELETE FROM conversation_sessions "
            f"WHERE agent_id = ? AND session_id = ? AND {_FENCE}",
            (self.agent_id, session_id, self.agent_id, epoch),
        )

    async def _store(self, projection: SessionProjection, epoch: int) -> int:
        """Upsert one row, if this pass still owns the projection.

        Returns 1 when the row landed and 0 when the fence refused it — a newer
        repair has taken the epoch, so this pass's answer is derived from rows
        that pass has already superseded.

        ``SELECT ... WHERE EXISTS`` rather than ``VALUES``, because the guard has
        to be evaluated by the statement that writes; a check in Python before an
        unguarded write is exactly the window this is here to close. Both engines
        accept this spelling — and both need the SELECT's own WHERE clause for
        SQLite's parser to tell where the ``ON CONFLICT`` belongs, which the
        fence supplies. PostgreSQL takes the parameters' types from the INSERT's
        target columns, so no cast is needed for the ``TIMESTAMP`` pair.
        """
        assignments = ", ".join(
            f"{column} = excluded.{column}" for column in PROJECTION_COLUMNS
        )
        return await self.db.execute(
            "INSERT INTO conversation_sessions "
            f"(agent_id, session_id, {', '.join(PROJECTION_COLUMNS)}) "
            "SELECT ?, ?, ?, ?, ?, ?, ?, ? "
            f"WHERE {_FENCE} "
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
                self.agent_id,
                epoch,
            ),
        )

    async def _ensure_watermark_row(self) -> None:
        """Make sure there is a row to take the epoch from.

        Written invalid, which is both the truth and the safe direction: a first
        repair must rebuild and sweep rather than compare zeroes.

        Without this the first repair would try to claim a row that does not
        exist, match nothing, and conclude it had lost a race it was never in —
        leaving the projection permanently stale.
        """
        columns = (*WATERMARK_COLUMNS, EPOCH_COLUMN)
        await self.db.execute(
            "INSERT INTO conversation_session_watermarks "
            f"(agent_id, {', '.join(columns)}) "
            f"VALUES (?, {', '.join('0' for _ in columns)}) "
            "ON CONFLICT (agent_id) DO NOTHING",
            (self.agent_id,),
        )

    async def _claim(self) -> Optional[_Claim]:
        """Take the epoch, so that from here only this pass's writes are kept.

        This is the whole of this module's concurrency story, and it replaces the
        serialization that failed review twice. Two repairs racing on one session
        can each derive a projection row from rows the other has already changed,
        and nothing in a single upsert orders them — so ownership is settled
        *before* either derives anything, by bumping a counter no two passes can
        bump to the same value.

        The bump is a compare-and-swap against the epoch this pass just read, and
        that is what makes the returned ``accounted`` trustworthy: had another
        pass claimed in between, the swap would match nothing and this one would
        re-read rather than reason from a state that had already moved.

        Claiming **invalidates** the watermark for the duration of the pass, and
        that is deliberate in two directions. Any pass arriving mid-flight reads
        invalid, so it rebuilds from history instead of catching up over rows a
        predecessor may have half-written; and a pass that dies leaves the
        watermark invalid rather than valid-but-wrong.

        ``None`` when :data:`CLAIM_ATTEMPTS` passes have each taken the epoch
        away first. Nothing has been written in that case, and nothing is owed:
        the passes that displaced this one are doing the same work.

        A claim taken just after another pass published is not wasted effort
        worth avoiding — it recomputes and republishes the same answer. Only the
        cheap ``CURRENT`` check in :meth:`repair`, which runs before any claim,
        exists to keep that off the hot path.
        """
        for _ in range(CLAIM_ATTEMPTS):
            accounted = await self.accounted()
            # Read before the claim, never after it: a row landing during the
            # pass must fall OUTSIDE what this pass records, so the next repair
            # sees it. See :meth:`repair` on why behind is the safe direction.
            observed = await self.observed_changes()
            frontier = await self._frontier()
            epoch = accounted.epoch + 1
            taken = await self.db.execute(
                "UPDATE conversation_session_watermarks SET "
                + ", ".join(f"{column} = 0" for column in WATERMARK_COLUMNS)
                + f", {EPOCH_COLUMN} = ? "
                f"WHERE agent_id = ? AND {EPOCH_COLUMN} = ?",
                (epoch, self.agent_id, accounted.epoch),
            )
            if taken:
                return _Claim(
                    accounted, SessionWatermark(True, frontier, observed, epoch)
                )
        logger.info(
            "conversation_sessions: %s's projection was claimed by another "
            "repair on each of %d attempts; this pass wrote nothing",
            self.agent_id,
            CLAIM_ATTEMPTS,
        )
        return None

    async def _publish(self, target: SessionWatermark) -> bool:
        """Record what this pass accounted for, if it still owns the epoch.

        Fenced on the epoch exactly as the row writes are, and for one reason:
        the two must agree about who wrote. A pass that still owns the epoch
        wrote every projection row that landed since it claimed, so the state it
        publishes describes the rows that are actually there.

        ``False`` means a newer repair took the epoch mid-pass. That pass's
        writes are the ones in the table and its watermark is the one that will
        stand, so this one neither claims nor invalidates anything — the fence
        already made its writes no-ops, which is the difference between a
        superseded pass being harmless and a superseded pass being merely
        detected after the fact.
        """
        published = await self.db.execute(
            "UPDATE conversation_session_watermarks SET "
            + ", ".join(f"{column} = ?" for column in WATERMARK_COLUMNS)
            + f" WHERE agent_id = ? AND {EPOCH_COLUMN} = ?",
            (*target.as_params(), self.agent_id, target.epoch),
        )
        if published:
            return True
        logger.info(
            "conversation_sessions: another repair took %s's projection "
            "mid-pass; this pass's rows were refused and its watermark is not "
            "recorded",
            self.agent_id,
        )
        return False

    def _timestamp(self, value: str) -> Any:
        return timestamp_query_param(getattr(self.db, "backend_type", ""), value)


def _as_dict(row: Sequence[Any]) -> Dict[str, Any]:
    return {
        "session_id": row[0],
        **{column: row[index + 1] for index, column in enumerate(PROJECTION_COLUMNS)},
    }
