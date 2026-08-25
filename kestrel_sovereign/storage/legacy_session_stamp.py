"""Write down which session a legacy row is in, once (#3120).

A row's session was never recorded. ``group_messages_into_sessions`` derives it
from what the row fell NEXT TO — a gap, a marker, the id its neighbours carry —
and every reader re-derives it. That works while every reader sees the same
rows, and stops the moment they do not: a lifecycle op selects one deletion
state, the list has three views, and an unlabeled row that moved between them
has lost the only evidence of where it belonged.

Two codex rounds on the resolver found four P1s, two of them inside the
previous round's fix, and all of them that wall from a different side: a
session partially restored leaves rows in Trash that ARE its own, and an
archived session beside an unrelated active row looks *exactly* the same. No
rule over the rows can separate them, because the distinguishing fact was never
written down.

So write it down. Every row since #2012 carries its session id; this fills in
the ones from before, using the grouping the reader already shows, and after it
membership is a fact rather than an inference.

What it will and will not touch
===============================

**Only where nothing live disagrees.** Each candidate is a row the grouper
files under a canonical session whose own metadata says something else — in
practice a bare integer, which names a ROW rather than a session (#2012) and
which the grouper therefore reads as unlabeled. Measured across the four live
agents: 107 such rows, and every single affected row carries one. None carries
nothing at all.

That integer is a claim, and it is honoured:

* it names a row in the SAME session the grouper assigns — the claim and the
  grouping agree, so writing the grouping down is a re-spelling and no message
  moves (53 rows);
* it names no LIVE row at all — deleted, archived, or gone — so nothing live
  disagrees and the grouping is the only answer standing (34 rows);
* it names a live row in a DIFFERENT session — the claim and the grouping
  conflict, and which of them is right is not a question this can answer. Left
  exactly as it is, and logged (20 rows).

**And only where the value can be written.** A cluster keyed by a row id is not
touched at all: that key is not one the ``session_id`` column may hold (#2958),
and re-keying such a cluster is a different operation with different
consequences — it is what #3061 abandoned, because ``conversation_titles`` is
keyed on the legacy id and every rename would be lost. Nothing here moves a
session's key. It only fills in rows already displayed under one.

A document that does not parse to a JSON object is skipped rather than
replaced; a message with an unreadable blob is still a message, and overwriting
it would destroy what it holds.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, NamedTuple, Optional, Sequence, Tuple

from .session_grouping import (
    canonical_session_id,
    coalesce_sessions_by_session_id,
    group_messages_into_sessions,
    parse_message_metadata,
)
from .session_id_column import SESSION_ID_KEY, column_session_id

logger = logging.getLogger(__name__)

#: Records that an agent's legacy rows have been stamped.
#:
#: A marker rather than a probe of the rows, because "is there anything left to
#: do" cannot be asked cheaply: the rows this leaves alone — a legacy cluster's
#: own, a conflicting claim — look exactly like the rows it has not reached
#: yet, so any probe over them stays true for ever and the work runs on every
#: request. ``_init_schema`` runs on every ``from_pool()``, which is what made
#: that the objection that stopped #3061.
STAMP_TABLE = "conversation_session_stamped"

STAMP_DDL = f"""CREATE TABLE IF NOT EXISTS {STAMP_TABLE} (
    agent_id TEXT PRIMARY KEY,
    rows_stamped INTEGER NOT NULL,
    rows_refused INTEGER NOT NULL,
    completed_at TIMESTAMP NOT NULL
)"""

#: How many rows one transaction rewrites.
STAMP_BATCH = 200


#: The list's three views, each of which groups its own rows.
_VIEWS = (
    "deleted_at IS NULL AND archived_at IS NULL",
    "deleted_at IS NULL AND archived_at IS NOT NULL",
    "deleted_at IS NOT NULL",
)


def _transcript_in_view(backend_type: str, where: str) -> str:
    """:func:`live_transcript_sql`, for one view instead of the live one.

    The same columns and the same canonical order — the projection's own read —
    with its ``_LIVE`` predicate replaced by the view being planned.
    """
    from .conversation_sessions import _LIVE, live_transcript_sql

    return live_transcript_sql(backend_type).replace(_LIVE, where, 1)


def _null_safe_equality(backend_type: str) -> str:
    """``= ?`` that is true of two NULLs, in this engine's spelling.

    SQLite reads ``IS`` as null-safe equality and takes a parameter on the
    right. PostgreSQL's ``IS`` is a unary predicate — ``IS NULL``, ``IS TRUE``
    — and refuses a bind parameter outright, so the whole migration would raise
    on its first post-upgrade boot for any agent that had work to do.
    """
    return "IS NOT DISTINCT FROM ?" if backend_type == "postgres" else "IS ?"


def _ignore_conflict(backend_type: str) -> str:
    """Leave a marker another initializer has already written.

    Two of them plan the same agent, write the same rows through the same
    compare-and-set, and arrive here together; the second is not an error.
    """
    return "ON CONFLICT (agent_id) DO NOTHING"


class LegacyStamp(NamedTuple):
    """One row's session, written down."""

    row_id: int
    session_id: str
    metadata: str
    previous_metadata: Optional[str]


class StampPlan(NamedTuple):
    """What a pass would do, decided before anything is written."""

    stamps: List[LegacyStamp]
    #: ``(row_id, claimed, grouped)`` for a claim that conflicts with the
    #: grouping. Reported, never resolved.
    conflicts: List[Tuple[int, str, str]]


def plan_stamps(rows: Sequence[Sequence[Any]]) -> StampPlan:
    """Decide, from one agent's live transcript, what may be written down.

    ``rows`` are what :func:`live_transcript_sql` selects, read through
    :func:`_transcript_messages` — the projection's own shape and order, not a
    second spelling of "the live transcript", because this has to agree with
    what the list shows and a second spelling is where two derivations drift
    apart.

    Pure: it reads rows and returns statements. Nothing here decides when to
    run or how to write, which is what lets the whole decision be tested
    against a corpus without a database.
    """
    from .conversation_sessions import _transcript_messages

    messages, _stamped = _transcript_messages(rows)
    grouped = coalesce_sessions_by_session_id(
        group_messages_into_sessions(messages, collect_messages=True)
    )
    # Where the grouper puts every row, so a claim naming another row can be
    # asked what session THAT row is in.
    session_of: Dict[Any, str] = {}
    for session in grouped:
        for message in session["messages"]:
            session_of[message["id"]] = str(session["session_id"])
    # A ``new_session`` marker is structural: the grouper starts a session at
    # it and leaves it out of the messages it collects, so the loop above never
    # places one. It needs no grouping to place — a marker STARTS the session
    # keyed by its own canonical id, or by its row id when it names none, which
    # is exactly what ``_new_session`` does with it.
    for row in rows:
        marker = parse_message_metadata(row[2])
        if marker.get("new_session"):
            session_of.setdefault(
                row[0], canonical_session_id(marker) or str(row[0])
            )
    # And every row these candidates were read from, which is NOT the same set
    # even now: a row the grouper dropped is still live, and reading "not
    # placed" as "names nothing live" would overwrite a claim rather than
    # report it.
    live_ids = {row[0] for row in rows}
    # Every key some row NAMES. The guard below asks whether a session's key
    # came from metadata or from the grouper's row-id fallback, and a session
    # opened by a ``new_session`` marker takes its key from that marker — which
    # is not among the messages, so asking the messages alone would refuse a
    # marker-started session and leave its whole continuation unstamped.
    named_keys = {
        canonical_session_id(parse_message_metadata(row[2])) for row in rows
    }
    named_keys.discard(None)
    # The metadata AS STORED, for the compare-and-set below. A row rewritten
    # between the read and the write must not be clobbered by a document
    # derived from what it used to hold.
    raw: Dict[Any, Optional[str]] = {row[0]: row[2] for row in rows}

    stamps: List[LegacyStamp] = []
    conflicts: List[Tuple[int, str, str]] = []
    for session in grouped:
        key = str(session["session_id"])
        # The key has to have come from a row's METADATA, not from the row-id
        # fallback the grouper uses when no row names the cluster. Re-keying a
        # legacy cluster is #3061's operation, not this one — it moves a
        # session's identity, and `conversation_titles` is keyed on the old
        # one, so every rename would be lost.
        #
        # Asked as "does some row here say this", rather than "does the key
        # look like a row id". The second is what `canonical_session_id`
        # answers, and it answers it by a DIGIT test: a negative row id has a
        # sign in front of it, passes, and would have had `-5` written into
        # rows as though it were a session someone had named.
        if key not in named_keys:
            continue
        for message in session["messages"]:
            metadata = message["metadata"]
            if canonical_session_id(metadata) == key:
                continue  # already says so
            claim = metadata.get(SESSION_ID_KEY)
            if claim is not None:
                target = _claimed_row(claim)
                if target is None:
                    continue  # a claim that is not a row id: not ours to read
                if target in live_ids and session_of.get(target) != key:
                    conflicts.append((message["id"], str(claim), key))
                    continue
            previous = raw[message["id"]]
            document = _document(previous)
            if document is None:
                continue  # unreadable, and a message is still a message
            document[SESSION_ID_KEY] = key
            written = json.dumps(document)
            if column_session_id(written) != key:
                # The value cannot reach the column — charset, length, a
                # duplicate key the three readers resolve differently. Writing
                # metadata the column may not follow would put the two back
                # into the disagreement this exists to end.
                continue
            stamps.append(
                LegacyStamp(int(message["id"]), key, written, previous)
            )
    return StampPlan(stamps, conflicts)


def _claimed_row(claim: Any) -> Optional[int]:
    """The row a bare-integer ``session_id`` names, or ``None``."""
    text = str(claim)
    if not text.isdigit():
        return None
    try:
        return int(text)
    except ValueError:  # pragma: no cover - isdigit already refused these
        return None


def _document(raw: Optional[str]) -> Optional[Dict[str, Any]]:
    """The row's metadata as a mutable object, or ``None`` if it is not one.

    Only SQL NULL is absent. An empty string is a value someone stored, and
    ``json.loads`` refuses it — treating it as missing would replace it with a
    document of our own, which is the one thing this promises not to do.

    A duplicated top-level key is refused for the reason
    ``session_id_column`` refuses it: SQLite's ``json_extract`` takes the
    first, PostgreSQL's ``jsonb`` and Python's ``json.loads`` take the last, so
    whichever occurrence is "right", two of the three readers disagree. Reading
    such a document to rewrite it would silently pick one.
    """
    if raw is None:
        return {}
    try:
        parsed = json.loads(raw, object_pairs_hook=_no_duplicate_keys)
    except (TypeError, ValueError):
        return None
    return parsed if isinstance(parsed, dict) else None


def _no_duplicate_keys(pairs: Sequence[Tuple[str, Any]]) -> Dict[str, Any]:
    """``dict`` of one JSON object, refusing a key that appears twice."""
    seen: Dict[str, Any] = {}
    for key, value in pairs:
        if key in seen:
            raise ValueError(f"duplicate key {key!r}")
        seen[key] = value
    return seen


async def stamp_legacy_sessions(db: Any) -> None:
    """Write down each legacy row's session, once per agent (#3120).

    Ordered AFTER the projection's schema pass so the change triggers exist:
    the writes below bump the change stamp, the projection notices, and the
    next repair derives from rows that now say where they belong. Before it,
    the stamp would not move and a stored projection would go on describing
    rows that had been rewritten underneath it.

    Each agent is one gate, one plan and then bounded transactions. The plan is
    read OUTSIDE them — it is unbounded, and holding a write transaction across
    an unbounded read is what the projection's own rebuild refuses to do — so
    every write re-checks the row it is about: ``WHERE metadata = ?`` against
    the exact document the plan was made from. A row rewritten in between
    simply is not stamped, and the marker is not written, so the next boot
    plans again.
    """
    if not await db.table_exists("conversation_history"):
        return
    async with db.migration_lock(f"create_{STAMP_TABLE}"):
        if not await db.table_exists(STAMP_TABLE):
            await db.execute(STAMP_DDL)

    agents = await db.fetchall(
        f"SELECT DISTINCT agent_id FROM conversation_history c "
        f"WHERE NOT EXISTS (SELECT 1 FROM {STAMP_TABLE} s "
        f"WHERE s.agent_id = c.agent_id)",
        (),
    )
    for (agent_id,) in agents:
        await _stamp_one_agent(db, agent_id)


async def _stamp_one_agent(db: Any, agent_id: str) -> None:
    """Plan and write one agent's legacy rows, then record that it is done."""
    # Imported here rather than at module scope: ``async_database`` imports
    # this module, and this imports back through it.
    from .async_conversation_store import _rows_affected

    # No lock around any of this. ``migration_lock`` opens a TRANSACTION —
    # ``BEGIN IMMEDIATE`` on SQLite, a transaction-scoped advisory lock on
    # PostgreSQL — so holding it here would put the unbounded transcript read
    # and every "batch" inside one transaction, taking SQLite's writer slot for
    # the whole pass and making ``STAMP_BATCH`` bound nothing at all.
    #
    # Correctness does not need it. Two initializers that plan the same agent
    # write the same rows; the compare-and-set makes the second a no-op, and
    # the marker's primary key makes the second insert one.
    stamped = 0
    planned = 0
    conflicts: List[Tuple[int, str, str]] = []
    # Each of the list's three views separately, because each groups its OWN
    # rows and an agent whose legacy session is archived or in Trash would
    # otherwise be recorded complete without those rows being read at all —
    # and the marker would then refuse every retry, including after a restore.
    for where in _VIEWS:
        rows = await db.fetchall(
            _transcript_in_view(db.backend_type, where), (agent_id,)
        )
        plan = plan_stamps(rows)
        planned += len(plan.stamps)
        conflicts.extend(plan.conflicts)
        for start in range(0, len(plan.stamps), STAMP_BATCH):
            batch = plan.stamps[start:start + STAMP_BATCH]
            async with db.transaction():
                for stamp in batch:
                    affected = await db.execute(
                        "UPDATE conversation_history "
                        "SET metadata = ?, session_id = ? "
                        f"WHERE id = ? AND agent_id = ? AND metadata "
                        f"{_null_safe_equality(db.backend_type)}",
                        (
                            stamp.metadata,
                            stamp.session_id,
                            stamp.row_id,
                            agent_id,
                            stamp.previous_metadata,
                        ),
                    )
                    stamped += _rows_affected(affected)

    if stamped != planned:
        # A row moved under the plan. The marker is the claim "this agent's
        # legacy rows say where they belong", and it is not true yet, so it is
        # not written: the next boot reads the rows as they now stand.
        logger.info(
            "%s: stamped %s of %s planned legacy rows for %s; a row was "
            "rewritten under the plan, so the pass will run again (#3120)",
            STAMP_TABLE, stamped, planned, agent_id,
        )
        return

    for row_id, claimed, grouped in conflicts:
        logger.warning(
            "conversation row %s claims session %r and the transcript "
            "groups it under %r; left as it stands (#3120)",
            row_id, claimed, grouped,
        )
    await db.execute(
        f"INSERT INTO {STAMP_TABLE} "
        "(agent_id, rows_stamped, rows_refused, completed_at) "
        f"VALUES (?, ?, ?, CURRENT_TIMESTAMP) {_ignore_conflict(db.backend_type)}",
        (agent_id, stamped, len(conflicts)),
    )
    if stamped or conflicts:
        logger.info(
            "%s: %s legacy rows now name their session for %s, %s left as "
            "they stand (#3120)",
            STAMP_TABLE, stamped, agent_id, len(conflicts),
        )
