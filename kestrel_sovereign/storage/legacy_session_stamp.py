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
    #: Rows left alone for a reason that is not a conflict — an unreadable
    #: document, a key the column may not hold. Counted so a run that stamps
    #: nothing and refuses nothing cannot be read as "there was nothing left".
    skipped: int = 0


def plan_stamps(
    rows: Sequence[Sequence[Any]],
    placements: Optional[Dict[Any, Optional[str]]] = None,
) -> StampPlan:
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
    #
    # ``placements`` carries the same question answered across ALL THREE views.
    # A claim can point out of the view being planned — an archived row naming
    # an active one, or the reverse — and asking only this view would read that
    # as "names nothing" and rewrite the row to whatever it happens to sit
    # beside here. The grouping stays per view; only the lookup is global,
    # because a claim names a ROW and a row exists in exactly one of them.
    if placements is None:
        placements = {}
        for row in rows:
            placements[row[0]] = None
    known = dict(placements)
    known.update(session_of)
    live_ids = set(known)
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
    skipped = 0
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
                    skipped += 1
                    continue  # a claim that is not a row id: not ours to read
                if target in live_ids and known.get(target) != key:
                    conflicts.append((message["id"], str(claim), key))
                    continue
            previous = raw[message["id"]]
            document = _document(previous)
            if document is None:
                skipped += 1
                continue  # unreadable, and a message is still a message
            document[SESSION_ID_KEY] = key
            written = json.dumps(document)
            if column_session_id(written) != key:
                # The value cannot reach the column — charset, length, a
                # duplicate key the three readers resolve differently. Writing
                # metadata the column may not follow would put the two back
                # into the disagreement this exists to end.
                skipped += 1
                continue
            stamps.append(
                LegacyStamp(int(message["id"]), key, written, previous)
            )
    return StampPlan(stamps, conflicts, skipped)


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


def _placements(rows: Sequence[Sequence[Any]]) -> Dict[Any, Optional[str]]:
    """Which session each of one view's rows is in, markers included."""
    from .conversation_sessions import _transcript_messages

    messages, _stamped = _transcript_messages(rows)
    placed: Dict[Any, Optional[str]] = {}
    for session in coalesce_sessions_by_session_id(
        group_messages_into_sessions(messages, collect_messages=True)
    ):
        for message in session["messages"]:
            placed[message["id"]] = str(session["session_id"])
    for row in rows:
        marker = parse_message_metadata(row[2])
        if marker.get("new_session"):
            placed.setdefault(row[0], canonical_session_id(marker) or str(row[0]))
        placed.setdefault(row[0], None)
    return placed


async def _change_stamp(db: Any, agent_id: str) -> Optional[int]:
    """How many row events the database has counted for this agent.

    ``None`` when the ledger is not there to ask — a database predating the
    #2959 triggers, where the compare-and-set is the only fence there is.
    """
    from .conversation_sessions import CHANGES_TABLE

    if not await db.table_exists(CHANGES_TABLE):
        return None
    row = await db.fetchone(
        f"SELECT COALESCE(SUM(changes), 0) FROM {CHANGES_TABLE} "
        "WHERE agent_id = ?",
        (agent_id,),
    )
    return int(row[0]) if row and row[0] is not None else 0


async def _unchanged(
    db: Any, agent_id: str, fence: Optional[int], ours: int
) -> bool:
    """Whether nothing but this pass has touched the agent's rows.

    Each stamp is exactly one row event, so the ledger should have moved by
    precisely what this pass has written. Anything else is a lifecycle
    operation running beside it, and the plan in hand was made from rows it may
    since have moved.
    """
    if fence is None:
        return True
    now = await _change_stamp(db, agent_id)
    return now is None or now - fence == ours


async def stamp_legacy_sessions(db: Any, agent_id: Optional[str] = None) -> Dict[str, int]:
    """Write down each legacy row's session. Idempotent, and run on demand.

    **Not a boot migration, and that is the finding this design came out of.**
    It was one, gated by a per-agent marker, and three separate review findings
    were the marker's rather than the work's: an agent DID left in a durable
    table after a privacy purge, a completion record that could not be
    published atomically against a concurrent restore, and an anti-join over
    the whole transcript on every ``from_pool()`` to discover which agents
    still needed it.

    None of them is about stamping rows. They are all about knowing whether
    stamping has happened — a question this does not need to ask, because
    running it twice writes nothing the second time: a row that names its
    session is not a candidate, and a row this refuses is refused again for the
    same reason. So it is called instead of gated, from the two places that
    reintroduce legacy rows and from ``kestrel storage stamp-sessions``.

    Each of the list's three views is planned separately — each groups its OWN
    rows — while a claim is resolved across all of them, because a claim names
    a ROW and a row lives in exactly one view. Every write re-checks the
    document its plan was made from, so a row rewritten under an unbounded read
    is skipped rather than clobbered, and a pass that skipped one is simply run
    again.

    Returns ``{"stamped", "refused"}``.
    """
    # Imported here rather than at module scope: both import back through the
    # database module this is called from.
    from .async_conversation_store import _rows_affected

    if not await db.table_exists("conversation_history"):
        return {"stamped": 0, "refused": 0, "skipped": 0}
    if agent_id is None:
        agents = [
            row[0]
            for row in await db.fetchall(
                "SELECT DISTINCT agent_id FROM conversation_history", ()
            )
        ]
    else:
        agents = [agent_id]

    total = {"stamped": 0, "refused": 0, "skipped": 0}
    for agent in agents:
        # Each view read ONCE and kept: the grouping is global within a view,
        # so the rows have to be in hand anyway. Measured on a synthetic
        # 100,000-row agent, a full pass is 1.2 seconds; on the live agents,
        # twenty milliseconds.
        # Taken BEFORE the first read, not after the last. A lifecycle
        # mutation committing between two of the view queries would otherwise
        # be part of the baseline — the plan built from rows it had already
        # moved, and the fence agreeing that nothing happened.
        fence = await _change_stamp(db, agent)

        by_view: List[Sequence[Sequence[Any]]] = []
        placements: Dict[Any, Optional[str]] = {}
        for where in _VIEWS:
            view_rows = await db.fetchall(
                _transcript_in_view(db.backend_type, where), (agent,)
            )
            by_view.append(view_rows)
            for row_id, session in _placements(view_rows).items():
                placements[row_id] = session

        stamped = 0
        skipped = 0
        conflicts: List[Tuple[int, str, str]] = []
        moved = False
        for rows in by_view:
            if moved:
                break
            plan = plan_stamps(rows, placements)
            conflicts.extend(plan.conflicts)
            skipped += plan.skipped
            for start in range(0, len(plan.stamps), STAMP_BATCH):
                batch = plan.stamps[start:start + STAMP_BATCH]
                # Revalidated INSIDE the transaction, not before it. The
                # compare-and-set below asks whether the ROW moved, and that is
                # not the whole question: a candidate's metadata can sit still
                # while the rows around it — the ones that decide which session
                # it is in — do not. A purge removing a canonical anchor and
                # missing its unlabeled tail would otherwise have this stamp
                # that tail back INTO the purged session, and a privacy purge
                # that leaves a row naming what it destroyed is not one.
                # Checked outside, a writer only has to commit in the gap
                # before ``BEGIN``.
                landed = 0
                async with db.transaction():
                    if await _unchanged(db, agent, fence, stamped):
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
                                    agent,
                                    stamp.previous_metadata,
                                ),
                            )
                            landed += _rows_affected(affected)
                    else:
                        # Nothing written, so nothing to roll back: the check
                        # is the first thing the transaction does.
                        moved = True
                stamped += landed
                if moved:
                    logger.info(
                        "history moved under the pass for %s; stopping after "
                        "%s rows, run it again (#3120)",
                        agent, stamped,
                    )
                    break

        for row_id, claimed, grouped in conflicts:
            logger.warning(
                "conversation row %s claims session %r and the transcript "
                "groups it under %r; left as it stands (#3120)",
                row_id, claimed, grouped,
            )
        if stamped or conflicts:
            logger.info(
                "%s legacy rows now name their session for %s, %s left as "
                "they stand (#3120)",
                stamped, agent, len(conflicts),
            )
        total["stamped"] += stamped
        total["refused"] += len(conflicts)
        total["skipped"] += skipped
    return total
