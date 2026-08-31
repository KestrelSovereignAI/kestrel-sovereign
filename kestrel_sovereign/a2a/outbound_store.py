"""Sender-side outbound A2A task store (#1576).

When an agent dispatches an A2A task to a peer, the receiver's
``a2a_tasks`` row carries the full envelope — message, artifacts,
history, lifecycle. The SENDER, by contrast, has historically had no
durable record of the dispatch at all: ``_post_a2a_task`` POSTed to
the peer and returned the envelope; nothing was written locally. That
left every outbound dispatch invisible to the sending agent's own
audit / introspection surfaces (`list_recent_tasks` only sees inbound
work; `tool_call_log` doesn't carry A2A semantics).

This module provides the canonical sender-side outbound log table.
Each row carries the assertion Emma pinned in #1576:

> Every outbound A2A dispatch writes a sender-side outbound task
> record and an audit/log row containing ``task_id``, ``recipient``,
> ``verb``, ``created_at``, ``dispatch tool/path``, and
> ``terminal/error state when known``.

Lifecycle:

* ``record_outbound_dispatch`` — reserve the sender-owned task id and stable
  recipient identity before a router accepts delivery.  The local-host
  adapter keeps no-store operation as a best-effort compatibility path, but a
  persisted local route follows the same reserve-to-activate transition.
* ``update_outbound_terminal_state`` — invoked when the agent fetches
  the peer's result via ``get_peer_task_result`` and learns the final
  state, OR when the dispatch itself failed (we already know the
  terminal state is the failure).
* ``list_outbound_tasks`` — paginated query for the agent's
  introspection / preturn-state surfaces.

Schema is dedicated (not piggybacking on ``a2a_tasks``) because the
inbound-side rows carry receiver-shape fields (message, history,
artifacts) that don't apply to a sender-side audit row, and mixing
directions would muddle every existing query. Same architectural
choice as ``restart_status_events`` vs ``restart_requests``.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from kestrel_sovereign.storage.db.interface import QueryError


ROUTE_STATE_RESERVED = "reserved"
"""A hosted dispatch has a stable recipient but no accepted peer task id yet."""

ROUTE_STATE_ROUTABLE = "routable"
"""The stable recipient and peer task id form a safe retained route."""

ROUTE_STATE_AMBIGUOUS = "ambiguous"
"""A historical duplicate route which must never be selected for routing."""

# Ordered, because the CHECK constraint below is generated from it and a
# frozenset would spell the same vocabulary differently between runs — the
# stored DDL is compared as text to decide whether a rebuild is needed.
ROUTE_STATES = (
    ROUTE_STATE_RESERVED,
    ROUTE_STATE_ROUTABLE,
    ROUTE_STATE_AMBIGUOUS,
)

_ROUTE_STATES = frozenset(ROUTE_STATES)

# Generated from the constants above rather than written out again, so the
# schema's vocabulary cannot drift from the code's (#2804).
ROUTE_STATE_CHECK = "route_state IN ({})".format(
    ", ".join(f"'{state}'" for state in ROUTE_STATES)
)


class OutboundTaskRouteAmbiguousError(RuntimeError):
    """A retained task id has multiple historical outbound owners.

    ``None`` means that no sender-side audit row exists at all, which is the
    legacy local-host compatibility case.  A duplicate key is materially
    different: it is known to be unsafe and callers must never convert it
    into a display-name lookup.  Keeping that distinction explicit avoids
    routing a retained fetch to a same-name replacement peer.
    """


@dataclass(frozen=True)
class OutboundTask:
    """One sender-side outbound A2A dispatch row."""

    id: str
    agent_id: str
    task_id: str
    recipient: str
    recipient_agent_id: Optional[str]
    verb: str
    session_id: str
    skill_id: Optional[str]
    dispatch_tool: str
    message_summary: Optional[str]
    created_at: str
    route_state: str
    terminal_state: Optional[str]
    terminal_at: Optional[str]
    error: Optional[str]

    def to_public_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "agent_id": self.agent_id,
            "task_id": self.task_id,
            "recipient": self.recipient,
            "verb": self.verb,
            "session_id": self.session_id,
            "skill_id": self.skill_id,
            "dispatch_tool": self.dispatch_tool,
            "message_summary": self.message_summary,
            "created_at": self.created_at,
            "route_state": self.route_state,
            "terminal_state": self.terminal_state,
            "terminal_at": self.terminal_at,
            "error": self.error,
        }


# ONE canonical shape, used to create the table fresh and to rebuild an
# upgraded one into it (#2804). ``{table}`` is substituted so the rebuild can
# stage under a temporary name; a second hand-copied spelling is exactly the
# divergence this fixes.
A2A_OUTBOUND_TASKS_DDL = f"""
    CREATE TABLE {{table}} (
        id TEXT PRIMARY KEY,
        agent_id TEXT NOT NULL,
        task_id TEXT NOT NULL,
        recipient TEXT NOT NULL,
        recipient_agent_id TEXT,
        verb TEXT NOT NULL,
        session_id TEXT NOT NULL,
        skill_id TEXT,
        dispatch_tool TEXT NOT NULL,
        message_summary TEXT,
        created_at TEXT NOT NULL,
        route_state TEXT NOT NULL DEFAULT 'routable' CHECK (
            {ROUTE_STATE_CHECK}
        ),
        terminal_state TEXT,
        terminal_at TEXT,
        error TEXT
    )
"""


async def ensure_a2a_outbound_tasks_table(db) -> None:
    """Create the table + indices if they don't already exist."""
    await db.execute(
        A2A_OUTBOUND_TASKS_DDL.replace(
            "CREATE TABLE {table}", "CREATE TABLE IF NOT EXISTS a2a_outbound_tasks"
        )
    )
    # The outbound table is feature-owned, so it is created independently of
    # AsyncDatabase's core schema.  Existing agent databases predate stable
    # recipient identity and retained-route authorization; migrate them in the
    # same backend-aware path used by core schema migrations.  NULL remains
    # the explicit legacy marker for ``recipient_agent_id``.
    await db.migrate_columns_once(
        "a2a_outbound_tasks",
        (
            ("recipient_agent_id", "TEXT DEFAULT NULL"),
            ("route_state", "TEXT NOT NULL DEFAULT 'routable'"),
        ),
    )
    # Older rows were created before retained-route authorization existed.
    # Treat a unique historical task id as routable for local-host backwards
    # compatibility, but quarantine every historical collision before adding
    # the canonical-owner constraint below.  Keeping the rows preserves audit
    # history; marking them ambiguous makes ``get_outbound_task`` fail closed.
    #
    # Deliberately NOT passed to ``migrate_columns_once`` as a backfill keyed
    # on ``route_state``.  A keyed backfill runs only when this call is the one
    # adding the column, and the unique index below DEPENDS on the quarantine
    # having run: a database whose ``route_state`` predates that index (a build
    # between the two changes, or a crash between them) would skip the
    # quarantine and then fail to create the index, because the collisions it
    # exists to clear are still routable.  Running it unconditionally costs a
    # grouped scan per init and guarantees the index can always be built.  That
    # trade is the reason this is not the ``restart_requests`` pattern.
    await db.execute(
        """
        UPDATE a2a_outbound_tasks
        SET route_state = 'ambiguous'
        WHERE route_state = 'routable'
          AND (agent_id, task_id) IN (
              SELECT agent_id, task_id
              FROM a2a_outbound_tasks
              WHERE route_state = 'routable'
              GROUP BY agent_id, task_id
              HAVING COUNT(*) > 1
          )
        """
    )
    # A database that gained ``route_state`` by ALTER carries it WITHOUT the
    # CHECK the fresh CREATE declares, permanently and undetectably (#2804).
    # ``route_state`` is routing *authorization* state and ``get_outbound_task``
    # fails closed on exactly one of its values, so "the column holds one of
    # three known values" has to be enforced rather than merely intended.
    #
    # The remediation runs first because both backends refuse to add a
    # constraint rows already violate. Anything outside the vocabulary is
    # quarantined to ``ambiguous`` rather than guessed into ``routable``: an
    # unrecognised authorization state must fail closed, and the rows are kept
    # so the audit history survives.
    await db.ensure_check_constraint(
        "a2a_outbound_tasks",
        "a2a_outbound_tasks_route_state_check",
        ROUTE_STATE_CHECK,
        canonical_ddl=A2A_OUTBOUND_TASKS_DDL,
        remediation=(
            "UPDATE a2a_outbound_tasks SET route_state = 'ambiguous' "
            f"WHERE route_state NOT IN ({', '.join('?' * len(ROUTE_STATES))})",
            ROUTE_STATES,
        ),
    )
    # ``NOT EXISTS`` in a rekey UPDATE is only a snapshot predicate.  It is
    # not an ownership invariant under PostgreSQL's READ COMMITTED isolation:
    # two concurrent rekeys can both observe no owner.  The partial unique
    # index makes only one route canonical.  Reserved rows intentionally do
    # not participate until their peer task id is accepted; ambiguous legacy
    # rows remain preserved but non-routable.
    await db.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS uq_a2a_outbound_tasks_canonical_route
        ON a2a_outbound_tasks(agent_id, task_id)
        WHERE route_state = 'routable'
        """
    )
    # Per-agent recency listing — the introspection surface, and the
    # multi-agent shared-backend safety guard (codex review #1576
    # round 3 P1). Without ``agent_id`` scoping, ``pending_a2a_questions``
    # already documents the precedent: shared-Postgres deployments
    # MUST scope by DID or one agent sees / overwrites another's rows.
    await db.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_a2a_outbound_tasks_agent_created
        ON a2a_outbound_tasks(agent_id, created_at DESC)
        """
    )
    # Per-agent task_id lookup for get_peer_task_result terminal stamp.
    # Compound index so the (agent_id, task_id) filter is index-served.
    await db.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_a2a_outbound_tasks_agent_task
        ON a2a_outbound_tasks(agent_id, task_id, created_at DESC)
        """
    )
    # Per-(agent, recipient) filter for "what did I send Claw lately?".
    await db.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_a2a_outbound_tasks_agent_recipient
        ON a2a_outbound_tasks(agent_id, recipient, created_at DESC)
        """
    )


def _summarize_message(message: Optional[str], max_len: int = 200) -> Optional[str]:
    """Truncate the outbound message for the audit row.

    Mirrors ``ApprovalQueue._summarize_args``: bounded width so a
    large payload doesn't bloat the audit log; never returns more
    than ``max_len`` characters.
    """
    if not message:
        return None
    s = str(message).replace("\n", " ").strip()
    if len(s) <= max_len:
        return s
    return s[: max_len - 3] + "..."


async def record_outbound_dispatch(
    db,
    *,
    agent_id: str,
    task_id: str,
    recipient: str,
    verb: str,
    session_id: str,
    dispatch_tool: str,
    recipient_agent_id: Optional[str] = None,
    skill_id: Optional[str] = None,
    message: Optional[str] = None,
    error: Optional[str] = None,
    route_state: str = ROUTE_STATE_ROUTABLE,
) -> OutboundTask:
    """Persist one outbound-dispatch audit row.

    ``agent_id`` scopes the row to the sending agent — required for
    shared-backend deployments where multiple agents share one
    Postgres (codex review #1576 round 3 P1). Without it, one agent
    would see / update another agent's outbound rows.

    ``error`` is populated when the dispatch itself failed at the transport
    layer (the peer was unreachable, returned 5xx, etc.); in that case
    ``terminal_state`` is also set so the row is self-describing.

    Routers reserve with ``route_state='reserved'`` before they send anything.
    A reservation is deliberately non-routable even though it has a stable
    recipient: the peer has not yet accepted a canonical task id.
    ``rekey_outbound_task(..., activate=True)`` is the only state transition
    to ``'routable'``.
    """
    if route_state not in _ROUTE_STATES:
        raise ValueError(f"Unknown outbound route state: {route_state!r}")
    if route_state == ROUTE_STATE_RESERVED and (
        not isinstance(recipient_agent_id, str) or not recipient_agent_id.strip()
    ):
        raise ValueError(
            "A reserved outbound route requires a stable recipient identity"
        )
    row_id = uuid.uuid4().hex
    now = datetime.now(timezone.utc).isoformat()
    terminal_state = "dispatch_failed" if error else None
    terminal_at = now if error else None
    await db.execute(
        """
        INSERT INTO a2a_outbound_tasks (
            id, agent_id, task_id, recipient, recipient_agent_id, verb, session_id,
            skill_id, dispatch_tool, message_summary, created_at, route_state,
            terminal_state, terminal_at, error
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            row_id, agent_id, task_id, recipient, recipient_agent_id, verb, session_id,
            skill_id, dispatch_tool, _summarize_message(message), now,
            route_state, terminal_state, terminal_at, error,
        ),
    )
    return OutboundTask(
        id=row_id,
        agent_id=agent_id,
        task_id=task_id,
        recipient=recipient,
        recipient_agent_id=recipient_agent_id,
        verb=verb,
        session_id=session_id,
        skill_id=skill_id,
        dispatch_tool=dispatch_tool,
        message_summary=_summarize_message(message),
        created_at=now,
        route_state=route_state,
        terminal_state=terminal_state,
        terminal_at=terminal_at,
        error=error,
    )


async def update_outbound_terminal_state(
    db,
    *,
    agent_id: str,
    task_id: str,
    terminal_state: str,
    error: Optional[str] = None,
) -> int:
    """Stamp the terminal lifecycle state on the matching row(s).

    ``agent_id`` scopes the update so an agent can never overwrite a
    peer's outbound row even on a task_id collision (codex review
    #1576 round 3 P1).

    Returns the number of rows updated. If the task_id was never
    recorded (or the audit table was dropped), returns 0 — never
    raises, so the cognition turn doesn't break on a stale fetch.
    Idempotent: updating with the same terminal_state twice is a no-op net of
    the ``terminal_at`` stamp. An authoritative remote cancellation may
    supersede sender-local provisional states (deadline expiry or a dispatch
    failure); it may not overwrite a contradictory remote terminal result.
    """
    now = datetime.now(timezone.utc).isoformat()
    affected = await db.execute(
        """
        UPDATE a2a_outbound_tasks
        SET terminal_state = ?, terminal_at = ?, error = COALESCE(?, error)
        WHERE agent_id = ? AND task_id = ?
          AND (
              terminal_state IS NULL
              OR (
                  ? = 'canceled'
                  AND terminal_state IN ('expired', 'dispatch_failed')
              )
          )
        """,
        (terminal_state, now, error, agent_id, task_id, terminal_state),
    )
    # AsyncDatabase.execute returns rows-affected as int. Older test
    # doubles may return a cursor-like object with .rowcount, or None;
    # bound defensively.
    if isinstance(affected, int):
        return affected
    return int(getattr(affected, "rowcount", 0) or 0)


async def rekey_outbound_task(
    db,
    *,
    record_id: str,
    agent_id: str,
    old_task_id: str,
    new_task_id: str,
    recipient_agent_id: str,
    activate: bool = False,
) -> int:
    """Atomically move an outbound id and optionally activate its route.

    An A2A sender includes its UUID in the envelope and compliant peers echo
    it.  This narrow migration path preserves compatibility with older peers
    that return a different id: the stable recipient binding moves only when
    the exact sender-owned row still has the recipient identity that was
    authorized for the delivery.  With ``activate=True`` (the hosted path),
    this is also the sole ``reserved -> routable`` transition and therefore
    runs even if the peer echoed ``old_task_id`` unchanged.  A caller must
    treat a zero-row result as a failed dispatch rather than route a retained
    request by a display name.

    In particular, a peer-controlled response must not be able to reuse an id
    already bound to another outbound task.  The unique canonical-route index
    enforces that invariant under concurrency; a uniqueness conflict is a
    failed rekey which leaves the source reservation intact.
    """
    if not isinstance(recipient_agent_id, str) or not recipient_agent_id.strip():
        raise ValueError("recipient_agent_id must be a non-empty stable identity")
    expected_route_state = (
        ROUTE_STATE_RESERVED if activate else ROUTE_STATE_ROUTABLE
    )
    try:
        affected = await db.execute(
            """
            UPDATE a2a_outbound_tasks
            SET task_id = ?, route_state = ?
            WHERE id = ? AND agent_id = ? AND task_id = ?
              AND recipient_agent_id = ? AND route_state = ?
            """,
            (
                new_task_id,
                ROUTE_STATE_ROUTABLE,
                record_id,
                agent_id,
                old_task_id,
                recipient_agent_id,
                expected_route_state,
            ),
        )
    except QueryError as exc:
        # Backends wrap their native integrity errors differently.  The two
        # supported forms share these substrings (SQLite: ``UNIQUE constraint
        # failed``; PostgreSQL: ``duplicate key value violates unique
        # constraint``).  A collision is an expected failed rekey, not a
        # storage outage.  Other failures propagate so callers can report the
        # failed durable binding without pretending it was a collision.
        message = str(exc).lower()
        if "unique constraint" in message or "duplicate key value" in message:
            return 0
        raise
    if isinstance(affected, int):
        return affected
    return int(getattr(affected, "rowcount", 0) or 0)


async def get_outbound_task(
    db,
    *,
    agent_id: str,
    task_id: str,
) -> Optional[OutboundTask]:
    """Return one sender-owned outbound task for retained-route recovery.

    A task id can be supplied to a result fetch long after the display name
    used at dispatch has changed.  The stored stable identity is intentionally
    internal: callers receive the historical display recipient through the
    normal public task view, while routing uses this field only after the
    router reauthorizes it in the current requester scope.  Returns ``None``
    only when the key is absent.  Raises ``OutboundTaskRouteAmbiguousError``
    when historical rows disagree, so callers can fail closed rather than
    treating known-unsafe history as a no-record legacy route.
    """
    rows = await db.fetchall(
        """
        SELECT id, agent_id, task_id, recipient, recipient_agent_id,
               verb, session_id, skill_id, dispatch_tool, message_summary,
               created_at, route_state, terminal_state, terminal_at, error
        FROM a2a_outbound_tasks
        WHERE agent_id = ? AND task_id = ?
        ORDER BY created_at DESC
        LIMIT 2
        """,
        (agent_id, task_id),
    )
    # A retained task id is a capability-like route key.  A historical or
    # externally-corrupted duplicate must never make us arbitrarily select a
    # newer row and thereby redirect a result fetch to another peer.  New
    # rekeys reject this state atomically; this guard protects old databases
    # too without deleting either audit record.
    if not rows:
        return None
    if len(rows) > 1:
        raise OutboundTaskRouteAmbiguousError(
            "Multiple outbound task records share this retained route key"
        )
    return _outbound_task_from_row(rows[0])


async def list_outbound_tasks(
    db,
    *,
    agent_id: str,
    limit: int = 50,
    recipient: Optional[str] = None,
    since: Optional[str] = None,
) -> List[OutboundTask]:
    """Return the most recent outbound rows for one agent, newest first.

    ``agent_id`` is REQUIRED so the introspection surface can never
    leak a peer's outbound dispatches in a shared-backend
    deployment (codex review #1576 round 3 P1).

    ``limit`` is clamped to [1, 1000] to defend against runaway
    callers asking for everything. ``recipient`` filters to one peer;
    ``since`` filters to ``created_at >= since`` (ISO8601 string).
    """
    capped = max(1, min(1000, int(limit) if limit else 50))
    conds: List[str] = ["agent_id = ?"]
    args: List[Any] = [agent_id]
    if recipient:
        conds.append("recipient = ?")
        args.append(recipient)
    if since:
        conds.append("created_at >= ?")
        args.append(since)
    where = "WHERE " + " AND ".join(conds)
    args.append(capped)
    rows = await db.fetchall(
        f"""
        SELECT id, agent_id, task_id, recipient, recipient_agent_id,
               verb, session_id, skill_id, dispatch_tool, message_summary,
               created_at, route_state, terminal_state, terminal_at, error
        FROM a2a_outbound_tasks
        {where}
        ORDER BY created_at DESC
        LIMIT ?
        """,
        tuple(args),
    )
    return [_outbound_task_from_row(row) for row in rows]


def _outbound_task_from_row(row) -> OutboundTask:
    """Map the canonical outbound-task SELECT order to its value object."""
    return OutboundTask(
        id=row[0], agent_id=row[1], task_id=row[2], recipient=row[3],
        recipient_agent_id=row[4], verb=row[5], session_id=row[6],
        skill_id=row[7], dispatch_tool=row[8], message_summary=row[9],
        created_at=row[10], route_state=row[11], terminal_state=row[12],
        terminal_at=row[13], error=row[14],
    )
