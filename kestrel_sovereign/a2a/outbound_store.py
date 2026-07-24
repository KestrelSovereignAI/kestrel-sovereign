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

* ``record_outbound_dispatch`` — write at the moment of successful
  POST (or after a transport-layer failure with ``error`` populated).
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

import json
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


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
            "terminal_state": self.terminal_state,
            "terminal_at": self.terminal_at,
            "error": self.error,
        }


async def ensure_a2a_outbound_tasks_table(db) -> None:
    """Create the table + indices if they don't already exist."""
    await db.execute(
        """
        CREATE TABLE IF NOT EXISTS a2a_outbound_tasks (
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
            terminal_state TEXT,
            terminal_at TEXT,
            error TEXT
        )
        """
    )
    # The outbound table is feature-owned, so it is created independently of
    # AsyncDatabase's core schema.  Existing agent databases predate stable
    # recipient identity; migrate them in the same backend-aware path used by
    # core schema migrations.  NULL remains the explicit legacy marker.
    await db._migrate_add_column(
        "a2a_outbound_tasks", "recipient_agent_id", "TEXT DEFAULT NULL",
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
) -> OutboundTask:
    """Persist one outbound-dispatch audit row.

    ``agent_id`` scopes the row to the sending agent — required for
    shared-backend deployments where multiple agents share one
    Postgres (codex review #1576 round 3 P1). Without it, one agent
    would see / update another agent's outbound rows.

    ``error`` is populated when the dispatch itself failed at the
    transport layer (the peer was unreachable, returned 5xx, etc.);
    in that case ``terminal_state`` is also set so the row is
    self-describing.
    """
    row_id = uuid.uuid4().hex
    now = datetime.now(timezone.utc).isoformat()
    terminal_state = "dispatch_failed" if error else None
    terminal_at = now if error else None
    await db.execute(
        """
        INSERT INTO a2a_outbound_tasks (
            id, agent_id, task_id, recipient, recipient_agent_id, verb, session_id,
            skill_id, dispatch_tool, message_summary, created_at,
            terminal_state, terminal_at, error
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            row_id, agent_id, task_id, recipient, recipient_agent_id, verb, session_id,
            skill_id, dispatch_tool, _summarize_message(message), now,
            terminal_state, terminal_at, error,
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
    Idempotent: updating with the same terminal_state twice is a
    no-op net of the ``terminal_at`` stamp.
    """
    now = datetime.now(timezone.utc).isoformat()
    affected = await db.execute(
        """
        UPDATE a2a_outbound_tasks
        SET terminal_state = ?, terminal_at = ?, error = COALESCE(?, error)
        WHERE agent_id = ? AND task_id = ? AND terminal_state IS NULL
        """,
        (terminal_state, now, error, agent_id, task_id),
    )
    # AsyncDatabase.execute returns rows-affected as int. Older test
    # doubles may return a cursor-like object with .rowcount, or None;
    # bound defensively.
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
    router reauthorizes it in the current requester scope.
    """
    rows = await db.fetchall(
        """
        SELECT id, agent_id, task_id, recipient, recipient_agent_id,
               verb, session_id, skill_id, dispatch_tool, message_summary,
               created_at, terminal_state, terminal_at, error
        FROM a2a_outbound_tasks
        WHERE agent_id = ? AND task_id = ?
        ORDER BY created_at DESC
        LIMIT 1
        """,
        (agent_id, task_id),
    )
    if not rows:
        return None
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
               created_at, terminal_state, terminal_at, error
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
        created_at=row[10], terminal_state=row[11], terminal_at=row[12],
        error=row[13],
    )
