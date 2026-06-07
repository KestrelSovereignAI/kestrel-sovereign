"""Typed event log for codex sandbox-approval declines (#1581).

Even after #1575's ApprovalQueue bridge, codex's sandbox emits some
RPCs we intentionally keep declining (the elicitation/permissions
defaults in ``_DEFAULT_APPROVAL_REPLIES``, plus the bridge's own
fail-closed paths when no policy gate is available or a payload is
malformed). Today those declines are completely invisible to the
agent — no audit row, no chat-side signal, no surface that lets her
reason about a tool call that never landed.

This mirrors the #1571 pattern (operational lifecycle events): write
a typed row each time we decline, then render a one-line summary in
the always-on operational state block on the next turn.

Acceptance (Emma): event payload fields are ``tool``, ``request``,
``status``, ``reason``. Status is always ``"declined"`` for this
table (per-row); ``reason`` carries provenance: ``auto_default``,
``policy_deny:<rule>``, ``queue_denied``, ``no_cu_feature``, etc.

Rows are DID-scoped — shared-backend deployments (multi-agent
Postgres) must not leak one agent's declines into another agent's
next-turn context.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class CodexDeclineEvent:
    """One declined codex sandbox-approval RPC row."""

    id: str
    agent_id: str
    request: str          # e.g. "item/commandExecution/requestApproval"
    tool: str             # e.g. "gh issue create -R ..." (truncated)
    status: str           # always "declined" today
    reason: str           # auto_default | policy_deny:<rule> | queue_denied | ...
    created_at: str

    def to_public_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "agent_id": self.agent_id,
            "request": self.request,
            "tool": self.tool,
            "status": self.status,
            "reason": self.reason,
            "created_at": self.created_at,
        }


async def ensure_codex_decline_events_table(db) -> None:
    """Create the table + indices if they don't already exist."""
    await db.execute(
        """
        CREATE TABLE IF NOT EXISTS codex_decline_events (
            id TEXT PRIMARY KEY,
            agent_id TEXT NOT NULL,
            request TEXT NOT NULL,
            tool TEXT NOT NULL,
            status TEXT NOT NULL,
            reason TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )
    # Per-agent recency lookup for the operational-state block's
    # decline section. Sorted DESC so the section gets the newest
    # rows in one index seek.
    await db.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_codex_decline_events_agent_created
        ON codex_decline_events(agent_id, created_at DESC)
        """
    )


def _truncate(text: Optional[str], max_len: int = 200) -> str:
    if not text:
        return ""
    s = str(text).replace("\n", " ").strip()
    if len(s) <= max_len:
        return s
    return s[: max_len - 3] + "..."


async def record_decline(
    db,
    *,
    agent_id: str,
    request: str,
    tool: str,
    reason: str,
    status: str = "declined",
) -> CodexDeclineEvent:
    """Append one decline row. Always inserts — the audit trail must
    show every decline, even repeats.

    ``tool`` is truncated to 200 chars (operational-state block has
    its own size cap; this is defense in depth so a giant command
    payload can't bloat the DB row)."""
    row_id = uuid.uuid4().hex
    now = datetime.now(timezone.utc).isoformat()
    await db.execute(
        """
        INSERT INTO codex_decline_events (
            id, agent_id, request, tool, status, reason, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            row_id, agent_id, request, _truncate(tool),
            status, reason, now,
        ),
    )
    return CodexDeclineEvent(
        id=row_id,
        agent_id=agent_id,
        request=request,
        tool=_truncate(tool),
        status=status,
        reason=reason,
        created_at=now,
    )


async def list_recent_declines_for_agent(
    db,
    *,
    agent_id: str,
    limit: int = 20,
    since: Optional[str] = None,
) -> List[CodexDeclineEvent]:
    """Return the most recent decline rows for one agent, newest first.

    ``agent_id`` is REQUIRED so shared-backend deployments can never
    leak rows across agents. ``limit`` is clamped to [1, 1000];
    ``since`` filters to ``created_at >= since`` (ISO8601 string).
    """
    capped = max(1, min(1000, int(limit) if limit else 20))
    conds: List[str] = ["agent_id = ?"]
    args: List[Any] = [agent_id]
    if since:
        conds.append("created_at >= ?")
        args.append(since)
    args.append(capped)
    rows = await db.fetchall(
        f"""
        SELECT id, agent_id, request, tool, status, reason, created_at
        FROM codex_decline_events
        WHERE {' AND '.join(conds)}
        ORDER BY created_at DESC
        LIMIT ?
        """,
        tuple(args),
    )
    return [
        CodexDeclineEvent(
            id=r[0], agent_id=r[1], request=r[2], tool=r[3],
            status=r[4], reason=r[5], created_at=r[6],
        )
        for r in rows
    ]
