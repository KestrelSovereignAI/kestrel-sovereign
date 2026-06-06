"""Durable typed-event store for restart_status events (#1562).

#1551 introduced the SSE-only ``restart_status`` UI event. That fixed
the live-chat visibility gap but left two architectural concerns open:

1. Events were ephemeral — a browser reload, a session reattach, or
   any history navigation lost the lifecycle trail.
2. The dedupe contract was implicit in the SSE payload's full JSON;
   the frontend ended up keying on volatile fields (``deferral_reason``
   contains a changing ``oldest 63s of 900s stale window`` substring
   after #1558), so every coordinator poll looked like a new state
   and produced duplicate bubbles (#1560).

This module persists every emit as a typed event row keyed by a stable
``dedupe_signature = "{request_id}:{state}"``. The frontend uses that
signature as the find-or-update key; the agent's pre-turn state block
(``preturn_state._restart_status_section``) reads the most recent rows
to surface restart lifecycle context as **non-instructional** state —
not as a developer message and not as an instruction-bearing system
prompt.

Schema is additive: no existing column is modified.
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# Canonical lifecycle states the coordinator emits. Surfaced for the
# frontend / agent context so they don't have to mirror the string
# constants. ``deferred`` is a distinct state from ``pending`` — the
# row stays ``pending`` in restart_requests but the coordinator
# emitted a deferral notification, which is what the UI bubble
# represents.
LIFECYCLE_STATES = (
    "pending",
    "deferred",
    "updating",
    "executing",
    "completed",
    "rejected",
    "canceled",
)


def dedupe_signature(request_id: str, state: str) -> str:
    """Stable client-side dedupe key for a restart status emit.

    ``{request_id}:{state}`` collapses repeated pending/deferred polls
    for the same request — the volatile age text in ``deferral_reason``
    is intentionally excluded so coordinator polling does not spawn
    duplicate bubbles (#1560).
    """
    return f"{request_id}:{state}"


@dataclass
class RestartStatusEvent:
    id: str
    request_id: str
    state: str
    agent_id: str
    operation: str
    urgency: str
    policy: str
    dedupe_signature: str
    payload_json: str
    created_at: str

    @classmethod
    def from_row(cls, row) -> "RestartStatusEvent":
        cols = list(row)
        return cls(
            id=str(cols[0]),
            request_id=str(cols[1] or ""),
            state=str(cols[2] or ""),
            agent_id=str(cols[3] or ""),
            operation=str(cols[4] or "restart_only"),
            urgency=str(cols[5] or "normal"),
            policy=str(cols[6] or "idle_agents_only"),
            dedupe_signature=str(cols[7] or ""),
            payload_json=str(cols[8] or "{}"),
            created_at=str(cols[9] or ""),
        )

    def to_public_dict(self) -> Dict[str, Any]:
        try:
            payload = json.loads(self.payload_json)
        except (ValueError, TypeError):
            payload = {}
        return {
            "id": self.id,
            "request_id": self.request_id,
            "state": self.state,
            "agent_id": self.agent_id,
            "operation": self.operation,
            "urgency": self.urgency,
            "policy": self.policy,
            "dedupe_signature": self.dedupe_signature,
            "payload": payload,
            "created_at": self.created_at,
        }


async def ensure_restart_status_events_table(db) -> None:
    """Create the table + indices if they don't already exist."""
    await db.execute(
        """
        CREATE TABLE IF NOT EXISTS restart_status_events (
            id TEXT PRIMARY KEY,
            request_id TEXT NOT NULL,
            state TEXT NOT NULL,
            agent_id TEXT NOT NULL,
            operation TEXT DEFAULT 'restart_only',
            urgency TEXT DEFAULT 'normal',
            policy TEXT DEFAULT 'idle_agents_only',
            dedupe_signature TEXT NOT NULL,
            payload_json TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )
    # Per-request lookup for the SSE replay + the chat history page.
    await db.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_restart_status_events_request
        ON restart_status_events(request_id, created_at)
        """
    )
    # Per-agent recency for the preturn_state restart-status section.
    await db.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_restart_status_events_agent
        ON restart_status_events(agent_id, created_at DESC)
        """
    )
    # Frontend find-or-update by stable signature.
    await db.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_restart_status_events_dedupe
        ON restart_status_events(dedupe_signature, created_at DESC)
        """
    )


async def record_event(
    db,
    *,
    request_id: str,
    state: str,
    agent_id: str,
    payload: Dict[str, Any],
    operation: str = "restart_only",
    urgency: str = "normal",
    policy: str = "idle_agents_only",
) -> RestartStatusEvent:
    """Append one typed event row. Always inserts — no per-signature
    dedupe at the DB layer; the audit trail must be complete so the
    operator can see every coordinator poll. Frontend dedupe collapses
    same-signature rows into a single visible bubble.
    """
    event_id = uuid.uuid4().hex
    now = datetime.now(timezone.utc).isoformat()
    sig = dedupe_signature(str(request_id), str(state))
    payload_str = json.dumps(payload, default=str, sort_keys=True)
    await db.execute(
        """
        INSERT INTO restart_status_events (
            id, request_id, state, agent_id, operation, urgency,
            policy, dedupe_signature, payload_json, created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            event_id, request_id, state, agent_id, operation,
            urgency, policy, sig, payload_str, now,
        ),
    )
    return RestartStatusEvent(
        id=event_id,
        request_id=str(request_id),
        state=str(state),
        agent_id=str(agent_id),
        operation=str(operation),
        urgency=str(urgency),
        policy=str(policy),
        dedupe_signature=sig,
        payload_json=payload_str,
        created_at=now,
    )


async def list_events_for_request(
    db, request_id: str,
) -> List[RestartStatusEvent]:
    """Full lifecycle trail for one request, chronological."""
    rows = await db.fetchall(
        """
        SELECT id, request_id, state, agent_id, operation, urgency,
               policy, dedupe_signature, payload_json, created_at
        FROM restart_status_events
        WHERE request_id = ?
        ORDER BY created_at ASC
        """,
        (str(request_id),),
    )
    return [RestartStatusEvent.from_row(r) for r in rows]


async def list_recent_events_for_history(
    db, *, limit: int = 100, since: Optional[str] = None,
) -> List[RestartStatusEvent]:
    """Most-recent events across all requests, newest-first.

    Used by chat-history reload to repaint the visible bubble trail.
    ``since`` is an ISO timestamp; rows newer than that are returned
    (lets the frontend page lazily through history).
    """
    if since is None:
        rows = await db.fetchall(
            """
            SELECT id, request_id, state, agent_id, operation, urgency,
                   policy, dedupe_signature, payload_json, created_at
            FROM restart_status_events
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (int(limit),),
        )
    else:
        rows = await db.fetchall(
            """
            SELECT id, request_id, state, agent_id, operation, urgency,
                   policy, dedupe_signature, payload_json, created_at
            FROM restart_status_events
            WHERE created_at > ?
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (str(since), int(limit)),
        )
    return [RestartStatusEvent.from_row(r) for r in rows]


async def list_recent_events_for_agent_context(
    db, *, agent_id: str, limit: int = 10,
) -> List[RestartStatusEvent]:
    """Last ``limit`` events relevant to one agent's preturn snapshot.

    Used by ``preturn_state._restart_status_section`` to render a
    non-instructional summary of recent restart lifecycle into the
    AGENT STATE block. The result is intentionally small (a handful
    of rows) so it never crowds the real prompt.
    """
    rows = await db.fetchall(
        """
        SELECT id, request_id, state, agent_id, operation, urgency,
               policy, dedupe_signature, payload_json, created_at
        FROM restart_status_events
        WHERE agent_id = ?
        ORDER BY created_at DESC
        LIMIT ?
        """,
        (str(agent_id), int(limit)),
    )
    return [RestartStatusEvent.from_row(r) for r in rows]


async def latest_event_for_signature(
    db, dedupe_sig: str,
) -> Optional[RestartStatusEvent]:
    """Return the most recent event with this dedupe_signature.

    Used by clients that want to know whether they have already
    surfaced this exact (request, state) pair before — they keep
    polling cheap by comparing only the event id.
    """
    rows = await db.fetchall(
        """
        SELECT id, request_id, state, agent_id, operation, urgency,
               policy, dedupe_signature, payload_json, created_at
        FROM restart_status_events
        WHERE dedupe_signature = ?
        ORDER BY created_at DESC
        LIMIT 1
        """,
        (str(dedupe_sig),),
    )
    if not rows:
        return None
    return RestartStatusEvent.from_row(rows[0])
