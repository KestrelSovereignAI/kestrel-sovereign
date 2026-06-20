"""Restart status-event API — repaint the bubble trail on chat reload.

The live restart lifecycle bubbles (requesting → executing → completed)
paint via the ``restart_status`` SSE side-channel (#1551) but vanish on
reload because nothing re-fetches them. The typed event records persisted
by #1562 are the durable home for that trail; this endpoint exposes them
to the Console so ``history.js`` can repaint the bubbles when a
conversation is loaded (#1816).

Mounted dynamically via ``RestartCoordinatorFeature.get_router()`` — it is
NOT a core router, so it only exists when the restart feature is loaded.
"""

import logging
from typing import Any, Dict

from fastapi import APIRouter, Query, Request

from kestrel_sovereign.endpoints.agent_helpers import get_agent
from kestrel_sovereign.features.storage_access import resolve_feature_database

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/restart", tags=["restart"])


@router.get("/status-events")
async def get_restart_status_events(
    request: Request,
    session: str = Query("", description="Filter to this origin session"),
    limit: int = Query(200, ge=1, le=1000),
) -> Dict[str, Any]:
    """Recent restart_status lifecycle events, newest-first.

    When ``session`` is given, only events whose originating request was
    filed from that chat session (``origin_session_id``, persisted in the
    event payload as of #1816) are returned — so reloading conversation A
    never repaints restart bubbles filed from conversation B.
    """
    agent = get_agent(request)
    db = resolve_feature_database(agent)
    if db is None:
        return {"events": [], "count": 0}

    try:
        from kestrel_sovereign.features.restart_coordinator.event_store import (
            list_recent_events_for_history,
        )

        rows = await list_recent_events_for_history(db, limit=limit)
    except Exception as e:
        # Restart feature not loaded / table absent — no trail to repaint.
        logger.debug("restart status-events lookup unavailable: %s", e)
        return {"events": [], "count": 0}

    events = [r.to_public_dict() for r in rows]
    wanted = session.strip()
    if wanted:
        events = [
            e
            for e in events
            if str((e.get("payload") or {}).get("origin_session_id") or "")
            == wanted
        ]

    return {"events": events, "count": len(events)}
