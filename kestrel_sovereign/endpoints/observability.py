"""Observability endpoint - query A2A observability events for debugging."""
from fastapi import APIRouter, Request, Query, HTTPException
from typing import Optional, Dict, Any, List
from datetime import datetime, timezone, timedelta
import logging

from kestrel_sovereign.endpoints.agent_helpers import get_agent

logger = logging.getLogger(__name__)

router = APIRouter(tags=["observability"])


@router.get("/api/observability/events")
async def get_observability_events(
    request: Request,
    agent_name: Optional[str] = Query(None, description="Filter by agent name/DID"),
    event_type: Optional[str] = Query(None, description="Filter by event type (tool_call, tool_response, agent_response, error, metric)"),
    session_id: Optional[str] = Query(None, description="Filter by session ID"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum events to return"),
) -> Dict[str, Any]:
    """
    Query observability events from the A2A observability store.

    Useful for debugging the agentic tool-calling loop:
    - See which tools were built and passed to LLM
    - Track LLM call timing
    - See if LLM returned tool_calls or just text
    - Monitor feature dispatch success/failure

    Event types:
    - `tool_call`: Start of a tool/LLM call (with metadata)
    - `tool_response`: Completion of a tool/LLM call (with timing)
    - `agent_response`: Agent response events
    - `error`: Error events (e.g., tool_calling_ignored)
    - `metric`: Metrics (e.g., feature_tools_built count)
    """
    try:
        agent = get_agent(request)
    except HTTPException:
        raise

    obs_store = getattr(agent, 'observability_store', None)
    if not obs_store:
        raise HTTPException(status_code=503, detail="Observability store not available")

    try:
        events = await obs_store.query_events(
            agent_name=agent_name,
            event_type=event_type,
            session_id=session_id,
            limit=limit,
        )

        # Convert to dicts for JSON response
        event_dicts = []
        for e in events:
            event_dicts.append({
                "event_id": e.event_id,
                "timestamp": str(e.timestamp) if e.timestamp else None,
                "agent_name": e.agent_name,
                "session_id": e.session_id,
                "event_type": e.event_type,
                "tool_name": e.tool_name,
                "duration_ms": e.duration_ms,
                "success": e.success,
                "error_message": e.error_message,
                "metadata": e.metadata,
            })

        return {
            "events": event_dicts,
            "count": len(event_dicts),
            "filters": {
                "agent_name": agent_name,
                "event_type": event_type,
                "session_id": session_id,
                "limit": limit,
            }
        }
    except Exception as e:
        logger.error(f"Error querying observability events: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to query observability events")


@router.get("/api/observability/summary")
async def get_observability_summary(
    request: Request,
    minutes: int = Query(60, ge=1, le=1440, description="Time window in minutes"),
) -> Dict[str, Any]:
    """
    Get a summary of recent observability events.

    Provides counts by event type and recent errors for quick health check.
    """
    try:
        agent = get_agent(request)
    except HTTPException:
        raise

    obs_store = getattr(agent, 'observability_store', None)
    if not obs_store:
        raise HTTPException(status_code=503, detail="Observability store not available")

    try:
        from datetime import timedelta
        from kestrel_sovereign.kestrel_config.constants import DEFAULT_OBSERVABILITY_LIMIT
        since = datetime.now(timezone.utc) - timedelta(minutes=minutes)

        # Query recent events
        events = await obs_store.query_events(
            since=since,
            limit=DEFAULT_OBSERVABILITY_LIMIT,
        )

        # Count by event type
        type_counts = {}
        error_count = 0
        errors = []
        tool_call_count = 0
        avg_duration_ms = 0
        durations = []
        # Break metrics out by name so otherwise-"dark" forensic metrics
        # (e.g. assistant_turn_persist_failed, #969) are visible at a glance
        # instead of being lumped into a single events_by_type["metric"] count.
        metrics_by_name: Dict[str, int] = {}

        for e in events:
            type_counts[e.event_type] = type_counts.get(e.event_type, 0) + 1

            if e.event_type == "error":
                error_count += 1
                if len(errors) < 10:  # Keep last 10 errors
                    errors.append({
                        "timestamp": str(e.timestamp),
                        "error_type": e.metadata.get("error_type") if e.metadata else None,
                        "error_message": e.error_message,
                    })

            if e.event_type == "metric" and e.metadata:
                name = e.metadata.get("metric_name")
                if name:
                    metrics_by_name[name] = metrics_by_name.get(name, 0) + 1

            if e.event_type == "tool_response" and e.duration_ms:
                durations.append(e.duration_ms)

        if durations:
            avg_duration_ms = sum(durations) / len(durations)

        return {
            "time_window_minutes": minutes,
            "total_events": len(events),
            "events_by_type": type_counts,
            "metrics_by_name": metrics_by_name,
            "error_count": error_count,
            "recent_errors": errors,
            "tool_responses_count": len(durations),
            "avg_tool_duration_ms": round(avg_duration_ms, 2) if durations else None,
        }
    except Exception as e:
        logger.error(f"Error getting observability summary: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to get observability summary")


@router.get("/api/observability/metrics/{metric_name}")
async def get_metric_summary(
    request: Request,
    metric_name: str,
    minutes: int = Query(1440, ge=1, le=43200, description="Time window in minutes (default 24h)"),
    agent_name: Optional[str] = Query(None, description="Filter to a single agent"),
) -> Dict[str, Any]:
    """Summarize a single named metric over a recent window.

    Surfaces otherwise-"dark" forensic metrics — notably
    ``assistant_turn_persist_failed`` (#969), emitted when the cancellation-safe
    assistant-turn persist falls back to the error path — with count, last_seen,
    per-agent breakdown, and recent samples carrying the metric's metadata
    (e.g. ``error_type``/``session_id``). Generic: works for any metric name.
    """
    try:
        agent = get_agent(request)
    except HTTPException:
        raise

    obs_store = getattr(agent, 'observability_store', None)
    if not obs_store:
        raise HTTPException(status_code=503, detail="Observability store not available")

    try:
        since = datetime.now(timezone.utc) - timedelta(minutes=minutes)
        summary = await obs_store.get_metric_summary(
            metric_name,
            agent_name=agent_name,
            since=since,
        )
        summary["time_window_minutes"] = minutes
        return summary
    except Exception as e:
        logger.error(f"Error getting metric summary for {metric_name}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to get metric summary")
