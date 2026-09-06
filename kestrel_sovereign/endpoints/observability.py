"""Observability endpoint - query A2A observability events for debugging."""
from fastapi import APIRouter, Request, Query, HTTPException
from typing import Dict, Any
from datetime import datetime, timezone, timedelta
import logging

from kestrel_sovereign.endpoints.agent_helpers import get_agent

logger = logging.getLogger(__name__)

router = APIRouter(tags=["observability"])


def _scope_did(agent) -> str:
    """The value `a2a_observability.agent_name` actually holds: the DID.

    Despite the column's name it is not the display name. Every live
    recorder writes `agent_name=self.did` — ten call sites across
    `agent/streaming.py`, `agent/orchestrator_engine.py` and
    `kestrel_agent.py` — and `a2a/task_manager.py` documents its
    parameter as "Durable DID". `kestrel_agent.py` says so itself where
    it wires the ephemeral purge: "Tool-call args in a2a_observability
    use the agent DID as agent_name ... so scope by DID on both
    columns".

    Scoping by `agent.agent_name` instead is not a narrower read, it is
    an empty one: no row carries that value, so every panel would go
    blank and the #969 forensic metric would answer "never happened".

    Missing identity refuses rather than falls back. An unscoped query
    is the defect this function exists to prevent, so it must not be
    what happens when the identity cannot be resolved.
    """
    did = getattr(agent, "did", None)
    if not did:
        raise HTTPException(
            status_code=503,
            detail="Agent identity unavailable; cannot scope observability.",
        )
    return did

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

    # Resolved BEFORE the try below, which turns every exception into a
    # generic 500 — a refusal that says "identity unavailable" is not the
    # same answer as "the query blew up", and only the first tells the
    # caller the read was declined rather than attempted.
    scope_did = _scope_did(agent)

    try:
        from datetime import timedelta
        from kestrel_sovereign.kestrel_config.constants import DEFAULT_OBSERVABILITY_LIMIT
        since = datetime.now(timezone.utc) - timedelta(minutes=minutes)

        # Scoped to the routed agent's own rows. Without the predicate
        # this returned every agent's events whenever the store is
        # shared — which it is on PostgreSQL, where one table serves the
        # whole host. SQLite-per-agent masked it: the file boundary was
        # doing the scoping, so the query never needed to (#3215).
        #
        # The identity comes from the resolved agent, never from the
        # request. Routing to an agent is not authority over it.
        events = await obs_store.query_events(
            agent_name=scope_did,
            since=since,
            limit=DEFAULT_OBSERVABILITY_LIMIT,
        )

        # Count by event type
        type_counts = {}
        error_count = 0
        errors = []
        avg_duration_ms = 0
        durations = []
        # Break metrics out by name so otherwise-"dark" forensic metrics
        # (e.g. assistant_turn_persist_failed, #969) are visible at a glance
        # instead of being lumped into a single events_by_type["metric"] count.
        metrics_by_name: Dict[str, int] = {}
        # Pre-aggregated series for the Metrics panel charts (#2317). Core no
        # longer serves the agent-scoped raw event feed (the fleet host feature
        # owns it), so the per-agent panel renders its Event Timeline and Tool
        # Duration charts from these aggregates instead of shipping the raw
        # event list to the browser.
        timeline: Dict[str, Dict[str, int]] = {}
        tool_duration_samples: Dict[str, list] = {}

        for e in events:
            type_counts[e.event_type] = type_counts.get(e.event_type, 0) + 1

            # 5-minute timeline buckets, counted per event type. Floors the
            # timestamp to the bucket boundary the same way the console used to.
            ts = getattr(e, "timestamp", None)
            if ts is not None and hasattr(ts, "replace"):
                try:
                    bucket = ts.replace(
                        minute=(ts.minute // 5) * 5, second=0, microsecond=0
                    )
                    bkey = bucket.isoformat()
                    etype = e.event_type or "unknown"
                    slot = timeline.setdefault(bkey, {})
                    slot[etype] = slot.get(etype, 0) + 1
                except (AttributeError, TypeError, ValueError):
                    pass

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
                tool_name = getattr(e, "tool_name", None) or "unknown"
                tool_duration_samples.setdefault(tool_name, []).append(e.duration_ms)

        if durations:
            avg_duration_ms = sum(durations) / len(durations)

        # Collapse the per-tool samples into average durations for the chart.
        tool_durations = {
            name: round(sum(samples) / len(samples), 2)
            for name, samples in tool_duration_samples.items()
            if samples
        }

        return {
            "time_window_minutes": minutes,
            "total_events": len(events),
            "events_by_type": type_counts,
            "metrics_by_name": metrics_by_name,
            "timeline": timeline,
            "tool_durations": tool_durations,
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
) -> Dict[str, Any]:
    """Summarize a single named metric over a recent window.

    Surfaces otherwise-"dark" forensic metrics — notably
    ``assistant_turn_persist_failed`` (#969), emitted when the cancellation-safe
    assistant-turn persist falls back to the error path — with count, last_seen,
    per-agent breakdown, and recent samples carrying the metric's metadata
    (e.g. ``error_type``/``session_id``). Generic: works for any metric name.

    Scoped to the routed agent. This used to take ``agent_name`` as a
    query parameter: omitting it summarised every agent's rows, and
    supplying it addressed another agent's (#3215). A caller-supplied
    identity is a request, not an authority, so the parameter is gone
    rather than validated — validating it would answer "that agent has
    no such metric" differently from "that agent is not you", which is
    the same leak one step removed. Fleet-wide observability belongs on
    an explicit sovereign surface, not on an agent-routed read.
    """
    try:
        agent = get_agent(request)
    except HTTPException:
        raise

    obs_store = getattr(agent, 'observability_store', None)
    if not obs_store:
        raise HTTPException(status_code=503, detail="Observability store not available")

    scope_did = _scope_did(agent)

    try:
        since = datetime.now(timezone.utc) - timedelta(minutes=minutes)
        summary = await obs_store.get_metric_summary(
            metric_name,
            agent_name=scope_did,
            since=since,
        )
        summary["time_window_minutes"] = minutes
        return summary
    except Exception as e:
        logger.error(f"Error getting metric summary for {metric_name}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to get metric summary")
