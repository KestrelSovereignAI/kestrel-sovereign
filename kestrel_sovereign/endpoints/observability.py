"""Observability endpoint - query A2A observability events for debugging."""
from fastapi import APIRouter, Request, Query, HTTPException
from typing import Optional, Dict, Any, List, Literal
from datetime import datetime, timezone, timedelta
import logging

from pydantic import BaseModel, Field, ValidationError, field_validator

from kestrel_sovereign.a2a.stores.unified.observability_store import _redact
from kestrel_sovereign.endpoints.agent_helpers import get_agent

logger = logging.getLogger(__name__)

router = APIRouter(tags=["observability"])

# event_type values accepted by the ingest endpoint, each mapped onto an
# existing observability_store.log_* method below.
EventType = Literal["tool_call", "tool_response", "agent_response", "error", "metric"]


class ObservabilityEventIn(BaseModel):
    """A single inbound telemetry event pushed by an external agent."""

    agent_name: str = Field(..., min_length=1)
    session_id: str = Field(..., min_length=1)
    event_type: EventType
    tool_name: Optional[str] = None
    duration_ms: Optional[int] = None
    success: bool = True
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    timestamp: Optional[str] = None

    @field_validator("timestamp")
    @classmethod
    def _validate_timestamp(cls, value: Optional[str]) -> Optional[str]:
        """Reject a malformed timestamp at validation time (→ 422).

        Keeps the stored string as-is for the producer's record; the ingest
        path re-parses it via ``_parse_timestamp`` to persist the row time.
        """
        if value is None:
            return None
        try:
            _parse_timestamp(value)
        except ValueError as exc:
            raise ValueError(f"invalid ISO-8601 timestamp: {value!r}") from exc
        return value


class ObservabilityIngestBatch(BaseModel):
    """Batch wrapper: ``{"events": [...]}``."""

    events: List[ObservabilityEventIn]


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


def _parse_timestamp(raw: Optional[str]) -> Optional[datetime]:
    """Parse an inbound ISO-8601 timestamp into an aware UTC datetime.

    Accepts a trailing ``Z`` (which ``datetime.fromisoformat`` rejected before
    3.11) and normalizes naive values to UTC so the stored row sorts correctly
    against internally-stamped events. Raises ``ValueError`` on a malformed
    value so the caller can surface a 422.
    """
    if raw is None:
        return None
    text = raw.strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


async def _ingest_one(obs_store, event: ObservabilityEventIn) -> Optional[str]:
    """Route one inbound event onto the matching ``observability_store.log_*``.

    Metadata is scrubbed with the same redaction the in-process loop uses
    before it ever touches the store. An optional ``timestamp`` is forwarded so
    after-the-fact telemetry keeps its original event time. Returns the created
    ``event_id``.
    """
    metadata = _redact(event.metadata or {})
    if not isinstance(metadata, dict):
        metadata = {}

    timestamp = _parse_timestamp(event.timestamp)

    if event.event_type == "tool_call":
        return await obs_store.log_tool_call(
            agent_name=event.agent_name,
            tool_name=event.tool_name or "",
            session_id=event.session_id,
            metadata=metadata,
            timestamp=timestamp,
        )

    if event.event_type == "tool_response":
        # No prior in-process row exists for an external push, so create the
        # row then fold in timing/outcome to land a single complete
        # tool_response record retrievable via GET.
        event_id = await obs_store.log_tool_call(
            agent_name=event.agent_name,
            tool_name=event.tool_name or "",
            session_id=event.session_id,
            metadata=metadata,
            timestamp=timestamp,
        )
        await obs_store.log_tool_response(
            event_id=event_id,
            success=event.success,
            duration_ms=int(event.duration_ms or 0),
            error_message=event.error_message,
        )
        return event_id

    if event.event_type == "agent_response":
        return await obs_store.log_agent_response(
            agent_name=event.agent_name,
            duration_ms=int(event.duration_ms or 0),
            success=event.success,
            session_id=event.session_id,
            metadata=metadata,
            timestamp=timestamp,
        )

    if event.event_type == "error":
        error_type = str(metadata.get("error_type") or "external_error")
        return await obs_store.log_error(
            agent_name=event.agent_name,
            error_type=error_type,
            error_message=event.error_message or "",
            session_id=event.session_id,
            metadata=metadata,
            timestamp=timestamp,
        )

    if event.event_type == "metric":
        metric_name = str(metadata.get("metric_name") or event.tool_name or "external_metric")
        try:
            metric_value = float(metadata.get("metric_value", event.duration_ms or 0) or 0)
        except (TypeError, ValueError):
            metric_value = 0.0
        return await obs_store.log_metric(
            agent_name=event.agent_name,
            metric_name=metric_name,
            metric_value=metric_value,
            metadata=metadata,
            session_id=event.session_id,
            timestamp=timestamp,
        )

    # EventType Literal makes this unreachable; guard anyway.
    raise ValueError(f"Unsupported event_type: {event.event_type}")


@router.post("/api/observability/events")
async def post_observability_events(
    request: Request,
    payload: Dict[str, Any],
) -> Dict[str, Any]:
    """Ingest tool-call telemetry pushed by external agents (e.g. talon).

    Accepts either a single event or ``{"events": [...]}``. Each event is
    routed onto the matching ``observability_store.log_*`` method, with the
    same redaction the in-process loop applies. Best-effort and non-blocking:
    a logging-store hiccup yields a soft per-event error, never a 500.
    """
    try:
        agent = get_agent(request)
    except HTTPException:
        raise

    obs_store = getattr(agent, "observability_store", None)
    if not obs_store:
        raise HTTPException(status_code=503, detail="Observability store not available")

    # Validate shape: batch wrapper or single event. Pydantic raises 422 on an
    # unknown event_type or a missing required field (agent_name/session_id).
    try:
        if isinstance(payload, dict) and "events" in payload:
            events = ObservabilityIngestBatch.model_validate(payload).events
        else:
            events = [ObservabilityEventIn.model_validate(payload)]
    except ValidationError as exc:
        # include_context=False drops the raw exception object pydantic stashes
        # in ``ctx`` for custom-validator errors, which is not JSON-serializable.
        raise HTTPException(status_code=422, detail=exc.errors(include_context=False))

    event_ids: List[str] = []
    errors: List[Dict[str, Any]] = []
    for index, event in enumerate(events):
        try:
            event_id = await _ingest_one(obs_store, event)
            if event_id:
                event_ids.append(event_id)
        except Exception as exc:  # best-effort: never 500 the caller
            logger.error(
                "Failed to ingest observability event %s: %s", index, exc, exc_info=True
            )
            errors.append({"index": index, "error": str(exc)})

    return {
        "event_ids": event_ids,
        "count": len(event_ids),
        "errors": errors,
    }


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
