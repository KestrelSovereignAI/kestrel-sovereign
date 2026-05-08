"""
Kestrel Observability Feature - Always-on lifecycle event observability.

Auto-discovers via the feature system, registers the ObservabilityHook
to log all lifecycle events to ObservabilityStore, and exposes tool
commands for querying observability data at runtime.
"""

import logging
from typing import Any, Dict, List, Optional

from kestrel_sdk.hooks.base import Hook
from kestrel_sdk.tools.base import ToolCategory
from kestrel_sdk.tools.result import ToolResult
from kestrel_sovereign.features.base import Feature, tool
from kestrel_sovereign.features.observability.hook import ObservabilityHook

logger = logging.getLogger(__name__)


class ObservabilityFeature(Feature):
    """Always-on observability via hook system. Logs all lifecycle events to ObservabilityStore."""

    def __init__(self, agent):
        super().__init__(agent)
        self._hook: Optional[ObservabilityHook] = None

    @property
    def tool_description(self) -> str:
        return "Lifecycle event observability and monitoring"

    def get_router(self):
        """Return the Observability HTTP router for dynamic mounting.

        The router is defined in endpoints/observability.py and mounted by
        the server only when ObservabilityFeature is discovered and enabled.
        """
        from kestrel_sovereign.endpoints.observability import router
        return router

    async def initialize(self):
        """Create the ObservabilityHook (auto-registered via get_hooks)."""
        self._hook = ObservabilityHook(agent=self.agent)

    def get_hooks(self) -> List[Hook]:
        """Return the observability hook for auto-registration."""
        if self._hook:
            return [self._hook]
        return []

    async def shutdown(self):
        """Clean up hook reference on shutdown."""
        self._hook = None

    @tool(
        "obs_status",
        "Show observability event counts",
        category=ToolCategory.SYSTEM,
        command_prefix="!obs",
    )
    async def obs_status(self) -> ToolResult:
        """Show observability summary: event counts by type, recent errors."""
        store = getattr(self.agent, "observability_store", None)
        if not store:
            return ToolResult.failed("ObservabilityStore not available")

        from datetime import datetime, timedelta, timezone

        since = datetime.now(timezone.utc) - timedelta(hours=1)
        events = await store.query_events(event_type="metric", since=since, limit=1000)

        counts: Dict[str, int] = {}
        for e in events:
            hook_event = e.metadata.get("hook_event", "unknown") if e.metadata else "unknown"
            metric_name = e.metadata.get("metric_name", "") if e.metadata else ""
            if metric_name.startswith("hook."):
                counts[hook_event] = counts.get(hook_event, 0) + 1

        error_events = await store.query_events(event_type="error", since=since, limit=10)
        recent_errors: List[Dict[str, Any]] = []
        for e in error_events:
            recent_errors.append({
                "timestamp": str(e.timestamp),
                "error_message": e.error_message,
                "metadata": e.metadata,
            })

        total_events = sum(counts.values())
        data = {
            "time_window": "last 1 hour",
            "hook_event_counts": counts,
            "total_hook_events": total_events,
            "recent_errors": recent_errors,
        }

        # Honesty: when there are recent errors, surface as PARTIAL so
        # the agent has to speak the count rather than narrate "obs
        # status retrieved" off a result that contains live errors.
        # Pre-migration the legacy dict let the LLM read just the
        # event counts and skip the recent_errors list.
        if recent_errors:
            return ToolResult.partial(
                confirmation=(
                    f"Observability: {total_events} hook event(s) in last 1h"
                ),
                error=(
                    f"{len(recent_errors)} error event(s) in the last hour; "
                    "see recent_errors in data for details"
                ),
                data=data,
            )
        return ToolResult.ok(
            confirmation=(
                f"Observability: {total_events} hook event(s) in last 1h, no errors"
            ),
            data=data,
        )

    @tool(
        "obs_events",
        "Query recent observability events",
        category=ToolCategory.SYSTEM,
        command_prefix="!obs-events",
    )
    async def obs_events(self, event_type: str = "", limit: int = 20) -> ToolResult:
        """Query recent events, optionally filtered by type.

        Args:
            event_type: Filter by event type (e.g., 'metric', 'error', 'tool_call')
            limit: Maximum events to return
        """
        store = getattr(self.agent, "observability_store", None)
        if not store:
            return ToolResult.failed("ObservabilityStore not available")

        events = await store.query_events(
            event_type=event_type if event_type else None,
            limit=limit,
        )

        event_dicts: List[Dict[str, Any]] = []
        for e in events:
            event_dicts.append({
                "event_id": e.event_id,
                "timestamp": str(e.timestamp),
                "agent_name": e.agent_name,
                "event_type": e.event_type,
                "tool_name": e.tool_name,
                "duration_ms": e.duration_ms,
                "success": e.success,
                "error_message": e.error_message,
                "metadata": e.metadata,
            })

        return ToolResult.ok(
            confirmation=(
                f"Listed {len(event_dicts)} event(s)"
                + (f" (filter={event_type!r})" if event_type else "")
            ),
            data={
                "events": event_dicts,
                "count": len(event_dicts),
            },
        )
