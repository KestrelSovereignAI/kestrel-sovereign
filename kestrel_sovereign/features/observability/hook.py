"""
Kestrel Observability Hook - Default observer for all lifecycle events.

Writes structured entries to ObservabilityStore for every hook event.
This hook is purely observational: it never blocks, denies, or modifies
anything. Exceptions are swallowed to avoid affecting agent operation.
"""

import logging
from typing import Any, Dict

from kestrel_sovereign.hooks.base import Hook, HookEvent, HookInput, HookOutput

logger = logging.getLogger(__name__)

# Maps hook event names to ObservabilityStore event types
EVENT_TYPE_MAP = {
    "SessionStart": "lifecycle",
    "UserPromptSubmit": "lifecycle",
    "PreToolUse": "tool_call",
    "PostToolUse": "tool_response",
    "PreSubagentCall": "subagent_call",
    "PostSubagentCall": "subagent_response",
    "Stop": "lifecycle",
}


class ObservabilityHook(Hook):
    """Default observer for all lifecycle events. Writes to ObservabilityStore."""

    def __init__(self, agent):
        super().__init__(
            name="observability",
            events=list(HookEvent),  # ALL events
            priority=999,            # Run LAST — after security, audit, etc.
            timeout=5.0,
        )
        self.agent = agent

    async def execute(self, input: HookInput) -> HookOutput:
        """Log the event to ObservabilityStore. Never blocks."""
        try:
            store = getattr(self.agent, "observability_store", None)
            if not store:
                return HookOutput.allow()

            agent_name = getattr(self.agent, "agent_name", "unknown")
            event_type = input.hook_event_name

            metadata: Dict[str, Any] = {
                "hook_event": event_type,
                "session_id": input.session_id,
            }

            # Enrich metadata based on event type
            if input.tool_name:
                metadata["tool_name"] = input.tool_name
            if input.feature_name:
                metadata["feature_name"] = input.feature_name
            if input.user_message:
                metadata["user_message_length"] = len(input.user_message)
                # Don't log full user message — privacy
            if input.execution_time_ms is not None:
                metadata["execution_time_ms"] = input.execution_time_ms
            if input.tool_response:
                success = (
                    input.tool_response.get("success", True)
                    if isinstance(input.tool_response, dict)
                    else True
                )
                metadata["success"] = success
                if not success and isinstance(input.tool_response, dict):
                    metadata["error"] = str(
                        input.tool_response.get("error", "")
                    )[:200]

            await store.log_metric(
                agent_name=agent_name,
                metric_name=f"hook.{event_type}",
                metric_value=1,
                metadata=metadata,
            )

        except Exception as e:
            # Never let observability failures affect agent operation
            logger.debug(f"ObservabilityHook error (non-fatal): {e}")

        return HookOutput.allow()  # NEVER block
