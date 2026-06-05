"""Chat-visible restart-status events (#1551).

A restart/update is an audited deployment primitive. When an agent
files one — and as the coordinator drives it through its lifecycle —
the Sovereign needs first-class, chat-visible evidence rather than
having to trust the agent's natural-language report.

This module builds the JSON payload for the ``restart_status`` UI
side-channel event. The feature emits one through ``agent.emit_event``
at each lifecycle point (filed/pending, deferred, executing/updating,
completed, rejected, canceled); the Sovereign Console renders it as a
system/status bubble in the conversation.

The payload deliberately mirrors the fields the issue calls out:
request id, requesting agent, operation, target ref + update profile,
policy/urgency, current state, and a deferral reason when the
coordinator defers execution.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

EVENT_NAME = "restart_status"


def build_restart_status_event(
    request,
    *,
    state: str,
    deferral_reason: str = "",
    status_reason: str = "",
    agent_did: str = "",
) -> Dict[str, Any]:
    """Build the ``restart_status`` UI event payload for one request.

    ``request`` is a ``RestartRequest`` row (or any object exposing the
    same id/operation/urgency/policy/reason fields). ``state`` is the
    lifecycle state being surfaced — pass it explicitly because the row
    object may carry a stale ``status`` at the emit site.

    ``deferral_reason`` is set only when the coordinator defers (the row
    stays ``pending`` but the attempt was held back, e.g. "agent busy").
    """
    return {
        "request_id": str(getattr(request, "id", "")),
        "requested_by_agent": str(
            getattr(request, "requested_by_agent", "") or agent_did
        ),
        "operation": str(getattr(request, "operation", "restart_only")),
        "urgency": str(getattr(request, "urgency", "normal")),
        "policy": str(getattr(request, "policy", "idle_agents_only")),
        "status": str(state),
        "reason": str(getattr(request, "reason", "")),
        "target_ref": str(getattr(request, "update_target_ref", "")),
        "update_profile": str(getattr(request, "update_profile", "")),
        "deferral_reason": str(deferral_reason or ""),
        "status_reason": str(status_reason or ""),
        "completed_at": _opt_str(getattr(request, "completed_at", None)),
    }


def _opt_str(value: Any) -> Optional[str]:
    return None if value is None else str(value)
