"""Signal source for ``restart.completed`` (#1512).

After a restart-coordinator-mediated host restart lands, the
post-restart sweep in :class:`RestartCoordinatorFeature.initialize`
emits one COGNITION signal per ``executing`` row owned by the
requesting agent so the agent wakes and can verify post-restart
runtime state without manual polling.
"""

from __future__ import annotations

import logging
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict

from kestrel_sdk.signals import (
    AttentionPolicy,
    RateLimit,
    RedactionPolicy,
    Signal,
    SignalMode,
    SourceRegistration,
    Trust,
    Visibility,
)

logger = logging.getLogger(__name__)


SOURCE_NAME = "restart.completed"
PROMPT_TEMPLATE = (
    Path(__file__).resolve().parents[2]
    / "prompts" / "signals" / "restart_completed.md"
)


def _schema(payload: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError(
            f"restart.completed payload must be a dict, got "
            f"{type(payload).__name__}"
        )
    if "request_id" not in payload:
        raise ValueError("restart.completed payload missing request_id")
    if not isinstance(payload["request_id"], str):
        raise ValueError("restart.completed payload request_id must be str")
    # Defaults for prompt template placeholders.
    payload.setdefault("reason", "")
    payload.setdefault("urgency", "normal")
    payload.setdefault("policy", "idle_agents_only")
    payload.setdefault("requested_at", "")
    payload.setdefault("completed_at", "")
    payload.setdefault("operation", "restart_only")
    payload.setdefault("update_profile", "")
    payload.setdefault("target_ref", "")
    payload.setdefault("resolved_ref", "")
    return payload


def _redact(payload: Dict[str, Any]) -> str:
    return (
        f"restart.completed "
        f"request_id={payload.get('request_id','?')} "
        f"operation={payload.get('operation','?')} "
        f"urgency={payload.get('urgency','?')} "
        f"policy={payload.get('policy','?')}"
    )


def _restart_completed_result_summary(body: Any) -> str:
    """Bounded inline body for the ``signal_completed`` UI side-channel (#1809).

    For a COGNITION dispatch ``result.artifact`` is the agent's own response
    string (the resumed post-restart turn). Surfacing it here is what lets an
    open chat tab render the wake live — the frontend's ``handleSignalCompleted``
    appends it the moment the wake's turn lands, instead of the turn staying
    invisible until a manual refresh. Mirrors a2a.question_answered (#1522). The
    store caps the returned text at ``MAX_RESULT_SUMMARY_BYTES``.
    """
    if body is None:
        return ""
    return body if isinstance(body, str) else str(body)


def build_restart_completed_registration() -> SourceRegistration:
    return SourceRegistration(
        name=SOURCE_NAME,
        schema=_schema,
        default_mode=SignalMode.COGNITION,
        allowed_modes=frozenset({SignalMode.COGNITION}),
        prompt_template=PROMPT_TEMPLATE,
        trust=Trust.TRUSTED,
        # Restart completions are rare events; the cap is defense
        # against an accidental sweep loop firing twice.
        rate_limit=RateLimit(per_minute=4, per_hour=20),
        coalescing_window=timedelta(seconds=30),
        attention_policy=AttentionPolicy(),
        resources=frozenset(),
        # Surface the resumed post-restart turn on the signal_completed UI
        # side-channel so an open chat tab renders the wake live. Paired with
        # visibility=USER_VISIBLE + session_id on the signal below (#1809).
        result_summary=_restart_completed_result_summary,
        allow_self_loops=False,
        log_redaction=RedactionPolicy(
            summarize=_redact,
            store_raw_trusted=False,
            redact_caller_identifier=True,
        ),
        retention_days=30,
    )


def build_signal_for_restart_completed(
    request, *, target_agent: str, completed_at: str,
) -> Signal:
    """Build the COGNITION signal envelope for a completed restart.

    ``request`` is the ``RestartRequest`` dataclass row (or any object
    exposing the same id/reason/urgency/policy/requested_at fields).

    For ``update_then_restart`` requests the payload also carries the
    requested operation, the target ref/branch, and the resolved commit
    the update landed on, so the woken agent can verify it booted into
    the code it asked for.
    """
    # The actual commit the update checked out, recorded by the
    # coordinator's update run (empty for restart_only requests).
    resolved_ref = ""
    log_dict_fn = getattr(request, "update_log_dict", None)
    if callable(log_dict_fn):
        try:
            resolved_ref = str(log_dict_fn().get("resolved_ref", "") or "")
        except Exception:
            resolved_ref = ""

    payload: Dict[str, Any] = {
        "request_id": str(getattr(request, "id", "")),
        "reason": str(getattr(request, "reason", "")),
        "urgency": str(getattr(request, "urgency", "normal")),
        "policy": str(getattr(request, "policy", "idle_agents_only")),
        "requested_at": str(getattr(request, "requested_at", "")),
        "completed_at": str(completed_at),
        "operation": str(getattr(request, "operation", "restart_only")),
        "update_profile": str(getattr(request, "update_profile", "")),
        "target_ref": str(getattr(request, "update_target_ref", "")),
        "resolved_ref": resolved_ref,
    }
    # Route the wake back into the session the request was filed from, so it
    # surfaces in the same chat window the agent asked from (#1809). Empty for
    # CLI/system-filed requests → None = system-initiated (a fresh session), the
    # prior behavior.
    origin_session_id = str(getattr(request, "origin_session_id", "") or "")
    return Signal(
        source=SOURCE_NAME,
        kind="inbound",
        mode=SignalMode.COGNITION,
        payload=payload,
        target_agent=target_agent,
        session_id=origin_session_id or None,
        # USER_VISIBLE so the dispatcher emits the signal_completed UI event for
        # the resumed turn (INTERNAL would log-only and the wake would never
        # surface live). Paired with the result_summary callback above (#1809).
        visibility=Visibility.USER_VISIBLE,
        dedupe_key=payload["request_id"],
    )
