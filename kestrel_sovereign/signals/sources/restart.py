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
    return payload


def _redact(payload: Dict[str, Any]) -> str:
    return (
        f"restart.completed "
        f"request_id={payload.get('request_id','?')} "
        f"urgency={payload.get('urgency','?')} "
        f"policy={payload.get('policy','?')}"
    )


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
    """
    payload: Dict[str, Any] = {
        "request_id": str(getattr(request, "id", "")),
        "reason": str(getattr(request, "reason", "")),
        "urgency": str(getattr(request, "urgency", "normal")),
        "policy": str(getattr(request, "policy", "idle_agents_only")),
        "requested_at": str(getattr(request, "requested_at", "")),
        "completed_at": str(completed_at),
    }
    return Signal(
        source=SOURCE_NAME,
        kind="inbound",
        mode=SignalMode.COGNITION,
        payload=payload,
        target_agent=target_agent,
        dedupe_key=payload["request_id"],
    )
