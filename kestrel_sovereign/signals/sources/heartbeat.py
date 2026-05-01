"""Source registration for the heartbeat (Phase 3 of #889).

The heartbeat fires periodically (configured interval; default 30 min) and
asks the bird to read HEARTBEAT.md and respond. Today this is a direct
call to `agent.process_input`; under the dispatcher it becomes a COGNITION
signal — race-safe via Phase 2's turn lifecycle, audit-logged via Phase 1's
signal_log.
"""

from __future__ import annotations

from datetime import time as dtime, timedelta
from pathlib import Path
from typing import Optional

from kestrel_sdk.signals import (
    AttentionPolicy,
    RateLimit,
    RedactionPolicy,
    SignalMode,
    SourceRegistration,
    Trust,
    Urgency,
)


SOURCE_NAME = "heartbeat"
PROMPT_TEMPLATE = (
    Path(__file__).resolve().parents[3] / "prompts" / "signals" / "heartbeat.md"
)


def _heartbeat_schema(payload: dict) -> dict:
    """Heartbeat payload is small and stable: optional `heartbeat_md` text
    appended to the prompt. Reject anything else so the registry surface
    can't be polluted with ad-hoc fields."""
    allowed = {"heartbeat_md"}
    extra = set(payload.keys()) - allowed
    if extra:
        raise ValueError(
            f"heartbeat payload has unexpected keys: {sorted(extra)}; "
            f"allowed: {sorted(allowed)}"
        )
    md = payload.get("heartbeat_md", "")
    if not isinstance(md, str):
        raise ValueError("heartbeat_md must be a string")
    return {"heartbeat_md": md}


def _heartbeat_redact(payload: dict) -> str:
    """The HEARTBEAT.md content is operator-authored (TRUSTED), but it can
    contain reminders about private matters. Store length + a short prefix
    rather than the full text — enough to debug "why did the bird ping at
    3am" without leaking content into the signal_log."""
    md = payload.get("heartbeat_md", "") or ""
    if not md:
        return "<empty heartbeat_md>"
    prefix = md[:80].replace("\n", " ").strip()
    return f"len={len(md)} prefix={prefix!r}"


def build_heartbeat_registration(
    *,
    interval_seconds: int,
    active_hours_start: Optional[str],
    active_hours_end: Optional[str],
    timezone_name: str = "UTC",
) -> SourceRegistration:
    """Construct the heartbeat source registration from existing
    HeartbeatConfig fields. Caller (HeartbeatRunner) holds the config and
    calls this at startup."""
    quiet_hours = _build_quiet_hours(active_hours_start, active_hours_end)

    # Rate limit: cap at twice the configured cadence per hour to absorb
    # manual triggers via /heartbeat/trigger without letting them flood.
    # interval_seconds=1800 (30 min) → ~2 ticks/hour; allow up to 4.
    per_hour = max(2, int(3600 / max(interval_seconds, 60)) * 2)

    return SourceRegistration(
        name=SOURCE_NAME,
        schema=_heartbeat_schema,
        default_mode=SignalMode.COGNITION,
        allowed_modes=frozenset({SignalMode.COGNITION}),
        prompt_template=PROMPT_TEMPLATE,
        trust=Trust.TRUSTED,
        rate_limit=RateLimit(per_hour=per_hour, burst=2),
        coalescing_window=timedelta(seconds=10),
        attention_policy=AttentionPolicy(
            quiet_hours=quiet_hours,
            tz=timezone_name,
            modes_governed=frozenset({SignalMode.COGNITION}),
            urgency_override=Urgency.HIGH,
        ),
        # CONVERSATION is owned solely by the turn lifecycle (Phase 2).
        # Heartbeat declares no other resources — its only side effect is
        # appending to conversation history, which the lifecycle covers.
        resources=frozenset(),
        allow_self_loops=False,
        log_redaction=RedactionPolicy(
            summarize=_heartbeat_redact,
            store_raw_trusted=False,  # HEARTBEAT.md may carry operator-private notes
            redact_caller_identifier=True,
        ),
        retention_days=14,
    )


def _build_quiet_hours(
    active_start: Optional[str], active_end: Optional[str]
) -> Optional[tuple[dtime, dtime]]:
    """Translate the legacy `active_hours_start/end` (when the bird IS
    awake) into `quiet_hours` (when it is NOT). The dispatcher's attention
    policy expresses inactive windows; heartbeat config expresses the
    inverse. Empty config → no quiet hours."""
    if not active_start or not active_end:
        return None
    try:
        start = _parse_hhmm(active_start)
        end = _parse_hhmm(active_end)
    except ValueError:
        return None
    # Quiet window is the complement of the active window. If active is
    # 09:00–22:00, quiet is 22:00–09:00 (wraps midnight; the dispatcher's
    # _time_in_window handles the wrap).
    return (end, start)


def _parse_hhmm(value: str) -> dtime:
    """Parse "HH:MM" → datetime.time. Raises ValueError on bad input."""
    parts = value.split(":")
    if len(parts) != 2:
        raise ValueError(f"expected HH:MM, got {value!r}")
    hour, minute = int(parts[0]), int(parts[1])
    return dtime(hour=hour, minute=minute)
