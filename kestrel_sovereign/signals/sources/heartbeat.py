"""Source registration for the heartbeat (Phase 3 of #889).

The heartbeat fires periodically (configured interval; default 30 min) and
asks the bird to read HEARTBEAT.md and respond. Today this is a direct
call to `agent.process_input`; under the dispatcher it becomes a COGNITION
signal — race-safe via Phase 2's turn lifecycle, audit-logged via Phase 1's
signal_log.
"""

from __future__ import annotations

import hashlib
from datetime import time as dtime, timedelta
from pathlib import Path
from typing import Any, Optional

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
    """HEARTBEAT.md content is operator-authored (TRUSTED) but routinely
    contains private reminders. Store length + content digest only — the
    digest proves "we saw the same content twice" for debugging without
    persisting any of the content itself. Truncated to 12 hex chars to
    keep the log readable; full sha256 is overkill for this purpose.
    """
    md = payload.get("heartbeat_md", "") or ""
    if not md:
        return "<empty heartbeat_md>"
    digest = hashlib.sha256(md.encode("utf-8")).hexdigest()[:12]
    return f"len={len(md)} sha256_12={digest}"


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
        # Phase 7 of #889: surface the bird's heartbeat response in
        # the UI side channel when the signal is non-INTERNAL. For
        # quiet HEARTBEAT_OK ticks, returning empty keeps the UI
        # clean; for alerts, the operator sees the alert text
        # directly in the side-channel event.
        result_summary=_heartbeat_result_summary,
        retention_days=14,
    )


def _heartbeat_result_summary(result_body: Any) -> str:
    """The cognition turn's result is the bird's response text. For a
    healthy heartbeat the bird replies "HEARTBEAT_OK" and we return
    empty (no UI noise for routine ticks). For an alert we surface
    the alert text so the operator's side channel renders something
    actionable."""
    if not result_body:
        return ""
    text = result_body if isinstance(result_body, str) else str(result_body)
    # Don't surface the all-clear in the UI side channel. Let routine
    # heartbeats stay metadata-only; only alerts get a body.
    if "HEARTBEAT_OK" in text and len(text.strip()) < 30:
        return ""
    if len(text) > 1000:
        return text[:1000] + "...(truncated)"
    return text


def _build_quiet_hours(
    active_start: Optional[str], active_end: Optional[str]
) -> Optional[tuple[dtime, dtime]]:
    """Translate the legacy `active_hours_start/end` (when the bird IS
    awake) into `quiet_hours` (when it is NOT).

    Boundary preservation: legacy `_is_within_active_hours` used
    `start <= now <= end` — INCLUSIVE on both ends. The dispatcher's
    `_time_in_window` is `[start, end)` — exclusive at end. Naively
    inverting (end, start) would flip a tick at exactly `active_end`
    from active to quiet, which is a behavior regression at the
    boundary.

    Fix: shift the quiet window's start by one minute so the active-end
    boundary minute stays active. Active 09:00–22:00 (inclusive both)
    becomes quiet 22:01–09:00 — equivalent to the legacy semantics at
    minute resolution.

    Empty config → no quiet hours.
    """
    if not active_start or not active_end:
        return None
    try:
        start = _parse_hhmm(active_start)
        end = _parse_hhmm(active_end)
    except ValueError:
        return None
    return (_add_minute(end), start)


def _parse_hhmm(value: str) -> dtime:
    """Parse "HH:MM" → datetime.time. Raises ValueError on bad input."""
    parts = value.split(":")
    if len(parts) != 2:
        raise ValueError(f"expected HH:MM, got {value!r}")
    hour, minute = int(parts[0]), int(parts[1])
    return dtime(hour=hour, minute=minute)


def _add_minute(t: dtime) -> dtime:
    """Add one minute, wrapping at midnight (23:59 → 00:00)."""
    minutes = t.hour * 60 + t.minute + 1
    minutes %= 24 * 60
    return dtime(hour=minutes // 60, minute=minutes % 60)
