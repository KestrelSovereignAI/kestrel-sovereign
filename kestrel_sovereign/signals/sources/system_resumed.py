"""Source registration for host resume / suspend recovery (#1545).

The ``ResumeMonitor`` (``kestrel_sovereign/resume_monitor.py``) detects a
host sleep/wake gap and dispatches one ``system.resumed`` signal through
the SignalDispatcher. This source is the routing target.

It is an ACTION source — deterministic, no LLM cost — whose handler
re-anchors the dispatcher's throttling windows (coalescing + rate-limit),
which otherwise disagree about elapsed time across a suspend. Routing it
through the dispatcher (rather than a bespoke side channel) gives the event
a durable, redaction-policed audit row in ``signal_log`` with the measured
gap, exactly like every other signal.

The handler is supplied by the wiring in ``KestrelAgent.initialize`` (it
needs the live dispatcher to call ``notify_resume``); this module owns the
schema, redaction, and throttling contract.
"""

from __future__ import annotations

from typing import Any, Awaitable, Callable

from kestrel_sdk.signals import (
    RateLimit,
    RedactionPolicy,
    SignalMode,
    SourceRegistration,
    Trust,
)

SOURCE_NAME = "system.resumed"


def _schema(payload: dict) -> dict:
    """Payload is a single non-negative ``gap_seconds`` float. Reject any
    other shape so the audit row and handler see a stable contract."""
    if not isinstance(payload, dict):
        raise ValueError(
            f"system.resumed payload must be a dict, got {type(payload).__name__}"
        )
    extra = set(payload.keys()) - {"gap_seconds"}
    if extra:
        raise ValueError(
            f"system.resumed payload has unexpected keys: {sorted(extra)}; "
            "allowed: ['gap_seconds']"
        )
    raw = payload.get("gap_seconds", 0.0)
    try:
        gap = float(raw)
    except (TypeError, ValueError):
        raise ValueError(f"gap_seconds must be numeric, got {raw!r}")
    if gap < 0:
        raise ValueError(f"gap_seconds must be >= 0, got {gap}")
    return {"gap_seconds": gap}


def _redact(payload: dict) -> str:
    """Nothing sensitive here — the gap duration is the whole payload and is
    safe (and useful) to store verbatim for post-hoc 'why did I miss things'
    debugging."""
    return f"gap_seconds={payload.get('gap_seconds', 0.0):.0f}"


def build_system_resumed_registration(
    *,
    handler: Callable[[dict], Awaitable[Any]],
) -> SourceRegistration:
    """Construct the ``system.resumed`` ACTION source registration.

    Args:
        handler: async callable receiving the validated payload
            (``{"gap_seconds": float}``). Wired in the agent so it can
            re-anchor the live dispatcher's throttling windows.
    """
    return SourceRegistration(
        name=SOURCE_NAME,
        schema=_schema,
        default_mode=SignalMode.ACTION,
        allowed_modes=frozenset({SignalMode.ACTION}),
        handler=handler,
        trust=Trust.TRUSTED,
        # Fires once per real suspend — rare by nature. A modest cap is
        # pure belt-and-suspenders against a pathological detector loop;
        # it must never throttle a legitimate resume, so the window is
        # generous and ACTION is never gated by quiet hours.
        rate_limit=RateLimit(per_minute=10, per_hour=120),
        # No dedupe_key is set by the emitter, so coalescing never applies;
        # the default attention policy leaves ACTION ungated.
        resources=frozenset(),
        allow_self_loops=False,
        log_redaction=RedactionPolicy(
            summarize=_redact,
            store_raw_trusted=True,  # gap duration is non-sensitive
            redact_caller_identifier=True,
        ),
        retention_days=14,
    )
