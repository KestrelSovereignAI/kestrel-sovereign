"""Signal source for Talon background-job state transitions.

When a Talon CLI-background job moves from ``running`` to a terminal
state (``complete``, ``failed``, ``finished_unknown``), the
periodically-scheduled ``talon_monitor`` task emits one COGNITION
signal so the agent wakes and can act on the result without the user
having to poll ``talon_status`` manually.

This is the signal-dispatcher half of #1510. The polling half lives
in ``TalonCoordinatorFeature.talon_monitor`` (built as an ACTION cron
task by ``signals/sources/scheduler.py``).

Idempotency: the monitor records ``last_signaled_status`` on each
job in ``jobs.json`` and only emits when the current status differs
from the persisted last-signalled value. The signal source itself
also coalesces by ``dedupe_key=job_id`` within a short window as a
defense-in-depth against double-firing across two adjacent polls.
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


SOURCE_NAME = "talon.job_complete"
PROMPT_TEMPLATE = (
    Path(__file__).resolve().parents[2]
    / "prompts" / "signals" / "talon_job_complete.md"
)


def _schema(payload: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError(
            f"talon.job_complete payload must be a dict, got "
            f"{type(payload).__name__}"
        )
    for key in ("job_id", "status"):
        if key not in payload:
            raise ValueError(
                f"talon.job_complete payload missing required key: {key}"
            )
        if not isinstance(payload[key], str):
            raise ValueError(
                f"talon.job_complete payload {key} must be a string"
            )
    # Inject defaults the prompt template indexes so any caller that
    # built a payload missing one of these optional fields still
    # renders cleanly through the dispatcher.
    payload.setdefault("repo", "")
    payload.setdefault("issue", "")
    payload.setdefault("label", "")
    payload.setdefault("returncode", "")
    payload.setdefault("log_path", "")
    payload.setdefault("log_tail", "")
    payload.setdefault("started_at", "")
    payload.setdefault("completed_at", "")
    # Implementation-side test evidence (#1542): a short rendered summary
    # of which targeted tests Talon ran and the CI status it observed, so
    # the reviewer wake can cite test evidence instead of re-deriving it.
    # Optional — older Talon builds won't populate it.
    payload.setdefault("test_evidence", "")
    payload.setdefault("ci_status", "")
    return payload


def _redact(payload: Dict[str, Any]) -> str:
    """Audit-log summary. Identifiers only — no log body."""
    return (
        f"talon.job_complete "
        f"job_id={payload.get('job_id','?')} "
        f"status={payload.get('status','?')} "
        f"repo={payload.get('repo','?')} "
        f"issue={payload.get('issue','?')}"
    )


def build_talon_job_complete_registration() -> SourceRegistration:
    return SourceRegistration(
        name=SOURCE_NAME,
        schema=_schema,
        default_mode=SignalMode.COGNITION,
        allowed_modes=frozenset({SignalMode.COGNITION}),
        prompt_template=PROMPT_TEMPLATE,
        trust=Trust.TRUSTED,
        # Rate limit is defense-in-depth against a runaway monitor;
        # legitimate bursts can be larger than 10/min (a fleet of
        # 15+ Talon jobs all finishing in one poll). The monitor
        # marks ``last_signaled_status`` BEFORE the dispatcher
        # actually delivers the signal, so a rate-limit drop would
        # silently lose those wakes. Cap is intentionally well
        # above any plausible real-world burst.
        rate_limit=RateLimit(per_minute=120, per_hour=600),
        coalescing_window=timedelta(seconds=60),
        attention_policy=AttentionPolicy(),
        resources=frozenset(),
        # No self-loop concern: this is a local-only signal sourced
        # by the agent's own cron polling, not from a peer.
        allow_self_loops=False,
        log_redaction=RedactionPolicy(
            summarize=_redact,
            store_raw_trusted=False,
            redact_caller_identifier=True,
        ),
        retention_days=14,
    )


def build_signal_for_completed_job(
    job_id: str,
    info: Dict[str, Any],
    *,
    target_agent: str,
    log_tail: str = "",
) -> Signal:
    """Build a COGNITION signal envelope for a Talon job that has
    transitioned to a terminal state.

    Caller is the monitor poll inside ``TalonCoordinatorFeature``.
    ``info`` is the in-memory/persisted job record. ``target_agent``
    is the agent's DID (or fallback identifier) — the dispatcher
    rejects signals lacking it.
    """
    payload: Dict[str, Any] = {
        "job_id": str(job_id),
        "status": str(info.get("status", "")),
        "repo": str(info.get("repo", "")),
        "issue": str(info.get("issue", "")),
        "label": str(info.get("label", "")),
        "returncode": str(info.get("returncode", "")),
        "log_path": str(info.get("log_path", "")),
        "log_tail": log_tail or "",
        "started_at": str(info.get("started_at", "")),
        "completed_at": str(info.get("completed_at", "")),
        "test_evidence": str(info.get("test_evidence", "")),
        "ci_status": str(info.get("ci_status", "")),
    }
    return Signal(
        source=SOURCE_NAME,
        kind="inbound",
        mode=SignalMode.COGNITION,
        payload=payload,
        target_agent=target_agent,
        # ``(job_id, status)`` lets a status correction inside the
        # coalescing window (e.g. running→finished_unknown→failed
        # if the sidecar lands a poll late) still fire a fresh wake
        # for the corrected terminal state — the monitor's contract
        # is "one signal per state transition", not "one per job".
        dedupe_key=f"{payload['job_id']}:{payload['status']}",
    )
