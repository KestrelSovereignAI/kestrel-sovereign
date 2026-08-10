"""Signal source for Talon background-job state transitions.

When a Talon CLI-background job moves from ``running`` to a terminal
state (``complete``, ``failed``, ``finished_unknown``), the generic wait
reconciler cron (Wave 2 of #1860) emits one COGNITION signal so the agent
wakes and can act on the result without the user having to poll
``talon_status`` manually.

This is the signal-dispatcher half of #1510. The polling half is now the
generic reconciler (``kestrel_sovereign/waits/reconciler.py``), which drives
this source via :class:`TalonWaitable` — that provider declares
``signal = "talon.job_complete"`` so terminal talon transitions route here
rather than to the generic ``wait.complete`` source. The talon-specific
``talon_monitor`` cron it replaced is retired.

Idempotency: the reconciler records ``last_signaled_outcome`` per
``(kind, handle)`` in the ``wait_signal_state`` table and only emits when the
current terminal outcome differs from the persisted value. The signal source
itself also coalesces by ``dedupe_key`` within a short window as a
defense-in-depth against double-firing across two adjacent ticks.
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
    # The chat session the job was dispatched from (#2877), captured on the
    # job record at dispatch and lifted onto ``Signal.session_id`` by the
    # reconciler so this wake resumes that session. Defaulted, not required:
    # job records dispatched before #2877 — and every unattended dispatch —
    # carry no origin, and those must stay valid and wake system-initiated.
    payload.setdefault("origin_session_id", "")
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


def _talon_job_complete_result_summary(body: Any) -> str:
    """Bounded inline body for the ``signal_completed`` UI side-channel (#2877).

    For a COGNITION dispatch ``result.artifact`` is the agent's own response
    string — the turn this job completion woke. Surfacing it here is what lets
    the chat window that dispatched the job render that turn live: the
    frontend's ``handleSignalCompleted`` requires BOTH a ``user_visible``
    signal (the reconciler builds one for a session-bound wake) and a
    non-empty ``result_summary``. With the callback absent this rail emitted
    metadata only, so a wake stayed invisible until a manual refresh even once
    it landed in the right session.

    Mirrors restart.completed (#1809) and a2a.question_answered (#1522). The
    store caps the returned text at ``MAX_RESULT_SUMMARY_BYTES``.
    """
    if body is None:
        return ""
    return body if isinstance(body, str) else str(body)


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
        # Surface the woken turn's response on the signal_completed UI
        # side-channel so the chat window that dispatched the job renders it
        # live. Paired with the reconciler building a USER_VISIBLE signal for
        # a session-bound wake (#2877); an unattended wake stays INTERNAL, and
        # the dispatcher never reaches this callback's emit for those.
        result_summary=_talon_job_complete_result_summary,
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
