"""Generic signal source for async-waitable terminal-state transitions.

When ANY :class:`~kestrel_sdk.tools.MonitorableWaitable` provider's handle
reaches a terminal :class:`~kestrel_sdk.tools.Outcome`, the generic wait
reconciler (Wave 2 of #1860) emits one COGNITION signal so the agent wakes
and can act on the result without having held a turn or explicitly waited.

This is the generic successor to ``talon.job_complete`` — the talon source
stays for talon jobs (TalonWaitable declares ``signal = "talon.job_complete"``
so the reconciler still routes there), and ``wait.complete`` is the fallback
for every provider that does NOT declare its own signal name.

Idempotency: the reconciler records ``last_signaled_outcome`` per
``(kind, handle)`` in the ``wait_signal_state`` table and only emits when the
current terminal outcome differs from the persisted value. The signal source
also coalesces by ``dedupe_key`` within a short window as defense-in-depth.
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


SOURCE_NAME = "wait.complete"
PROMPT_TEMPLATE = (
    Path(__file__).resolve().parents[2]
    / "prompts" / "signals" / "wait_complete.md"
)


def _schema(payload: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError(
            f"wait.complete payload must be a dict, got "
            f"{type(payload).__name__}"
        )
    for key in ("kind", "handle", "outcome"):
        if key not in payload:
            raise ValueError(
                f"wait.complete payload missing required key: {key}"
            )
        if not isinstance(payload[key], str):
            raise ValueError(
                f"wait.complete payload {key} must be a string"
            )
    # Inject defaults the prompt template indexes so a provider whose
    # WaitStatus.data omitted one of these still renders cleanly.
    payload.setdefault("summary", "")
    payload.setdefault("status", "")
    payload.setdefault("ref", f"{payload['kind']}:{payload['handle']}")
    return payload


def _redact(payload: Dict[str, Any]) -> str:
    """Audit-log summary. Identifiers only — no provider data body."""
    return (
        f"wait.complete "
        f"kind={payload.get('kind','?')} "
        f"handle={payload.get('handle','?')} "
        f"outcome={payload.get('outcome','?')}"
    )


def build_wait_complete_registration() -> SourceRegistration:
    return SourceRegistration(
        name=SOURCE_NAME,
        schema=_schema,
        default_mode=SignalMode.COGNITION,
        allowed_modes=frozenset({SignalMode.COGNITION}),
        prompt_template=PROMPT_TEMPLATE,
        trust=Trust.TRUSTED,
        # Rate limit is defense-in-depth against a runaway reconciler;
        # a fleet of many waitables all reaching terminal in one tick is
        # a legitimate burst, so the cap sits well above any plausible
        # real-world burst (mirrors talon.job_complete). The reconciler
        # records last_signaled_outcome BEFORE confirming delivery, so a
        # rate-limit drop would be re-detected and retried next tick.
        rate_limit=RateLimit(per_minute=120, per_hour=600),
        coalescing_window=timedelta(seconds=60),
        attention_policy=AttentionPolicy(),
        resources=frozenset(),
        # No self-loop concern: this is a local-only signal sourced by the
        # agent's own reconciler cron, not from a peer.
        allow_self_loops=False,
        log_redaction=RedactionPolicy(
            summarize=_redact,
            store_raw_trusted=False,
            redact_caller_identifier=True,
        ),
        retention_days=14,
    )
