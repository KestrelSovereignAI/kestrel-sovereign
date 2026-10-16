"""Signal source for ``queue.drained``: one triage wake per idle stretch (#3664).

On 2026-10-15 an agent finished its last item at 23:11Z with nothing queued
and nothing running in Talon. Nothing woke it for about 16 hours. Its
continuous-work doctrine says an empty queue triggers triage, but that rule
runs inside a turn, and every turn starts from a wake: a Talon completion, a
restart, a wait, an operator message. When the last item finishes and nothing
new is dispatched, no wake is pending, so the rule never gets to run.

The idle detector (:mod:`kestrel_sovereign.signals.idle_wake`) emits this
COGNITION signal when the agent has been idle for the configured number of
minutes, at most once per idle stretch. This module owns the source contract:
schema, redaction, rate limit and prompt. The wake only reports that the queue
is empty. It dispatches nothing; the agent decides what to do.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

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

SOURCE_NAME = "queue.drained"
SIGNAL_KIND = "idle"
PROMPT_TEMPLATE = (
    Path(__file__).resolve().parents[2]
    / "prompts" / "signals" / "queue_drained.md"
)

# The payload is built by the detector from its own observations; nothing in
# it is caller-supplied. A closed key set keeps it that way.
_STRING_KEYS = ("idle_since",)
_INT_KEYS = ("idle_minutes", "observed_idle_minutes", "attempt")


def _schema(payload: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError(
            f"queue.drained payload must be a dict, got {type(payload).__name__}"
        )
    allowed = set(_STRING_KEYS) | set(_INT_KEYS)
    extra = set(payload) - allowed
    if extra:
        raise ValueError(
            f"queue.drained payload has unexpected keys: {sorted(extra)}; "
            f"allowed: {sorted(allowed)}"
        )
    normalized: Dict[str, Any] = {}
    for key in _STRING_KEYS:
        value = payload.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"queue.drained payload {key} must be a non-empty string")
        normalized[key] = value.strip()
    for key in _INT_KEYS:
        value = payload.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(
                f"queue.drained payload {key} must be a non-negative integer"
            )
        normalized[key] = value
    return normalized


def _redact(payload: Dict[str, Any]) -> str:
    """Audit summary. The payload holds only the detector's own timings."""
    return (
        f"queue.drained idle_since={payload.get('idle_since', '?')} "
        f"idle_minutes={payload.get('idle_minutes', '?')} "
        f"attempt={payload.get('attempt', '?')}"
    )


def _result_summary(body: Any) -> str:
    """Bounded inline body for the ``signal_completed`` UI side channel.

    A wake bound to the agent's chain session is built ``USER_VISIBLE``, and
    the frontend paints it only with a non-empty ``result_summary``. For a
    COGNITION dispatch the body is the triage turn's response. The store caps
    the returned text.
    """
    if body is None:
        return ""
    return body if isinstance(body, str) else str(body)


def build_queue_drained_registration() -> SourceRegistration:
    return SourceRegistration(
        name=SOURCE_NAME,
        schema=_schema,
        default_mode=SignalMode.COGNITION,
        allowed_modes=frozenset({SignalMode.COGNITION}),
        prompt_template=PROMPT_TEMPLATE,
        # Built by the agent's own detector from local state.
        trust=Trust.TRUSTED,
        # The detector fires at most once per idle stretch and retries a
        # failed wake a bounded number of times, so a real burst is
        # impossible. The cap is defense against a detector bug spending
        # turns in a loop.
        rate_limit=RateLimit(per_minute=1, per_hour=6),
        # The default window. Each attempt carries its own dedupe key, so it
        # only folds a duplicate enqueue of one attempt, never a retry into
        # the attempt that failed.
        coalescing_window=None,
        # The configured idle threshold is the attention policy: an agent
        # that wants quiet hours leaves this off.
        attention_policy=AttentionPolicy(),
        # CONVERSATION is owned by the turn lifecycle.
        resources=frozenset(),
        result_summary=_result_summary,
        allow_self_loops=False,
        log_redaction=RedactionPolicy(
            summarize=_redact,
            store_raw_trusted=False,
            redact_caller_identifier=True,
        ),
        retention_days=14,
    )


def build_queue_drained_signal(
    *,
    target_agent: str,
    idle_since: str,
    idle_minutes: int,
    observed_idle_minutes: int,
    attempt: int,
    session_id: Optional[str],
) -> Signal:
    """Build the triage wake.

    ``session_id`` is the agent's most recent chain session, resolved by the
    detector from local state only. A bound wake is ``USER_VISIBLE`` so it
    renders live in that session (SIGNAL_SOURCES_GUIDE rule 1). With no chain
    session the signal carries none, the dispatcher mints a fresh session for
    the turn, and the wake stays ``INTERNAL``: nobody is watching a minted
    session. There is no fallback to an operator's open chat.
    """
    bound = bool(session_id)
    return Signal(
        source=SOURCE_NAME,
        kind=SIGNAL_KIND,
        mode=SignalMode.COGNITION,
        payload={
            "idle_since": idle_since,
            "idle_minutes": idle_minutes,
            "observed_idle_minutes": observed_idle_minutes,
            "attempt": attempt,
        },
        target_agent=target_agent,
        session_id=session_id if bound else None,
        visibility=Visibility.USER_VISIBLE if bound else Visibility.INTERNAL,
        dedupe_key=f"{SOURCE_NAME}:{idle_since}#{attempt}",
    )
