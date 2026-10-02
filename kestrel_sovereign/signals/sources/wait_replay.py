"""Signal source for wait wakes that re-announce an event older than the
handle's last delivered wake (#3390).

The wait reconciler wakes the agent once per terminal transition of a handle,
on the provider's own source (``talon.job_complete``) or on the generic
``wait.complete``. A provider's prompt can tell the agent what to do in the
same turn — Talon's tells it to answer a clarification and re-dispatch the
claim. That is right when the event has just happened and wrong when it
has not.

A wake is a REPLAY when the provider dates its terminal event
(:data:`~kestrel_sovereign.waits.engine.TERMINAL_EVENT_AT_KEY`) before the
handle's last delivered wake: the reconciler sees a state it has not delivered,
but the work finished before the agent was last woken for it. On 2026-09-29 a
Talon classifier change re-delivered weeks-old completions this way. They
carried same-turn instructions that would have re-answered settled questions
and re-claimed finished issues.

The reconciler announces a replay here instead of on the provider's source. This
template says it is a replay and withholds the provider's act-now instructions:
it directs the agent to check the handle's current state before doing
anything. A durable resume consumer, which listens on the provider's source,
does not receive a replay. Its handle's earlier wake was delivered after the
event happened, so the consumer has already been woken for it.
"""

from __future__ import annotations

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

SOURCE_NAME = "wait.replay"
PROMPT_TEMPLATE = (
    Path(__file__).resolve().parents[2]
    / "prompts" / "signals" / "wait_replay.md"
)


def _schema(payload: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError(
            f"wait.replay payload must be a dict, got {type(payload).__name__}"
        )
    for key in ("kind", "handle", "outcome"):
        if key not in payload:
            raise ValueError(f"wait.replay payload missing required key: {key}")
        if not isinstance(payload[key], str):
            raise ValueError(f"wait.replay payload {key} must be a string")
    # Defaults for every key the prompt template indexes, so a provider whose
    # WaitStatus.data omitted one still renders.
    payload.setdefault("summary", "")
    payload.setdefault("status", "")
    payload.setdefault("ref", f"{payload['kind']}:{payload['handle']}")
    payload.setdefault("terminal_event_at", "")
    payload.setdefault("delivery_last_delivered_at", "")
    return payload


def _redact(payload: Dict[str, Any]) -> str:
    """Audit-log summary. Identifiers only — no provider data body."""
    return (
        f"wait.replay "
        f"kind={payload.get('kind', '?')} "
        f"handle={payload.get('handle', '?')} "
        f"outcome={payload.get('outcome', '?')}"
    )


def _result_summary(body: Any) -> str:
    """Bounded inline body for the ``signal_completed`` UI side channel.

    A replay of a wake that was bound to a chat session is bound to that
    session too (#2877), and the frontend paints it only with a non-empty
    ``result_summary``. For a COGNITION dispatch the body is the woken turn's
    response.
    """
    if body is None:
        return ""
    return body if isinstance(body, str) else str(body)


def build_wait_replay_registration() -> SourceRegistration:
    return SourceRegistration(
        name=SOURCE_NAME,
        schema=_schema,
        default_mode=SignalMode.COGNITION,
        allowed_modes=frozenset({SignalMode.COGNITION}),
        prompt_template=PROMPT_TEMPLATE,
        trust=Trust.TRUSTED,
        # The same defense-in-depth ceiling as wait.complete: a classifier
        # change can turn a whole history into replays in one tick, and a
        # dropped replay is re-detected and retried by the reconciler.
        rate_limit=RateLimit(per_minute=120, per_hour=600),
        coalescing_window=timedelta(seconds=60),
        attention_policy=AttentionPolicy(),
        resources=frozenset(),
        result_summary=_result_summary,
        # Local-only: sourced by the agent's own reconciler, not by a peer.
        allow_self_loops=False,
        log_redaction=RedactionPolicy(
            summarize=_redact,
            store_raw_trusted=False,
            redact_caller_identifier=True,
        ),
        retention_days=14,
    )
