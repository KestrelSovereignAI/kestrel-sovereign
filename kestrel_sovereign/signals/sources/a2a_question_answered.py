"""Sender-owned wake from an authorized native question receipt or local deadline.

The native question outbox and the canonical durable dispatcher own recovery.
The wire always uses official A2A Task/Message values; these local payloads
retain the immutable sender dispatch ID, independent of remote task identity.
"""

from __future__ import annotations

import logging
from datetime import timedelta
from pathlib import Path
from typing import Any

from kestrel_sdk.signals import (
    AttentionPolicy,
    RateLimit,
    RedactionPolicy,
    Signal,
    SignalMode,
    SourceRegistration,
    Trust,
    Urgency,
    Visibility,
)

logger = logging.getLogger(__name__)


SOURCE_NAME = "a2a.question_answered"
DURABLE_COGNITION_CONSUMER_ID = "core.a2a.question_answered.cognition"
PROMPT_TEMPLATE = (
    Path(__file__).resolve().parents[2]
    / "prompts"
    / "signals"
    / "a2a_question_answered.md"
)

# Soft cap on inline reply text. Overflow truncates with a hint to call
# ``get_a2a_task(task_id)`` for the full body. Sized so a 16K-token
# context budget can absorb it without crowding memories/RAG. See module
# docstring.
REPLY_TEXT_INLINE_CAP_BYTES = 8 * 1024
REPLY_TEXT_OVERFLOW_HINT = (
    ' ...[truncated; call get_peer_task_result("{dispatch_id}") for the full body]'
)


def _a2a_question_answered_schema(payload: dict) -> dict:
    if not isinstance(payload, dict):
        raise ValueError(
            f"a2a.question_answered payload must be a dict, got "
            f"{type(payload).__name__}"
        )
    for key in (
        "question_id",
        "dispatch_id",
        "recipient_id",
        "original_question",
        "reply_text",
        "state",
    ):
        if not isinstance(payload.get(key), str):
            raise ValueError(f"a2a.question_answered payload requires string {key}")
    for key in ("question_id", "dispatch_id", "recipient_id"):
        if not payload[key].strip():
            raise ValueError(f"a2a.question_answered payload requires concrete {key}")
    if payload["state"] not in {
        "completed",
        "failed",
        "canceled",
        "rejected",
        "expired",
    }:
        raise ValueError(
            "a2a.question_answered state must be a native terminal state or expired"
        )
    if not isinstance(payload.get("request"), dict):
        raise ValueError("a2a.question_answered requires the original native Message")
    # Optional correlation fields injected by the subscription supervisor
    # so the prompt template can cite them in long-form context.
    if "origin_session_id" in payload and not isinstance(
        payload["origin_session_id"], str
    ):
        raise ValueError(
            "a2a.question_answered payload origin_session_id must be a string"
        )
    if "truncated" in payload and not isinstance(payload["truncated"], bool):
        raise ValueError("a2a.question_answered payload truncated must be a bool")
    if "late" in payload:
        if not isinstance(payload["late"], bool):
            raise ValueError("a2a.question_answered payload late must be a bool")
        if payload["late"] and payload["state"] == "expired":
            raise ValueError(
                "a2a.question_answered late wake must carry the recipient's "
                "terminal state, not 'expired'"
            )
    if "withdrawn" in payload:
        if not isinstance(payload["withdrawn"], bool):
            raise ValueError("a2a.question_answered payload withdrawn must be a bool")
        if payload["withdrawn"] and payload["state"] != "expired":
            raise ValueError(
                "a2a.question_answered withdrawn wake must carry state "
                "'expired': only an expiry withdraws the question"
            )
    # Inject defaults for the prompt template's full placeholder set so
    # legacy callers (test fixtures, external integrations, future code
    # paths) building a payload with only required keys don't KeyError
    # at render time. Same hygiene posture as a2a_task_submitted (#1438
    # codex round 2).
    payload.setdefault("origin_session_id", "")
    payload.setdefault("truncated", False)
    payload.setdefault("late", False)
    payload.setdefault("withdrawn", False)
    return payload


def _a2a_question_answered_redact(payload: dict) -> str:
    """One-line redaction summary for signal_log. Does NOT include
    reply_text — that lives in payload_raw with ``store_raw_trusted=True``
    so an auditor can fetch it but the public summary stays tight."""
    state = payload.get("state", "?")
    return (
        f"a2a.question_answered "
        f"dispatch_id={payload.get('dispatch_id', '?')} "
        f"recipient_id={payload.get('recipient_id', '?')} "
        f"state={state} "
        f"late={bool(payload.get('late', False))} "
        f"withdrawn={bool(payload.get('withdrawn', False))} "
        f"reply_chars={len(payload.get('reply_text', ''))}"
    )


def _a2a_question_answered_result_summary(body: Any) -> str:
    """Bounded inline body for the ``signal_completed`` UI side-channel
    (#1522).

    For a COGNITION dispatch ``result.artifact`` is the agent's own
    response string (see ``SignalDispatcher._success`` —
    ``artifact=cognition_result``). Surfacing it here is what lets the
    active chat UI render the resumed turn live: a reply-capable A2A
    wake resolves while a chat tab is open, the dispatcher emits
    ``signal_completed`` with this text, and the frontend appends it
    inline instead of leaving the turn invisible until a manual
    refresh. The store caps the returned text at
    ``MAX_RESULT_SUMMARY_BYTES`` as defense in depth."""
    if body is None:
        return ""
    if isinstance(body, str):
        return body
    return str(body)


def build_a2a_question_answered_registration() -> SourceRegistration:
    return SourceRegistration(
        name=SOURCE_NAME,
        schema=_a2a_question_answered_schema,
        default_mode=SignalMode.COGNITION,
        allowed_modes=frozenset({SignalMode.COGNITION}),
        prompt_template=PROMPT_TEMPLATE,
        trust=Trust.TRUSTED,
        # Higher cardinality than a2a.task_complete — one signal per
        # asked question, not one per task lifecycle event. Sized for a
        # busy operator + small fleet without throttling legitimate
        # multi-turn conversations.
        rate_limit=RateLimit(per_minute=30, per_hour=300),
        # Tight dedupe — a subscription racing the startup-replay sweep
        # MUST not fire two cognition wakes for the same answer.
        coalescing_window=timedelta(seconds=10),
        attention_policy=AttentionPolicy(),
        resources=frozenset(),
        # #1522: surface the resumed turn's response on the
        # ``signal_completed`` UI side-channel so an open chat tab
        # renders the wake live. Paired with visibility=USER_VISIBLE +
        # session_id on the signal (see build_signal_for_question_answered).
        result_summary=_a2a_question_answered_result_summary,
        # The sender firing a signal back to itself would only happen if
        # they sent a question to themselves — already blocked at the
        # PeersFeature layer, but redundant guard.
        allow_self_loops=False,
        log_redaction=RedactionPolicy(
            summarize=_a2a_question_answered_redact,
            # Reply text matters for audit — preserve it raw for trusted
            # consumers. Public signal_log row still summarized.
            store_raw_trusted=True,
            redact_caller_identifier=True,
        ),
        # Sovereign decision on #1444 — resumed-turn recall sometimes
        # benefits from looking back weeks later.
        retention_days=90,
    )


def build_signal_for_question_answered(
    *, event_id: str, payload: dict, target_agent: str
) -> Signal:
    """Build one immutable outbox event; retries retain the same event identity."""
    from kestrel_sovereign.a2a.extensions import (
        CAUSATION_URI,
        deserialize_causation_chain,
    )

    body = _a2a_question_answered_schema(dict(payload))
    chain = deserialize_causation_chain(
        {CAUSATION_URI: body.pop("causation", None)}
        if body.get("causation") is not None
        else {}
    )
    reply = body["reply_text"]
    if len(reply.encode("utf-8")) > REPLY_TEXT_INLINE_CAP_BYTES:
        hint = REPLY_TEXT_OVERFLOW_HINT.format(dispatch_id=body["dispatch_id"])
        room = REPLY_TEXT_INLINE_CAP_BYTES - len(hint.encode("utf-8"))
        body["reply_text"] = (
            reply.encode("utf-8")[:room].decode("utf-8", errors="ignore") + hint
        )
        body["truncated"] = True
    return Signal(
        id=event_id,
        source=SOURCE_NAME,
        kind="answered",
        mode=SignalMode.COGNITION,
        payload=body,
        target_agent=target_agent,
        session_id=body["origin_session_id"] or None,
        visibility=Visibility.USER_VISIBLE,
        caller=body["recipient_id"],
        urgency=Urgency.NORMAL,
        dedupe_key=event_id,
        causation_chain=chain,
    )
