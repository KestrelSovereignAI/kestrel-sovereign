"""Sender-side wake when a previously-asked A2A question reaches a
terminal state on the receiver (#1444).

This is the resumption rail for the new fire-and-resume
``send_a2a_question`` contract. Mechanism:

  1. Sender calls ``send_a2a_question(recipient, message, ...)`` — tool
     POSTs to recipient's ``/tasks/send``, writes a row to the sender's
     ``pending_a2a_questions`` table, spawns a tracked SSE subscription
     against ``GET /tasks/{task_id}/subscribe`` on the recipient.
  2. Asking turn ENDS immediately. ``ToolResult.ok(data={
     awaiting_reply: True, task_id, resume_via: 'a2a.question_answered'})``.
  3. Eventually the recipient calls ``respond_to_a2a_task`` which fires
     a ``status`` event over the SSE stream with ``final=true`` and the
     reply text in ``status.message.parts[0].text``.
  4. The subscription supervisor enqueues an ``a2a.question_answered``
     COGNITION signal on the sender's dispatcher with the full reply
     inline.
  5. A fresh COGNITION turn fires on the sender; its prompt template
     cites the original question by task_id + recipient and inlines the
     reply text. The agent integrates the answer and continues.

Distinction from ``a2a.task_complete``:

- ``a2a.task_complete`` is sent to the RECEIVER's dispatcher when one
  of THEIR tasks (created via ``TaskManager.create_task``) completes —
  including outbound tasks they sent that came back. Its prompt template
  says "do not re-spawn the same task" — wrong doctrine for an answered
  question, which the asking agent DOES want to react to.
- ``a2a.question_answered`` is fired LOCALLY by the sender's own
  subscription supervisor. The signal never crosses the wire — it's a
  cognition-wake produced when a peer's SSE stream tells us our
  outbound question is terminal. Dedicated source = dedicated prompt
  template + dedicated rate limit + dedicated redaction policy sized
  for full reply text.

Cycle safety: ``allow_self_loops=False``. The causation chain
threaded into the original outbound task's metadata round-trips
through the receiver's response handling and is rehydrated here when
the supervisor builds the signal, so A→B→A→B chains still hit the
dispatcher's depth-2 cycle cap.

Reply text size: 8 KiB inline soft cap. Overflow is truncated with an
explicit ``call get_peer_task_result('<recipient>', '<task_id>')`` hint
in the prompt — the receiver-side task store still has the full reply,
and ``PeersFeature.get_peer_task_result`` fetches it through the host
proxy. 8 KiB picked so a normal prompt token budget can absorb the
inline reply without crowding out memories / RAG context (~2K tokens).

Retention: 90 days per Sovereign decision on #1444 — resumed-turn
context-recall sometimes benefits from looking back at the original
question/answer pair weeks later.
"""

from __future__ import annotations

import logging
from datetime import timedelta
from pathlib import Path
from typing import Any, Optional

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
PROMPT_TEMPLATE = (
    Path(__file__).resolve().parents[2]
    / "prompts" / "signals" / "a2a_question_answered.md"
)

# Soft cap on inline reply text. Overflow truncates with a hint to call
# ``get_a2a_task(task_id)`` for the full body. Sized so a 16K-token
# context budget can absorb it without crowding memories/RAG. See module
# docstring.
REPLY_TEXT_INLINE_CAP_BYTES = 8 * 1024
REPLY_TEXT_OVERFLOW_HINT = (
    " ...[truncated; call get_peer_task_result(\"{recipient}\", "
    "\"{task_id}\") for the full body]"
)


def _a2a_question_answered_schema(payload: dict) -> dict:
    if not isinstance(payload, dict):
        raise ValueError(
            f"a2a.question_answered payload must be a dict, got "
            f"{type(payload).__name__}"
        )
    for key in ("task_id", "recipient", "original_question",
                "reply_text", "state"):
        if key not in payload:
            raise ValueError(
                f"a2a.question_answered payload missing required key: {key}"
            )
        if not isinstance(payload[key], str):
            raise ValueError(
                f"a2a.question_answered payload {key} must be a string"
            )
    if payload["state"] not in (
        "completed", "failed", "canceled", "expired",
    ):
        raise ValueError(
            f"a2a.question_answered payload state must be one of "
            f"completed/failed/canceled/expired, got {payload['state']!r}"
        )
    # Optional correlation fields injected by the subscription supervisor
    # so the prompt template can cite them in long-form context.
    if "origin_session_id" in payload and not isinstance(
        payload["origin_session_id"], str
    ):
        raise ValueError(
            "a2a.question_answered payload origin_session_id must be a string"
        )
    if "truncated" in payload and not isinstance(payload["truncated"], bool):
        raise ValueError(
            "a2a.question_answered payload truncated must be a bool"
        )
    # Inject defaults for the prompt template's full placeholder set so
    # legacy callers (test fixtures, external integrations, future code
    # paths) building a payload with only required keys don't KeyError
    # at render time. Same hygiene posture as a2a_task_submitted (#1438
    # codex round 2).
    payload.setdefault("origin_session_id", "")
    payload.setdefault("truncated", False)
    return payload


def _a2a_question_answered_redact(payload: dict) -> str:
    """One-line redaction summary for signal_log. Does NOT include
    reply_text — that lives in payload_raw with ``store_raw_trusted=True``
    so an auditor can fetch it but the public summary stays tight."""
    state = payload.get("state", "?")
    return (
        f"a2a.question_answered "
        f"task_id={payload.get('task_id', '?')} "
        f"recipient={payload.get('recipient', '?')} "
        f"state={state} "
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
    *,
    task_id: str,
    recipient: str,
    original_question: str,
    reply_text: str,
    state: str,
    target_agent: str,
    origin_session_id: str = "",
    causation_chain: Optional[list] = None,
) -> Signal:
    """Build the resumption signal.

    Args mirror the schema. ``reply_text`` is truncated to
    ``REPLY_TEXT_INLINE_CAP_BYTES`` here, NOT at fire-time — by the
    time the signal is in flight the payload size is locked. The
    overflow hint references ``task_id`` so the resumed turn can fetch
    the full body via ``get_a2a_task``.

    ``causation_chain`` is opaque (a list of frame dicts as emitted by
    ``serialize_chain_for_metadata`` and rehydrated by
    ``_deserialize_chain``). The caller (subscription supervisor) reads
    it from the receiver's terminal task metadata and threads it
    through here so the dispatcher's depth-2 cycle cap still applies.
    """
    truncated = False
    if len(reply_text.encode("utf-8")) > REPLY_TEXT_INLINE_CAP_BYTES:
        # Trim to the budget then append the hint. Use byte-aware
        # truncation so multi-byte UTF-8 sequences don't get cut.
        encoded = reply_text.encode("utf-8")
        # Reserve room for the hint so the total payload stays under
        # cap. The hint cites get_peer_task_result with BOTH recipient
        # and task_id since fetching the peer's task requires both
        # (the sender's own store only has tasks it received, not
        # tasks it sent — codex round 2 P2b on PR #1453).
        hint = REPLY_TEXT_OVERFLOW_HINT.format(
            recipient=recipient, task_id=task_id,
        )
        hint_bytes = hint.encode("utf-8")
        room = REPLY_TEXT_INLINE_CAP_BYTES - len(hint_bytes)
        clipped = encoded[:room].decode("utf-8", errors="ignore")
        reply_text = clipped + hint
        truncated = True

    return Signal(
        source=SOURCE_NAME,
        kind="answered",
        mode=SignalMode.COGNITION,
        payload={
            "task_id": task_id,
            "recipient": recipient,
            "original_question": original_question,
            "reply_text": reply_text,
            "state": state,
            "origin_session_id": origin_session_id,
            "truncated": truncated,
        },
        target_agent=target_agent,
        # #1522: USER_VISIBLE (was INTERNAL) so the dispatcher emits a
        # ``signal_completed`` SSE event after the resumed turn logs,
        # letting an open chat tab render the response live instead of
        # only on a manual refresh. The ``result_summary`` callback on
        # the registration supplies the agent's response text as the
        # event body; the frontend ``signal_completed`` handler
        # (chat.js ``handleSignalCompleted``) appends it to the active
        # pane.
        #
        # session_id is intentionally left at its default (None). The
        # conversation session this wake lands in is resolved INSIDE
        # process_input by the time-gap heuristic at dispatch time and
        # is not surfaced back onto the signal — it is NOT the A2A
        # protocol sessionId carried in origin_session_id (a uuid4 from
        # send time, different namespace). The frontend does not match
        # on session_id: the notifications SSE stream is pinned to the
        # selected agent, so the wake renders into the visible pane,
        # the same precedent as task_notification.
        visibility=Visibility.USER_VISIBLE,
        caller=recipient or None,
        urgency=Urgency.NORMAL,
        # Dedupe key prevents subscription + startup-replay racing for
        # the same terminal event from waking two cognition turns.
        dedupe_key=f"{task_id}:answered",
        causation_chain=list(causation_chain or []),
    )


__all__ = [
    "PROMPT_TEMPLATE",
    "REPLY_TEXT_INLINE_CAP_BYTES",
    "REPLY_TEXT_OVERFLOW_HINT",
    "SOURCE_NAME",
    "build_a2a_question_answered_registration",
    "build_signal_for_question_answered",
]
