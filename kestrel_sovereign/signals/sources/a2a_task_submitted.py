"""Native inbound A2A cognition wakes after task admission commits.

The manager delivers a native task snapshot with server-owned admission facts.
The recipient builds its signal from those facts, maps the peer context into
an isolated chat projection, and parses the declared correlation and causation
extensions. Caller metadata cannot choose the trusted sender. The persisted
task and lifecycle revision also drive restart reconciliation.
"""

from __future__ import annotations

import logging
from datetime import timedelta
from pathlib import Path

from a2a.types import TaskState
from google.protobuf.json_format import MessageToDict
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

from kestrel_sovereign.a2a.extensions import (
    deserialize_causation_chain,
    parse_question_correlation,
)
from kestrel_sovereign.a2a.projections import message_text
from kestrel_sovereign.a2a.scope import NativeRecipientScope
from kestrel_sovereign.a2a.stores.unified.task_store import (
    NativeTaskSnapshot,
    require_native_task_snapshot,
)

logger = logging.getLogger(__name__)


SOURCE_NAME = "a2a.task_submitted"
DURABLE_COGNITION_CONSUMER_ID = "core.a2a-task-submitted-cognition-v1"
PROMPT_TEMPLATE = (
    # Inside the package (#1415); see a2a.py for the full rationale.
    Path(__file__).resolve().parents[2]
    / "prompts" / "signals" / "a2a_task_submitted.md"
)


# ---------------------------------------------------------------------------
# Schema / redaction
# ---------------------------------------------------------------------------


def _a2a_submitted_schema(payload: dict) -> dict:
    if not isinstance(payload, dict):
        raise ValueError(
            f"a2a.task_submitted payload must be a dict, got "
            f"{type(payload).__name__}"
        )
    for key in ("task_id", "session_id", "sender"):
        if key not in payload:
            raise ValueError(
                f"a2a.task_submitted payload missing required key: {key}"
            )
        if not isinstance(payload[key], str):
            raise ValueError(
                f"a2a.task_submitted payload {key} must be a string"
            )
    # Optional discriminator fields (added in #1380). Empty strings are
    # valid (legacy / non-PeersFeature task creators leave them blank).
    if "a2a_verb" in payload and not isinstance(payload["a2a_verb"], str):
        raise ValueError(
            "a2a.task_submitted payload a2a_verb must be a string"
        )
    if "reply_expected" in payload and not isinstance(
        payload["reply_expected"], bool
    ):
        raise ValueError(
            "a2a.task_submitted payload reply_expected must be a bool"
        )
    # ``request_content`` is the sender's actual question/message text
    # extracted from ``task.history[0]``. Added in #1433 so the
    # cognition prompt has the question inline rather than forcing the
    # receiver to call ``check_task_status`` first — without it the
    # receiver hallucinated "null body" three times in a row on real
    # questions. Optional (empty string accepted) so historical signal
    # payloads that predate the field still validate; the prompt
    # template guards against the missing case.
    if "request_content" in payload and not isinstance(
        payload["request_content"], str
    ):
        raise ValueError(
            "a2a.task_submitted payload request_content must be a string"
        )
    # Inject defaults for every field the prompt template indexes so any
    # legacy caller (test fixtures, prior code paths, external
    # integrations) building a payload with only the required keys still
    # renders cleanly. Without this, ``Dispatcher._render_prompt()`` would
    # KeyError on `{payload[a2a_verb]}` / `{payload[skill_id]}` /
    # `{payload[request_content]}` for older payloads and the cognition
    # turn would silently never fire (codex review #1433 P2 rounds 1+2).
    # Keep this list in lockstep with the placeholders in
    # ``prompts/signals/a2a_task_submitted.md``.
    payload.setdefault("request_content", "")
    payload.setdefault("a2a_verb", "")
    payload.setdefault("skill_id", "")
    payload.setdefault("reply_expected", False)
    return payload


def _a2a_submitted_redact(payload: dict) -> str:
    """One-line redaction summary for audit logs. Don't leak full
    message bodies into log storage — just identifiers and the
    summary fields a debugger needs to find the corresponding task."""
    return (
        f"a2a.task_submitted "
        f"task_id={payload.get('task_id','?')} "
        f"sender={payload.get('sender','?')} "
        f"verb={payload.get('a2a_verb','?')} "
        f"skill={payload.get('skill_id','?')}"
    )


def build_a2a_task_submitted_registration() -> SourceRegistration:
    return SourceRegistration(
        name=SOURCE_NAME,
        schema=_a2a_submitted_schema,
        default_mode=SignalMode.COGNITION,
        allowed_modes=frozenset({SignalMode.COGNITION}),
        prompt_template=PROMPT_TEMPLATE,
        trust=Trust.TRUSTED,
        # Mirror a2a.task_complete throttling. Inbound bursts (e.g.
        # multiple peers fanning out work) shouldn't pin the bird at
        # full LLM cost — coalesce repeated submissions of the same
        # task within the window via dedupe_key.
        rate_limit=RateLimit(per_minute=10, per_hour=60),
        coalescing_window=timedelta(seconds=5),
        attention_policy=AttentionPolicy(),
        resources=frozenset(),
        # A→B→A self-spawn is the cycle risk; block it. A peer that
        # creates a task addressed to itself bypasses this source's
        # purpose (the inbound wake) and would loop without bound.
        allow_self_loops=False,
        log_redaction=RedactionPolicy(
            summarize=_a2a_submitted_redact,
            store_raw_trusted=False,
            redact_caller_identifier=True,
        ),
        retention_days=14,
    )


# ---------------------------------------------------------------------------
# Signal builder
# ---------------------------------------------------------------------------


def build_signal_for_submitted_task(
    candidate: NativeTaskSnapshot, target_agent: str,
) -> Signal:
    """Wake the admitted recipient using ownership facts outside peer metadata."""
    candidate = require_native_task_snapshot(candidate, target_agent)
    task = candidate.task
    if task.status.state not in {TaskState.TASK_STATE_SUBMITTED, TaskState.TASK_STATE_WORKING}:
        raise ValueError("Task submission wake requires a submitted or working native task")
    metadata = MessageToDict(task.metadata)
    correlation = parse_question_correlation(metadata)
    chain = deserialize_causation_chain(metadata)
    request_content = (message_text(task.history[0]) or "") if task.history else ""
    scope = NativeRecipientScope(candidate.ownership.recipient_agent_id, candidate.ownership.tenant)
    return Signal(
        source=SOURCE_NAME, kind="inbound", mode=SignalMode.COGNITION,
        payload={
            "task_id": task.id,
            "session_id": scope.session_id_for_context(task.context_id),
            "sender": candidate.ownership.creator_agent_id,
            "skill_id": str(metadata.get("skill") or ""),
            "a2a_verb": correlation.kind if correlation is not None else "task",
            "reply_expected": correlation is not None and correlation.kind == "question",
            "question_id": correlation.question_id if correlation is not None else "",
            "request_content": request_content,
        },
        target_agent=target_agent, visibility=Visibility.INTERNAL,
        caller=candidate.ownership.creator_agent_id, urgency=Urgency.NORMAL,
        origin_trust=Trust.TRUSTED, dedupe_key=task.id,
        causation_chain=list(chain),
    )

__all__ = [
    "PROMPT_TEMPLATE",
    "SOURCE_NAME",
    "build_a2a_task_submitted_registration",
    "build_signal_for_submitted_task",
]
