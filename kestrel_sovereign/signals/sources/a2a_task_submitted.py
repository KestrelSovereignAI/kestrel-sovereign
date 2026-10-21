"""Native inbound A2A cognition wakes after task admission commits.

The manager delivers a native task snapshot with server-owned admission facts.
The recipient builds its signal from those facts, maps the peer context into
an isolated chat projection, and parses the declared correlation and causation
extensions. Caller metadata cannot choose the trusted sender. The persisted
task and lifecycle revision also drive restart reconciliation.
"""

from __future__ import annotations

import json
import logging
from datetime import timedelta
from pathlib import Path

from a2a.types import Role, TaskState
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
    parse_delivery_intent,
    parse_question_correlation,
    parse_skill_hint,
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
    / "prompts"
    / "signals"
    / "a2a_task_submitted.md"
)


# ---------------------------------------------------------------------------
# Schema / redaction
# ---------------------------------------------------------------------------


def submitted_input_identity(
    task_id: str, input_revision: int, *, working_revision: int | None = None
) -> str:
    """Encode an unambiguous durable identity for exactly one admitted input."""
    if not isinstance(task_id, str) or not task_id:
        raise ValueError("Input wake requires a concrete task ID")
    if type(input_revision) is not int or input_revision < 0:
        raise ValueError("Input wake requires a nonnegative input revision")
    values = [task_id, input_revision]
    if working_revision is not None:
        if type(working_revision) is not int or working_revision < 0:
            raise ValueError("Working wake requires a nonnegative lifecycle revision")
        values.append(working_revision)
    return json.dumps(values, separators=(",", ":"))


def parse_submitted_input_identity(value: str) -> tuple[str, int]:
    """Read privacy-retained identity without ever substituting a newer input."""
    try:
        parts = json.loads(value)
    except (TypeError, ValueError) as error:
        raise ValueError("Invalid task input wake identity") from error
    if not isinstance(parts, list) or len(parts) not in {2, 3}:
        raise ValueError("Invalid task input wake identity")
    canonical = submitted_input_identity(
        parts[0], parts[1], working_revision=parts[2] if len(parts) == 3 else None
    )
    if canonical != value:
        raise ValueError("Noncanonical task input wake identity")
    return parts[0], parts[1]


def _a2a_submitted_schema(payload: dict) -> dict:
    if not isinstance(payload, dict):
        raise ValueError(
            f"a2a.task_submitted payload must be a dict, got {type(payload).__name__}"
        )
    if type(payload.get("input_revision")) is not int or payload["input_revision"] < 0:
        raise ValueError("a2a.task_submitted requires a nonnegative input_revision")
    for key in ("task_id", "session_id", "sender"):
        if key not in payload:
            raise ValueError(f"a2a.task_submitted payload missing required key: {key}")
        if not isinstance(payload[key], str):
            raise ValueError(f"a2a.task_submitted payload {key} must be a string")
    # A native task may have no application-specific verb or skill hint.
    if "a2a_verb" in payload and not isinstance(payload["a2a_verb"], str):
        raise ValueError("a2a.task_submitted payload a2a_verb must be a string")
    if "reply_expected" in payload and not isinstance(payload["reply_expected"], bool):
        raise ValueError("a2a.task_submitted payload reply_expected must be a bool")
    # Keep cumulative user text in the prompt; full structured parts remain
    # available through the native task's history.
    if "request_content" in payload and not isinstance(payload["request_content"], str):
        raise ValueError("a2a.task_submitted payload request_content must be a string")
    # Optional application labels still render with the canonical template.
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
        f"task_id={payload.get('task_id', '?')} "
        f"sender={payload.get('sender', '?')} "
        f"verb={payload.get('a2a_verb', '?')} "
        f"skill={payload.get('skill_id', '?')}"
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
    candidate: NativeTaskSnapshot,
    target_agent: str,
) -> Signal:
    """Wake the admitted recipient using ownership facts outside peer metadata."""
    candidate = require_native_task_snapshot(candidate, target_agent)
    task = candidate.task
    if not candidate.cognition_wake_eligible:
        raise ValueError("This task is not eligible for a cognition wake")
    if task.status.state not in {
        TaskState.TASK_STATE_SUBMITTED,
        TaskState.TASK_STATE_WORKING,
    }:
        raise ValueError(
            "Task submission wake requires a submitted or working native task"
        )
    metadata = MessageToDict(task.metadata)
    correlation = parse_question_correlation(metadata)
    intent = parse_delivery_intent(metadata)
    admitted = candidate.input_request
    if candidate.input_revision and admitted is None:
        raise ValueError("Task input wake has no admitted request")
    latest_metadata = (
        MessageToDict(admitted.metadata) if admitted is not None else metadata
    )
    chain = deserialize_causation_chain(latest_metadata)
    request_content = "\n\n".join(
        text
        for message in task.history
        if message.role == Role.ROLE_USER and (text := message_text(message))
    )
    scope = NativeRecipientScope(
        candidate.ownership.recipient_agent_id, candidate.ownership.tenant
    )
    return Signal(
        source=SOURCE_NAME,
        kind="inbound",
        mode=SignalMode.COGNITION,
        payload={
            "task_id": task.id,
            "input_revision": candidate.input_revision,
            "session_id": scope.session_id_for_context(task.context_id),
            "sender": candidate.ownership.creator_agent_id,
            "skill_id": parse_skill_hint(latest_metadata) or "",
            "a2a_verb": "question"
            if correlation is not None
            else "message"
            if intent == "notification"
            else "task",
            "reply_expected": correlation is not None or intent == "assignment",
            "question_id": correlation.question_id if correlation is not None else "",
            "request_content": request_content,
        },
        target_agent=target_agent,
        visibility=Visibility.INTERNAL,
        caller=candidate.ownership.creator_agent_id,
        urgency=Urgency.NORMAL,
        origin_trust=Trust.TRUSTED,
        dedupe_key=submitted_input_identity(task.id, candidate.input_revision),
        causation_chain=list(chain),
    )


__all__ = [
    "PROMPT_TEMPLATE",
    "SOURCE_NAME",
    "build_a2a_task_submitted_registration",
    "build_signal_for_submitted_task",
    "parse_submitted_input_identity",
    "submitted_input_identity",
]
