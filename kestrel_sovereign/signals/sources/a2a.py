"""Native A2A completion wakes from committed task ownership.

The task manager passes an official Task together with its durable creator,
recipient, tenant and lifecycle revision. Metadata supplies application labels
and the versioned causation extension; it never establishes wake authority.
The dispatcher uses the restored causation frames for depth and cycle checks.
"""

from __future__ import annotations

import logging
from datetime import timedelta
from pathlib import Path
from typing import Any

from a2a.types import Task
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

from kestrel_sovereign.a2a.extensions import CAUSATION_URI, deserialize_causation_chain
from kestrel_sovereign.a2a.extensions import (
    serialize_chain_for_metadata as serialize_chain_for_metadata,
)
from kestrel_sovereign.a2a.lifecycle import TERMINAL_STATES, task_state_label
from kestrel_sovereign.a2a.stores.unified.task_store import (
    NativeTaskSnapshot,
    require_native_task_snapshot,
)

logger = logging.getLogger(__name__)


SOURCE_NAME = "a2a.task_complete"
DURABLE_COGNITION_CONSUMER_ID = "core.a2a-task-complete-cognition-v1"
PROMPT_TEMPLATE = (
    # Inside the package (#1415). ``parents[2]`` resolves to
    # ``<package>/kestrel_sovereign/`` so the template ships with the
    # wheel; ``parents[3]`` pointed at the repo root, which only
    # worked from a source checkout and silently failed on every PyPI
    # install — every COGNITION wake-up using this source raised
    # FileNotFoundError in ``_render_prompt`` and the recipient agent
    # was never woken.
    Path(__file__).resolve().parents[2] / "prompts" / "signals" / "a2a_task_complete.md"
)
METADATA_KEY = CAUSATION_URI


# ---------------------------------------------------------------------------
# Schema / redaction
# ---------------------------------------------------------------------------


def _a2a_schema(payload: dict) -> dict:
    """Required fields: task_id, task_state, result_summary. Reject
    anything malformed so the bird never thinks about a half-built
    notification."""
    if not isinstance(payload, dict):
        raise ValueError(f"a2a payload must be a dict, got {type(payload).__name__}")
    for key in ("task_id", "task_state", "result_summary"):
        if key not in payload:
            raise ValueError(f"a2a payload missing required key: {key}")
    if not isinstance(payload["task_id"], str):
        raise ValueError("task_id must be a string")
    if not isinstance(payload["task_state"], str):
        raise ValueError("task_state must be a string")
    return payload


def _a2a_redact(payload: dict) -> str:
    """Task summaries can contain peer-agent output (work artifacts,
    chat replies, etc.). Cap the summary at 200 chars in signal_log so
    we keep enough context to debug "what task completed" without
    persisting potentially long artifact text."""
    task_id = payload.get("task_id", "<missing>")
    state = payload.get("task_state", "<missing>")
    summary = payload.get("result_summary", "") or ""
    if len(summary) > 200:
        summary = summary[:200] + "...(truncated)"
    return f"task_id={task_id} state={state} summary={summary!r}"


def _a2a_result_summary(result_body: Any) -> str:
    """Phase 7 of #889: bounded body for the UI side channel. The
    cognition turn's result is whatever process_input returned — the
    bird's response text after deciding what to do with the peer-task
    completion. Surface it so a USER_VISIBLE/ADMIN_VISIBLE
    a2a.task_complete signal renders meaningfully rather than a
    metadata-only "the bird did a thing" toast.

    The store hard-caps at MAX_RESULT_SUMMARY_BYTES; the per-source
    cap here is gentler (1000 chars) so reasonably-sized bird responses
    come through intact."""
    if result_body is None:
        return ""
    text = result_body if isinstance(result_body, str) else str(result_body)
    if len(text) > 1000:
        return text[:1000] + "...(truncated)"
    return text


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def build_a2a_task_complete_registration() -> SourceRegistration:
    """A2A peers are TRUSTED in the v1 trust model — they're other
    sovereign agents we explicitly registered with our TaskManager.
    No sanitizer required (the schema validator catches malformed
    payloads). If we ever federate with untrusted peers, set
    trust=UNTRUSTED here and add a sanitizer for the payload's
    `result_summary` field."""
    return SourceRegistration(
        name=SOURCE_NAME,
        schema=_a2a_schema,
        default_mode=SignalMode.COGNITION,
        allowed_modes=frozenset({SignalMode.COGNITION}),
        prompt_template=PROMPT_TEMPLATE,
        trust=Trust.TRUSTED,
        # Conservative throttle: A2A completion is the cycle-risk
        # surface (#894). Cap per-minute to absorb bursty completions
        # without letting a runaway peer pin the bird at 100% LLM cost.
        rate_limit=RateLimit(per_minute=10, per_hour=60),
        # Multiple completions of the same task within a short window
        # would be the same notification re-fired (e.g. retry storm).
        coalescing_window=timedelta(seconds=5),
        # No quiet hours — task completions wake the bird regardless of
        # operator-configured quiet windows. If you don't want this,
        # use a runtime config to suppress the source via rate_limit=0.
        attention_policy=AttentionPolicy(),
        # CONVERSATION is owned by the turn lifecycle (Phase 2).
        # No other resources declared — the cognition turn already
        # serializes against other turns through CONVERSATION.
        resources=frozenset(),
        # NEVER allow self-loops — that's the entire point of cycle
        # detection on this source. A→B→A→B→A would otherwise be
        # bounded only by TTL, which is too generous.
        allow_self_loops=False,
        log_redaction=RedactionPolicy(
            summarize=_a2a_redact,
            store_raw_trusted=False,
            redact_caller_identifier=True,
        ),
        result_summary=_a2a_result_summary,
        retention_days=14,
    )


# ---------------------------------------------------------------------------
# Signal builder
# ---------------------------------------------------------------------------


def build_signal_for_completed_task(
    candidate: NativeTaskSnapshot, target_agent: str
) -> Signal:
    """Build a terminal wake from the native row and its durable principals."""
    candidate = require_native_task_snapshot(candidate, target_agent)
    task = candidate.task
    if not candidate.cognition_wake_eligible:
        raise ValueError("This task is not eligible for a cognition wake")
    if task.status.state not in TERMINAL_STATES:
        raise ValueError("Task completion wake requires a terminal native task")
    chain = deserialize_causation_chain(MessageToDict(task.metadata))
    state = task_state_label(task.status.state)
    return Signal(
        source=SOURCE_NAME,
        kind="terminal",
        mode=SignalMode.COGNITION,
        payload={
            "task_id": task.id,
            "task_state": state,
            "result_summary": _summarize_task_result(task),
        },
        target_agent=target_agent,
        caller=candidate.ownership.creator_agent_id,
        visibility=Visibility.INTERNAL,
        urgency=Urgency.NORMAL,
        origin_trust=Trust.TRUSTED,
        dedupe_key=f"{task.id}:{state}",
        causation_chain=list(chain),
    )


def _summarize_task_result(task: Task) -> str:
    """Return a short text summary of the task outcome. Uses the last
    status message text if present, else the task state. Bounded to
    keep the prompt template render manageable."""
    status_msg = getattr(getattr(task, "status", None), "message", None)
    if status_msg is not None and getattr(status_msg, "parts", None):
        for part in status_msg.parts:
            text = getattr(part, "text", None)
            if isinstance(text, str) and text:
                return text if len(text) <= 1000 else text[:1000] + "...(truncated)"
    return f"<no message; final state: {task_state_label(task.status.state)}>"
