"""Source registration for A2A task completion (Phase 5 of #889).

When a peer agent finishes work an agent spawned, the local
`TaskManager._notify_status_update` callback fires. Today that callback
queues an SSE notification for the browser. Under the dispatcher, it
ALSO enqueues a COGNITION signal so the bird wakes up and decides what
to do with the result.

Causation chain propagation:
- When a turn dispatches an outbound A2A task, the caller attaches the
  current causation chain (serialized) to `task.metadata["causation_chain"]`.
- When the task completes locally, `build_signal_for_completed_task`
  rehydrates the chain and threads it into the Signal envelope.
- The dispatcher's append-and-cycle-check rejects A→B→A ping-pong at
  depth 2 by spotting `(A, a2a.task_complete)` as a repeated frame.

Outbound task creation reads the in-flight causation chain from the
agent/TaskManager seam and serializes it into task metadata. The receive
side (this module) rehydrates that chain whenever a completion arrives.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

from kestrel_sdk.signals import (
    AttentionPolicy,
    CausationFrame,
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


SOURCE_NAME = "a2a.task_complete"
PROMPT_TEMPLATE = (
    Path(__file__).resolve().parents[3]
    / "prompts" / "signals" / "a2a_task_complete.md"
)
METADATA_KEY = "causation_chain"


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


def build_signal_for_completed_task(task: Any, target_agent: str) -> Signal:
    """Build a COGNITION signal envelope for a terminal A2A task.

    Reads the causation chain from `task.metadata["causation_chain"]`
    if present (serialized as a list of dicts). Falls back to an empty
    chain for tasks created without dispatcher awareness — those signals
    pass cycle detection by definition (no prior frames).

    `task` is duck-typed (anything with .id, .status.state, .metadata,
    optionally .artifacts) so this helper is testable without
    constructing real pydantic Task objects.
    """
    chain = _deserialize_chain(getattr(task, "metadata", None) or {})
    task_id = str(getattr(task, "id", "<unknown>"))
    task_state = _state_to_str(task)

    return Signal(
        source=SOURCE_NAME,
        kind="terminal",
        mode=SignalMode.COGNITION,
        payload={
            "task_id": task_id,
            "task_state": task_state,
            "result_summary": _summarize_task_result(task),
        },
        target_agent=target_agent,
        visibility=Visibility.INTERNAL,
        urgency=Urgency.NORMAL,
        # dedupe_key is (task_id, terminal_state) so retry storms or
        # double-fired terminal callbacks for the same task collapse
        # within the registration's coalescing_window. Without this
        # the dispatcher can't coalesce — see #905 review P2.
        dedupe_key=f"{task_id}:{task_state}",
        causation_chain=list(chain),
    )


def serialize_chain_for_metadata(chain: list[CausationFrame]) -> list[dict]:
    """Serialize a causation chain for storage in `task.metadata`. Use
    when an in-flight turn spawns an outbound A2A task — the receiver
    (us, when the task completes) reconstructs the chain via
    `_deserialize_chain`."""
    return [
        {
            "agent_id": f.agent_id,
            "source": f.source,
            "signal_id": f.signal_id,
            "turn_id": f.turn_id,
            "depth": f.depth,
            "emitted_at": f.emitted_at.isoformat(),
        }
        for f in chain
    ]


def _deserialize_chain(metadata: dict) -> list[CausationFrame]:
    """Inverse of `serialize_chain_for_metadata`. Defensive against
    missing keys / wrong shapes (peer-supplied metadata is TRUSTED but
    A2A messages traverse the wire — corruption is possible)."""
    raw = metadata.get(METADATA_KEY)
    if not raw or not isinstance(raw, list):
        return []
    frames: list[CausationFrame] = []
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        try:
            frames.append(
                CausationFrame(
                    agent_id=str(entry["agent_id"]),
                    source=str(entry["source"]),
                    signal_id=str(entry["signal_id"]),
                    turn_id=entry.get("turn_id"),
                    depth=int(entry["depth"]),
                    emitted_at=_parse_iso(entry["emitted_at"]),
                )
            )
        except (KeyError, ValueError, TypeError) as e:
            logger.warning(
                "Dropping malformed CausationFrame from a2a metadata: %s", e
            )
    return frames


def _parse_iso(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value
    return datetime.fromisoformat(str(value))


def _state_to_str(task: Any) -> str:
    state = getattr(getattr(task, "status", None), "state", None)
    if state is None:
        return "unknown"
    # TaskState is a str-Enum; .value is the canonical string.
    return getattr(state, "value", str(state))


def _summarize_task_result(task: Any) -> str:
    """Return a short text summary of the task outcome. Uses the last
    status message text if present, else the task state. Bounded to
    keep the prompt template render manageable."""
    status_msg = getattr(getattr(task, "status", None), "message", None)
    if status_msg is not None and getattr(status_msg, "parts", None):
        for part in status_msg.parts:
            text = getattr(part, "text", None)
            if isinstance(text, str) and text:
                return text if len(text) <= 1000 else text[:1000] + "...(truncated)"
    return f"<no message; final state: {_state_to_str(task)}>"
