"""Signal source for inbound A2A task submission.

Closes the missing wakeup gap surfaced by Emma/Meridian: until this source
existed, when peer agent A created an A2A task addressed to agent B, B's
TaskStore got the row but B's cognition loop was never poked. The
``a2a.task_complete`` source (already in `a2a.py`) wakes the CALLER when
their outbound task finishes; this source is the symmetric INBOUND wake
that fires on the CALLEE when a new task is submitted to them.

The wiring mirrors `channels.feature.py:425`, the proven inbound pattern:

    1. Sender calls TaskManager.create_task → task saved to store.
    2. create_task builds a signal via build_signal_for_a2a_task_submitted.
    3. create_task calls dispatcher.enqueue_signal(signal).
    4. Dispatcher wakes the target agent.
    5. Next cognition turn sees the task in context and decides what to do.

Trust posture: A2A peers are TRUSTED in the v1 model (other sovereign
agents we explicitly registered). Same posture as `a2a.task_complete`. If
we ever federate with untrusted peers, flip to UNTRUSTED and add a
sanitizer.

Cycle safety: `allow_self_loops=False` and a dedupe key keyed on
``task_id`` mean a peer that retries the same task creation (idempotency
retry, network duplicate) collapses to one cognition wake.
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


SOURCE_NAME = "a2a.task_submitted"
PROMPT_TEMPLATE = (
    Path(__file__).resolve().parents[3]
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
    task: Any, target_agent: str, sender: str = "",
) -> Signal:
    """Build an inbound COGNITION signal for an A2A task that just
    landed in the target's store.

    ``task`` is duck-typed (anything with .id, .sessionId, .metadata)
    so this helper is testable without constructing real pydantic Task
    objects. ``sender`` is the calling agent's identifier (DID or
    name) — defaults to empty when the task wasn't created via an
    inter-agent send path (e.g. local agent spawning its own
    background task).

    Causation chain: rehydrated from ``task.metadata["causation_chain"]``
    (serialized form, same shape ``serialize_chain_for_metadata``
    produces). Threading the chain into the Signal lets the dispatcher's
    cycle detection catch A→B→A ping-pong at depth 2 — without it,
    every inbound task started fresh at depth 1 and the loop bound
    was only the per-source rate limit (codex P1 on PR #1366).
    """
    # Import locally to avoid a circular dependency between this
    # source module and a2a.py.
    from kestrel_sovereign.signals.sources.a2a import _deserialize_chain

    task_id = str(getattr(task, "id", "<unknown>"))
    session_id = str(getattr(task, "sessionId", "") or "")
    metadata = getattr(task, "metadata", {}) or {}
    skill_id = ""
    # ``a2a_verb`` discriminates message vs question vs task at the
    # cognition-prompt level — codex P2 on PR #1380. Without this, an
    # empty-skill message looks indistinguishable from an empty-skill
    # task to the receiver. The PeersFeature send_a2a_* tools each
    # stamp it; tasks created via other paths (local self-spawn,
    # subagent dispatch, etc.) leave it empty and the prompt falls
    # back to generic "task" framing.
    a2a_verb = ""
    reply_expected = False
    if isinstance(metadata, dict):
        skill_id = str(metadata.get("skill") or metadata.get("skill_id") or "")
        a2a_verb = str(metadata.get("a2a_verb") or "")
        reply_expected = bool(metadata.get("reply_expected", False))
    chain = _deserialize_chain(metadata if isinstance(metadata, dict) else {})
    return Signal(
        source=SOURCE_NAME,
        kind="inbound",
        mode=SignalMode.COGNITION,
        payload={
            "task_id": task_id,
            "session_id": session_id,
            "sender": sender or str(metadata.get("sender", "") or ""),
            "skill_id": skill_id,
            "a2a_verb": a2a_verb,
            "reply_expected": reply_expected,
        },
        target_agent=target_agent,
        visibility=Visibility.INTERNAL,
        caller=sender or None,
        urgency=Urgency.NORMAL,
        # Idempotency retries (same task_id resubmitted) collapse to
        # one wake within the registration's coalescing window.
        dedupe_key=task_id,
        # Carry the lineage from the upstream turn that spawned this
        # task so cycle detection rejects A→B→A loops at depth 2.
        causation_chain=list(chain),
    )


__all__ = [
    "PROMPT_TEMPLATE",
    "SOURCE_NAME",
    "build_a2a_task_submitted_registration",
    "build_signal_for_submitted_task",
]
