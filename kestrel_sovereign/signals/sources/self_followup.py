"""COGNITION cron source for agent-authored follow-up turns (#3101).

An intention formed inside a turn ("verify PR N once CI settles, then
merge") has no way to execute itself: the turn ends and the intention ends
with it. This source is the substrate that carries one across the turn
boundary. The agent persists a one-shot ``self_followup`` schedule through
the ordinary ``schedule_add_deadline`` tool; when the deadline arrives the
scheduler dispatches ``cron.self_followup`` in COGNITION mode and the
intention text is rendered into a genuine turn — not a liveness ping, not
an echo.

Two properties are load-bearing:

* **It really fires a turn.** COGNITION means the dispatcher renders
  :data:`PROMPT_TEMPLATE` and runs the turn lifecycle. An accept that
  produced no turn would be worse than the explicit refusal the Talon
  ``schedule_work_rescue`` shim already gives.
* **It is visible either way.** :func:`self_followup_result_summary` is
  what lets a session-bound wake render live in the originating chat pane
  (SIGNAL_SOURCES_GUIDE rules 1-5); an unattended wake stays ``INTERNAL``
  and log-only. ``SchedulerFeature`` refuses to persist a session-bound
  follow-up whose registration cannot surface, so "bound but invisible"
  is not a reachable state.

Single hop is deliberate (#3101 clarification): ``allow_self_loops=False``
covers the in-chain case, and ``SchedulerFeature`` separately refuses to
persist a new follow-up from inside a follow-up turn — a persisted row
resets the causation chain, so the registry flag alone would not bound the
spend.
"""

from __future__ import annotations

import hashlib
import re
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

# Bare scheduler task name. `cron.self_followup` is the source name; the
# scheduler derives it via `cron_source_name(TASK_NAME)`.
TASK_NAME = "self_followup"

PROMPT_TEMPLATE = (
    Path(__file__).resolve().parents[2]
    / "prompts" / "signals" / "self_followup.md"
)

# The intent is rendered verbatim into a prompt. Cap it so one schedule row
# cannot dominate the turn's context window, and strip control characters so
# a pasted fragment cannot forge template structure.
MAX_INTENT_CHARS = 4000

# Payload keys the schema accepts. `origin_session_id` is written by the
# scheduler from the live turn binding, never by the caller, so an intent
# author cannot choose which chat window its follow-up renders into
# (SIGNAL_SOURCES_GUIDE rule 2).
ALLOWED_PAYLOAD_KEYS = frozenset({"intent", "origin_session_id", "scheduled_at"})

_CONTROL_CHARS = frozenset(
    chr(c) for c in range(0x20) if chr(c) not in "\n\t"
) | {chr(0x7F)}

# Three-or-more backticks: the run length that can close the prompt
# template's fixed fence. Matched greedily so a longer run is defused whole
# rather than leaving a residual fence behind.
_FENCE_RUN = re.compile(r"`{3,}")


class SelfFollowupIntentError(ValueError):
    """The intent text is missing or unusable."""


def normalize_intent(value: Any) -> str:
    """Validate and clean one agent-authored intention.

    Raises :class:`SelfFollowupIntentError` rather than silently accepting an
    empty intention: a follow-up turn with nothing to act on is exactly the
    no-op accept this feature exists to avoid.
    """
    if not isinstance(value, str):
        raise SelfFollowupIntentError(
            f"intent must be a string, got {type(value).__name__}"
        )
    cleaned = "".join(ch for ch in value if ch not in _CONTROL_CHARS).strip()
    if not cleaned:
        raise SelfFollowupIntentError(
            "intent must be a non-empty description of the follow-up work"
        )
    # Neutralize Markdown fences (#3112 gate-2 P2). The prompt template wraps
    # this text in a FIXED three-backtick block. A run of three or more
    # backticks in the intent closes that block early; the template's own
    # closing delimiter then OPENS a second one, which puts the single-hop
    # guidance inside a code block and lets part of the intent render outside
    # the boundary it is advertised as sitting inside. On a feature whose
    # subject is a trust boundary, caller text escaping its rendered boundary
    # is the defect, not a cosmetic one.
    #
    # Neutralized here rather than by computing a longer fence in the template
    # because the template is a static file and the payload contract is a
    # closed key set -- there is nowhere to carry a per-payload delimiter. The
    # substitution is visible in the rendered text rather than silent: an
    # agent that wrote a fence sees that it was defused, instead of seeing its
    # note mysteriously reformatted.
    cleaned = _FENCE_RUN.sub(
        lambda m: "'" * len(m.group(0)) + " (fence defused)", cleaned
    )
    if len(cleaned) > MAX_INTENT_CHARS:
        cleaned = cleaned[:MAX_INTENT_CHARS] + "...(truncated)"
    return cleaned


def _schema(payload: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError(
            f"cron.self_followup payload must be a dict, got "
            f"{type(payload).__name__}"
        )
    extra = set(payload.keys()) - ALLOWED_PAYLOAD_KEYS
    if extra:
        raise ValueError(
            f"cron.self_followup payload has unexpected keys: {sorted(extra)}; "
            f"allowed: {sorted(ALLOWED_PAYLOAD_KEYS)}"
        )
    if "intent" not in payload:
        raise ValueError("cron.self_followup payload missing required key: intent")

    normalized: Dict[str, Any] = {"intent": normalize_intent(payload["intent"])}
    for key in ("origin_session_id", "scheduled_at"):
        value = payload.get(key, "")
        if value is None:
            value = ""
        if not isinstance(value, str):
            raise ValueError(f"cron.self_followup payload {key} must be a string")
        normalized[key] = value
    return normalized


def _redact(payload: Dict[str, Any]) -> str:
    """Audit summary: shape and digest, never the intention body.

    An intention routinely names unreleased work, people, or credentialed
    systems. The digest still proves "this is the same intention we
    persisted" when reconciling a schedule row against its dispatch.
    """
    intent = payload.get("intent", "") or ""
    digest = hashlib.sha256(intent.encode("utf-8")).hexdigest()[:12]
    bound = "yes" if (payload.get("origin_session_id") or "") else "no"
    return (
        f"cron.self_followup intent_len={len(intent)} "
        f"intent_sha256_12={digest} session_bound={bound}"
    )


def self_followup_result_summary(body: Any) -> str:
    """Bounded inline body for the ``signal_completed`` UI side channel.

    For a COGNITION dispatch the result is the agent's own response text.
    Returning it is what lets an open chat tab render the follow-up turn the
    moment it lands, instead of it staying invisible until a refresh
    (#2877/#2922). The store caps the returned text.
    """
    if body is None:
        return ""
    text = body if isinstance(body, str) else str(body)
    if len(text) > 2000:
        return text[:2000] + "...(truncated)"
    return text


def build_self_followup_registration() -> SourceRegistration:
    """Source registration for the agent's own scheduled follow-up turn."""
    return SourceRegistration(
        name=f"cron.{TASK_NAME}",
        schema=_schema,
        default_mode=SignalMode.COGNITION,
        allowed_modes=frozenset({SignalMode.COGNITION}),
        prompt_template=PROMPT_TEMPLATE,
        # The intention is authored by this agent inside its own turn, at the
        # same trust level as any other thing the agent decides to do. The
        # schema still normalizes it, and the template fences it.
        trust=Trust.TRUSTED,
        # Each follow-up is a persisted one-shot row, so the deadline is the
        # real throttle. The cap is defense against a burst of rows all
        # coming due together; drops are recorded as `missed`, never success.
        rate_limit=RateLimit(per_minute=10, per_hour=60),
        # Distinct intentions must not coalesce into one another. Deadlines
        # are per-row and never share a dedupe key, so keep the window off.
        coalescing_window=None,
        # A follow-up is scheduled for a specific moment the agent chose;
        # deferring it to a quiet-hours boundary would silently change what
        # the agent asked for.
        attention_policy=AttentionPolicy(),
        # CONVERSATION is owned solely by the turn lifecycle.
        resources=frozenset(),
        # Single hop: a follow-up may not chain a further follow-up. The
        # schedule-time refusal in SchedulerFeature is the load-bearing half
        # (a persisted row resets the causation chain); this covers the
        # in-chain case.
        allow_self_loops=False,
        log_redaction=RedactionPolicy(
            summarize=_redact,
            store_raw_trusted=False,
            redact_caller_identifier=True,
        ),
        # Paired with session_id + USER_VISIBLE on the dispatched signal when
        # the follow-up is bound to a chat session.
        result_summary=self_followup_result_summary,
        retention_days=30,
    )
