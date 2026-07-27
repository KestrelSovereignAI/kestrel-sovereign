"""
Typed section results and the canonical context-build plan.

``ContextManager.build_context_plan`` is the only context orchestrator.
Production commits the plan's declared side effects before rendering it;
status surfaces render the same plan in dry-run mode.  The legacy
``ContextBuilder.measure_context_breakdown`` method is only a compatibility
adapter over that plan (#2523 / #2534).

Design contract (#2523 required invariant): context construction is a
composition of explicit section results followed by one auditable
finalization boundary. Every section reports content/messages, token
cost, item count, provenance, and any required persistence. The
coordinator applies budget/slack policy; the section *definitions* live here.

Deliberately dependency-light: this module imports only the standard
library so both ``context_manager`` and ``context_builder`` can import it
without an import cycle. It never imports either of them back.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Per-message overhead used by the salvage token estimate and the lumpy
# anchor. Mirrors ``context_builder._MESSAGE_OVERHEAD`` /
# ``format_conversation_history``'s internal ``MESSAGE_OVERHEAD`` so the
# stage math stays in lock-step with the LLM call path. Kept local (not
# imported from context_builder) to preserve the leaf-module property.
_MESSAGE_OVERHEAD = 4


# ---------------------------------------------------------------------------
# Canonical section-content vocabulary
# ---------------------------------------------------------------------------
#
# These strings are byte-load-bearing: downstream LLM prompt caches
# (Anthropic position-indexed cache_control, llama.cpp per-slot KV,
# OpenAI prefix cache) hit only when the emitted bytes are stable across
# turns. Do not reflow — see ``project_anthropic_cache_markers``.

#: Wrapper for the retrieved-memory block placed in dynamic user context.
def wrap_memories(text: str) -> str:
    """Wrap a formatted memory block in its ``<memories>`` envelope."""
    return f"<memories>\n{text}\n</memories>"


#: Wrapper for the retrieved-RAG block placed in dynamic user context.
def wrap_documents(text: str) -> str:
    """Wrap a formatted RAG block in its ``<documents>`` envelope."""
    return f"<documents>\n{text}\n</documents>"


#: The empty ``<retrieved_context>`` envelope used to attribute the one shared
#: dynamic wrapper without charging it once per retrieved section.
RETRIEVED_CONTEXT_EMPTY_ENVELOPE = "<retrieved_context>\n\n</retrieved_context>"


def assemble_dynamic_user_context(blocks: List[str]) -> str:
    """Join dynamic blocks into the per-turn ``<retrieved_context>`` block.

    Returns ``""`` when ``blocks`` is empty so callers can drop the block
    straight into a ``format()`` without emitting dangling wrapper tags.
    This is kept OUT of the system prompt by construction so the cacheable
    system prefix stays byte-stable across turns.
    """
    if not blocks:
        return ""
    return "<retrieved_context>\n" + "\n".join(blocks) + "\n</retrieved_context>"


def count_memory_blocks(text: str) -> int:
    """Count ``[Memory`` markers in a formatted memory block."""
    return text.count("[Memory")


def count_rag_chunks(text: str) -> int:
    """Count RAG chunks by ``[Document`` markers, falling back to ``Source:``."""
    return text.count("[Document") or text.count("Source:")


def build_reflection_guidance_block(items: List[str]) -> str:
    """Render the ACTIVE REFLECTION GUIDANCE block from guidance lines.

    Byte-identical to the former in-line constructions.
    """
    return (
        "\n--- ACTIVE REFLECTION GUIDANCE ---\n"
        + "Based on self-reflection, keep these insights in mind:\n"
        + "".join(f"- {item}\n" for item in items)
        + "--- END GUIDANCE ---"
    )


#: Fixed notice appended to the system prompt in EPHEMERAL privacy mode.
EPHEMERAL_NOTICE = (
    "--- EPHEMERAL MODE ACTIVE ---\n"
    "This conversation is not being recorded. "
    "No history or memories are available.\n"
    "--- END NOTICE ---"
)


# ---------------------------------------------------------------------------
# Typed per-build plan
# ---------------------------------------------------------------------------


class ContextBuildMode(str, Enum):
    """How a context plan will be consumed.

    Planning is read-only in both modes.  ``LIVE`` means the caller will
    subsequently commit the plan's declared side effects before rendering it;
    ``DRY_RUN`` means those requirements are only reported.
    """

    LIVE = "live"
    DRY_RUN = "dry_run"


class SectionStatus(str, Enum):
    """Whether a section was measured and admitted to the model view."""

    INCLUDED = "included"
    EMPTY = "empty"
    EXCLUDED = "excluded"
    SKIPPED = "skipped"
    UNKNOWN = "unknown"
    ERROR = "error"


class SectionDestination(str, Enum):
    """Where a section's produced bytes land in the final assembly."""

    #: Appended to the stable, cacheable system prompt (episodes).
    SYSTEM = "system"
    #: Placed in the per-turn ``<retrieved_context>`` block (memories, RAG).
    DYNAMIC = "dynamic"
    #: Conversation history messages.
    HISTORY = "history"
    #: Provider tool-schema payload, outside chat-message content.
    TOOLS = "tools"


@dataclass
class SectionResult:
    """What a single context section reports back to the coordinator.

    Producers build this (retrieve + format + count) *without* mutating
    the budget; the coordinator's commit step applies budget/slack policy
    and folds the result into :class:`ContextAssembly`. Keeping the two
    apart is what lets the same producer logic be exercised in isolation
    and prevents a content stage from silently owning budget policy.
    """

    name: str
    destination: SectionDestination
    tokens: int = 0
    items: int = 0
    #: For ``SYSTEM`` sections: the text appended after ``"\n\n"``.
    append_text: Optional[str] = None
    #: For ``DYNAMIC`` sections: the already-wrapped block to append.
    dynamic_block: Optional[str] = None
    #: Memory rehearsal-effect targets, recorded only after insertion.
    message_ids: Tuple[int, ...] = ()
    #: True when the retrieved memories came back as a structured block.
    is_memory_block: bool = False
    #: Populated when retrieval raised; surfaced as a build warning.
    warning: Optional[str] = None
    #: Set by the coordinator when the section's bytes were committed.
    committed: bool = False
    #: Content-free producer observability carried into the context plan.
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ContextAssembly:
    """Typed per-build state for ``ContextManager.build_context``.

    Exactly one instance is created per ``build_context`` call (a local),
    replacing the loose locals the former 667-line procedure mutated in
    place. Because it is never a shared instance/class attribute,
    concurrent builds cannot cross-contaminate counters or results —
    per-task isolation is a structural property, not a discipline.

    ``dynamic_user_context`` is a *computed* view over ``dynamic_blocks``
    so per-turn retrieved context can never leak into ``system_prompt``:
    the stable system prefix and the volatile user context are separate
    fields by construction.
    """

    system_prompt: str = ""
    dynamic_blocks: List[str] = field(default_factory=list)
    formatted_history: List[Dict[str, str]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    episode_count: int = 0
    memory_count: int = 0
    rag_chunks: int = 0
    injected_clauses: Optional[List[str]] = None
    dropped_clauses: Optional[List[str]] = None

    @property
    def dynamic_user_context(self) -> str:
        """Assemble the per-turn ``<retrieved_context>`` block (or ``""``)."""
        return assemble_dynamic_user_context(self.dynamic_blocks)


@dataclass
class ContextSectionPlan:
    """Final, model-visible decision for one section of a context plan.

    ``tokens`` is ``None`` when the caller deliberately chose the cheap
    measurement path.  This distinction is load-bearing: an omitted RAG or
    memory lookup is unknown/skipped, never a measured zero.
    """

    name: str
    destination: SectionDestination
    status: SectionStatus
    tokens: Optional[int]
    budget: Optional[int] = None
    items: Optional[int] = None
    provenance: Tuple[str, ...] = ()
    reason: Optional[str] = None
    raw_tokens: Optional[int] = None
    details: Dict[str, Any] = field(default_factory=dict)

    @property
    def included(self) -> bool:
        return self.status is SectionStatus.INCLUDED


@dataclass
class ContextBuildPlan:
    """Read-only description of one production context build.

    The plan owns the exact rendered artifacts, prune decisions, section
    accounting, provenance, warnings, and required writes for a turn.
    ``ContextManager.build_context`` commits ``memory_access_ids`` and
    ``salvage_requirement`` before rendering a live result; diagnostics render
    the same object in :attr:`ContextBuildMode.DRY_RUN` without committing.
    """

    mode: ContextBuildMode
    model: str
    assembly: ContextAssembly
    sections: Dict[str, ContextSectionPlan]
    budget_summary: Dict[str, Any]
    context_limit: int
    response_reserve: int
    total_budget: int
    total_tokens: int
    mandatory_system_tokens: int = 0
    state_of_mind: Any = None
    degraded_mode: bool = False
    degraded_reason: Optional[str] = None
    memory_access_ids: Tuple[int, ...] = ()
    salvage_requirement: Optional["PrunedSpan"] = None
    pruned_span: Optional["PrunedSpan"] = None
    durable_salvage_enabled: bool = False
    measurement_complete: bool = True
    microcompacted_tool_results: int = 0

    @property
    def utilization_percent(self) -> float:
        if self.total_budget <= 0:
            return 0.0
        return round(min(100.0, self.total_tokens / self.total_budget * 100.0), 1)

    @property
    def warnings(self) -> List[str]:
        return self.assembly.warnings

    def to_breakdown(self) -> Dict[str, Any]:
        """Render the stable API measurement shape without artifact bodies."""

        rendered: Dict[str, Dict[str, Any]] = {}
        for name, section in self.sections.items():
            row: Dict[str, Any] = {
                "tokens": section.tokens,
                "status": section.status.value,
                "provenance": list(section.provenance),
            }
            if section.budget is not None:
                row["budget"] = section.budget
            if section.items is not None:
                row["count"] = section.items
            if section.raw_tokens is not None:
                row["raw_tokens"] = section.raw_tokens
            if section.reason:
                row["reason"] = section.reason
            row.update(section.details)
            rendered[name] = row

        pruned_span = self.pruned_span
        salvage_requirement = self.salvage_requirement
        unmappable_count = (
            pruned_span.unmappable_count if pruned_span is not None else 0
        )
        if salvage_requirement is not None and unmappable_count:
            salvage_status = "partial_required_not_committed"
        elif salvage_requirement is not None:
            salvage_status = "required_not_committed"
        elif pruned_span is not None and self.durable_salvage_enabled:
            salvage_status = "unavailable_no_persistent_ids"
        elif pruned_span is not None:
            salvage_status = "disabled"
        else:
            salvage_status = "not_required"

        return {
            "model": self.model,
            "context_limit": self.context_limit,
            "response_reserve": self.response_reserve,
            "total_budget": self.total_budget,
            "total_measured": self.total_tokens,
            "utilization_percent": self.utilization_percent,
            "budget_summary": self.budget_summary,
            "sections": rendered,
            "notes": list(self.warnings),
            "measurement_complete": self.measurement_complete,
            "dry_run": self.mode is ContextBuildMode.DRY_RUN,
            "salvage": {
                "feature_enabled": self.durable_salvage_enabled,
                "required": salvage_requirement is not None,
                "status": salvage_status,
                "message_count": (
                    len(salvage_requirement.dropped_ids)
                    if salvage_requirement is not None
                    else 0
                ),
                "pruned_message_count": (
                    pruned_span.total_dropped_count
                    if pruned_span is not None
                    else 0
                ),
                "unmappable_message_count": unmappable_count,
                "token_estimate": (
                    salvage_requirement.token_estimate
                    if salvage_requirement is not None
                    else 0
                ),
                "silent_prune_possible": (
                    not self.durable_salvage_enabled
                    or (
                        pruned_span is not None
                        and unmappable_count > 0
                    )
                ),
            },
            "microcompacted_tool_results": self.microcompacted_tool_results,
        }


# ---------------------------------------------------------------------------
# Canonical section producers
# ---------------------------------------------------------------------------
#
# These turn already-retrieved raw section content into a typed
# :class:`SectionResult`: token cost (on the RAW block — the byte the
# budget gate charges), item count, and the wrapped/append bytes that
# actually land in the assembly. Retrieval remains read-only while the plan is
# built, and normalization after retrieval — count, wrap, gate-input — is
# single-sourced here so the plan's bytes/counts cannot drift (#2523 / #2534).
# The canonical coordinator applies budget policy and records any required
# rehearsal/salvage effects on the plan for the live commit boundary.


def build_memory_section(
    memory_text: str,
    count_tokens: Callable[[str], int],
    *,
    message_ids: Tuple[int, ...] = (),
    is_memory_block: bool = False,
) -> SectionResult:
    """Normalize a retrieved memory block into the shared DYNAMIC section.

    ``tokens`` is the RAW block cost (what both budget gates charge);
    ``dynamic_block`` is the ``<memories>``-wrapped text that lands in the
    per-turn ``<retrieved_context>`` envelope.
    """
    return SectionResult(
        name="memories",
        destination=SectionDestination.DYNAMIC,
        tokens=count_tokens(memory_text),
        items=count_memory_blocks(memory_text),
        dynamic_block=wrap_memories(memory_text),
        message_ids=message_ids,
        is_memory_block=is_memory_block,
    )


def build_rag_section(
    rag_context: str,
    count_tokens: Callable[[str], int],
) -> SectionResult:
    """Normalize retrieved RAG context into the shared DYNAMIC section.

    ``tokens`` is the RAW block cost (the budget-gate input);
    ``dynamic_block`` is the ``<documents>``-wrapped text.
    """
    return SectionResult(
        name="rag",
        destination=SectionDestination.DYNAMIC,
        tokens=count_tokens(rag_context),
        items=count_rag_chunks(rag_context),
        dynamic_block=wrap_documents(rag_context),
    )


def build_episode_section(
    episode_context: str,
    episode_count: int,
    count_tokens: Callable[[str], int],
) -> SectionResult:
    """Normalize a formatted episode block into the shared SYSTEM section.

    ``append_text`` is joined onto the stable system prompt with ``"\\n\\n"``
    by the caller; ``items`` is the true ``len(episodes)`` count (never the
    legacy ``"**"`` heuristic).
    """
    return SectionResult(
        name="episodes",
        destination=SectionDestination.SYSTEM,
        tokens=count_tokens(episode_context),
        items=episode_count,
        append_text=episode_context,
    )


# ---------------------------------------------------------------------------
# Elastic finalization boundary (centralized slack release)
# ---------------------------------------------------------------------------


def finalize_section(budget: Any, name: str) -> None:
    """Release ``name``'s unused budget into the elastic pool.

    Centralizes the former per-stage
    ``if hasattr(budget, "mark_section_finalized"): ...`` policy branch so
    content stages no longer carry budget-shape knowledge. A no-op for the
    legacy non-elastic budgets that lack the method.
    """
    marker = getattr(budget, "mark_section_finalized", None)
    if marker is not None:
        marker(name)


# ---------------------------------------------------------------------------
# History-emit byte selection + lumpy anchoring (production stage)
# ---------------------------------------------------------------------------


def emit_content_for_msg(
    msg: Dict[str, Any],
    wrap_user_input: Callable[[str], str],
) -> str:
    """Select the bytes a message will emit in ``format_conversation_history``.

    Mirrors the emit-byte selection in
    ``ContextBuilder.format_conversation_history`` so the lumpy anchor
    counts the SAME bytes the LLM will see. Without this, the anchor can
    over-estimate fit for ``sent_form`` user/system rows (whose
    ``rendered_content`` may be larger than raw ``content``) and let the
    formatter fall into its just-enough skip path — recreating the
    per-turn cache churn the anchor is supposed to prevent.
    """
    role = msg.get("role", "user")
    if role not in ("user", "assistant", "system"):
        role = "user" if role == "human" else "assistant"
    raw = msg.get("content", "") or ""
    rendered = msg.get("rendered_content")
    meta = msg.get("metadata") or {}
    if role in ("user", "system") and meta.get("sent_form") and rendered is not None:
        return rendered
    if role == "user" and not meta.get("sent_form"):
        return wrap_user_input(raw)
    return rendered if rendered is not None else raw


def compute_lumpy_anchor(
    history: List[Dict[str, Any]],
    max_tokens: int,
    *,
    prune_target_frac: float,
    count_msg_tokens: Callable[[Dict[str, Any]], int],
) -> int:
    """Compute the oldest-message index to KEEP for a cache-stable window.

    When the full history fits the ceiling, returns 0 (include all). When
    it doesn't, advances the anchor in chunks of
    ``(1 - prune_target_frac) * max_tokens`` so the anchor stays put across
    multiple turns of growth before jumping forward. That hysteresis is
    the whole point: the prefix at ``messages[-2]`` / ``messages[-4]`` is
    byte-stable across the turns between anchor jumps, so Anthropic's
    position-indexed cache markers compound.

    ``count_msg_tokens`` must count the SAME bytes the formatter emits
    (see :func:`emit_content_for_msg`); the caller adds the per-message
    overhead here so the estimate matches ``count_messages``.
    """
    if not history or max_tokens <= 0:
        return 0
    msg_tokens = [count_msg_tokens(m) + _MESSAGE_OVERHEAD for m in history]
    total = sum(msg_tokens)
    if total <= max_tokens:
        return 0
    chunk = max(1, int(max_tokens * (1.0 - prune_target_frac)))
    overage = total - max_tokens
    # Round drop UP to the next chunk boundary so the anchor only advances
    # in lumpy steps. Between steps, overage growth within a chunk's worth
    # of tokens leaves the anchor untouched.
    chunks = max(1, math.ceil(overage / chunk))
    target_drop = chunks * chunk
    dropped = 0
    anchor = 0
    while anchor < len(history) - 1 and dropped < target_drop:
        dropped += msg_tokens[anchor]
        anchor += 1
    return anchor


# ---------------------------------------------------------------------------
# Durable salvage span (production stage — fail-closed boundary input)
# ---------------------------------------------------------------------------


@dataclass
class PrunedSpan:
    """The leading history messages that left the model-visible slice.

    ``dropped_messages`` contains only rows with durable integer ids because
    those are the only rows the salvage transaction can link.  The aggregate
    counters preserve the full prune shape so diagnostics never describe a
    mixed persistent/in-memory span as fully salvageable.
    """

    dropped_messages: List[Dict[str, Any]]
    dropped_ids: List[int]
    token_estimate: int
    session_id: Optional[str]
    total_dropped_count: int
    unmappable_count: int


def compute_pruned_span(
    history: List[Dict[str, Any]],
    formatted_history: List[Dict[str, Any]],
    count_tokens: Callable[[str], int],
) -> Optional[PrunedSpan]:
    """Identify the oldest span pruned from the model-visible history.

    Returns ``None`` only when nothing was dropped. Rows without durable
    integer ids remain represented by ``unmappable_count`` but are excluded
    from ``dropped_messages`` and the token estimate passed to the salvage
    transaction. The session id is derived from the first dropped row carrying
    it, including an id-less row, so a mixed span's marker stays scoped to the
    already session-filtered acquisition input (#713).
    """
    if len(formatted_history) >= len(history):
        return None
    dropped_count = len(history) - len(formatted_history)
    dropped = history[:dropped_count]
    mappable = [
        m
        for m in dropped
        if isinstance(m.get("id"), int) and not isinstance(m.get("id"), bool)
    ]
    dropped_ids = [m["id"] for m in mappable]
    token_estimate = sum(
        count_tokens(m.get("content", "") or "") + _MESSAGE_OVERHEAD
        for m in mappable
    )
    session_id: Optional[str] = None
    for m in dropped:
        meta = m.get("metadata")
        if isinstance(meta, dict):
            sid = meta.get("session_id")
            if sid:
                session_id = sid
                break
    return PrunedSpan(
        dropped_messages=mappable,
        dropped_ids=dropped_ids,
        token_estimate=token_estimate,
        session_id=session_id,
        total_dropped_count=len(dropped),
        unmappable_count=len(dropped) - len(mappable),
    )


# ---------------------------------------------------------------------------
# Microcompaction (production stage — zero-cost tool-result clearing)
# ---------------------------------------------------------------------------


def microcompact_tool_results(history: List[Dict], keep_recent: int) -> int:
    """Clear stale tool-result content from conversation history in place.

    Replaces old tool-result content with JSON markers while preserving
    ``tool_call_id`` pairing (required by LLM APIs). The most recent
    ``keep_recent`` tool results are kept intact. Runs BEFORE
    ``format_conversation_history`` normalizes roles, so ``role="tool"``
    is still identifiable.

    Returns the number of tool results cleared.
    """
    if keep_recent < 1:
        keep_recent = 1

    # Collect indices of tool result messages (preserving order)
    tool_indices: List[int] = []
    for i, msg in enumerate(history):
        if msg.get("role") != "tool":
            continue
        # Skip protected or already excluded messages
        meta = msg.get("metadata") or {}
        if isinstance(meta, str):
            try:
                meta = json.loads(meta)
            except (json.JSONDecodeError, TypeError):
                # Can't read protection flags — assume protected, skip
                continue
        if meta.get("context_priority") == "protected":
            continue
        if meta.get("excluded_from_context"):
            continue
        if meta.get("decay_protected"):
            continue
        tool_indices.append(i)

    if len(tool_indices) <= keep_recent:
        return 0

    # Keep the last N, clear the rest
    to_clear = tool_indices[:-keep_recent]
    cleared = 0
    for idx in to_clear:
        msg = history[idx]
        content = msg.get("content", "")

        # Already cleared?
        if isinstance(content, str) and content.startswith('{"cleared":'):
            continue

        # Build informative marker
        tool_name = ""
        summary = ""
        meta = msg.get("metadata") or {}
        if isinstance(meta, str):
            try:
                meta = json.loads(meta)
            except (json.JSONDecodeError, TypeError):
                meta = {}
        tool_name = meta.get("tool_name", "")

        # Extract summary from content (first 100 chars of the result)
        if isinstance(content, str):
            summary = content[:100].replace('"', '\\"')

        marker = json.dumps(
            {
                "cleared": True,
                "tool_name": tool_name,
                "summary": summary,
                # Stable across live/dry planning and repeated turns.  A wall
                # clock here made identical histories produce different
                # model-visible bytes on every status poll.
                "cleared_at": msg.get("created_at") or "context-plan",
            }
        )
        msg["content"] = marker
        cleared += 1

    return cleared
