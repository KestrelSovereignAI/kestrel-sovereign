"""
Typed section/build plan shared by the two context assemblers.

``ContextManager.build_context`` (the production LLM path) and
``ContextBuilder.measure_context_breakdown`` (the read-only introspection
path) historically each open-coded their own copy of the section
vocabulary — memory/RAG wrappers, the ``<retrieved_context>`` envelope,
episode/reflection formatting, the ``[Memory]`` / ``[Document]`` counters
— and drifted (#2523 / #2534). This module is the *single* definition of
that vocabulary plus the production-only stage primitives (lumpy history
anchoring, tool-result microcompaction, elastic section finalization,
durable-salvage span computation), so the two callers cannot disagree on
the bytes each section contributes.

Design contract (#2523 required invariant): context construction is a
composition of explicit section results followed by one auditable
finalization boundary. Every section reports content/messages, token
cost, item count, provenance, and any required persistence. The
coordinator (``build_context`` vs ``measure_context_breakdown``) applies
budget/slack policy; the section *definitions* live here.

Deliberately dependency-light: this module imports only the standard
library so both ``context_manager`` and ``context_builder`` can import it
without an import cycle. It never imports either of them back.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
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
# Shared section-content vocabulary (consumed by BOTH assemblers)
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


#: The empty ``<retrieved_context>`` envelope. ``measure_context_breakdown``
#: counts this once as ``dynamic_context_overhead`` when at least one
#: dynamic block is present, mirroring the single-wrapper behavior below.
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

    Byte-identical to the two former in-line constructions in
    ``build_context`` and ``measure_context_breakdown``.
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


class SectionDestination(Enum):
    """Where a section's produced bytes land in the final assembly."""

    #: Appended to the stable, cacheable system prompt (episodes).
    SYSTEM = "system"
    #: Placed in the per-turn ``<retrieved_context>`` block (memories, RAG).
    DYNAMIC = "dynamic"
    #: Conversation history messages.
    HISTORY = "history"


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


# ---------------------------------------------------------------------------
# Shared section producers (consumed by BOTH assemblers)
# ---------------------------------------------------------------------------
#
# These turn already-retrieved raw section content into a typed
# :class:`SectionResult`: token cost (on the RAW block — the byte the
# budget gate charges), item count, and the wrapped/append bytes that
# actually land in the assembly. The *retrieval* differs between the two
# callers (production runs the side-effectful ``MemoryManager``;
# measurement runs a side-effect-free adapter), but the normalization
# after retrieval — count, wrap, gate-input — is single-sourced here so
# ``build_context`` and ``measure_context_breakdown`` cannot disagree on
# the bytes/counts a section contributes (#2523 / #2534). Each caller
# then applies its OWN budget/persistence policy to the returned result.


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
    """The leading history messages that left the model-visible slice."""

    dropped_messages: List[Dict[str, Any]]
    dropped_ids: List[int]
    token_estimate: int
    session_id: Optional[str]


def compute_pruned_span(
    history: List[Dict[str, Any]],
    formatted_history: List[Dict[str, Any]],
    count_tokens: Callable[[str], int],
) -> Optional[PrunedSpan]:
    """Identify the oldest span pruned from the model-visible history.

    Returns ``None`` when nothing was dropped or when no dropped row
    carries an ``id`` (legacy un-tagged rows have nothing durable to
    salvage against). The session id is derived from the first dropped
    row's own metadata so the salvage marker stays non-leaking across
    sessions (#713).
    """
    if len(formatted_history) >= len(history):
        return None
    dropped_count = len(history) - len(formatted_history)
    dropped = history[:dropped_count]
    dropped_ids = [m["id"] for m in dropped if m.get("id") is not None]
    if not dropped_ids:
        return None
    token_estimate = sum(
        count_tokens(m.get("content", "") or "") + _MESSAGE_OVERHEAD for m in dropped
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
        dropped_messages=dropped,
        dropped_ids=dropped_ids,
        token_estimate=token_estimate,
        session_id=session_id,
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
    now = datetime.now(timezone.utc).isoformat()

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
                "cleared_at": now,
            }
        )
        msg["content"] = marker
        cleared += 1

    return cleared
