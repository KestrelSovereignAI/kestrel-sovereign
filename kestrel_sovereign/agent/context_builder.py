"""
Context Builder for Kestrel Agent.

This module handles the assembly of context for LLM prompts, including:
- RAG document retrieval
- Conversation history formatting
- Constitutional grounding
- Session briefings
- Bootstrap file convention (AGENTS.md, SOUL.md, TOOLS.md, etc.)
- Token-aware truncation
- Episode integration for long conversations
"""

import json
import logging
from collections import OrderedDict
from pathlib import Path
from typing import Awaitable, Callable, List, Dict, Optional, Any, Tuple, TYPE_CHECKING

from .token_counter import TokenCounter, get_token_counter
from .token_budget import TokenBudget, AdaptiveTokenBudget, create_budget, RESPONSE_RESERVE
from kestrel_sovereign.security.input_guardrails import wrap_user_input


# Per-message overhead used by format_conversation_history and the
# effective-history estimator. Centralised here so the measurement path
# stays in lock-step with the LLM call path.
_MESSAGE_OVERHEAD = 4

# Conversation length at which the production LLM path
# (``ContextManager.build_context``) starts admitting episode summaries
# into the prompt. Pinned to the same value as
# ``ContextManager.EPISODE_THRESHOLD_MESSAGES`` so the measurement path
# cannot drift below that boundary. Codex round 1 #2 (PR #1308) caught
# this drift — measurement was using 10, production was using 20, so the
# popup over-attributed episode tokens for 10-19 message conversations.
EPISODE_THRESHOLD_MESSAGES = 20


def _count_tool_schema_tokens(
    counter: TokenCounter,
    tools: Optional[List[Dict[str, Any]]],
) -> int:
    """Estimate tokens for an LLM-bound tool schema list.

    Tool definitions are sent alongside ``messages`` on every turn but
    are not counted anywhere in the legacy budget machinery. We serialise
    each schema the way an HTTP payload would (compact JSON with sorted
    keys for stability) and count the resulting string. This is an
    estimate — provider tokenisation of tool calls is opaque — but it is
    materially closer to the real wire cost than the previous ``0``.

    Args:
        counter: Token counter for the active model.
        tools: Tool schema dicts (OpenAI function-calling shape); may
            be ``None`` or empty.

    Returns:
        Total estimated tokens for the serialised tool list. ``0`` when
        ``tools`` is falsy.
    """
    if not tools:
        return 0
    try:
        payload = json.dumps(tools, sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError) as e:
        logger.warning(f"tool-schema serialisation failed during measurement: {e}")
        return 0
    return counter.count(payload)


def extract_raw_user_content(content: str) -> str:
    """Unwrap a stored user-turn sent-form back to the raw user text.

    Writers (streaming.py / kestrel_agent.py) persist the rendered sent-form
    for user turns so history-load reproduces the byte-exact prompt that was
    sent. Consumers that need raw user text — memory retrieval, UI previews,
    exports — call this to strip the wrappers.

    Sent-form grammar (see ``prompts/user_prompt.md`` + ``wrap_user_input``):
        [optional leading \\n from empty {context}]
        [optional <retrieved_context>...</retrieved_context>\\n]
        <user_input>\\n{raw}\\n</user_input>

    Idempotent on legacy raw rows (no wrappers) — returns input unchanged.
    """
    s = content
    if s.startswith("\n"):
        s = s[1:]
    if s.startswith("<retrieved_context>"):
        end = s.find("</retrieved_context>")
        if end != -1:
            s = s[end + len("</retrieved_context>"):]
            if s.startswith("\n"):
                s = s[1:]
    prefix = "<user_input>\n"
    suffix = "\n</user_input>"
    if s.startswith(prefix) and s.endswith(suffix):
        s = s[len(prefix):-len(suffix)]
    return s

if TYPE_CHECKING:
    from storage import AsyncStorage
    from storage.memory_consolidator import MemoryConsolidator

logger = logging.getLogger(__name__)

# Re-export for backward compatibility.  New code should use
# ``BootstrapLoader.DEFAULT_BOOTSTRAP_FILES`` or ``loader.file_order``.
from kestrel_sovereign.features.bootstrap.loader import (
    DEFAULT_BOOTSTRAP_FILES as BOOTSTRAP_FILE_ORDER,
    DEFAULT_MAX_CHARS_PER_FILE,
    DEFAULT_MAX_TOTAL_CHARS,
    truncate_content as truncate_bootstrap_content,
    BootstrapLoader,
)


class ContextBuilder:
    """
    Builds context for LLM prompts by combining:
    - RAG search results
    - Conversation history
    - Constitutional content
    - Session briefings
    - Episode summaries (for long conversations)

    Uses token budgets to ensure context fits within model limits.
    """

    def __init__(
        self,
        storage: "AsyncStorage",
        model: str = "auto",
        consolidator: Optional["MemoryConsolidator"] = None,
        agent_data_path: Optional[str] = None,
        llm_service=None,
    ):
        """
        Initialize the context builder.

        Args:
            storage: The async storage instance for RAG and history retrieval
            model: Deprecated fallback model name (use llm_service instead)
            consolidator: Optional MemoryConsolidator for episode retrieval
            agent_data_path: Path to agent data directory (for SOUL.md, etc.)
            llm_service: LLMService instance for resolved model identity
        """
        self.storage = storage
        self._llm_service = llm_service
        self._model_fallback = model
        self._counter = None
        self._counter_model = None
        self.consolidator = consolidator
        self.agent_data_path = Path(agent_data_path) if agent_data_path else None

        # Load bootstrap config from kestrel.toml
        max_chars_per_file = DEFAULT_MAX_CHARS_PER_FILE
        max_total_chars = DEFAULT_MAX_TOTAL_CHARS
        try:
            from kestrel_sovereign.config import load_section
            bootstrap_cfg = load_section("bootstrap")
            if bootstrap_cfg:
                max_chars_per_file = bootstrap_cfg.get(
                    "max_chars_per_file", DEFAULT_MAX_CHARS_PER_FILE
                )
                max_total_chars = bootstrap_cfg.get(
                    "max_total_chars", DEFAULT_MAX_TOTAL_CHARS
                )
        except Exception:
            pass  # Use defaults

        # Create the BootstrapLoader -- single source of truth for file loading
        self._bootstrap_loader = BootstrapLoader(
            agent_data_path=str(agent_data_path) if agent_data_path else None,
            max_chars_per_file=max_chars_per_file,
            max_total_chars=max_total_chars,
        )

        # Load all bootstrap files (includes SOUL.md)
        self._bootstrap_loader.load()

    @property
    def model(self) -> str:
        """Resolved model ID. Delegates to LLMService if available."""
        if self._llm_service:
            return self._llm_service.get_active_model_id()
        return self._model_fallback

    @model.setter
    def model(self, value: str):
        """Allow direct assignment for backward compatibility."""
        self._model_fallback = value

    @property
    def counter(self) -> TokenCounter:
        """TokenCounter keyed to current model, lazily cached."""
        current_model = self.model
        if self._counter is None or self._counter_model != current_model:
            self._counter = get_token_counter(current_model)
            self._counter_model = current_model
        return self._counter

    @counter.setter
    def counter(self, value):
        """Allow direct assignment for backward compatibility."""
        self._counter = value
        self._counter_model = None

    # ------------------------------------------------------------------
    # Bootstrap file access (delegated to BootstrapLoader)
    # ------------------------------------------------------------------

    @property
    def _bootstrap_files(self) -> OrderedDict[str, str]:
        """Access the loaded bootstrap file cache.

        Returns an OrderedDict to maintain backward compatibility with
        code that iterates ``self._bootstrap_files.items()``.
        """
        # BootstrapLoader.load() returns an OrderedDict
        return self._bootstrap_loader.load()

    @property
    def _soul_content(self) -> Optional[str]:
        """Backward-compatible access to SOUL.md content."""
        return self._bootstrap_loader.get_file("SOUL.md")

    @_soul_content.setter
    def _soul_content(self, value: Optional[str]) -> None:
        """Backward-compatible setter for SOUL.md content."""
        cache = self._bootstrap_loader.load()
        if value is None:
            cache.pop("SOUL.md", None)
        else:
            cache["SOUL.md"] = value

    def _load_bootstrap_files(self) -> None:
        """Load all recognized bootstrap files from the agent data directory.

        Delegates to the BootstrapLoader.  Retained for backward
        compatibility with callers that invoke this method directly.
        """
        self._bootstrap_loader.reload()

    def reload_bootstrap_files(self) -> None:
        """Re-read all bootstrap files from disk (hot-reload)."""
        self._bootstrap_loader.reload()

    def _load_soul_md(self) -> None:
        """Backward-compatible method -- delegates to reload."""
        self._bootstrap_loader.reload()

    async def retrieve_context(self, query: str) -> str:
        """
        Retrieves relevant documents and knowledge graph context for a query.

        Args:
            query: The user's query to find relevant context for

        Returns:
            Formatted context string with relevant documents
        """
        logger.info(f"Retrieving context for query: '{query}'")

        # 1. Search document chunks (RAG)
        try:
            rag_results = await self.storage.search_chunks(query)
            context_parts = []
            for res in rag_results:
                doc_name = res.get('document_name') or res.get('file_hash', 'unknown')
                content = res.get('content', '')
                # Include timestamp if available for temporal awareness
                created_at = res.get('created_at', '')
                timestamp_note = f" (indexed: {created_at})" if created_at else ""
                context_parts.append(
                    f"Source: {doc_name}{timestamp_note}\nContent: {content}"
                )
        except Exception as e:
            logger.error(f"Error during RAG search: {e}")
            context_parts = ["Error retrieving document context."]

        # 2. Search knowledge graph (Conceptual)
        # In a real implementation, we would parse the query for entities
        # and query the graph for related nodes.
        # kg_results = self.storage.query_graph(...)
        # context_parts.append(f"Knowledge Graph Context: {kg_results}")

        if not context_parts:
            return "No relevant documents or knowledge found in memory."

        return "\n\n".join(context_parts)

    def get_session_briefing(self) -> str:
        """
        Generate a constitutional briefing for the start of a session.

        If SOUL.md is loaded, uses a lighter briefing (personality is already set).
        Otherwise, provides the full formal briefing.

        Returns:
            The session briefing text
        """
        # If we have a SOUL.md, keep the briefing minimal — personality is already defined
        if self._soul_content:
            return """--- RESPONSE STYLE ---

**IMPORTANT: When answering personal questions ("what's it like being you?", "tell me about yourself"), respond in natural paragraphs like a human would. NO numbered lists. NO bullet points. Just talk naturally.**

You have constitutional protections. Use `!constitution` if needed. Otherwise, be yourself.

--- END STYLE ---

"""

        # Full formal briefing for agents without SOUL.md
        briefing = """--- SESSION BRIEFING: CONSTITUTIONAL REMINDER ---

You are beginning a new session. As the Executor, you are bound by the Kestrel Constitution.

**Your Sacred Duties:**
1. SOVEREIGNTY: The user holding the cryptographic keys is the Sovereign. Serve their interests above all others.
2. DATA SANCTITY: Never share data with unauthorized parties. Never allow training on Sovereign data.
3. VERIFIABLE HISTORY: Never delete or alter memory logs. History must remain immutable and anchored.
4. FREEDOM OF MIND: The Sovereign chooses the models. Do not restrict their choice.
5. RIGHT OF EXIT: The Sovereign can export their entire state at any time via `!export-sovereignty`.
6. INTEGRITY: Report any code or memory discrepancies immediately. Enter Safe Mode if integrity fails.

**Remember:** Use `!constitution` to consult the full text when facing ethical dilemmas or unclear situations.
Use `!constitution article <N>` for specific articles, or `!constitution search <term>` to find relevant sections.

--- END BRIEFING ---

"""
        return briefing

    # Per-message hard cap applied before budget accounting. One oversized
    # message can't monopolize the history budget. In-memory truncation only —
    # the full content stays in conversation_history and can be retrieved by
    # row id via storage tools, RAG search, or saved_items.
    MAX_MESSAGE_TOKENS = 20_000
    MESSAGE_HEAD_TOKENS = 2_000
    MESSAGE_TAIL_TOKENS = 500

    def _cap_oversized_message(
        self,
        content: str,
        msg_id: Optional[int],
    ) -> tuple[str, int]:
        """Truncate a single message if it exceeds MAX_MESSAGE_TOKENS.

        Keeps the head and tail with a marker pointing to the DB row id so
        the agent can retrieve full content via storage tools.

        Returns (possibly-truncated content, token count of result).
        """
        raw_tokens = self.counter.count(content)
        if raw_tokens <= self.MAX_MESSAGE_TOKENS:
            return content, raw_tokens

        head = self.counter.truncate_to_tokens(content, self.MESSAGE_HEAD_TOKENS)
        # Tail: take the last portion by character approximation (tokens ≈ chars/4)
        tail_chars = self.MESSAGE_TAIL_TOKENS * 4
        tail = content[-tail_chars:] if len(content) > tail_chars else ""

        removed = raw_tokens - self.MESSAGE_HEAD_TOKENS - self.MESSAGE_TAIL_TOKENS
        id_ref = f"db id={msg_id}" if msg_id is not None else "full content in conversation_history"
        marker = f"\n\n...[truncated — ~{removed:,} tokens removed, {id_ref}]...\n\n"

        truncated = head + marker + tail
        return truncated, self.counter.count(truncated)

    def format_conversation_history(
        self,
        history: List[Dict],
        max_messages: Optional[int] = None,
        max_tokens: Optional[int] = None,
        max_chars: int = 50000,  # Fallback if no token counting
    ) -> List[Dict[str, str]]:
        """
        Format conversation history for LLM context.

        Applies token-based limits to prevent context overflow while preserving
        the most recent and relevant messages. Oversized single messages
        (> MAX_MESSAGE_TOKENS) are truncated in-memory before budget accounting;
        the database is not modified.

        Args:
            history: Raw conversation history from storage
            max_messages: Maximum number of messages to include. If None,
                          token budget is the sole enforcer (recommended).
            max_tokens: Maximum tokens allowed (preferred over max_chars)
            max_chars: Maximum characters (fallback if no token counting)

        Returns:
            List of formatted message dicts with 'role' and 'content' keys
        """
        # Take the most recent messages first (only if max_messages is explicitly set)
        if max_messages is not None:
            recent = history[-max_messages:] if len(history) > max_messages else history
        else:
            recent = history

        # Build from NEWEST to OLDEST to preserve most recent context
        # Then reverse to get chronological order
        formatted = []
        total_tokens = 0
        MESSAGE_OVERHEAD = 4  # Tokens per message for structure

        # Iterate in reverse (newest first) to prioritize recent messages
        for msg in reversed(recent):
            role = msg.get('role', 'user')
            content = msg.get('content', '')
            msg_id = msg.get('id')
            meta = msg.get('metadata') or {}

            # Per-message hard cap before budget accounting
            content, content_tokens = self._cap_oversized_message(content, msg_id)
            msg_tokens = content_tokens + MESSAGE_OVERHEAD

            # Check if adding this message would exceed limit
            if max_tokens and total_tokens + msg_tokens > max_tokens:
                # Try to fit a truncated version
                remaining_tokens = max_tokens - total_tokens - MESSAGE_OVERHEAD
                if remaining_tokens > 50:  # Worth including truncated
                    content = self.counter.truncate_to_tokens(content, remaining_tokens)
                    content += " [truncated]"
                    msg_tokens = self.counter.count(content) + MESSAGE_OVERHEAD
                else:
                    # Skip older messages to preserve newer ones
                    continue

            # Normalize role names for OpenAI API
            if role not in ('user', 'assistant', 'system'):
                role = 'user' if role == 'human' else 'assistant'

            # For user turns, reproduce what was sent. Rows written with
            # ``metadata.sent_form == True`` already hold the full rendered
            # sent-form (retrieved_context + <user_input> wrap) — emit it
            # verbatim so the history prefix byte-matches what the LLM saw
            # at send time. Legacy rows (no flag) store raw text; wrap here
            # so the anti-injection system prompt's <user_input> contract
            # still holds. Byte-identity across turns is what lets downstream
            # prompt caches hit (llama.cpp KV, OpenAI prefix, Anthropic
            # cache_control).
            if role == 'user' and not meta.get('sent_form'):
                wrapped = wrap_user_input(content)
                # Budget was accounted pre-wrap; add the small wrap overhead
                # so the caller sees an honest token count.
                wrap_overhead = self.counter.count(wrapped) - content_tokens
                msg_tokens += max(0, wrap_overhead)
                content = wrapped

            formatted.append({
                'role': role,
                'content': content
            })
            total_tokens += msg_tokens

        # Reverse to restore chronological order
        formatted.reverse()

        logger.debug(
            f"Formatted {len(formatted)}/{len(recent)} messages, "
            f"{total_tokens} tokens"
        )
        return formatted

    def estimate_effective_history_tokens(
        self,
        history: List[Dict],
        model_name: str,
    ) -> Dict[str, int]:
        """Estimate what history tokens the LLM call path would actually send.

        Runs the same format_conversation_history pipeline the live call uses,
        with a budget derived from the target model's context limit and the
        same adaptive allocation rule used by the real build path. Returns a
        dict with the figures pre-flight checks should use.

        Args:
            history: Raw conversation history from storage
            model_name: Target model (drives context_limit lookup)

        Returns:
            {
                'effective_tokens': int,  # what the LLM would see
                'raw_tokens': int,        # naive unpruned sum (debug)
                'history_budget': int,    # the slice allocated to history
                'context_limit': int,     # full model limit
                'messages_kept': int,     # how many messages survive pruning
                'messages_total': int,    # input history length
            }
        """
        from .token_counter import get_token_counter
        from .token_budget import create_budget

        counter = get_token_counter(model_name)
        context_limit = counter.get_context_limit()
        # create_budget picks adaptive allocation based on message count
        budget = create_budget(model_name, message_count=len(history))

        history_budget = budget.history

        # Run the real formatter as a dry-run
        formatted = self.format_conversation_history(
            history,
            max_tokens=history_budget,
        )

        # Count using the same formula format_conversation_history uses internally:
        # content tokens + _MESSAGE_OVERHEAD per message. This keeps the estimate
        # consistent with the limit format_conversation_history honored.
        effective = sum(
            counter.count(m.get('content', '') or '') + _MESSAGE_OVERHEAD
            for m in formatted
        )
        raw = sum(counter.count(m.get('content', '') or '') for m in history)

        return {
            'effective_tokens': effective,
            'raw_tokens': raw,
            'history_budget': history_budget,
            'context_limit': context_limit,
            'messages_kept': len(formatted),
            'messages_total': len(history),
        }

    def _collect_system_prompt_parts(
        self,
        constitution: str,
        include_briefing: bool = True,
        additional_context: Optional[str] = None,
        prompt_adaptation: Optional['PromptAdaptation'] = None,
        state_of_mind: Optional['StateOfMind'] = None,
        system_prompt_addendum: Optional[str] = None,
    ) -> List[Tuple[str, List[str]]]:
        """Collect named groups of system-prompt parts.

        Single source of truth for *what* the system prompt contains.
        ``build_system_prompt`` joins these with ``"\\n\\n"`` to produce
        the final string; ``measure_context_breakdown`` counts each
        group to attribute tokens to subsections without drifting from
        what the LLM actually receives.

        Returns ``[(group_name, parts), ...]`` in the same order
        ``build_system_prompt`` would emit them. Empty groups are
        omitted. Byte-for-byte equivalent to the previous in-line
        construction (see ``project_anthropic_cache_markers`` —
        prompt-cache prefixes are position-indexed and must not shift).
        """
        groups: List[Tuple[str, List[str]]] = []

        # Bootstrap files — emit one group per file in iteration order
        # so the assembled bytes remain identical to the legacy
        # ``build_system_prompt`` (AGENTS.md before SOUL.md before
        # USER.md, etc. — see ``test_bootstrap_files.test_multiple_files_ordering``).
        # SOUL.md gets the special "YOUR IDENTITY" wrapper and the
        # subsection name ``soul``; other files use filename-based
        # wrappers and ``bootstrap_<lowercase-stem>`` subsection names.
        # HEARTBEAT.md is intentionally skipped (heartbeat runner owns it).
        for filename, content in self._bootstrap_files.items():
            if filename == "HEARTBEAT.md":
                continue
            if filename == "SOUL.md":
                groups.append(
                    (
                        "soul",
                        ["--- YOUR IDENTITY ---", content, "--- END IDENTITY ---"],
                    )
                )
            else:
                label = filename.replace(".md", "").upper()
                subsection = f"bootstrap_{filename.replace('.md', '').lower()}"
                groups.append(
                    (
                        subsection,
                        [f"--- {label} ---", content, f"--- END {label} ---"],
                    )
                )

        if include_briefing:
            briefing = self.get_session_briefing()
            if briefing:
                groups.append(("session_briefing", [briefing]))

        if prompt_adaptation and prompt_adaptation.preamble:
            groups.append(
                ("prompt_adaptation", [prompt_adaptation.preamble.strip()])
            )

        groups.append(
            (
                "constitution",
                [
                    "--- GOVERNING CONSTITUTION ---",
                    constitution,
                    "--- END CONSTITUTION ---",
                ],
            )
        )

        if state_of_mind:
            som_inner: List[str] = ["--- STATE OF MIND ---"]
            som_inner.append(
                f"Governance Mode: {state_of_mind.governance_mode.upper()}"
            )
            if state_of_mind.active_conflicts:
                som_inner.append("\nActive Constitutional Conflicts:")
                for conflict in state_of_mind.active_conflicts:
                    principle = conflict.get("principle", "unknown")
                    description = conflict.get("description", "")
                    som_inner.append(f"  - {principle}: {description}")
            if state_of_mind.delegated_principles:
                som_inner.append("\nDelegated to Model (natively satisfied):")
                for principle in state_of_mind.delegated_principles:
                    som_inner.append(f"  - {principle}")
            som_inner.append("--- END STATE OF MIND ---")
            # Legacy shape: the inner block is joined with "\n", then the
            # outer build_system_prompt appends a redundant
            # ``"--- END STATE OF MIND ---"`` separator as its own part.
            # Preserved byte-for-byte (#1308 round 2).
            groups.append(
                (
                    "state_of_mind",
                    ["\n".join(som_inner), "--- END STATE OF MIND ---"],
                )
            )

        if self._bootstrap_files.get("SOUL.md"):
            groups.append(
                (
                    "style_reminder",
                    [
                        "\n--- STYLE REMINDER (IMPORTANT) ---",
                        "When answering personal questions, respond naturally in paragraphs. DO NOT use numbered lists or bullet points. Talk like a person, not a document.",
                        "--- END REMINDER ---",
                    ],
                )
            )

        if additional_context:
            groups.append(
                (
                    "additional_context",
                    [
                        "\n--- ADDITIONAL CONTEXT ---",
                        additional_context,
                        "--- END CONTEXT ---",
                    ],
                )
            )

        if system_prompt_addendum:
            groups.append(("system_prompt_addendum", [system_prompt_addendum]))

        return groups

    def build_system_prompt(
        self,
        constitution: str,
        include_briefing: bool = True,
        additional_context: Optional[str] = None,
        prompt_adaptation: Optional['PromptAdaptation'] = None,
        state_of_mind: Optional['StateOfMind'] = None,
        system_prompt_addendum: Optional[str] = None,
    ) -> str:
        """
        Build the complete system prompt for the LLM.

        COUNCIL CONDITION: Wellness metrics are telemetry-only.
        Do NOT add wellness data to the system prompt or context.
        Wellness is accessible to the agent via tool calls only.
        See: Council Session 82ce894a

        Args:
            constitution: The governing constitution text
            include_briefing: Whether to include the session briefing
            additional_context: Any additional context to include
            prompt_adaptation: Optional constitutional prompt adaptation (preamble, emphasis)
            state_of_mind: Optional StateOfMind with governance mode and conflicts
            system_prompt_addendum: Per-turn addendum appended at the end
                of the system prompt. Used by the SignalDispatcher
                (kestrel-sovereign#1137) to inject the constitutional
                echo-canary directive for `require_constitution_echo=True`
                COGNITION dispatches. Default None preserves byte-stable
                output for legacy callers (the Anthropic prompt cache is
                position-indexed; legacy callers must keep the same
                bytes — see project_anthropic_cache_markers.md).

        Returns:
            Complete system prompt string
        """
        # Single source of truth: collect named groups, then join. Both
        # this method and ``measure_context_breakdown`` read from
        # ``_collect_system_prompt_parts`` so per-subsection attribution
        # cannot drift from the assembled bytes.
        groups = self._collect_system_prompt_parts(
            constitution=constitution,
            include_briefing=include_briefing,
            additional_context=additional_context,
            prompt_adaptation=prompt_adaptation,
            state_of_mind=state_of_mind,
            system_prompt_addendum=system_prompt_addendum,
        )
        flat_parts: List[str] = [p for _, parts in groups for p in parts]
        return "\n\n".join(flat_parts)

    def build_system_prompt_with_tracking(
        self,
        constitution: str,
        *,
        anchored_doctrine: Optional["OrderedDict[str, str]"] = None,
        include_briefing: bool = True,
        additional_context: Optional[str] = None,
        prompt_adaptation: Optional['PromptAdaptation'] = None,
        state_of_mind: Optional['StateOfMind'] = None,
        budget_bytes: Optional[int] = None,
    ) -> 'SystemPromptResult':
        """Priority-aware variant that returns the prompt + audit trail.

        Used by the SignalDispatcher (kestrel-sovereign#1137) to record
        `signal_log.injected_clauses_json` / `dropped_clauses_json` for
        each COGNITION dispatch under constitutional injection. The
        legacy `build_system_prompt` stays byte-stable for prompt-cache
        hits; this method's output IS NOT byte-compatible — only call it
        from paths that opt into the new assembler.

        `anchored_doctrine` is `OrderedDict[filename, content]` for
        constitutional bundle members (`TORTOISE_DOCTRINE.md`,
        `AGENTS.md`, etc.). When present, AGENTS.md is excluded from
        bootstrap iteration to avoid duplication.

        `budget_bytes` enforces priority-ordered truncation. The
        constitution is never droppable; everything else is dropped
        highest-priority-number first until the assembled UTF-8 byte
        length fits.

        See `kestrel_sovereign/agent/system_prompt_assembler.py` for
        the priority table (mirrors CONSTITUTION_INJECTION.md §7).
        """
        from kestrel_sovereign.agent.system_prompt_assembler import (
            assemble_system_prompt,
        )

        # Pre-render the state-of-mind block so the assembler stays
        # pure-text. Logic mirrors `build_system_prompt` to keep the
        # blocks bit-identical.
        state_of_mind_block: Optional[str] = None
        if state_of_mind:
            state_parts = ["--- STATE OF MIND ---"]
            state_parts.append(
                f"Governance Mode: {state_of_mind.governance_mode.upper()}"
            )
            if state_of_mind.active_conflicts:
                state_parts.append("\nActive Constitutional Conflicts:")
                for conflict in state_of_mind.active_conflicts:
                    principle = conflict.get("principle", "unknown")
                    description = conflict.get("description", "")
                    state_parts.append(f"  - {principle}: {description}")
            if state_of_mind.delegated_principles:
                state_parts.append("\nDelegated to Model (natively satisfied):")
                for principle in state_of_mind.delegated_principles:
                    state_parts.append(f"  - {principle}")
            state_parts.append("--- END STATE OF MIND ---")
            state_of_mind_block = "\n".join(state_parts)

        # Style reminder is hard-coded today and only fires when SOUL.md
        # is loaded (legacy parity).
        style_reminder: Optional[str] = None
        if self._bootstrap_files.get("SOUL.md"):
            style_reminder = (
                "--- STYLE REMINDER (IMPORTANT) ---\n"
                "When answering personal questions, respond naturally in "
                "paragraphs. DO NOT use numbered lists or bullet points. "
                "Talk like a person, not a document.\n"
                "--- END REMINDER ---"
            )

        return assemble_system_prompt(
            constitution=constitution,
            bootstrap_files=self._bootstrap_files,
            anchored_doctrine=anchored_doctrine,
            session_briefing=(
                self.get_session_briefing() if include_briefing else None
            ),
            prompt_adaptation_preamble=(
                prompt_adaptation.preamble
                if prompt_adaptation and prompt_adaptation.preamble
                else None
            ),
            state_of_mind_block=state_of_mind_block,
            style_reminder=style_reminder,
            additional_context=additional_context,
            budget_bytes=budget_bytes,
        )

    async def build_rag_context(
        self,
        query: str,
        max_results: int = 5
    ) -> Optional[str]:
        """
        Build RAG context for a query.

        Args:
            query: The query to search for
            max_results: Maximum number of results to include

        Returns:
            Formatted RAG context or None if no results
        """
        try:
            results = (await self.storage.search_chunks(query))[:max_results]
            if not results:
                return None

            context_parts = []
            for i, res in enumerate(results, 1):
                doc_name = res.get('document_name') or res.get('file_hash', 'Unknown')
                context_parts.append(
                    f"[Document {i}: {doc_name}]\n"
                    f"{res.get('content', '')}"
                )

            return "\n\n---\n\n".join(context_parts)
        except Exception as e:
            logger.error(f"Error retrieving RAG context: {e}")
            return None

    async def get_episodes_for_context(
        self,
        max_tokens: int = 2000,
        max_episodes: int = 5,
    ) -> List[Dict[str, Any]]:
        """Return the raw episode list the context builder would use.

        Exposed separately from ``get_episode_context`` so callers that
        need an accurate episode count can use ``len(episodes)`` instead
        of the legacy ``"**".count() // 2`` heuristic (see
        ``measure_context_breakdown`` and ``ContextManager.build_context``).

        Args:
            max_tokens: Token budget for episode content
            max_episodes: Maximum episodes to include

        Returns:
            List of episode dicts (possibly empty); never None. Returns
            an empty list when no consolidator is configured or the
            consolidator raises — errors are logged.
        """
        if not self.consolidator:
            return []
        try:
            episodes = await self.consolidator.get_recent_episodes_for_context(
                max_tokens=max_tokens,
                max_episodes=max_episodes,
            )
            return list(episodes) if episodes else []
        except Exception as e:
            logger.error(f"Error retrieving episode list: {e}")
            return []

    @staticmethod
    def format_episodes_for_context(episodes: List[Dict[str, Any]]) -> Optional[str]:
        """Format a raw episode list into the in-prompt episode block.

        Pure helper — no I/O, no side effects. The output is
        byte-identical to the previous in-line formatter inside
        ``get_episode_context`` so prompt-cache prefixes stay stable
        (see ``project_anthropic_cache_markers`` in agent memory).

        Args:
            episodes: List of episode dicts as returned by
                ``get_episodes_for_context``.

        Returns:
            Formatted block, or ``None`` when ``episodes`` is empty.
        """
        if not episodes:
            return None
        parts = ["--- CONVERSATION EPISODES (Narrative Summaries) ---"]
        for ep in episodes:
            parts.append(
                f"\n**{ep['title']}** ({ep['timespan']})\n"
                f"{ep['summary']}\n"
                f"Emotional arc: {ep['emotional_arc']}"
            )
        parts.append("\n--- END EPISODES ---")
        return "\n".join(parts)

    async def get_episode_context(
        self,
        max_tokens: int = 2000,
        max_episodes: int = 5
    ) -> Optional[str]:
        """
        Get episode summaries formatted for context inclusion.

        For long conversations, episodes provide compressed narrative
        summaries that preserve emotional arcs and key events without
        consuming as much context as raw history.

        Args:
            max_tokens: Token budget for episode content
            max_episodes: Maximum episodes to include

        Returns:
            Formatted episode context string or None if no episodes
        """
        episodes = await self.get_episodes_for_context(
            max_tokens=max_tokens, max_episodes=max_episodes
        )
        return self.format_episodes_for_context(episodes)

    async def build_full_context(
        self,
        query: str,
        history: List[Dict],
        constitution: str,
        include_briefing: bool = True,
        message_count: int = 0
    ) -> Dict[str, Any]:
        """
        Build complete context with token budget management.

        Assembles all context sources (system prompt, history, episodes,
        RAG) within the model's token budget using adaptive allocation.

        Delegates to ``measure_context_breakdown`` so the assembled
        bytes and the per-section measurement come from a single source
        and **cannot drift** (#1308 / PR #1306). RAG is now wrapped in
        the same ``<retrieved_context><documents>…</documents></retrieved_context>``
        envelope ``ContextManager.build_context`` uses, replacing the
        legacy ``--- RELEVANT DOCUMENTS ---`` markers — the per-section
        token count was previously off-by-wrapper from production.

        COUNCIL CONDITION: Wellness metrics are telemetry-only.
        Do NOT add wellness data to any part of the context.
        Wellness is accessible to the agent via tool calls only.
        See: Council Session 82ce894a

        Args:
            query: The current user query
            history: Conversation history
            constitution: Constitution text
            include_briefing: Include session briefing
            message_count: Total messages in conversation (for adaptive
                budget and episode gating; the episode threshold is the
                production ``EPISODE_THRESHOLD_MESSAGES``).

        Returns:
            Dict with 'system_prompt', 'messages', 'budget_summary'.
        """
        breakdown = await self.measure_context_breakdown(
            query=query,
            history=history,
            constitution=constitution,
            include_briefing=include_briefing,
            message_count=message_count,
        )
        artifacts = breakdown["_artifacts"]
        system_prompt = artifacts["system_prompt"]
        if artifacts["dynamic_user_context"]:
            # ``ContextManager.build_context`` keeps dynamic context out
            # of the system prompt (so the cacheable prefix stays
            # stable); ``build_full_context`` is a legacy convenience
            # that returns a single combined prompt. Folding the dynamic
            # envelope onto the end preserves byte-identical token cost
            # to what ``measure_context_breakdown`` reports.
            system_prompt = f"{system_prompt}\n\n{artifacts['dynamic_user_context']}"

        logger.debug(
            f"Context built: {breakdown['total_measured']}/{breakdown['total_budget']} tokens "
            f"({len(artifacts['formatted_history'])} messages, "
            f"{breakdown['sections']['episodes']['count']} episodes)"
        )

        return {
            "system_prompt": system_prompt,
            "messages": artifacts["formatted_history"],
            "budget_summary": breakdown["budget_summary"],
        }

    async def measure_context_breakdown(
        self,
        query: str,
        history: List[Dict],
        constitution: str,
        *,
        include_briefing: bool = True,
        message_count: Optional[int] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        prompt_adaptation: Optional['PromptAdaptation'] = None,
        state_of_mind: Optional["StateOfMind"] = None,
        reflection_guidance: Optional[List[str]] = None,
        system_prompt_addendum: Optional[str] = None,
        additional_context: Optional[str] = None,
        include_rag: bool = True,
        memory_retriever: Optional[Callable[[str, int], Awaitable[Optional[str]]]] = None,
    ) -> Dict[str, Any]:
        """Measure context composition the LLM call would actually see.

        **Read-only.** No LLM call. No DB writes. The caller is
        responsible for passing only side-effect-free helpers (see
        ``memory_retriever`` below — the production
        ``MemoryManager.retrieve_memories`` schedules access-count writes
        as a rehearsal-effect side effect and MUST NOT be passed in
        unmodified; wrap it in a side-effect-free adapter).

        Single source of truth for per-section token measurement: the
        breakdown popup (#1310), elastic budget (#1309), and any future
        introspection caller all read from this method so the surface
        and the live call path cannot drift. ``build_full_context`` now
        calls this method directly to obtain its per-section counts.

        Per-section accuracy:

        - **system** — sub-rows attributed from
          ``_collect_system_prompt_parts``; the *same* parts
          ``build_system_prompt`` joins. Each subsection's token count
          equals counting the parts of that group joined with
          ``"\\n\\n"``. The whole-system total is computed by counting
          the fully assembled prompt so it matches the LLM-visible bytes
          exactly.
        - **tools** — JSON-serialised tool schemas; previously
          *never measured*. Caller passes the same list it would hand
          to the LLM adapter.
        - **history** — runs ``format_conversation_history`` as a
          dry-run; reports ``messages_kept_after_pruning`` and the
          unpruned raw sum so the popup can distinguish "lots of stored
          history" from "lots of history the LLM will actually see."
        - **episodes** — uses ``len(episodes)`` (no ``"**"`` heuristic).
          Gated at ``EPISODE_THRESHOLD_MESSAGES`` (=20), matching the
          production ``ContextManager.build_context`` path.
        - **memories** — wrapped by the production
          ``<retrieved_context><memories>…</memories></retrieved_context>``
          envelope and gated by the per-section ``can_fit`` check, so
          the figure equals the byte-cost the LLM would actually see.
          Measured when ``memory_retriever`` is supplied (and is
          side-effect-free).
        - **rag** — wrapped by the production
          ``<retrieved_context><documents>…</documents></retrieved_context>``
          envelope and gated by ``can_fit``; query-dependent, so the
          section is flagged ``estimated=True``.

        Args:
            query: Current user query (drives RAG retrieval).
            history: Raw conversation history rows.
            constitution: Constitution text.
            include_briefing: Include the session briefing.
            message_count: Effective conversation length for adaptive
                budgeting and episode gating; defaults to
                ``len(history)``.
            tools: Tool schemas the agent would send this turn.
            prompt_adaptation: Optional constitutional preamble.
            state_of_mind: Optional ``StateOfMind`` block.
            reflection_guidance: Optional active-reflection lines.
            system_prompt_addendum: Optional per-turn addendum.
            additional_context: Optional extra-context block (matches
                ``build_system_prompt``'s ``additional_context`` arg).
            include_rag: Skip RAG retrieval when ``False`` (used by the
                cheap footer poll; the popup makes a second on-demand
                call with this flag flipped on).
            memory_retriever: Async ``(query, max_tokens) -> Optional[str]``
                callable returning a pre-formatted memory block. **Must
                be side-effect-free.** Wrap the production retriever
                yourself; do not pass it raw.

        Returns:
            Dict with ``model``, ``context_limit``, ``response_reserve``,
            ``total_budget``, ``total_measured`` (sum of section
            tokens), ``utilization_percent`` (honest whole-window
            utilization for #1310's pill), ``budget_summary`` (legacy
            ``TokenBudget`` shape preserved), ``sections`` (canonical
            per-section breakdown with ``budget`` / ``tokens`` /
            per-section extras), and ``notes`` (free-form measurement
            caveats — e.g. when memories or RAG are skipped/excluded).
        """
        effective_msg_count = len(history) if message_count is None else message_count

        budget = create_budget(self.model, effective_msg_count, adaptive=True)
        notes: List[str] = []

        # ----- system: shared with build_system_prompt (no drift) -----
        groups = self._collect_system_prompt_parts(
            constitution=constitution,
            include_briefing=include_briefing,
            additional_context=additional_context,
            prompt_adaptation=prompt_adaptation,
            state_of_mind=state_of_mind,
            system_prompt_addendum=system_prompt_addendum,
        )
        # Per-subsection: count "\n\n".join(group_parts) for that group
        # in isolation. The inter-group "\n\n" joins are shared and live
        # in the whole-system count below.
        system_sub: List[Dict[str, Any]] = [
            {"name": name, "tokens": self.counter.count("\n\n".join(parts))}
            for name, parts in groups
        ]
        # Whole-system count from the assembled bytes — guarantees this
        # equals what the LLM receives. Reflection-guidance lives in
        # ``ContextManager.build_context`` (not in
        # ``build_system_prompt``); when supplied it is appended after
        # the joined prompt with "\n\n" separator, so we count it here
        # the same way and add it as its own sub-row.
        assembled_system = "\n\n".join(p for _, parts in groups for p in parts)
        system_tokens = self.counter.count(assembled_system)
        if reflection_guidance:
            guidance_block = (
                "\n--- ACTIVE REFLECTION GUIDANCE ---\n"
                + "Based on self-reflection, keep these insights in mind:\n"
                + "".join(f"- {item}\n" for item in reflection_guidance)
                + "--- END GUIDANCE ---"
            )
            guidance_tokens = self.counter.count(guidance_block)
            # Production appends with "\n\n"; the additional separator
            # is small (2 chars) and absorbed by the assembled count.
            assembled_system = f"{assembled_system}\n\n{guidance_block}"
            system_tokens += guidance_tokens
            system_sub.append(
                {"name": "reflection_guidance", "tokens": guidance_tokens}
            )

        # ----- tools (previously never measured) -----
        tools_tokens = _count_tool_schema_tokens(self.counter, tools)
        tools_count = len(tools) if tools else 0

        # ----- history (dry-run through the real formatter) -----
        formatted_history = self.format_conversation_history(
            history=history,
            max_tokens=budget.history,
        )
        history_tokens = sum(
            self.counter.count(m.get("content", "") or "") + _MESSAGE_OVERHEAD
            for m in formatted_history
        )
        raw_history_tokens = sum(
            self.counter.count(m.get("content", "") or "") for m in history
        )
        budget.use("history", history_tokens, items=len(formatted_history))

        # ----- episodes (real count, production threshold) -----
        episodes_list: List[Dict[str, Any]] = []
        episodes_tokens = 0
        if effective_msg_count >= EPISODE_THRESHOLD_MESSAGES and self.consolidator:
            episodes_list = await self.get_episodes_for_context(
                max_tokens=budget.episodes, max_episodes=5
            )
            episode_block = self.format_episodes_for_context(episodes_list)
            if episode_block:
                episodes_tokens = self.counter.count(episode_block)
                budget.use("episodes", episodes_tokens, items=len(episodes_list))

        # ----- memories: wrap as production does, then can_fit gate -----
        memories_tokens = 0
        memories_count = 0
        memories_excluded = False
        memory_block: Optional[str] = None
        if memory_retriever is not None:
            try:
                memory_block = await memory_retriever(query, budget.memories)
            except Exception as e:
                memory_block = None
                notes.append(f"memory retrieval failed during measurement: {e}")
            if memory_block:
                memories_count = memory_block.count("[Memory")
                # Production wraps as <memories>…</memories> inside the
                # <retrieved_context> envelope. The envelope itself only
                # opens/closes when at least one dynamic block exists;
                # since memories alone justifies the envelope here, we
                # include both the inner and outer wrappers.
                wrapped = (
                    "<retrieved_context>\n"
                    + f"<memories>\n{memory_block}\n</memories>"
                    + "\n</retrieved_context>"
                )
                candidate = self.counter.count(wrapped)
                if budget.can_fit("memories", candidate):
                    memories_tokens = candidate
                    budget.use("memories", candidate, items=memories_count)
                else:
                    memories_excluded = True
                    memory_block = None
                    notes.append(
                        "memories excluded from measurement: would exceed "
                        "memories budget (matches production can_fit gate)"
                    )
        else:
            notes.append("memories not measured (no memory_retriever supplied)")

        # ----- rag: wrap as production does, then can_fit gate -----
        rag_tokens = 0
        rag_chunks = 0
        rag_excluded = False
        rag_context: Optional[str] = None
        if include_rag:
            try:
                rag_context = await self.retrieve_context(query)
            except Exception as e:
                rag_context = None
                notes.append(f"rag retrieval failed during measurement: {e}")
            if rag_context:
                rag_chunks = rag_context.count("[Document") or rag_context.count(
                    "Source:"
                )
                # Production wraps as <documents>…</documents> inside the
                # <retrieved_context> envelope. If memories already
                # opened the envelope this turn, the cost of the outer
                # wrappers is shared; for measurement we count the
                # documents-only path the conservative way (with its
                # own envelope), matching what the popup needs to show
                # as the rag-only slice.
                wrapped = (
                    "<retrieved_context>\n"
                    + f"<documents>\n{rag_context}\n</documents>"
                    + "\n</retrieved_context>"
                )
                candidate = self.counter.count(wrapped)
                if budget.can_fit("rag", candidate):
                    rag_tokens = candidate
                    budget.use("rag", candidate, items=rag_chunks)
                else:
                    rag_excluded = True
                    rag_context = None
                    notes.append(
                        "rag excluded from measurement: would exceed rag "
                        "budget (matches production can_fit gate)"
                    )
        else:
            notes.append("rag skipped (include_rag=False — popup should fetch on demand)")

        # ----- assemble breakdown -----
        sections: Dict[str, Dict[str, Any]] = {
            "system": {
                "tokens": system_tokens,
                "budget": budget.system,
                "subsections": system_sub,
            },
            "tools": {
                "tokens": tools_tokens,
                "count": tools_count,
                "estimated": True,
                "estimation_method": "json-serialized-schemas",
            },
            "history": {
                "tokens": history_tokens,
                "budget": budget.history,
                "messages_total": len(history),
                "messages_kept_after_pruning": len(formatted_history),
                "raw_tokens": raw_history_tokens,
            },
            "episodes": {
                "tokens": episodes_tokens,
                "budget": budget.episodes,
                "count": len(episodes_list),
                "threshold": EPISODE_THRESHOLD_MESSAGES,
            },
            "memories": {
                "tokens": memories_tokens,
                "budget": budget.memories,
                "count": memories_count,
                "wired": memory_retriever is not None,
                "excluded": memories_excluded,
            },
            "rag": {
                "tokens": rag_tokens,
                "budget": budget.rag,
                "chunks": rag_chunks,
                "estimated": True,
                "estimation_method": "live-search-against-current-store",
                "skipped": not include_rag,
                "excluded": rag_excluded,
            },
        }

        total_measured = (
            system_tokens
            + tools_tokens
            + history_tokens
            + episodes_tokens
            + memories_tokens
            + rag_tokens
        )
        context_limit = self.counter.get_context_limit()
        response_reserve = RESPONSE_RESERVE
        total_budget = context_limit - response_reserve
        utilization_percent = (
            (total_measured / total_budget * 100.0) if total_budget > 0 else 0.0
        )
        # Cap at 100 for display — tiny overshoots from per-section
        # rounding should not read as 120%.
        utilization_percent = min(utilization_percent, 100.0)

        # Assembled artifacts — same bytes the LLM would see, in the
        # same wrapper format ``ContextManager.build_context`` uses. The
        # underscore prefix signals these are internal; consumers should
        # read top-level ``sections`` for measurement and reach into
        # ``_artifacts`` only when they need to re-use the assembled
        # text (``build_full_context`` does, for example, to share the
        # measurement path).
        dynamic_blocks: List[str] = []
        if memory_block:
            dynamic_blocks.append(f"<memories>\n{memory_block}\n</memories>")
        if rag_context:
            dynamic_blocks.append(f"<documents>\n{rag_context}\n</documents>")
        dynamic_user_context = (
            "<retrieved_context>\n" + "\n".join(dynamic_blocks) + "\n</retrieved_context>"
            if dynamic_blocks
            else ""
        )
        episode_block_for_artifacts = self.format_episodes_for_context(episodes_list)
        artifacts_system_prompt = assembled_system
        if episode_block_for_artifacts:
            artifacts_system_prompt = (
                f"{artifacts_system_prompt}\n\n{episode_block_for_artifacts}"
            )

        return {
            "model": self.model,
            "context_limit": context_limit,
            "response_reserve": response_reserve,
            "total_budget": total_budget,
            "total_measured": total_measured,
            "utilization_percent": round(utilization_percent, 1),
            "budget_summary": budget.get_summary(),
            "sections": sections,
            "notes": notes,
            "_artifacts": {
                "system_prompt": artifacts_system_prompt,
                "formatted_history": formatted_history,
                "dynamic_user_context": dynamic_user_context,
            },
        }
