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

import logging
from collections import OrderedDict
from pathlib import Path
from typing import List, Dict, Optional, Any, TYPE_CHECKING

from .token_counter import TokenCounter, get_token_counter
from .token_budget import TokenBudget, AdaptiveTokenBudget, create_budget
from kestrel_sovereign.security.input_guardrails import wrap_user_input


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
        # content tokens + MESSAGE_OVERHEAD per message. This keeps the estimate
        # consistent with the limit format_conversation_history honored.
        MESSAGE_OVERHEAD = 4
        effective = sum(
            counter.count(m.get('content', '') or '') + MESSAGE_OVERHEAD
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

    def build_system_prompt(
        self,
        constitution: str,
        include_briefing: bool = True,
        additional_context: Optional[str] = None,
        prompt_adaptation: Optional['PromptAdaptation'] = None,
        state_of_mind: Optional['StateOfMind'] = None
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

        Returns:
            Complete system prompt string
        """
        parts = []

        # Bootstrap files — injected into system prompt in order.
        # SOUL.md gets special "YOUR IDENTITY" wrapper; others use filename-based wrappers.
        # AGENTS.md goes first (if present), then SOUL.md, then the rest.
        for filename, content in self._bootstrap_files.items():
            if filename == "SOUL.md":
                parts.append("--- YOUR IDENTITY ---")
                parts.append(content)
                parts.append("--- END IDENTITY ---")
            elif filename == "HEARTBEAT.md":
                # HEARTBEAT.md is loaded by the heartbeat runner separately;
                # skip it in the normal system prompt to avoid duplication.
                continue
            else:
                label = filename.replace(".md", "").upper()
                parts.append(f"--- {label} ---")
                parts.append(content)
                parts.append(f"--- END {label} ---")

        if include_briefing:
            parts.append(self.get_session_briefing())

        # Add constitutional preamble if provided
        if prompt_adaptation and prompt_adaptation.preamble:
            parts.append(prompt_adaptation.preamble.strip())

        parts.append("--- GOVERNING CONSTITUTION ---")
        parts.append(constitution)
        parts.append("--- END CONSTITUTION ---")

        # Add state of mind section if available
        if state_of_mind:
            state_parts = []
            state_parts.append("--- STATE OF MIND ---")
            state_parts.append(f"Governance Mode: {state_of_mind.governance_mode.upper()}")
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
            parts.append("\n".join(state_parts))
            parts.append("--- END STATE OF MIND ---")

        # Add style reminder at the end (models pay more attention to end of context)
        if self._bootstrap_files.get("SOUL.md"):
            parts.append("\n--- STYLE REMINDER (IMPORTANT) ---")
            parts.append("When answering personal questions, respond naturally in paragraphs. DO NOT use numbered lists or bullet points. Talk like a person, not a document.")
            parts.append("--- END REMINDER ---")

        if additional_context:
            parts.append("\n--- ADDITIONAL CONTEXT ---")
            parts.append(additional_context)
            parts.append("--- END CONTEXT ---")

        return "\n\n".join(parts)

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
        if not self.consolidator:
            return None

        try:
            episodes = await self.consolidator.get_recent_episodes_for_context(
                max_tokens=max_tokens,
                max_episodes=max_episodes
            )

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

        except Exception as e:
            logger.error(f"Error retrieving episode context: {e}")
            return None

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

        COUNCIL CONDITION: Wellness metrics are telemetry-only.
        Do NOT add wellness data to any part of the context.
        Wellness is accessible to the agent via tool calls only.
        See: Council Session 82ce894a

        Args:
            query: The current user query
            history: Conversation history
            constitution: Constitution text
            include_briefing: Include session briefing
            message_count: Total messages in conversation (for adaptive budget)

        Returns:
            Dict with 'system_prompt', 'messages', 'budget_summary'
        """
        # Create adaptive budget based on conversation length
        budget = create_budget(self.model, message_count, adaptive=True)

        # 1. Build system prompt (uses 'system' allocation)
        system_prompt = self.build_system_prompt(
            constitution=constitution,
            include_briefing=include_briefing
        )
        system_tokens = self.counter.count(system_prompt)
        budget.use("system", system_tokens)

        # 2. Get episodes for long conversations (uses 'episodes' allocation)
        episode_context = None
        if message_count >= 10 and self.consolidator:
            episode_context = await self.get_episode_context(
                max_tokens=budget.episodes,
                max_episodes=5
            )
            if episode_context:
                episode_tokens = self.counter.count(episode_context)
                budget.use("episodes", episode_tokens)
                # Append episodes to system prompt
                system_prompt = f"{system_prompt}\n\n{episode_context}"

        # 3. Format conversation history (uses 'history' allocation)
        formatted_history = self.format_conversation_history(
            history=history,
            max_tokens=budget.history
        )
        history_tokens = self.counter.count_messages(formatted_history)
        budget.use("history", history_tokens, items=len(formatted_history))

        # 4. Retrieve RAG context (uses 'rag' allocation)
        rag_context = await self.retrieve_context(query)
        if rag_context:
            rag_tokens = self.counter.count(rag_context)
            if budget.can_fit("rag", rag_tokens):
                budget.use("rag", rag_tokens)
                # Add RAG to system prompt
                system_prompt = (
                    f"{system_prompt}\n\n"
                    f"--- RELEVANT DOCUMENTS ---\n{rag_context}\n--- END DOCUMENTS ---"
                )

        logger.debug(
            f"Context built: {budget.total_used}/{budget.total_budget} tokens "
            f"({len(formatted_history)} messages, "
            f"{'with' if episode_context else 'no'} episodes)"
        )

        return {
            "system_prompt": system_prompt,
            "messages": formatted_history,
            "budget_summary": budget.get_summary(),
        }
