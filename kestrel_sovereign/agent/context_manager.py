"""
Unified Context Manager for Kestrel Agent.

Orchestrates all context sources:
- Conversation history with token-aware truncation
- Episode summaries for long conversations
- Emotionally-weighted memory retrieval
- RAG document search (hybrid: embeddings + BM25)
- Constitutional grounding

This is the single integration point for context assembly,
replacing scattered context retrieval throughout the codebase.

Refactored to serve as an orchestration layer that delegates to specialized managers.
"""

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, TYPE_CHECKING

from .context_builder import ContextBuilder
from .token_counter import TokenCounter, get_token_counter
from .token_budget import TokenBudget, create_budget
from .conversation_manager import ConversationManager
from .memory_manager import MemoryManager
from .tool_context_manager import ToolContextManager

if TYPE_CHECKING:
    from kestrel_sovereign.storage import AsyncStorage
    from kestrel_sovereign.storage.memory_consolidator import MemoryConsolidator
    from kestrel_sovereign.storage.memory_retriever import MemoryRetriever
    from kestrel_sovereign.llm.service import LLMService

logger = logging.getLogger(__name__)


@dataclass
class ContextResult:
    """Result of context assembly."""
    system_prompt: str
    messages: List[Dict[str, str]]
    total_tokens: int
    budget_summary: Dict[str, Any]
    episode_count: int = 0
    memory_count: int = 0
    rag_chunks: int = 0
    warnings: List[str] = field(default_factory=list)
    # Query-dependent context retrieved for this turn (memories + RAG).
    # Lives OUTSIDE the system message so the system prefix stays stable
    # across turns and downstream LLM prompt caches (llama.cpp per-slot KV,
    # OpenAI prefix cache, Anthropic cache_control) can actually hit.
    # Callers prepend this to the current user message content.
    dynamic_user_context: str = ""


class ContextManager:
    """
    Unified context manager for the Kestrel agent.

    Responsibilities:
    1. Orchestrate specialized managers for different context concerns
    2. Manage token budgets across all context sources
    3. Decide when to use episodes vs raw history
    4. Coordinate RAG search with hybrid approach
    5. Ensure privacy mode compliance

    Usage:
        manager = ContextManager(storage)
        result = await manager.build_context(
            query="Tell me about our conversation yesterday",
            agent_id="agent-123",
            constitution=constitution_text
        )
    """

    # Threshold for using episodes vs raw history
    EPISODE_THRESHOLD_MESSAGES = 20

    def __init__(
        self,
        storage: "AsyncStorage",
        model: str = "auto",
        agent_id: Optional[str] = None,
        consolidator: Optional["MemoryConsolidator"] = None,
        memory_retriever: Optional["MemoryRetriever"] = None,
        llm_service: Optional["LLMService"] = None,
        context_builder: Optional[ContextBuilder] = None,
    ):
        """
        Initialize the context manager.

        Args:
            storage: AsyncStorage instance for all storage operations
            model: Deprecated fallback model name (use llm_service instead)
            agent_id: Agent ID for scoped queries
            consolidator: MemoryConsolidator for episode access
            memory_retriever: MemoryRetriever for emotional memory access
            llm_service: LLMService for model identity and constitutional awareness
            context_builder: Pre-configured ContextBuilder to use (dependency
                injection).  Production callers (KestrelAgent) pass their own
                instance, which has bootstrap files (SOUL.md, AGENTS.md, …)
                loaded into it — that's how the agent's identity flows into
                the system prompt.  When None, a minimal ContextBuilder is
                built here; that fallback is for isolated test contexts and
                will NOT carry bootstrap content, so any production path
                that omits this argument is wrong.
        """
        self.storage = storage
        self._llm_service = llm_service
        self._model_fallback = model
        self._counter = None
        self._counter_model = None
        self.agent_id = agent_id
        self.consolidator = consolidator
        self.memory_retriever = memory_retriever
        # Keep public reference for constitutional awareness (used by build_context)
        self.llm_service = llm_service

        # Initialize sub-components.  Prefer the injected ContextBuilder —
        # that's the one the agent pre-loaded SOUL.md into.  Constructing a
        # fresh one here would leave its BootstrapLoader empty, which is
        # exactly the bug that caused the agent to not know its own name
        # when asked in chat.
        if context_builder is not None:
            self.context_builder = context_builder
        else:
            self.context_builder = ContextBuilder(
                storage=storage,
                consolidator=consolidator,
                llm_service=llm_service,
                model=model,
            )

        # Initialize specialized managers
        self.conversation_manager = ConversationManager(storage, agent_id)
        self.memory_manager = MemoryManager(storage, agent_id, consolidator, memory_retriever)
        self.tool_context_manager = ToolContextManager(
            storage=storage,
            model=self.model,
            agent_id=agent_id,
            llm_service=llm_service,
        )

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

    async def build_context(
        self,
        query: str,
        constitution: str,
        include_briefing: bool = True,
        include_memories: bool = True,
        include_rag: bool = True,
        privacy_mode: str = "NORMAL",
        emotional_context: Optional[Dict[str, Any]] = None,
        conversation_history: Optional[List[Dict]] = None,
        reflection_guidance: Optional[List[str]] = None,
        system_prompt_addendum: Optional[str] = None,
    ) -> ContextResult:
        """
        Build complete context for an LLM request.

        This is the main entry point for context assembly. It:
        1. Creates an adaptive token budget based on conversation length
        2. Builds the system prompt with constitution
        3. Retrieves and formats conversation history
        4. Adds episode summaries for long conversations
        5. Retrieves emotionally-relevant memories
        6. Searches RAG for relevant documents
        7. Ensures everything fits within token limits

        Args:
            query: The current user query
            constitution: Constitution text to include
            include_briefing: Include session briefing in system prompt
            include_memories: Include emotionally-weighted memories
            include_rag: Include RAG document search
            privacy_mode: Current privacy mode (affects what's retrieved)
            emotional_context: Current emotional state for mood-congruent recall
            conversation_history: Pre-fetched conversation history (e.g., session-filtered).
                                  If None, fetches full history from storage.

        Returns:
            ContextResult with assembled context and metadata
        """
        warnings: List[str] = []

        # Handle EPHEMERAL mode - no retrieval
        if privacy_mode == "EPHEMERAL":
            return await self._build_ephemeral_context(
                query=query,
                constitution=constitution,
                include_briefing=include_briefing,
                system_prompt_addendum=system_prompt_addendum,
            )

        # Use provided history or fetch from storage
        if conversation_history is not None:
            history = conversation_history
        else:
            history = await self.conversation_manager.get_conversation_history()
        message_count = len(history)

        # Create adaptive budget
        budget = create_budget(self.model, message_count, adaptive=True)

        # Get constitutional awareness (state of mind includes prompt adaptation)
        prompt_adaptation = None
        state_of_mind = None
        if self.llm_service and hasattr(self.llm_service, 'get_state_of_mind'):
            try:
                state_of_mind = self.llm_service.get_state_of_mind()
                prompt_adaptation = state_of_mind.prompt_adaptation
            except Exception as e:
                logger.warning(f"Failed to get constitutional state of mind: {e}")

        # 1. Build system prompt
        system_prompt = self.context_builder.build_system_prompt(
            constitution=constitution,
            include_briefing=include_briefing,
            prompt_adaptation=prompt_adaptation,
            state_of_mind=state_of_mind,
            system_prompt_addendum=system_prompt_addendum,
        )
        system_tokens = self.counter.count(system_prompt)
        budget.use("system", system_tokens)

        # Track what we include
        episode_count = 0
        memory_count = 0
        rag_chunks = 0
        # Per-turn retrieved context (memories + RAG). Kept OUT of system_prompt
        # so the system prefix stays stable across turns and prompt caches hit.
        dynamic_blocks: List[str] = []

        # 1b. Inject reflection guidance into system prompt
        if reflection_guidance:
            guidance_text = "\n--- ACTIVE REFLECTION GUIDANCE ---\n"
            guidance_text += "Based on self-reflection, keep these insights in mind:\n"
            for item in reflection_guidance:
                guidance_text += f"- {item}\n"
            guidance_text += "--- END GUIDANCE ---"
            guidance_tokens = self.counter.count(guidance_text)
            budget.use("system", guidance_tokens)
            system_prompt = f"{system_prompt}\n\n{guidance_text}"
            logger.info(f"Injected {len(reflection_guidance)} reflection guidance items into prompt")

        # 1c. Microcompact: clear stale tool results (zero-cost, no LLM)
        microcompact_savings = self._microcompact_tool_results(history)
        if microcompact_savings > 0:
            logger.info(f"Microcompact cleared {microcompact_savings} stale tool results")

        # 2. Add episodes for long conversations
        if message_count >= self.EPISODE_THRESHOLD_MESSAGES and self.consolidator:
            episode_context = await self.context_builder.get_episode_context(
                max_tokens=budget.episodes,
                max_episodes=5
            )
            if episode_context:
                episode_tokens = self.counter.count(episode_context)
                budget.use("episodes", episode_tokens)
                system_prompt = f"{system_prompt}\n\n{episode_context}"
                # Count episodes (rough estimate from formatting)
                episode_count = episode_context.count("**") // 2
                logger.debug(f"Added {episode_count} episodes to context")

        # 3. Retrieve emotionally-weighted memories (placed in dynamic user
        # context, not system, so the system prefix stays cacheable).
        if include_memories and self.memory_retriever:
            try:
                memories = await self.memory_manager.retrieve_memories(
                    query=query,
                    max_tokens=budget.memories,
                    counter=self.counter,
                    emotional_context=emotional_context
                )
                if memories:
                    memory_tokens = self.counter.count(memories)
                    if budget.can_fit("memories", memory_tokens):
                        budget.use("memories", memory_tokens)
                        dynamic_blocks.append(f"<memories>\n{memories}\n</memories>")
                        memory_count = memories.count("[Memory")
                        logger.debug(f"Added {memory_count} memories to dynamic context")
            except Exception as e:
                logger.warning(f"Memory retrieval failed: {e}")
                warnings.append(f"Memory retrieval unavailable: {e}")

        # 4. Retrieve RAG context (placed in dynamic user context, not system).
        if include_rag:
            try:
                rag_context = await self.context_builder.retrieve_context(query)
                if rag_context:
                    rag_tokens = self.counter.count(rag_context)
                    if budget.can_fit("rag", rag_tokens):
                        budget.use("rag", rag_tokens)
                        dynamic_blocks.append(
                            "<documents>\n"
                            f"{rag_context}\n"
                            "</documents>"
                        )
                        rag_chunks = rag_context.count("[Document") or rag_context.count("Source:")
                        logger.debug(f"Added {rag_chunks} RAG chunks to dynamic context")
            except Exception as e:
                logger.warning(f"RAG retrieval failed: {e}")
                warnings.append(f"Document search unavailable: {e}")

        # Assemble the per-turn retrieved-context block. Empty string when
        # nothing was retrieved — caller can use this as-is in a format()
        # without producing dangling wrapper tags.
        if dynamic_blocks:
            dynamic_user_context = (
                "<retrieved_context>\n"
                + "\n".join(dynamic_blocks)
                + "\n</retrieved_context>"
            )
        else:
            dynamic_user_context = ""

        # 5. Format conversation history with remaining budget
        formatted_history = self.context_builder.format_conversation_history(
            history=history,
            max_tokens=budget.history
        )
        history_tokens = self.counter.count_messages(formatted_history)
        budget.use("history", history_tokens, items=len(formatted_history))

        # Check if we had to truncate significantly
        if len(formatted_history) < len(history) * 0.5:
            warnings.append(
                f"History truncated: {len(formatted_history)}/{len(history)} messages"
            )

        # Pre-send budget enforcement: if total exceeds budget, drop oldest history
        if budget.total_used > budget.total_budget and len(formatted_history) > 1:
            overage = budget.total_used - budget.total_budget
            logger.warning(
                f"Context budget exceeded by {overage} tokens — auto-pruning oldest history"
            )
            pruned_tokens = 0
            while formatted_history and pruned_tokens < overage:
                dropped = formatted_history.pop(0)  # Drop oldest
                dropped_tokens = self.counter.count(dropped.get("content", "")) + 4
                pruned_tokens += dropped_tokens
            # Update budget tracking
            new_history_tokens = self.counter.count_messages(formatted_history)
            alloc = budget.allocations["history"]
            alloc.used = new_history_tokens
            alloc.items = len(formatted_history)
            warnings.append(
                f"Auto-pruned {pruned_tokens} tokens from history to fit budget"
            )

        logger.info(
            f"Context built: {budget.total_used}/{budget.total_budget} tokens "
            f"({len(formatted_history)} msgs, {episode_count} episodes, "
            f"{memory_count} memories, {rag_chunks} docs)"
        )

        return ContextResult(
            system_prompt=system_prompt,
            messages=formatted_history,
            total_tokens=budget.total_used,
            budget_summary=budget.get_summary(),
            episode_count=episode_count,
            memory_count=memory_count,
            rag_chunks=rag_chunks,
            warnings=warnings,
            dynamic_user_context=dynamic_user_context,
        )

    async def _build_ephemeral_context(
        self,
        query: str,
        constitution: str,
        include_briefing: bool,
        system_prompt_addendum: Optional[str] = None,
    ) -> ContextResult:
        """
        Build minimal context for EPHEMERAL privacy mode.

        In EPHEMERAL mode, no history or memories are retrieved.
        Only the system prompt and constitution are included.
        """
        # Get constitutional awareness (state of mind includes prompt adaptation)
        prompt_adaptation = None
        state_of_mind = None
        if self.llm_service and hasattr(self.llm_service, 'get_state_of_mind'):
            try:
                state_of_mind = self.llm_service.get_state_of_mind()
                prompt_adaptation = state_of_mind.prompt_adaptation
            except Exception as e:
                logger.warning(f"Failed to get constitutional state of mind: {e}")

        system_prompt = self.context_builder.build_system_prompt(
            constitution=constitution,
            include_briefing=include_briefing,
            prompt_adaptation=prompt_adaptation,
            state_of_mind=state_of_mind,
            system_prompt_addendum=system_prompt_addendum,
        )

        # Add ephemeral mode notice
        system_prompt = (
            f"{system_prompt}\n\n"
            "--- EPHEMERAL MODE ACTIVE ---\n"
            "This conversation is not being recorded. "
            "No history or memories are available.\n"
            "--- END NOTICE ---"
        )

        tokens = self.counter.count(system_prompt)

        return ContextResult(
            system_prompt=system_prompt,
            messages=[],
            total_tokens=tokens,
            budget_summary={"mode": "ephemeral"},
            warnings=["EPHEMERAL mode: no history available"],
        )

    # Delegate to ConversationManager
    async def compress_session(self, llm_service, preserve_recent: int = 10, force: bool = False) -> Dict[str, Any]:
        """Delegate to ConversationManager."""
        return await self.conversation_manager.compress_session(
            llm_service, self.counter, preserve_recent, force
        )

    async def check_compression_needed(self, utilization_threshold: float = 70.0) -> Dict[str, Any]:
        """Delegate to ConversationManager."""
        return await self.conversation_manager.check_compression_needed(
            self.counter, self.model, utilization_threshold
        )

    async def get_messages_for_selection(self, mode: str, criteria: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Delegate to ConversationManager."""
        return await self.conversation_manager.get_messages_for_selection(mode, criteria, limit)

    async def mark_messages(self, message_ids: List[int], action: str, reason: str = "") -> Dict[str, Any]:
        """Delegate to ConversationManager."""
        return await self.conversation_manager.mark_messages(message_ids, action, reason)

    async def exclude_messages(self, message_ids: List[int], reason: str) -> Dict[str, Any]:
        """Delegate to ConversationManager."""
        return await self.conversation_manager.exclude_messages(message_ids, reason)

    async def restore_messages(self, message_ids: Optional[List[int]] = None) -> Dict[str, Any]:
        """Delegate to ConversationManager."""
        return await self.conversation_manager.restore_messages(message_ids)

    async def summarize_messages(self, llm_service, message_ids: List[int], preserve_key_facts: bool = True) -> Dict[str, Any]:
        """Delegate to ConversationManager."""
        return await self.conversation_manager.summarize_messages(
            llm_service, self.counter, message_ids, preserve_key_facts
        )

    # Delegate to MemoryManager
    async def check_episode_needed(self, session_messages: int = 0) -> bool:
        """Delegate to MemoryManager."""
        return await self.memory_manager.check_episode_needed(session_messages)

    async def create_episode_if_needed(self, session_messages: int = 0, force: bool = False) -> Optional[Any]:
        """Delegate to MemoryManager."""
        return await self.memory_manager.create_episode_if_needed(session_messages, force)

    async def stash_messages(self, message_ids: Optional[List[int]] = None, name: Optional[str] = None, last_n: Optional[int] = None) -> Dict[str, Any]:
        """Delegate to MemoryManager."""
        return await self.memory_manager.stash_messages(message_ids, name, last_n)

    async def stash_pop(self, stash_id: Optional[str] = None) -> Dict[str, Any]:
        """Delegate to MemoryManager."""
        return await self.memory_manager.stash_pop(stash_id)

    async def stash_apply(self, stash_id: Optional[str] = None) -> Dict[str, Any]:
        """Delegate to MemoryManager."""
        return await self.memory_manager.stash_apply(stash_id)

    async def stash_list(self) -> Dict[str, Any]:
        """Delegate to MemoryManager."""
        return await self.memory_manager.stash_list()

    async def stash_drop(self, stash_id: Optional[str] = None) -> Dict[str, Any]:
        """Delegate to MemoryManager."""
        return await self.memory_manager.stash_drop(stash_id)

    async def stash_save(self, stash_id: Optional[str] = None, name: Optional[str] = None, summary: Optional[str] = None, tags: Optional[List[str]] = None) -> Dict[str, Any]:
        """Delegate to MemoryManager."""
        return await self.memory_manager.stash_save(stash_id, name, summary, tags)

    async def stash_peek(self, stash_id: Optional[str] = None, max_chars: int = 5000) -> Dict[str, Any]:
        """Delegate to MemoryManager."""
        return await self.memory_manager.stash_peek(stash_id, max_chars)

    async def hierarchical_compress(self, llm_service, chunk_size: int = 4000, preserve_recent: int = 5, max_depth: int = 3) -> Dict[str, Any]:
        """Delegate to MemoryManager."""
        return await self.memory_manager.hierarchical_compress(
            llm_service, self.counter, chunk_size, preserve_recent, max_depth
        )

    # Delegate to ToolContextManager
    async def get_status(self, history: Optional[list] = None, context_stats=None) -> Dict[str, Any]:
        """Delegate to ToolContextManager. Pass session-filtered history for accurate reporting."""
        return await self.tool_context_manager.get_status(self.counter, history=history, context_stats=context_stats)

    def get_budget_status(self, message_count: int = 0) -> Dict[str, Any]:
        """Delegate to ToolContextManager."""
        return self.tool_context_manager.get_budget_status(message_count)

    # Helper methods exposed for testing
    def _build_message_chunks(self, messages: List[Dict], chunk_size: int) -> List[str]:
        """Delegate to MemoryManager."""
        return self.memory_manager._build_message_chunks(messages, chunk_size)

    # --- Microcompact: zero-cost tool result clearing ---

    MICROCOMPACT_KEEP_RECENT = int(os.environ.get("KESTREL_MICROCOMPACT_KEEP_RECENT", "5"))

    def _microcompact_tool_results(self, history: List[Dict]) -> int:
        """
        Clear stale tool result content from conversation history.

        Replaces old tool result content with JSON markers while preserving
        tool_call_id pairing (required by LLM APIs). The most recent N
        tool results are kept intact.

        This runs BEFORE format_conversation_history() normalizes roles,
        so role="tool" is still identifiable.

        Args:
            history: Conversation history (mutated in place)

        Returns:
            Number of tool results cleared
        """
        keep_recent = self.MICROCOMPACT_KEEP_RECENT
        if keep_recent < 1:
            keep_recent = 1

        # Collect indices of tool result messages (preserving order)
        tool_indices = []
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

            marker = json.dumps({
                "cleared": True,
                "tool_name": tool_name,
                "summary": summary,
                "cleared_at": now,
            })
            msg["content"] = marker
            cleared += 1

        return cleared
