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

import logging
from dataclasses import dataclass, field
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
        manager = ContextManager(storage, model="gpt-4")
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
        model: str = "gpt-4",
        agent_id: Optional[str] = None,
        consolidator: Optional["MemoryConsolidator"] = None,
        memory_retriever: Optional["MemoryRetriever"] = None,
        llm_service: Optional["LLMService"] = None,
    ):
        """
        Initialize the context manager.

        Args:
            storage: AsyncStorage instance for all storage operations
            model: Model name for token counting/limits
            agent_id: Agent ID for scoped queries
            consolidator: MemoryConsolidator for episode access
            memory_retriever: MemoryRetriever for emotional memory access
            llm_service: Optional LLM service for constitutional awareness
        """
        self.storage = storage
        self.model = model
        self.agent_id = agent_id
        self.consolidator = consolidator
        self.memory_retriever = memory_retriever
        self.llm_service = llm_service

        # Initialize sub-components
        self.counter = get_token_counter(model)
        self.context_builder = ContextBuilder(
            storage=storage,
            model=model,
            consolidator=consolidator
        )

        # Initialize specialized managers
        self.conversation_manager = ConversationManager(storage, agent_id)
        self.memory_manager = MemoryManager(storage, agent_id, consolidator, memory_retriever)
        self.tool_context_manager = ToolContextManager(storage, model, agent_id)

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
                include_briefing=include_briefing
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
            state_of_mind=state_of_mind
        )
        system_tokens = self.counter.count(system_prompt)
        budget.use("system", system_tokens)

        # Track what we include
        episode_count = 0
        memory_count = 0
        rag_chunks = 0

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

        # 3. Add emotionally-weighted memories
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
                        system_prompt = f"{system_prompt}\n\n{memories}"
                        memory_count = memories.count("[Memory")
                        logger.debug(f"Added {memory_count} memories to context")
            except Exception as e:
                logger.warning(f"Memory retrieval failed: {e}")
                warnings.append(f"Memory retrieval unavailable: {e}")

        # 4. Add RAG context
        if include_rag:
            try:
                rag_context = await self.context_builder.retrieve_context(query)
                if rag_context:
                    rag_tokens = self.counter.count(rag_context)
                    if budget.can_fit("rag", rag_tokens):
                        budget.use("rag", rag_tokens)
                        system_prompt = (
                            f"{system_prompt}\n\n"
                            f"--- RELEVANT DOCUMENTS (from indexed files, not current conversation) ---\n"
                            f"{rag_context}\n"
                            f"--- END DOCUMENTS ---"
                        )
                        rag_chunks = rag_context.count("[Document") or rag_context.count("Source:")
                        logger.debug(f"Added {rag_chunks} RAG chunks to context")
            except Exception as e:
                logger.warning(f"RAG retrieval failed: {e}")
                warnings.append(f"Document search unavailable: {e}")

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
        )

    async def _build_ephemeral_context(
        self,
        query: str,
        constitution: str,
        include_briefing: bool
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
            state_of_mind=state_of_mind
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
    async def get_status(self) -> Dict[str, Any]:
        """Delegate to ToolContextManager."""
        return await self.tool_context_manager.get_status(self.counter)

    def get_budget_status(self, message_count: int = 0) -> Dict[str, Any]:
        """Delegate to ToolContextManager."""
        return self.tool_context_manager.get_budget_status(message_count)

    # Helper methods exposed for testing
    def _build_message_chunks(self, messages: List[Dict], chunk_size: int) -> List[str]:
        """Delegate to MemoryManager."""
        return self.memory_manager._build_message_chunks(messages, chunk_size)