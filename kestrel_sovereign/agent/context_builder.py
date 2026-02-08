"""
Context Builder for Kestrel Agent.

This module handles the assembly of context for LLM prompts, including:
- RAG document retrieval
- Conversation history formatting
- Constitutional grounding
- Session briefings
- Token-aware truncation
- Episode integration for long conversations
"""

import logging
from pathlib import Path
from typing import List, Dict, Optional, Any, TYPE_CHECKING

from .token_counter import TokenCounter, get_token_counter
from .token_budget import TokenBudget, AdaptiveTokenBudget, create_budget

if TYPE_CHECKING:
    from storage import AsyncStorage
    from storage.memory_consolidator import MemoryConsolidator

logger = logging.getLogger(__name__)


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
        model: str = "gpt-4",
        consolidator: Optional["MemoryConsolidator"] = None,
        agent_data_path: Optional[str] = None
    ):
        """
        Initialize the context builder.

        Args:
            storage: The async storage instance for RAG and history retrieval
            model: The model name for token counting (default: gpt-4)
            consolidator: Optional MemoryConsolidator for episode retrieval
            agent_data_path: Path to agent data directory (for SOUL.md, etc.)
        """
        self.storage = storage
        self.model = model
        self.counter = get_token_counter(model)
        self.consolidator = consolidator
        self.agent_data_path = Path(agent_data_path) if agent_data_path else None
        self._soul_content: Optional[str] = None
        
        # Load SOUL.md if it exists
        self._load_soul_md()

    def _load_soul_md(self) -> None:
        """Load or reload SOUL.md from the agent data directory."""
        if self.agent_data_path:
            soul_path = self.agent_data_path / "SOUL.md"
            if soul_path.exists():
                try:
                    self._soul_content = soul_path.read_text()
                    logger.info(f"Loaded SOUL.md from {soul_path}")
                except Exception as e:
                    logger.warning(f"Failed to load SOUL.md: {e}")
            else:
                self._soul_content = None
                logger.debug(f"No SOUL.md found at {soul_path}")

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

    def format_conversation_history(
        self,
        history: List[Dict],
        max_messages: int = 20,
        max_tokens: Optional[int] = None,
        max_chars: int = 50000,  # Fallback if no token counting
    ) -> List[Dict[str, str]]:
        """
        Format conversation history for LLM context.

        Applies token-based limits to prevent context overflow while preserving
        the most recent and relevant messages.

        Args:
            history: Raw conversation history from storage
            max_messages: Maximum number of messages to include
            max_tokens: Maximum tokens allowed (preferred over max_chars)
            max_chars: Maximum characters (fallback if no token counting)

        Returns:
            List of formatted message dicts with 'role' and 'content' keys
        """
        # Take the most recent messages first
        recent = history[-max_messages:] if len(history) > max_messages else history

        # Build from NEWEST to OLDEST to preserve most recent context
        # Then reverse to get chronological order
        formatted = []
        total_tokens = 0
        MESSAGE_OVERHEAD = 4  # Tokens per message for structure

        # Iterate in reverse (newest first) to prioritize recent messages
        for msg in reversed(recent):
            role = msg.get('role', 'user')
            content = msg.get('content', '')

            # Count tokens for this message
            msg_tokens = self.counter.count(content) + MESSAGE_OVERHEAD

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

        # SOUL.md comes first — it defines personality and overrides tone
        if self._soul_content:
            parts.append("--- YOUR IDENTITY ---")
            parts.append(self._soul_content)
            parts.append("--- END IDENTITY ---")

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
        if self._soul_content:
            parts.append("\n--- STYLE REMINDER (IMPORTANT) ---")
            parts.append("When answering personal questions, respond naturally in paragraphs. DO NOT use numbered lists or bullet points. Talk like a person, not a document.")
            parts.append("--- END REMINDER ---")
        
        if additional_context:
            parts.append("\n--- ADDITIONAL CONTEXT ---")
            parts.append(additional_context)
            parts.append("--- END CONTEXT ---")
        
        return "\n\n".join(parts)

    def build_rag_context(
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
            results = self.storage.search_chunks(query)[:max_results]
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
