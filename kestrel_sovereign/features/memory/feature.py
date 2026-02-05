"""
Memory access tools for the Kestrel agent.

Allows the agent to search, recall, and inspect their conversation history,
RAG documents, and consolidated memory episodes. Tools are available for both
LLM function calling and explicit !commands.

Integrates with the human-like memory system:
- Emotional congruence (mood-matching retrieval)
- Importance weighting (life events stick)
- Recency with decay (Ebbinghaus forgetting curve)
- Access frequency (rehearsal effect)
"""

import logging
from typing import Any, Dict, List, Optional

from kestrel_sovereign.features.base import Feature, tool
from kestrel_sovereign.tools.base import ToolCategory

logger = logging.getLogger(__name__)


class MemoryFeature(Feature):
    """
    Memory access tools for the agent.

    Provides tools for:
    - Searching conversation history
    - Retrieving recent messages
    - Searching RAG documents (files, knowledge)
    - Getting consolidated memory episodes
    - Checking memory system status

    EPHEMERAL mode: Nothing is stored, so queries return empty results.
    """

    @property
    def tool_description(self) -> str:
        return "ALWAYS USE when asked 'do you remember', 'what did we discuss', 'recall', or any question about past conversations. Searches and retrieves conversation history with full decryption support."

    async def initialize(self):
        """Initialize the memory feature with storage references."""
        self.storage = self.agent.storage
        self.consolidator = getattr(self.agent, 'memory_consolidator', None)
        # Memory retriever will be lazily loaded since memory_system is initialized after features
        self._memory_retriever = None
        # Get agent_id through storage hierarchy
        self.agent_id = (
            getattr(self.storage, 'agent_id', '') or
            getattr(getattr(self.storage, '_storage', None), 'agent_id', '')
        )
        logger.info(f"MemoryFeature initialized for agent: {self.agent_id[:30]}...")

    @property
    def memory_retriever(self):
        """Lazy-load memory retriever from memory_system (initialized after features)."""
        if self._memory_retriever is None:
            memory_system = getattr(self.agent, 'memory_system', None)
            if memory_system and hasattr(memory_system, 'retriever'):
                self._memory_retriever = memory_system.retriever
        return self._memory_retriever

    def _get_conversation_store(self):
        """Navigate storage hierarchy to get the conversation store."""
        return (
            getattr(self.storage, 'conversation', None) or
            getattr(getattr(self.storage, '_storage', None), 'conversation', None)
        )

    @tool(
        name="search_memory",
        description="Search conversation history for matching content. Decrypts and searches all messages client-side for reliable results.",
        category=ToolCategory.MEMORY,
        command_prefix="!memory search"
    )
    async def search_memory(self, query: str, limit: int = 10) -> Dict[str, Any]:
        """
        Search conversation history for matching content.

        Uses client-side decryption and search to work reliably with encrypted storage.

        Args:
            query: Search term or phrase to find in past conversations
            limit: Maximum number of results to return (default 10)
        """
        try:
            # Get full decrypted history and search client-side
            # This works with encryption because we decrypt before searching
            all_history = await self.storage.get_conversation_history(limit=5000)

            # Search through decrypted content
            query_lower = query.lower()
            matches = []
            for msg in all_history:
                content = msg.get("content", "")
                if query_lower in content.lower():
                    matches.append(msg)
                    if len(matches) >= limit:
                        break

            return {
                "success": True,
                "results": matches,
                "count": len(matches),
                "query": query,
                "total_searched": len(all_history)
            }
        except Exception as e:
            logger.error(f"search_memory failed: {e}")
            return {"success": False, "error": str(e)}

    @tool(
        name="recall_recent",
        description="Get my most recent conversation messages. Use this to recall what we just discussed or to provide context about our recent interactions.",
        category=ToolCategory.MEMORY,
        command_prefix="!memory recent"
    )
    async def recall_recent(self, limit: int = 20) -> Dict[str, Any]:
        """
        Get recent conversation history.

        Args:
            limit: Number of recent messages to retrieve (default 20)
        """
        try:
            history = await self.storage.get_conversation_history(limit=limit)
            return {
                "success": True,
                "messages": history,
                "count": len(history)
            }
        except Exception as e:
            logger.error(f"recall_recent failed: {e}")
            return {"success": False, "error": str(e)}

    @tool(
        name="search_documents",
        description="Search my knowledge base and RAG documents for relevant information. Use this when I need to find information from files, documents, or other stored knowledge.",
        category=ToolCategory.MEMORY,
        command_prefix="!memory docs"
    )
    async def search_documents(self, query: str, limit: int = 5) -> Dict[str, Any]:
        """
        Search RAG document chunks using hybrid semantic + keyword search.

        Args:
            query: Search query for finding relevant documents
            limit: Maximum number of document chunks to return (default 5)
        """
        try:
            results = await self.storage.search_chunks(query, limit)
            # Format results for readability
            formatted = []
            for res in results:
                formatted.append({
                    "source": res.get("document_name") or res.get("file_hash", "unknown"),
                    "content": res.get("content", "")[:500],  # Truncate for preview
                    "score": res.get("score", 0),
                    "full_content": res.get("content", "")
                })
            return {
                "success": True,
                "results": formatted,
                "count": len(formatted),
                "query": query
            }
        except Exception as e:
            logger.error(f"search_documents failed: {e}")
            return {"success": False, "error": str(e)}

    @tool(
        name="search_case_law",
        description="Search past audit decisions and constitutional interpretations. Use this when I need precedent for ethical or governance decisions.",
        category=ToolCategory.MEMORY,
        command_prefix="!memory cases"
    )
    async def search_case_law(self, query: str, limit: int = 3) -> Dict[str, Any]:
        """
        Search past audit decisions for precedent.

        Args:
            query: Query describing the ethical/governance situation
            limit: Maximum number of cases to return (default 3)
        """
        try:
            # Use the case law search if available
            if hasattr(self.storage, 'search_case_law'):
                results = await self.storage.search_case_law(query, limit)
                return {
                    "success": True,
                    "cases": results,
                    "count": len(results),
                    "query": query
                }
            else:
                return {
                    "success": False,
                    "error": "Case law search not available",
                    "note": "This feature requires audit history to be enabled"
                }
        except Exception as e:
            logger.error(f"search_case_law failed: {e}")
            return {"success": False, "error": str(e)}

    @tool(
        name="get_episodes",
        description="Get consolidated memory episodes - narrative summaries of past conversation themes. Use this for high-level recall of what we've discussed over time.",
        category=ToolCategory.MEMORY,
        command_prefix="!memory episodes"
    )
    async def get_episodes(self, limit: int = 10) -> Dict[str, Any]:
        """
        Get memory episodes from consolidation.

        Args:
            limit: Maximum episodes to return (default 10)
        """
        if not self.consolidator:
            return {
                "success": False,
                "error": "Memory consolidator not available",
                "note": "Episodes are created during memory consolidation cycles"
            }
        try:
            episodes = await self.consolidator.get_recent_episodes_for_context(
                max_episodes=limit
            )
            return {
                "success": True,
                "episodes": episodes,
                "count": len(episodes)
            }
        except Exception as e:
            logger.error(f"get_episodes failed: {e}")
            return {"success": False, "error": str(e)}

    @tool(
        name="memory_status",
        description="Check memory system health and statistics. Use this to understand my memory capabilities and current state.",
        category=ToolCategory.SYSTEM,
        command_prefix="!memory status"
    )
    async def memory_status(self) -> Dict[str, Any]:
        """
        Get memory system status and statistics.

        Returns:
            Dict with 'success': True and status info on success.
            Dict with 'success': False, 'error': str on failure.
        """
        try:
            # Get conversation count
            history = await self.storage.get_conversation_history(limit=10000)
            total_messages = len(history)

            # Get conversation store encryption status
            conv_store = self._get_conversation_store()
            encryption_enabled = getattr(conv_store, 'encryption_enabled', False) if conv_store else False

            # Get RAG stats if available
            rag_stats = {}
            if hasattr(self.storage, 'rag'):
                rag = self.storage.rag
                # Try to get chunk count
                try:
                    count_result = await self.storage.db.fetchone(
                        "SELECT COUNT(*) FROM document_chunks"
                    )
                    rag_stats["document_chunks"] = count_result[0] if count_result else 0
                except Exception:
                    rag_stats["document_chunks"] = "unknown"

            # Get file count
            file_count = 0
            try:
                file_result = await self.storage.db.fetchone(
                    "SELECT COUNT(*) FROM files WHERE agent_id = ?",
                    (self.agent_id,)
                )
                file_count = file_result[0] if file_result else 0
            except Exception:
                pass

            return {
                "success": True,
                "total_messages": total_messages,
                "files_stored": file_count,
                "encryption_enabled": encryption_enabled,
                "agent_id": (self.agent_id[:30] + "...") if len(self.agent_id) > 30 else self.agent_id,
                "consolidator_available": self.consolidator is not None,
                "rag": rag_stats
            }
        except Exception as e:
            logger.error(f"memory_status failed: {e}")
            return {"success": False, "error": str(e)}

    @tool(
        name="recall_emotional",
        description="Recall memories with human-like weighting (importance, emotion, recency). Use alongside full_history_search for emotionally-aware recall. Scores memories like a human would - important moments and emotionally-charged memories surface first.",
        category=ToolCategory.MEMORY,
        command_prefix="!memory recall"
    )
    async def recall_emotional(
        self,
        query: str,
        mood: str = "neutral",
        limit: int = 10
    ) -> Dict[str, Any]:
        """
        Retrieve memories with human-like weighting.

        Uses the MemoryRetriever which scores memories on:
        - Semantic relevance (30%)
        - Emotional congruence (25%) - matches current mood
        - Importance (20%) - life events, personal disclosures
        - Recency (15%) - with Ebbinghaus decay curve
        - Access frequency (10%) - rehearsal strengthens memory

        Args:
            query: What you're trying to remember
            mood: Current emotional context (positive, negative, neutral)
            limit: Maximum memories to retrieve (default 10)
        """
        if not self.memory_retriever:
            return {
                "success": False,
                "error": "Memory retriever not available",
                "note": "Falling back to basic search",
                "fallback": await self.search_memory(query, limit)
            }

        try:
            # Import here to avoid circular dependency
            from kestrel_sovereign.storage.memory_models import MemoryMetadata

            # Create emotional context based on mood
            mood_valence = {
                "positive": 0.6,
                "negative": -0.6,
                "neutral": 0.0
            }.get(mood.lower(), 0.0)

            emotional_context = MemoryMetadata(
                emotional_valence=mood_valence,
                emotional_intensity=0.5 if mood != "neutral" else 0.0
            )

            # Retrieve with human-like weighting
            memories = await self.memory_retriever.retrieve(
                query=query,
                agent_id=self.agent_id,
                emotional_context=emotional_context,
                limit=limit
            )

            # Format for readability
            formatted = []
            for mem in memories:
                meta = mem.get("metadata", {})
                formatted.append({
                    "content": mem.get("content", ""),
                    "role": mem.get("role", "unknown"),
                    "score": mem.get("score", 0),
                    "emotional_valence": meta.get("emotional_valence", 0),
                    "importance": meta.get("importance", 0.5),
                    "timestamp": mem.get("timestamp", "")
                })

            return {
                "success": True,
                "memories": formatted,
                "count": len(formatted),
                "query": query,
                "mood_context": mood,
                "note": "Retrieved using human-like memory weighting"
            }
        except Exception as e:
            logger.error(f"recall_emotional failed: {e}")
            return {"success": False, "error": str(e)}

    @tool(
        name="full_history_search",
        description="PRIMARY TOOL for recalling past conversations. Use this when asked 'do you remember', 'what did we discuss', or any question about past conversations. Decrypts and searches all conversation history reliably.",
        category=ToolCategory.MEMORY,
        command_prefix="!memory fullsearch"
    )
    async def full_history_search(self, query: str, limit: int = 20) -> Dict[str, Any]:
        """
        Get full history and search client-side (works with encryption).

        This is slower but works with encrypted content because it retrieves
        and decrypts all messages before searching.

        Args:
            query: Search term or phrase to find
            limit: Maximum results to return (default 20)
        """
        try:
            # Get full decrypted history
            all_history = await self.storage.get_conversation_history(limit=5000)

            # Search through decrypted content
            query_lower = query.lower()
            matches = []
            for msg in all_history:
                content = msg.get("content", "")
                if query_lower in content.lower():
                    matches.append(msg)
                    if len(matches) >= limit:
                        break

            return {
                "success": True,
                "results": matches,
                "count": len(matches),
                "query": query,
                "total_searched": len(all_history)
            }
        except Exception as e:
            logger.error(f"full_history_search failed: {e}")
            return {"success": False, "error": str(e)}

    @tool(
        name="delete_messages",
        description="Delete conversation messages matching a pattern. Use for cleaning up test data or removing unwanted messages. Requires Sovereign authorization.",
        category=ToolCategory.MEMORY,
        command_prefix="!memory delete"
    )
    async def delete_messages(self, pattern: str, confirm: bool = False) -> Dict[str, Any]:
        """
        Delete messages matching a content pattern.

        Args:
            pattern: Text pattern to match (case-insensitive)
            confirm: Must be True to actually delete (safety check)
        """
        try:
            conv_store = self._get_conversation_store()
            if not conv_store:
                return {"success": False, "error": "Conversation store not available"}

            if not confirm:
                # Preview mode - show what would be deleted
                history = await conv_store.get_full_history_with_ids(include_excluded=True, include_stashed=True)
                pattern_lower = pattern.lower()
                matches = [
                    {"id": msg["id"], "role": msg["role"], "preview": msg.get("content", "")[:100]}
                    for msg in history
                    if pattern_lower in msg.get("content", "").lower()
                ]
                return {
                    "success": True,
                    "mode": "preview",
                    "would_delete": len(matches),
                    "matches": matches[:20],  # Limit preview
                    "note": "Set confirm=True to actually delete these messages"
                }

            # Actually delete
            deleted = await conv_store.delete_messages_matching(pattern)
            return {
                "success": True,
                "deleted": deleted,
                "pattern": pattern
            }
        except Exception as e:
            logger.error(f"delete_messages failed: {e}")
            return {"success": False, "error": str(e)}
