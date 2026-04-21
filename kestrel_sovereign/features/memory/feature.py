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
from kestrel_sovereign.features.storage_access import (
    resolve_feature_conversation_store,
    resolve_feature_database,
)
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
        self._db = resolve_feature_database(self.agent)
        # MemorySystem is the single facade for all memory components.
        # Lazily loaded since it's initialized on the agent after feature registration.
        self._memory_system = None
        # Agent identity (DID is the canonical source of truth)
        self.agent_id = self.agent.did
        logger.info(f"MemoryFeature initialized for agent: {self.agent_id[:30]}...")

    @property
    def memory_system(self):
        """Lazy-load MemorySystem facade from agent (initialized after features)."""
        if self._memory_system is None:
            self._memory_system = getattr(self.agent, 'memory_system', None)
        return self._memory_system

    @property
    def consolidator(self):
        """Access consolidator through MemorySystem facade."""
        ms = self.memory_system
        return ms.consolidator if ms else None

    @property
    def memory_retriever(self):
        """Access retriever through MemorySystem facade."""
        ms = self.memory_system
        return ms.retriever if ms else None

    def _get_conversation_store(self):
        """Navigate storage hierarchy to get the conversation store."""
        return resolve_feature_conversation_store(self.agent)

    @tool(
        name="search_memory",
        description="PRIMARY TOOL for recalling past conversations. Use this when asked 'do you remember', 'what did we discuss', or any question about past conversations. Decrypts and searches conversation history client-side for reliable results. Pass session_id to scope to a single conversation thread.",
        category=ToolCategory.MEMORY,
        command_prefix="!memory search"
    )
    async def search_memory(
        self,
        query: str,
        limit: int = 20,
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Search conversation history for matching content.

        Uses the conversation store's encryption-aware search_history,
        which decrypts client-side before matching. Optionally scope to
        a single session.

        Args:
            query: Search term or phrase to find in past conversations
            limit: Maximum number of results to return (default 20)
            session_id: If provided, only search messages tagged with this
                session_id. Useful for "what did we discuss in this
                conversation" queries.
        """
        try:
            conv_store = self._get_conversation_store()
            if not conv_store:
                return {"success": False, "error": "Conversation store unavailable"}

            results = await conv_store.search_history(
                query=query,
                limit=limit,
                session_id=session_id,
            )

            return {
                "success": True,
                "results": results,
                "count": len(results),
                "query": query,
                "session_id": session_id,
            }
        except (AttributeError, TypeError, KeyError) as e:
            logger.error(f"search_memory failed: {e}")
            return {"success": False, "error": str(e)}
        except Exception as e:
            logger.error(f"search_memory failed: {e}", exc_info=True)
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
        except (AttributeError, TypeError) as e:
            logger.error(f"recall_recent failed: {e}")
            return {"success": False, "error": str(e)}
        except Exception as e:
            logger.error(f"recall_recent failed: {e}", exc_info=True)
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
        except (AttributeError, TypeError, KeyError) as e:
            logger.error(f"search_documents failed: {e}")
            return {"success": False, "error": str(e)}
        except Exception as e:
            logger.error(f"search_documents failed: {e}", exc_info=True)
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
        except (AttributeError, TypeError, KeyError) as e:
            logger.error(f"search_case_law failed: {e}")
            return {"success": False, "error": str(e)}
        except Exception as e:
            logger.error(f"search_case_law failed: {e}", exc_info=True)
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
        except (AttributeError, TypeError) as e:
            logger.error(f"get_episodes failed: {e}")
            return {"success": False, "error": str(e)}
        except Exception as e:
            logger.error(f"get_episodes failed: {e}", exc_info=True)
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
                    count_result = await self._db.fetchone(
                        "SELECT COUNT(*) FROM document_chunks"
                    )
                    rag_stats["document_chunks"] = count_result[0] if count_result else 0
                except Exception:
                    rag_stats["document_chunks"] = "unknown"

            # Get file count
            file_count = 0
            try:
                file_result = await self._db.fetchone(
                    "SELECT COUNT(*) FROM files WHERE agent_id = ?",
                    (self.agent_id,)
                )
                file_count = file_result[0] if file_result else 0
            except Exception:
                pass

            # Include MemorySystem summary if available
            memory_system_info = {}
            if self.memory_system:
                memory_system_info = self.memory_system.get_summary()

            return {
                "success": True,
                "total_messages": total_messages,
                "files_stored": file_count,
                "encryption_enabled": encryption_enabled,
                "agent_id": (self.agent_id[:30] + "...") if len(self.agent_id) > 30 else self.agent_id,
                "consolidator_available": self.consolidator is not None,
                "retriever_available": self.memory_retriever is not None,
                "memory_system": memory_system_info,
                "rag": rag_stats
            }
        except (AttributeError, TypeError, KeyError) as e:
            logger.error(f"memory_status failed: {e}")
            return {"success": False, "error": str(e)}
        except Exception as e:
            logger.error(f"memory_status failed: {e}", exc_info=True)
            return {"success": False, "error": str(e)}

    @tool(
        name="recall_emotional",
        description="Recall memories with human-like weighting (importance, emotion, recency). Use alongside search_memory for emotionally-aware recall. Scores memories like a human would - important moments and emotionally-charged memories surface first.",
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
        except (AttributeError, TypeError, KeyError, ValueError) as e:
            logger.error(f"recall_emotional failed: {e}")
            return {"success": False, "error": str(e)}
        except Exception as e:
            logger.error(f"recall_emotional failed: {e}", exc_info=True)
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
        except (AttributeError, TypeError, KeyError) as e:
            logger.error(f"delete_messages failed: {e}")
            return {"success": False, "error": str(e)}
        except Exception as e:
            logger.error(f"delete_messages failed: {e}", exc_info=True)
            return {"success": False, "error": str(e)}

    @tool(
        name="memory_consolidate",
        description="Consolidate recent messages into narrative episodes, detect temporal patterns, and archive decayed memories. Runs the cognitive memory pipeline that turns raw conversation into structured long-term memory. Safe to schedule periodically (e.g. nightly).",
        category=ToolCategory.MEMORY,
        command_prefix="!memory consolidate"
    )
    async def memory_consolidate(self) -> Dict[str, Any]:
        """
        Run the memory consolidation pipeline.

        Invokes MemoryConsolidator.run_consolidation() which:
        - Creates narrative episodes from recent message clusters
        - Detects temporal behavioral patterns
        - Archives memories whose decay strength has fallen below threshold

        This is the missing automatic invocation — without this tool being
        scheduled, memory_episodes table stays empty and the cognitive
        memory layer never compounds beyond the raw conversation history.

        Returns:
            Dict with episodes_created, patterns_found, messages_archived counts
        """
        try:
            memory_system = getattr(self.agent, "memory_system", None)
            if not memory_system:
                return {"success": False, "error": "MemorySystem not available on agent"}

            # Go through the MemorySystem facade so None-safety is consistent
            # with all other consolidate callers (sleep cycle, etc.)
            result = await memory_system.consolidate()
            if "error" in result:
                return {"success": False, **result}
            return {"success": True, **result}
        except Exception as e:
            logger.error(f"memory_consolidate failed: {e}", exc_info=True)
            return {"success": False, "error": str(e)}

    # ------------------------------------------------------------------
    # Schema-aware recall tools (#628)
    # These read from the typed stores populated by SchemaRouter during
    # message enrichment. Empty results in EPHEMERAL/ISOLATED modes where
    # routing never ran.
    # ------------------------------------------------------------------

    @tool(
        name="recall_action_items",
        description="Retrieve action items the user committed to. Filters: status, creation-date window (days), assignee.",
        category=ToolCategory.MEMORY,
        command_prefix="!memory actions",
    )
    async def recall_action_items(
        self,
        status: Optional[str] = None,
        days: Optional[int] = None,
        assignee_concept_id: Optional[str] = None,
        limit: int = 25,
    ) -> Dict[str, Any]:
        """Query action_item graph nodes with optional property filters.

        Action items live as graph nodes of type `action_item` (no
        separate SQL table). The graph's `node_type` index makes the
        initial scan fast; remaining filters run in memory over
        at-most a few thousand nodes per agent.

        Args:
            status: Filter by status ("pending", "done", "cancelled").
                If None, all statuses are returned.
            days: If set, only return items whose `created_at` is within
                the last N days. Note: this is a creation-date window,
                not a due-date window — use update_action_item to set
                a due_date explicitly if you need that.
            assignee_concept_id: Optional person concept id to filter by.
            limit: Max rows returned (1-200, default 25).
        """
        storage = getattr(self.agent, "storage", None)
        if storage is None or not hasattr(storage, "graph"):
            return {"success": False, "error": "Graph store not available"}

        try:
            limit = int(limit)
        except (TypeError, ValueError):
            return {"success": False, "error": f"limit must be an integer, got {limit!r}"}
        if limit < 1 or limit > 200:
            return {"success": False, "error": "limit must be in [1, 200]"}

        if status is not None and status not in ("pending", "done", "cancelled"):
            return {"success": False, "error": f"status must be pending/done/cancelled, got {status!r}"}

        since: Optional[str] = None
        if days is not None:
            try:
                days = int(days)
            except (TypeError, ValueError):
                return {"success": False, "error": f"days must be an integer, got {days!r}"}
            if days < 1 or days > 3650:
                return {"success": False, "error": "days must be in [1, 3650]"}
            from datetime import datetime, timezone, timedelta
            since = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()

        filters: dict = {"agent_id": self.agent_id}
        if status is not None:
            filters["status"] = status
        if assignee_concept_id is not None:
            filters["assignee_concept_id"] = assignee_concept_id

        try:
            nodes = await storage.graph.query_nodes_by_type_and_property(
                "action_item",
                filters=filters,
                created_since=since,
                order_by_created=True,
                limit=limit,
            )
        except Exception as e:
            logger.error("recall_action_items query failed: %s", e)
            return {"success": False, "error": str(e)}

        matching = [
            {
                "id": n.node_id,
                "source_message_id": (n.properties or {}).get("source_message_id"),
                "text": (n.properties or {}).get("text"),
                "status": (n.properties or {}).get("status"),
                "assignee_concept_id": (n.properties or {}).get("assignee_concept_id"),
                "due_date": (n.properties or {}).get("due_date"),
                "confidence": (n.properties or {}).get("confidence"),
                "created_at": (n.properties or {}).get("created_at"),
            }
            for n in nodes
        ]
        return {"action_items": matching, "count": len(matching)}

    @tool(
        name="update_action_item",
        description="Update an action item's status (pending/done/cancelled), due date, or assignee.",
        category=ToolCategory.MEMORY,
        command_prefix="!memory action update",
    )
    async def update_action_item(
        self,
        item_id: str,
        status: Optional[str] = None,
        due_date: Optional[str] = None,
        assignee_concept_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Update a single action item graph node. Null fields are preserved."""
        storage = getattr(self.agent, "storage", None)
        if storage is None or not hasattr(storage, "graph"):
            return {"success": False, "error": "Graph store not available"}

        if status is not None and status not in ("pending", "done", "cancelled"):
            return {"success": False, "error": f"status must be pending/done/cancelled, got {status!r}"}

        if due_date is not None:
            # Accept ISO-8601 date or datetime strings.
            from datetime import datetime as _dt
            try:
                _dt.fromisoformat(due_date)
            except (TypeError, ValueError):
                return {
                    "success": False,
                    "error": f"due_date must be ISO-8601 (YYYY-MM-DD or full datetime), got {due_date!r}",
                }

        if status is None and due_date is None and assignee_concept_id is None:
            return {"success": False, "error": "no fields to update"}

        try:
            node = await storage.graph.get_node(item_id)
        except Exception as e:
            logger.error("update_action_item lookup failed: %s", e)
            return {"success": False, "error": str(e)}
        if node is None or node.node_type != "action_item":
            return {"success": False, "error": f"Action item {item_id} not found"}

        props = dict(node.properties or {})
        if props.get("agent_id") != self.agent_id:
            # Don't leak cross-agent mutations.
            return {"success": False, "error": f"Action item {item_id} not found"}

        if status is not None:
            props["status"] = status
        if due_date is not None:
            props["due_date"] = due_date
        if assignee_concept_id is not None:
            props["assignee_concept_id"] = assignee_concept_id
        props["updated_at"] = _utc_now_iso()

        from kestrel_sovereign.storage.async_graph_store import GraphNode
        try:
            await storage.graph.add_node(GraphNode(
                node_id=node.node_id,
                node_type=node.node_type,
                label=node.label,
                properties=props,
            ))
        except Exception as e:
            logger.error("update_action_item write failed: %s", e)
            return {"success": False, "error": str(e)}
        return {"success": True, "item_id": item_id}

    @tool(
        name="recall_decisions",
        description="Retrieve decisions the user has recorded (stored as graph nodes of type 'decision').",
        category=ToolCategory.MEMORY,
        command_prefix="!memory decisions",
    )
    async def recall_decisions(self, limit: int = 25) -> Dict[str, Any]:
        """List decisions from the graph."""
        storage = getattr(self.agent, "storage", None)
        if storage is None or not hasattr(storage, "graph"):
            return {"success": False, "error": "Graph store not available"}

        try:
            limit = int(limit)
        except (TypeError, ValueError):
            return {"success": False, "error": f"limit must be an integer, got {limit!r}"}
        if limit < 1 or limit > 200:
            return {"success": False, "error": "limit must be in [1, 200]"}

        try:
            nodes = await storage.graph.query_nodes_by_type_and_property(
                "decision",
                filters={"agent_id": self.agent_id},
                order_by_created=True,
                limit=limit,
            )
        except Exception as e:
            logger.error("recall_decisions failed: %s", e)
            return {"success": False, "error": str(e)}

        own = [
            {
                "id": n.node_id,
                "label": n.label,
                "text": (n.properties or {}).get("text"),
                "source_message_id": (n.properties or {}).get("source_message_id"),
                "confidence": (n.properties or {}).get("confidence"),
                "created_at": (n.properties or {}).get("created_at"),
            }
            for n in nodes
        ]
        return {"decisions": own, "count": len(own)}

    @tool(
        name="recall_interactions",
        description="List recent message→person interactions for a given person concept, with sentiment and topics.",
        category=ToolCategory.MEMORY,
        command_prefix="!memory interactions",
    )
    async def recall_interactions(
        self,
        person_concept_id: str,
        limit: int = 25,
    ) -> Dict[str, Any]:
        """Return recent mentions edges pointing at a person concept."""
        storage = getattr(self.agent, "storage", None)
        if storage is None or not hasattr(storage, "graph"):
            return {"success": False, "error": "Graph store not available"}

        try:
            limit = int(limit)
        except (TypeError, ValueError):
            return {"success": False, "error": f"limit must be an integer, got {limit!r}"}
        if limit < 1 or limit > 200:
            return {"success": False, "error": "limit must be in [1, 200]"}

        try:
            edges = await storage.graph.get_edges(person_concept_id, direction="in")
        except Exception as e:
            logger.error("recall_interactions failed: %s", e)
            return {"success": False, "error": str(e)}

        # Message nodes are namespaced by agent id: `message:{agent_id}:{msg}`.
        # Without this guard, a caller passing a shared concept id could
        # retrieve edges from other agents' messages.
        agent_prefix = f"message:{self.agent_id}:"
        interactions = [
            {
                "message_node_id": e.source_id,
                "properties": e.properties or {},
            }
            for e in edges
            if e.label == "mentions" and e.source_id.startswith(agent_prefix)
        ]
        interactions.sort(
            key=lambda i: (i.get("properties") or {}).get("recorded_at") or "",
            reverse=True,
        )
        return {"interactions": interactions[:limit], "count": min(len(interactions), limit)}

    @tool(
        name="confirm_person_match",
        description="Resolve an ambiguous person mention by confirming which existing concept it refers to.",
        category=ToolCategory.MEMORY,
        command_prefix="!memory confirm-person",
    )
    async def confirm_person_match(
        self,
        message_id: str,
        mentioned_label: str,
        concept_id: str,
    ) -> Dict[str, Any]:
        """Resolve an ambiguous person mention.

        Removes the ambiguous label-based mentions edge
        (message → concept:{agent}:{mentioned_label}) and writes a canonical
        mentions edge pointing at the confirmed concept id. After this runs,
        recall_interactions on the confirmed concept includes the message;
        recall on the ambiguous label-concept does not.
        """
        storage = getattr(self.agent, "storage", None)
        if storage is None or not hasattr(storage, "graph"):
            return {"success": False, "error": "Graph store not available"}

        target = await storage.graph.get_node(concept_id)
        if target is None:
            return {"success": False, "error": f"Concept {concept_id} not found"}

        message_node = f"message:{self.agent_id}:{message_id}"
        # The ambiguous edge was written by SchemaRouter using the same
        # deterministic id shape the linker uses. Normalize the label the
        # same way the linker/router did (lowercased, stripped) so we can
        # find and delete it.
        ambiguous_target = (
            f"concept:{self.agent_id}:{mentioned_label.strip().lower()}"
        )

        try:
            # Write the canonical edge first so we never have a window
            # where neither edge exists.
            await storage.graph.add_edge(
                message_node,
                concept_id,
                "mentions",
                properties={
                    "resolved_from": mentioned_label,
                    "confirmed": True,
                    "confirmed_at": _utc_now_iso(),
                },
            )
            # Then remove the ambiguous one — but only if it's different
            # from the canonical target. Otherwise we would be deleting
            # the edge we just wrote.
            removed = False
            if ambiguous_target != concept_id:
                await storage.graph.delete_edge(
                    message_node, ambiguous_target, "mentions"
                )
                removed = True
        except Exception as e:
            logger.error("confirm_person_match failed: %s", e)
            return {"success": False, "error": str(e)}
        return {
            "success": True,
            "message_id": message_id,
            "resolved_to": concept_id,
            "ambiguous_edge_removed": removed,
        }


def _utc_now_iso() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()
