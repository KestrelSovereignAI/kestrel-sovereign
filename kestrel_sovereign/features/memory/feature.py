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

@tool methods return ``kestrel_sdk.tools.result.ToolResult`` per the
kestrel-sovereign #1042 narration-honesty contract (see #1061).
"""

import logging
from typing import Any, Dict, List, Optional

from kestrel_sovereign.agent.context_builder import extract_raw_user_content
from kestrel_sovereign.features.base import Feature, tool
from kestrel_sovereign.features.storage_access import (
    resolve_feature_conversation_store,
    resolve_feature_database,
)
from kestrel_sovereign.features.memory.reflection_hook import ReflectionSleepHook
from kestrel_sdk.tools.base import ToolCategory
from kestrel_sdk.tools.result import ToolResult

logger = logging.getLogger(__name__)


def _strip_sent_form_for_recall(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Strip the sent-form template from user-role rows before returning
    them to the LLM via a memory-recall tool result.

    User-turn rows are persisted in their fully-rendered prompt form
    (``<retrieved_context>...</retrieved_context>\\n<user_input>...</user_input>``)
    so the conversation-history loader can replay byte-exact bytes for
    Anthropic prompt-cache stability. That's the right shape for prompt
    replay, but the wrong shape for memory recall: when an agent asks
    ``search_memory("what did we discuss")``, the model receives the
    retrieved-context block as the ``user`` content and treats the
    previous turn's retrieved memories as if the user had spoken them.

    Real-world symptom: April 28 cluster of "Based on the retrieved
    context, I can tell you that..." memories — the model paraphrasing
    its own prior retrieval block back at itself.

    Fix is local to the recall path: strip the wrappers only for
    ``role == "user"`` rows. Assistant rows are persisted as raw text
    and pass through unchanged. ``extract_raw_user_content`` is
    idempotent on legacy/raw rows.
    """
    out = []
    for msg in messages:
        if msg.get("role") == "user" and isinstance(msg.get("content"), str):
            cleaned = dict(msg)
            cleaned["content"] = extract_raw_user_content(msg["content"])
            out.append(cleaned)
        else:
            out.append(msg)
    return out


def _coerce_int(value: Any, name: str, *, lo: int, hi: int) -> tuple[Optional[int], Optional[str]]:
    """Coerce a tool arg to an int in [lo, hi].

    Returns ``(value, None)`` on success or ``(None, error_string)`` so
    callers can return a ToolResult.failed without re-formatting. Used
    for the schema-aware recall tools where malformed JSON args from
    the LLM (string ``"25"``, float ``25.0``) are common.
    """
    try:
        coerced = int(value)
    except (TypeError, ValueError):
        return None, f"{name} must be an integer, got {value!r}"
    if coerced < lo or coerced > hi:
        return None, f"{name} must be in [{lo}, {hi}], got {coerced}"
    return coerced, None


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

    async def post_all_features_loaded(self, agent):
        """Subscribe memory application attestation to the sleep cycle."""
        hooks = getattr(agent, "sleep_hooks", None)
        if hooks is None:
            agent.sleep_hooks = []
            hooks = agent.sleep_hooks
        if not any(isinstance(hook, ReflectionSleepHook) for hook in hooks):
            hooks.append(ReflectionSleepHook())

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

    # ------------------------------------------------------------------
    # Internal helper used by `recall_emotional` fallback. Keeping it
    # separate from the @tool wrapper means the fallback consumer
    # doesn't have to unpack a ToolResult envelope.
    # ------------------------------------------------------------------

    async def _search_memory_impl(
        self,
        query: str,
        limit: int,
        session_id: Optional[str],
    ) -> Dict[str, Any]:
        """Run the conversation-history search and return a dict.

        Returns ``{"ok": True, "results": [...], "count": N, ...}`` or
        ``{"ok": False, "error": str}``.
        """
        try:
            conv_store = self._get_conversation_store()
            if not conv_store:
                return {"ok": False, "error": "Conversation store unavailable"}

            results = await conv_store.search_history(
                query=query,
                limit=limit,
                session_id=session_id,
            )
            return {
                "ok": True,
                "results": _strip_sent_form_for_recall(results),
                "count": len(results),
                "query": query,
                "session_id": session_id,
            }
        except (AttributeError, TypeError, KeyError) as e:
            logger.error(f"_search_memory_impl failed: {e}")
            return {"ok": False, "error": str(e)}
        except Exception as e:
            logger.error(f"_search_memory_impl failed: {e}", exc_info=True)
            return {"ok": False, "error": str(e)}

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
    ) -> ToolResult:
        """
        Search conversation history for matching content.

        Uses the conversation store's encryption-aware search_history,
        which decrypts client-side before matching. Optionally scope to
        a single session.

        Args:
            query: Search term or phrase to find in past conversations
            limit: Maximum number of results to return (the request — actual
                   count returned may be lower if fewer matches exist).
            session_id: If provided, only search messages tagged with this
                session_id. Useful for "what did we discuss in this
                conversation" queries.
        """
        try:
            limit_val = int(limit)
        except (TypeError, ValueError):
            return ToolResult.failed(f"limit must be an integer, got {limit!r}")
        if limit_val < 1:
            return ToolResult.failed("limit must be >= 1")

        data = await self._search_memory_impl(query, limit_val, session_id)
        if not data["ok"]:
            return ToolResult.failed(data["error"])

        scope = f" in session {session_id}" if session_id else ""
        return ToolResult.ok(
            confirmation=(
                f"Found {data['count']} match(es) for {query!r}{scope} "
                f"(limit requested: {limit_val})"
            ),
            data={
                "results": data["results"],
                "count": data["count"],
                "query": data["query"],
                "session_id": data["session_id"],
                "limit_requested": limit_val,
            },
        )

    @tool(
        name="recall_recent",
        description="Get my most recent conversation messages. Use this to recall what we just discussed or to provide context about our recent interactions.",
        category=ToolCategory.MEMORY,
        command_prefix="!memory recent"
    )
    async def recall_recent(self, limit: int = 20) -> ToolResult:
        """
        Get recent conversation history.

        Args:
            limit: Number of recent messages to retrieve (the request —
                   actual count returned may be lower if fewer messages exist).
        """
        try:
            limit_val = int(limit)
        except (TypeError, ValueError):
            return ToolResult.failed(f"limit must be an integer, got {limit!r}")
        if limit_val < 1:
            return ToolResult.failed("limit must be >= 1")

        try:
            history = await self.storage.get_conversation_history(limit=limit_val)
        except (AttributeError, TypeError) as e:
            logger.error(f"recall_recent failed: {e}")
            return ToolResult.failed(str(e))
        except Exception as e:
            logger.error(f"recall_recent failed: {e}", exc_info=True)
            return ToolResult.failed(str(e))

        cleaned = _strip_sent_form_for_recall(history)
        return ToolResult.ok(
            confirmation=(
                f"Retrieved {len(cleaned)} recent message(s) "
                f"(limit requested: {limit_val})"
            ),
            data={
                "messages": cleaned,
                "count": len(cleaned),
                "limit_requested": limit_val,
            },
        )

    @tool(
        name="search_documents",
        description="Search my knowledge base and RAG documents for relevant information. Use this when I need to find information from files, documents, or other stored knowledge.",
        category=ToolCategory.MEMORY,
        command_prefix="!memory docs"
    )
    async def search_documents(self, query: str, limit: int = 5) -> ToolResult:
        """
        Search RAG document chunks using hybrid semantic + keyword search.

        Args:
            query: Search query for finding relevant documents
            limit: Maximum number of document chunks to return (the request).
        """
        try:
            limit_val = int(limit)
        except (TypeError, ValueError):
            return ToolResult.failed(f"limit must be an integer, got {limit!r}")
        if limit_val < 1:
            return ToolResult.failed("limit must be >= 1")

        try:
            results = await self.storage.search_chunks(query, limit_val)
        except (AttributeError, TypeError, KeyError) as e:
            logger.error(f"search_documents failed: {e}")
            return ToolResult.failed(str(e))
        except Exception as e:
            logger.error(f"search_documents failed: {e}", exc_info=True)
            return ToolResult.failed(str(e))

        formatted = []
        for res in results:
            formatted.append({
                "source": res.get("document_name") or res.get("file_hash", "unknown"),
                "content": res.get("content", "")[:500],
                "score": res.get("score", 0),
                "full_content": res.get("content", ""),
            })
        return ToolResult.ok(
            confirmation=(
                f"Found {len(formatted)} document chunk(s) for {query!r} "
                f"(limit requested: {limit_val})"
            ),
            data={
                "results": formatted,
                "count": len(formatted),
                "query": query,
                "limit_requested": limit_val,
            },
        )

    @tool(
        name="search_case_law",
        description="Search past audit decisions and constitutional interpretations. Use this when I need precedent for ethical or governance decisions.",
        category=ToolCategory.MEMORY,
        command_prefix="!memory cases"
    )
    async def search_case_law(self, query: str, limit: int = 3) -> ToolResult:
        """
        Search past audit decisions for precedent.

        Args:
            query: Query describing the ethical/governance situation
            limit: Maximum number of cases to return (the request).
        """
        try:
            limit_val = int(limit)
        except (TypeError, ValueError):
            return ToolResult.failed(f"limit must be an integer, got {limit!r}")
        if limit_val < 1:
            return ToolResult.failed("limit must be >= 1")

        if not hasattr(self.storage, 'search_case_law'):
            return ToolResult.failed(
                "Case law search not available",
                data={"reason": "audit history is not enabled on this storage"},
            )

        try:
            results = await self.storage.search_case_law(query, limit_val)
        except (AttributeError, TypeError, KeyError) as e:
            logger.error(f"search_case_law failed: {e}")
            return ToolResult.failed(str(e))
        except Exception as e:
            logger.error(f"search_case_law failed: {e}", exc_info=True)
            return ToolResult.failed(str(e))

        return ToolResult.ok(
            confirmation=(
                f"Found {len(results)} case(s) for {query!r} "
                f"(limit requested: {limit_val})"
            ),
            data={
                "cases": results,
                "count": len(results),
                "query": query,
                "limit_requested": limit_val,
            },
        )

    @tool(
        name="get_episodes",
        description="Get consolidated memory episodes - narrative summaries of past conversation themes. Use this for high-level recall of what we've discussed over time. Pass `query` to recall episodes RELEVANT to a topic (semantic search, can surface older episodes); omit it for the most recent episodes.",
        category=ToolCategory.MEMORY,
        command_prefix="!memory episodes"
    )
    async def get_episodes(
        self, limit: int = 10, query: Optional[str] = None
    ) -> ToolResult:
        """
        Get memory episodes from consolidation.

        Args:
            limit: Maximum episodes to return (the request).
            query: Optional topic to recall episodes by RELEVANCE (semantic
                recall via the shared vector backend; #1674 P2). Surfaced
                episodes are marked as accessed so consulted memories resist
                the forgetting deletion tier. When omitted, returns the most
                recent episodes (unchanged legacy behavior).
        """
        if not self.consolidator:
            return ToolResult.failed(
                "Memory consolidator not available",
                data={"reason": "episodes are created during memory consolidation cycles"},
            )

        try:
            limit_val = int(limit)
        except (TypeError, ValueError):
            return ToolResult.failed(f"limit must be an integer, got {limit!r}")
        if limit_val < 1:
            return ToolResult.failed("limit must be >= 1")

        query_str = (query or "").strip()
        try:
            if query_str:
                # Relevance recall — returns MemoryEpisode objects and bumps
                # their access_count. Format to the same dict shape the
                # recency path emits so callers see one contract.
                found = await self.consolidator.search_episodes(
                    query_str, limit=limit_val
                )
                episodes = [
                    {
                        "title": ep.title,
                        "summary": ep.summary,
                        "emotional_arc": ep.emotional_arc,
                        "timespan": (
                            ep.timespan_start.strftime("%Y-%m-%d")
                            if ep.timespan_start else "unknown"
                        ),
                    }
                    for ep in found
                ]
            else:
                episodes = await self.consolidator.get_recent_episodes_for_context(
                    max_episodes=limit_val
                )
        except (AttributeError, TypeError) as e:
            logger.error(f"get_episodes failed: {e}")
            return ToolResult.failed(str(e))
        except Exception as e:
            logger.error(f"get_episodes failed: {e}", exc_info=True)
            return ToolResult.failed(str(e))

        mode = "relevance" if query_str else "recency"
        return ToolResult.ok(
            confirmation=(
                f"Retrieved {len(episodes)} episode(s) by {mode} "
                f"(limit requested: {limit_val})"
            ),
            data={
                "episodes": episodes,
                "count": len(episodes),
                "limit_requested": limit_val,
                "mode": mode,
            },
        )

    @tool(
        name="memory_status",
        description="Check memory system health and statistics. Use this to understand my memory capabilities and current state.",
        category=ToolCategory.SYSTEM,
        command_prefix="!memory status"
    )
    async def memory_status(self) -> ToolResult:
        """Get memory system status and statistics."""
        try:
            history = await self.storage.get_conversation_history(limit=10000)
            total_messages = len(history)

            conv_store = self._get_conversation_store()
            encryption_enabled = (
                getattr(conv_store, 'encryption_enabled', False) if conv_store else False
            )

            rag_stats: Dict[str, Any] = {}
            if hasattr(self.storage, 'rag'):
                try:
                    count_result = await self._db.fetchone(
                        "SELECT COUNT(*) FROM document_chunks"
                    )
                    rag_stats["document_chunks"] = count_result[0] if count_result else 0
                except Exception:
                    rag_stats["document_chunks"] = "unknown"

            file_count = 0
            try:
                file_result = await self._db.fetchone(
                    "SELECT COUNT(*) FROM files WHERE agent_id = ?",
                    (self.agent_id,)
                )
                file_count = file_result[0] if file_result else 0
            except Exception:
                pass

            memory_system_info: Dict[str, Any] = {}
            episode_count = 0
            if self.memory_system:
                memory_system_info = self.memory_system.get_summary()
            if self.consolidator:
                try:
                    episodes = await self.consolidator.get_episodes(limit=10000)
                    episode_count = len(episodes)
                except Exception:
                    pass
        except (AttributeError, TypeError, KeyError) as e:
            logger.error(f"memory_status failed: {e}")
            return ToolResult.failed(str(e))
        except Exception as e:
            logger.error(f"memory_status failed: {e}", exc_info=True)
            return ToolResult.failed(str(e))

        agent_id_short = (
            (self.agent_id[:30] + "...") if len(self.agent_id) > 30 else self.agent_id
        )
        return ToolResult.ok(
            confirmation=(
                f"Memory status: {total_messages} message(s), "
                f"{episode_count} episode(s), "
                f"{file_count} file(s), "
                f"encryption={'on' if encryption_enabled else 'off'}"
            ),
            data={
                "total_messages": total_messages,
                "episode_count": episode_count,
                "files_stored": file_count,
                "encryption_enabled": encryption_enabled,
                "agent_id": agent_id_short,
                "consolidator_available": self.consolidator is not None,
                "retriever_available": self.memory_retriever is not None,
                "memory_system": memory_system_info,
                "rag": rag_stats,
            },
        )

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
    ) -> ToolResult:
        """
        Retrieve memories with human-like weighting.

        Uses the MemoryRetriever which scores memories on:
        - Semantic relevance (25%)
        - Emotional congruence (20%) - matches current mood
        - Importance (20%) - life events, personal disclosures
        - Recency (15%) - with Ebbinghaus decay curve
        - Certainty (10%) - epistemic confidence weight
        - Access frequency (10%) - rehearsal strengthens memory

        Args:
            query: What you're trying to remember
            mood: Current emotional context (positive, negative, neutral)
            limit: Maximum memories to retrieve (the request).
        """
        try:
            limit_val = int(limit)
        except (TypeError, ValueError):
            return ToolResult.failed(f"limit must be an integer, got {limit!r}")
        if limit_val < 1:
            return ToolResult.failed("limit must be >= 1")

        if not self.memory_retriever:
            # Fallback path: surface as PARTIAL so the LLM cannot claim
            # the emotionally-weighted recall ran when the retriever was
            # unavailable and we degraded to keyword search.
            fallback = await self._search_memory_impl(query, limit_val, None)
            if not fallback["ok"]:
                return ToolResult.failed(
                    "Memory retriever not available; basic search also failed: "
                    f"{fallback['error']}",
                    data={"limit_requested": limit_val},
                )
            return ToolResult.partial(
                confirmation=(
                    f"Memory retriever unavailable; fell back to keyword "
                    f"search and found {fallback['count']} match(es)"
                ),
                error=(
                    "emotional weighting (mood/importance/recency) was NOT "
                    "applied — results are basic keyword matches"
                ),
                data={
                    "fallback_results": fallback["results"],
                    "count": fallback["count"],
                    "query": query,
                    "limit_requested": limit_val,
                },
            )

        try:
            from kestrel_sovereign.storage.memory_models import MemoryMetadata

            mood_valence = {
                "positive": 0.6,
                "negative": -0.6,
                "neutral": 0.0,
            }.get(mood.lower(), 0.0)

            emotional_context = MemoryMetadata(
                emotional_valence=mood_valence,
                emotional_intensity=0.5 if mood != "neutral" else 0.0,
            )

            memories = await self.memory_retriever.retrieve(
                query=query,
                agent_id=self.agent_id,
                emotional_context=emotional_context,
                limit=limit_val,
            )
        except (AttributeError, TypeError, KeyError, ValueError) as e:
            logger.error(f"recall_emotional failed: {e}")
            return ToolResult.failed(str(e))
        except Exception as e:
            logger.error(f"recall_emotional failed: {e}", exc_info=True)
            return ToolResult.failed(str(e))

        formatted = []
        for mem in memories:
            meta = mem.get("metadata", {})
            role = mem.get("role", "unknown")
            content = mem.get("content", "")
            if role == "user" and isinstance(content, str):
                content = extract_raw_user_content(content)
            formatted.append({
                "content": content,
                "role": role,
                "score": mem.get("score", 0),
                "emotional_valence": meta.get("emotional_valence", 0),
                "importance": meta.get("importance", 0.5),
                "timestamp": mem.get("timestamp", ""),
            })

        return ToolResult.ok(
            confirmation=(
                f"Retrieved {len(formatted)} memory(ies) with human-like "
                f"weighting (mood={mood}, limit requested: {limit_val})"
            ),
            data={
                "memories": formatted,
                "count": len(formatted),
                "query": query,
                "mood_context": mood,
                "limit_requested": limit_val,
            },
        )

    @tool(
        name="delete_messages",
        description="Delete conversation messages matching a pattern. Use for cleaning up test data or removing unwanted messages. Requires Sovereign authorization.",
        category=ToolCategory.MEMORY,
        command_prefix="!memory delete"
    )
    async def delete_messages(self, pattern: str, confirm: bool = False) -> ToolResult:
        """
        Delete messages matching a content pattern.

        Args:
            pattern: Text pattern to match (case-insensitive)
            confirm: Must be True to actually delete (safety check). The
                LLM occasionally passes truthy strings like "false"; we
                accept only the literal Python bool ``True`` for the
                destructive path.
        """
        # ``confirm`` is destructive — refuse anything that isn't the
        # actual ``True`` bool, including the truthy strings "true"/"false".
        if not isinstance(confirm, bool):
            return ToolResult.failed(
                "confirm must be a boolean (True/False), "
                f"got {type(confirm).__name__}={confirm!r}"
            )

        conv_store = self._get_conversation_store()
        if not conv_store:
            return ToolResult.failed("Conversation store not available")

        if not confirm:
            try:
                history = await conv_store.get_full_history_with_ids(
                    include_excluded=True, include_stashed=True
                )
            except (AttributeError, TypeError, KeyError) as e:
                logger.error(f"delete_messages preview failed: {e}")
                return ToolResult.failed(str(e))
            except Exception as e:
                logger.error(f"delete_messages preview failed: {e}", exc_info=True)
                return ToolResult.failed(str(e))

            pattern_lower = pattern.lower()
            matches = [
                {"id": msg["id"], "role": msg["role"], "preview": msg.get("content", "")[:100]}
                for msg in history
                if pattern_lower in msg.get("content", "").lower()
            ]
            return ToolResult.ok(
                confirmation=(
                    f"Preview only — {len(matches)} message(s) match pattern "
                    f"{pattern!r} (call with confirm=True to delete)"
                ),
                data={
                    "mode": "preview",
                    "would_delete": len(matches),
                    "matches": matches[:20],
                    "pattern": pattern,
                },
            )

        try:
            deleted = await conv_store.delete_messages_matching(pattern)
        except (AttributeError, TypeError, KeyError) as e:
            logger.error(f"delete_messages failed: {e}")
            return ToolResult.failed(str(e))
        except Exception as e:
            logger.error(f"delete_messages failed: {e}", exc_info=True)
            return ToolResult.failed(str(e))

        return ToolResult.ok(
            confirmation=f"Deleted {deleted} message(s) matching {pattern!r}",
            data={
                "mode": "delete",
                "deleted": deleted,
                "pattern": pattern,
            },
        )

    @tool(
        name="memory_consolidate",
        description="Consolidate recent messages into narrative episodes, detect temporal patterns, and archive decayed memories. Runs the cognitive memory pipeline that turns raw conversation into structured long-term memory. Safe to schedule periodically (e.g. nightly).",
        category=ToolCategory.MEMORY,
        command_prefix="!memory consolidate"
    )
    async def memory_consolidate(self) -> ToolResult:
        """
        Run the memory consolidation pipeline.

        Invokes MemoryConsolidator.run_consolidation() which:
        - Creates narrative episodes from recent message clusters
        - Detects temporal behavioral patterns
        - Archives memories whose decay strength has fallen below threshold
        """
        memory_system = getattr(self.agent, "memory_system", None)
        if not memory_system:
            return ToolResult.failed("MemorySystem not available on agent")

        from contextlib import AsyncExitStack
        from kestrel_sdk.signals import ResourceLock

        # Serialize against the scheduled cron consolidation, which holds
        # ResourceLock.MEMORY via the dispatcher. Without this, a manual
        # `!memory consolidate` racing the cron tick reads the same empty
        # covered-message-id set and both runs emit duplicate episodes for the
        # same span (round-3 finding). No-op when no dispatcher lock manager is
        # wired (standalone / tests).
        dispatcher = getattr(self.agent, "dispatcher", None)
        lock_manager = getattr(dispatcher, "_locks", None) if dispatcher is not None else None

        try:
            async with AsyncExitStack() as stack:
                if lock_manager is not None:
                    await stack.enter_async_context(
                        lock_manager.acquire([ResourceLock.MEMORY])
                    )
                # consolidate() is the single chokepoint — it runs consolidation
                # AND the forgetting deletion tier (#1674), so the tool and the
                # nightly sleep cycle forget identically. The MEMORY lock here
                # serializes a manual `!memory consolidate` against the cron.
                result = await memory_system.consolidate()
        except Exception as e:
            logger.error(f"memory_consolidate failed: {e}", exc_info=True)
            return ToolResult.failed(str(e))

        if isinstance(result, dict) and "error" in result:
            return ToolResult.failed(
                str(result["error"]),
                data={k: v for k, v in result.items() if k != "error"},
            )

        result = dict(result or {})
        episodes_deleted = result.get("episodes_deleted", 0)
        episodes_created = result.get("episodes_created", 0)
        patterns_found = result.get("patterns_found", 0)
        messages_archived = result.get("messages_archived", 0)
        forget_clause = (
            f", {episodes_deleted} episode(s) forgotten"
            if episodes_deleted else ""
        )
        return ToolResult.ok(
            confirmation=(
                f"Consolidation complete: {episodes_created} episode(s), "
                f"{patterns_found} pattern(s), "
                f"{messages_archived} message(s) archived{forget_clause}"
            ),
            data=result,
        )


    # ------------------------------------------------------------------
    # Schema-aware recall tools (#628)
    # These read from the typed stores populated by SchemaRouter during
    # message enrichment. Empty results in EPHEMERAL/ISOLATED modes where
    # routing never ran.
    # ------------------------------------------------------------------

    @tool(
        name="recall_action_items",
        description="Retrieve action items the user committed to. Filters: status, creation-date window (days), assignee. Superseded items are excluded by default; pass include_superseded=True to see them.",
        category=ToolCategory.MEMORY,
        command_prefix="!memory actions",
    )
    async def recall_action_items(
        self,
        status: Optional[str] = None,
        days: Optional[int] = None,
        assignee_concept_id: Optional[str] = None,
        limit: int = 25,
        include_superseded: bool = False,
    ) -> ToolResult:
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
            include_superseded: If False (default), excludes items that
                have been superseded by a newer claim.
        """
        if not isinstance(include_superseded, bool):
            return ToolResult.failed(
                "include_superseded must be a boolean, "
                f"got {type(include_superseded).__name__}={include_superseded!r}"
            )

        storage = getattr(self.agent, "storage", None)
        if storage is None or not hasattr(storage, "graph"):
            return ToolResult.failed("Graph store not available")

        limit_val, err = _coerce_int(limit, "limit", lo=1, hi=200)
        if err:
            return ToolResult.failed(err)

        if status is not None and status not in ("pending", "done", "cancelled"):
            return ToolResult.failed(
                f"status must be pending/done/cancelled, got {status!r}"
            )

        since: Optional[str] = None
        if days is not None:
            days_val, err = _coerce_int(days, "days", lo=1, hi=3650)
            if err:
                return ToolResult.failed(err)
            from datetime import datetime, timezone, timedelta
            since = (datetime.now(timezone.utc) - timedelta(days=days_val)).isoformat()

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
                limit=limit_val,
            )
        except Exception as e:
            logger.error("recall_action_items query failed: %s", e)
            return ToolResult.failed(str(e))

        matching = []
        for n in nodes:
            props = n.properties or {}
            if not include_superseded and props.get("superseded_by"):
                continue
            matching.append({
                "id": n.node_id,
                "source_message_id": props.get("source_message_id"),
                "text": props.get("text"),
                "status": props.get("status"),
                "assignee_concept_id": props.get("assignee_concept_id"),
                "due_date": props.get("due_date"),
                "confidence": props.get("confidence"),
                "created_at": props.get("created_at"),
                "claim_certainty": props.get("claim_certainty"),
                "claim_source": props.get("claim_source"),
                "temporal_validity": props.get("temporal_validity"),
                "superseded_by": props.get("superseded_by"),
            })

        filter_clause = ""
        if status:
            filter_clause += f" status={status}"
        if days is not None:
            filter_clause += f" within last {days_val}d"
        if assignee_concept_id:
            filter_clause += f" assignee={assignee_concept_id}"
        return ToolResult.ok(
            confirmation=(
                f"Retrieved {len(matching)} action item(s){filter_clause} "
                f"(limit requested: {limit_val})"
            ),
            data={
                "action_items": matching,
                "count": len(matching),
                "limit_requested": limit_val,
                "include_superseded": include_superseded,
            },
        )

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
    ) -> ToolResult:
        """Update a single action item graph node. Null fields are preserved."""
        storage = getattr(self.agent, "storage", None)
        if storage is None or not hasattr(storage, "graph"):
            return ToolResult.failed("Graph store not available")

        if status is not None and status not in ("pending", "done", "cancelled"):
            return ToolResult.failed(
                f"status must be pending/done/cancelled, got {status!r}"
            )

        if due_date is not None:
            from datetime import datetime as _dt
            try:
                _dt.fromisoformat(due_date)
            except (TypeError, ValueError):
                return ToolResult.failed(
                    "due_date must be ISO-8601 (YYYY-MM-DD or full datetime), "
                    f"got {due_date!r}"
                )

        if status is None and due_date is None and assignee_concept_id is None:
            return ToolResult.failed("no fields to update")

        try:
            node = await storage.graph.get_node(item_id)
        except Exception as e:
            logger.error("update_action_item lookup failed: %s", e)
            return ToolResult.failed(str(e))
        if node is None or node.node_type != "action_item":
            return ToolResult.failed(f"Action item {item_id} not found")

        props = dict(node.properties or {})
        if props.get("agent_id") != self.agent_id:
            # Don't leak cross-agent mutations.
            return ToolResult.failed(f"Action item {item_id} not found")

        updates: List[str] = []
        if status is not None:
            props["status"] = status
            updates.append(f"status={status}")
        if due_date is not None:
            props["due_date"] = due_date
            updates.append(f"due_date={due_date}")
        if assignee_concept_id is not None:
            props["assignee_concept_id"] = assignee_concept_id
            updates.append(f"assignee={assignee_concept_id}")
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
            return ToolResult.failed(str(e))

        return ToolResult.ok(
            confirmation=f"Updated action item {item_id}: {', '.join(updates)}",
            data={"item_id": item_id, "updates": updates},
        )

    @tool(
        name="recall_decisions",
        description="Retrieve decisions the user has recorded (stored as graph nodes of type 'decision'). Superseded decisions are excluded by default; pass include_superseded=True to see them.",
        category=ToolCategory.MEMORY,
        command_prefix="!memory decisions",
    )
    async def recall_decisions(
        self,
        limit: int = 25,
        include_superseded: bool = False,
    ) -> ToolResult:
        """List decisions from the graph."""
        if not isinstance(include_superseded, bool):
            return ToolResult.failed(
                "include_superseded must be a boolean, "
                f"got {type(include_superseded).__name__}={include_superseded!r}"
            )

        storage = getattr(self.agent, "storage", None)
        if storage is None or not hasattr(storage, "graph"):
            return ToolResult.failed("Graph store not available")

        limit_val, err = _coerce_int(limit, "limit", lo=1, hi=200)
        if err:
            return ToolResult.failed(err)

        try:
            nodes = await storage.graph.query_nodes_by_type_and_property(
                "decision",
                filters={"agent_id": self.agent_id},
                order_by_created=True,
                limit=limit_val,
            )
        except Exception as e:
            logger.error("recall_decisions failed: %s", e)
            return ToolResult.failed(str(e))

        own = []
        for n in nodes:
            props = n.properties or {}
            if not include_superseded and props.get("superseded_by"):
                continue
            own.append({
                "id": n.node_id,
                "label": n.label,
                "text": props.get("text"),
                "source_message_id": props.get("source_message_id"),
                "confidence": props.get("confidence"),
                "created_at": props.get("created_at"),
                "claim_certainty": props.get("claim_certainty"),
                "claim_source": props.get("claim_source"),
                "temporal_validity": props.get("temporal_validity"),
                "superseded_by": props.get("superseded_by"),
            })
        return ToolResult.ok(
            confirmation=(
                f"Retrieved {len(own)} decision(s) "
                f"(limit requested: {limit_val})"
            ),
            data={
                "decisions": own,
                "count": len(own),
                "limit_requested": limit_val,
                "include_superseded": include_superseded,
            },
        )

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
    ) -> ToolResult:
        """Return recent mentions edges pointing at a person concept."""
        storage = getattr(self.agent, "storage", None)
        if storage is None or not hasattr(storage, "graph"):
            return ToolResult.failed("Graph store not available")

        limit_val, err = _coerce_int(limit, "limit", lo=1, hi=200)
        if err:
            return ToolResult.failed(err)

        try:
            edges = await storage.graph.get_edges(person_concept_id, direction="in")
        except Exception as e:
            logger.error("recall_interactions failed: %s", e)
            return ToolResult.failed(str(e))

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
        truncated = interactions[:limit_val]
        return ToolResult.ok(
            confirmation=(
                f"Retrieved {len(truncated)} interaction(s) for "
                f"{person_concept_id} (limit requested: {limit_val})"
            ),
            data={
                "interactions": truncated,
                "count": len(truncated),
                "person_concept_id": person_concept_id,
                "limit_requested": limit_val,
            },
        )

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
    ) -> ToolResult:
        """Resolve an ambiguous person mention."""
        storage = getattr(self.agent, "storage", None)
        if storage is None or not hasattr(storage, "graph"):
            return ToolResult.failed("Graph store not available")

        target = await storage.graph.get_node(concept_id)
        if target is None:
            return ToolResult.failed(f"Concept {concept_id} not found")

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
        except Exception as e:
            logger.error("confirm_person_match canonical edge write failed: %s", e)
            return ToolResult.failed(str(e))

        # Honesty: AsyncGraphStore.delete_edge() is a SQL DELETE that
        # returns no affected-row count and does not raise when the
        # edge isn't there. We can't actually verify the ambiguous
        # edge existed and was removed — the call is best-effort. So
        # we phrase the field and confirmation as "remove attempted"
        # rather than "removed" (round 5 codex finding).
        attempted_removal = False
        ambiguous_remove_error: Optional[str] = None
        if ambiguous_target != concept_id:
            try:
                await storage.graph.delete_edge(
                    message_node, ambiguous_target, "mentions"
                )
                attempted_removal = True
            except Exception as e:
                # The canonical edge IS in place; the ambiguous-edge
                # removal failed. Surface as PARTIAL so the LLM cannot
                # claim a clean resolution.
                logger.error(
                    "confirm_person_match: canonical edge written but "
                    "ambiguous edge removal failed: %s", e
                )
                ambiguous_remove_error = str(e)

        if ambiguous_remove_error:
            return ToolResult.partial(
                confirmation=(
                    f"Resolved {message_id} → {concept_id}; canonical edge "
                    "written"
                ),
                error=(
                    f"orphaned ambiguous edge {ambiguous_target} could not "
                    f"be removed: {ambiguous_remove_error}"
                ),
                data={
                    "message_id": message_id,
                    "resolved_to": concept_id,
                    "ambiguous_remove_attempted": False,
                    "orphan_edge_target": ambiguous_target,
                },
            )

        return ToolResult.ok(
            confirmation=(
                f"Resolved {message_id} → {concept_id}"
                + (
                    f" (delete_edge issued for {ambiguous_target}; "
                    "removal not verified)"
                    if attempted_removal else ""
                )
            ),
            data={
                "message_id": message_id,
                "resolved_to": concept_id,
                "ambiguous_remove_attempted": attempted_removal,
                "ambiguous_target": ambiguous_target if attempted_removal else None,
            },
        )

    @tool(
        name="mark_superseded",
        description="Mark a claim node (decision or action_item) as superseded by a newer one. Creates a 'supersedes' edge from new to old and sets superseded_by on the old node.",
        category=ToolCategory.MEMORY,
        command_prefix="!memory supersede",
    )
    async def mark_superseded(
        self,
        old_id: str,
        new_id: str,
        reason: Optional[str] = None,
    ) -> ToolResult:
        """Mark old_id as superseded by new_id.

        Only claim-shaped node types (decision, action_item) can be
        superseded. Both nodes must belong to this agent.

        Args:
            old_id: Node ID of the claim being replaced.
            new_id: Node ID of the replacing claim.
            reason: Optional human-readable reason for the supersession.
        """
        from kestrel_sovereign.storage.schema_router import CLAIM_SHAPED_NODE_TYPES

        storage = getattr(self.agent, "storage", None)
        if storage is None or not hasattr(storage, "graph"):
            return ToolResult.failed("Graph store not available")

        try:
            old_node = await storage.graph.get_node(old_id)
            new_node = await storage.graph.get_node(new_id)
        except Exception as e:
            logger.error("mark_superseded lookup failed: %s", e)
            return ToolResult.failed(str(e))

        if old_node is None:
            return ToolResult.failed(f"Node {old_id} not found")
        old_props = old_node.properties or {}
        if old_props.get("agent_id") != self.agent_id:
            return ToolResult.failed(f"Node {old_id} not found")

        if new_node is None:
            return ToolResult.failed(f"Node {new_id} not found")
        new_props = new_node.properties or {}
        if new_props.get("agent_id") != self.agent_id:
            return ToolResult.failed(f"Node {new_id} not found")

        if old_node.node_type not in CLAIM_SHAPED_NODE_TYPES:
            return ToolResult.failed(
                f"Cannot supersede node of type '{old_node.node_type}'. "
                f"Only {sorted(CLAIM_SHAPED_NODE_TYPES)} nodes can be superseded."
            )
        if new_node.node_type not in CLAIM_SHAPED_NODE_TYPES:
            return ToolResult.failed(
                f"Replacement node must be a claim type "
                f"({sorted(CLAIM_SHAPED_NODE_TYPES)}), "
                f"got '{new_node.node_type}'."
            )

        edge_props = {"reason": reason, "superseded_at": _utc_now_iso()}
        try:
            await storage.graph.add_edge(
                new_id, old_id, "supersedes", properties=edge_props
            )
        except Exception as e:
            logger.error("mark_superseded edge write failed: %s", e)
            return ToolResult.failed(str(e))

        old_props["superseded_by"] = new_id
        old_props["superseded_at"] = edge_props["superseded_at"]
        if reason:
            old_props["superseded_reason"] = reason

        from kestrel_sovereign.storage.async_graph_store import GraphNode
        try:
            await storage.graph.add_node(GraphNode(
                node_id=old_node.node_id,
                node_type=old_node.node_type,
                label=old_node.label,
                properties=old_props,
            ))
        except Exception as e:
            # The 'supersedes' edge has been written; the old node's
            # superseded_by property is stale. Surface as PARTIAL so the
            # LLM cannot claim a clean supersession.
            logger.error("mark_superseded property update failed: %s", e)
            return ToolResult.partial(
                confirmation=(
                    f"supersedes edge written: {new_id} -> {old_id}"
                ),
                error=(
                    f"superseded_by property could not be set on {old_id}: "
                    f"{e}; recall_action_items / recall_decisions will not "
                    "filter the old claim out until this is repaired"
                ),
                data={
                    "old_id": old_id,
                    "new_id": new_id,
                    "reason": reason,
                    "edge_written": True,
                    "property_updated": False,
                },
            )

        return ToolResult.ok(
            confirmation=(
                f"Marked {old_id} as superseded by {new_id}"
                + (f" ({reason})" if reason else "")
            ),
            data={
                "old_id": old_id,
                "new_id": new_id,
                "reason": reason,
                "edge_written": True,
                "property_updated": True,
            },
        )


def _utc_now_iso() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()
