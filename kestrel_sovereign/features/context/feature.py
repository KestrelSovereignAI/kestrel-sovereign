"""
Context Management Feature for Kestrel Agent.

Allows the agent to introspect, optimize, and manage its own context window.
This gives the agent tools to:
- See current context utilization
- Summarize specific conversation sections
- Mark content as protected or droppable
- Proactively trigger compression
- Exclude irrelevant content (soft removal)
- Restore excluded content
- Stash context for context-switching (like git stash)

Security safeguards:
- No permanent deletion (soft exclusion only)
- Protected content cannot be excluded or stashed
- All operations logged for audit trail
- Rate limiting to prevent manipulation loops
"""

import logging
from typing import Any, Dict, List, Optional

from kestrel_sovereign.features.base import Feature, tool
from kestrel_sovereign.tools.base import ToolCategory

logger = logging.getLogger(__name__)


class ContextFeature(Feature):
    """
    Context management tools for the agent.

    Provides tools for:
    - Checking context window utilization
    - Summarizing specific conversation sections
    - Marking content priority (protected/droppable)
    - Triggering compression
    - Excluding/restoring content
    - Stashing context for context-switching (stash/pop/apply/list/drop)
    """

    @property
    def tool_description(self) -> str:
        return "Manage context window - check status, summarize sections, mark content priority, compress, exclude/restore content"

    async def initialize(self):
        """Initialize the context feature with required references."""
        self.context_manager = getattr(self.agent, 'context_manager', None)
        self.llm_service = getattr(self.agent, 'llm_service', None)

        if not self.context_manager:
            logger.warning("ContextFeature initialized without context_manager - some tools may not work")

        logger.info("ContextFeature initialized")

    @tool(
        name="context_status",
        description="Check current context window utilization. Use this to understand how much context space is available before deciding to summarize or prune.",
        category=ToolCategory.SYSTEM,
        command_prefix="!context status"
    )
    async def context_status(self) -> Dict[str, Any]:
        """
        Get detailed context window status.

        Returns information about:
        - Total budget and used tokens by category
        - Message count and estimated tokens
        - Compression recommendation (if utilization > 70%)
        - Available headroom for new content
        """
        if not self.context_manager:
            return {"success": False, "error": "Context manager not available"}

        try:
            status = await self.context_manager.get_status()
            return status
        except Exception as e:
            logger.error(f"context_status failed: {e}")
            return {"success": False, "error": str(e)}

    @tool(
        name="summarize_section",
        description="Summarize a specific section of conversation history to save context space. Use this to compress verbose exchanges while preserving key information.",
        category=ToolCategory.MEMORY,
        command_prefix="!context summarize"
    )
    async def summarize_section(
        self,
        mode: str,
        criteria: str,
        preserve_key_facts: bool = True
    ) -> Dict[str, Any]:
        """
        Summarize a section of conversation.

        Args:
            mode: Selection mode - "time_range", "topic", "messages", or "last_n"
                - time_range: Use criteria like "before_today", "last_2_hours"
                - topic: Semantic search query like "debugging issues"
                - messages: Comma-separated message IDs like "1,2,3,4,5"
                - last_n: Number of messages like "10"
            criteria: Selection criteria based on mode
            preserve_key_facts: Keep explicit facts, decisions, commitments (default True)
        """
        if not self.context_manager:
            return {"success": False, "error": "Context manager not available"}

        if not self.llm_service:
            return {"success": False, "error": "LLM service not available for summarization"}

        try:
            # Get messages matching criteria
            messages = await self.context_manager.get_messages_for_selection(
                mode=mode,
                criteria=criteria
            )

            if not messages:
                return {
                    "success": False,
                    "error": f"No messages found for {mode}={criteria}"
                }

            if len(messages) < 2:
                return {
                    "success": False,
                    "error": "Need at least 2 messages to summarize",
                    "found": len(messages)
                }

            # Extract message IDs
            message_ids = [m["id"] for m in messages]

            # Perform summarization
            result = await self.context_manager.summarize_messages(
                llm_service=self.llm_service,
                message_ids=message_ids,
                preserve_key_facts=preserve_key_facts
            )

            return result

        except Exception as e:
            logger.error(f"summarize_section failed: {e}")
            return {"success": False, "error": str(e)}

    @tool(
        name="mark_content",
        description="Mark conversation content for context management. Use 'protect' to ensure important content is never pruned, 'droppable' to suggest low-priority content for removal.",
        category=ToolCategory.MEMORY,
        command_prefix="!context mark"
    )
    async def mark_content(
        self,
        action: str,
        target: str,
        reason: str = ""
    ) -> Dict[str, Any]:
        """
        Mark content for context management.

        Args:
            action: "protect" (never auto-prune), "droppable" (first to remove), "clear" (remove marking)
            target: Message selection - "last_5", "search:error handling", "ids:23,24,25"
            reason: Optional reason for marking (logged for audit)
        """
        if not self.context_manager:
            return {"success": False, "error": "Context manager not available"}

        try:
            # Parse target to get messages
            if target.startswith("last_"):
                n = target.split("_")[1]
                messages = await self.context_manager.get_messages_for_selection(
                    mode="last_n",
                    criteria=n
                )
            elif target.startswith("search:"):
                query = target[7:]  # Remove "search:" prefix
                messages = await self.context_manager.get_messages_for_selection(
                    mode="topic",
                    criteria=query
                )
            elif target.startswith("ids:"):
                ids_str = target[4:]  # Remove "ids:" prefix
                messages = await self.context_manager.get_messages_for_selection(
                    mode="messages",
                    criteria=ids_str
                )
            else:
                return {
                    "success": False,
                    "error": f"Invalid target format: {target}. Use 'last_N', 'search:query', or 'ids:1,2,3'"
                }

            if not messages:
                return {
                    "success": False,
                    "error": f"No messages found for target: {target}"
                }

            message_ids = [m["id"] for m in messages]

            # Perform marking
            result = await self.context_manager.mark_messages(
                message_ids=message_ids,
                action=action,
                reason=reason
            )

            return result

        except Exception as e:
            logger.error(f"mark_content failed: {e}")
            return {"success": False, "error": str(e)}

    @tool(
        name="compress_context",
        description="Compress context by summarizing older messages. Use when context utilization is high and you need space for new information.",
        category=ToolCategory.MEMORY,
        command_prefix="!context compress"
    )
    async def compress_context(
        self,
        keep_recent: int = 10,
        force: bool = False,
        dry_run: bool = False
    ) -> Dict[str, Any]:
        """
        Compress context window by summarizing older messages.

        Args:
            keep_recent: Number of recent messages to preserve verbatim (default 10)
            force: Compress even if utilization is below threshold (default False)
            dry_run: Show what would be compressed without doing it (default False)
        """
        if not self.context_manager:
            return {"success": False, "error": "Context manager not available"}

        try:
            # Check compression status first
            if dry_run:
                status = await self.context_manager.check_compression_needed()
                return {
                    "success": True,
                    "dry_run": True,
                    "compression_recommended": status["compression_recommended"],
                    "utilization_percent": status["utilization_percent"],
                    "message_count": status["message_count"],
                    "would_compress": max(0, status["message_count"] - keep_recent),
                    "would_preserve": min(keep_recent, status["message_count"])
                }

            # Check if llm_service is available
            if not self.llm_service:
                return {"success": False, "error": "LLM service not available for compression"}

            # Perform compression
            result = await self.context_manager.compress_session(
                llm_service=self.llm_service,
                preserve_recent=keep_recent,
                force=force
            )

            return result

        except Exception as e:
            logger.error(f"compress_context failed: {e}")
            return {"success": False, "error": str(e)}

    @tool(
        name="exclude_from_context",
        description="Exclude messages from context window (they remain in storage but won't be included in context). Use for redundant, superseded, or irrelevant content.",
        category=ToolCategory.MEMORY,
        command_prefix="!context exclude"
    )
    async def exclude_from_context(
        self,
        target: str,
        reason: str
    ) -> Dict[str, Any]:
        """
        Exclude content from context assembly.

        Args:
            target: Message selection - "ids:1,2,3", "search:old debug output", "last_5"
            reason: Required reason for exclusion (logged for audit)

        Note: Cannot exclude protected content. User messages with explicit
        importance markers are protected by default.
        """
        if not self.context_manager:
            return {"success": False, "error": "Context manager not available"}

        if not reason:
            return {"success": False, "error": "Reason is required for exclusion"}

        try:
            # Parse target to get messages
            if target.startswith("last_"):
                n = target.split("_")[1]
                messages = await self.context_manager.get_messages_for_selection(
                    mode="last_n",
                    criteria=n
                )
            elif target.startswith("search:"):
                query = target[7:]
                messages = await self.context_manager.get_messages_for_selection(
                    mode="topic",
                    criteria=query
                )
            elif target.startswith("ids:"):
                ids_str = target[4:]
                messages = await self.context_manager.get_messages_for_selection(
                    mode="messages",
                    criteria=ids_str
                )
            else:
                return {
                    "success": False,
                    "error": f"Invalid target format: {target}. Use 'last_N', 'search:query', or 'ids:1,2,3'"
                }

            if not messages:
                return {
                    "success": False,
                    "error": f"No messages found for target: {target}"
                }

            message_ids = [m["id"] for m in messages]

            # Perform exclusion
            result = await self.context_manager.exclude_messages(
                message_ids=message_ids,
                reason=reason
            )

            return result

        except Exception as e:
            logger.error(f"exclude_from_context failed: {e}")
            return {"success": False, "error": str(e)}

    @tool(
        name="restore_excluded",
        description="Restore previously excluded content back to context.",
        category=ToolCategory.MEMORY,
        command_prefix="!context restore"
    )
    async def restore_excluded(
        self,
        target: str = "all"
    ) -> Dict[str, Any]:
        """
        Restore excluded content.

        Args:
            target: What to restore - "all", "recent" (last exclusion), or "ids:1,2,3"
        """
        if not self.context_manager:
            return {"success": False, "error": "Context manager not available"}

        try:
            if target == "all":
                # Restore all excluded messages
                result = await self.context_manager.restore_messages(message_ids=None)
            elif target == "recent":
                # Get recently excluded and restore them
                conv_store = self.context_manager._get_conversation_store()
                if conv_store:
                    excluded = await conv_store.get_excluded_messages(limit=10)
                    if excluded:
                        # Sort by excluded_at to get most recent
                        excluded.sort(
                            key=lambda m: m.get("metadata", {}).get("excluded_at", ""),
                            reverse=True
                        )
                        # Restore the most recent exclusion batch
                        # (messages excluded at the same time)
                        recent_time = excluded[0].get("metadata", {}).get("excluded_at")
                        recent_ids = [
                            m["id"] for m in excluded
                            if m.get("metadata", {}).get("excluded_at") == recent_time
                        ]
                        result = await self.context_manager.restore_messages(message_ids=recent_ids)
                    else:
                        result = {"success": True, "restored_count": 0, "note": "No excluded messages found"}
                else:
                    result = {"success": False, "error": "Conversation store not available"}
            elif target.startswith("ids:"):
                ids_str = target[4:]
                try:
                    message_ids = [int(x.strip()) for x in ids_str.split(",")]
                    result = await self.context_manager.restore_messages(message_ids=message_ids)
                except ValueError:
                    return {"success": False, "error": f"Invalid message IDs: {ids_str}"}
            else:
                return {
                    "success": False,
                    "error": f"Invalid target: {target}. Use 'all', 'recent', or 'ids:1,2,3'"
                }

            return result

        except Exception as e:
            logger.error(f"restore_excluded failed: {e}")
            return {"success": False, "error": str(e)}

    # =========================================================================
    # Stash Tools (Temporary Context Parking)
    # =========================================================================

    @tool(
        name="context_stash",
        description="Stash current working context (like git stash). Use when you need to context-switch to a different topic and want to restore the current discussion later.",
        category=ToolCategory.MEMORY,
        command_prefix="!context stash"
    )
    async def context_stash(
        self,
        target: str = "last_10",
        name: str = ""
    ) -> Dict[str, Any]:
        """
        Stash messages for later restoration.

        Args:
            target: What to stash - "last_N" (e.g., "last_10") or "ids:1,2,3"
            name: Optional name for this stash (e.g., "debugging-session")
        """
        if not self.context_manager:
            return {"success": False, "error": "Context manager not available"}

        try:
            if target.startswith("last_"):
                try:
                    n = int(target.split("_")[1])
                    result = await self.context_manager.stash_messages(
                        last_n=n,
                        name=name if name else None
                    )
                except ValueError:
                    return {"success": False, "error": f"Invalid last_N format: {target}"}
            elif target.startswith("ids:"):
                try:
                    ids_str = target[4:]
                    message_ids = [int(x.strip()) for x in ids_str.split(",")]
                    result = await self.context_manager.stash_messages(
                        message_ids=message_ids,
                        name=name if name else None
                    )
                except ValueError:
                    return {"success": False, "error": f"Invalid message IDs: {target}"}
            else:
                return {
                    "success": False,
                    "error": f"Invalid target: {target}. Use 'last_N' or 'ids:1,2,3'"
                }

            return result

        except Exception as e:
            logger.error(f"context_stash failed: {e}")
            return {"success": False, "error": str(e)}

    @tool(
        name="context_stash_pop",
        description="Pop the most recent stash (restore messages and remove from stash list). Like git stash pop.",
        category=ToolCategory.MEMORY,
        command_prefix="!context stash pop"
    )
    async def context_stash_pop(
        self,
        stash_id: str = ""
    ) -> Dict[str, Any]:
        """
        Pop a stash - restore messages to context and remove from stash list.

        Args:
            stash_id: Optional specific stash ID to pop (default: most recent)
        """
        if not self.context_manager:
            return {"success": False, "error": "Context manager not available"}

        try:
            result = await self.context_manager.stash_pop(
                stash_id=stash_id if stash_id else None
            )
            return result
        except Exception as e:
            logger.error(f"context_stash_pop failed: {e}")
            return {"success": False, "error": str(e)}

    @tool(
        name="context_stash_apply",
        description="Apply a stash without removing it (restore messages but keep stash for reuse). Like git stash apply.",
        category=ToolCategory.MEMORY,
        command_prefix="!context stash apply"
    )
    async def context_stash_apply(
        self,
        stash_id: str = ""
    ) -> Dict[str, Any]:
        """
        Apply a stash - restore messages to context but keep stash reference.

        Args:
            stash_id: Optional specific stash ID to apply (default: most recent)
        """
        if not self.context_manager:
            return {"success": False, "error": "Context manager not available"}

        try:
            result = await self.context_manager.stash_apply(
                stash_id=stash_id if stash_id else None
            )
            return result
        except Exception as e:
            logger.error(f"context_stash_apply failed: {e}")
            return {"success": False, "error": str(e)}

    @tool(
        name="context_stash_list",
        description="List all stashes with their names and message counts.",
        category=ToolCategory.MEMORY,
        command_prefix="!context stash list"
    )
    async def context_stash_list(self) -> Dict[str, Any]:
        """
        List all available stashes.

        Returns list of stashes with:
        - stash_id: Short identifier for the stash
        - name: Human-readable name
        - message_count: Number of messages in stash
        - stashed_at: When the stash was created
        """
        if not self.context_manager:
            return {"success": False, "error": "Context manager not available"}

        try:
            result = await self.context_manager.stash_list()
            return result
        except Exception as e:
            logger.error(f"context_stash_list failed: {e}")
            return {"success": False, "error": str(e)}

    @tool(
        name="context_stash_drop",
        description="Drop a stash without restoring (discard stashed messages). Messages become excluded from context.",
        category=ToolCategory.MEMORY,
        command_prefix="!context stash drop"
    )
    async def context_stash_drop(
        self,
        stash_id: str = ""
    ) -> Dict[str, Any]:
        """
        Drop a stash without restoring messages.

        Messages are excluded from context (can be recovered with restore_excluded).

        Args:
            stash_id: Optional specific stash ID to drop (default: most recent)
        """
        if not self.context_manager:
            return {"success": False, "error": "Context manager not available"}

        try:
            result = await self.context_manager.stash_drop(
                stash_id=stash_id if stash_id else None
            )
            return result
        except Exception as e:
            logger.error(f"context_stash_drop failed: {e}")
            return {"success": False, "error": str(e)}

    @tool(
        name="context_stash_save",
        description="Save a stash to long-term storage with semantic search capability. Use when you want to preserve context for future retrieval via !recall.",
        category=ToolCategory.MEMORY,
        command_prefix="!context stash save"
    )
    async def context_stash_save(
        self,
        stash_id: str = "",
        name: str = "",
        summary: str = "",
        tags: str = ""
    ) -> Dict[str, Any]:
        """
        Save a stash to SavedItems for long-term retrieval.

        The stash content is stored with an embedding for semantic search,
        allowing later retrieval via !recall or search_saved_items.

        Args:
            stash_id: Optional specific stash ID to save (default: most recent)
            name: Optional name for the saved item (default: stash name)
            summary: Optional summary for search (auto-generated if not provided)
            tags: Comma-separated tags for filtering (e.g., "debugging,session")
        """
        if not self.context_manager:
            return {"success": False, "error": "Context manager not available"}

        try:
            tags_list = [t.strip() for t in tags.split(",")] if tags else None

            result = await self.context_manager.stash_save(
                stash_id=stash_id if stash_id else None,
                name=name if name else None,
                summary=summary if summary else None,
                tags=tags_list
            )
            return result
        except Exception as e:
            logger.error(f"context_stash_save failed: {e}")
            return {"success": False, "error": str(e)}

    @tool(
        name="context_stash_peek",
        description="Peek at stash contents without restoring. Use to explore stashed context programmatically (RLM-inspired context-as-variable).",
        category=ToolCategory.MEMORY,
        command_prefix="!context stash peek"
    )
    async def context_stash_peek(
        self,
        stash_id: str = "",
        max_chars: int = 5000
    ) -> Dict[str, Any]:
        """
        Peek at stash contents without fully restoring.

        This allows programmatic exploration of stashed context
        without loading everything into the context window.

        Args:
            stash_id: Optional specific stash ID to peek (default: most recent)
            max_chars: Maximum characters to return (default: 5000)
        """
        if not self.context_manager:
            return {"success": False, "error": "Context manager not available"}

        try:
            result = await self.context_manager.stash_peek(
                stash_id=stash_id if stash_id else None,
                max_chars=max_chars
            )
            return result
        except Exception as e:
            logger.error(f"context_stash_peek failed: {e}")
            return {"success": False, "error": str(e)}

    # =========================================================================
    # RLM-Inspired Advanced Compression
    # =========================================================================

    @tool(
        name="hierarchical_compress",
        description="Compress context using hierarchical tree-structured summarization (RLM-inspired). Better preserves structure than linear compression.",
        category=ToolCategory.MEMORY,
        command_prefix="!context compress hierarchical"
    )
    async def hierarchical_compress(
        self,
        chunk_size: int = 4000,
        keep_recent: int = 5,
        max_depth: int = 3
    ) -> Dict[str, Any]:
        """
        Hierarchical compression using recursive summarization.

        Unlike linear compression which flattens everything, this:
        1. Splits messages into chunks
        2. Summarizes each chunk
        3. Recursively merges summaries
        4. Preserves more structure

        Inspired by RLM paper's approach to long context.

        Args:
            chunk_size: Target characters per chunk (default: 4000)
            keep_recent: Messages to preserve verbatim (default: 5)
            max_depth: Maximum recursion depth (default: 3)
        """
        if not self.context_manager:
            return {"success": False, "error": "Context manager not available"}

        if not self.llm_service:
            return {"success": False, "error": "LLM service not available for compression"}

        try:
            result = await self.context_manager.hierarchical_compress(
                llm_service=self.llm_service,
                chunk_size=chunk_size,
                preserve_recent=keep_recent,
                max_depth=max_depth
            )
            return result
        except Exception as e:
            logger.error(f"hierarchical_compress failed: {e}")
            return {"success": False, "error": str(e)}

    @tool(
        name="recursive_query",
        description="Query a subset of context using a cheaper model (RLM-inspired sub-LM call). Use for exploring large context sections, compressed originals, or excluded messages without using main model quota.",
        category=ToolCategory.MEMORY,
        command_prefix="!context query"
    )
    async def recursive_query(
        self,
        context_source: str,
        query: str,
        use_cheap_model: bool = True
    ) -> Dict[str, Any]:
        """
        Query context subset using recursive sub-LM call.

        This allows exploring stashed, excluded, or compressed context using a
        cheaper model, preserving main model quota for important work.

        Args:
            context_source: Source - "stash:name", "excluded", "compressed:ID", "summary:ID", "last_N"
            query: Question to ask about the context
            use_cheap_model: Use cheaper model for query (default: True)
        """
        if not self.context_manager:
            return {"success": False, "error": "Context manager not available"}

        if not self.llm_service:
            return {"success": False, "error": "LLM service not available"}

        try:
            # Get the context to query
            context_text = ""

            if context_source.startswith("stash:"):
                stash_name = context_source[6:]
                peek_result = await self.context_manager.stash_peek(
                    stash_id=stash_name,
                    max_chars=10000  # Allow larger for query
                )
                if peek_result.get("success"):
                    context_text = peek_result.get("preview", "")
                else:
                    return {"success": False, "error": peek_result.get("error", "Stash not found")}

            elif context_source == "excluded":
                conv_store = self.context_manager._get_conversation_store()
                if conv_store:
                    excluded = await conv_store.get_excluded_messages(limit=50)
                    context_text = "\n".join([
                        f"{m.get('role', 'user').upper()}: {m.get('content', '')}"
                        for m in excluded
                    ])[:10000]
                else:
                    return {"success": False, "error": "Conversation store not available"}

            elif context_source.startswith("compressed:") or context_source.startswith("summary:"):
                # View original messages that were compressed/summarized
                try:
                    marker_id = context_source.split(":", 1)[1]
                    conv_store = self.context_manager._get_conversation_store()
                    if not conv_store:
                        return {"success": False, "error": "Conversation store not available"}

                    # Get the compression/summary marker
                    marker_messages = await conv_store.get_messages_by_ids([int(marker_id)])
                    if not marker_messages:
                        return {"success": False, "error": f"Marker message {marker_id} not found"}

                    marker = marker_messages[0]
                    meta = marker.get("metadata", {})
                    original_ids = meta.get("original_message_ids", [])

                    if not original_ids:
                        return {"success": False, "error": f"No original message IDs found in marker {marker_id}"}

                    # Get the original messages
                    original_messages = await conv_store.get_messages_by_ids(original_ids)
                    if not original_messages:
                        return {"success": False, "error": "Original messages not found"}

                    context_text = "\n".join([
                        f"{m.get('role', 'user').upper()}: {m.get('content', '')}"
                        for m in original_messages
                    ])[:10000]

                except (ValueError, IndexError) as e:
                    return {"success": False, "error": f"Invalid format: {context_source} - {str(e)}"}

            elif context_source.startswith("last_"):
                try:
                    n = int(context_source.split("_")[1])
                    messages = await self.context_manager.get_messages_for_selection(
                        mode="last_n",
                        criteria=str(n)
                    )
                    context_text = "\n".join([
                        f"{m.get('role', 'user').upper()}: {m.get('content', '')}"
                        for m in messages
                    ])[:10000]
                except ValueError:
                    return {"success": False, "error": f"Invalid format: {context_source}"}
            else:
                return {
                    "success": False,
                    "error": f"Invalid source: {context_source}. Use 'stash:name', 'excluded', 'compressed:ID', 'summary:ID', or 'last_N'"
                }

            if not context_text:
                return {"success": False, "error": "No context found to query"}

            # Build prompt for recursive query
            prompt = f"""Answer the following question based ONLY on this context:

CONTEXT:
{context_text}

QUESTION: {query}

ANSWER:"""

            # Use cheap model if requested
            model_override = None
            if use_cheap_model and hasattr(self.llm_service, 'get_cheap_model'):
                model_override = self.llm_service.get_cheap_model()

            # Generate response
            response = await self.llm_service.generate(
                prompt=prompt,
                system_prompt="You are answering questions about conversation context. Be concise and accurate.",
                model_override=model_override
            )

            return {
                "success": True,
                "answer": response.strip() if isinstance(response, str) else str(response),
                "context_source": context_source,
                "query": query,
                "model_used": model_override or "default",
                "context_chars": len(context_text)
            }

        except Exception as e:
            logger.error(f"recursive_query failed: {e}")
            return {"success": False, "error": str(e)}
