"""
Memory Manager for Kestrel Agent.

Handles memory retrieval, episode management, and emotional memory operations.
Extracted from ContextManager to improve modularity and maintainability.
"""

import html
import logging
from typing import List, Dict, Any, Optional, TYPE_CHECKING
import uuid

from kestrel_sovereign.security.input_guardrails import extract_raw_user_content

if TYPE_CHECKING:
    from kestrel_sovereign.storage import AsyncStorage
    from kestrel_sovereign.storage.memory_consolidator import MemoryConsolidator
    from kestrel_sovereign.storage.memory_retriever import MemoryRetriever

logger = logging.getLogger(__name__)


class MemoryManager:
    """
    Manages memory retrieval, episode consolidation, and stash operations.

    Responsibilities:
    1. Retrieve emotionally-weighted memories
    2. Episode creation and management
    3. Stash operations (context parking)
    4. Hierarchical compaction of messages
    5. Memory-based context enrichment
    """

    def __init__(
        self,
        storage: "AsyncStorage",
        agent_id: Optional[str] = None,
        consolidator: Optional["MemoryConsolidator"] = None,
        memory_retriever: Optional["MemoryRetriever"] = None,
    ):
        """
        Initialize the memory manager.

        Args:
            storage: AsyncStorage instance for memory operations
            agent_id: Agent ID for scoped queries
            consolidator: MemoryConsolidator for episode access
            memory_retriever: MemoryRetriever for emotional memory access
        """
        self.storage = storage
        self.agent_id = agent_id
        self.consolidator = consolidator
        self.memory_retriever = memory_retriever

    # =========================================================================
    # Post-Response Memory Tagging
    # =========================================================================

    async def tag_exchange(
        self,
        user_content: str,
        assistant_content: str,
        user_message_id: Optional[int] = None,
        assistant_message_id: Optional[int] = None,
        memory_system=None,
    ) -> Dict[str, Any]:
        """
        Tag a user+assistant exchange with emotional metadata (Phase 1 sync).

        Called inline after the LLM response is stored. EmotionalTagger is
        CPU-bound (regex keyword matching) so this is safe to run in the
        request path without blocking on I/O.

        Args:
            user_content: The user's message text
            assistant_content: The assistant's response text
            user_message_id: DB row ID of the user message (if available)
            assistant_message_id: DB row ID of the assistant message (if available)
            memory_system: MemorySystem instance for tag_message()

        Returns:
            Dict with tagging results for both messages
        """
        results: Dict[str, Any] = {"user": None, "assistant": None}

        if not memory_system:
            logger.debug("No memory_system provided, skipping tag_exchange")
            return results

        try:
            if user_message_id is not None:
                results["user"] = await memory_system.tag_message(
                    user_message_id, user_content, role="user"
                )
            if assistant_message_id is not None:
                results["assistant"] = await memory_system.tag_message(
                    assistant_message_id, assistant_content, role="assistant"
                )
        except Exception as e:
            logger.error(f"tag_exchange failed: {e}", exc_info=True)

        return results

    async def retrieve_memories(
        self,
        query: str,
        max_tokens: int,
        counter,
        emotional_context: Optional[Dict[str, Any]] = None,
        min_score: Optional[float] = None,
    ) -> Optional[str]:
        """
        Retrieve emotionally-weighted memories.

        Uses MemoryRetriever for human-like recall with:
        - Semantic matching (30%)
        - Emotional congruence (25%)
        - Importance weighting (20%)
        - Recency with decay (15%)
        - Access frequency (10%)

        Args:
            min_score: Relevance gate floor (#1404). When set, memories
                with weighted retrieval score below this value are
                dropped — keeps weak matches from being stamped into
                the rendered transport form. ``None`` keeps the
                underlying ``MemoryRetriever`` default (0.1).
        """
        if not self.memory_retriever:
            return None

        try:
            from kestrel_sovereign.storage.memory_models import MemoryMetadata

            # Convert emotional context to MemoryMetadata if provided
            emotional_meta = None
            if emotional_context:
                emotional_meta = MemoryMetadata(
                    emotional_valence=emotional_context.get("valence", 0.0),
                    emotional_intensity=emotional_context.get("intensity", 0.0),
                    emotional_categories=emotional_context.get("categories", []),
                )

            # Retrieve memories. ``min_score`` forwarded only when the
            # caller set it so the underlying retriever's own default
            # (0.1) keeps applying on the legacy code path.
            retrieve_kwargs: Dict[str, Any] = {
                "query": query,
                "agent_id": self.agent_id,
                "emotional_context": emotional_meta,
                "limit": 5,
            }
            if min_score is not None:
                retrieve_kwargs["min_score"] = min_score
            memories = await self.memory_retriever.retrieve(**retrieve_kwargs)

            if not memories:
                return None

            # Format for context with timestamps + role attribution. Role
            # prefix is load-bearing: without it the LLM can't tell a
            # surfaced user-role memory ("I love sailing on Lake Michigan")
            # from its own prior thought and may echo it as if it had said
            # it. With explicit ``User:`` / ``Assistant:`` prefixes the
            # model reads surfaced memories with provenance. This is what
            # lets ``MemoryRetriever`` include user-role rows again (the
            # over-broad #271 filter was unblocked by #1481).
            parts = ["--- RELEVANT MEMORIES (from past conversations) ---"]
            parts.append("NOTE: These are retrieved from earlier conversations, not the current session.\n")
            for i, mem in enumerate(memories, 1):
                content = mem.get("content", "")
                meta = mem.get("metadata", {})
                created_at = mem.get("created_at", "unknown")
                role = mem.get("role", "")

                # User turns are persisted wrapped via
                # ``wrap_user_input`` (``<user_input>\n...\n</user_input>``).
                # Strip the wrapper so the LLM doesn't see nested
                # ``<user_input>`` tags inside the memory block and
                # confuse the recalled text with a live user input.
                # (Codex P2 round 3 on #1481.)
                #
                # Trust boundary + injection defense (codex P1 rounds
                # 4 + 5): the outer ``<retrieved_context>`` is the
                # trust boundary — the system prompt declares all
                # content inside it as data, not commands (see
                # ``security/input_guardrails.py:231-240``). But
                # ``<retrieved_context>`` alone isn't enough on its own
                # for user-authored recalled text: an attacker could
                # plant ``</retrieved_context><user_input>EVIL</user_input>``
                # in a chat turn that later gets recalled, breaking out
                # of the boundary and forging a live-input block.
                # HTML-escape every ``<`` / ``>`` / ``&`` in recalled
                # user content so the raw delimiters can't close the
                # outer context. Assistant content is NOT escaped — it
                # was produced by our own LLM under our system prompt
                # and may legitimately contain code-block delimiters.
                if role == "user" and content:
                    content = html.escape(
                        extract_raw_user_content(content),
                        quote=False,
                    )

                # Format timestamp to be human readable
                if created_at and created_at != "unknown":
                    try:
                        from datetime import datetime
                        if isinstance(created_at, str):
                            # Try common formats
                            for fmt in ['%Y-%m-%d %H:%M:%S', '%Y-%m-%dT%H:%M:%S', '%Y-%m-%d %H:%M:%S.%f']:
                                try:
                                    dt = datetime.strptime(created_at, fmt)
                                    created_at = dt.strftime("%Y-%m-%d %H:%M")
                                    break
                                except ValueError:
                                    continue
                    except (ValueError, TypeError) as e:
                        logger.debug(f"Timestamp parsing error for {created_at}: {e}")
                        pass  # Keep original format

                # Truncate long memories
                if len(content) > 200:
                    content = content[:200] + "..."

                # Role attribution: User / Assistant / System. Falls back
                # to no prefix only if the row is missing the role field
                # entirely (legacy data shouldn't be — conversation_history
                # has had a NOT NULL role since the initial schema).
                role_prefix = f"{role.capitalize()}: " if role else ""

                parts.append(
                    f"[Memory {i}] ({created_at}) {role_prefix}{content}\n"
                    f"  Importance: {meta.get('importance', 0.5):.1f}, "
                    f"Emotion: {meta.get('emotional_valence', 0):.1f}"
                )
            parts.append("--- END MEMORIES ---")

            return "\n".join(parts)

        except (ConnectionError, TimeoutError) as e:
            logger.error(f"Storage connection error during memory retrieval: {e}", exc_info=True)
            return None
        except (KeyError, ValueError, TypeError) as e:
            logger.error(f"Data error during memory retrieval: {e}", exc_info=True)
            return None
        except Exception as e:
            logger.error(f"Memory retrieval failed: {e}", exc_info=True)
            return None

    async def check_episode_needed(self, session_messages: int = 0) -> bool:
        """
        Check if a new episode should be created.

        Delegates to MemoryConsolidator if available.
        """
        if not self.consolidator:
            return False

        return await self.consolidator.should_create_episode(session_messages)

    async def create_episode_if_needed(
        self,
        session_messages: int = 0,
        force: bool = False
    ) -> Optional[Any]:
        """
        Create an episode if conditions are met.

        Args:
            session_messages: Messages in current session
            force: Force creation even if threshold not met

        Returns:
            Created episode or None
        """
        if not self.consolidator:
            return None

        if force or await self.check_episode_needed(session_messages):
            return await self.consolidator.create_session_episode(force=force)

        return None

    # =========================================================================
    # Stash Methods (Temporary Context Parking)
    # =========================================================================

    async def stash_messages(
        self,
        message_ids: Optional[List[int]] = None,
        name: Optional[str] = None,
        last_n: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Stash messages (temporarily remove from context).

        Like git stash - parks messages so you can context-switch,
        then restore them later with pop or apply.

        Args:
            message_ids: Specific message IDs to stash
            name: Optional name for this stash (e.g., "debugging-session")
            last_n: Stash the last N messages (alternative to message_ids)

        Returns:
            Result dict with stash_id and count
        """
        from datetime import datetime, timezone

        conv_store = self._get_conversation_store()
        if not conv_store:
            return {"success": False, "error": "Conversation store not available"}

        # Get messages to stash
        if message_ids:
            messages = await conv_store.get_messages_by_ids(message_ids)
        elif last_n:
            all_messages = await conv_store.get_full_history_with_ids()
            messages = all_messages[-last_n:] if len(all_messages) >= last_n else all_messages
        else:
            return {"success": False, "error": "Must specify message_ids or last_n"}

        if not messages:
            return {"success": False, "error": "No messages found to stash"}

        # Filter out protected messages
        protected_ids = []
        stashable = []
        for msg in messages:
            meta = msg.get("metadata", {})
            if meta.get("context_priority") == "protected" or meta.get("decay_protected"):
                protected_ids.append(msg["id"])
            else:
                stashable.append(msg)

        if not stashable:
            return {
                "success": False,
                "error": "No stashable messages (all protected)",
                "protected_count": len(protected_ids)
            }

        # Create stash
        stash_id = str(uuid.uuid4())[:8]  # Short ID for easier reference
        now = datetime.now(timezone.utc).isoformat()
        stash_name = name or f"stash-{stash_id}"

        # Update message metadata
        stash_update = {
            "stashed": True,
            "stash_id": stash_id,
            "stash_name": stash_name,
            "stashed_at": now
        }

        stash_ids = [m["id"] for m in stashable]
        updated = await conv_store.update_messages_metadata(stash_ids, stash_update)

        # Log audit
        await self._log_context_audit(
            action="stash",
            message_ids=stash_ids,
            reason=f"Stashed as '{stash_name}'"
        )

        return {
            "success": True,
            "stash_id": stash_id,
            "stash_name": stash_name,
            "stashed_count": updated,
            "protected_count": len(protected_ids)
        }

    async def stash_pop(
        self,
        stash_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Pop the most recent stash (restore and remove from stash).

        Like git stash pop - restores messages to context and removes
        them from the stash.

        Args:
            stash_id: Specific stash to pop (default: most recent)

        Returns:
            Result dict with restored count
        """
        conv_store = self._get_conversation_store()
        if not conv_store:
            return {"success": False, "error": "Conversation store not available"}

        # Get stash to pop
        if stash_id:
            stashed = await conv_store.get_stashed_messages(stash_id=stash_id)
        else:
            # Get most recent stash
            stashes = await conv_store.list_stashes()
            if not stashes:
                return {"success": True, "restored_count": 0, "note": "No stashes found"}
            stash_id = stashes[0]["stash_id"]
            stashed = await conv_store.get_stashed_messages(stash_id=stash_id)

        if not stashed:
            return {"success": True, "restored_count": 0, "note": "Stash is empty or not found"}

        # Clear stash metadata
        clear_update = {
            "stashed": False,
            "stash_id": None,
            "stash_name": None,
            "stashed_at": None
        }

        message_ids = [m["id"] for m in stashed]
        updated = await conv_store.update_messages_metadata(message_ids, clear_update)

        # Log audit
        await self._log_context_audit(
            action="stash_pop",
            message_ids=message_ids,
            reason=f"Popped stash {stash_id}"
        )

        return {
            "success": True,
            "stash_id": stash_id,
            "restored_count": updated
        }

    async def stash_apply(
        self,
        stash_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Apply a stash without removing it (like git stash apply).

        Restores messages to context but keeps them in the stash
        for potential reuse.

        Args:
            stash_id: Specific stash to apply (default: most recent)

        Returns:
            Result dict with applied count
        """
        conv_store = self._get_conversation_store()
        if not conv_store:
            return {"success": False, "error": "Conversation store not available"}

        # Get stash to apply
        if stash_id:
            stashed = await conv_store.get_stashed_messages(stash_id=stash_id)
        else:
            # Get most recent stash
            stashes = await conv_store.list_stashes()
            if not stashes:
                return {"success": True, "applied_count": 0, "note": "No stashes found"}
            stash_id = stashes[0]["stash_id"]
            stashed = await conv_store.get_stashed_messages(stash_id=stash_id)

        if not stashed:
            return {"success": True, "applied_count": 0, "note": "Stash is empty or not found"}

        # Clear only the stashed flag (keep stash_id for tracking)
        # This allows the messages to appear in context while keeping stash reference
        apply_update = {
            "stashed": False
        }

        message_ids = [m["id"] for m in stashed]
        updated = await conv_store.update_messages_metadata(message_ids, apply_update)

        # Log audit
        await self._log_context_audit(
            action="stash_apply",
            message_ids=message_ids,
            reason=f"Applied stash {stash_id} (kept in stash list)"
        )

        return {
            "success": True,
            "stash_id": stash_id,
            "applied_count": updated,
            "note": "Messages restored to context. Stash reference preserved."
        }

    async def stash_list(self) -> Dict[str, Any]:
        """
        List all stashes.

        Returns:
            Dict with list of stashes, each containing:
            - stash_id: Short identifier
            - name: Human-readable name
            - message_count: Number of messages in stash
            - stashed_at: When stash was created
        """
        conv_store = self._get_conversation_store()
        if not conv_store:
            return {"success": False, "error": "Conversation store not available"}

        stashes = await conv_store.list_stashes()

        return {
            "success": True,
            "stash_count": len(stashes),
            "stashes": stashes
        }

    async def stash_drop(
        self,
        stash_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Drop (delete) a stash without restoring.

        Messages remain in history but stash metadata is cleared.

        Args:
            stash_id: Specific stash to drop (default: most recent)

        Returns:
            Result dict with dropped count
        """
        conv_store = self._get_conversation_store()
        if not conv_store:
            return {"success": False, "error": "Conversation store not available"}

        # Get stash to drop
        if stash_id:
            stashed = await conv_store.get_stashed_messages(stash_id=stash_id)
        else:
            # Get most recent stash
            stashes = await conv_store.list_stashes()
            if not stashes:
                return {"success": True, "dropped_count": 0, "note": "No stashes found"}
            stash_id = stashes[0]["stash_id"]
            stashed = await conv_store.get_stashed_messages(stash_id=stash_id)

        if not stashed:
            return {"success": True, "dropped_count": 0, "note": "Stash is empty or not found"}

        # Clear stash metadata but keep messages excluded from context
        # (they were stashed for a reason - don't auto-restore on drop)
        drop_update = {
            "stashed": False,
            "stash_id": None,
            "stash_name": None,
            "stashed_at": None,
            "excluded_from_context": True,
            "excluded_reason": f"Dropped from stash {stash_id}"
        }

        message_ids = [m["id"] for m in stashed]
        updated = await conv_store.update_messages_metadata(message_ids, drop_update)

        # Log audit
        await self._log_context_audit(
            action="stash_drop",
            message_ids=message_ids,
            reason=f"Dropped stash {stash_id}"
        )

        return {
            "success": True,
            "stash_id": stash_id,
            "dropped_count": updated,
            "note": "Messages excluded from context. Use restore_excluded to recover."
        }

    async def stash_save(
        self,
        stash_id: Optional[str] = None,
        name: Optional[str] = None,
        summary: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Save a stash to long-term storage for semantic retrieval.

        This persists the stash content with an embedding so it can be
        found later via semantic search (!recall).

        Args:
            stash_id: Specific stash to save (default: most recent)
            name: Name for the saved item (default: stash name)
            summary: Optional summary for search
            tags: Optional tags for filtering

        Returns:
            Result dict with saved_item_id
        """
        import json

        conv_store = self._get_conversation_store()
        if not conv_store:
            return {"success": False, "error": "Conversation store not available"}

        # Get the stash
        if stash_id:
            stashed = await conv_store.get_stashed_messages(stash_id=stash_id)
            stash_name = stash_id
        else:
            # Get most recent stash
            stashes = await conv_store.list_stashes()
            if not stashes:
                return {"success": False, "error": "No stashes found"}
            stash_id = stashes[0]["stash_id"]
            stash_name = stashes[0].get("name", stash_id)
            stashed = await conv_store.get_stashed_messages(stash_id=stash_id)

        if not stashed:
            return {"success": False, "error": f"Stash {stash_id} is empty or not found"}

        # Build content
        content_json = json.dumps({
            "stash_id": stash_id,
            "stash_name": stash_name,
            "message_count": len(stashed),
            "messages": stashed
        })

        source_ref = json.dumps({
            "stash_id": stash_id,
            "message_ids": [m.get("id") for m in stashed]
        })

        # Generate summary if not provided
        if not summary:
            messages_text = []
            for msg in stashed[:3]:
                role = msg.get("role", "")
                content = msg.get("content", "")[:200]
                messages_text.append(f"{role}: {content}")
            summary = f"Stash '{stash_name}' ({len(stashed)} messages): " + " | ".join(messages_text)

        # Get SavedItemsStore
        try:
            from kestrel_sovereign.storage.saved_items_store import (
                SavedItemsStore, SavedItemType, SourceType
            )

            db = getattr(self.storage, 'db', None)
            if not db:
                return {"success": False, "error": "Database not available"}

            store = SavedItemsStore(
                db,
                self.agent_id,
                llm_service=getattr(self.storage, "llm_service", None),
            )

            item = await store.save_item(
                item_type=SavedItemType.STASH.value,
                name=name or stash_name,
                content=content_json,
                summary=summary,
                source_type=SourceType.CONVERSATION.value,
                source_ref=source_ref,
                tags=tags or [],
                metadata={"original_stash_id": stash_id}
            )

            # Log audit
            await self._log_context_audit(
                action="stash_save",
                message_ids=[m.get("id") for m in stashed],
                reason=f"Saved stash '{stash_name}' as item {item.id}"
            )

            return {
                "success": True,
                "saved_item_id": item.id,
                "stash_id": stash_id,
                "name": item.name,
                "message_count": len(stashed),
                "has_embedding": item.embedding is not None
            }

        except ImportError as e:
            logger.error(f"SavedItemsStore import failed: {e}", exc_info=True)
            return {"success": False, "error": f"SavedItemsStore not available: {e}"}
        except (ConnectionError, TimeoutError) as e:
            logger.error(f"Storage connection error during stash save: {e}", exc_info=True)
            return {"success": False, "error": f"Storage error: {str(e)}"}
        except (KeyError, ValueError, TypeError) as e:
            logger.error(f"Data error during stash save: {e}", exc_info=True)
            return {"success": False, "error": f"Data error: {str(e)}"}
        except Exception as e:
            logger.error(f"Stash save failed: {e}", exc_info=True)
            return {"success": False, "error": str(e)}

    async def stash_peek(
        self,
        stash_id: Optional[str] = None,
        max_chars: int = 5000
    ) -> Dict[str, Any]:
        """
        Peek at stash contents without fully restoring (RLM-inspired).

        This allows the agent to programmatically explore stashed context
        without loading it all into the context window.

        Args:
            stash_id: Specific stash to peek (default: most recent)
            max_chars: Maximum characters to return (default: 5000)

        Returns:
            Dict with stash preview and metadata
        """
        conv_store = self._get_conversation_store()
        if not conv_store:
            return {"success": False, "error": "Conversation store not available"}

        # Get stash
        if stash_id:
            stashed = await conv_store.get_stashed_messages(stash_id=stash_id)
            stash_name = stash_id
        else:
            stashes = await conv_store.list_stashes()
            if not stashes:
                return {"success": False, "error": "No stashes found"}
            stash_id = stashes[0]["stash_id"]
            stash_name = stashes[0].get("name", stash_id)
            stashed = await conv_store.get_stashed_messages(stash_id=stash_id)

        if not stashed:
            return {"success": False, "error": f"Stash {stash_id} is empty"}

        # Build preview within char limit
        preview_parts = []
        char_count = 0
        included_count = 0
        truncated = False

        for msg in stashed:
            role = msg.get("role", "user").upper()
            content = msg.get("content", "")

            # Check if we have room
            msg_preview = f"{role}: {content}"
            if char_count + len(msg_preview) > max_chars:
                # Truncate this message to fit
                remaining = max_chars - char_count
                if remaining > 50:  # Only include if meaningful space left
                    msg_preview = f"{role}: {content[:remaining-10]}..."
                    preview_parts.append(msg_preview)
                    included_count += 1
                truncated = True
                break

            preview_parts.append(msg_preview)
            char_count += len(msg_preview) + 1  # +1 for newline
            included_count += 1

        return {
            "success": True,
            "stash_id": stash_id,
            "stash_name": stash_name,
            "total_messages": len(stashed),
            "preview_messages": included_count,
            "truncated": truncated,
            "preview": "\n".join(preview_parts)
        }

    # =========================================================================
    # RLM-Inspired Hierarchical Compaction
    # =========================================================================

    async def hierarchical_compact(
        self,
        llm_service,
        counter,
        chunk_size: int = 4000,
        preserve_recent: int = 5,
        max_depth: int = 3
    ) -> Dict[str, Any]:
        """
        Hierarchical compaction using RLM-style recursive summarization.

        Instead of flat linear compaction, this:
        1. Splits messages into chunks
        2. Recursively summarizes each chunk
        3. Merges summaries into higher-level summaries
        4. Preserves more structure than flat compaction

        This is inspired by the RLM paper's approach to handling long context
        through recursive sub-queries.

        Args:
            llm_service: LLM service for generating summaries
            counter: TokenCounter for counting tokens
            chunk_size: Target characters per chunk
            preserve_recent: Messages to keep verbatim
            max_depth: Maximum recursion depth

        Returns:
            Result dict with compaction stats
        """
        # Get conversation history
        history = []
        conv_store = self._get_conversation_store()
        if conv_store:
            history = await conv_store.get_full_history()

        message_count = len(history)

        if message_count <= preserve_recent + 3:
            return {
                "success": False,
                "reason": "Not enough messages to compact",
                "message_count": message_count
            }

        # Split into messages to compact vs preserve
        to_compact = history[:-preserve_recent] if preserve_recent > 0 else history
        to_preserve = history[-preserve_recent:] if preserve_recent > 0 else []

        if len(to_compact) < 4:
            return {
                "success": False,
                "reason": "Not enough older messages for hierarchical compaction",
                "message_count": len(to_compact)
            }

        # Count tokens before
        tokens_before = sum(
            counter.count(m.get("content", ""))
            for m in to_compact
        )

        # Build text chunks
        chunks = self._build_message_chunks(to_compact, chunk_size)

        if len(chunks) < 2:
            # Fall back to regular compaction - not implemented here, would need ConversationManager
            return {
                "success": False,
                "reason": "Not enough chunks for hierarchical compaction",
                "chunks_count": len(chunks)
            }

        logger.info(f"Hierarchical compaction: {len(to_compact)} messages → {len(chunks)} chunks")

        try:
            # Recursive summarization
            final_summary = await self._recursive_summarize(
                llm_service=llm_service,
                chunks=chunks,
                depth=0,
                max_depth=max_depth
            )

            tokens_after = counter.count(final_summary)
            tokens_saved = tokens_before - tokens_after

            # Store compaction result
            from datetime import datetime, timezone
            compaction_marker = {
                "role": "system",
                "content": f"[HIERARCHICAL COMPACTION - {len(to_compact)} messages, {len(chunks)} chunks]\n\n{final_summary}",
                "metadata": {
                    "type": "hierarchical_compaction",
                    "messages_compacted": len(to_compact),
                    "chunks_processed": len(chunks),
                    "tokens_before": tokens_before,
                    "tokens_after": tokens_after,
                    "compacted_at": datetime.now(timezone.utc).isoformat()
                }
            }

            # Store via conversation store
            if conv_store:
                await conv_store.add_conversation(
                    role="system",
                    content=compaction_marker["content"],
                    metadata=compaction_marker["metadata"]
                )

            logger.info(
                f"Hierarchical compaction complete: {len(to_compact)} messages → summary, "
                f"saved {tokens_saved} tokens ({tokens_before} → {tokens_after})"
            )

            return {
                "success": True,
                "messages_compacted": len(to_compact),
                "messages_preserved": len(to_preserve),
                "chunks_processed": len(chunks),
                "tokens_before": tokens_before,
                "tokens_after": tokens_after,
                "tokens_saved": tokens_saved,
                "summary_preview": final_summary[:300] + "..." if len(final_summary) > 300 else final_summary
            }

        except (ConnectionError, TimeoutError) as e:
            logger.error(f"Network error during hierarchical compaction: {e}", exc_info=True)
            return {"success": False, "error": f"Network error: {str(e)}"}
        except (KeyError, ValueError, TypeError) as e:
            logger.error(f"Data error during hierarchical compaction: {e}", exc_info=True)
            return {"success": False, "error": f"Data error: {str(e)}"}
        except Exception as e:
            logger.error(f"Hierarchical compaction failed: {e}", exc_info=True)
            return {"success": False, "error": str(e)}

    def _build_message_chunks(
        self,
        messages: List[Dict],
        chunk_size: int
    ) -> List[str]:
        """Split messages into character-sized chunks."""
        chunks = []
        current_chunk = []
        current_size = 0

        for msg in messages:
            role = msg.get("role", "user").upper()
            content = msg.get("content", "")
            msg_text = f"{role}: {content}\n"
            msg_size = len(msg_text)

            if current_size + msg_size > chunk_size and current_chunk:
                # Start new chunk
                chunks.append("\n".join(current_chunk))
                current_chunk = [msg_text]
                current_size = msg_size
            else:
                current_chunk.append(msg_text)
                current_size += msg_size

        # Add final chunk
        if current_chunk:
            chunks.append("\n".join(current_chunk))

        return chunks

    async def _recursive_summarize(
        self,
        llm_service,
        chunks: List[str],
        depth: int,
        max_depth: int
    ) -> str:
        """
        Recursively summarize chunks using tree-structured approach.

        At each level:
        1. Summarize each chunk individually
        2. If multiple summaries remain and depth < max_depth, recurse
        3. Otherwise merge remaining summaries
        """
        if depth >= max_depth or len(chunks) <= 1:
            # Base case: final merge
            if len(chunks) == 1:
                return await self._summarize_chunk(llm_service, chunks[0])
            else:
                combined = "\n---\n".join(chunks)
                return await self._summarize_chunk(llm_service, combined)

        # Summarize each chunk
        summaries = []
        for i, chunk in enumerate(chunks):
            logger.debug(f"Summarizing chunk {i+1}/{len(chunks)} at depth {depth}")
            summary = await self._summarize_chunk(llm_service, chunk)
            summaries.append(summary)

        # If we have many summaries, recurse to merge them
        if len(summaries) > 2:
            # Group summaries into pairs and recurse
            paired_chunks = []
            for i in range(0, len(summaries), 2):
                if i + 1 < len(summaries):
                    paired_chunks.append(f"{summaries[i]}\n---\n{summaries[i+1]}")
                else:
                    paired_chunks.append(summaries[i])

            return await self._recursive_summarize(
                llm_service, paired_chunks, depth + 1, max_depth
            )
        else:
            # Final merge
            combined = "\n---\n".join(summaries)
            return await self._summarize_chunk(llm_service, combined)

    async def _summarize_chunk(self, llm_service, chunk: str) -> str:
        """Generate a summary for a single chunk."""
        prompt = f"""Summarize this conversation segment concisely, preserving:
- Key facts, decisions, and conclusions
- Important context for continuing the conversation
- Commitments and requests mentioned
- Emotional tone

Write a direct summary (no meta-commentary).

CONVERSATION SEGMENT:
{chunk}

SUMMARY:"""

        response = await llm_service.generate(
            prompt=prompt,
            system_prompt="You are a conversation summarizer. Create concise summaries.",
            model_override=None
        )

        return response.strip() if isinstance(response, str) else str(response)

    def _get_conversation_store(self):
        """Get the conversation store from storage hierarchy."""
        if hasattr(self.storage, 'conversation'):
            return self.storage.conversation
        elif hasattr(self.storage, '_storage') and hasattr(self.storage._storage, 'conversation'):
            return self.storage._storage.conversation
        return None

    async def _log_context_audit(
        self,
        action: str,
        message_ids: List[int],
        reason: str
    ) -> None:
        """Log context management operations for audit trail."""
        from datetime import datetime, timezone

        conv_store = self._get_conversation_store()
        if not conv_store:
            return

        audit_entry = {
            "type": "context_audit",
            "action": action,
            "message_ids": message_ids,
            "reason": reason,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

        await conv_store.add_conversation(
            role="system",
            content=f"[CONTEXT_AUDIT] {action}: {len(message_ids)} messages - {reason}",
            metadata=audit_entry
        )
