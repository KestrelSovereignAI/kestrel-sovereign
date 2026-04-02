"""
Conversation Manager for Kestrel Agent.

Handles conversation state, history retrieval, and message management operations.
Extracted from ContextManager to improve modularity and maintainability.
"""

import logging
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from kestrel_sovereign.storage import AsyncStorage

logger = logging.getLogger(__name__)


class ConversationManager:
    """
    Manages conversation state and history operations.

    Responsibilities:
    1. Retrieve conversation history from storage
    2. Format conversation messages for context
    3. Handle message selection and filtering
    4. Manage message metadata (protect, exclude, restore)
    5. Compression and summarization of conversation history
    """

    def __init__(
        self,
        storage: "AsyncStorage",
        agent_id: Optional[str] = None,
    ):
        """
        Initialize the conversation manager.

        Args:
            storage: AsyncStorage instance for conversation operations
            agent_id: Agent ID for scoped queries
        """
        self.storage = storage
        self.agent_id = agent_id

    async def get_conversation_history(
        self, session_id: Optional[str] = None, limit: int = 100
    ) -> List[Dict]:
        """Get conversation history from storage.

        Uses session-aware, limited retrieval that filters out excluded messages
        (compressed, summarized, etc.).

        Args:
            session_id: Optional session ID to filter by specific session.
            limit: Maximum number of messages to return.
        """
        try:
            if hasattr(self.storage, 'conversation'):
                return await self.storage.conversation.get_conversation_history(
                    limit, session_id=session_id
                )
            elif hasattr(self.storage, 'get_conversation_history'):
                return await self.storage.get_conversation_history(
                    limit, session_id=session_id
                )
            else:
                logger.warning("No conversation history method available")
                return []
        except (ConnectionError, TimeoutError) as e:
            logger.error(f"Storage connection error while fetching conversation history: {e}", exc_info=True)
            return []
        except (KeyError, ValueError, TypeError) as e:
            logger.error(f"Data error while fetching conversation history: {e}", exc_info=True)
            return []
        except Exception as e:
            logger.error(f"Failed to get conversation history: {e}", exc_info=True)
            return []

    async def compress_session(
        self,
        llm_service,
        counter,
        preserve_recent: int = 10,
        force: bool = False
    ) -> Dict[str, Any]:
        """
        Compress the current session by summarizing older messages.

        This is an in-session compression that:
        1. Takes all messages except the most recent N
        2. Uses the LLM to create a summary
        3. Replaces older messages with the summary
        4. Preserves recent messages verbatim

        Args:
            llm_service: LLM service for generating summary
            counter: TokenCounter for token counting
            preserve_recent: Number of recent messages to keep verbatim
            force: Compress even if utilization is low

        Returns:
            Dict with compression results (messages_compressed, tokens_saved, etc.)
        """
        # Compression needs full unfiltered history to see what to compress
        conv_store = self._get_conversation_store()
        if conv_store:
            history = await conv_store.get_full_history()
        else:
            history = await self.get_conversation_history()
        message_count = len(history)

        # Check if compression is needed
        if not force and message_count <= preserve_recent + 5:
            return {
                "success": False,
                "reason": "Not enough messages to compress",
                "message_count": message_count
            }

        # Messages to compress (older) vs preserve (recent)
        messages_to_compress = history[:-preserve_recent] if preserve_recent > 0 else history
        messages_to_preserve = history[-preserve_recent:] if preserve_recent > 0 else []

        if len(messages_to_compress) < 3:
            return {
                "success": False,
                "reason": "Not enough older messages to compress",
                "message_count": message_count
            }

        # Count tokens in messages to compress
        tokens_before = sum(
            counter.count(m.get("content", ""))
            for m in messages_to_compress
        )

        # Build summary prompt
        conversation_text = "\n".join([
            f"{m.get('role', 'user').upper()}: {m.get('content', '')}"
            for m in messages_to_compress
        ])

        summary_prompt = f"""Summarize this conversation segment concisely, preserving:
- Key facts, decisions, and conclusions reached
- Important context needed to continue the conversation
- Any commitments, preferences, or requests mentioned
- The emotional tone and relationship context

Do NOT include meta-commentary. Write as a direct summary that could replace this conversation segment.

CONVERSATION:
{conversation_text}

SUMMARY:"""

        try:
            # Generate summary using LLM
            summary_response = await llm_service.generate(
                prompt=summary_prompt,
                system_prompt="You are a conversation summarizer. Create concise summaries that preserve essential context.",
                model_override=None  # Use default model
            )

            summary_text = summary_response.strip() if isinstance(summary_response, str) else str(summary_response)

            # Count tokens saved
            tokens_after = counter.count(summary_text)
            tokens_saved = tokens_before - tokens_after

            # Collect original message IDs for transcript reference
            original_message_ids = [m.get("id") for m in messages_to_compress if m.get("id")]
            first_id = original_message_ids[0] if original_message_ids else None
            last_id = original_message_ids[-1] if original_message_ids else None

            # Store the compression in the database
            # Create a summary message that replaces the compressed portion
            # Note: Transcript reference is in metadata only, not in content (LLM context)
            compression_marker = {
                "role": "system",
                "content": f"[COMPRESSED CONTEXT - {len(messages_to_compress)} messages summarized]\n\n{summary_text}",
                "metadata": {
                    "type": "compression",
                    "messages_compressed": len(messages_to_compress),
                    "tokens_before": tokens_before,
                    "tokens_after": tokens_after,
                    "compressed_at": datetime.now(timezone.utc).isoformat(),
                    "original_message_ids": original_message_ids,
                    "message_range": {
                        "first": first_id,
                        "last": last_id
                    }
                }
            }

            # Store the compression result
            # Add the compression marker to conversation
            conv_store = self._get_conversation_store()
            if conv_store:
                await conv_store.add_conversation(
                    role="system",
                    content=compression_marker["content"],
                    metadata=compression_marker["metadata"]
                )

                # Get the ID of the compression marker we just created
                # We need this to populate summarized_into on original messages
                # Query for just the last ID instead of full history (performance fix)
                compression_marker_id = None
                if hasattr(conv_store, 'db'):
                    row = await conv_store.db.fetchone(
                        "SELECT id FROM conversation_history WHERE agent_id = ? ORDER BY id DESC LIMIT 1",
                        (conv_store.agent_id,)
                    )
                    compression_marker_id = row[0] if row else None

                # Mark original messages as excluded and link to compression marker
                if compression_marker_id and original_message_ids:
                    now = datetime.now(timezone.utc).isoformat()
                    exclusion_metadata = {
                        "excluded_from_context": True,
                        "excluded_at": now,
                        "excluded_reason": "Replaced by compression",
                        "summarized_into": str(compression_marker_id)
                    }
                    await conv_store.update_messages_metadata(
                        original_message_ids,
                        exclusion_metadata
                    )

            logger.info(
                f"Session compressed: {len(messages_to_compress)} messages → summary, "
                f"saved {tokens_saved} tokens ({tokens_before} → {tokens_after})"
            )

            return {
                "success": True,
                "messages_compressed": len(messages_to_compress),
                "messages_preserved": len(messages_to_preserve),
                "tokens_before": tokens_before,
                "tokens_after": tokens_after,
                "tokens_saved": tokens_saved,
                "summary_preview": summary_text[:200] + "..." if len(summary_text) > 200 else summary_text
            }

        except (ConnectionError, TimeoutError) as e:
            logger.error(f"Network error during session compression: {e}", exc_info=True)
            return {
                "success": False,
                "reason": f"Network error during compression: {str(e)}",
                "message_count": message_count
            }
        except (KeyError, ValueError, TypeError) as e:
            logger.error(f"Data error during session compression: {e}", exc_info=True)
            return {
                "success": False,
                "reason": f"Data error during compression: {str(e)}",
                "message_count": message_count
            }
        except Exception as e:
            logger.error(f"Session compression failed: {e}", exc_info=True)
            return {
                "success": False,
                "reason": f"Compression failed: {str(e)}",
                "message_count": message_count
            }

    async def check_compression_needed(self, counter, model: str, utilization_threshold: float = 70.0, history: Optional[list] = None) -> Dict[str, Any]:
        """
        Check if session compression is recommended.

        Args:
            counter: TokenCounter for token counting
            model: Model name for budget calculation
            utilization_threshold: Percentage at which compression is recommended
            history: Pre-fetched session-filtered history (same as LLM sees).
                     If None, fetches limited history to approximate LLM path.

        Returns:
            Dict with recommendation and current stats
        """
        from .token_budget import create_budget

        # Use same data path as LLM — session-filtered, limited
        if history is None:
            history = await self.get_conversation_history(limit=50)
        message_count = len(history)
        budget = create_budget(model, message_count, adaptive=True)

        # Calculate against the history allocation, not total budget
        total_tokens = sum(
            counter.count(m.get("content", ""))
            for m in history
        )
        history_utilization = (total_tokens / budget.history * 100) if budget.history > 0 else 0

        return {
            "compression_recommended": history_utilization >= utilization_threshold,
            "utilization_percent": round(history_utilization, 1),
            "message_count": message_count,
            "total_tokens": total_tokens,
            "budget_limit": budget.history,
            "threshold": utilization_threshold,
            "note": "Utilization measured against history allocation, not total context budget."
        }

    async def get_messages_for_selection(
        self,
        mode: str,
        criteria: str,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Select messages based on mode and criteria.

        Args:
            mode: Selection mode - "time_range", "topic", "messages", "last_n"
            criteria: Selection criteria based on mode
            limit: Maximum messages to return

        Returns:
            List of matching messages with IDs
        """
        conv_store = self._get_conversation_store()
        if not conv_store:
            return []

        if mode == "messages":
            # Direct message ID selection
            try:
                message_ids = [int(x.strip()) for x in criteria.split(",")]
                return await conv_store.get_messages_by_ids(message_ids)
            except ValueError:
                logger.error(f"Invalid message IDs: {criteria}")
                return []

        elif mode == "last_n":
            # Get last N messages
            try:
                n = int(criteria)
                all_messages = await conv_store.get_full_history_with_ids()
                return all_messages[-n:] if len(all_messages) >= n else all_messages
            except ValueError:
                logger.error(f"Invalid count: {criteria}")
                return []

        elif mode == "topic":
            # Semantic search by topic/content
            return await conv_store.search_messages_by_content(criteria, limit)

        elif mode == "time_range":
            # Time-based selection
            all_messages = await conv_store.get_full_history_with_ids()

            now = datetime.now(timezone.utc)
            if criteria == "before_today":
                cutoff = now.replace(hour=0, minute=0, second=0, microsecond=0)
            elif criteria.startswith("last_"):
                # Parse "last_2_hours", "last_1_day", etc.
                parts = criteria.split("_")
                if len(parts) >= 3:
                    try:
                        amount = int(parts[1])
                        unit = parts[2]
                        if unit in ("hour", "hours"):
                            cutoff = now - timedelta(hours=amount)
                        elif unit in ("day", "days"):
                            cutoff = now - timedelta(days=amount)
                        elif unit in ("minute", "minutes"):
                            cutoff = now - timedelta(minutes=amount)
                        else:
                            cutoff = now - timedelta(hours=1)
                    except ValueError:
                        cutoff = now - timedelta(hours=1)
                else:
                    cutoff = now - timedelta(hours=1)
            else:
                # Try parsing as date range "YYYY-MM-DD..YYYY-MM-DD"
                if ".." in criteria:
                    try:
                        start_str, end_str = criteria.split("..")
                        start = datetime.fromisoformat(start_str)
                        end = datetime.fromisoformat(end_str)
                        # Return messages in range
                        return [
                            m for m in all_messages
                            if self._message_in_range(m, start, end)
                        ]
                    except ValueError:
                        pass
                # Default to last hour
                cutoff = now - timedelta(hours=1)

            # Return messages before cutoff (for "before_today") or after (for "last_X")
            if criteria == "before_today":
                return [m for m in all_messages if self._message_before(m, cutoff)]
            else:
                return [m for m in all_messages if self._message_after(m, cutoff)]

        return []

    def _get_conversation_store(self):
        """Get the conversation store from storage hierarchy."""
        if hasattr(self.storage, 'conversation'):
            return self.storage.conversation
        elif hasattr(self.storage, '_storage') and hasattr(self.storage._storage, 'conversation'):
            return self.storage._storage.conversation
        return None

    def _message_before(self, msg: Dict, cutoff: 'datetime') -> bool:
        """Check if message timestamp is before cutoff."""
        created_at = msg.get("created_at")
        if not created_at:
            return False
        if isinstance(created_at, str):
            try:
                ts = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
            except ValueError:
                return False
        else:
            ts = created_at
        # Make cutoff timezone-aware if needed
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        if cutoff.tzinfo is None:
            cutoff = cutoff.replace(tzinfo=timezone.utc)
        return ts < cutoff

    def _message_after(self, msg: Dict, cutoff: 'datetime') -> bool:
        """Check if message timestamp is after cutoff."""
        created_at = msg.get("created_at")
        if not created_at:
            return True  # Include messages without timestamp by default
        if isinstance(created_at, str):
            try:
                ts = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
            except ValueError:
                return True
        else:
            ts = created_at
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        if cutoff.tzinfo is None:
            cutoff = cutoff.replace(tzinfo=timezone.utc)
        return ts >= cutoff

    def _message_in_range(self, msg: Dict, start: 'datetime', end: 'datetime') -> bool:
        """Check if message timestamp is in range."""
        created_at = msg.get("created_at")
        if not created_at:
            return False
        if isinstance(created_at, str):
            try:
                ts = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
            except ValueError:
                return False
        else:
            ts = created_at
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        if start.tzinfo is None:
            start = start.replace(tzinfo=timezone.utc)
        if end.tzinfo is None:
            end = end.replace(tzinfo=timezone.utc)
        return start <= ts <= end

    async def mark_messages(
        self,
        message_ids: List[int],
        action: str,
        reason: str = ""
    ) -> Dict[str, Any]:
        """
        Mark messages for context management.

        Args:
            message_ids: List of message IDs to mark
            action: "protect", "droppable", or "clear"
            reason: Optional reason for marking

        Returns:
            Result dict with success status and count
        """
        conv_store = self._get_conversation_store()
        if not conv_store:
            return {"success": False, "error": "Conversation store not available"}

        # Check for protected messages that cannot be modified
        messages = await conv_store.get_messages_by_ids(message_ids)
        protected_ids = []
        valid_ids = []

        for msg in messages:
            meta = msg.get("metadata", {})
            # User-protected messages cannot be marked as droppable
            if action == "droppable" and meta.get("decay_protected"):
                protected_ids.append(msg["id"])
            else:
                valid_ids.append(msg["id"])

        if not valid_ids:
            return {
                "success": False,
                "error": "No valid messages to mark",
                "protected_count": len(protected_ids)
            }

        # Prepare metadata update
        if action == "protect":
            metadata_update = {"context_priority": "protected"}
        elif action == "droppable":
            metadata_update = {"context_priority": "droppable"}
        elif action == "clear":
            metadata_update = {"context_priority": None}
        else:
            return {"success": False, "error": f"Invalid action: {action}"}

        # Update messages
        updated = await conv_store.update_messages_metadata(valid_ids, metadata_update)

        # Log audit trail
        await self._log_context_audit(
            action=f"mark_{action}",
            message_ids=valid_ids,
            reason=reason
        )

        return {
            "success": True,
            "marked_count": updated,
            "protected_count": len(protected_ids),
            "action": action,
            "reason": reason
        }

    async def exclude_messages(
        self,
        message_ids: List[int],
        reason: str
    ) -> Dict[str, Any]:
        """
        Exclude messages from context (soft removal).

        Args:
            message_ids: List of message IDs to exclude
            reason: Required reason for exclusion

        Returns:
            Result dict with success status and count
        """
        conv_store = self._get_conversation_store()
        if not conv_store:
            return {"success": False, "error": "Conversation store not available"}

        # Check for protected messages
        messages = await conv_store.get_messages_by_ids(message_ids)
        protected_ids = []
        valid_ids = []

        for msg in messages:
            meta = msg.get("metadata", {})
            # Cannot exclude protected messages
            if meta.get("context_priority") == "protected" or meta.get("decay_protected"):
                protected_ids.append(msg["id"])
            else:
                valid_ids.append(msg["id"])

        if not valid_ids:
            return {
                "success": False,
                "error": "No valid messages to exclude (all protected)",
                "protected_count": len(protected_ids)
            }

        # Prepare metadata update
        now = datetime.now(timezone.utc).isoformat()
        metadata_update = {
            "excluded_from_context": True,
            "excluded_at": now,
            "excluded_reason": reason
        }

        # Update messages
        updated = await conv_store.update_messages_metadata(valid_ids, metadata_update)

        # Log audit trail
        await self._log_context_audit(
            action="exclude",
            message_ids=valid_ids,
            reason=reason
        )

        return {
            "success": True,
            "excluded_count": updated,
            "protected_count": len(protected_ids),
            "reason": reason
        }

    async def restore_messages(
        self,
        message_ids: Optional[List[int]] = None
    ) -> Dict[str, Any]:
        """
        Restore excluded messages back to context.

        Args:
            message_ids: Specific IDs to restore, or None for all excluded

        Returns:
            Result dict with success status and count
        """
        conv_store = self._get_conversation_store()
        if not conv_store:
            return {"success": False, "error": "Conversation store not available"}

        # If no IDs specified, restore all excluded
        if message_ids is None:
            excluded = await conv_store.get_excluded_messages(limit=1000)
            message_ids = [m["id"] for m in excluded]

        if not message_ids:
            return {"success": True, "restored_count": 0, "note": "No excluded messages found"}

        # Clear exclusion metadata
        metadata_update = {
            "excluded_from_context": False,
            "excluded_at": None,
            "excluded_reason": None
        }

        updated = await conv_store.update_messages_metadata(message_ids, metadata_update)

        # Log audit trail
        await self._log_context_audit(
            action="restore",
            message_ids=message_ids,
            reason="Restored to context"
        )

        return {
            "success": True,
            "restored_count": updated
        }

    async def summarize_messages(
        self,
        llm_service,
        counter,
        message_ids: List[int],
        preserve_key_facts: bool = True
    ) -> Dict[str, Any]:
        """
        Summarize specific messages and replace them with a summary.

        Args:
            llm_service: LLM service for generating summary
            counter: TokenCounter for token counting
            message_ids: Message IDs to summarize
            preserve_key_facts: Keep facts, decisions, commitments in summary

        Returns:
            Result dict with summary and stats
        """
        conv_store = self._get_conversation_store()
        if not conv_store:
            return {"success": False, "error": "Conversation store not available"}

        # Get messages to summarize
        messages = await conv_store.get_messages_by_ids(message_ids)

        if len(messages) < 2:
            return {"success": False, "error": "Need at least 2 messages to summarize"}

        # Filter out protected messages
        protected_ids = []
        summarizable = []
        for msg in messages:
            meta = msg.get("metadata", {})
            if meta.get("context_priority") == "protected" or meta.get("decay_protected"):
                protected_ids.append(msg["id"])
            else:
                summarizable.append(msg)

        if len(summarizable) < 2:
            return {
                "success": False,
                "error": "Not enough non-protected messages to summarize",
                "protected_count": len(protected_ids)
            }

        # Count tokens before
        tokens_before = sum(
            counter.count(m.get("content", ""))
            for m in summarizable
        )

        # Build summary prompt
        conversation_text = "\n".join([
            f"{m.get('role', 'user').upper()}: {m.get('content', '')}"
            for m in summarizable
        ])

        key_facts_instruction = ""
        if preserve_key_facts:
            key_facts_instruction = """
Preserve:
- Key facts, decisions, and conclusions reached
- Important context needed to continue the conversation
- Any commitments, preferences, or requests mentioned
- The emotional tone and relationship context
"""

        summary_prompt = f"""Summarize this conversation segment concisely.
{key_facts_instruction}
Do NOT include meta-commentary. Write as a direct summary that could replace this conversation segment.

CONVERSATION:
{conversation_text}

SUMMARY:"""

        try:
            # Generate summary
            summary_response = await llm_service.generate(
                prompt=summary_prompt,
                system_prompt="You are a conversation summarizer. Create concise summaries that preserve essential context.",
                model_override=None
            )

            summary_text = summary_response.strip() if isinstance(summary_response, str) else str(summary_response)
            tokens_after = counter.count(summary_text)

            # Collect original message IDs for transcript reference
            original_message_ids = [m["id"] for m in summarizable]
            first_id = original_message_ids[0] if original_message_ids else None
            last_id = original_message_ids[-1] if original_message_ids else None

            # Create summary message with transcript reference
            # Note: Transcript reference is in metadata only, not in content (LLM context)
            now = datetime.now(timezone.utc).isoformat()
            summary_meta = {
                "type": "context_summary",
                "summarized_message_ids": original_message_ids,
                "original_message_ids": original_message_ids,
                "tokens_before": tokens_before,
                "tokens_after": tokens_after,
                "created_at": now,
                "message_range": {
                    "first": first_id,
                    "last": last_id
                }
            }

            # Add summary to conversation
            await conv_store.add_conversation(
                role="system",
                content=f"[SUMMARY of {len(summarizable)} messages]\n\n{summary_text}",
                metadata=summary_meta
            )

            # Get the ID of the summary marker we just created
            # Query for just the last ID instead of full history (performance fix)
            summary_marker_id = None
            if hasattr(conv_store, 'db'):
                row = await conv_store.db.fetchone(
                    "SELECT id FROM conversation_history WHERE agent_id = ? ORDER BY id DESC LIMIT 1",
                    (conv_store.agent_id,)
                )
                summary_marker_id = row[0] if row else None

            # Mark original messages as summarized and link to summary marker
            summary_update = {
                "summarized": True,
                "excluded_from_context": True,
                "excluded_at": now,
                "excluded_reason": "Replaced by summary",
                "summarized_into": str(summary_marker_id) if summary_marker_id else None
            }
            await conv_store.update_messages_metadata(
                original_message_ids,
                summary_update
            )

            # Log audit
            await self._log_context_audit(
                action="summarize",
                message_ids=[m["id"] for m in summarizable],
                reason=f"Summarized {len(summarizable)} messages, saved {tokens_before - tokens_after} tokens"
            )

            return {
                "success": True,
                "messages_summarized": len(summarizable),
                "protected_count": len(protected_ids),
                "tokens_before": tokens_before,
                "tokens_after": tokens_after,
                "tokens_saved": tokens_before - tokens_after,
                "summary_preview": summary_text[:200] + "..." if len(summary_text) > 200 else summary_text
            }

        except (ConnectionError, TimeoutError) as e:
            logger.error(f"Network error during message summarization: {e}", exc_info=True)
            return {"success": False, "error": f"Network error: {str(e)}"}
        except (KeyError, ValueError, TypeError) as e:
            logger.error(f"Data error during message summarization: {e}", exc_info=True)
            return {"success": False, "error": f"Data error: {str(e)}"}
        except Exception as e:
            logger.error(f"Summarization failed: {e}", exc_info=True)
            return {"success": False, "error": str(e)}

    async def _log_context_audit(
        self,
        action: str,
        message_ids: List[int],
        reason: str
    ) -> None:
        """Log context management operations for audit trail."""
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