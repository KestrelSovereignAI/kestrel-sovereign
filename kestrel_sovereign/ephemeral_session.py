#!/usr/bin/env python3
"""
Ephemeral Session Handler for Kestrel Privacy System.
In-memory only session that stores NOTHING to disk.
"""

from datetime import datetime, timezone
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)

class EphemeralSession:
    """
    In-memory session for EPHEMERAL privacy mode.
    Stores conversation history in memory only, with a limited buffer size.
    All data is lost when the session ends.
    """

    def __init__(self, max_messages: int = 50):
        """
        Initialize an ephemeral session.

        Args:
            max_messages: Maximum number of messages to keep in memory buffer
        """
        self.messages: List[Dict] = []
        self.context: Dict = {}
        self.max_messages = max_messages
        self.created_at = datetime.now(timezone.utc)
        logger.info(f"Ephemeral session created at {self.created_at}")

    def add_message(self, role: str, content: str, metadata: Optional[Dict] = None,
                    rendered_content: Optional[str] = None):
        """
        Add a message to the in-memory buffer.
        Old messages are automatically removed if buffer is full.

        Args:
            role: Message role (user/assistant/system)
            content: Canonical raw message content
            metadata: Optional metadata (not persisted)
            rendered_content: Byte-stable transport form (#1402). Replayed
                verbatim by ``format_conversation_history`` so prompt-cache
                prefixes stay stable across turns within the ephemeral
                session.
        """
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "metadata": metadata or {}
        }
        if rendered_content is not None:
            message["rendered_content"] = rendered_content

        self.messages.append(message)

        # Limit buffer size - remove oldest messages
        if len(self.messages) > self.max_messages:
            removed = self.messages.pop(0)
            logger.debug(f"Removed oldest message from ephemeral buffer: {removed['timestamp']}")

    def get_context(self, limit: int = 10) -> str:
        """
        Get recent conversation context for LLM prompting.

        Args:
            limit: Number of recent messages to include

        Returns:
            Formatted conversation history string
        """
        recent_messages = self.messages[-limit:] if len(self.messages) > limit else self.messages

        context_lines = []
        for msg in recent_messages:
            context_lines.append(f"{msg['role']}: {msg['content']}")

        return "\n".join(context_lines)

    def get_history(self, limit: int = 100) -> List[Dict]:
        """
        Get conversation history from in-memory buffer.

        Args:
            limit: Maximum number of messages to return

        Returns:
            List of message dictionaries
        """
        return self.messages[-limit:] if len(self.messages) > limit else self.messages.copy()

    def clear(self):
        """
        Clear all session data from memory.
        This is called when the session ends.
        """
        message_count = len(self.messages)
        self.messages.clear()
        self.context.clear()

        logger.info(f"Ephemeral session cleared. {message_count} messages deleted from memory.")

    def get_stats(self) -> Dict:
        """
        Get session statistics.

        Returns:
            Dictionary with session stats
        """
        return {
            "message_count": len(self.messages),
            "created_at": self.created_at.isoformat(),
            "max_buffer_size": self.max_messages,
            "storage_mode": "in-memory only",
            "persistent_storage": False
        }

    def __del__(self):
        """Destructor - ensure data is cleared when object is destroyed"""
        if self.messages:
            logger.warning("Ephemeral session destroyed with data still in memory. Clearing...")
            self.clear()
