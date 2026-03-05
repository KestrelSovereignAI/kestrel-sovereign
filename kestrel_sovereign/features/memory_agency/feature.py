"""
Memory Agency -- agent-controlled memory pinning and release.

Allows the agent to actively participate in its own memory by:
- Pinning important memories (resist Ebbinghaus decay)
- Releasing memories it wants to let go of
- Listing currently pinned memories
- Viewing pin statistics

Pinned memories get ``decay_protected = True`` in their metadata,
which the MemoryConsolidator already respects (skips archival) and
the MemoryRetriever boosts in scoring.
"""

import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict

from kestrel_sovereign.features.base import Feature, tool
from kestrel_sovereign.tools.base import ToolCategory

logger = logging.getLogger(__name__)


class MemoryAgencyFeature(Feature):
    """
    Memory agency tools for the agent.

    Provides tools for:
    - Pinning a memory to protect it from decay
    - Releasing a previously pinned memory
    - Listing all active pins
    - Viewing pin statistics
    """

    @property
    def tool_description(self) -> str:
        return (
            "Pin and release memories -- protect important memories from "
            "decay or let go of ones no longer needed."
        )

    async def initialize(self):
        """Initialize the memory agency feature and create the memory_pins table."""
        self.storage = self.agent.storage
        self.agent_id = (
            getattr(self.storage, "agent_id", "")
            or getattr(getattr(self.storage, "_storage", None), "agent_id", "")
        )

        # Create the memory_pins tracking table
        await self.storage.db.execute(
            """CREATE TABLE IF NOT EXISTS memory_pins (
                id TEXT PRIMARY KEY,
                message_id INTEGER NOT NULL,
                agent_id TEXT,
                pin_reason TEXT DEFAULT '',
                pinned_at TEXT NOT NULL,
                released_at TEXT
            )"""
        )

        logger.info(
            "MemoryAgencyFeature initialized for agent: %s...",
            self.agent_id[:30] if self.agent_id else "(none)",
        )

    # ------------------------------------------------------------------
    # Tools
    # ------------------------------------------------------------------

    @tool(
        name="memory_pin",
        description=(
            "Pin a memory so it resists decay and stays retrievable. "
            "Use this for memories the agent considers important to preserve."
        ),
        category=ToolCategory.MEMORY,
        command_prefix="!memory-pin",
    )
    async def memory_pin(self, message_id: int, reason: str = "") -> Dict[str, Any]:
        """
        Pin a conversation message to protect it from memory decay.

        Sets ``decay_protected = True`` in the message metadata and records
        the pin in the memory_pins table.

        Args:
            message_id: The database ID of the message to pin
            reason: Optional reason for pinning this memory
        """
        db = self.storage.db

        # Fetch the message
        row = await db.fetchone(
            "SELECT id, content, metadata FROM conversation_history WHERE id = ?",
            (message_id,),
        )
        if not row:
            return {"error": f"Message {message_id} not found"}

        msg_id, content, raw_metadata = row

        # Parse metadata
        metadata = {}
        if raw_metadata:
            try:
                metadata = json.loads(raw_metadata) if isinstance(raw_metadata, str) else raw_metadata
            except (json.JSONDecodeError, TypeError):
                metadata = {}

        # Set decay_protected
        metadata["decay_protected"] = True

        # Write metadata back
        await db.execute(
            "UPDATE conversation_history SET metadata = ? WHERE id = ?",
            (json.dumps(metadata), msg_id),
        )

        # Record in memory_pins (idempotent -- check for existing active pin)
        existing = await db.fetchone(
            "SELECT id FROM memory_pins WHERE message_id = ? AND released_at IS NULL",
            (message_id,),
        )
        if not existing:
            pin_id = uuid.uuid4().hex[:12]
            await db.execute(
                """INSERT INTO memory_pins (id, message_id, agent_id, pin_reason, pinned_at)
                   VALUES (?, ?, ?, ?, ?)""",
                (
                    pin_id,
                    message_id,
                    self.agent_id,
                    reason,
                    datetime.now(timezone.utc).isoformat(),
                ),
            )

        preview = (content[:80] + "...") if len(content) > 80 else content
        return {
            "pinned": True,
            "message_id": message_id,
            "preview": preview,
            "reason": reason,
        }

    @tool(
        name="memory_release",
        description=(
            "Release a pinned memory so it resumes normal decay. "
            "Use this when a previously important memory is no longer needed."
        ),
        category=ToolCategory.MEMORY,
        command_prefix="!memory-release",
    )
    async def memory_release(self, message_id: int) -> Dict[str, Any]:
        """
        Release a pinned memory, allowing it to decay normally again.

        Clears ``decay_protected`` from the message metadata and marks
        the pin record with a ``released_at`` timestamp.

        Args:
            message_id: The database ID of the message to release
        """
        db = self.storage.db

        # Fetch the message
        row = await db.fetchone(
            "SELECT id, metadata FROM conversation_history WHERE id = ?",
            (message_id,),
        )
        if not row:
            return {"error": f"Message {message_id} not found"}

        msg_id, raw_metadata = row

        # Parse and update metadata
        metadata = {}
        if raw_metadata:
            try:
                metadata = json.loads(raw_metadata) if isinstance(raw_metadata, str) else raw_metadata
            except (json.JSONDecodeError, TypeError):
                metadata = {}

        metadata["decay_protected"] = False

        await db.execute(
            "UPDATE conversation_history SET metadata = ? WHERE id = ?",
            (json.dumps(metadata), msg_id),
        )

        # Update memory_pins record
        now = datetime.now(timezone.utc).isoformat()
        await db.execute(
            "UPDATE memory_pins SET released_at = ? WHERE message_id = ? AND released_at IS NULL",
            (now, message_id),
        )

        return {"released": True, "message_id": message_id}

    @tool(
        name="memory_pinned",
        description=(
            "List all currently pinned memories with their reasons. "
            "Use this to review what the agent has chosen to protect."
        ),
        category=ToolCategory.MEMORY,
        command_prefix="!memory-pinned",
    )
    async def memory_pinned(self) -> Dict[str, Any]:
        """
        List all active (non-released) pinned memories.

        Joins memory_pins with conversation_history to show message
        content alongside pin metadata.
        """
        db = self.storage.db

        rows = await db.fetchall(
            """SELECT mp.id, mp.message_id, mp.pin_reason, mp.pinned_at,
                      ch.content
               FROM memory_pins mp
               JOIN conversation_history ch ON ch.id = mp.message_id
               WHERE mp.released_at IS NULL
               ORDER BY mp.pinned_at DESC""",
        )

        pins = []
        for row in rows:
            pin_id, message_id, reason, pinned_at, content = row
            preview = (content[:100] + "...") if len(content) > 100 else content
            pins.append({
                "pin_id": pin_id,
                "message_id": message_id,
                "reason": reason or "",
                "pinned_at": pinned_at,
                "preview": preview,
            })

        return {"pins": pins, "count": len(pins)}

    @tool(
        name="memory_pin_stats",
        description=(
            "Show memory pin statistics -- total messages, pinned count, "
            "released count, and ratios."
        ),
        category=ToolCategory.SYSTEM,
        command_prefix="!memory-pin-stats",
    )
    async def memory_pin_stats(self) -> Dict[str, Any]:
        """
        Return statistics about memory pinning activity.

        Counts total messages, active pins, and released pins, then
        calculates the pin ratio.
        """
        db = self.storage.db

        total_messages = await db.fetchval(
            "SELECT COUNT(*) FROM conversation_history WHERE agent_id = ?",
            (self.agent_id,),
        ) or 0

        pinned_count = await db.fetchval(
            "SELECT COUNT(*) FROM memory_pins WHERE released_at IS NULL",
        ) or 0

        released_count = await db.fetchval(
            "SELECT COUNT(*) FROM memory_pins WHERE released_at IS NOT NULL",
        ) or 0

        pin_ratio = round(pinned_count / total_messages, 4) if total_messages > 0 else 0.0

        return {
            "total_messages": total_messages,
            "pinned": pinned_count,
            "released": released_count,
            "pin_ratio": pin_ratio,
        }
