"""
Memory Agency -- agent-controlled memory pinning, release, and fact storage.

Allows the agent to actively participate in its own memory by:
- Pinning important memories (resist Ebbinghaus decay)
- Releasing memories it wants to let go of
- Listing currently pinned memories
- Viewing pin statistics
- Saving learned facts to the Knowledge Graph
- Administrative bulk-unpin for sovereign/admin control

Pinned memories get ``decay_protected = True`` in their metadata,
which the MemoryConsolidator already respects (skips archival) and
the MemoryRetriever boosts in scoring.

Pin quotas prevent decay circumvention and database bloat by limiting
the number of active pins per agent.
"""

import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict

from kestrel_sovereign.features.base import Feature, tool
from kestrel_sovereign.tools.base import ToolCategory

logger = logging.getLogger(__name__)

# -- Pin quota configuration --------------------------------------------------
# Maximum number of active pins per agent (can be overridden per-instance).
PIN_QUOTA_DEFAULT = 100

# If the ratio of pinned memories to total memories exceeds this threshold,
# a warning is included in pin responses and stats output.
PIN_RATIO_ALERT_THRESHOLD = 0.5


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

        # Pin quota -- configurable per-instance, defaults to module constant
        self.pin_quota = PIN_QUOTA_DEFAULT

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
    # Helpers
    # ------------------------------------------------------------------

    async def _active_pin_count(self) -> int:
        """Return the number of currently active (non-released) pins."""
        return (
            await self.storage.db.fetchval(
                "SELECT COUNT(*) FROM memory_pins WHERE released_at IS NULL",
            )
            or 0
        )

    async def _pin_ratio(self) -> float:
        """Return the ratio of pinned memories to total memories."""
        total = (
            await self.storage.db.fetchval(
                "SELECT COUNT(*) FROM conversation_history WHERE agent_id = ?",
                (self.agent_id,),
            )
            or 0
        )
        if total == 0:
            return 0.0
        pinned = await self._active_pin_count()
        return pinned / total

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
        the pin in the memory_pins table.  Enforces the per-agent pin quota
        and emits a warning when the pin ratio exceeds the alert threshold.

        Args:
            message_id: The database ID of the message to pin
            reason: Optional reason for pinning this memory
        """
        db = self.storage.db

        # ----- Quota enforcement -----
        # Check whether this message already has an active pin (idempotent
        # re-pin should not count against the quota).
        existing = await db.fetchone(
            "SELECT id FROM memory_pins WHERE message_id = ? AND released_at IS NULL",
            (message_id,),
        )

        if not existing:
            current_pins = await self._active_pin_count()
            if current_pins >= self.pin_quota:
                return {
                    "error": (
                        f"Pin quota reached ({self.pin_quota}). "
                        "Release existing pins before pinning new memories."
                    ),
                    "pinned": False,
                    "quota": self.pin_quota,
                    "current_pins": current_pins,
                }

        # ----- Fetch the message -----
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

        # Record in memory_pins (idempotent -- skip if already active)
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
        result: Dict[str, Any] = {
            "pinned": True,
            "message_id": message_id,
            "preview": preview,
            "reason": reason,
        }

        # ----- Ratio alert -----
        ratio = await self._pin_ratio()
        if ratio > PIN_RATIO_ALERT_THRESHOLD:
            result["warning"] = (
                f"Pin ratio is {ratio:.1%} (threshold: "
                f"{PIN_RATIO_ALERT_THRESHOLD:.0%}). "
                "Consider releasing less important pins to avoid over-pinning."
            )
            logger.warning(
                "Agent %s pin ratio %.2f exceeds threshold %.2f",
                self.agent_id[:30] if self.agent_id else "(none)",
                ratio,
                PIN_RATIO_ALERT_THRESHOLD,
            )

        return result

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

    # ------------------------------------------------------------------
    # Sovereign override -- pins CANNOT resist sovereign actions
    # ------------------------------------------------------------------

    async def sovereign_override_pins(
        self,
        agent_id: str,
        message_ids: list[int] | None = None,
        reason: str = "sovereign_override",
    ) -> int:
        """
        Remove pins by sovereign authority.

        Sovereign deletion, privacy mode changes, and compliance erasure
        MUST override pins immediately -- pins CANNOT block, delay, or
        resurrect erased content.

        Args:
            agent_id: The agent whose pins to override.
            message_ids: Specific message IDs to unpin.  If ``None``,
                all active pins for the agent are removed.
            reason: Audit reason for the override (default: ``sovereign_override``).

        Returns:
            Number of pins overridden.
        """
        db = self.storage.db

        if message_ids:
            placeholders = ",".join("?" for _ in message_ids)
            # Remove pin records for the specified messages
            await db.execute_commit(
                f"DELETE FROM memory_pins WHERE agent_id = ? AND message_id IN ({placeholders})",
                [agent_id] + list(message_ids),
            )
            # Clear decay_protected flag on each message
            for mid in message_ids:
                await self._clear_decay_protected(mid, agent_id)

            logger.info(
                "Sovereign override: cleared %d pin(s) for agent %s (reason=%s)",
                len(message_ids),
                agent_id[:30],
                reason,
            )
            return len(message_ids)
        else:
            # Remove ALL active pins for the agent
            count = await db.execute_commit(
                "DELETE FROM memory_pins WHERE agent_id = ? AND released_at IS NULL",
                (agent_id,),
            )
            # Clear all decay_protected flags in conversation_history metadata
            await db.execute_commit(
                """UPDATE conversation_history
                   SET metadata = json_set(COALESCE(metadata, '{}'), '$.decay_protected', json('false'))
                   WHERE agent_id = ?
                     AND json_extract(metadata, '$.decay_protected') = 1""",
                (agent_id,),
            )
            logger.info(
                "Sovereign override: cleared ALL active pins for agent %s "
                "(count=%s, reason=%s)",
                agent_id[:30],
                count,
                reason,
            )
            return count

    async def _clear_decay_protected(self, message_id: int, agent_id: str) -> None:
        """
        Clear the ``decay_protected`` flag from a single message's metadata.

        Used by :meth:`sovereign_override_pins` to ensure the metadata flag
        is consistent with the pin record removal.
        """
        db = self.storage.db

        row = await db.fetchone(
            "SELECT metadata FROM conversation_history WHERE id = ? AND agent_id = ?",
            (message_id, agent_id),
        )
        if not row:
            return

        raw_metadata = row[0]
        metadata = {}
        if raw_metadata:
            try:
                metadata = json.loads(raw_metadata) if isinstance(raw_metadata, str) else raw_metadata
            except (json.JSONDecodeError, TypeError):
                metadata = {}

        if metadata.get("decay_protected"):
            metadata["decay_protected"] = False
            await db.execute(
                "UPDATE conversation_history SET metadata = ? WHERE id = ? AND agent_id = ?",
                (json.dumps(metadata), message_id, agent_id),
            )

    @tool(
        name="memory_pin_stats",
        description=(
            "Show memory pin statistics -- total messages, pinned count, "
            "released count, ratios, quota usage, and pin age information."
        ),
        category=ToolCategory.SYSTEM,
        command_prefix="!memory-pin-stats",
    )
    async def memory_pin_stats(self) -> Dict[str, Any]:
        """
        Return statistics about memory pinning activity.

        Counts total messages, active pins, and released pins, then
        calculates the pin ratio.  Also includes quota information,
        oldest/average pin age, and an alert when the ratio exceeds
        the configured threshold.
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

        # ----- Pin age information -----
        oldest_pin_at = await db.fetchval(
            "SELECT MIN(pinned_at) FROM memory_pins WHERE released_at IS NULL",
        )
        newest_pin_at = await db.fetchval(
            "SELECT MAX(pinned_at) FROM memory_pins WHERE released_at IS NULL",
        )

        now = datetime.now(timezone.utc)
        oldest_pin_age_seconds = None
        average_pin_age_seconds = None

        if oldest_pin_at:
            try:
                oldest_dt = datetime.fromisoformat(oldest_pin_at)
                oldest_pin_age_seconds = round((now - oldest_dt).total_seconds())
            except (ValueError, TypeError):
                pass

        # Compute average pin age from all active pins
        all_pin_times = await db.fetchall(
            "SELECT pinned_at FROM memory_pins WHERE released_at IS NULL",
        )
        if all_pin_times:
            ages = []
            for (pin_time,) in all_pin_times:
                try:
                    dt = datetime.fromisoformat(pin_time)
                    ages.append((now - dt).total_seconds())
                except (ValueError, TypeError):
                    continue
            if ages:
                average_pin_age_seconds = round(sum(ages) / len(ages))

        result: Dict[str, Any] = {
            "total_messages": total_messages,
            "pinned": pinned_count,
            "released": released_count,
            "pin_ratio": pin_ratio,
            "quota": self.pin_quota,
            "quota_remaining": max(self.pin_quota - pinned_count, 0),
            "oldest_pin_age_seconds": oldest_pin_age_seconds,
            "average_pin_age_seconds": average_pin_age_seconds,
        }

        if pin_ratio > PIN_RATIO_ALERT_THRESHOLD:
            result["alert"] = (
                f"Pin ratio {pin_ratio:.1%} exceeds alert threshold "
                f"({PIN_RATIO_ALERT_THRESHOLD:.0%}). Consider releasing "
                "less important pins to prevent over-pinning."
            )

        return result

    # ------------------------------------------------------------------
    # Knowledge Graph -- learned facts
    # ------------------------------------------------------------------

    @tool(
        name="save_fact",
        description=(
            "Save a learned fact to the Knowledge Graph. Use this when the "
            "user tells you something worth remembering permanently, like "
            "preferences, personal details, or important information. "
            "The fact appears immediately in the Knowledge Graph panel. "
            "Use 'user' as the subject for facts about the user. "
            "Call once per distinct fact — do not save the same fact with different subject names."
        ),
        category=ToolCategory.MEMORY,
        command_prefix="!memory-save-fact",
    )
    async def save_fact(
        self,
        subject: str,
        predicate: str,
        value: str,
        confidence: float = 1.0,
    ) -> Dict[str, Any]:
        """
        Save a learned fact as a Knowledge Graph node.

        Creates a ``learned_fact`` node linked to the agent via a
        ``knows`` edge.  Facts are immediately visible in the KG panel.

        Args:
            subject: Who or what the fact is about (e.g. "user", "project")
            predicate: The relationship or attribute (e.g. "favorite_color", "lives_in")
            value: The fact value (e.g. "blue", "Portland")
            confidence: Confidence level 0.0-1.0 (default 1.0)
        """
        from kestrel_sovereign.storage.async_graph_store import GraphNode

        graph = getattr(self.storage, "graph", None)
        if graph is None:
            return {"error": "Knowledge graph not available"}

        # Clamp confidence
        confidence = max(0.0, min(1.0, confidence))

        # Generate a deterministic-ish node ID for upsert on same subject+predicate
        fact_id = f"fact:{self.agent_id}:{subject}:{predicate}"

        node = GraphNode(
            node_id=fact_id,
            node_type="learned_fact",
            label=f"{predicate.replace('_', ' ').title()}: {value}",
            properties={
                "subject": subject,
                "predicate": predicate,
                "value": value,
                "confidence": confidence,
                "source": "agent_tool",
                "saved_at": datetime.now(timezone.utc).isoformat(),
            },
        )

        await graph.add_node(node)
        await graph.add_edge(self.agent_id, fact_id, "knows")

        logger.info(
            "Saved fact to KG: %s.%s = %s (confidence=%.2f)",
            subject, predicate, value, confidence,
        )

        return {
            "saved": True,
            "node_id": fact_id,
            "subject": subject,
            "predicate": predicate,
            "value": value,
            "confidence": confidence,
        }

    # ------------------------------------------------------------------
    # Admin / Sovereign tools
    # ------------------------------------------------------------------

    @tool(
        name="memory_admin_unpin_all",
        description=(
            "Administrative command: remove ALL active pins for this agent. "
            "Sovereign/admin use only."
        ),
        category=ToolCategory.SYSTEM,
        command_prefix="!memory-admin-unpin-all",
    )
    async def memory_admin_unpin_all(self) -> Dict[str, Any]:
        """
        Remove every active pin for the current agent.

        Marks all active memory_pins records as released and clears
        ``decay_protected`` on the corresponding messages.  This is an
        administrative override intended for sovereign/admin use.
        """
        db = self.storage.db
        now = datetime.now(timezone.utc).isoformat()

        # Fetch all active pin message IDs before releasing
        active_pins = await db.fetchall(
            "SELECT message_id FROM memory_pins WHERE released_at IS NULL",
        )

        if not active_pins:
            return {"unpinned": 0, "message": "No active pins to remove."}

        # Release all active pins
        await db.execute(
            "UPDATE memory_pins SET released_at = ? WHERE released_at IS NULL",
            (now,),
        )

        # Clear decay_protected on all affected messages
        for (message_id,) in active_pins:
            row = await db.fetchone(
                "SELECT id, metadata FROM conversation_history WHERE id = ?",
                (message_id,),
            )
            if row:
                msg_id, raw_metadata = row
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

        count = len(active_pins)
        logger.info("Admin bulk-unpin: removed %d pins for agent %s", count, self.agent_id[:30] if self.agent_id else "(none)")
        return {"unpinned": count}

    @tool(
        name="memory_admin_unpin_oldest",
        description=(
            "Administrative command: remove the N oldest active pins. "
            "Sovereign/admin use only."
        ),
        category=ToolCategory.SYSTEM,
        command_prefix="!memory-admin-unpin-oldest",
    )
    async def memory_admin_unpin_oldest(self, count: int) -> Dict[str, Any]:
        """
        Remove the *count* oldest active pins for this agent.

        Selects the oldest active pins by ``pinned_at``, marks them as
        released, and clears ``decay_protected`` on the corresponding
        messages.

        Args:
            count: Number of oldest pins to release
        """
        db = self.storage.db
        now = datetime.now(timezone.utc).isoformat()

        # Fetch the N oldest active pins
        oldest_pins = await db.fetchall(
            "SELECT id, message_id FROM memory_pins "
            "WHERE released_at IS NULL ORDER BY pinned_at ASC LIMIT ?",
            (count,),
        )

        if not oldest_pins:
            return {"unpinned": 0, "message": "No active pins to remove."}

        for pin_id, message_id in oldest_pins:
            # Release the pin
            await db.execute(
                "UPDATE memory_pins SET released_at = ? WHERE id = ?",
                (now, pin_id),
            )

            # Clear decay_protected on the message
            row = await db.fetchone(
                "SELECT id, metadata FROM conversation_history WHERE id = ?",
                (message_id,),
            )
            if row:
                msg_id, raw_metadata = row
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

        released = len(oldest_pins)
        logger.info(
            "Admin unpin-oldest: removed %d oldest pins for agent %s",
            released,
            self.agent_id[:30] if self.agent_id else "(none)",
        )
        return {"unpinned": released, "requested": count}
