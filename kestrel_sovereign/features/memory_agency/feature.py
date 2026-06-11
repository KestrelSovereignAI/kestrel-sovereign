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

@tool methods return ``kestrel_sdk.tools.result.ToolResult`` per the
kestrel-sovereign #1042 narration-honesty contract (see #1061).
"""

import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from kestrel_sovereign.features.base import Feature, tool
from kestrel_sovereign.features.storage_access import (
    hides_persisted_user_content,
    resolve_feature_database,
)
from kestrel_sdk.tools.base import ToolCategory
from kestrel_sdk.tools.result import ToolResult

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
        self.agent_id = self.agent.did

        # Pin quota -- configurable per-instance, defaults to module constant
        self.pin_quota = PIN_QUOTA_DEFAULT

        self._db = None
        if hides_persisted_user_content(self.agent):
            logger.info(
                "MemoryAgencyFeature: persistent memory pin storage "
                "unavailable in current privacy mode"
            )
            return

        self._db = resolve_feature_database(self.agent)
        if self._db is None:
            raise RuntimeError("MemoryAgencyFeature requires database storage")

        await self._ensure_memory_pins_table()

        logger.info(
            "MemoryAgencyFeature initialized for agent: %s...",
            self.agent_id[:30] if self.agent_id else "(none)",
        )

    async def _ensure_memory_pins_table(self) -> None:
        # Create the memory_pins tracking table
        await self._db.execute(
            """CREATE TABLE IF NOT EXISTS memory_pins (
                id TEXT PRIMARY KEY,
                message_id INTEGER NOT NULL,
                agent_id TEXT,
                pin_reason TEXT DEFAULT '',
                pinned_at TEXT NOT NULL,
                released_at TEXT
            )"""
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    async def _ensure_persistent_db(self) -> Any:
        if self._persistent_memory_hidden():
            return None
        if self._db is None:
            self._db = resolve_feature_database(self.agent)
            if self._db is not None:
                await self._ensure_memory_pins_table()
        return self._db

    def _resolve_override_db(self) -> Any:
        """Resolve raw DB for sovereign deletion/compliance cleanup.

        Override paths remove or relax persistence constraints. They must work
        even when the current privacy mode hides persisted user content.
        """
        if self._db is not None:
            return self._db
        self._db = resolve_feature_database(self.agent)
        return self._db

    async def _active_pin_count(self) -> int:
        """Return the number of currently active (non-released) pins."""
        db = await self._ensure_persistent_db()
        if db is None:
            return 0
        return (
            await db.fetchval(
                "SELECT COUNT(*) FROM memory_pins WHERE released_at IS NULL",
            )
            or 0
        )

    async def _pin_ratio(self) -> float:
        """Return the ratio of pinned memories to total memories."""
        db = await self._ensure_persistent_db()
        if db is None:
            return 0.0
        total = (
            await db.fetchval(
                "SELECT COUNT(*) FROM conversation_history WHERE agent_id = ?",
                (self.agent_id,),
            )
            or 0
        )
        if total == 0:
            return 0.0
        pinned = await self._active_pin_count()
        return pinned / total

    @staticmethod
    def _parse_metadata(raw_metadata: Any) -> Dict[str, Any]:
        """Defensively parse metadata from a conversation_history row."""
        if not raw_metadata:
            return {}
        if isinstance(raw_metadata, dict):
            return dict(raw_metadata)
        if isinstance(raw_metadata, str):
            try:
                parsed = json.loads(raw_metadata)
                return parsed if isinstance(parsed, dict) else {}
            except (json.JSONDecodeError, TypeError):
                return {}
        return {}

    def _persistent_memory_hidden(self) -> bool:
        return hides_persisted_user_content(self.agent)

    def _privacy_unavailable_result(self) -> ToolResult:
        return ToolResult.failed(
            "Memory pinning is unavailable in the current privacy mode",
            data={"privacy_mode_blocks_persistent_storage": True},
        )

    def _storage_unavailable_result(self) -> ToolResult:
        return ToolResult.failed("Memory pinning requires database storage")

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
    async def memory_pin(self, message_id: int, reason: str = "") -> ToolResult:
        """
        Pin a conversation message to protect it from memory decay.

        Args:
            message_id: The database ID of the message to pin
            reason: Optional reason for pinning this memory
        """
        try:
            message_id_val = int(message_id)
        except (TypeError, ValueError):
            return ToolResult.failed(
                f"message_id must be an integer, got {message_id!r}"
            )

        db = await self._ensure_persistent_db()
        if db is None:
            return (
                self._privacy_unavailable_result()
                if self._persistent_memory_hidden()
                else self._storage_unavailable_result()
            )

        # Quota enforcement (idempotent re-pin doesn't count).
        try:
            existing = await db.fetchone(
                "SELECT id FROM memory_pins WHERE message_id = ? AND released_at IS NULL",
                (message_id_val,),
            )
            current_pins = await self._active_pin_count()
        except Exception as e:
            logger.error(f"memory_pin quota check failed: {e}", exc_info=True)
            return ToolResult.failed(str(e))

        if not existing and current_pins >= self.pin_quota:
            return ToolResult.failed(
                f"Pin quota reached ({self.pin_quota}). "
                "Release existing pins before pinning new memories.",
                data={
                    "quota": self.pin_quota,
                    "current_pins": current_pins,
                    "message_id": message_id_val,
                },
            )

        try:
            row = await db.fetchone(
                "SELECT id, content, metadata FROM conversation_history WHERE id = ?",
                (message_id_val,),
            )
        except Exception as e:
            logger.error(f"memory_pin fetch failed: {e}", exc_info=True)
            return ToolResult.failed(str(e))

        if not row:
            return ToolResult.failed(
                f"Message {message_id_val} not found",
                data={"message_id": message_id_val},
            )

        msg_id, content, raw_metadata = row
        metadata = self._parse_metadata(raw_metadata)
        metadata["decay_protected"] = True

        try:
            await db.execute(
                "UPDATE conversation_history SET metadata = ? WHERE id = ?",
                (json.dumps(metadata), msg_id),
            )

            if not existing:
                pin_id = uuid.uuid4().hex[:12]
                await db.execute(
                    """INSERT INTO memory_pins (id, message_id, agent_id, pin_reason, pinned_at)
                       VALUES (?, ?, ?, ?, ?)""",
                    (
                        pin_id,
                        message_id_val,
                        self.agent_id,
                        reason,
                        datetime.now(timezone.utc).isoformat(),
                    ),
                )
        except Exception as e:
            logger.error(f"memory_pin write failed: {e}", exc_info=True)
            return ToolResult.failed(str(e))

        preview = (content[:80] + "...") if len(content) > 80 else content
        data: Dict[str, Any] = {
            "pinned": True,
            "message_id": message_id_val,
            "preview": preview,
            "reason": reason,
            "idempotent_repin": bool(existing),
        }

        # Ratio alert. Honesty: the pin DID succeed, but the ratio
        # is now over the alert threshold — surface as PARTIAL so the
        # LLM cannot claim "pinned successfully" without speaking the
        # over-pinning caveat.
        try:
            ratio = await self._pin_ratio()
        except Exception as e:
            logger.warning(f"memory_pin ratio computation failed: {e}")
            ratio = 0.0
        data["pin_ratio"] = ratio

        if ratio > PIN_RATIO_ALERT_THRESHOLD:
            logger.warning(
                "Agent %s pin ratio %.2f exceeds threshold %.2f",
                self.agent_id[:30] if self.agent_id else "(none)",
                ratio,
                PIN_RATIO_ALERT_THRESHOLD,
            )
            return ToolResult.partial(
                confirmation=(
                    f"Pinned message {message_id_val}"
                    + (f" (re-pin)" if existing else "")
                    + (f" — reason: {reason}" if reason else "")
                ),
                error=(
                    f"pin ratio is {ratio:.1%} (alert threshold "
                    f"{PIN_RATIO_ALERT_THRESHOLD:.0%}); the agent may be "
                    "over-pinning. Consider releasing less important pins "
                    "to avoid evading the decay system."
                ),
                data=data,
            )

        return ToolResult.ok(
            confirmation=(
                f"Pinned message {message_id_val}"
                + (f" (re-pin)" if existing else "")
                + (f" — reason: {reason}" if reason else "")
            ),
            data=data,
        )

    @tool(
        name="memory_release",
        description=(
            "Release a pinned memory so it resumes normal decay. "
            "Use this when a previously important memory is no longer needed."
        ),
        category=ToolCategory.MEMORY,
        command_prefix="!memory-release",
    )
    async def memory_release(self, message_id: int) -> ToolResult:
        """
        Release a pinned memory, allowing it to decay normally again.

        Args:
            message_id: The database ID of the message to release
        """
        try:
            message_id_val = int(message_id)
        except (TypeError, ValueError):
            return ToolResult.failed(
                f"message_id must be an integer, got {message_id!r}"
            )

        db = await self._ensure_persistent_db()
        if db is None:
            return (
                self._privacy_unavailable_result()
                if self._persistent_memory_hidden()
                else self._storage_unavailable_result()
            )

        try:
            row = await db.fetchone(
                "SELECT id, metadata FROM conversation_history WHERE id = ?",
                (message_id_val,),
            )
        except Exception as e:
            logger.error(f"memory_release fetch failed: {e}", exc_info=True)
            return ToolResult.failed(str(e))

        if not row:
            return ToolResult.failed(
                f"Message {message_id_val} not found",
                data={"message_id": message_id_val},
            )

        msg_id, raw_metadata = row
        metadata = self._parse_metadata(raw_metadata)
        was_pinned = bool(metadata.get("decay_protected"))
        metadata["decay_protected"] = False

        # Honesty: if the message wasn't pinned, the release is a
        # no-op. Tell the LLM that explicitly so it can't say
        # "released the pin" when there was no pin to release.
        try:
            await db.execute(
                "UPDATE conversation_history SET metadata = ? WHERE id = ?",
                (json.dumps(metadata), msg_id),
            )

            now = datetime.now(timezone.utc).isoformat()
            await db.execute(
                "UPDATE memory_pins SET released_at = ? WHERE message_id = ? AND released_at IS NULL",
                (now, message_id_val),
            )
        except Exception as e:
            logger.error(f"memory_release write failed: {e}", exc_info=True)
            return ToolResult.failed(str(e))

        if not was_pinned:
            return ToolResult.ok(
                confirmation=(
                    f"Message {message_id_val} was not pinned — no-op "
                    "(decay_protected flag was already False)"
                ),
                data={
                    "released": False,
                    "was_pinned": False,
                    "message_id": message_id_val,
                },
            )

        return ToolResult.ok(
            confirmation=f"Released pin on message {message_id_val}",
            data={
                "released": True,
                "was_pinned": True,
                "message_id": message_id_val,
            },
        )

    @tool(
        name="memory_pinned",
        description=(
            "List all currently pinned memories with their reasons. "
            "Use this to review what the agent has chosen to protect."
        ),
        category=ToolCategory.MEMORY,
        command_prefix="!memory-pinned",
    )
    async def memory_pinned(self) -> ToolResult:
        """List all active (non-released) pinned memories."""
        db = await self._ensure_persistent_db()
        if db is None:
            return (
                self._privacy_unavailable_result()
                if self._persistent_memory_hidden()
                else self._storage_unavailable_result()
            )

        try:
            rows = await db.fetchall(
                """SELECT mp.id, mp.message_id, mp.pin_reason, mp.pinned_at,
                          ch.content
                   FROM memory_pins mp
                   JOIN conversation_history ch ON ch.id = mp.message_id
                   WHERE mp.released_at IS NULL
                   ORDER BY mp.pinned_at DESC""",
            )
        except Exception as e:
            logger.error(f"memory_pinned query failed: {e}", exc_info=True)
            return ToolResult.failed(str(e))

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

        return ToolResult.ok(
            confirmation=f"Listed {len(pins)} active pin(s)",
            data={"pins": pins, "count": len(pins)},
        )

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
        db = self._resolve_override_db()
        if db is None:
            return 0

        if message_ids:
            placeholders = ",".join("?" for _ in message_ids)
            # Remove pin records for the specified messages
            await db.execute_commit(
                f"DELETE FROM memory_pins WHERE agent_id = ? AND message_id IN ({placeholders})",
                [agent_id] + list(message_ids),
            )
            # Clear decay_protected flag on each message
            for mid in message_ids:
                await self._clear_decay_protected(mid, agent_id, db=db)

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

    async def _clear_decay_protected(
        self,
        message_id: int,
        agent_id: str,
        *,
        db: Any | None = None,
    ) -> None:
        """
        Clear the ``decay_protected`` flag from a single message's metadata.

        Used by :meth:`sovereign_override_pins` to ensure the metadata flag
        is consistent with the pin record removal.
        """
        db = db or self._resolve_override_db()
        if db is None:
            return

        row = await db.fetchone(
            "SELECT metadata FROM conversation_history WHERE id = ? AND agent_id = ?",
            (message_id, agent_id),
        )
        if not row:
            return

        raw_metadata = row[0]
        metadata = self._parse_metadata(raw_metadata)

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
    async def memory_pin_stats(self) -> ToolResult:
        """Return statistics about memory pinning activity."""
        db = await self._ensure_persistent_db()
        if db is None:
            return (
                self._privacy_unavailable_result()
                if self._persistent_memory_hidden()
                else self._storage_unavailable_result()
            )

        try:
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

            oldest_pin_at = await db.fetchval(
                "SELECT MIN(pinned_at) FROM memory_pins WHERE released_at IS NULL",
            )

            now = datetime.now(timezone.utc)
            oldest_pin_age_seconds: Optional[int] = None
            average_pin_age_seconds: Optional[int] = None

            if oldest_pin_at:
                try:
                    oldest_dt = datetime.fromisoformat(oldest_pin_at)
                    oldest_pin_age_seconds = round((now - oldest_dt).total_seconds())
                except (ValueError, TypeError):
                    pass

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
        except Exception as e:
            logger.error(f"memory_pin_stats failed: {e}", exc_info=True)
            return ToolResult.failed(str(e))

        data: Dict[str, Any] = {
            "total_messages": total_messages,
            "pinned": pinned_count,
            "released": released_count,
            "pin_ratio": pin_ratio,
            "quota": self.pin_quota,
            "quota_remaining": max(self.pin_quota - pinned_count, 0),
            "oldest_pin_age_seconds": oldest_pin_age_seconds,
            "average_pin_age_seconds": average_pin_age_seconds,
        }

        confirmation = (
            f"Pin stats: {pinned_count} active, {released_count} released, "
            f"ratio={pin_ratio:.1%}, quota_remaining={data['quota_remaining']}"
        )

        # Honesty: stats themselves are observational (the tool ran),
        # but a high pin ratio is a finding the LLM should surface
        # rather than buried in data. PARTIAL with the alert framing.
        if pin_ratio > PIN_RATIO_ALERT_THRESHOLD:
            return ToolResult.partial(
                confirmation=confirmation,
                error=(
                    f"pin ratio {pin_ratio:.1%} exceeds alert threshold "
                    f"({PIN_RATIO_ALERT_THRESHOLD:.0%}); the agent is "
                    "over-pinning relative to total memories. Consider "
                    "releasing less important pins to prevent evading "
                    "the decay system."
                ),
                data=data,
            )
        return ToolResult.ok(confirmation=confirmation, data=data)

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
    ) -> ToolResult:
        """
        Save a learned fact as a Knowledge Graph node.

        Args:
            subject: Who or what the fact is about (e.g. "user", "project")
            predicate: The relationship or attribute (e.g. "favorite_color", "lives_in")
            value: The fact value (e.g. "blue", "Portland")
            confidence: Confidence level 0.0-1.0 (default 1.0)
        """
        from kestrel_sovereign.storage.async_graph_store import GraphNode

        try:
            confidence_val = float(confidence)
        except (TypeError, ValueError):
            return ToolResult.failed(
                f"confidence must be a number, got {confidence!r}"
            )

        graph = getattr(self.storage, "graph", None)
        if graph is None:
            return ToolResult.failed("Knowledge graph not available")

        # Honesty: confidence is silently clamped to [0, 1]. Pre-fix
        # this was hidden — the agent could pass 1.5 and the saved
        # node would have 1.0 with no signal. Surface as PARTIAL when
        # the input was actually clamped.
        clamped_confidence = max(0.0, min(1.0, confidence_val))
        was_clamped = clamped_confidence != confidence_val

        fact_id = f"fact:{self.agent_id}:{subject}:{predicate}"
        node = GraphNode(
            node_id=fact_id,
            node_type="learned_fact",
            label=f"{predicate.replace('_', ' ').title()}: {value}",
            properties={
                "subject": subject,
                "predicate": predicate,
                "value": value,
                "confidence": clamped_confidence,
                "source": "agent_tool",
                "saved_at": datetime.now(timezone.utc).isoformat(),
            },
        )

        try:
            await graph.add_node(node)
            await graph.add_edge(self.agent_id, fact_id, "knows")
        except Exception as e:
            logger.error(f"save_fact write failed: {e}", exc_info=True)
            return ToolResult.failed(str(e))

        logger.info(
            "Saved fact to KG: %s.%s = %s (confidence=%.2f)",
            subject, predicate, value, clamped_confidence,
        )

        data = {
            "saved": True,
            "node_id": fact_id,
            "subject": subject,
            "predicate": predicate,
            "value": value,
            "confidence": clamped_confidence,
            "confidence_requested": confidence_val,
            "confidence_clamped": was_clamped,
        }

        if was_clamped:
            return ToolResult.partial(
                confirmation=(
                    f"Saved fact {subject}.{predicate}={value} "
                    f"(confidence={clamped_confidence:.2f})"
                ),
                error=(
                    f"requested confidence={confidence_val} was outside "
                    f"[0.0, 1.0]; clamped to {clamped_confidence:.2f}"
                ),
                data=data,
            )

        return ToolResult.ok(
            confirmation=(
                f"Saved fact {subject}.{predicate}={value} "
                f"(confidence={clamped_confidence:.2f})"
            ),
            data=data,
        )

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
    async def memory_admin_unpin_all(self) -> ToolResult:
        """Remove every active pin for the current agent."""
        db = await self._ensure_persistent_db()
        if db is None:
            return (
                self._privacy_unavailable_result()
                if self._persistent_memory_hidden()
                else self._storage_unavailable_result()
            )
        now = datetime.now(timezone.utc).isoformat()

        try:
            active_pins = await db.fetchall(
                "SELECT message_id FROM memory_pins WHERE released_at IS NULL",
            )
        except Exception as e:
            logger.error(f"memory_admin_unpin_all query failed: {e}", exc_info=True)
            return ToolResult.failed(str(e))

        if not active_pins:
            return ToolResult.ok(
                confirmation="No active pins to remove (no-op)",
                data={"unpinned": 0},
            )

        try:
            await db.execute(
                "UPDATE memory_pins SET released_at = ? WHERE released_at IS NULL",
                (now,),
            )
        except Exception as e:
            logger.error(f"memory_admin_unpin_all release failed: {e}", exc_info=True)
            return ToolResult.failed(str(e))

        # Clear decay_protected on each affected message. If a row
        # write fails, track it — release succeeded but the metadata
        # is inconsistent for that message until repaired.
        metadata_failures: List[Dict[str, Any]] = []
        for (message_id,) in active_pins:
            try:
                row = await db.fetchone(
                    "SELECT id, metadata FROM conversation_history WHERE id = ?",
                    (message_id,),
                )
                if row:
                    msg_id, raw_metadata = row
                    metadata = self._parse_metadata(raw_metadata)
                    metadata["decay_protected"] = False
                    await db.execute(
                        "UPDATE conversation_history SET metadata = ? WHERE id = ?",
                        (json.dumps(metadata), msg_id),
                    )
            except Exception as e:
                metadata_failures.append({"message_id": message_id, "error": str(e)})

        count = len(active_pins)
        logger.info(
            "Admin bulk-unpin: removed %d pins for agent %s",
            count,
            self.agent_id[:30] if self.agent_id else "(none)",
        )

        if metadata_failures:
            return ToolResult.partial(
                confirmation=f"Released {count} pin(s) (memory_pins records updated)",
                error=(
                    f"{len(metadata_failures)} message(s) had metadata "
                    "update failures — pin records are released but the "
                    "decay_protected flag in conversation_history is stale "
                    "for those rows. Manual reconciliation needed."
                ),
                data={
                    "unpinned": count,
                    "metadata_failures": metadata_failures,
                },
            )
        return ToolResult.ok(
            confirmation=f"Released {count} pin(s)",
            data={"unpinned": count},
        )

    @tool(
        name="memory_admin_unpin_oldest",
        description=(
            "Administrative command: remove the N oldest active pins. "
            "Sovereign/admin use only."
        ),
        category=ToolCategory.SYSTEM,
        command_prefix="!memory-admin-unpin-oldest",
    )
    async def memory_admin_unpin_oldest(self, count: int) -> ToolResult:
        """
        Remove the *count* oldest active pins for this agent.

        Args:
            count: Number of oldest pins to release
        """
        try:
            count_val = int(count)
        except (TypeError, ValueError):
            return ToolResult.failed(
                f"count must be an integer, got {count!r}"
            )
        if count_val < 1:
            return ToolResult.failed("count must be >= 1")

        db = await self._ensure_persistent_db()
        if db is None:
            return (
                self._privacy_unavailable_result()
                if self._persistent_memory_hidden()
                else self._storage_unavailable_result()
            )
        now = datetime.now(timezone.utc).isoformat()

        try:
            oldest_pins = await db.fetchall(
                "SELECT id, message_id FROM memory_pins "
                "WHERE released_at IS NULL ORDER BY pinned_at ASC LIMIT ?",
                (count_val,),
            )
        except Exception as e:
            logger.error(f"memory_admin_unpin_oldest query failed: {e}", exc_info=True)
            return ToolResult.failed(str(e))

        if not oldest_pins:
            return ToolResult.ok(
                confirmation="No active pins to remove (no-op)",
                data={"unpinned": 0, "requested": count_val},
            )

        # Honesty: separate pin-release from metadata clear so a write
        # failure on the memory_pins UPDATE doesn't get reported as a
        # successful release. If the pin record can't be marked
        # released_at, the pin is still active — it must NOT count
        # toward `unpinned` or the caller will free quota that wasn't
        # actually freed.
        released_count = 0
        release_failures: List[Dict[str, Any]] = []
        metadata_failures: List[Dict[str, Any]] = []
        for pin_id, message_id in oldest_pins:
            try:
                await db.execute(
                    "UPDATE memory_pins SET released_at = ? WHERE id = ?",
                    (now, pin_id),
                )
            except Exception as e:
                release_failures.append(
                    {"pin_id": pin_id, "message_id": message_id, "error": str(e)}
                )
                continue
            released_count += 1

            try:
                row = await db.fetchone(
                    "SELECT id, metadata FROM conversation_history WHERE id = ?",
                    (message_id,),
                )
                if row:
                    msg_id, raw_metadata = row
                    metadata = self._parse_metadata(raw_metadata)
                    metadata["decay_protected"] = False
                    await db.execute(
                        "UPDATE conversation_history SET metadata = ? WHERE id = ?",
                        (json.dumps(metadata), msg_id),
                    )
            except Exception as e:
                metadata_failures.append({"message_id": message_id, "error": str(e)})

        released = released_count
        logger.info(
            "Admin unpin-oldest: removed %d oldest pins for agent %s",
            released,
            self.agent_id[:30] if self.agent_id else "(none)",
        )

        data = {
            "unpinned": released,
            "requested": count_val,
            "shortfall": max(0, count_val - released),
        }

        partial_errs: List[str] = []
        if released < count_val:
            partial_errs.append(
                f"requested {count_val} pin(s) released but only "
                f"{released} were active; {count_val - released} fewer "
                "than requested"
            )
        if release_failures:
            data["release_failures"] = release_failures
            partial_errs.append(
                f"{len(release_failures)} pin(s) could not be released "
                "(memory_pins UPDATE failed); those pins are still "
                "active and continue to occupy quota"
            )
        if metadata_failures:
            data["metadata_failures"] = metadata_failures
            partial_errs.append(
                f"{len(metadata_failures)} message(s) had metadata "
                "update failures — pin records released but "
                "decay_protected flag is stale; manual reconciliation needed"
            )

        if partial_errs:
            return ToolResult.partial(
                confirmation=f"Released {released} oldest pin(s)",
                error=" | ".join(partial_errs),
                data=data,
            )

        return ToolResult.ok(
            confirmation=f"Released {released} oldest pin(s)",
            data=data,
        )
