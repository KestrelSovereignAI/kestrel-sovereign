"""
Delivery Feature -- durable outbound message queue with retry and dead letter queue.

Provides a disk-backed delivery queue for outbound messages. Messages are
persisted in the database, retried with exponential backoff on failure, and
moved to a dead letter queue after max retries are exhausted.

A background asyncio worker polls every 10 seconds for pending/retryable
messages and processes them in batches of up to 10.

Tools:
    !delivery status              -- show queue status (pending, failed, delivered counts)
    !delivery queue               -- list pending messages
    !delivery failed              -- list dead letter queue
    !delivery retry <message_id>  -- manually retry a failed message
    !delivery purge               -- clear delivered messages older than 24h
"""

import logging
from typing import Any, Dict, Optional

from kestrel_sovereign.features.base import Feature, tool
from kestrel_sovereign.features.delivery.queue import DeliveryQueue
from kestrel_sovereign.features.storage_access import resolve_feature_database
from kestrel_sdk.tools.base import ToolCategory
from kestrel_sdk.tools.result import ToolResult

logger = logging.getLogger(__name__)


class DeliveryFeature(Feature):
    """
    Durable outbound delivery queue with retry and dead letter queue.

    On initialize(), resolves the database handle from agent storage,
    creates the delivery queue tables, and starts a background worker
    that polls for pending messages every 10 seconds.
    """

    @property
    def tool_description(self) -> str:
        return (
            "Manage outbound message delivery - view queue status, list pending "
            "and dead-letter messages, retry failed deliveries, and purge old entries"
        )

    async def initialize(self):
        """Initialize the delivery feature: set up DB refs and start the queue worker."""
        self._db = None
        self._agent_id = ""
        self._queue: Optional[DeliveryQueue] = None

        self._db = resolve_feature_database(self.agent)

        # Agent identity (DID is the canonical source of truth)
        self._agent_id = self.agent.did

        if self._db is None:
            logger.warning("DeliveryFeature: no database available, running in no-op mode")
            return

        # Create and start the delivery queue
        self._queue = DeliveryQueue(
            db=self._db,
            agent_id=self._agent_id,
        )
        await self._queue.start()
        logger.info("DeliveryFeature initialized")

    async def shutdown(self):
        """Stop the background delivery worker."""
        if self._queue:
            await self._queue.stop()

    # ------------------------------------------------------------------
    # Public API for programmatic access
    # ------------------------------------------------------------------

    async def enqueue_message(
        self,
        channel_type: str,
        recipient: str,
        content: Dict[str, Any],
        max_retries: Optional[int] = None,
    ) -> Optional[str]:
        """Enqueue a message for delivery (programmatic API).

        Args:
            channel_type: Delivery channel (e.g. "webhook", "email", "ws").
            recipient: Target address or identifier.
            content: Message payload dict.
            max_retries: Override default max retries.

        Returns:
            Queue entry ID, or None if queue is not available.
        """
        if not self._queue:
            logger.warning("DeliveryFeature: cannot enqueue, queue not available")
            return None
        return await self._queue.enqueue(channel_type, recipient, content, max_retries)

    # ------------------------------------------------------------------
    # Tools
    # ------------------------------------------------------------------

    @tool(
        "delivery_status",
        "Show delivery queue status with counts of pending, failed, delivered, and dead letter messages",
        category=ToolCategory.COMMUNICATION,
        command_prefix="!delivery status",
    )
    async def delivery_status(self) -> ToolResult:
        """
        Show current delivery queue status.

        Returns:
            ToolResult.ok with counts + total + queue_healthy flag.
            PARTIAL when the dead-letter queue is non-empty (the
            queue is operating, but messages have permanently failed
            and the LLM should speak that). ERROR when the queue is
            not available (no DB attached) or the count query raises.
        """
        if not self._queue:
            return ToolResult.failed(error="Delivery queue not available")

        try:
            counts = await self._queue.get_status_counts()
            total = sum(counts.values())
            dead = counts.get("dead_letter", 0)
            healthy = dead == 0
            data = {
                "agent_id": self._agent_id,
                "counts": counts,
                "total": total,
                "queue_healthy": healthy,
            }
            confirmation = (
                f"Delivery queue: {total} message(s) "
                + ", ".join(f"{k}={v}" for k, v in counts.items())
                + "."
            )
            if not healthy:
                return ToolResult.partial(
                    confirmation,
                    (
                        f"{dead} message(s) in dead_letter — they exhausted "
                        "max_retries and will NOT be delivered without a "
                        "manual !delivery retry."
                    ),
                    data=data,
                )
            return ToolResult.ok(confirmation, data=data)
        except Exception as e:
            logger.error("Failed to get delivery status: %s", e)
            return ToolResult.failed(error=str(e))

    @tool(
        "delivery_queue_list",
        "List pending messages in the delivery queue",
        category=ToolCategory.COMMUNICATION,
        command_prefix="!delivery queue",
    )
    async def delivery_queue_list(self, limit: int = 20) -> ToolResult:
        """
        List pending and retryable messages.

        Args:
            limit: Maximum number of entries to return (default: 20)

        Returns:
            ToolResult.ok with the list of pending queue entries;
            ERROR when the queue is not available or the query raises.
        """
        if not self._queue:
            return ToolResult.failed(error="Delivery queue not available")

        try:
            entries = await self._queue.get_pending_entries(limit=limit)
            return ToolResult.ok(
                f"Returned {len(entries)} pending delivery entry(ies).",
                data={
                    "entries": [e.to_dict() for e in entries],
                    "count": len(entries),
                },
            )
        except Exception as e:
            logger.error("Failed to list delivery queue: %s", e)
            return ToolResult.failed(error=str(e))

    @tool(
        "delivery_failed",
        "List messages in the dead letter queue (permanently failed)",
        category=ToolCategory.COMMUNICATION,
        command_prefix="!delivery failed",
    )
    async def delivery_failed(self, limit: int = 20) -> ToolResult:
        """
        List messages in the dead letter queue.

        Args:
            limit: Maximum number of entries to return (default: 20)

        Returns:
            ToolResult.ok with dead-letter entries (empty list is the
            healthy case); ERROR when the queue is not available or
            the query raises.
        """
        if not self._queue:
            return ToolResult.failed(error="Delivery queue not available")

        try:
            entries = await self._queue.get_dead_letter_entries(limit=limit)
            return ToolResult.ok(
                f"Returned {len(entries)} dead-letter entry(ies).",
                data={
                    "entries": entries,
                    "count": len(entries),
                },
            )
        except Exception as e:
            logger.error("Failed to list dead letter queue: %s", e)
            return ToolResult.failed(error=str(e))

    @tool(
        "delivery_retry",
        "Manually retry a failed or dead-lettered message by its entry ID",
        category=ToolCategory.COMMUNICATION,
        command_prefix="!delivery retry",
    )
    async def delivery_retry(self, message_id: str) -> ToolResult:
        """
        Manually retry a failed or dead-lettered message.

        Args:
            message_id: The queue entry ID or dead-letter ID to retry

        Returns:
            ToolResult.ok when the queue accepted the retry request;
            ERROR when the queue is not available, the entry isn't
            found, the message is already delivered or in flight, or
            the queue raised. The underlying ``DeliveryQueue.retry``
            returns ``{"success": True/False, ...}``; we route those
            into OK / ERROR accordingly so callers don't have to
            keep grepping a stringly-typed shape.
        """
        if not self._queue:
            return ToolResult.failed(error="Delivery queue not available")

        try:
            payload = await self._queue.retry(message_id)
        except Exception as e:
            logger.error("Failed to retry delivery %s: %s", message_id, e)
            return ToolResult.failed(error=str(e))

        # ``DeliveryQueue.retry`` returns the legacy
        # ``{"success": ..., "error"?: ..., ...}`` shape; map it onto
        # the envelope without dropping any caller-visible fields.
        if not isinstance(payload, dict):
            return ToolResult.ok(
                f"Retried delivery for {message_id}.",
                data={"message_id": message_id, "raw": payload},
            )
        if payload.get("success"):
            confirmation = (
                f"Queued retry for delivery {message_id}"
                + (f" ({payload['status']})." if payload.get("status") else ".")
            )
            return ToolResult.ok(confirmation, data=payload)
        err = payload.get("error") or f"Could not retry delivery {message_id}"
        return ToolResult.failed(error=err, data=payload)

    @tool(
        "delivery_purge",
        "Clear delivered messages older than 24 hours from the queue",
        category=ToolCategory.COMMUNICATION,
        command_prefix="!delivery purge",
    )
    async def delivery_purge(self, older_than_hours: int = 24) -> ToolResult:
        """
        Purge delivered messages older than the specified threshold.

        Args:
            older_than_hours: Remove entries delivered more than this many hours ago (default: 24)

        Returns:
            ToolResult.ok with the count purged. ERROR when the queue
            is not available or the purge raises.
        """
        if not self._queue:
            return ToolResult.failed(error="Delivery queue not available")

        try:
            purged = await self._queue.purge_delivered(older_than_hours=older_than_hours)
            return ToolResult.ok(
                f"Purged {purged} delivered message(s) older than {older_than_hours}h.",
                data={"purged": purged, "older_than_hours": older_than_hours},
            )
        except Exception as e:
            logger.error("Failed to purge delivered messages: %s", e)
            return ToolResult.failed(error=str(e))
