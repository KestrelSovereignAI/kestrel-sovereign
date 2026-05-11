"""
Durable delivery queue with retry logic, exponential backoff, and dead letter queue.

The DeliveryQueue persists messages in SQLite, processes them via a background
asyncio task, and moves exhausted entries to a dead letter table after max
retries are exceeded.

Retry backoff formula: base_delay * (5 ** attempt), capped at 1 hour.
Default: 5s -> 25s -> 2m5s -> 10m25s -> 52m5s

Deduplication: content_hash + recipient within a 60-second window prevents
duplicate enqueues of the same message.
"""

import asyncio
import json
import logging
import time
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Coroutine, Dict, List, Optional

from kestrel_sovereign.features.delivery.models import (
    DeliveryResult,
    DeliveryStatus,
    DeliveryTask,
    QueueEntry,
)

logger = logging.getLogger(__name__)

# Background worker constants
POLL_INTERVAL_SECONDS = 10
BATCH_SIZE = 10

# Retry constants
DEFAULT_MAX_RETRIES = 5
BASE_DELAY_SECONDS = 5
MAX_DELAY_SECONDS = 3600  # 1 hour cap

# Deduplication window
DEDUP_WINDOW_SECONDS = 60

# Type alias for the delivery callback
DeliveryCallback = Callable[[str, str, Dict[str, Any]], Coroutine[Any, Any, DeliveryResult]]


def _compute_backoff(attempt: int) -> float:
    """Compute exponential backoff delay for the given attempt number.

    Formula: base_delay * (5 ** attempt), capped at MAX_DELAY_SECONDS.

    Args:
        attempt: Zero-based attempt number (0 = first retry).

    Returns:
        Delay in seconds before the next retry.
    """
    delay = BASE_DELAY_SECONDS * (5 ** attempt)
    return min(delay, MAX_DELAY_SECONDS)


class DeliveryQueue:
    """
    Disk-backed outbound message queue with retry and dead letter support.

    Usage:
        queue = DeliveryQueue(db, agent_id, deliver_fn)
        await queue.start()     # creates tables + starts background worker
        entry_id = await queue.enqueue(message)
        ...
        await queue.stop()      # graceful shutdown
    """

    def __init__(
        self,
        db,
        agent_id: str,
        deliver: Optional[DeliveryCallback] = None,
        allow_noop_delivery: bool = False,
        poll_interval: int = POLL_INTERVAL_SECONDS,
        batch_size: int = BATCH_SIZE,
        max_retries: int = DEFAULT_MAX_RETRIES,
    ):
        """
        Args:
            db: AsyncDatabase-like object with execute/fetchall/fetchone.
            agent_id: The owning agent's identifier.
            deliver: Async callable(channel_type, recipient, content) -> DeliveryResult.
            allow_noop_delivery: Explicit test/development opt-in for marking
                messages delivered when no delivery provider is configured.
            poll_interval: Seconds between background poll cycles.
            batch_size: Max messages to process per tick.
            max_retries: Default maximum retries before dead-lettering.
        """
        self._db = db
        self._agent_id = agent_id
        self._deliver = deliver
        self._allow_noop_delivery = allow_noop_delivery
        self._poll_interval = poll_interval
        self._batch_size = batch_size
        self._max_retries = max_retries
        self._task: Optional[asyncio.Task] = None
        self._running = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self):
        """Create DB tables and launch the background worker."""
        await self._ensure_tables()
        self._running = True
        self._task = asyncio.create_task(self._loop(), name="delivery-queue-worker")
        logger.info(
            "DeliveryQueue started (poll every %ds, batch %d, max_retries %d)",
            self._poll_interval,
            self._batch_size,
            self._max_retries,
        )

    async def stop(self):
        """Gracefully stop the background worker."""
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("DeliveryQueue stopped")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def enqueue(
        self,
        channel_type: str,
        recipient: str,
        content: Dict[str, Any],
        max_retries: Optional[int] = None,
    ) -> str:
        """Persist a new message to the queue.

        Performs deduplication: if an identical message (same recipient + content)
        was enqueued within the last DEDUP_WINDOW_SECONDS, returns the existing ID
        instead of creating a duplicate.

        Args:
            channel_type: Delivery channel (e.g. "webhook", "email", "ws").
            recipient: Target address/identifier.
            content: Message payload (will be JSON-serialized).
            max_retries: Override default max retries for this entry.

        Returns:
            The queue entry ID (existing if deduplicated, new otherwise).
        """
        content_json = json.dumps(content, default=str)
        content_hash = QueueEntry.compute_content_hash(recipient, content_json)
        retries = max_retries if max_retries is not None else self._max_retries

        # Deduplication check
        dedup_cutoff = (
            datetime.now(timezone.utc) - timedelta(seconds=DEDUP_WINDOW_SECONDS)
        ).isoformat()

        existing = await self._db.fetchone(
            """
            SELECT id FROM delivery_queue
            WHERE agent_id = ? AND content_hash = ? AND recipient = ?
                  AND created_at >= ?
            """,
            (self._agent_id, content_hash, recipient, dedup_cutoff),
        )
        if existing:
            logger.debug("Deduplicated delivery entry: %s", existing[0])
            return existing[0]

        entry_id = str(uuid.uuid4())
        now_iso = datetime.now(timezone.utc).isoformat()

        await self._db.execute(
            """
            INSERT INTO delivery_queue
                (id, agent_id, channel_type, recipient, content_json, content_hash,
                 status, attempts, max_retries, next_retry_at, last_error,
                 created_at, delivered_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, 0, ?, ?, NULL, ?, NULL)
            """,
            (
                entry_id,
                self._agent_id,
                channel_type,
                recipient,
                content_json,
                content_hash,
                DeliveryStatus.PENDING.value,
                retries,
                now_iso,  # next_retry_at = now (immediately eligible)
                now_iso,  # created_at
            ),
        )

        logger.info(
            "Enqueued delivery %s -> %s/%s",
            entry_id, channel_type, recipient,
        )
        return entry_id

    async def process_pending(self) -> int:
        """Process the next batch of pending/retryable messages.

        Returns:
            Number of messages processed in this batch.
        """
        now_iso = datetime.now(timezone.utc).isoformat()
        rows = await self._db.fetchall(
            """
            SELECT id, agent_id, channel_type, recipient, content_json, content_hash,
                   status, attempts, max_retries, next_retry_at, last_error,
                   created_at, delivered_at
            FROM delivery_queue
            WHERE agent_id = ?
                  AND status IN (?, ?)
                  AND next_retry_at <= ?
            ORDER BY next_retry_at ASC
            LIMIT ?
            """,
            (
                self._agent_id,
                DeliveryStatus.PENDING.value,
                DeliveryStatus.FAILED.value,
                now_iso,
                self._batch_size,
            ),
        )

        processed = 0
        for row in rows:
            entry = self._row_to_entry(row)
            await self._attempt_delivery(entry)
            processed += 1

        return processed

    async def retry(self, entry_id: str) -> Dict[str, Any]:
        """Manually retry a failed or dead-lettered message.

        For dead-lettered entries, this moves them back to the main queue.

        Args:
            entry_id: The queue entry ID to retry.

        Returns:
            Dict with status information.
        """
        # First check the main queue
        row = await self._db.fetchone(
            """
            SELECT id, agent_id, channel_type, recipient, content_json, content_hash,
                   status, attempts, max_retries, next_retry_at, last_error,
                   created_at, delivered_at
            FROM delivery_queue
            WHERE id = ? AND agent_id = ?
            """,
            (entry_id, self._agent_id),
        )

        if row:
            entry = self._row_to_entry(row)
            if entry.status == DeliveryStatus.DELIVERED:
                return {"success": False, "error": "Message already delivered"}
            if entry.status == DeliveryStatus.IN_FLIGHT:
                return {"success": False, "error": "Message is currently in flight"}

            # Reset for retry
            now_iso = datetime.now(timezone.utc).isoformat()
            await self._db.execute(
                """
                UPDATE delivery_queue
                SET status = ?, next_retry_at = ?, last_error = NULL
                WHERE id = ? AND agent_id = ?
                """,
                (DeliveryStatus.PENDING.value, now_iso, entry_id, self._agent_id),
            )
            return {"success": True, "entry_id": entry_id, "status": "queued_for_retry"}

        # Check dead letter queue
        dl_row = await self._db.fetchone(
            """
            SELECT id, original_id, agent_id, channel_type, recipient,
                   content_json, error, attempts, created_at
            FROM delivery_dead_letter
            WHERE (id = ? OR original_id = ?) AND agent_id = ?
            """,
            (entry_id, entry_id, self._agent_id),
        )

        if dl_row:
            # Re-enqueue from dead letter
            now_iso = datetime.now(timezone.utc).isoformat()
            new_id = str(uuid.uuid4())
            content_hash = QueueEntry.compute_content_hash(dl_row[4], dl_row[5])

            await self._db.execute(
                """
                INSERT INTO delivery_queue
                    (id, agent_id, channel_type, recipient, content_json, content_hash,
                     status, attempts, max_retries, next_retry_at, last_error,
                     created_at, delivered_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, 0, ?, ?, NULL, ?, NULL)
                """,
                (
                    new_id,
                    self._agent_id,
                    dl_row[3],  # channel_type
                    dl_row[4],  # recipient
                    dl_row[5],  # content_json
                    content_hash,
                    DeliveryStatus.PENDING.value,
                    self._max_retries,
                    now_iso,  # next_retry_at
                    now_iso,  # created_at
                ),
            )

            # Remove from dead letter
            await self._db.execute(
                "DELETE FROM delivery_dead_letter WHERE id = ? AND agent_id = ?",
                (dl_row[0], self._agent_id),
            )

            return {
                "success": True,
                "entry_id": new_id,
                "original_id": dl_row[1],
                "status": "re-enqueued_from_dead_letter",
            }

        return {"success": False, "error": f"Entry {entry_id} not found"}

    async def move_to_dead_letter(self, entry_id: str, reason: str) -> None:
        """Move an entry from the main queue to the dead letter table.

        Args:
            entry_id: The queue entry ID to move.
            reason: Human-readable reason for dead-lettering.
        """
        row = await self._db.fetchone(
            """
            SELECT id, agent_id, channel_type, recipient, content_json,
                   attempts, created_at
            FROM delivery_queue
            WHERE id = ? AND agent_id = ?
            """,
            (entry_id, self._agent_id),
        )
        if not row:
            logger.warning("Cannot dead-letter unknown entry: %s", entry_id)
            return

        dl_id = str(uuid.uuid4())
        now_iso = datetime.now(timezone.utc).isoformat()

        await self._db.execute(
            """
            INSERT INTO delivery_dead_letter
                (id, original_id, agent_id, channel_type, recipient,
                 content_json, error, attempts, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                dl_id,
                row[0],       # original_id
                row[1],       # agent_id
                row[2],       # channel_type
                row[3],       # recipient
                row[4],       # content_json
                reason,
                row[5],       # attempts
                now_iso,
            ),
        )

        # Remove from main queue
        await self._db.execute(
            "DELETE FROM delivery_queue WHERE id = ? AND agent_id = ?",
            (entry_id, self._agent_id),
        )

        logger.info("Dead-lettered delivery %s: %s", entry_id, reason)

    async def get_status_counts(self) -> Dict[str, int]:
        """Get counts of entries by status.

        Returns:
            Dict mapping status names to counts, plus dead_letter count.
        """
        counts: Dict[str, int] = {
            "pending": 0,
            "in_flight": 0,
            "delivered": 0,
            "failed": 0,
        }

        rows = await self._db.fetchall(
            """
            SELECT status, COUNT(*) FROM delivery_queue
            WHERE agent_id = ?
            GROUP BY status
            """,
            (self._agent_id,),
        )
        for row in rows:
            if row[0] in counts:
                counts[row[0]] = row[1]

        # Dead letter count
        dl_row = await self._db.fetchone(
            "SELECT COUNT(*) FROM delivery_dead_letter WHERE agent_id = ?",
            (self._agent_id,),
        )
        counts["dead_letter"] = dl_row[0] if dl_row else 0

        return counts

    async def get_pending_entries(self, limit: int = 20) -> List[QueueEntry]:
        """Get pending and failed entries ordered by next retry time.

        Args:
            limit: Maximum entries to return.

        Returns:
            List of QueueEntry objects.
        """
        rows = await self._db.fetchall(
            """
            SELECT id, agent_id, channel_type, recipient, content_json, content_hash,
                   status, attempts, max_retries, next_retry_at, last_error,
                   created_at, delivered_at
            FROM delivery_queue
            WHERE agent_id = ? AND status IN (?, ?)
            ORDER BY next_retry_at ASC
            LIMIT ?
            """,
            (
                self._agent_id,
                DeliveryStatus.PENDING.value,
                DeliveryStatus.FAILED.value,
                limit,
            ),
        )
        return [self._row_to_entry(row) for row in rows]

    async def get_dead_letter_entries(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get entries from the dead letter queue.

        Args:
            limit: Maximum entries to return.

        Returns:
            List of dead letter dicts.
        """
        rows = await self._db.fetchall(
            """
            SELECT id, original_id, agent_id, channel_type, recipient,
                   content_json, error, attempts, created_at
            FROM delivery_dead_letter
            WHERE agent_id = ?
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (self._agent_id, limit),
        )
        results = []
        for row in rows:
            content = {}
            try:
                content = json.loads(row[5]) if row[5] else {}
            except (json.JSONDecodeError, TypeError):
                pass
            results.append({
                "id": row[0],
                "original_id": row[1],
                "agent_id": row[2],
                "channel_type": row[3],
                "recipient": row[4],
                "content": content,
                "error": row[6],
                "attempts": row[7],
                "created_at": row[8],
            })
        return results

    async def purge_delivered(self, older_than_hours: int = 24) -> int:
        """Delete delivered entries older than the given threshold.

        Args:
            older_than_hours: Remove entries delivered more than this many hours ago.

        Returns:
            Number of entries purged.
        """
        cutoff = (
            datetime.now(timezone.utc) - timedelta(hours=older_than_hours)
        ).isoformat()

        # Count first
        row = await self._db.fetchone(
            """
            SELECT COUNT(*) FROM delivery_queue
            WHERE agent_id = ? AND status = ? AND delivered_at < ?
            """,
            (self._agent_id, DeliveryStatus.DELIVERED.value, cutoff),
        )
        count = row[0] if row else 0

        if count > 0:
            await self._db.execute(
                """
                DELETE FROM delivery_queue
                WHERE agent_id = ? AND status = ? AND delivered_at < ?
                """,
                (self._agent_id, DeliveryStatus.DELIVERED.value, cutoff),
            )
            logger.info("Purged %d delivered entries older than %dh", count, older_than_hours)

        return count

    # ------------------------------------------------------------------
    # Background worker
    # ------------------------------------------------------------------

    async def _loop(self):
        """Background poll loop."""
        while self._running:
            try:
                processed = await self.process_pending()
                if processed > 0:
                    logger.debug("Delivery worker processed %d messages", processed)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("DeliveryQueue worker tick error")
            try:
                await asyncio.sleep(self._poll_interval)
            except asyncio.CancelledError:
                raise

    # ------------------------------------------------------------------
    # Delivery attempt
    # ------------------------------------------------------------------

    async def _attempt_delivery(self, entry: QueueEntry) -> None:
        """Attempt to deliver a single message.

        On success: mark as delivered.
        On failure: increment attempts, compute next backoff, or dead-letter.
        """
        # Mark in-flight
        await self._db.execute(
            "UPDATE delivery_queue SET status = ? WHERE id = ? AND agent_id = ?",
            (DeliveryStatus.IN_FLIGHT.value, entry.id, self._agent_id),
        )

        result: DeliveryResult
        if self._deliver:
            try:
                result = await self._deliver(
                    entry.channel_type, entry.recipient, entry.content,
                )
            except Exception as e:
                result = DeliveryResult(success=False, error=str(e))
        elif self._allow_noop_delivery:
            result = DeliveryResult(
                success=True,
                metadata={"noop_delivery": True},
            )
        else:
            task = DeliveryTask(
                id=entry.id,
                agent_id=entry.agent_id,
                channel_type=entry.channel_type,
                recipient=entry.recipient,
                content=entry.content,
            )
            logger.warning(
                "No delivery provider configured for %s/%s; leaving task %s retryable",
                task.channel_type,
                task.recipient,
                task.id,
            )
            result = DeliveryResult(
                success=False,
                error=(
                    "No delivery provider configured; install or register a "
                    f"provider for channel '{task.channel_type}'"
                ),
            )

        new_attempts = entry.attempts + 1

        if result.success:
            now_iso = datetime.now(timezone.utc).isoformat()
            await self._db.execute(
                """
                UPDATE delivery_queue
                SET status = ?, attempts = ?, delivered_at = ?, last_error = NULL
                WHERE id = ? AND agent_id = ?
                """,
                (
                    DeliveryStatus.DELIVERED.value,
                    new_attempts,
                    now_iso,
                    entry.id,
                    self._agent_id,
                ),
            )
            logger.info("Delivered %s after %d attempt(s)", entry.id, new_attempts)
        elif new_attempts >= entry.max_retries:
            # Exhausted retries -- dead letter
            reason = f"Max retries ({entry.max_retries}) exceeded. Last error: {result.error}"
            await self._db.execute(
                """
                UPDATE delivery_queue
                SET status = ?, attempts = ?, last_error = ?
                WHERE id = ? AND agent_id = ?
                """,
                (
                    DeliveryStatus.FAILED.value,
                    new_attempts,
                    result.error,
                    entry.id,
                    self._agent_id,
                ),
            )
            await self.move_to_dead_letter(entry.id, reason)
            logger.warning(
                "Dead-lettered %s after %d attempts: %s",
                entry.id, new_attempts, result.error,
            )
        else:
            # Schedule retry with exponential backoff
            delay = _compute_backoff(new_attempts)
            next_retry = (
                datetime.now(timezone.utc) + timedelta(seconds=delay)
            ).isoformat()
            await self._db.execute(
                """
                UPDATE delivery_queue
                SET status = ?, attempts = ?, next_retry_at = ?, last_error = ?
                WHERE id = ? AND agent_id = ?
                """,
                (
                    DeliveryStatus.FAILED.value,
                    new_attempts,
                    next_retry,
                    result.error,
                    entry.id,
                    self._agent_id,
                ),
            )
            logger.info(
                "Delivery %s failed (attempt %d/%d), retry in %.0fs: %s",
                entry.id, new_attempts, entry.max_retries, delay, result.error,
            )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _row_to_entry(row) -> QueueEntry:
        """Convert a database row tuple to a QueueEntry."""
        return QueueEntry(
            id=row[0],
            agent_id=row[1],
            channel_type=row[2],
            recipient=row[3],
            content_json=row[4] or "{}",
            content_hash=row[5],
            status=DeliveryStatus(row[6]),
            attempts=row[7],
            max_retries=row[8],
            next_retry_at=row[9],
            last_error=row[10],
            created_at=row[11],
            delivered_at=row[12],
        )

    async def _ensure_tables(self):
        """Create the delivery_queue and delivery_dead_letter tables if needed."""
        await self._db.execute(
            """
            CREATE TABLE IF NOT EXISTS delivery_queue (
                id TEXT PRIMARY KEY,
                agent_id TEXT NOT NULL,
                channel_type TEXT NOT NULL,
                recipient TEXT NOT NULL,
                content_json TEXT NOT NULL DEFAULT '{}',
                content_hash TEXT,
                status TEXT NOT NULL DEFAULT 'pending',
                attempts INTEGER NOT NULL DEFAULT 0,
                max_retries INTEGER NOT NULL DEFAULT 5,
                next_retry_at TEXT,
                last_error TEXT,
                created_at TEXT NOT NULL,
                delivered_at TEXT
            )
            """
        )
        await self._db.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_delivery_queue_agent_status
            ON delivery_queue(agent_id, status, next_retry_at)
            """
        )
        await self._db.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_delivery_queue_dedup
            ON delivery_queue(agent_id, content_hash, recipient, created_at)
            """
        )
        await self._db.execute(
            """
            CREATE TABLE IF NOT EXISTS delivery_dead_letter (
                id TEXT PRIMARY KEY,
                original_id TEXT NOT NULL,
                agent_id TEXT NOT NULL,
                channel_type TEXT NOT NULL,
                recipient TEXT NOT NULL,
                content_json TEXT NOT NULL DEFAULT '{}',
                error TEXT,
                attempts INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL
            )
            """
        )
        await self._db.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_delivery_dead_letter_agent
            ON delivery_dead_letter(agent_id, created_at DESC)
            """
        )
