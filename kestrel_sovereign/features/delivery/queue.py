"""
Durable delivery queue with retry logic, exponential backoff, and dead letter queue.

The DeliveryQueue persists messages in SQLite, processes them via a background
asyncio task, and moves exhausted entries to a dead letter table after max
retries are exceeded.

Retry backoff formula: base_delay * (5 ** attempt), capped at 1 hour.
Default: 5s -> 25s -> 2m5s -> 10m25s -> 52m5s

Deduplication: legacy and canonical content hashes + recipient within a
60-second window prevent duplicate enqueues of the same JSON message, whether
or not the caller also supplies an idempotency key. The split preserves rolling
compatibility while making mapping order semantically irrelevant.

Callers that need durable replay safety can additionally supply an opaque
idempotency key. Its SHA-256 digest is scoped to the owning agent in a separate
ledger, which avoids storing the raw key but is not a confidentiality boundary.
Safe replays adopt the canonical queue ID; reusing the key for a different
request fails closed.
"""

import asyncio
import hashlib
import json
import logging
import math
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
CANONICAL_BACKFILL_BATCH_SIZE = 500

# Type alias for the delivery callback
DeliveryCallback = Callable[[str, str, Dict[str, Any]], Coroutine[Any, Any, DeliveryResult]]

MAX_IDEMPOTENCY_KEY_BYTES = 4096


class DeliveryIdempotencyError(RuntimeError):
    """Base class for durable delivery idempotency failures."""


class DeliveryIdempotencyConflict(DeliveryIdempotencyError):
    """Raised when an idempotency key is replayed with a different request."""


class DeliveryIdempotencyTerminal(DeliveryIdempotencyError):
    """Raised when a replay targets a delivery that is in dead-letter state."""


class DeliveryIdempotencyStateError(DeliveryIdempotencyError):
    """Raised when the durable replay ledger cannot be reconciled safely."""


def _canonical_content_json(
    content: Dict[str, Any], *, allow_string_fallback: bool
) -> str:
    """Return stable JSON for deduplication and durable request identity."""
    if not allow_string_fallback:
        _validate_json_value(content, path="content")
    try:
        # Preserve the legacy unkeyed API's ``json.dumps(default=str)``
        # acceptance before sorting. Sorting the caller's mapping directly
        # rejects otherwise-supported mixtures such as string and integer
        # keys because Python cannot order those key types. A JSON round trip
        # first applies the encoder's historical key coercion and value
        # fallback; the second pass can then produce a stable representation.
        canonical_value = content
        if allow_string_fallback:
            canonical_value = json.loads(
                json.dumps(content, default=str, allow_nan=True)
            )
        return json.dumps(
            canonical_value,
            default=None,
            allow_nan=allow_string_fallback,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as error:
        raise ValueError(
            "idempotent delivery content must contain only JSON-serializable values"
        ) from error


def _persisted_content_hashes(recipient: str, content_json: str) -> tuple[str, str]:
    """Return legacy and canonical hashes for an already-persisted payload."""
    raw_hash = QueueEntry.compute_content_hash(recipient, content_json)
    try:
        content = json.loads(content_json)
    except (json.JSONDecodeError, TypeError):
        return raw_hash, raw_hash
    canonical_json = _canonical_content_json(content, allow_string_fallback=True)
    return (
        raw_hash,
        QueueEntry.compute_content_hash(recipient, canonical_json),
    )


def _validate_json_value(
    value: Any, *, path: str, _active: Optional[set[int]] = None
) -> None:
    """Reject lossy Python-to-JSON coercions and name the invalid value path."""
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if math.isfinite(value):
            return
        raise ValueError(f"idempotent delivery {path} must be a finite JSON number")
    if isinstance(value, (list, dict)):
        active = _active if _active is not None else set()
        identity = id(value)
        if identity in active:
            raise ValueError(f"idempotent delivery {path} contains a JSON cycle")
        active.add(identity)
        try:
            if isinstance(value, list):
                for index, item in enumerate(value):
                    _validate_json_value(
                        item, path=f"{path}[{index}]", _active=active
                    )
            else:
                for key, item in value.items():
                    if not isinstance(key, str):
                        raise ValueError(
                            f"idempotent delivery {path} has non-string JSON key {key!r}"
                        )
                    _validate_json_value(item, path=f"{path}.{key}", _active=active)
        finally:
            active.remove(identity)
        return
    raise ValueError(
        f"idempotent delivery {path} contains non-JSON value "
        f"of type {type(value).__name__}"
    )


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
        await self._reclaim_in_flight()
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

    async def _reclaim_in_flight(self) -> int:
        """Requeue rows left IN_FLIGHT by a crash/restart.

        DeliveryQueue is single-worker-per-agent, so any IN_FLIGHT row observed
        at start() is by definition stale -- the previous worker died mid-flight
        and will never resolve it. Move such rows back to PENDING (preserving
        ``attempts``, resetting ``next_retry_at`` to now) so the worker retries
        them. Providers already tolerate at-least-once redelivery via the
        existing retry semantics, so requeue-on-startup is safe.

        Returns:
            Number of entries reclaimed.
        """
        now_iso = datetime.now(timezone.utc).isoformat()

        row = await self._db.fetchone(
            """
            SELECT COUNT(*) FROM delivery_queue
            WHERE agent_id = ? AND status = ?
              AND NOT EXISTS (
                  SELECT 1 FROM delivery_dead_letter
                  WHERE delivery_dead_letter.agent_id = delivery_queue.agent_id
                    AND delivery_dead_letter.original_id = delivery_queue.id
              )
              AND NOT EXISTS (
                  SELECT 1 FROM delivery_dead_letter
                  WHERE delivery_dead_letter.agent_id = delivery_queue.agent_id
                    AND delivery_dead_letter.retry_entry_id = delivery_queue.id
              )
            """,
            (self._agent_id, DeliveryStatus.IN_FLIGHT.value),
        )
        count = row[0] if row else 0

        if count > 0:
            await self._db.execute(
                """
                UPDATE delivery_queue
                SET status = ?, next_retry_at = ?
                WHERE agent_id = ? AND status = ?
                  AND NOT EXISTS (
                      SELECT 1 FROM delivery_dead_letter
                      WHERE delivery_dead_letter.agent_id = delivery_queue.agent_id
                        AND delivery_dead_letter.original_id = delivery_queue.id
                  )
                  AND NOT EXISTS (
                      SELECT 1 FROM delivery_dead_letter
                      WHERE delivery_dead_letter.agent_id = delivery_queue.agent_id
                        AND delivery_dead_letter.retry_entry_id = delivery_queue.id
                  )
                """,
                (
                    DeliveryStatus.PENDING.value,
                    now_iso,
                    self._agent_id,
                    DeliveryStatus.IN_FLIGHT.value,
                ),
            )
            logger.warning(
                "Reclaimed %d stale in-flight delivery ent(ies) on start", count
            )

        return count

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def enqueue(
        self,
        channel_type: str,
        recipient: str,
        content: Dict[str, Any],
        max_retries: Optional[int] = None,
        idempotency_key: Optional[str] = None,
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
            idempotency_key: Optional opaque, owner-scoped replay key. Only its
                SHA-256 digest is persisted (this avoids storing the raw value,
                but does not make a guessable key secret). A replay must preserve
                whether ``max_retries`` was omitted or explicitly supplied; a
                different request fails closed.

        Returns:
            The queue entry ID (existing if deduplicated, new otherwise).

        Raises:
            ValueError: The key is empty/oversized or keyed content is not JSON.
            DeliveryIdempotencyConflict: The key names a different request.
            DeliveryIdempotencyTerminal: The keyed delivery is dead-lettered.
            DeliveryIdempotencyStateError: Durable state cannot be reconciled.
        """
        retries = max_retries if max_retries is not None else self._max_retries

        if idempotency_key is not None:
            return await self._enqueue_idempotent(
                channel_type=channel_type,
                recipient=recipient,
                content=content,
                retries=retries,
                requested_max_retries=max_retries,
                idempotency_key=idempotency_key,
            )

        # Snapshot permissive legacy values exactly once. Some supported
        # ``default=str`` objects are stateful, so serializing the caller's
        # object again could assign this row a hash for content it did not
        # persist.
        content_json = json.dumps(content, default=str)
        legacy_content_hash, canonical_content_hash = _persisted_content_hashes(
            recipient, content_json
        )

        dedup_cutoff = (
            datetime.now(timezone.utc) - timedelta(seconds=DEDUP_WINDOW_SECONDS)
        ).isoformat()
        entry_id = str(uuid.uuid4())
        now_iso = datetime.now(timezone.utc).isoformat()

        # Plain and keyed enqueues share one content-scoped serialization gate.
        # SQLite's IMMEDIATE transaction owns the writer slot; PostgreSQL also
        # needs the same advisory transaction lock used by keyed requests.
        async with self._db.transaction(immediate=True):
            await self._lock_content_dedup(recipient, canonical_content_hash)
            existing = await self._find_recent_duplicate(
                recipient=recipient,
                canonical_content_hash=canonical_content_hash,
                legacy_content_hash=legacy_content_hash,
                dedup_cutoff=dedup_cutoff,
            )
            if existing is not None:
                logger.debug("Deduplicated delivery entry: %s", existing)
                return existing

            await self._db.execute(
                """
                INSERT INTO delivery_queue
                    (id, agent_id, channel_type, recipient, content_json,
                     content_hash, canonical_content_hash, status, attempts,
                     max_retries, next_retry_at, last_error, created_at,
                     delivered_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0, ?, ?, NULL, ?, NULL)
                """,
                (
                    entry_id,
                    self._agent_id,
                    channel_type,
                    recipient,
                    content_json,
                    legacy_content_hash,
                    canonical_content_hash,
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

    async def _lock_content_dedup(
        self, recipient: str, canonical_content_hash: str
    ) -> None:
        """Serialize one owner's content-dedup decision on PostgreSQL."""
        if self._db.backend_type != "postgres":
            return
        lock_identity = (
            f"delivery-dedup:{self._agent_id}:{recipient}:"
            f"{canonical_content_hash}"
        )
        await self._db.fetchone(
            "SELECT pg_advisory_xact_lock(hashtextextended(?, 0))",
            (lock_identity,),
        )

    async def _find_recent_duplicate(
        self,
        *,
        recipient: str,
        canonical_content_hash: str,
        legacy_content_hash: str,
        dedup_cutoff: str,
        channel_type: Optional[str] = None,
        max_retries: Optional[int] = None,
    ) -> Optional[str]:
        """Find an eligible recent duplicate and repair rolling-writer hashes."""
        request_filter = ""
        request_params: tuple[Any, ...] = ()
        if channel_type is not None and max_retries is not None:
            request_filter = (
                " AND delivery_queue.channel_type = ?"
                " AND delivery_queue.max_retries = ?"
            )
            request_params = (channel_type, max_retries)
        existing = await self._db.fetchone(
            f"""
            SELECT id FROM (
                SELECT delivery_queue.id, delivery_queue.created_at
                FROM delivery_queue
                WHERE delivery_queue.agent_id = ?
                      AND delivery_queue.content_hash IN (?, ?)
                      AND delivery_queue.recipient = ?
                      AND delivery_queue.created_at >= ?
                      {request_filter}
                      AND NOT EXISTS (
                          SELECT 1 FROM delivery_dead_letter
                          WHERE delivery_dead_letter.agent_id = delivery_queue.agent_id
                            AND delivery_dead_letter.original_id = delivery_queue.id
                      )
                      AND NOT EXISTS (
                          SELECT 1 FROM delivery_dead_letter
                          WHERE delivery_dead_letter.agent_id = delivery_queue.agent_id
                            AND delivery_dead_letter.retry_entry_id = delivery_queue.id
                      )
                UNION ALL
                SELECT delivery_queue.id, delivery_queue.created_at
                FROM delivery_queue
                WHERE delivery_queue.agent_id = ?
                      AND delivery_queue.canonical_content_hash = ?
                      AND delivery_queue.recipient = ?
                      AND delivery_queue.created_at >= ?
                      {request_filter}
                      AND NOT EXISTS (
                          SELECT 1 FROM delivery_dead_letter
                          WHERE delivery_dead_letter.agent_id = delivery_queue.agent_id
                            AND delivery_dead_letter.original_id = delivery_queue.id
                      )
                      AND NOT EXISTS (
                          SELECT 1 FROM delivery_dead_letter
                          WHERE delivery_dead_letter.agent_id = delivery_queue.agent_id
                            AND delivery_dead_letter.retry_entry_id = delivery_queue.id
                      )
            ) AS candidates
            ORDER BY created_at DESC
            LIMIT 1
            """,
            (
                self._agent_id,
                canonical_content_hash,
                legacy_content_hash,
                recipient,
                dedup_cutoff,
                *request_params,
                self._agent_id,
                canonical_content_hash,
                recipient,
                dedup_cutoff,
                *request_params,
            ),
        )
        if existing is not None:
            return existing[0]

        # An old process in a rolling deployment can insert a NULL canonical
        # hash after startup backfill. Reconcile relevant rows on every enqueue;
        # the partial index keeps this bounded to legacy-writer residue.
        missing = await self._db.fetchall(
            f"""
            SELECT delivery_queue.id, delivery_queue.content_json
            FROM delivery_queue
            WHERE delivery_queue.agent_id = ?
                  AND delivery_queue.recipient = ?
                  AND delivery_queue.created_at >= ?
                  AND delivery_queue.canonical_content_hash IS NULL
                  {request_filter}
                  AND NOT EXISTS (
                      SELECT 1 FROM delivery_dead_letter
                      WHERE delivery_dead_letter.agent_id = delivery_queue.agent_id
                        AND delivery_dead_letter.original_id = delivery_queue.id
                  )
                  AND NOT EXISTS (
                      SELECT 1 FROM delivery_dead_letter
                      WHERE delivery_dead_letter.agent_id = delivery_queue.agent_id
                        AND delivery_dead_letter.retry_entry_id = delivery_queue.id
                  )
            ORDER BY delivery_queue.created_at DESC
            """,
            (self._agent_id, recipient, dedup_cutoff, *request_params),
        )
        for missing_id, persisted_json in missing:
            _, repaired_hash = _persisted_content_hashes(
                recipient, persisted_json or "{}"
            )
            await self._db.execute(
                """
                UPDATE delivery_queue SET canonical_content_hash = ?
                WHERE id = ? AND agent_id = ?
                      AND canonical_content_hash IS NULL
                """,
                (repaired_hash, missing_id, self._agent_id),
            )
            if repaired_hash == canonical_content_hash:
                return missing_id
        return None

    async def _has_unlinked_compatible_queue_row(
        self,
        *,
        recipient: str,
        canonical_content_hash: str,
        legacy_content_hash: str,
        channel_type: str,
        claim_created_at: str,
    ) -> bool:
        """Detect an unreconciled rolling-writer retry outside dedup time.

        Adopting such a row could collapse a legitimate independent delivery,
        while inserting another row could duplicate an older process's retry.
        The only safe automatic outcome is therefore to fail closed.
        """
        eligibility = """
            AND delivery_queue.recipient = ?
            AND delivery_queue.channel_type = ?
            AND delivery_queue.created_at >= ?
            AND NOT EXISTS (
                  SELECT 1 FROM delivery_dead_letter
                  WHERE delivery_dead_letter.agent_id = delivery_queue.agent_id
                    AND delivery_dead_letter.original_id = delivery_queue.id
            )
            AND NOT EXISTS (
                  SELECT 1 FROM delivery_dead_letter
                  WHERE delivery_dead_letter.agent_id = delivery_queue.agent_id
                    AND delivery_dead_letter.retry_entry_id = delivery_queue.id
            )
            AND NOT EXISTS (
                  SELECT 1 FROM delivery_idempotency
                  WHERE delivery_idempotency.agent_id = delivery_queue.agent_id
                    AND delivery_idempotency.entry_id = delivery_queue.id
            )
        """
        row = await self._db.fetchone(
            f"""
            SELECT id FROM (
                SELECT delivery_queue.id
                FROM delivery_queue
                WHERE delivery_queue.agent_id = ?
                  AND delivery_queue.canonical_content_hash = ?
                  {eligibility}
                UNION ALL
                SELECT delivery_queue.id
                FROM delivery_queue
                WHERE delivery_queue.agent_id = ?
                  AND delivery_queue.content_hash IN (?, ?)
                  {eligibility}
            ) AS compatible_queue_rows
            LIMIT 1
            """,
            (
                self._agent_id,
                canonical_content_hash,
                recipient,
                channel_type,
                claim_created_at,
                self._agent_id,
                canonical_content_hash,
                legacy_content_hash,
                recipient,
                channel_type,
                claim_created_at,
            ),
        )
        return row is not None

    async def _find_replacement_for_prior_entry(
        self,
        *,
        prior_entry_id: str,
        recipient: str,
        canonical_content_hash: str,
        legacy_content_hash: str,
        channel_type: str,
        max_retries: int,
    ) -> Optional[str]:
        """Find a replacement durably anchored to a shared prior queue ID."""
        rows = await self._db.fetchall(
            """
            SELECT DISTINCT delivery_queue.id
            FROM delivery_queue
            JOIN delivery_idempotency replacement_claim
              ON replacement_claim.agent_id = delivery_queue.agent_id
             AND replacement_claim.entry_id = delivery_queue.id
            WHERE delivery_queue.agent_id = ?
              AND replacement_claim.previous_entry_id = ?
              AND delivery_queue.recipient = ?
              AND delivery_queue.channel_type = ?
              AND delivery_queue.max_retries = ?
              AND (
                    delivery_queue.canonical_content_hash = ?
                    OR delivery_queue.content_hash IN (?, ?)
              )
              AND NOT EXISTS (
                    SELECT 1 FROM delivery_dead_letter
                    WHERE delivery_dead_letter.agent_id = delivery_queue.agent_id
                      AND delivery_dead_letter.original_id = delivery_queue.id
              )
              AND NOT EXISTS (
                    SELECT 1 FROM delivery_dead_letter
                    WHERE delivery_dead_letter.agent_id = delivery_queue.agent_id
                      AND delivery_dead_letter.retry_entry_id = delivery_queue.id
              )
            LIMIT 2
            """,
            (
                self._agent_id,
                prior_entry_id,
                recipient,
                channel_type,
                max_retries,
                canonical_content_hash,
                canonical_content_hash,
                legacy_content_hash,
            ),
        )
        if len(rows) > 1:
            raise DeliveryIdempotencyStateError(
                "stale delivery claim has multiple anchored replacements"
            )
        return rows[0][0] if rows else None

    async def _lock_dead_letter(self, entry_id: str) -> Optional[tuple[Any, ...]]:
        """Lock and return a tombstone addressed by any durable queue identity."""
        identity_lookup = """
            SELECT id FROM (
                SELECT id, 0 AS identity_priority
                FROM delivery_dead_letter
                WHERE agent_id = ? AND id = ?
                UNION ALL
                SELECT id, 1 AS identity_priority
                FROM delivery_dead_letter
                WHERE agent_id = ? AND original_id = ?
                UNION ALL
                SELECT id, 2 AS identity_priority
                FROM delivery_dead_letter
                WHERE agent_id = ? AND retry_entry_id = ?
            ) AS matching_dead_letter
            ORDER BY identity_priority
            LIMIT 1
        """
        identity_params = (
            self._agent_id,
            entry_id,
            self._agent_id,
            entry_id,
            self._agent_id,
            entry_id,
        )
        locked = await self._db.execute(
            f"""
            UPDATE delivery_dead_letter SET id = id
            WHERE agent_id = ? AND id = ({identity_lookup})
            """,
            (self._agent_id, *identity_params),
        )
        if locked == 0:
            return None
        return await self._db.fetchone(
            f"""
            SELECT id, original_id, agent_id, channel_type, recipient,
                   content_json, error, attempts, created_at, max_retries,
                   retry_entry_id, legacy_content_hash
            FROM delivery_dead_letter
            WHERE agent_id = ? AND id = ({identity_lookup})
            """,
            (self._agent_id, *identity_params),
        )

    async def _find_dead_letter_for_queue_id(
        self, entry_id: str
    ) -> Optional[tuple[Any, ...]]:
        """Find a tombstone through independently indexable queue identities."""
        return await self._db.fetchone(
            """
            SELECT id FROM (
                SELECT id, 0 AS identity_priority
                FROM delivery_dead_letter
                WHERE agent_id = ? AND original_id = ?
                UNION ALL
                SELECT id, 1 AS identity_priority
                FROM delivery_dead_letter
                WHERE agent_id = ? AND retry_entry_id = ?
            ) AS matching_dead_letter
            ORDER BY identity_priority
            LIMIT 1
            """,
            (self._agent_id, entry_id, self._agent_id, entry_id),
        )

    async def _enqueue_idempotent(
        self,
        *,
        channel_type: str,
        recipient: str,
        content: Dict[str, Any],
        retries: int,
        requested_max_retries: Optional[int],
        idempotency_key: str,
    ) -> str:
        """Atomically insert or adopt an owner-scoped logical delivery."""
        key_bytes = idempotency_key.encode("utf-8")
        if not key_bytes:
            raise ValueError("idempotency_key must not be empty")
        if len(key_bytes) > MAX_IDEMPOTENCY_KEY_BYTES:
            raise ValueError(
                f"idempotency_key must be at most {MAX_IDEMPOTENCY_KEY_BYTES} UTF-8 bytes"
            )

        content_json = _canonical_content_json(
            content, allow_string_fallback=False
        )
        key_digest = hashlib.sha256(key_bytes).hexdigest()
        payload_json = json.dumps(
            {
                "channel_type": channel_type,
                "recipient": recipient,
                "content": json.loads(content_json),
                # Preserve the caller's request identity across restarts even
                # if this queue instance's configured default has changed.
                "max_retries": requested_max_retries,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        payload_digest = hashlib.sha256(payload_json.encode("utf-8")).hexdigest()
        entry_id = str(uuid.uuid4())
        candidate_id = entry_id
        now_iso = datetime.now(timezone.utc).isoformat()
        canonical_content_hash = QueueEntry.compute_content_hash(
            recipient, content_json
        )
        legacy_content_hash = QueueEntry.compute_content_hash(
            recipient, json.dumps(content, default=str)
        )
        dedup_cutoff = (
            datetime.now(timezone.utc) - timedelta(seconds=DEDUP_WINDOW_SECONDS)
        ).isoformat()
        nesting_strategy = getattr(self._db, "nested_transaction_strategy", None)
        if nesting_strategy not in {"savepoint", "joined"}:
            raise DeliveryIdempotencyStateError(
                "database backend does not declare safe nested transaction semantics"
            )

        try:
            async with self._db.transaction(immediate=True):
                try:
                    await self._lock_content_dedup(
                        recipient, canonical_content_hash
                    )
                    deduplicated = await self._find_recent_duplicate(
                        recipient=recipient,
                        canonical_content_hash=canonical_content_hash,
                        legacy_content_hash=legacy_content_hash,
                        dedup_cutoff=dedup_cutoff,
                        channel_type=channel_type,
                        max_retries=retries,
                    )
                    # Always claim with a fresh unguessable candidate. That lets
                    # ambiguous INSERT completion compensate by exact entry ID
                    # without risking a pre-existing claim that merely shares a
                    # 60-second dedup target.
                    await self._db.execute(
                        """
                        INSERT INTO delivery_idempotency
                            (agent_id, idempotency_key_digest, entry_id,
                             payload_digest, created_at, effective_max_retries,
                             legacy_content_hash)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT (agent_id, idempotency_key_digest) DO NOTHING
                        """,
                        (
                            self._agent_id,
                            key_digest,
                            candidate_id,
                            payload_digest,
                            now_iso,
                            retries,
                            legacy_content_hash,
                        ),
                    )
                    # Lock the canonical ledger row portably. SQLite already
                    # owns the sole writer slot; PostgreSQL's no-op UPDATE waits
                    # for and locks a concurrently inserted claim.
                    await self._db.execute(
                        """
                        UPDATE delivery_idempotency SET entry_id = entry_id
                        WHERE agent_id = ? AND idempotency_key_digest = ?
                        """,
                        (self._agent_id, key_digest),
                    )
                    existing = await self._db.fetchone(
                        """
                        SELECT entry_id, payload_digest, effective_max_retries,
                               created_at, legacy_content_hash,
                               previous_entry_id
                        FROM delivery_idempotency
                        WHERE agent_id = ? AND idempotency_key_digest = ?
                        """,
                        (self._agent_id, key_digest),
                    )
                    if existing is None:
                        raise DeliveryIdempotencyStateError(
                            "delivery idempotency record was not persisted"
                        )
                    if existing[1] != payload_digest:
                        raise DeliveryIdempotencyConflict(
                            "idempotency_key was already used for a different "
                            "delivery request"
                        )

                    stored_retries = existing[2]
                    claim_created_at = existing[3]
                    stored_legacy_hash = existing[4]
                    previous_entry_id = existing[5]

                    canonical_id = existing[0]
                    dead_letter = await self._find_dead_letter_for_queue_id(
                        canonical_id
                    )
                    if dead_letter is not None:
                        raise DeliveryIdempotencyTerminal(
                            "idempotent delivery is in the dead-letter queue; "
                            "retry that entry explicitly before replaying it"
                        )

                    queue_row = await self._db.fetchone(
                        """
                        SELECT status, max_retries, content_hash
                        FROM delivery_queue
                        WHERE id = ? AND agent_id = ?
                        """,
                        (canonical_id, self._agent_id),
                    )
                    if queue_row is not None:
                        if stored_retries is None:
                            if previous_entry_id is not None:
                                # A replacement anchor proves this claim was
                                # already stale and followed another claim's
                                # repair. It therefore has no live pre-upgrade
                                # row from which its own policy can be learned.
                                raise DeliveryIdempotencyStateError(
                                    "stale delivery idempotency record has no "
                                    "durable retry policy"
                                )
                            # Upgrade old ledger rows lazily from their live
                            # queue entry. Keeping the column nullable lets old
                            # rolling-deployment writers continue inserting.
                            stored_retries = queue_row[1]
                            await self._db.execute(
                                """
                                UPDATE delivery_idempotency
                                SET effective_max_retries = ?
                                WHERE agent_id = ?
                                  AND idempotency_key_digest = ?
                                  AND effective_max_retries IS NULL
                                """,
                                (stored_retries, self._agent_id, key_digest),
                            )
                        if stored_legacy_hash is None:
                            stored_legacy_hash = queue_row[2]
                            await self._db.execute(
                                """
                                UPDATE delivery_idempotency
                                SET legacy_content_hash = ?
                                WHERE agent_id = ?
                                  AND idempotency_key_digest = ?
                                  AND legacy_content_hash IS NULL
                                """,
                                (stored_legacy_hash, self._agent_id, key_digest),
                            )
                        logger.debug(
                            "Adopted idempotent delivery entry: %s", canonical_id
                        )
                        return canonical_id

                    # PostgreSQL READ COMMITTED takes a new snapshot for each
                    # statement. A dead-letter move may therefore commit after
                    # the optimistic tombstone read above but before the queue
                    # lookup. Lock and re-check the tombstone before treating
                    # the missing row as stale; SQLite's writer transaction
                    # makes this harmless but keeps the contract identical.
                    dead_letter = await self._lock_dead_letter(canonical_id)
                    if dead_letter is not None:
                        raise DeliveryIdempotencyTerminal(
                            "idempotent delivery is in the dead-letter queue; "
                            "retry that entry explicitly before replaying it"
                        )

                    if stored_retries is None:
                        # A pre-upgrade orphan contains no recoverable record
                        # of the effective default. Recreating it under today's
                        # policy would silently change the logical request.
                        raise DeliveryIdempotencyStateError(
                            "stale delivery idempotency record has no durable "
                            "retry policy"
                        )

                    if stored_retries != retries:
                        # Omitted defaults are intentionally excluded from the
                        # request digest, but stale repair must retain the
                        # original effective policy. Re-evaluate any content
                        # adoption under that policy before recreating the row.
                        deduplicated = await self._find_recent_duplicate(
                            recipient=recipient,
                            canonical_content_hash=canonical_content_hash,
                            legacy_content_hash=legacy_content_hash,
                            dedup_cutoff=dedup_cutoff,
                            channel_type=channel_type,
                            max_retries=stored_retries,
                        )

                    if deduplicated is not None:
                        adopted_row = await self._db.fetchone(
                            """
                            SELECT content_hash FROM delivery_queue
                            WHERE id = ? AND agent_id = ?
                            """,
                            (deduplicated, self._agent_id),
                        )
                        if adopted_row is None:
                            raise DeliveryIdempotencyStateError(
                                "deduplicated delivery disappeared while locked"
                            )
                        await self._db.execute(
                            """
                            UPDATE delivery_idempotency
                            SET entry_id = ?, created_at = ?, compensating = 0,
                                previous_entry_id = ?,
                                legacy_content_hash = ?
                            WHERE agent_id = ? AND entry_id = ?
                            """,
                            (
                                deduplicated,
                                now_iso,
                                canonical_id,
                                adopted_row[0],
                                self._agent_id,
                                canonical_id,
                            ),
                        )
                        logger.debug(
                            "Mapped idempotent delivery claim to deduplicated "
                            "entry: %s",
                            deduplicated,
                        )
                        return deduplicated

                    anchored_replacement = await self._find_replacement_for_prior_entry(
                        prior_entry_id=canonical_id,
                        recipient=recipient,
                        canonical_content_hash=canonical_content_hash,
                        legacy_content_hash=stored_legacy_hash
                        or legacy_content_hash,
                        channel_type=channel_type,
                        max_retries=stored_retries,
                    )
                    if anchored_replacement is not None:
                        await self._db.execute(
                            """
                            UPDATE delivery_idempotency
                            SET entry_id = ?, created_at = ?, compensating = 0,
                                previous_entry_id = ?
                            WHERE agent_id = ? AND entry_id = ?
                            """,
                            (
                                anchored_replacement,
                                now_iso,
                                canonical_id,
                                self._agent_id,
                                canonical_id,
                            ),
                        )
                        return anchored_replacement

                    if (
                        canonical_id != candidate_id
                        and await self._has_unlinked_compatible_queue_row(
                            recipient=recipient,
                            canonical_content_hash=canonical_content_hash,
                            legacy_content_hash=stored_legacy_hash
                            or legacy_content_hash,
                            channel_type=channel_type,
                            claim_created_at=claim_created_at,
                        )
                    ):
                        raise DeliveryIdempotencyStateError(
                            "stale delivery idempotency record has an unlinked "
                            "compatible queue row; manual reconciliation is required"
                        )

                    if canonical_id != candidate_id:
                        # The queue row was removed independently of its ledger
                        # (or by a pre-v0.53.12 purge). Repair the claim under the
                        # row lock and recreate the logical delivery.
                        await self._db.execute(
                            """
                            UPDATE delivery_idempotency
                            SET entry_id = ?, created_at = ?, compensating = 0,
                                previous_entry_id = ?
                            WHERE agent_id = ? AND idempotency_key_digest = ?
                                  AND entry_id = ?
                            """,
                            (
                                candidate_id,
                                now_iso,
                                canonical_id,
                                self._agent_id,
                                key_digest,
                                canonical_id,
                            ),
                        )

                    await self._db.execute(
                        """
                        INSERT INTO delivery_queue
                            (id, agent_id, channel_type, recipient, content_json,
                             content_hash, canonical_content_hash,
                             status, attempts, max_retries,
                             next_retry_at, last_error, created_at, delivered_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0, ?, ?, NULL, ?, NULL)
                        """,
                        (
                            candidate_id,
                            self._agent_id,
                            channel_type,
                            recipient,
                            content_json,
                            stored_legacy_hash or legacy_content_hash,
                            canonical_content_hash,
                            DeliveryStatus.PENDING.value,
                            stored_retries,
                            now_iso,
                            now_iso,
                        ),
                    )
                    if canonical_id != candidate_id:
                        await self._db.execute(
                            """
                            UPDATE delivery_idempotency
                            SET entry_id = ?, created_at = ?, compensating = 0,
                                previous_entry_id = ?
                            WHERE agent_id = ? AND entry_id = ?
                            """,
                            (
                                candidate_id,
                                now_iso,
                                canonical_id,
                                self._agent_id,
                                canonical_id,
                            ),
                        )
                    # Keep previous_entry_id as a durable compensation anchor.
                    # If the INSERT above completed but cancellation was
                    # reported ambiguously, a joined SQLite caller may catch
                    # the cancellation and commit. The trigger must still be
                    # able to restore the prior fail-closed claim.
                except BaseException:
                    if nesting_strategy == "joined":
                        # A joined nested transaction cannot roll back only this
                        # call when its caller catches the error and commits the
                        # outer scope. One marker UPDATE invokes a scoped trigger
                        # that removes this candidate queue row and ledger claim
                        # atomically. Savepoint backends roll back normally.
                        await self._db.execute(
                            """
                            UPDATE delivery_idempotency SET compensating = 1
                            WHERE agent_id = ? AND idempotency_key_digest = ?
                                  AND entry_id = ? AND compensating = 0
                            """,
                            (self._agent_id, key_digest, candidate_id),
                        )
                    raise
        except Exception as error:
            public_error = self._find_idempotency_error(error)
            if public_error is not None:
                raise public_error
            raise

        logger.info(
            "Enqueued idempotent delivery %s -> %s/%s",
            entry_id,
            channel_type,
            recipient,
        )
        return candidate_id

    @staticmethod
    def _find_idempotency_error(
        error: BaseException,
    ) -> Optional[DeliveryIdempotencyError]:
        """Recover a public idempotency error from explicit backend causes."""
        seen: set[int] = set()
        current: Optional[BaseException] = error
        while current is not None and id(current) not in seen:
            if isinstance(current, DeliveryIdempotencyError):
                return current
            seen.add(id(current))
            current = current.__cause__
        return None

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
                  AND NOT EXISTS (
                      SELECT 1 FROM delivery_dead_letter
                      WHERE delivery_dead_letter.agent_id = delivery_queue.agent_id
                        AND delivery_dead_letter.original_id = delivery_queue.id
                  )
                  AND NOT EXISTS (
                      SELECT 1 FROM delivery_dead_letter
                      WHERE delivery_dead_letter.agent_id = delivery_queue.agent_id
                        AND delivery_dead_letter.retry_entry_id = delivery_queue.id
                  )
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

        Raises:
            DeliveryIdempotencyStateError: A durable retry transition cannot
                be reconciled safely.
        """
        try:
            return await self._retry_entry(entry_id)
        except Exception as error:
            public_error = self._find_idempotency_error(error)
            if public_error is not None:
                raise public_error
            raise

    async def _retry_entry(self, entry_id: str) -> Dict[str, Any]:
        """Execute a retry while allowing the public wrapper to unwrap errors."""
        # A dead-letter tombstone is authoritative even if a joined SQLite
        # caller committed the recoverable live+tombstone intermediate state.
        # Check and lock it before considering the main queue row.
        async with self._db.transaction(immediate=True):
            dl_row = await self._lock_dead_letter(entry_id)
            queue_locked = None
            if dl_row is None:
                # Lock the live row before deciding its status. A concurrent
                # dead-letter move can win between the first tombstone miss and
                # this lock attempt, so a missing live row requires one final
                # tombstone check inside the same transaction.
                queue_locked = await self._db.execute(
                    """
                    UPDATE delivery_queue SET id = id
                    WHERE id = ? AND agent_id = ?
                    """,
                    (entry_id, self._agent_id),
                )
                if queue_locked == 0:
                    dl_row = await self._lock_dead_letter(entry_id)
                    if dl_row is None:
                        return {
                            "success": False,
                            "error": f"Entry {entry_id} not found or already retried",
                        }

            if dl_row is not None:
                now_iso = datetime.now(timezone.utc).isoformat()
                new_id = dl_row[10] or str(uuid.uuid4())
                computed_legacy_hash, canonical_hash = _persisted_content_hashes(
                    dl_row[4], dl_row[5]
                )
                ledger_rows = await self._db.fetchall(
                    """
                    SELECT effective_max_retries, legacy_content_hash
                    FROM delivery_idempotency
                    WHERE agent_id = ? AND entry_id = ?
                    """,
                    (self._agent_id, dl_row[1]),
                )
                ledger_policies = {
                    row[0] for row in ledger_rows if row[0] is not None
                }
                ledger_hashes = {
                    row[1] for row in ledger_rows if row[1] is not None
                }
                if len(ledger_policies) > 1:
                    raise DeliveryIdempotencyStateError(
                        "dead-letter retry found inconsistent replay metadata"
                    )
                ledger_policy = next(iter(ledger_policies), None)
                ledger_legacy_hash = next(iter(ledger_hashes), None)
                if dl_row[11] is None and len(ledger_hashes) > 1:
                    raise DeliveryIdempotencyStateError(
                        "dead-letter retry has no authoritative compatibility hash"
                    )
                legacy_hash = dl_row[11] or ledger_legacy_hash or computed_legacy_hash
                # The replay ledger is authoritative when attached. Early
                # prerelease schemas gave the tombstone column DEFAULT 5, so
                # a rolling old writer that omitted it can leave a fabricated
                # value on both backends. Prefer the request policy captured
                # by the ledger over that ambiguous tombstone value.
                retry_policy = (
                    ledger_policy
                    if ledger_policy is not None
                    else dl_row[9]
                    if dl_row[9] is not None
                    else self._max_retries
                )
                await self._db.execute(
                    """
                    UPDATE delivery_dead_letter SET retry_entry_id = ?
                    WHERE id = ? AND agent_id = ? AND retry_entry_id IS NULL
                    """,
                    (new_id, dl_row[0], self._agent_id),
                )
                # Complete a previously interrupted move-to-dead-letter before
                # recreating the live row. Otherwise consuming the tombstone
                # would make both the residual original and retry row eligible.
                if dl_row[1] != new_id:
                    await self._db.execute(
                        """
                        DELETE FROM delivery_queue
                        WHERE id = ? AND agent_id = ?
                        """,
                        (dl_row[1], self._agent_id),
                    )
                await self._db.execute(
                    """
                    INSERT INTO delivery_queue
                        (id, agent_id, channel_type, recipient, content_json,
                         content_hash, canonical_content_hash, status,
                         attempts, max_retries,
                         next_retry_at, last_error, created_at, delivered_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0, ?, ?, NULL, ?, NULL)
                    ON CONFLICT (id) DO NOTHING
                    """,
                    (
                        new_id,
                        self._agent_id,
                        dl_row[3],  # channel_type
                        dl_row[4],  # recipient
                        dl_row[5],  # content_json
                        legacy_hash,
                        canonical_hash,
                        DeliveryStatus.PENDING.value,
                        retry_policy,
                        now_iso,  # next_retry_at
                        now_iso,  # created_at
                    ),
                )
                # Keep every replay key attached to the new canonical queue ID.
                await self._db.execute(
                    """
                    UPDATE delivery_idempotency
                    SET entry_id = ?, created_at = ?, compensating = 0,
                        previous_entry_id = COALESCE(previous_entry_id, ?)
                    WHERE agent_id = ? AND entry_id = ?
                    """,
                    (new_id, now_iso, dl_row[1], self._agent_id, dl_row[1]),
                )
                # This is intentionally the final awaited mutation. If an
                # earlier step fails, the dead letter retains retry_entry_id
                # and a later call can safely resume. If this DELETE completes
                # ambiguously, the queue row and replay mapping already exist.
                deleted = await self._db.execute(
                    "DELETE FROM delivery_dead_letter WHERE id = ? AND agent_id = ?",
                    (dl_row[0], self._agent_id),
                )
                if deleted == 0:
                    raise DeliveryIdempotencyStateError(
                        "dead-letter retry lost its locked source row"
                    )
                return {
                    "success": True,
                    "entry_id": new_id,
                    "original_id": dl_row[1],
                    "status": "re-enqueued_from_dead_letter",
                }

            row = await self._db.fetchone(
                """
                SELECT id, agent_id, channel_type, recipient, content_json,
                       content_hash, status, attempts, max_retries,
                       next_retry_at, last_error, created_at, delivered_at
                FROM delivery_queue
                WHERE id = ? AND agent_id = ?
                """,
                (entry_id, self._agent_id),
            )
            if row is None:
                raise DeliveryIdempotencyStateError(
                    "delivery retry lost its locked queue row"
                )
            entry = self._row_to_entry(row)
            if entry.status == DeliveryStatus.DELIVERED:
                return {"success": False, "error": "Message already delivered"}
            if entry.status == DeliveryStatus.IN_FLIGHT:
                return {"success": False, "error": "Message is currently in flight"}

            now_iso = datetime.now(timezone.utc).isoformat()
            await self._db.execute(
                """
                UPDATE delivery_queue
                SET status = ?, next_retry_at = ?, last_error = NULL
                WHERE id = ? AND agent_id = ?
                """,
                (DeliveryStatus.PENDING.value, now_iso, entry_id, self._agent_id),
            )
            return {
                "success": True,
                "entry_id": entry_id,
                "status": "queued_for_retry",
            }

    async def move_to_dead_letter(self, entry_id: str, reason: str) -> None:
        """Move an entry from the main queue to the dead letter table.

        Args:
            entry_id: The queue entry ID to move.
            reason: Human-readable reason for dead-lettering.
        """
        async with self._db.transaction(immediate=True):
            locked = await self._db.execute(
                """
                UPDATE delivery_queue SET id = id
                WHERE id = ? AND agent_id = ?
                """,
                (entry_id, self._agent_id),
            )
            if locked == 0:
                logger.warning("Cannot dead-letter unknown entry: %s", entry_id)
                return
            row = await self._db.fetchone(
                """
                SELECT id, agent_id, channel_type, recipient, content_json,
                       attempts, created_at, max_retries, content_hash
                FROM delivery_queue
                WHERE id = ? AND agent_id = ?
                """,
                (entry_id, self._agent_id),
            )
            if not row:
                logger.warning("Cannot dead-letter unknown entry: %s", entry_id)
                return

            existing = await self._find_dead_letter_for_queue_id(entry_id)
            if existing is None:
                await self._db.execute(
                    """
                    INSERT INTO delivery_dead_letter
                        (id, original_id, agent_id, channel_type, recipient,
                         content_json, error, attempts, created_at, max_retries,
                         legacy_content_hash)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        str(uuid.uuid4()),
                        row[0],
                        row[1],
                        row[2],
                        row[3],
                        row[4],
                        reason,
                        row[5],
                        datetime.now(timezone.utc).isoformat(),
                        row[7],
                        row[8],
                    ),
                )

            # Final awaited mutation: if it fails, the dead-letter tombstone
            # suppresses delivery and replay until this transition is resumed.
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
              AND NOT EXISTS (
                  SELECT 1 FROM delivery_dead_letter
                  WHERE delivery_dead_letter.agent_id = delivery_queue.agent_id
                    AND delivery_dead_letter.original_id = delivery_queue.id
              )
              AND NOT EXISTS (
                  SELECT 1 FROM delivery_dead_letter
                  WHERE delivery_dead_letter.agent_id = delivery_queue.agent_id
                    AND delivery_dead_letter.retry_entry_id = delivery_queue.id
              )
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
              AND NOT EXISTS (
                  SELECT 1 FROM delivery_dead_letter
                  WHERE delivery_dead_letter.agent_id = delivery_queue.agent_id
                    AND delivery_dead_letter.original_id = delivery_queue.id
              )
              AND NOT EXISTS (
                  SELECT 1 FROM delivery_dead_letter
                  WHERE delivery_dead_letter.agent_id = delivery_queue.agent_id
                    AND delivery_dead_letter.retry_entry_id = delivery_queue.id
              )
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
        """Delete delivered entries and their replay claims after retention.

        The threshold is also the idempotency replay-safety window for completed
        deliveries. Reusing a key after its delivered row is purged creates a
        new logical delivery. Dead-letter claims remain retained with their
        dead-letter record until an explicit retry moves the claim.

        Args:
            older_than_hours: Remove entries delivered more than this many hours ago.

        Returns:
            Number of entries purged.
        """
        cutoff = (
            datetime.now(timezone.utc) - timedelta(hours=older_than_hours)
        ).isoformat()

        count = 0
        async with self._db.transaction(immediate=True):
            # Delete queue rows first and capture their IDs atomically. If a
            # joined SQLite transaction is interrupted afterward and its caller
            # commits, stale replay claims safely recreate a missing queue row;
            # deleting claims first could instead leave the old delivered row
            # alongside a newly accepted same-key delivery.
            while True:
                purged_rows = await self._db.fetchall(
                    """
                    DELETE FROM delivery_queue
                    WHERE id IN (
                        SELECT id FROM delivery_queue
                        WHERE agent_id = ? AND status = ? AND delivered_at < ?
                        ORDER BY delivered_at, id
                        LIMIT 500
                    )
                    RETURNING id
                    """,
                    (self._agent_id, DeliveryStatus.DELIVERED.value, cutoff),
                )
                purged_ids = [row[0] for row in purged_rows]
                if not purged_ids:
                    break
                count += len(purged_ids)
                placeholders = ", ".join("?" for _ in purged_ids)
                await self._db.execute(
                    f"""
                    DELETE FROM delivery_idempotency
                    WHERE agent_id = ? AND entry_id IN ({placeholders})
                    """,
                    (self._agent_id, *purged_ids),
                )
                if len(purged_ids) < 500:
                    break
            # Clean pre-v0.53.12 or independently orphaned claims only after the
            # same retention period, while preserving dead-letter tombstones.
            await self._db.execute(
                """
                DELETE FROM delivery_idempotency
                WHERE agent_id = ? AND created_at < ?
                  AND NOT EXISTS (
                      SELECT 1 FROM delivery_queue
                      WHERE delivery_queue.agent_id = delivery_idempotency.agent_id
                        AND delivery_queue.id = delivery_idempotency.entry_id
                  )
                  AND NOT EXISTS (
                      SELECT 1 FROM delivery_idempotency replacement_claim
                      JOIN delivery_queue
                        ON delivery_queue.agent_id = replacement_claim.agent_id
                       AND delivery_queue.id = replacement_claim.entry_id
                      WHERE replacement_claim.agent_id = delivery_idempotency.agent_id
                        AND replacement_claim.previous_entry_id = delivery_idempotency.entry_id
                  )
                  AND NOT EXISTS (
                      SELECT 1 FROM delivery_dead_letter
                      WHERE delivery_dead_letter.agent_id = delivery_idempotency.agent_id
                        AND delivery_dead_letter.original_id = delivery_idempotency.entry_id
                  )
                  AND NOT EXISTS (
                      SELECT 1 FROM delivery_dead_letter
                      WHERE delivery_dead_letter.agent_id = delivery_idempotency.agent_id
                        AND delivery_dead_letter.retry_entry_id = delivery_idempotency.entry_id
                  )
                """,
                (self._agent_id, cutoff),
            )

        if count > 0:
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
        """Create the delivery tables under one concurrency-safe migration."""
        async with self._db.migration_lock("delivery_queue_schema_v3"):
            # Run both idempotent DDL and explicit catalog probes only after
            # the lock is held, so every decision is a serialized re-probe and
            # the complete schema change stays atomic.
            await self._ensure_tables_locked()

    async def _ensure_tables_locked(self) -> None:
        """Create delivery schema while ``delivery_queue_schema_v3`` is held."""
        await self._db.execute(
            """
            CREATE TABLE IF NOT EXISTS delivery_queue (
                id TEXT PRIMARY KEY,
                agent_id TEXT NOT NULL,
                channel_type TEXT NOT NULL,
                recipient TEXT NOT NULL,
                content_json TEXT NOT NULL DEFAULT '{}',
                content_hash TEXT,
                canonical_content_hash TEXT,
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
        if not await self._db.column_exists(
            "delivery_queue", "canonical_content_hash"
        ):
            await self._db.execute(
                """
                ALTER TABLE delivery_queue
                ADD COLUMN canonical_content_hash TEXT
                """
            )
        await self._db.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_delivery_queue_missing_canonical
            ON delivery_queue(agent_id, id)
            WHERE canonical_content_hash IS NULL
            """
        )
        missing_canonical_hashes = await self._db.fetchall(
            """
            SELECT id, recipient, content_json FROM delivery_queue
            WHERE agent_id = ? AND canonical_content_hash IS NULL
            ORDER BY id
            LIMIT ?
            """,
            (self._agent_id, CANONICAL_BACKFILL_BATCH_SIZE),
        )
        for entry_id, recipient, content_json in missing_canonical_hashes:
            _, canonical_hash = _persisted_content_hashes(
                recipient, content_json or "{}"
            )
            await self._db.execute(
                """
                UPDATE delivery_queue SET canonical_content_hash = ?
                WHERE id = ? AND agent_id = ?
                      AND canonical_content_hash IS NULL
                """,
                (canonical_hash, entry_id, self._agent_id),
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
            CREATE INDEX IF NOT EXISTS idx_delivery_queue_canonical_dedup
            ON delivery_queue(
                agent_id, canonical_content_hash, recipient, created_at
            )
            """
        )
        await self._db.execute(
            """
            CREATE TABLE IF NOT EXISTS delivery_idempotency (
                agent_id TEXT NOT NULL,
                idempotency_key_digest TEXT NOT NULL,
                entry_id TEXT NOT NULL,
                payload_digest TEXT NOT NULL,
                created_at TEXT NOT NULL,
                compensating INTEGER NOT NULL DEFAULT 0,
                previous_entry_id TEXT,
                effective_max_retries INTEGER,
                legacy_content_hash TEXT,
                PRIMARY KEY (agent_id, idempotency_key_digest)
            )
            """
        )
        needs_v3_upgrade = not await self._db.column_exists(
            "delivery_idempotency", "compensating"
        )
        if needs_v3_upgrade:
            await self._db.execute(
                """
                ALTER TABLE delivery_idempotency
                ADD COLUMN compensating INTEGER NOT NULL DEFAULT 0
                """
            )
        needs_v4_upgrade = not await self._db.column_exists(
            "delivery_idempotency", "previous_entry_id"
        )
        if needs_v4_upgrade:
            await self._db.execute(
                """
                ALTER TABLE delivery_idempotency
                ADD COLUMN previous_entry_id TEXT
                """
            )
        if not await self._db.column_exists(
            "delivery_idempotency", "effective_max_retries"
        ):
            await self._db.execute(
                """
                ALTER TABLE delivery_idempotency
                ADD COLUMN effective_max_retries INTEGER
                """
            )
        if not await self._db.column_exists(
            "delivery_idempotency", "legacy_content_hash"
        ):
            await self._db.execute(
                """
                ALTER TABLE delivery_idempotency
                ADD COLUMN legacy_content_hash TEXT
                """
            )
        # v2 accidentally made ledger deletion cascade into the live queue on
        # SQLite. Remove that schema-wide behavior before installing the v3
        # marker trigger used only by a failed joined-transaction enqueue.
        if self._db.backend_type == "sqlite":
            if needs_v4_upgrade:
                await self._db.execute(
                    "DROP TRIGGER IF EXISTS trg_delivery_idempotency_compensate"
                )
            await self._db.execute(
                "DROP TRIGGER IF EXISTS trg_delivery_idempotency_delete"
            )
        if needs_v3_upgrade:
            # v2 declared this mapping UNIQUE. v3 deliberately permits several
            # replay keys to adopt one short-window dedup row. The marker column
            # is the durable one-time migration boundary, so ordinary startups
            # never rebuild this potentially large index.
            await self._db.execute(
                "DROP INDEX IF EXISTS idx_delivery_idempotency_entry"
            )
        await self._db.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_delivery_idempotency_entry
            ON delivery_idempotency(agent_id, entry_id)
            """
        )
        await self._db.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_delivery_idempotency_previous
            ON delivery_idempotency(agent_id, previous_entry_id)
            """
        )
        await self._db.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_delivery_idempotency_retention
            ON delivery_idempotency(agent_id, created_at)
            """
        )
        if self._db.backend_type == "sqlite":
            await self._db.execute(
                "DROP TRIGGER IF EXISTS trg_delivery_idempotency_compensate"
            )
            await self._db.execute(
                """
                CREATE TRIGGER IF NOT EXISTS trg_delivery_idempotency_compensate_v2
                AFTER UPDATE OF compensating ON delivery_idempotency
                WHEN NEW.compensating = 1
                BEGIN
                    DELETE FROM delivery_queue
                    WHERE id = NEW.entry_id AND agent_id = NEW.agent_id;
                    UPDATE delivery_idempotency
                    SET entry_id = previous_entry_id,
                        previous_entry_id = NULL,
                        compensating = 0
                    WHERE agent_id = NEW.agent_id
                      AND entry_id = NEW.entry_id
                      AND previous_entry_id IS NOT NULL;
                    DELETE FROM delivery_idempotency
                    WHERE agent_id = NEW.agent_id
                      AND idempotency_key_digest = NEW.idempotency_key_digest
                      AND entry_id = NEW.entry_id
                      AND compensating = 1
                      AND previous_entry_id IS NULL;
                END
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
                created_at TEXT NOT NULL,
                max_retries INTEGER,
                retry_entry_id TEXT,
                legacy_content_hash TEXT
            )
            """
        )
        if not await self._db.column_exists(
            "delivery_dead_letter", "max_retries"
        ):
            await self._db.execute(
                """
                ALTER TABLE delivery_dead_letter
                ADD COLUMN max_retries INTEGER
                """
            )
        elif self._db.backend_type == "postgres":
            # Early v0.53.12 prerelease schemas declared this NOT NULL. A
            # rolling old writer cannot persist the new value, so the durable
            # ledger recovery path requires the compatibility column to remain
            # nullable on upgraded PostgreSQL databases too.
            if not await self._db.column_accepts_null(
                "delivery_dead_letter", "max_retries"
            ):
                await self._db.execute(
                    """
                    ALTER TABLE delivery_dead_letter
                    ALTER COLUMN max_retries DROP NOT NULL
                    """
                )
            # DROP NOT NULL does not remove the prerelease DEFAULT 5. Probe it
            # independently so a database already visited by an earlier
            # v0.53.12 build still converges and old writers persist unknown
            # policy as NULL rather than silently widening it to five.
            if await self._db.column_has_default(
                "delivery_dead_letter", "max_retries"
            ):
                await self._db.execute(
                    """
                    ALTER TABLE delivery_dead_letter
                    ALTER COLUMN max_retries DROP DEFAULT
                    """
                )
        if not await self._db.column_exists(
            "delivery_dead_letter", "retry_entry_id"
        ):
            await self._db.execute(
                """
                ALTER TABLE delivery_dead_letter
                ADD COLUMN retry_entry_id TEXT
                """
            )
        if not await self._db.column_exists(
            "delivery_dead_letter", "legacy_content_hash"
        ):
            await self._db.execute(
                """
                ALTER TABLE delivery_dead_letter
                ADD COLUMN legacy_content_hash TEXT
                """
            )
        await self._db.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_delivery_dead_letter_agent
            ON delivery_dead_letter(agent_id, created_at DESC)
            """
        )
        await self._db.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_delivery_dead_letter_original
            ON delivery_dead_letter(agent_id, original_id)
            """
        )
        await self._db.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_delivery_dead_letter_retry_entry
            ON delivery_dead_letter(agent_id, retry_entry_id)
            """
        )
