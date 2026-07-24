"""Durable, scoped delivery for normalized signal envelopes.

``signal_log`` is an outcome/audit trail.  This module is deliberately a
separate ledger: it commits a normalized signal before a durable consumer can
claim it, and retains the lease/acknowledgement state required to resume after
a process loss.  It is used through :class:`SignalDispatcher`; callers should
not bypass the dispatcher to turn an external event into a workflow wake.

The ledger is at-least-once by design.  A consumer's side effects must be
idempotent on ``event_id``/``delivery_id``: a process can die after the side
effect and before ``ack_delivery``.  The lease token prevents a stale executor
from acknowledging a delivery that was reclaimed by another executor.
"""

from __future__ import annotations

import json
import re
import secrets
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Iterable, Optional

from kestrel_sdk.signals import Signal

from kestrel_sovereign.a2a.stores.unified.base import UnifiedStoreBase
from kestrel_sovereign.signals.store import _json_default, _serialize_chain
from kestrel_sovereign.storage.db.interface import DatabaseBackend


PENDING = "pending"
LEASED = "leased"
RETRY = "retry"
ACKNOWLEDGED = "acknowledged"
FAILED = "failed"
_TERMINAL_STATUSES = frozenset({ACKNOWLEDGED, FAILED})
_CLAIMABLE_STATUSES = frozenset({PENDING, RETRY})
_SELECTOR_KEY = re.compile(r"^(?:payload\.[A-Za-z0-9_-]+(?:\.[A-Za-z0-9_-]+)*|session_id|kind)=(.+)$")
_PERSISTED_PAYLOAD = object()


@dataclass(frozen=True)
class DurableConsumerRegistration:
    """A durable subscriber owned by one agent/tenant.

    ``correlation_selector`` is intentionally a tiny, non-SQL selector.  It
    is either ``None`` (receive every event from ``source``) or an exact
    comparison such as ``"payload.workflow_id=wf-42"``.  The left side may
    name a sanitized payload path, ``session_id``, or ``kind``.  Keeping the
    selector declarative makes subscriptions replayable after restart and
    prevents callers from injecting an ad-hoc database predicate.
    """

    consumer_id: str
    source: str
    agent_id: str
    correlation_selector: Optional[str] = None
    max_attempts: int = 5
    lease_seconds: int = 60
    active: bool = True


@dataclass(frozen=True)
class DurableSignalEvent:
    """The canonical, post-sanitization signal persisted for consumers."""

    event_id: str
    source_event_id: Optional[str]
    agent_id: str
    target_agent: str
    source: str
    kind: str
    mode: str
    payload: Any
    session_id: Optional[str]
    visibility: str
    urgency: str
    dedupe_key: Optional[str]
    causation_chain: list[dict[str, Any]]
    arrived_at: datetime
    committed_at: datetime
    retention_until: datetime


@dataclass(frozen=True)
class DurableEventPersistence:
    """Result of persisting a source event.

    ``created`` is false when the source event ID had already been accepted
    for the same source and agent/tenant.  The original ``event_id`` is
    returned so callers can record a useful audit result without creating a
    duplicate delivery.  ``delivery_ids`` identifies only the initial
    deliveries inserted by this persistence transaction.  ``initial_leases``
    is populated only for a payload-elided live dispatch: those rows were
    atomically reserved to the emitting dispatcher before commit, so another
    worker cannot claim the marker-only delivery during the sidecar handoff.
    """

    event_id: str
    created: bool
    delivery_ids: tuple[str, ...] = ()
    retention_until: Optional[datetime] = None
    initial_leases: tuple["DurableInitialDeliveryLease", ...] = ()


@dataclass(frozen=True)
class DurableInitialDeliveryLease:
    """An initial live-delivery reservation created with its event row.

    The token is a lease capability, never user payload.  It lets the
    emitting dispatcher transfer this reservation to its chosen executor
    without exposing a newly committed marker-only delivery to another store
    instance first.
    """

    delivery_id: str
    consumer_id: str
    lease_token: str
    lease_expires_at: datetime
    created_at: datetime


@dataclass(frozen=True)
class DurableDelivery:
    """A delivery, optionally leased to an executor, plus its event."""

    delivery_id: str
    consumer_id: str
    agent_id: str
    event_id: str
    status: str
    attempts: int
    max_attempts: int
    lease_owner: Optional[str]
    lease_token: Optional[str]
    lease_expires_at: Optional[datetime]
    next_attempt_at: Optional[datetime]
    last_error: Optional[str]
    acknowledged_at: Optional[datetime]
    terminal_at: Optional[datetime]
    created_at: datetime
    updated_at: datetime
    event: DurableSignalEvent


def _json_dump(value: Any) -> str:
    return json.dumps(value, default=_json_default, ensure_ascii=False, sort_keys=True)


def _json_load(value: Any) -> Any:
    if isinstance(value, (dict, list)) or value is None:
        return value
    return json.loads(value)


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


class DurableSignalStore(UnifiedStoreBase):
    """Backend-neutral pending-delivery ledger.

    Every agent-facing query includes ``agent_id`` in SQL.  This is an
    authorization boundary for a shared PostgreSQL database, not a caller-side
    post-filter.  SQLite uses the same predicates so standalone behaviour is
    identical.
    """

    EVENTS = "durable_signal_events"
    CONSUMERS = "durable_signal_consumers"
    DELIVERIES = "durable_signal_deliveries"

    def __init__(self, backend: DatabaseBackend):
        # ``SignalLogStore`` historically accepts the ``AsyncDatabase``
        # compatibility facade as well as a native ``DatabaseBackend``.  The
        # dispatcher derives this ledger from that store, so unwrap only the
        # legacy facade (which exposes ``fetchall`` but not ``fetch_all``)
        # rather than requiring every existing signal-store embedding to
        # migrate its construction path. Durable delivery uses the native
        # ``fetch_one`` / ``fetch_all`` contract and must share the same
        # transaction domain as the signal log. The capability check leaves
        # native backends and test doubles untouched.
        native_backend = backend
        if not hasattr(backend, "fetch_all") and hasattr(backend, "backend"):
            native_backend = backend.backend
        super().__init__(native_backend)

    async def initialize(self) -> None:
        ts_type = self.timestamp_type()
        ts_default = self.now_default()
        json_type = self.json_type()
        bool_type = self.boolean_type()
        await self._backend.execute_script(
            f"""
            CREATE TABLE IF NOT EXISTS {self.EVENTS} (
                event_id TEXT PRIMARY KEY,
                source_event_id TEXT,
                agent_id TEXT NOT NULL,
                target_agent TEXT NOT NULL,
                source TEXT NOT NULL,
                kind TEXT NOT NULL,
                mode TEXT NOT NULL,
                payload {json_type} NOT NULL,
                session_id TEXT,
                visibility TEXT NOT NULL,
                urgency TEXT NOT NULL,
                dedupe_key TEXT,
                causation_chain {json_type} NOT NULL,
                arrived_at {ts_type} NOT NULL,
                committed_at {ts_type} {ts_default},
                retention_until {ts_type} NOT NULL,
                UNIQUE (agent_id, source, source_event_id)
            );

            CREATE TABLE IF NOT EXISTS {self.CONSUMERS} (
                agent_id TEXT NOT NULL,
                consumer_id TEXT NOT NULL,
                source TEXT NOT NULL,
                correlation_selector TEXT,
                max_attempts INTEGER NOT NULL,
                lease_seconds INTEGER NOT NULL,
                active {bool_type} NOT NULL,
                created_at {ts_type} {ts_default},
                updated_at {ts_type} {ts_default},
                PRIMARY KEY (agent_id, consumer_id)
            );

            CREATE TABLE IF NOT EXISTS {self.DELIVERIES} (
                delivery_id TEXT PRIMARY KEY,
                agent_id TEXT NOT NULL,
                consumer_id TEXT NOT NULL,
                event_id TEXT NOT NULL,
                status TEXT NOT NULL,
                attempts INTEGER NOT NULL DEFAULT 0,
                max_attempts INTEGER NOT NULL,
                lease_owner TEXT,
                lease_token TEXT,
                lease_expires_at {ts_type},
                next_attempt_at {ts_type},
                last_error TEXT,
                acknowledged_at {ts_type},
                terminal_at {ts_type},
                created_at {ts_type} {ts_default},
                updated_at {ts_type} {ts_default},
                UNIQUE (agent_id, consumer_id, event_id),
                FOREIGN KEY (event_id) REFERENCES {self.EVENTS}(event_id)
                    ON DELETE CASCADE,
                FOREIGN KEY (agent_id, consumer_id)
                    REFERENCES {self.CONSUMERS}(agent_id, consumer_id)
                    ON DELETE RESTRICT
            );
            """
        )
        await self._backend.execute(
            f"CREATE INDEX IF NOT EXISTS idx_{self.EVENTS}_scope_retention "
            f"ON {self.EVENTS}(agent_id, source, retention_until)"
        )
        await self._backend.execute(
            f"CREATE INDEX IF NOT EXISTS idx_{self.DELIVERIES}_claim "
            f"ON {self.DELIVERIES}(agent_id, consumer_id, status, next_attempt_at)"
        )
        await self._backend.execute(
            f"CREATE INDEX IF NOT EXISTS idx_{self.DELIVERIES}_lease "
            f"ON {self.DELIVERIES}(agent_id, consumer_id, lease_expires_at)"
        )

    # ------------------------------------------------------------------
    # Subscription registration and event persistence
    # ------------------------------------------------------------------

    async def register_consumer(
        self, registration: DurableConsumerRegistration
    ) -> None:
        """Persist an idempotent consumer registration and backfill it.

        A different registration reusing an existing consumer ID is rejected;
        silently changing a workflow's selector/retry policy would make old
        pending deliveries ambiguous.  Backfill is idempotent because delivery
        identity is unique on ``(agent_id, consumer_id, event_id)``.
        """
        self._validate_registration(registration)
        now = self.now_utc()
        async with self._backend.transaction():
            # Consumer registration and event persistence must share one
            # serialization point.  Without it on PostgreSQL, an event can
            # commit after this transaction fails to see it during backfill
            # while that event's consumer lookup still cannot see this
            # uncommitted registration — permanently losing the delivery.
            # SQLite's transaction writer lock is reserved explicitly below;
            # the advisory transaction lock gives hosted PostgreSQL identical
            # handoff semantics at the narrow (agent, source) scope.
            await self._lock_scope_handoff(
                agent_id=registration.agent_id, source=registration.source
            )
            row = await self._backend.fetch_one(
                f"""
                SELECT source, correlation_selector, max_attempts,
                       lease_seconds, active
                FROM {self.CONSUMERS}
                WHERE agent_id = ? AND consumer_id = ?
                """,
                (registration.agent_id, registration.consumer_id),
            )
            expected = (
                registration.source,
                registration.correlation_selector,
                registration.max_attempts,
                registration.lease_seconds,
                self.to_bool_param(registration.active),
            )
            if row is not None:
                actual = (row[0], row[1], int(row[2]), int(row[3]), row[4])
                if actual != expected:
                    raise ValueError(
                        "Durable consumer registration conflicts with the "
                        f"existing contract for '{registration.consumer_id}'."
                    )
            else:
                await self._backend.execute(
                    f"""
                    INSERT INTO {self.CONSUMERS} (
                        agent_id, consumer_id, source, correlation_selector,
                        max_attempts, lease_seconds, active, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        registration.agent_id,
                        registration.consumer_id,
                        registration.source,
                        registration.correlation_selector,
                        registration.max_attempts,
                        registration.lease_seconds,
                        self.to_bool_param(registration.active),
                        self.to_timestamp_param(now),
                        self.to_timestamp_param(now),
                    ),
                )
            # Inactive registrations are retained as configuration but do not
            # materialize work that no executor is allowed to claim.
            if registration.active:
                await self._backfill_consumer(registration, now=now)

    async def persist_signal(
        self,
        signal: Signal,
        *,
        agent_id: str,
        source_event_id: Optional[str],
        retention_days: int,
        transient_selector_payload: Any = _PERSISTED_PAYLOAD,
        initial_lease_owner: Optional[str] = None,
        before_commit: Optional[Callable[[DurableEventPersistence], None]] = None,
        on_rollback: Optional[Callable[[DurableEventPersistence], None]] = None,
    ) -> DurableEventPersistence:
        """Commit a persisted signal and all matching initial deliveries.

        This is the durable boundary called by the dispatcher *after*
        sanitization/schema normalization and causation validation, but before
        handlers, cognition, or any durable consumer can execute.  When a
        privacy projection fully elides payload content, ``signal`` is that
        safe persisted projection while ``transient_selector_payload`` is the
        normalized in-memory payload used only to materialize deliveries for
        consumers already registered in this transaction.  It is never
        serialized, returned, or used for restart backfill.  Projections that
        retain a replayable payload, including ANONYMOUS redaction, leave this
        argument unset so initial and replayed selector behavior is identical.

        ``initial_lease_owner`` reserves each initially matched delivery to
        one live dispatcher in the same transaction as the event insert.  The
        dispatcher installs its process-local raw-payload sidecars through
        ``before_commit`` before this transaction becomes visible.  A rollback
        invokes ``on_rollback`` so those sidecars cannot outlive rows that did
        not commit.  Both callbacks are synchronous deliberately: yielding
        between installing the sidecar and committing would reopen the very
        visibility race this handoff closes.
        """
        if retention_days < 0:
            raise ValueError("retention_days must be >= 0")
        self._require_nonempty("agent_id", agent_id)
        self._require_nonempty("source", signal.source)
        if initial_lease_owner is not None:
            self._require_nonempty("initial_lease_owner", initial_lease_owner)
        source_event_id = self._normalize_source_event_id(source_event_id)
        payload_json = _json_dump(signal.payload)
        chain_json = _json_dump(_serialize_chain(signal.causation_chain))
        persistence: Optional[DurableEventPersistence] = None
        try:
            async with self._backend.transaction():
                await self._lock_scope_handoff(agent_id=agent_id, source=signal.source)
                # The transaction may have waited behind the cross-instance
                # handoff lock. Start persisted event timing only after that
                # contention has cleared, never from method entry.
                now = self.now_utc()
                retention_until = now + timedelta(days=retention_days)
                inserted = await self._backend.execute(
                    f"""
                    INSERT OR IGNORE INTO {self.EVENTS} (
                        event_id, source_event_id, agent_id, target_agent, source, kind, mode,
                        payload, session_id, visibility, urgency, dedupe_key,
                        causation_chain, arrived_at, committed_at, retention_until
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        signal.id,
                        source_event_id,
                        agent_id,
                        signal.target_agent,
                        signal.source,
                        signal.kind,
                        signal.mode.value,
                        payload_json,
                        signal.session_id,
                        signal.visibility.value,
                        signal.urgency.value,
                        signal.dedupe_key,
                        chain_json,
                        self.to_timestamp_param(signal.arrived_at),
                        self.to_timestamp_param(now),
                        self.to_timestamp_param(retention_until),
                    ),
                )
                if inserted == 0:
                    existing = await self._find_existing_event_locked(
                        agent_id, signal, source_event_id
                    )
                    return DurableEventPersistence(event_id=existing, created=False)

                consumer_rows = await self._backend.fetch_all(
                    f"""
                    SELECT consumer_id, correlation_selector, max_attempts, lease_seconds
                    FROM {self.CONSUMERS}
                    WHERE agent_id = ? AND source = ? AND active = ?
                    """,
                    (agent_id, signal.source, self.to_bool_param(True)),
                )
                event = self._event_from_signal(
                    signal,
                    agent_id=agent_id,
                    source_event_id=source_event_id,
                    committed_at=now,
                    retention_until=retention_until,
                )
                selector_event = (
                    event
                    if transient_selector_payload is _PERSISTED_PAYLOAD
                    else replace(event, payload=transient_selector_payload)
                )
                delivery_ids: list[str] = []
                initial_lease_specs: list[tuple[str, str, str, int]] = []
                for consumer_id, selector, max_attempts, lease_seconds in consumer_rows:
                    if not self._matches_selector(selector_event, selector):
                        continue
                    lease_token = None
                    lease_expires_at = None
                    if initial_lease_owner is not None:
                        lease_token = secrets.token_urlsafe(24)
                        lease_expires_at = now + timedelta(seconds=int(lease_seconds))
                    delivery_id = await self._insert_delivery_locked(
                        agent_id=agent_id,
                        consumer_id=consumer_id,
                        event_id=signal.id,
                        max_attempts=int(max_attempts),
                        now=now,
                        initial_lease_owner=initial_lease_owner,
                        initial_lease_token=lease_token,
                        initial_lease_expires_at=lease_expires_at,
                    )
                    if delivery_id is not None:
                        delivery_ids.append(delivery_id)
                        if lease_token is not None and lease_expires_at is not None:
                            initial_lease_specs.append(
                                (
                                    delivery_id,
                                    consumer_id,
                                    lease_token,
                                    int(lease_seconds),
                                )
                            )

                # The provisional lease values above are private to this
                # transaction. Refresh them immediately before the synchronous
                # pre-commit sidecar callback, after consumer matching and
                # delivery materialization. Thus lock contention cannot make a
                # newly committed initial reservation already expired.
                initial_leases: list[DurableInitialDeliveryLease] = []
                if initial_lease_specs:
                    reservation_now = self.now_utc()
                    for (
                        delivery_id,
                        consumer_id,
                        lease_token,
                        lease_seconds,
                    ) in initial_lease_specs:
                        lease_expires_at = reservation_now + timedelta(
                            seconds=lease_seconds
                        )
                        refreshed = await self._backend.execute(
                            f"""
                            UPDATE {self.DELIVERIES}
                            SET lease_expires_at = ?, updated_at = ?
                            WHERE delivery_id = ? AND agent_id = ?
                              AND consumer_id = ? AND status = ?
                              AND lease_owner = ? AND lease_token = ?
                            """,
                            (
                                self.to_timestamp_param(lease_expires_at),
                                self.to_timestamp_param(reservation_now),
                                delivery_id,
                                agent_id,
                                consumer_id,
                                LEASED,
                                initial_lease_owner,
                                lease_token,
                            ),
                        )
                        if refreshed != 1:
                            raise RuntimeError(
                                "initial durable delivery reservation disappeared "
                                "before commit"
                            )
                        initial_leases.append(
                            DurableInitialDeliveryLease(
                                delivery_id=delivery_id,
                                consumer_id=consumer_id,
                                lease_token=lease_token,
                                lease_expires_at=lease_expires_at,
                                created_at=reservation_now,
                            )
                        )
                persistence = DurableEventPersistence(
                    event_id=signal.id,
                    created=True,
                    delivery_ids=tuple(delivery_ids),
                    retention_until=retention_until,
                    initial_leases=tuple(initial_leases),
                )
                if before_commit is not None:
                    before_commit(persistence)
        except BaseException:
            if persistence is not None and on_rollback is not None:
                on_rollback(persistence)
            raise
        assert persistence is not None
        return persistence

    # ------------------------------------------------------------------
    # Claim / lease / acknowledgement API
    # ------------------------------------------------------------------

    async def claim_delivery(
        self,
        *,
        agent_id: str,
        consumer_id: str,
        executor_id: str,
        now: Optional[datetime] = None,
    ) -> Optional[DurableDelivery]:
        """Atomically lease one due delivery for this scoped consumer.

        The conditional UPDATE is the ownership handoff.  Two SQLite
        connections or two PostgreSQL executors may choose the same candidate
        in their subqueries, but only one can change its still-claimable state;
        the loser observes a zero row count and receives no delivery.
        """
        self._require_nonempty("agent_id", agent_id)
        self._require_nonempty("consumer_id", consumer_id)
        self._require_nonempty("executor_id", executor_id)
        now = _as_utc(now or self.now_utc())
        consumer = await self._get_consumer(agent_id, consumer_id)
        if consumer is None or not consumer[4]:
            return None
        await self._recover_expired_leases(
            agent_id=agent_id, consumer_id=consumer_id, now=now
        )
        # Catch events committed while a consumer was restarting.  The unique
        # delivery key makes this safe to do on every poll.
        registration = DurableConsumerRegistration(
            agent_id=agent_id,
            consumer_id=consumer_id,
            source=consumer[0],
            correlation_selector=consumer[1],
            max_attempts=int(consumer[2]),
            lease_seconds=int(consumer[3]),
            active=bool(consumer[4]),
        )
        # Backfill is idempotent (unique delivery identity) but deliberately
        # separate from the claim statement.  Keeping the actual conditional
        # handoff as one autocommit UPDATE means two independent SQLite
        # processes never hold competing read transactions while deciding who
        # owns a delivery; PostgreSQL receives the same atomic predicate.
        await self._backfill_consumer(registration, now=now)
        lease_token = secrets.token_urlsafe(24)
        lease_expires_at = now + timedelta(seconds=registration.lease_seconds)
        updated = await self._backend.execute(
            f"""
            UPDATE {self.DELIVERIES}
            SET status = ?, attempts = attempts + 1, lease_owner = ?,
                lease_token = ?, lease_expires_at = ?, next_attempt_at = NULL,
                updated_at = ?
            WHERE delivery_id = (
                SELECT delivery_id FROM {self.DELIVERIES}
                WHERE agent_id = ? AND consumer_id = ?
                  AND status IN ('{PENDING}', '{RETRY}')
                  AND attempts < max_attempts
                  AND (next_attempt_at IS NULL OR next_attempt_at <= ?)
                ORDER BY created_at, delivery_id
                LIMIT 1
            )
              AND agent_id = ? AND consumer_id = ?
              AND status IN ('{PENDING}', '{RETRY}')
              AND attempts < max_attempts
              AND (next_attempt_at IS NULL OR next_attempt_at <= ?)
            """,
            (
                LEASED,
                executor_id,
                lease_token,
                self.to_timestamp_param(lease_expires_at),
                self.to_timestamp_param(now),
                agent_id,
                consumer_id,
                self.to_timestamp_param(now),
                agent_id,
                consumer_id,
                self.to_timestamp_param(now),
            ),
        )
        if updated == 0:
            return None
        return await self._delivery_for_lease_locked(
            agent_id=agent_id,
            consumer_id=consumer_id,
            lease_token=lease_token,
        )

    async def claim_initial_delivery(
        self,
        *,
        agent_id: str,
        consumer_id: str,
        delivery_id: str,
        initial_lease_owner: str,
        initial_lease_token: str,
        executor_id: str,
        now: Optional[datetime] = None,
    ) -> Optional[DurableDelivery]:
        """Transfer one emitting-dispatcher's initial reservation to a worker.

        Initial payload-elided deliveries are inserted as ``LEASED`` before
        their event transaction commits.  A normal claimant cannot see them as
        due.  Only the dispatcher holding this unpersisted capability may make
        the first worker claim; after that, ordinary retry/lease rules apply.
        """
        self._require_nonempty("agent_id", agent_id)
        self._require_nonempty("consumer_id", consumer_id)
        self._require_nonempty("delivery_id", delivery_id)
        self._require_nonempty("initial_lease_owner", initial_lease_owner)
        self._require_nonempty("initial_lease_token", initial_lease_token)
        self._require_nonempty("executor_id", executor_id)
        now = _as_utc(now or self.now_utc())
        consumer = await self._get_consumer(agent_id, consumer_id)
        if consumer is None or not consumer[4]:
            return None
        lease_token = secrets.token_urlsafe(24)
        lease_expires_at = now + timedelta(seconds=int(consumer[3]))
        updated = await self._backend.execute(
            f"""
            UPDATE {self.DELIVERIES}
            SET attempts = attempts + 1, lease_owner = ?, lease_token = ?,
                lease_expires_at = ?, next_attempt_at = NULL, updated_at = ?
            WHERE agent_id = ? AND consumer_id = ? AND delivery_id = ?
              AND status = ? AND lease_owner = ? AND lease_token = ?
              AND lease_expires_at > ?
            """,
            (
                executor_id,
                lease_token,
                self.to_timestamp_param(lease_expires_at),
                self.to_timestamp_param(now),
                agent_id,
                consumer_id,
                delivery_id,
                LEASED,
                initial_lease_owner,
                initial_lease_token,
                self.to_timestamp_param(now),
            ),
        )
        if updated == 0:
            return None
        return await self._delivery_for_lease_locked(
            agent_id=agent_id,
            consumer_id=consumer_id,
            lease_token=lease_token,
        )

    async def ack_delivery(
        self,
        *,
        agent_id: str,
        consumer_id: str,
        delivery_id: str,
        lease_token: str,
        now: Optional[datetime] = None,
    ) -> bool:
        """Acknowledge a live lease.  Stale/foreign tokens cannot ack it."""
        now = _as_utc(now or self.now_utc())
        updated = await self._backend.execute(
            f"""
            UPDATE {self.DELIVERIES}
            SET status = ?, lease_owner = NULL, lease_token = NULL,
                lease_expires_at = NULL, acknowledged_at = ?, terminal_at = ?,
                updated_at = ?
            WHERE agent_id = ? AND consumer_id = ? AND delivery_id = ?
              AND status = ? AND lease_token = ? AND lease_expires_at > ?
            """,
            (
                ACKNOWLEDGED,
                self.to_timestamp_param(now),
                self.to_timestamp_param(now),
                self.to_timestamp_param(now),
                agent_id,
                consumer_id,
                delivery_id,
                LEASED,
                lease_token,
                self.to_timestamp_param(now),
            ),
        )
        return updated == 1

    async def nack_delivery(
        self,
        *,
        agent_id: str,
        consumer_id: str,
        delivery_id: str,
        lease_token: str,
        error: str,
        retry_delay: timedelta = timedelta(),
        terminal: bool = False,
        now: Optional[datetime] = None,
    ) -> Optional[DurableDelivery]:
        """Release a failed lease for retry or mark a terminal failure.

        Retry is bounded by the delivery's persisted ``max_attempts``.  A
        caller may request terminal failure earlier, but cannot request an
        unbounded retry by passing a fresh policy on every invocation.
        """
        self._require_nonempty("error", error)
        if retry_delay.total_seconds() < 0:
            raise ValueError("retry_delay must not be negative")
        now = _as_utc(now or self.now_utc())
        retry_at = now + retry_delay
        timestamp = self._timestamp_placeholder()
        updated = await self._backend.execute(
            f"""
            UPDATE {self.DELIVERIES}
            SET status = CASE WHEN ? OR attempts >= max_attempts THEN ? ELSE ? END,
                lease_owner = NULL, lease_token = NULL, lease_expires_at = NULL,
                next_attempt_at = CASE
                    WHEN ? OR attempts >= max_attempts THEN NULL ELSE {timestamp} END,
                last_error = ?,
                terminal_at = CASE WHEN ? OR attempts >= max_attempts
                    THEN {timestamp} ELSE NULL END,
                updated_at = ?
            WHERE agent_id = ? AND consumer_id = ? AND delivery_id = ?
              AND status = ? AND lease_token = ? AND lease_expires_at > ?
            """,
            (
                self.to_bool_param(terminal),
                FAILED,
                RETRY,
                self.to_bool_param(terminal),
                self.to_timestamp_param(retry_at),
                error,
                self.to_bool_param(terminal),
                self.to_timestamp_param(now),
                self.to_timestamp_param(now),
                agent_id,
                consumer_id,
                delivery_id,
                LEASED,
                lease_token,
                self.to_timestamp_param(now),
            ),
        )
        if updated == 0:
            return None
        return await self.get_delivery(
            agent_id=agent_id, consumer_id=consumer_id, delivery_id=delivery_id
        )

    async def get_delivery(
        self, *, agent_id: str, consumer_id: str, delivery_id: str
    ) -> Optional[DurableDelivery]:
        row = await self._backend.fetch_one(
            self._delivery_select_sql(
                "d.agent_id = ? AND d.consumer_id = ? AND d.delivery_id = ?"
            ),
            (agent_id, consumer_id, delivery_id),
        )
        return self._row_to_delivery(row) if row is not None else None

    async def list_deliveries(
        self,
        *,
        agent_id: str,
        consumer_id: Optional[str] = None,
        statuses: Optional[Iterable[str]] = None,
        limit: int = 100,
    ) -> list[DurableDelivery]:
        """List observable delivery state within one agent/tenant only."""
        self._require_nonempty("agent_id", agent_id)
        if limit < 1 or limit > 1000:
            raise ValueError("limit must be between 1 and 1000")
        where = ["d.agent_id = ?"]
        params: list[Any] = [agent_id]
        if consumer_id is not None:
            where.append("d.consumer_id = ?")
            params.append(consumer_id)
        if statuses is not None:
            wanted = tuple(statuses)
            valid = _CLAIMABLE_STATUSES | {LEASED} | _TERMINAL_STATUSES
            if not wanted or any(status not in valid for status in wanted):
                raise ValueError("statuses must be durable delivery states")
            where.append("d.status IN (" + ", ".join("?" for _ in wanted) + ")")
            params.extend(wanted)
        rows = await self._backend.fetch_all(
            self._delivery_select_sql(" AND ".join(where))
            + " ORDER BY d.created_at, d.delivery_id LIMIT ?",
            tuple(params + [limit]),
        )
        return [self._row_to_delivery(row) for row in rows]

    async def purge_expired(
        self, *, agent_id: str, now: Optional[datetime] = None
    ) -> int:
        """Delete only this agent's retained, terminal event histories.

        Pending, retriable, and leased work is never cleaned up by retention;
        operators must first resolve it to an observable terminal state.
        ``agent_id`` is mandatory because the periodic sweep is owned by one
        dispatcher even when multiple agents share a PostgreSQL database.
        """
        self._require_nonempty("agent_id", agent_id)
        now = _as_utc(now or self.now_utc())
        return await self._backend.execute(
            f"""
            DELETE FROM {self.EVENTS}
            WHERE agent_id = ?
              AND retention_until < ?
              AND NOT EXISTS (
                  SELECT 1 FROM {self.DELIVERIES} d
                  WHERE d.event_id = {self.EVENTS}.event_id
                    AND d.status NOT IN ('{ACKNOWLEDGED}', '{FAILED}')
              )
            """,
            (agent_id, self.to_timestamp_param(now)),
        )

    # ------------------------------------------------------------------
    # Internal storage helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _require_nonempty(name: str, value: str) -> None:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} must be a non-empty string")

    def _validate_registration(self, registration: DurableConsumerRegistration) -> None:
        self._require_nonempty("consumer_id", registration.consumer_id)
        self._require_nonempty("source", registration.source)
        self._require_nonempty("agent_id", registration.agent_id)
        if (
            not isinstance(registration.max_attempts, int)
            or isinstance(registration.max_attempts, bool)
            or registration.max_attempts < 1
        ):
            raise ValueError("max_attempts must be >= 1")
        if (
            not isinstance(registration.lease_seconds, int)
            or isinstance(registration.lease_seconds, bool)
            or registration.lease_seconds < 1
        ):
            raise ValueError("lease_seconds must be >= 1")
        if registration.correlation_selector is not None:
            if (
                not isinstance(registration.correlation_selector, str)
                or not _SELECTOR_KEY.match(registration.correlation_selector)
            ):
                raise ValueError(
                    "correlation_selector must be 'payload.<path>=<value>', "
                    "'session_id=<value>', or 'kind=<value>'"
                )

    @staticmethod
    def _normalize_source_event_id(value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        if not isinstance(value, str) or not value.strip():
            raise ValueError("source_event_id must be a non-empty string when set")
        return value.strip()

    async def _find_existing_event_locked(
        self, agent_id: str, signal: Signal, source_event_id: Optional[str]
    ) -> str:
        if source_event_id is not None:
            row = await self._backend.fetch_one(
                f"""
                SELECT event_id FROM {self.EVENTS}
                WHERE agent_id = ? AND source = ? AND source_event_id = ?
                """,
                (agent_id, signal.source, source_event_id),
            )
            if row is not None:
                return row[0]
        row = await self._backend.fetch_one(
            f"SELECT event_id FROM {self.EVENTS} WHERE event_id = ?",
            (signal.id,),
        )
        if row is None:
            raise RuntimeError("durable event insert conflicted without an existing event")
        return row[0]

    async def _lock_scope_handoff(self, *, agent_id: str, source: str) -> None:
        """Serialize registration and persistence for one subscription scope.

        PostgreSQL transactions otherwise use independent snapshots: a new
        registration could backfill before a concurrent event commits, while
        that event's consumer query runs before the registration commits.
        ``pg_advisory_xact_lock`` keeps those two atomic handoff paths in one
        order and releases automatically on commit or rollback.  SQLite
        transactions begin deferred, so a no-op write reserves its one writer
        slot before either handoff path can read stale state.  This matters
        across backend instances: each instance's in-memory write lock is
        local to that instance.
        """
        if self.is_postgres:
            await self._backend.fetch_val(
                "SELECT pg_advisory_xact_lock(hashtextextended(?, 0))",
                (f"durable-signal:{agent_id}:{source}",),
            )
            return
        if self._backend.backend_type == "sqlite":
            await self._backend.execute(f"DELETE FROM {self.CONSUMERS} WHERE 0")
            return
        raise RuntimeError(
            "Durable signal handoff serialization does not support backend "
            f"{self._backend.backend_type!r}"
        )

    async def _get_consumer(
        self, agent_id: str, consumer_id: str
    ) -> Optional[tuple[Any, ...]]:
        return await self._backend.fetch_one(
            f"""
            SELECT source, correlation_selector, max_attempts, lease_seconds, active
            FROM {self.CONSUMERS}
            WHERE agent_id = ? AND consumer_id = ?
            """,
            (agent_id, consumer_id),
        )

    async def _recover_expired_leases(
        self, *, agent_id: str, consumer_id: str, now: datetime
    ) -> None:
        timestamp = self._timestamp_placeholder()
        await self._backend.execute(
            f"""
            UPDATE {self.DELIVERIES}
            SET status = CASE WHEN attempts >= max_attempts THEN ? ELSE ? END,
                lease_owner = NULL, lease_token = NULL, lease_expires_at = NULL,
                next_attempt_at = CASE WHEN attempts >= max_attempts
                    THEN NULL ELSE {timestamp} END,
                last_error = 'lease expired before acknowledgement',
                terminal_at = CASE WHEN attempts >= max_attempts
                    THEN {timestamp} ELSE NULL END,
                updated_at = ?
            WHERE agent_id = ? AND consumer_id = ? AND status = ?
              AND lease_expires_at <= ?
            """,
            (
                FAILED,
                RETRY,
                self.to_timestamp_param(now),
                self.to_timestamp_param(now),
                self.to_timestamp_param(now),
                agent_id,
                consumer_id,
                LEASED,
                self.to_timestamp_param(now),
            ),
        )

    def _timestamp_placeholder(self) -> str:
        """Return a timestamp parameter expression for the active dialect.

        PostgreSQL cannot infer the type of a bind value in a ``CASE`` arm
        whose other arm is ``NULL``.  Without the cast asyncpg treats it as
        text and rejects the update against a ``TIMESTAMPTZ`` column.  SQLite
        stores timestamps as text, so its ordinary placeholder is correct.
        """
        return "?::TIMESTAMPTZ" if self.is_postgres else "?"

    async def _backfill_consumer(
        self, registration: DurableConsumerRegistration, *, now: datetime
    ) -> None:
        rows = await self._backend.fetch_all(
            f"""
            SELECT event_id, source_event_id, agent_id, target_agent, source, kind, mode, payload,
                   session_id, visibility, urgency, dedupe_key,
                   causation_chain, arrived_at, committed_at, retention_until
            FROM {self.EVENTS}
            WHERE agent_id = ? AND source = ? AND retention_until >= ?
            """,
            (
                registration.agent_id,
                registration.source,
                self.to_timestamp_param(now),
            ),
        )
        for row in rows:
            event = self._row_to_event(row)
            if self._matches_selector(event, registration.correlation_selector):
                await self._insert_delivery_locked(
                    agent_id=registration.agent_id,
                    consumer_id=registration.consumer_id,
                    event_id=event.event_id,
                    max_attempts=registration.max_attempts,
                    now=now,
                )

    async def _insert_delivery_locked(
        self,
        *,
        agent_id: str,
        consumer_id: str,
        event_id: str,
        max_attempts: int,
        now: datetime,
        initial_lease_owner: Optional[str] = None,
        initial_lease_token: Optional[str] = None,
        initial_lease_expires_at: Optional[datetime] = None,
    ) -> Optional[str]:
        initial_lease = initial_lease_owner is not None
        if initial_lease != (initial_lease_token is not None):
            raise ValueError("initial lease owner and token must be set together")
        if initial_lease != (initial_lease_expires_at is not None):
            raise ValueError("initial lease owner and expiry must be set together")
        delivery_id = secrets.token_urlsafe(18)
        inserted = await self._backend.execute(
            f"""
            INSERT OR IGNORE INTO {self.DELIVERIES} (
                delivery_id, agent_id, consumer_id, event_id, status,
                attempts, max_attempts, lease_owner, lease_token,
                lease_expires_at, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, 0, ?, ?, ?, ?, ?, ?)
            """,
            (
                delivery_id,
                agent_id,
                consumer_id,
                event_id,
                LEASED if initial_lease else PENDING,
                max_attempts,
                initial_lease_owner,
                initial_lease_token,
                (
                    self.to_timestamp_param(initial_lease_expires_at)
                    if initial_lease_expires_at is not None
                    else None
                ),
                self.to_timestamp_param(now),
                self.to_timestamp_param(now),
            ),
        )
        return delivery_id if inserted == 1 else None

    def _event_from_signal(
        self,
        signal: Signal,
        *,
        agent_id: str,
        source_event_id: Optional[str],
        committed_at: datetime,
        retention_until: datetime,
    ) -> DurableSignalEvent:
        return DurableSignalEvent(
            event_id=signal.id,
            source_event_id=source_event_id,
            agent_id=agent_id,
            target_agent=signal.target_agent,
            source=signal.source,
            kind=signal.kind,
            mode=signal.mode.value,
            payload=signal.payload,
            session_id=signal.session_id,
            visibility=signal.visibility.value,
            urgency=signal.urgency.value,
            dedupe_key=signal.dedupe_key,
            causation_chain=_serialize_chain(signal.causation_chain),
            arrived_at=_as_utc(signal.arrived_at),
            committed_at=committed_at,
            retention_until=retention_until,
        )

    def _row_to_event(self, row: tuple[Any, ...]) -> DurableSignalEvent:
        return DurableSignalEvent(
            event_id=row[0],
            source_event_id=row[1],
            agent_id=row[2],
            target_agent=row[3],
            source=row[4],
            kind=row[5],
            mode=row[6],
            payload=_json_load(row[7]),
            session_id=row[8],
            visibility=row[9],
            urgency=row[10],
            dedupe_key=row[11],
            causation_chain=_json_load(row[12]),
            arrived_at=_as_utc(self.from_timestamp_field(row[13])),
            committed_at=_as_utc(self.from_timestamp_field(row[14])),
            retention_until=_as_utc(self.from_timestamp_field(row[15])),
        )

    @staticmethod
    def _matches_selector(
        event: DurableSignalEvent, selector: Optional[str]
    ) -> bool:
        if selector is None:
            return True
        match = _SELECTOR_KEY.match(selector)
        if match is None:  # registrations are validated; keep this fail-closed.
            return False
        left, expected = selector.split("=", 1)
        if left == "session_id":
            actual = event.session_id
        elif left == "kind":
            actual = event.kind
        else:
            actual: Any = event.payload
            for key in left.removeprefix("payload.").split("."):
                if not isinstance(actual, dict) or key not in actual:
                    return False
                actual = actual[key]
        if isinstance(actual, (dict, list)) or actual is None:
            return False
        return str(actual) == expected

    def _delivery_select_sql(self, where: str) -> str:
        return f"""
            SELECT
                d.delivery_id, d.consumer_id, d.agent_id, d.event_id, d.status,
                d.attempts, d.max_attempts, d.lease_owner, d.lease_token,
                d.lease_expires_at, d.next_attempt_at, d.last_error,
                d.acknowledged_at, d.terminal_at, d.created_at, d.updated_at,
                e.event_id, e.source_event_id, e.agent_id, e.target_agent,
                e.source, e.kind, e.mode, e.payload, e.session_id,
                e.visibility, e.urgency, e.dedupe_key, e.causation_chain,
                e.arrived_at, e.committed_at, e.retention_until
            FROM {self.DELIVERIES} d
            JOIN {self.EVENTS} e ON e.event_id = d.event_id
            WHERE {where}
        """

    async def _delivery_for_lease_locked(
        self, *, agent_id: str, consumer_id: str, lease_token: str
    ) -> DurableDelivery:
        row = await self._backend.fetch_one(
            self._delivery_select_sql(
                "d.agent_id = ? AND d.consumer_id = ? AND d.status = ? "
                "AND d.lease_token = ?"
            ),
            (agent_id, consumer_id, LEASED, lease_token),
        )
        if row is None:
            raise RuntimeError("claimed durable delivery disappeared before handoff")
        return self._row_to_delivery(row)

    def _row_to_delivery(self, row: tuple[Any, ...]) -> DurableDelivery:
        event = self._row_to_event(row[16:])
        return DurableDelivery(
            delivery_id=row[0],
            consumer_id=row[1],
            agent_id=row[2],
            event_id=row[3],
            status=row[4],
            attempts=int(row[5]),
            max_attempts=int(row[6]),
            lease_owner=row[7],
            lease_token=row[8],
            lease_expires_at=(
                _as_utc(self.from_timestamp_field(row[9])) if row[9] is not None else None
            ),
            next_attempt_at=(
                _as_utc(self.from_timestamp_field(row[10])) if row[10] is not None else None
            ),
            last_error=row[11],
            acknowledged_at=(
                _as_utc(self.from_timestamp_field(row[12])) if row[12] is not None else None
            ),
            terminal_at=(
                _as_utc(self.from_timestamp_field(row[13])) if row[13] is not None else None
            ),
            created_at=_as_utc(self.from_timestamp_field(row[14])),
            updated_at=_as_utc(self.from_timestamp_field(row[15])),
            event=event,
        )


__all__ = [
    "ACKNOWLEDGED",
    "FAILED",
    "LEASED",
    "PENDING",
    "RETRY",
    "DurableConsumerRegistration",
    "DurableDelivery",
    "DurableEventPersistence",
    "DurableInitialDeliveryLease",
    "DurableSignalEvent",
    "DurableSignalStore",
]
