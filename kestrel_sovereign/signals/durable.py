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
from contextlib import asynccontextmanager
from typing import (
    Any,
    AsyncContextManager,
    AsyncIterator,
    Callable,
    Iterable,
    Optional,
    Protocol,
    cast,
    runtime_checkable,
)

from kestrel_sdk.signals import Signal

from kestrel_sovereign.a2a.stores.unified.base import UnifiedStoreBase
from kestrel_sovereign.signals.store import _json_default, _serialize_chain
from kestrel_sovereign.storage.db.interface import DatabaseBackend

PENDING = "pending"
INITIAL_RESERVED = "initial_reserved"
LEASED = "leased"
RETRY = "retry"
ACKNOWLEDGED = "acknowledged"
FAILED = "failed"
# A validation/cycle refusal has no effects to retry, but an ACK-bearing
# provider can redeliver it when its provider-side ACK was lost.  Keep that
# fact distinct from a normal terminal worker failure: only this state is a
# durable, idempotent receipt for a redelivery.
TERMINAL_ACKABLE = "terminal_ackable"
_TERMINAL_STATUSES = frozenset({ACKNOWLEDGED, FAILED, TERMINAL_ACKABLE})
_CLAIMABLE_STATUSES = frozenset({PENDING, RETRY})
_SELECTOR_KEY = re.compile(r"^(?:payload\.[A-Za-z0-9_-]+(?:\.[A-Za-z0-9_-]+)*|session_id|kind)=(.+)$")
_PERSISTED_PAYLOAD = object()
# Callers that own a managed dispatcher normally supply their configured
# threshold explicitly.  Keep direct-store compatibility conservative: a
# recently registered dispatcher must not lose a lease simply because a
# polling caller has no dispatcher instance from which to obtain that policy.
_DEFAULT_RUNTIME_OWNER_STALE_AFTER = timedelta(minutes=2)


@runtime_checkable
class SQLiteImmediateTransactionBackend(Protocol):
    """SQLite capability required to serialize durable-ledger bootstrap."""

    def transaction(self, *, immediate: bool = False) -> AsyncContextManager[None]: ...


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
    # Zero is intentional: a cursor-owning external producer must retain its
    # event until this consumer has acknowledged it, rather than converting a
    # transient outage into a terminal loss after an arbitrary retry budget.
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
    # ``caller_identity`` is an opaque, dispatcher-produced ciphertext for a
    # persistence-allowed caller, a keyless ``v2:opaque:...`` event label, or
    # the ``v1:none`` sentinel.  It is never a raw caller identifier in
    # storage. Payload-elided rows deliberately leave it NULL and bind the
    # caller only through their keyed integrity proof plus the verified live
    # retry envelope.
    caller_identity: Optional[str]
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
    deliveries inserted by this persistence transaction.
    ``initial_reservations`` is populated only for a payload-elided live
    dispatch: those rows are atomically reserved to the emitting dispatcher
    before commit, but deliberately have no delivery lease deadline until the
    dispatcher activates them after the commit is visible.
    """

    event_id: str
    created: bool
    delivery_ids: tuple[str, ...] = ()
    retention_until: Optional[datetime] = None
    initial_reservations: tuple["DurableInitialDeliveryReservation", ...] = ()


@dataclass(frozen=True)
class DurableInitialDeliveryReservation:
    """A non-claimable initial reservation created with its event row.

    The token is a reservation capability, never user payload.  It lets the
    emitting dispatcher activate this row *after* commit, then transfer the
    resulting live lease to its chosen executor without exposing a newly
    committed marker-only delivery to another store instance first.
    """

    delivery_id: str
    consumer_id: str
    reservation_token: str
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
    RUNTIME_OWNERS = "durable_signal_runtime_owners"
    # Payload-eliding privacy modes cannot retain their canonical input in the
    # event row. This side table stores only a fixed-width integrity binding.
    EVENT_INTEGRITY = "durable_signal_event_integrity"

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
        """Bootstrap/evolve the ledger under one cross-process schema lock.

        The delivery tables are shared by independently restarted dispatchers.
        In particular, the integrity side table was added after the original
        ledger, so a plain sequence of ``CREATE IF NOT EXISTS`` calls is not a
        migration protocol: another process can observe a partial schema and
        begin routing before the additive migration finishes.
        """

        ts_type = self.timestamp_type()
        ts_default = self.now_default()
        json_type = self.json_type()
        bool_type = self.boolean_type()
        statements = (
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
                caller_identity TEXT,
                visibility TEXT NOT NULL,
                urgency TEXT NOT NULL,
                dedupe_key TEXT,
                causation_chain {json_type} NOT NULL,
                arrived_at {ts_type} NOT NULL,
                committed_at {ts_type} {ts_default},
                retention_until {ts_type} NOT NULL,
                UNIQUE (agent_id, source, source_event_id)
            )
            """,
            f"""
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
            )
            """,
            f"""
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
            )
            """,
            f"""
            CREATE TABLE IF NOT EXISTS {self.RUNTIME_OWNERS} (
                agent_id TEXT NOT NULL,
                owner_id TEXT NOT NULL,
                heartbeat_at {ts_type} NOT NULL,
                stopped_at {ts_type},
                created_at {ts_type} {ts_default},
                updated_at {ts_type} {ts_default},
                PRIMARY KEY (agent_id, owner_id)
            )
            """,
            f"""
            CREATE TABLE IF NOT EXISTS {self.EVENT_INTEGRITY} (
                event_id TEXT PRIMARY KEY,
                integrity_binding TEXT NOT NULL,
                FOREIGN KEY (event_id) REFERENCES {self.EVENTS}(event_id)
                    ON DELETE CASCADE
            )
            """,
        )
        async with self._schema_bootstrap_transaction():
            for statement in statements:
                await self._backend.execute(statement)
            # ``CREATE TABLE IF NOT EXISTS`` cannot evolve an existing durable
            # ledger. Keep the caller representation additive: legacy rows
            # lack it and are rejected for caller-bearing replay rather than
            # silently accepting an unbound live caller.
            await self._ensure_caller_identity_column()
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
            await self._backend.execute(
                f"CREATE INDEX IF NOT EXISTS idx_{self.RUNTIME_OWNERS}_liveness "
                f"ON {self.RUNTIME_OWNERS}(agent_id, heartbeat_at, stopped_at)"
            )

    @asynccontextmanager
    async def _schema_bootstrap_transaction(self) -> AsyncIterator[None]:
        """Serialize every fresh and additive ledger migration.

        PostgreSQL uses a fixed transaction advisory key; SQLite reserves its
        writer before schema inspection.  Both keep the catalog recheck, DDL,
        and indexes in one transaction rather than relying on an instance-local
        ``initialized`` flag.
        """

        if self.is_postgres:
            async with self._backend.transaction():
                await self._backend.fetch_val(
                    "SELECT pg_advisory_xact_lock(hashtext('kestrel.durable_signal.bootstrap'))"
                )
                yield
            return
        if self.is_sqlite:
            transaction = self._sqlite_immediate_transaction()
            async with transaction:
                yield
            return
        raise RuntimeError("Durable signal delivery supports only sqlite or postgres databases")

    def _sqlite_immediate_transaction(self) -> AsyncContextManager[None]:
        """Return a SQLite transaction that reserves the writer before reads."""

        if not isinstance(self._backend, SQLiteImmediateTransactionBackend):
            raise RuntimeError(
                "SQLite durable signal delivery requires transaction(immediate=True) "
                "for schema bootstrap"
            )
        transaction = cast(SQLiteImmediateTransactionBackend, self._backend).transaction
        try:
            return transaction(immediate=True)
        except TypeError as exc:
            raise RuntimeError(
                "SQLite durable signal delivery requires transaction(immediate=True) "
                "for schema bootstrap"
            ) from exc

    async def _ensure_caller_identity_column(self) -> None:
        """Apply the sole additive event-table migration under the schema lock."""

        if self.is_postgres:
            await self._backend.execute(
                f"ALTER TABLE {self.EVENTS} ADD COLUMN IF NOT EXISTS caller_identity TEXT"
            )
            return
        columns = await self._backend.fetch_all(f"PRAGMA table_info({self.EVENTS})")
        if not any(row[1] == "caller_identity" for row in columns):
            await self._backend.execute(
                f"ALTER TABLE {self.EVENTS} ADD COLUMN caller_identity TEXT"
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
            # The SQLite writer reservation / PostgreSQL advisory lock is the
            # serialization point. Sampling before it can make a retention
            # backfill admit an event that is already expired by the time this
            # transaction owns the handoff.
            now = self.now_utc()
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
        integrity_binding: Optional[str] = None,
        caller_identity: Optional[str] = None,
        caller_identity_factory: Optional[Callable[[], str]] = None,
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
        one live dispatcher in the same transaction as the event insert.  It
        creates an ``INITIAL_RESERVED`` capability, not a lease: there is no
        countdown to expire before the transaction is visible.  The dispatcher
        installs its process-local raw-payload sidecars through
        ``before_commit`` before this transaction becomes visible, then
        activates the reservation after this method returns from commit.  A
        transaction failure invokes ``on_rollback`` so those sidecars cannot
        outlive rows that did not commit.  Async database drivers can report
        cancellation after their worker has completed ``commit``, however, so
        the callback is an *ambiguous transaction-outcome* notification rather
        than proof of rollback. Callers that retained an owner/token capability
        must conditionally repair it; that repair is a no-op for a confirmed
        rollback and releases a row whose commit was already durable. Both
        callbacks are synchronous deliberately:
        yielding between installing the sidecar and committing would reopen
        the very visibility race this handoff closes.
        """
        if retention_days < 0:
            raise ValueError("retention_days must be >= 0")
        self._require_nonempty("agent_id", agent_id)
        self._require_nonempty("source", signal.source)
        if initial_lease_owner is not None:
            self._require_nonempty("initial_lease_owner", initial_lease_owner)
        if integrity_binding is not None and (
            type(integrity_binding) is not str
            or re.fullmatch(r"[0-9a-f]{64}", integrity_binding) is None
        ):
            raise ValueError("integrity_binding must be a SHA-256 hex digest")
        if caller_identity is not None and type(caller_identity) is not str:
            raise ValueError("caller_identity must be an opaque string when set")
        if caller_identity is not None and caller_identity_factory is not None:
            raise ValueError(
                "caller_identity and caller_identity_factory are mutually exclusive"
            )
        if caller_identity_factory is not None and not callable(caller_identity_factory):
            raise ValueError("caller_identity_factory must be callable when set")
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
                        payload, session_id, caller_identity, visibility, urgency,
                        dedupe_key, causation_chain, arrived_at, committed_at,
                        retention_until
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                        caller_identity,
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

                if caller_identity_factory is not None:
                    caller_identity = caller_identity_factory()
                    if type(caller_identity) is not str:
                        raise RuntimeError(
                            "caller_identity_factory returned a non-string value"
                        )
                    await self._backend.execute(
                        f"""
                        UPDATE {self.EVENTS}
                        SET caller_identity = ?
                        WHERE event_id = ? AND agent_id = ?
                        """,
                        (caller_identity, signal.id, agent_id),
                    )

                if integrity_binding is not None:
                    await self._backend.execute(
                        f"""
                        INSERT INTO {self.EVENT_INTEGRITY} (event_id, integrity_binding)
                        VALUES (?, ?)
                        """,
                        (signal.id, integrity_binding),
                    )

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
                    caller_identity=caller_identity,
                    committed_at=now,
                    retention_until=retention_until,
                )
                selector_event = (
                    event
                    if transient_selector_payload is _PERSISTED_PAYLOAD
                    else replace(event, payload=transient_selector_payload)
                )
                delivery_ids: list[str] = []
                initial_reservations: list[DurableInitialDeliveryReservation] = []
                for consumer_id, selector, max_attempts, lease_seconds in consumer_rows:
                    if not self._matches_selector(selector_event, selector):
                        continue
                    reservation_token = None
                    if initial_lease_owner is not None:
                        reservation_token = secrets.token_urlsafe(24)
                    delivery_id = await self._insert_delivery_locked(
                        agent_id=agent_id,
                        consumer_id=consumer_id,
                        event_id=signal.id,
                        max_attempts=int(max_attempts),
                        now=now,
                        initial_reservation_owner=initial_lease_owner,
                        initial_reservation_token=reservation_token,
                    )
                    if delivery_id is not None:
                        delivery_ids.append(delivery_id)
                        if reservation_token is not None:
                            initial_reservations.append(
                                DurableInitialDeliveryReservation(
                                    delivery_id=delivery_id,
                                    consumer_id=consumer_id,
                                    reservation_token=reservation_token,
                                    created_at=now,
                                )
                            )
                persistence = DurableEventPersistence(
                    event_id=signal.id,
                    created=True,
                    delivery_ids=tuple(delivery_ids),
                    retention_until=retention_until,
                    initial_reservations=tuple(initial_reservations),
                )
                if before_commit is not None:
                    before_commit(persistence)
        except BaseException:
            if persistence is not None and on_rollback is not None:
                on_rollback(persistence)
            raise
        assert persistence is not None
        return persistence

    async def upgrade_legacy_delivery_for_redelivery(
        self,
        *,
        agent_id: str,
        consumer_id: str,
        event_id: str,
        source_event_id: str,
        expected_signal: Signal,
        caller_identity_factory: Callable[[], str],
    ) -> bool:
        """Atomically add one delivery to a verified pre-consumer event.

        This deliberately is *not* a consumer backfill.  It upgrades only the
        immutable event named by a provider's current redelivery, after the
        caller has proved that its normalized live envelope matches the old
        retained event.  The caller identity was not protected by the
        pre-upgrade schema, so it is sealed and the exact delivery is created
        in the same transaction.  Privacy-elided rows carry an integrity row
        and are refused here; their keyed-MAC retry path remains fail-closed.
        """

        self._require_nonempty("agent_id", agent_id)
        self._require_nonempty("consumer_id", consumer_id)
        self._require_nonempty("event_id", event_id)
        self._require_nonempty("source_event_id", source_event_id)
        self._require_nonempty("source", expected_signal.source)
        if not callable(caller_identity_factory):
            raise ValueError("caller_identity_factory must be callable")

        async with self._backend.transaction():
            await self._lock_scope_handoff(
                agent_id=agent_id, source=expected_signal.source
            )
            # This upgrade is a live redelivery operation, not a historical
            # repair.  Sample once after the transaction has acquired the
            # source handoff lock and use that same instant for both retention
            # admission and the new delivery's timestamps.  Backfill retains
            # the exact boundary (``retention_until >= now``), so a row at the
            # boundary is still eligible here as well.
            now = self.now_utc()
            consumer = await self._get_consumer(agent_id, consumer_id)
            if consumer is None or not consumer[4] or consumer[0] != expected_signal.source:
                return False
            row = await self._backend.fetch_one(
                f"""
                SELECT event_id, source_event_id, agent_id, target_agent, source, kind, mode, payload,
                       session_id, caller_identity, visibility, urgency, dedupe_key,
                       causation_chain, arrived_at, committed_at, retention_until
                FROM {self.EVENTS}
                WHERE event_id = ? AND agent_id = ? AND source = ?
                """,
                (event_id, agent_id, expected_signal.source),
            )
            if row is None:
                return False
            event = self._row_to_event(row)
            if (
                event.retention_until < now
                or
                event.caller_identity is not None
                or not self._legacy_event_matches_redelivery(
                    event, expected_signal, source_event_id
                )
                # An event that would already have matched the registered
                # selector is not a marker-era legacy row.  Never use a
                # redelivery to alter such historical work.
                or self._matches_selector(event, consumer[1])
            ):
                return False
            integrity = await self._backend.fetch_one(
                f"SELECT 1 FROM {self.EVENT_INTEGRITY} WHERE event_id = ?",
                (event_id,),
            )
            if integrity is not None:
                return False
            delivery = await self._backend.fetch_one(
                f"""
                SELECT 1 FROM {self.DELIVERIES}
                WHERE agent_id = ? AND consumer_id = ? AND event_id = ?
                """,
                (agent_id, consumer_id, event_id),
            )
            if delivery is not None:
                return False

            caller_identity = caller_identity_factory()
            if type(caller_identity) is not str or not caller_identity:
                raise RuntimeError(
                    "caller_identity_factory returned an invalid protected value"
                )

            updated = await self._backend.execute(
                f"""
                UPDATE {self.EVENTS}
                SET caller_identity = ?
                WHERE event_id = ? AND agent_id = ? AND caller_identity IS NULL
                """,
                (caller_identity, event_id, agent_id),
            )
            if updated != 1:
                return False
            delivery_id = await self._insert_delivery_locked(
                agent_id=agent_id,
                consumer_id=consumer_id,
                event_id=event_id,
                max_attempts=int(consumer[2]),
                now=now,
            )
            if delivery_id is None:
                # This transaction owns the source handoff lock, so an
                # unexpected duplicate would otherwise leave an identity-only
                # partial upgrade.  Raising rolls every change back together.
                raise RuntimeError("legacy delivery upgrade conflicted unexpectedly")
            return True

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
        runtime_owner_stale_before: Optional[datetime] = None,
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
        explicit_now = _as_utc(now) if now is not None else None
        consumer = await self._get_consumer(agent_id, consumer_id)
        if consumer is None or not consumer[4]:
            return None
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
        # Claim, recovery, and restart backfill all serialize with event
        # persistence for this source.  Besides avoiding a registration/event
        # visibility gap, this makes SQLite's single writer serialization
        # explicit. PostgreSQL still needs to serialize the actual delivery
        # row below before it observes an implicit lease clock.
        async with self._backend.transaction():
            await self._lock_scope_handoff(agent_id=agent_id, source=registration.source)
            recovery_now = explicit_now or self.now_utc()
            await self._recover_expired_leases(
                agent_id=agent_id,
                consumer_id=consumer_id,
                now=recovery_now,
                runtime_owner_stale_before=(
                    _as_utc(runtime_owner_stale_before)
                    if runtime_owner_stale_before is not None
                    else recovery_now - _DEFAULT_RUNTIME_OWNER_STALE_AFTER
                ),
            )
            # Backfill is idempotent because delivery identity is unique.
            await self._backfill_consumer(registration, now=recovery_now)
            delivery_id = await self._lock_claimable_delivery(
                agent_id=agent_id,
                consumer_id=consumer_id,
                now=recovery_now,
            )
            if delivery_id is None:
                return None
            # A PostgreSQL row lock can wait behind a worker that is slower
            # than this consumer's entire lease.  Preserve an explicitly
            # supplied timestamp exactly, but otherwise sample only after the
            # selected delivery is serialized so the new lease is full-lived.
            effective_now = explicit_now or self.now_utc()
            lease_token = secrets.token_urlsafe(24)
            lease_expires_at = effective_now + timedelta(
                seconds=registration.lease_seconds
            )
            updated = await self._backend.execute(
                f"""
                UPDATE {self.DELIVERIES}
                SET status = ?, attempts = attempts + 1, lease_owner = ?,
                    lease_token = ?, lease_expires_at = ?, next_attempt_at = NULL,
                    updated_at = ?
                WHERE delivery_id = ? AND agent_id = ? AND consumer_id = ?
                  AND status IN ('{PENDING}', '{RETRY}')
                  AND (max_attempts = 0 OR attempts < max_attempts)
                  AND (next_attempt_at IS NULL OR next_attempt_at <= ?)
                """,
                (
                    LEASED,
                    executor_id,
                    lease_token,
                    self.to_timestamp_param(lease_expires_at),
                    self.to_timestamp_param(effective_now),
                    delivery_id,
                    agent_id,
                    consumer_id,
                    self.to_timestamp_param(effective_now),
                ),
            )
        if updated == 0:
            return None
        return await self._delivery_for_lease_locked(
            agent_id=agent_id,
            consumer_id=consumer_id,
            lease_token=lease_token,
        )

    async def claim_delivery_for_event(
        self,
        *,
        agent_id: str,
        consumer_id: str,
        event_id: str,
        executor_id: str,
        now: Optional[datetime] = None,
        runtime_owner_stale_before: Optional[datetime] = None,
    ) -> Optional[DurableDelivery]:
        """Atomically claim this consumer's delivery for one persisted event.

        Cursor-owning ingress must never let a concurrent callback claim an
        unrelated pending event and then acknowledge the wrong provider
        cursor.  This is the exact-event counterpart of ``claim_delivery``.
        """
        self._require_nonempty("agent_id", agent_id)
        self._require_nonempty("consumer_id", consumer_id)
        self._require_nonempty("event_id", event_id)
        self._require_nonempty("executor_id", executor_id)
        explicit_now = _as_utc(now) if now is not None else None
        consumer = await self._get_consumer(agent_id, consumer_id)
        if consumer is None or not consumer[4]:
            return None
        async with self._backend.transaction():
            await self._lock_scope_handoff(agent_id=agent_id, source=consumer[0])
            recovery_now = explicit_now or self.now_utc()
            await self._recover_expired_leases(
                agent_id=agent_id,
                consumer_id=consumer_id,
                now=recovery_now,
                runtime_owner_stale_before=(
                    _as_utc(runtime_owner_stale_before)
                    if runtime_owner_stale_before is not None
                    else recovery_now - _DEFAULT_RUNTIME_OWNER_STALE_AFTER
                ),
            )
            delivery_id = await self._lock_claimable_delivery(
                agent_id=agent_id,
                consumer_id=consumer_id,
                event_id=event_id,
                now=recovery_now,
            )
            if delivery_id is None:
                return None
            # See claim_delivery: do not publish an already-expired implicit
            # lease after waiting for this exact delivery row.
            effective_now = explicit_now or self.now_utc()
            lease_token = secrets.token_urlsafe(24)
            lease_expires_at = effective_now + timedelta(seconds=int(consumer[3]))
            updated = await self._backend.execute(
                f"""
                UPDATE {self.DELIVERIES}
                SET status = ?, attempts = attempts + 1, lease_owner = ?,
                    lease_token = ?, lease_expires_at = ?, next_attempt_at = NULL,
                    updated_at = ?
                WHERE delivery_id = ? AND agent_id = ? AND consumer_id = ? AND event_id = ?
                  AND status IN ('{PENDING}', '{RETRY}')
                  AND (max_attempts = 0 OR attempts < max_attempts)
                  AND (next_attempt_at IS NULL OR next_attempt_at <= ?)
                """,
                (
                    LEASED,
                    executor_id,
                    lease_token,
                    self.to_timestamp_param(lease_expires_at),
                    self.to_timestamp_param(effective_now),
                    delivery_id,
                    agent_id,
                    consumer_id,
                    event_id,
                    self.to_timestamp_param(effective_now),
                ),
            )
        if updated == 0:
            return None
        return await self._delivery_for_lease_locked(
            agent_id=agent_id,
            consumer_id=consumer_id,
            lease_token=lease_token,
        )

    async def _lock_claimable_delivery(
        self,
        *,
        agent_id: str,
        consumer_id: str,
        now: datetime,
        event_id: Optional[str] = None,
    ) -> Optional[str]:
        """Serialize one due delivery before assigning an implicit lease clock.

        ``_lock_scope_handoff`` protects event/consumer registration handoff,
        but a PostgreSQL transaction may still block on the selected delivery
        row itself (for example, an operator repair holding that row).  Select
        and lock that exact row before the caller samples its implicit clock.
        SQLite has already acquired its single writer in ``_lock_scope_handoff``;
        the same select keeps both backends on one claim contract.
        """

        where = [
            "agent_id = ?",
            "consumer_id = ?",
            f"status IN ('{PENDING}', '{RETRY}')",
            "(max_attempts = 0 OR attempts < max_attempts)",
            "(next_attempt_at IS NULL OR next_attempt_at <= ?)",
        ]
        params: list[Any] = [agent_id, consumer_id, self.to_timestamp_param(now)]
        if event_id is not None:
            where.insert(2, "event_id = ?")
            params.insert(2, event_id)
        lock_clause = " FOR UPDATE" if self.is_postgres else ""
        row = await self._backend.fetch_one(
            f"""
            SELECT delivery_id FROM {self.DELIVERIES}
            WHERE {' AND '.join(where)}
            ORDER BY created_at, delivery_id
            LIMIT 1{lock_clause}
            """,
            tuple(params),
        )
        return str(row[0]) if row is not None else None

    async def renew_delivery_lease(
        self,
        *,
        agent_id: str,
        consumer_id: str,
        delivery_id: str,
        lease_token: str,
        now: Optional[datetime] = None,
    ) -> Optional[DurableDelivery]:
        """Extend one still-owned lease using the consumer's persisted policy.

        A long cognition turn keeps its original token. This conditional update
        refuses an expired or foreign lease, so a late worker can never revive
        work another executor may already have claimed.
        """

        self._require_nonempty("agent_id", agent_id)
        self._require_nonempty("consumer_id", consumer_id)
        self._require_nonempty("delivery_id", delivery_id)
        self._require_nonempty("lease_token", lease_token)
        consumer = await self._get_consumer(agent_id, consumer_id)
        if consumer is None or not consumer[4]:
            return None
        now = _as_utc(now or self.now_utc())
        lease_expires_at = now + timedelta(seconds=int(consumer[3]))
        updated = await self._backend.execute(
            f"""
            UPDATE {self.DELIVERIES}
            SET lease_expires_at = ?, updated_at = ?
            WHERE agent_id = ? AND consumer_id = ? AND delivery_id = ?
              AND status = ? AND lease_token = ? AND lease_expires_at > ?
            """,
            (
                self.to_timestamp_param(lease_expires_at),
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
        """Transfer one activated emitting-dispatcher lease to a worker.

        An initial reservation first becomes a real ``LEASED`` row through
        :meth:`activate_initial_delivery`, which runs only after the event
        transaction has committed.  A normal claimant cannot claim either the
        reservation or the activated owner lease.  Only the dispatcher holding
        this unpersisted capability may make the first worker claim; after
        that, ordinary retry/lease rules apply.
        """
        self._require_nonempty("agent_id", agent_id)
        self._require_nonempty("consumer_id", consumer_id)
        self._require_nonempty("delivery_id", delivery_id)
        self._require_nonempty("initial_lease_owner", initial_lease_owner)
        self._require_nonempty("initial_lease_token", initial_lease_token)
        self._require_nonempty("executor_id", executor_id)
        consumer = await self._get_consumer(agent_id, consumer_id)
        if consumer is None or not consumer[4]:
            return None
        requested_now = _as_utc(now) if now is not None else None
        async with self._backend.transaction():
            # Acquire the delivery's write/row serialization point before
            # observing time.  SQLite transactions begin deferred, so the
            # targeted no-op update reserves the actual writer slot; PostgreSQL
            # locks this row with ``FOR UPDATE``.  Sampling first can publish a
            # worker lease that has already expired while waiting here.
            initial = await self._lock_initial_delivery_transfer(
                agent_id=agent_id,
                consumer_id=consumer_id,
                delivery_id=delivery_id,
            )
            if initial is None:
                return None
            status, owner, token, expires_at = initial
            transfer_now = requested_now or self.now_utc()
            if (
                status != LEASED
                or owner != initial_lease_owner
                or token != initial_lease_token
                or expires_at is None
                or _as_utc(self.from_timestamp_field(expires_at)) <= transfer_now
            ):
                return None
            lease_token = secrets.token_urlsafe(24)
            lease_expires_at = transfer_now + timedelta(seconds=int(consumer[3]))
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
                    self.to_timestamp_param(transfer_now),
                    agent_id,
                    consumer_id,
                    delivery_id,
                    LEASED,
                    initial_lease_owner,
                    initial_lease_token,
                    self.to_timestamp_param(transfer_now),
                ),
            )
        if updated == 0:
            return None
        return await self._delivery_for_lease_locked(
            agent_id=agent_id,
            consumer_id=consumer_id,
            lease_token=lease_token,
        )

    async def activate_initial_delivery(
        self,
        *,
        agent_id: str,
        consumer_id: str,
        delivery_id: str,
        initial_lease_owner: str,
        initial_lease_token: str,
        now: Optional[datetime] = None,
    ) -> Optional[DurableDelivery]:
        """Start a reservation's first real lease after its event commits.

        ``persist_signal`` inserts an ``INITIAL_RESERVED`` row with no lease
        deadline.  This conditional transition is intentionally a separate
        post-commit operation: it is the first place a live lease countdown is
        calculated, so a paused commit can never publish an already-expired
        delivery.  The runtime-owner heartbeat and this transition share a
        transaction so stale-owner recovery cannot take a live emitter that is
        actively activating its own reservation.
        """
        self._require_nonempty("agent_id", agent_id)
        self._require_nonempty("consumer_id", consumer_id)
        self._require_nonempty("delivery_id", delivery_id)
        self._require_nonempty("initial_lease_owner", initial_lease_owner)
        self._require_nonempty("initial_lease_token", initial_lease_token)
        consumer = await self._get_consumer(agent_id, consumer_id)
        if consumer is None or not consumer[4]:
            return None
        requested_now = _as_utc(now) if now is not None else None
        async with self._backend.transaction():
            await self._lock_runtime_owner_scope(agent_id=agent_id)
            # The transaction may wait behind a real database writer. Sample
            # time only after that contention has cleared and immediately
            # before the activation write; this is the first live delivery
            # deadline and must not inherit any pre-commit delay.
            activation_now = requested_now or self.now_utc()
            await self._touch_runtime_owner_locked(
                agent_id=agent_id,
                owner_id=initial_lease_owner,
                now=activation_now,
            )
            activation_now = requested_now or self.now_utc()
            lease_expires_at = activation_now + timedelta(
                seconds=int(consumer[3])
            )
            updated = await self._backend.execute(
                f"""
                UPDATE {self.DELIVERIES}
                SET status = ?, lease_expires_at = ?, updated_at = ?
                WHERE agent_id = ? AND consumer_id = ? AND delivery_id = ?
                  AND status = ? AND lease_owner = ? AND lease_token = ?
                  AND lease_expires_at IS NULL
                """,
                (
                    LEASED,
                    self.to_timestamp_param(lease_expires_at),
                    self.to_timestamp_param(activation_now),
                    agent_id,
                    consumer_id,
                    delivery_id,
                    INITIAL_RESERVED,
                    initial_lease_owner,
                    initial_lease_token,
                ),
            )
        if updated == 0:
            return None
        return await self._delivery_for_lease_locked(
            agent_id=agent_id,
            consumer_id=consumer_id,
            lease_token=initial_lease_token,
        )

    async def register_runtime_owner(
        self,
        *,
        agent_id: str,
        owner_id: str,
        now: Optional[datetime] = None,
    ) -> None:
        """Record a live dispatcher generation before it can reserve work."""
        self._require_nonempty("agent_id", agent_id)
        self._require_nonempty("owner_id", owner_id)
        requested_now = _as_utc(now) if now is not None else None
        async with self._backend.transaction():
            await self._lock_runtime_owner_scope(agent_id=agent_id)
            # Entering this transaction may wait behind a real writer.  A
            # default timestamp is liveness evidence, so sample it only after
            # that contention clears rather than publishing an old heartbeat.
            touch_now = requested_now or self.now_utc()
            await self._touch_runtime_owner_locked(
                agent_id=agent_id, owner_id=owner_id, now=touch_now
            )

    async def heartbeat_runtime_owner(
        self,
        *,
        agent_id: str,
        owner_id: str,
        now: Optional[datetime] = None,
    ) -> None:
        """Refresh one dispatcher generation's liveness record."""
        self._require_nonempty("agent_id", agent_id)
        self._require_nonempty("owner_id", owner_id)
        requested_now = _as_utc(now) if now is not None else None
        async with self._backend.transaction():
            # Recovery uses this exact scope before it decides whether a
            # managed lease owner is stale.  Without the common lock a
            # PostgreSQL recovery snapshot can classify the old heartbeat as
            # stale while this refresh is concurrently committing.
            await self._lock_runtime_owner_scope(agent_id=agent_id)
            # See register_runtime_owner: a heartbeat taken before waiting on
            # this transaction is not trustworthy liveness evidence.
            touch_now = requested_now or self.now_utc()
            await self._touch_runtime_owner_locked(
                agent_id=agent_id, owner_id=owner_id, now=touch_now
            )

    async def release_initial_reservations(
        self,
        *,
        agent_id: str,
        owner_id: str,
        now: Optional[datetime] = None,
        mark_owner_stopped: bool = True,
    ) -> int:
        """Release this runtime's unactivated rows into marker replay.

        This is deliberately scoped to one runtime owner.  A concurrent live
        dispatcher cannot release another emitter's raw-payload reservation.
        A cancellation-resistant cognition task can outlive ordinary shutdown;
        callers retain that owner's liveness fence by passing
        ``mark_owner_stopped=False`` until the task is actually settled.
        """
        self._require_nonempty("agent_id", agent_id)
        self._require_nonempty("owner_id", owner_id)
        now = _as_utc(now or self.now_utc())
        async with self._backend.transaction():
            await self._lock_runtime_owner_scope(agent_id=agent_id)
            released = await self._backend.execute(
                f"""
                UPDATE {self.DELIVERIES}
                SET status = ?, lease_owner = NULL, lease_token = NULL,
                    lease_expires_at = NULL, next_attempt_at = ?,
                    last_error = 'initial reservation owner stopped before activation',
                    updated_at = ?
                WHERE agent_id = ? AND status = ? AND lease_owner = ?
                """,
                (
                    RETRY,
                    self.to_timestamp_param(now),
                    self.to_timestamp_param(now),
                    agent_id,
                    INITIAL_RESERVED,
                    owner_id,
                ),
            )
            if mark_owner_stopped:
                await self._backend.execute(
                    f"""
                    UPDATE {self.RUNTIME_OWNERS}
                    SET heartbeat_at = ?, stopped_at = ?, updated_at = ?
                    WHERE agent_id = ? AND owner_id = ?
                    """,
                    (
                        self.to_timestamp_param(now),
                        self.to_timestamp_param(now),
                        self.to_timestamp_param(now),
                        agent_id,
                        owner_id,
                    ),
                )
        return released

    async def abandon_initial_reservation(
        self,
        *,
        agent_id: str,
        consumer_id: str,
        delivery_id: str,
        owner_id: str,
        reservation_token: str,
        now: Optional[datetime] = None,
    ) -> bool:
        """Release one initial capability whose raw handoff cannot complete.

        The owner/token pair identifies both an unactivated reservation and
        its just-activated first lease.  The latter case is possible only if
        the activation write committed before its readback failed; no worker
        can own it yet because the dispatcher still holds its local handoff
        lock.  Both forms must become marker-only retry work.
        """
        self._require_nonempty("agent_id", agent_id)
        self._require_nonempty("consumer_id", consumer_id)
        self._require_nonempty("delivery_id", delivery_id)
        self._require_nonempty("owner_id", owner_id)
        self._require_nonempty("reservation_token", reservation_token)
        now = _as_utc(now or self.now_utc())
        async with self._backend.transaction():
            released = await self._backend.execute(
                f"""
                UPDATE {self.DELIVERIES}
                SET status = ?, lease_owner = NULL, lease_token = NULL,
                    lease_expires_at = NULL, next_attempt_at = ?,
                    last_error = 'initial reservation activation unavailable',
                    updated_at = ?
                WHERE agent_id = ? AND consumer_id = ? AND delivery_id = ?
                  AND status IN (?, ?) AND lease_owner = ? AND lease_token = ?
                """,
                (
                    RETRY,
                    self.to_timestamp_param(now),
                    self.to_timestamp_param(now),
                    agent_id,
                    consumer_id,
                    delivery_id,
                    INITIAL_RESERVED,
                    LEASED,
                    owner_id,
                    reservation_token,
                ),
            )
        return released == 1

    async def recover_abandoned_initial_reservations(
        self,
        *,
        agent_id: str,
        recovering_owner_id: str,
        stale_before: datetime,
        now: Optional[datetime] = None,
    ) -> int:
        """Requeue reservations only from stale managed dispatcher owners.

        Generic claim/retry recovery intentionally never touches
        ``INITIAL_RESERVED``.  Startup uses this owner-aware path instead;
        another live dispatcher remains protected by its heartbeat even when
        it shares the same tenant and consumer IDs. Public executor owners and
        unknown owner namespaces are never recovery candidates.
        """
        self._require_nonempty("agent_id", agent_id)
        self._require_nonempty("recovering_owner_id", recovering_owner_id)
        stale_before = _as_utc(stale_before)
        now = _as_utc(now or self.now_utc())
        async with self._backend.transaction():
            await self._lock_runtime_owner_scope(agent_id=agent_id)
            released = await self._backend.execute(
                f"""
                UPDATE {self.DELIVERIES}
                SET status = ?, lease_owner = NULL, lease_token = NULL,
                    lease_expires_at = NULL, next_attempt_at = ?,
                    last_error = 'initial reservation owner unavailable before activation',
                    updated_at = ?
                WHERE agent_id = ? AND status = ? AND lease_owner <> ?
                  AND lease_owner LIKE 'dispatcher:%'
                  AND EXISTS (
                      SELECT 1 FROM {self.RUNTIME_OWNERS} owner
                      WHERE owner.agent_id = {self.DELIVERIES}.agent_id
                        AND owner.owner_id = {self.DELIVERIES}.lease_owner
                        AND (
                            owner.stopped_at IS NOT NULL
                            OR owner.heartbeat_at < ?
                        )
                  )
                """,
                (
                    RETRY,
                    self.to_timestamp_param(now),
                    self.to_timestamp_param(now),
                    agent_id,
                    INITIAL_RESERVED,
                    recovering_owner_id,
                    self.to_timestamp_param(stale_before),
                ),
            )
        return released

    async def recover_abandoned_leases(
        self,
        *,
        agent_id: str,
        recovering_owner_id: str,
        stale_before: datetime,
        now: Optional[datetime] = None,
    ) -> int:
        """Requeue only managed dispatcher leases whose owner is stale/stopped.

        A normal lease can be committed before cognition begins.  On restart,
        retaining that lease until its deadline would make a provider callback
        look like a duplicate even though its cognition was never made safe.
        Owner-aware recovery restores that delivery without disturbing a live
        sibling dispatcher sharing the same tenant. It deliberately preserves
        public executor leases and unknown ownership domains.
        """
        self._require_nonempty("agent_id", agent_id)
        self._require_nonempty("recovering_owner_id", recovering_owner_id)
        stale_before = _as_utc(stale_before)
        now = _as_utc(now or self.now_utc())
        timestamp = self._timestamp_placeholder()
        async with self._backend.transaction():
            await self._lock_runtime_owner_scope(agent_id=agent_id)
            released = await self._backend.execute(
                f"""
                UPDATE {self.DELIVERIES}
                SET status = CASE
                        WHEN max_attempts > 0 AND attempts >= max_attempts THEN ?
                        ELSE ?
                    END,
                    lease_owner = NULL, lease_token = NULL, lease_expires_at = NULL,
                    next_attempt_at = CASE
                        WHEN max_attempts > 0 AND attempts >= max_attempts THEN NULL
                        ELSE {timestamp}
                    END,
                    last_error = 'lease owner unavailable before acknowledgement',
                    terminal_at = CASE
                        WHEN max_attempts > 0 AND attempts >= max_attempts
                        THEN {timestamp} ELSE NULL
                    END,
                    updated_at = ?
                WHERE agent_id = ? AND status = ? AND lease_owner <> ?
                  AND lease_owner LIKE 'dispatcher:%'
                  AND EXISTS (
                      SELECT 1 FROM {self.RUNTIME_OWNERS} owner
                      WHERE owner.agent_id = {self.DELIVERIES}.agent_id
                        AND owner.owner_id = {self.DELIVERIES}.lease_owner
                        AND (
                            owner.stopped_at IS NOT NULL
                            OR owner.heartbeat_at < ?
                        )
                  )
                """,
                (
                    FAILED,
                    RETRY,
                    self.to_timestamp_param(now),
                    self.to_timestamp_param(now),
                    self.to_timestamp_param(now),
                    agent_id,
                    LEASED,
                    recovering_owner_id,
                    self.to_timestamp_param(stale_before),
                ),
            )
        return released

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
        terminal_ackable: bool = False,
        now: Optional[datetime] = None,
    ) -> Optional[DurableDelivery]:
        """Release a failed lease for retry or mark a terminal failure.

        Retry is bounded by the delivery's persisted ``max_attempts`` unless
        that value is zero, which intentionally means retry until an explicit
        terminal failure or acknowledgement.
        """
        self._require_nonempty("error", error)
        if retry_delay.total_seconds() < 0:
            raise ValueError("retry_delay must not be negative")
        if terminal_ackable and not terminal:
            raise ValueError("terminal_ackable deliveries must be terminal")
        now = _as_utc(now or self.now_utc())
        retry_at = now + retry_delay
        timestamp = self._timestamp_placeholder()
        updated = await self._backend.execute(
            f"""
            UPDATE {self.DELIVERIES}
            SET status = CASE
                    WHEN ? THEN ?
                    WHEN ? OR (max_attempts > 0 AND attempts >= max_attempts) THEN ?
                    ELSE ?
                END,
                lease_owner = NULL, lease_token = NULL, lease_expires_at = NULL,
                next_attempt_at = CASE
                    WHEN ? OR ? OR (max_attempts > 0 AND attempts >= max_attempts)
                        THEN NULL ELSE {timestamp} END,
                last_error = ?,
                terminal_at = CASE
                    WHEN ? OR ? OR (max_attempts > 0 AND attempts >= max_attempts)
                        THEN {timestamp} ELSE NULL
                    END,
                updated_at = ?
            WHERE agent_id = ? AND consumer_id = ? AND delivery_id = ?
              AND status = ? AND lease_token = ? AND lease_expires_at > ?
            """,
            (
                self.to_bool_param(terminal_ackable),
                TERMINAL_ACKABLE,
                self.to_bool_param(terminal),
                FAILED,
                RETRY,
                self.to_bool_param(terminal_ackable),
                self.to_bool_param(terminal),
                self.to_timestamp_param(retry_at),
                error,
                self.to_bool_param(terminal_ackable),
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

    async def release_managed_delivery_after_task(
        self,
        *,
        agent_id: str,
        consumer_id: str,
        delivery_id: str,
        lease_token: str,
        owner_id: str,
        error: str,
        terminal: bool = False,
        terminal_ackable: bool = False,
        now: Optional[datetime] = None,
    ) -> Optional[DurableDelivery]:
        """Release a completed local task's exact managed lease.

        A dispatcher may learn that its renewal path is lost while the
        cognition coroutine ignores cancellation.  Normal NACK intentionally
        refuses an expired lease, but the live managed-owner heartbeat keeps
        that lease out of generic expiry recovery until this exact coroutine
        has settled.  At that point this owner/token conditional transition is
        safe even after the nominal deadline: no other claimant could have
        acquired the row while the owner was live.  The owner predicate keeps
        this narrow escape hatch unavailable to public or unknown executors.
        """

        self._require_nonempty("agent_id", agent_id)
        self._require_nonempty("consumer_id", consumer_id)
        self._require_nonempty("delivery_id", delivery_id)
        self._require_nonempty("lease_token", lease_token)
        self._require_nonempty("owner_id", owner_id)
        self._require_nonempty("error", error)
        if terminal_ackable and not terminal:
            raise ValueError("terminal_ackable deliveries must be terminal")
        if not owner_id.startswith("dispatcher:"):
            raise ValueError("owner_id must identify a managed dispatcher")
        now = _as_utc(now or self.now_utc())
        timestamp = self._timestamp_placeholder()
        updated = await self._backend.execute(
            f"""
            UPDATE {self.DELIVERIES}
            SET status = CASE
                    WHEN ? THEN ?
                    WHEN ? OR (max_attempts > 0 AND attempts >= max_attempts) THEN ?
                    ELSE ?
                END,
                lease_owner = NULL, lease_token = NULL, lease_expires_at = NULL,
                next_attempt_at = CASE
                    WHEN ? OR ? OR (max_attempts > 0 AND attempts >= max_attempts)
                        THEN NULL ELSE {timestamp}
                END,
                last_error = ?,
                terminal_at = CASE
                    WHEN ? OR ? OR (max_attempts > 0 AND attempts >= max_attempts)
                        THEN {timestamp} ELSE NULL
                END,
                updated_at = ?
            WHERE agent_id = ? AND consumer_id = ? AND delivery_id = ?
              AND status = ? AND lease_owner = ? AND lease_token = ?
            """,
            (
                self.to_bool_param(terminal_ackable),
                TERMINAL_ACKABLE,
                self.to_bool_param(terminal),
                FAILED,
                RETRY,
                self.to_bool_param(terminal_ackable),
                self.to_bool_param(terminal),
                self.to_timestamp_param(now),
                error,
                self.to_bool_param(terminal_ackable),
                self.to_bool_param(terminal),
                self.to_timestamp_param(now),
                self.to_timestamp_param(now),
                agent_id,
                consumer_id,
                delivery_id,
                LEASED,
                owner_id,
                lease_token,
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

    async def get_delivery_for_event(
        self, *, agent_id: str, consumer_id: str, event_id: str
    ) -> Optional[DurableDelivery]:
        """Read one consumer delivery by its immutable persisted event ID."""
        row = await self._backend.fetch_one(
            self._delivery_select_sql(
                "d.agent_id = ? AND d.consumer_id = ? AND d.event_id = ?"
            ),
            (agent_id, consumer_id, event_id),
        )
        return self._row_to_delivery(row) if row is not None else None

    async def get_event_integrity(
        self, *, agent_id: str, event_id: str
    ) -> Optional[str]:
        """Return one agent-scoped privacy-safe durable event binding."""

        self._require_nonempty("agent_id", agent_id)
        self._require_nonempty("event_id", event_id)
        row = await self._backend.fetch_one(
            f"""
            SELECT integrity.integrity_binding
            FROM {self.EVENT_INTEGRITY} integrity
            JOIN {self.EVENTS} event ON event.event_id = integrity.event_id
            WHERE integrity.event_id = ? AND event.agent_id = ?
            """,
            (event_id, agent_id),
        )
        return str(row[0]) if row is not None else None

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
            valid = (
                _CLAIMABLE_STATUSES
                | {INITIAL_RESERVED, LEASED}
                | _TERMINAL_STATUSES
            )
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
                    AND d.status NOT IN ('{ACKNOWLEDGED}', '{FAILED}', '{TERMINAL_ACKABLE}')
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
            or registration.max_attempts < 0
        ):
            raise ValueError("max_attempts must be >= 0")
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

    @staticmethod
    def _legacy_event_matches_redelivery(
        event: DurableSignalEvent,
        signal: Signal,
        source_event_id: str,
    ) -> bool:
        """Compare the retained canonical envelope, excluding fresh attempt IDs.

        A provider retry receives a new signal/outcome ID and causation frame,
        neither of which was stable in the pre-consumer ledger.  Every source
        identity and normalized payload field that *was* retained must match.
        """

        return (
            event.source_event_id == source_event_id
            and event.target_agent == signal.target_agent
            and event.source == signal.source
            and event.kind == signal.kind
            and event.mode == signal.mode.value
            and event.payload == signal.payload
            and event.session_id == signal.session_id
            and event.visibility == signal.visibility.value
            and event.urgency == signal.urgency.value
            and event.dedupe_key == signal.dedupe_key
        )

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

    async def _lock_runtime_owner_scope(self, *, agent_id: str) -> None:
        """Serialize liveness heartbeats and recovery for one tenant.

        The durable owner row is read by recovery predicates but updated by a
        separate heartbeat transaction.  PostgreSQL's statement snapshots do
        not make that read/update pair mutually exclusive on their own, so
        both paths take one transaction-scoped advisory key.  SQLite reserves
        its single writer before either path inspects owner liveness.  This is
        intentionally tenant-wide: recovery can assess several dispatcher
        generations in one statement, and a per-owner lock would leave the
        predicate race open for every other owner it scans.
        """
        if self.is_postgres:
            await self._backend.fetch_val(
                "SELECT pg_advisory_xact_lock(hashtextextended(?, 0))",
                (f"durable-signal-runtime-owner:{agent_id}",),
            )
            return
        if self._backend.backend_type == "sqlite":
            await self._backend.execute(
                f"UPDATE {self.RUNTIME_OWNERS} SET updated_at = updated_at WHERE agent_id = ?",
                (agent_id,),
            )
            return
        raise RuntimeError(
            "Durable runtime-owner serialization does not support backend "
            f"{self._backend.backend_type!r}"
        )

    async def _lock_initial_delivery_transfer(
        self, *, agent_id: str, consumer_id: str, delivery_id: str
    ) -> Optional[tuple[Any, ...]]:
        """Lock an activated initial delivery before starting its worker lease.

        This deliberately uses the narrow delivery row rather than the
        registration/persistence scope lock: all we need here is stable
        ownership and a current lease deadline for one post-commit handoff.
        """
        params = (agent_id, consumer_id, delivery_id)
        if self.is_postgres:
            return await self._backend.fetch_one(
                f"""
                SELECT status, lease_owner, lease_token, lease_expires_at
                FROM {self.DELIVERIES}
                WHERE agent_id = ? AND consumer_id = ? AND delivery_id = ?
                FOR UPDATE
                """,
                params,
            )
        if self._backend.backend_type == "sqlite":
            # ``BEGIN`` is deferred in SQLite. Updating the exact delivery to
            # its current value claims the writer slot before the following
            # read/time sample, while preserving every persisted field.
            await self._backend.execute(
                f"""
                UPDATE {self.DELIVERIES}
                SET updated_at = updated_at
                WHERE agent_id = ? AND consumer_id = ? AND delivery_id = ?
                """,
                params,
            )
            return await self._backend.fetch_one(
                f"""
                SELECT status, lease_owner, lease_token, lease_expires_at
                FROM {self.DELIVERIES}
                WHERE agent_id = ? AND consumer_id = ? AND delivery_id = ?
                """,
                params,
            )
        raise RuntimeError(
            "Durable signal initial-delivery transfer does not support backend "
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

    async def _touch_runtime_owner_locked(
        self, *, agent_id: str, owner_id: str, now: datetime
    ) -> None:
        """Upsert an active owner while the caller holds a transaction.

        A delayed heartbeat can finish after a newer activation or heartbeat.
        Preserve the newest liveness evidence rather than regressing it and
        allowing another dispatcher to misclassify this owner as stale.
        """
        updated = await self._backend.execute(
            f"""
            UPDATE {self.RUNTIME_OWNERS}
            SET heartbeat_at = CASE
                    WHEN heartbeat_at >= ? THEN heartbeat_at ELSE ? END,
                stopped_at = NULL,
                updated_at = CASE
                    WHEN heartbeat_at >= ? THEN heartbeat_at ELSE ? END
            WHERE agent_id = ? AND owner_id = ?
            """,
            (
                self.to_timestamp_param(now),
                self.to_timestamp_param(now),
                self.to_timestamp_param(now),
                self.to_timestamp_param(now),
                agent_id,
                owner_id,
            ),
        )
        if updated == 0:
            await self._backend.execute(
                f"""
                INSERT OR IGNORE INTO {self.RUNTIME_OWNERS} (
                    agent_id, owner_id, heartbeat_at, stopped_at, created_at, updated_at
                ) VALUES (?, ?, ?, NULL, ?, ?)
                """,
                (
                    agent_id,
                    owner_id,
                    self.to_timestamp_param(now),
                    self.to_timestamp_param(now),
                    self.to_timestamp_param(now),
                ),
            )

    async def _recover_expired_leases(
        self,
        *,
        agent_id: str,
        consumer_id: str,
        now: datetime,
        runtime_owner_stale_before: datetime,
    ) -> None:
        """Requeue expired work without stealing from a live dispatcher.

        Leases owned by public or unknown executors retain ordinary expiry
        behavior because there is no durable liveness contract for them.  A
        ``dispatcher:`` owner is different: its heartbeat proves that an
        in-process cognition task may still be draining a cancellation.  Such
        a row is recoverable only once that managed owner is explicitly
        stopped or its heartbeat is stale.
        """

        # Claim/recovery calls this inside their existing transaction.  Take
        # the same tenant liveness scope as heartbeat before evaluating the
        # managed-owner predicate.
        await self._lock_runtime_owner_scope(agent_id=agent_id)
        timestamp = self._timestamp_placeholder()
        await self._backend.execute(
            f"""
            UPDATE {self.DELIVERIES}
            SET status = CASE WHEN max_attempts > 0 AND attempts >= max_attempts THEN ? ELSE ? END,
                lease_owner = NULL, lease_token = NULL, lease_expires_at = NULL,
                next_attempt_at = CASE WHEN max_attempts > 0 AND attempts >= max_attempts
                    THEN NULL ELSE {timestamp} END,
                last_error = 'lease expired before acknowledgement',
                terminal_at = CASE WHEN max_attempts > 0 AND attempts >= max_attempts
                    THEN {timestamp} ELSE NULL END,
                updated_at = ?
            WHERE agent_id = ? AND consumer_id = ? AND status = ?
              AND lease_expires_at <= ?
              AND (
                  lease_owner IS NULL
                  OR lease_owner NOT LIKE 'dispatcher:%'
                  OR NOT EXISTS (
                      SELECT 1 FROM {self.RUNTIME_OWNERS} owner
                      WHERE owner.agent_id = {self.DELIVERIES}.agent_id
                        AND owner.owner_id = {self.DELIVERIES}.lease_owner
                        AND owner.stopped_at IS NULL
                        AND owner.heartbeat_at >= ?
                  )
              )
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
                self.to_timestamp_param(runtime_owner_stale_before),
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
                   session_id, caller_identity, visibility, urgency, dedupe_key,
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
        initial_reservation_owner: Optional[str] = None,
        initial_reservation_token: Optional[str] = None,
    ) -> Optional[str]:
        initial_reservation = initial_reservation_owner is not None
        if initial_reservation != (initial_reservation_token is not None):
            raise ValueError(
                "initial reservation owner and token must be set together"
            )
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
                INITIAL_RESERVED if initial_reservation else PENDING,
                max_attempts,
                initial_reservation_owner,
                initial_reservation_token,
                None,
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
        caller_identity: Optional[str],
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
            caller_identity=caller_identity,
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
            caller_identity=row[9],
            visibility=row[10],
            urgency=row[11],
            dedupe_key=row[12],
            causation_chain=_json_load(row[13]),
            arrived_at=_as_utc(self.from_timestamp_field(row[14])),
            committed_at=_as_utc(self.from_timestamp_field(row[15])),
            retention_until=_as_utc(self.from_timestamp_field(row[16])),
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
                e.caller_identity, e.visibility, e.urgency, e.dedupe_key,
                e.causation_chain,
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
    "INITIAL_RESERVED",
    "LEASED",
    "PENDING",
    "RETRY",
    "TERMINAL_ACKABLE",
    "DurableConsumerRegistration",
    "DurableDelivery",
    "DurableEventPersistence",
    "DurableInitialDeliveryReservation",
    "DurableSignalEvent",
    "DurableSignalStore",
]
