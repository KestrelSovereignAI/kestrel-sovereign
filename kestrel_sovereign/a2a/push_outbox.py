"""Native push subscriptions over the canonical durable signal consumer ledger."""

import hashlib
from dataclasses import dataclass, field

from a2a.types import TaskPushNotificationConfig
from kestrel_sdk.signals import Signal, SignalMode

from kestrel_sovereign.a2a.push_transport import validate_push_configuration
from kestrel_sovereign.a2a.scope import NativeRecipientScope
from kestrel_sovereign.a2a.stores.base import json_dumps
from kestrel_sovereign.a2a.stores.unified.base import UnifiedStoreBase
from kestrel_sovereign.a2a.stores.unified.push_config_store import (
    PushConfigConflict,
    PushConfigStore,
    PushConfiguration,
)
from kestrel_sovereign.signals.durable import (
    DurableConsumerRegistration,
    DurableSignalStore,
)


@dataclass(frozen=True)
class PushSubscription:
    """Immutable delivery contract; credentials stay in encrypted configuration."""

    task_id: str
    config_id: str
    generation: str
    creator_id: str
    consumer_id: str
    after_sequence: int


@dataclass(frozen=True)
class PreparedPushConfig:
    """A validated, immutable secret projection for atomic task admission.

    This is an internal storage handoff, never a wire type. Delivery validates
    egress again at its actual network boundary. Preparation must happen before
    acquiring task or consumer row locks, including for inline configurations.
    """

    scope: NativeRecipientScope
    _configuration_bytes: bytes = field(repr=False)

    @property
    def config(self) -> TaskPushNotificationConfig:
        return TaskPushNotificationConfig.FromString(self._configuration_bytes)


class NativePushOutbox(UnifiedStoreBase):
    """Forward references atomically; reuse consumer registration, claims and ACKs.

    Only server-owned IDs enter this source. User message content, callback
    addresses and credentials are never signal payloads. Network delivery
    reads the latest canonical Task under this subscription's creator authority.
    The task journal and the signal ledger must share the exact primary backend.
    """

    SOURCE = "a2a.task_changed"

    def __init__(
        self,
        configurations: PushConfigStore,
        signals: DurableSignalStore,
        *,
        retention_days: int,
        lease_seconds: int,
    ):
        if not isinstance(configurations, PushConfigStore) or not isinstance(
            signals, DurableSignalStore
        ):
            raise TypeError(
                "Native push requires its canonical configuration and signal stores"
            )
        if configurations.backend is not signals.backend:
            raise ValueError("Native push requires the same primary backend")
        if type(retention_days) is not int or retention_days <= 0:
            raise ValueError("Push retention must be a positive number of days")
        if type(lease_seconds) is not int or lease_seconds <= 0:
            raise ValueError("Push delivery lease must be a positive number of seconds")
        super().__init__(configurations.backend, owns_backend=False)
        self.configurations = configurations
        self.signals = signals
        self.changes = configurations.task_store.changes
        self.retention_days = retention_days
        self.lease_seconds = lease_seconds
        self.agent_id = (
            "a2a.push:"
            + hashlib.sha256(
                json_dumps(
                    [configurations.scope.recipient_id, configurations.scope.tenant]
                ).encode()
            ).hexdigest()
        )

    async def initialize(self) -> None:
        """Initialize only after the host's canonical signal ledger is ready."""
        from kestrel_sovereign.a2a.native_schema import (
            NativeStoreSchema,
            native_schema_initialization,
            native_store_schema_ready,
            publish_native_store_schema,
        )

        self.configurations.require_encryption_ready()
        if not await self.backend.table_exists(self.signals.EVENTS):
            raise RuntimeError(
                "Native push requires an initialized durable signal ledger"
            )
        async with native_schema_initialization(self.backend):
            if await native_store_schema_ready(self.backend, NativeStoreSchema.PUSH):
                return
            await self.configurations.initialize()
            await self.backend.execute_script("""
                CREATE TABLE IF NOT EXISTS a2a_push_subscriptions (
                    recipient_id TEXT NOT NULL,
                    tenant TEXT NOT NULL,
                    task_id TEXT NOT NULL,
                    config_id TEXT NOT NULL,
                    generation TEXT NOT NULL,
                    creator_id TEXT NOT NULL,
                    agent_id TEXT NOT NULL,
                    consumer_id TEXT NOT NULL,
                    after_sequence BIGINT NOT NULL CHECK (after_sequence >= 0),
                    PRIMARY KEY (recipient_id, tenant, task_id, config_id, generation),
                    UNIQUE (agent_id, consumer_id),
                    FOREIGN KEY (agent_id, consumer_id)
                        REFERENCES durable_signal_consumers (agent_id, consumer_id),
                    FOREIGN KEY (recipient_id, tenant, task_id, config_id, generation)
                        REFERENCES a2a_push_configs (recipient_id, tenant, task_id, config_id, generation)
                        ON DELETE CASCADE
                )
            """)
            await self.changes.enable_push_capture()
            await publish_native_store_schema(self.backend, NativeStoreSchema.PUSH)

    async def register(
        self, config: TaskPushNotificationConfig, *, creator_id: str
    ) -> PushConfiguration:
        """Prepare before locking, then use the one atomic registration path."""
        return await self.register_prepared(
            await self.prepare(config), creator_id=creator_id
        )

    async def prepare(self, config: TaskPushNotificationConfig) -> PreparedPushConfig:
        """Validate an independent callback copy before the admission transaction."""
        if not isinstance(config, TaskPushNotificationConfig):
            raise TypeError(
                "Native push requires an official TaskPushNotificationConfig"
            )
        config = TaskPushNotificationConfig.FromString(config.SerializeToString())
        if config.tenant != self.configurations.scope.tenant:
            raise PushConfigConflict("Push preparation belongs to another tenant")
        config = await validate_push_configuration(config)
        return PreparedPushConfig(
            self.configurations.scope, config.SerializeToString(deterministic=True)
        )

    async def register_prepared(
        self,
        prepared: PreparedPushConfig,
        *,
        creator_id: str,
        task_id: str | None = None,
    ) -> PushConfiguration:
        """Commit encrypted configuration, consumer and initial reference together.

        The task row lock orders this boundary against all mutation triggers.
        Older journal entries may still be forwarded later; the immutable
        boundary lets their consumer ACK them without sending old notifications.
        An inline config leaves its task ID empty; only the server's successful
        task admission supplies that ID here. No DNS or network work occurs.
        """
        if not isinstance(prepared, PreparedPushConfig):
            raise TypeError(
                "Atomic push registration requires a prepared configuration"
            )
        if prepared.scope != self.configurations.scope:
            raise PushConfigConflict(
                "Prepared push configuration belongs to another recipient scope"
            )
        config = prepared.config
        if task_id is not None:
            if not isinstance(task_id, str) or not task_id.strip():
                raise ValueError(
                    "Inline push registration requires its admitted task ID"
                )
            if config.task_id and config.task_id != task_id:
                raise PushConfigConflict(
                    "Prepared push configuration belongs to another task"
                )
            config.task_id = task_id
        if not config.task_id:
            raise PushConfigConflict("Push registration requires its admitted task ID")
        async with self.write_transaction():
            boundary = await self.changes.lock_task_boundary(
                config.task_id, creator_id=creator_id
            )
            configuration = await self.configurations.create(
                config, creator_id=creator_id
            )
            native = configuration.config
            consumer_id = "a2a.push:" + configuration.generation
            existing = await self.backend.fetch_one(
                "SELECT creator_id, agent_id, consumer_id FROM a2a_push_subscriptions "
                "WHERE recipient_id = ? AND tenant = ? AND task_id = ? AND config_id = ? "
                "AND generation = ?",
                (*self.configurations._scope_parameters, native.task_id, native.id,
                 configuration.generation),
            )
            if existing is not None:
                if existing != (creator_id, self.agent_id, consumer_id):
                    raise PushConfigConflict("Repeated push configuration lost its original subscription")
                return configuration
            previous = await self.backend.fetch_all(
                "SELECT s.consumer_id FROM a2a_push_subscriptions s "
                "JOIN a2a_push_configs c ON c.recipient_id = s.recipient_id "
                "AND c.tenant = s.tenant AND c.task_id = s.task_id AND c.config_id = s.config_id "
                "AND c.generation = s.generation "
                "WHERE s.recipient_id = ? AND s.tenant = ? AND s.task_id = ? "
                "AND s.config_id = ? AND s.creator_id = ? AND s.agent_id = ? "
                "AND c.state = 'deleting'",
                (*self.configurations._scope_parameters, native.task_id, native.id,
                 creator_id, self.agent_id),
            )
            for row in previous:
                if not await self.signals.deactivate_consumer(
                    agent_id=self.agent_id, consumer_id=row[0]
                ):
                    raise PushConfigConflict("Push replacement lost an original consumer")
            await self.signals.register_consumer(
                DurableConsumerRegistration(
                    agent_id=self.agent_id,
                    consumer_id=consumer_id,
                    source=self.SOURCE,
                    correlation_selector="payload.task_id=" + native.task_id,
                    max_attempts=0,
                    lease_seconds=self.lease_seconds,
                    replay_retained=False,
                )
            )
            await self.backend.execute(
                "INSERT INTO a2a_push_subscriptions (recipient_id, tenant, task_id, config_id, "
                "generation, creator_id, agent_id, consumer_id, after_sequence) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    self.configurations.scope.recipient_id,
                    self.configurations.scope.tenant,
                    native.task_id,
                    native.id,
                    configuration.generation,
                    creator_id,
                    self.agent_id,
                    consumer_id,
                    boundary,
                ),
            )
            await self.changes.enqueue_snapshot(native.task_id, creator_id=creator_id)
        return configuration

    async def subscriptions(self, *, due_limit: int | None = None) -> list[PushSubscription]:
        """Inventory live contracts, optionally restricted to a finite due batch."""
        due = None
        parameters = [
            self.configurations.scope.recipient_id,
            self.configurations.scope.tenant,
            self.signals.to_bool_param(True),
        ]
        predicate = ""
        if due_limit is not None:
            due = await self.signals.due_consumer_ids(
                agent_id=self.agent_id, source=self.SOURCE, limit=due_limit
            )
            if not due:
                return []
            predicate = " AND s.consumer_id IN (" + ", ".join("?" for _ in due) + ")"
            parameters.extend(due)
        rows = await self.backend.fetch_all(
            "SELECT s.task_id, s.config_id, s.generation, s.creator_id, s.consumer_id, s.after_sequence "
            "FROM a2a_push_subscriptions AS s JOIN a2a_push_configs AS c "
            "ON c.recipient_id = s.recipient_id AND c.tenant = s.tenant "
            "AND c.task_id = s.task_id AND c.config_id = s.config_id AND c.generation = s.generation "
            "JOIN durable_signal_consumers consumer "
            "ON consumer.agent_id = s.agent_id AND consumer.consumer_id = s.consumer_id "
            "WHERE s.recipient_id = ? AND s.tenant = ? AND consumer.active = ? "
            "AND c.state = 'active' " + predicate + " "
            "ORDER BY s.task_id, s.config_id",
            tuple(parameters),
        )
        if due is not None:
            found = {row[4] for row in rows}
            for consumer_id in set(due) - found:
                if await self.signals.has_active_consumer(self.agent_id, consumer_id):
                    raise PushConfigConflict("Due push consumer lost its original subscription")
        return [PushSubscription(*row) for row in rows]

    async def retire_history(self, *, limit: int) -> int:
        """Retire one exact source batch and its journal references atomically."""
        async with self.write_transaction():
            deleted = await self.signals.purge_expired(
                agent_id=self.agent_id, source=self.SOURCE, limit=limit
            )
            await self.changes.retire_forwarded(deleted)
            await self.configurations.retire_ready(limit=limit)
        return len(deleted)

    async def revoke(
        self, task_id: str, config_id: str, *, creator_id: str
    ) -> tuple[PushConfiguration, ...]:
        """Revoke config admission and consumer claims in one primary transaction.

        This is not a deletion acknowledgement. The worker must join original
        HTTP owners and obtain ``finish_delete`` before returning success.
        """
        async with self.write_transaction():
            configurations = await self.configurations.begin_delete(
                task_id, config_id, creator_id=creator_id
            )
            for config in configurations:
                row = await self.backend.fetch_one(
                    "SELECT consumer_id FROM a2a_push_subscriptions WHERE recipient_id = ? "
                    "AND tenant = ? AND task_id = ? AND config_id = ? AND creator_id = ? "
                    "AND generation = ?",
                    (
                        self.configurations.scope.recipient_id,
                        self.configurations.scope.tenant,
                        task_id,
                        config_id,
                        creator_id,
                        config.generation,
                    ),
                )
                if row is None:
                    raise PushConfigConflict(
                        "Push configuration has no original durable consumer"
                    )
                if not await self.signals.deactivate_consumer(
                    agent_id=self.agent_id, consumer_id=row[0]
                ):
                    raise PushConfigConflict(
                        "Push configuration lost its durable consumer"
                    )
        return configurations

    async def forward_pending(self, *, limit: int) -> int:
        """Atomically persist metadata-only events, fan-out and journal receipts."""
        async with self.write_transaction():
            pending = await self.changes.pending(limit=limit)
            for change in pending:
                persistence = await self.signals.persist_signal(
                    Signal(
                        source=self.SOURCE,
                        kind="task_changed",
                        mode=SignalMode.ACTION,
                        payload={
                            "task_id": change.task_id,
                            "journal_sequence": change.sequence,
                        },
                        target_agent=self.configurations.scope.recipient_id,
                    ),
                    agent_id=self.agent_id,
                    source_event_id="a2a.task_changed:" + str(change.sequence),
                    retention_days=self.retention_days,
                )
                await self.changes.mark_forwarded(change, persistence.event_id)
        return len(pending)
