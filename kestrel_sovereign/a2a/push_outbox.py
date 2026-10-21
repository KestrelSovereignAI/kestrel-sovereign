"""Native push subscriptions over the canonical durable signal consumer ledger."""

import hashlib
from dataclasses import dataclass

from a2a.types import TaskPushNotificationConfig
from kestrel_sdk.signals import Signal, SignalMode

from kestrel_sovereign.a2a.push_transport import validate_push_configuration
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
        if not await self.backend.table_exists(self.signals.EVENTS):
            raise RuntimeError(
                "Native push requires an initialized durable signal ledger"
            )
        await self.configurations.initialize()
        async with self.write_transaction():
            await self.backend.execute_script("""
                CREATE TABLE IF NOT EXISTS a2a_push_subscriptions (
                    recipient_id TEXT NOT NULL,
                    tenant TEXT NOT NULL,
                    task_id TEXT NOT NULL,
                    config_id TEXT NOT NULL,
                    generation TEXT NOT NULL,
                    creator_id TEXT NOT NULL,
                    consumer_id TEXT NOT NULL,
                    after_sequence BIGINT NOT NULL CHECK (after_sequence >= 0),
                    PRIMARY KEY (recipient_id, tenant, task_id, config_id),
                    FOREIGN KEY (recipient_id, tenant, task_id, config_id)
                        REFERENCES a2a_push_configs (recipient_id, tenant, task_id, config_id)
                        ON DELETE CASCADE
                )
            """)

    async def register(
        self, config: TaskPushNotificationConfig, *, creator_id: str
    ) -> PushConfiguration:
        """Commit the encrypted config, consumer and first notification together.

        The task row lock orders this boundary against all mutation triggers.
        Older journal entries may still be forwarded later; the immutable
        boundary lets their consumer ACK them without sending old notifications.
        Validation performs no network request and completes before row locks.
        """
        config = await validate_push_configuration(config)
        async with self.write_transaction():
            boundary = await self.changes.lock_task_boundary(
                config.task_id, creator_id=creator_id
            )
            configuration = await self.configurations.create(
                config, creator_id=creator_id
            )
            native = configuration.config
            consumer_id = "a2a.push:" + configuration.generation
            await self.signals.register_consumer(
                DurableConsumerRegistration(
                    agent_id=self.agent_id,
                    consumer_id=consumer_id,
                    source=self.SOURCE,
                    correlation_selector="payload.task_id=" + native.task_id,
                    max_attempts=0,
                    lease_seconds=self.lease_seconds,
                )
            )
            await self.backend.execute(
                "INSERT INTO a2a_push_subscriptions (recipient_id, tenant, task_id, config_id, "
                "generation, creator_id, consumer_id, after_sequence) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    self.configurations.scope.recipient_id,
                    self.configurations.scope.tenant,
                    native.task_id,
                    native.id,
                    configuration.generation,
                    creator_id,
                    consumer_id,
                    boundary,
                ),
            )
            await self.changes.enqueue_snapshot(native.task_id, creator_id=creator_id)
        return configuration

    async def subscriptions(self) -> list[PushSubscription]:
        """Inventory active contracts for this exact recipient and tenant."""
        rows = await self.backend.fetch_all(
            "SELECT s.task_id, s.config_id, s.generation, s.creator_id, s.consumer_id, s.after_sequence "
            "FROM a2a_push_subscriptions AS s JOIN a2a_push_configs AS c "
            "ON c.recipient_id = s.recipient_id AND c.tenant = s.tenant "
            "AND c.task_id = s.task_id AND c.config_id = s.config_id AND c.generation = s.generation "
            "WHERE s.recipient_id = ? AND s.tenant = ? AND c.state = 'active' "
            "ORDER BY s.task_id, s.config_id",
            (self.configurations.scope.recipient_id, self.configurations.scope.tenant),
        )
        return [PushSubscription(*row) for row in rows]

    async def revoke(
        self, task_id: str, config_id: str, *, creator_id: str
    ) -> PushConfiguration | None:
        """Revoke config admission and consumer claims in one primary transaction.

        This is not a deletion acknowledgement. The worker must join original
        HTTP owners and obtain ``finish_delete`` before returning success.
        """
        async with self.write_transaction():
            config = await self.configurations.begin_delete(
                task_id, config_id, creator_id=creator_id
            )
            if config is not None:
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
        return config

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
