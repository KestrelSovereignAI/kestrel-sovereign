"""Creator-owned native push configuration with encrypted callback credentials."""

from dataclasses import dataclass, field
from uuid import uuid4

from a2a.types import TaskPushNotificationConfig
from a2a.utils.proto_utils import validate_proto_required_fields
from kestrel_sdk.security.aead import AEADCipher
from kestrel_sdk.security.encryption import get_agent_key

from kestrel_sovereign.a2a.scope import NativeRecipientScope
from kestrel_sovereign.a2a.stores.base import json_dumps
from kestrel_sovereign.a2a.stores.unified.base import (
    NativeStoreConflict,
    UnifiedStoreBase,
)
from kestrel_sovereign.a2a.stores.unified.task_store import TaskStore


class PushConfigConflict(NativeStoreConflict):
    """A push configuration violates its immutable ownership or lifecycle."""


class PushTaskNotFound(PushConfigConflict):
    """The caller cannot observe the selected native task."""


@dataclass(frozen=True)
class PushConfiguration:
    """A copied official value and an immutable delivery generation."""

    creator_id: str
    generation: str
    _configuration_bytes: bytes = field(repr=False)

    @classmethod
    def from_config(
        cls, creator_id: str, generation: str, config: TaskPushNotificationConfig
    ):
        return cls(creator_id, generation, config.SerializeToString(deterministic=True))

    @property
    def config(self) -> TaskPushNotificationConfig:
        """Readers receive their own official protobuf, never a mutable cache."""
        return TaskPushNotificationConfig.FromString(self._configuration_bytes)


@dataclass(frozen=True)
class PushDeliveryAttempt:
    """A durable record of one admitted external request's physical lifetime."""

    task_id: str
    config_id: str
    creator_id: str
    generation: str
    attempt_id: str
    executor_id: str


class PushConfigStore(UnifiedStoreBase):
    """Borrow the task store's primary backend and recipient/tenant scope.

    Retirement has two durable phases. ``begin_delete`` first denies new
    delivery admission. The service must physically retire admitted delivery
    owners before calling ``finish_delete`` and acknowledging Delete. A retry
    retains the same generation, including after a process restart.
    """

    def __init__(self, task_store: TaskStore):
        if not isinstance(task_store, TaskStore):
            raise TypeError(
                "Native push configuration requires its canonical task store"
            )
        super().__init__(task_store.backend, owns_backend=False)
        self.task_store = task_store

    async def initialize(self) -> None:
        from kestrel_sovereign.a2a.native_schema import native_schema_initialization

        async with native_schema_initialization(self.backend):
            await self.backend.execute_script("""
                CREATE TABLE IF NOT EXISTS a2a_push_configs (
                    recipient_id TEXT NOT NULL,
                    tenant TEXT NOT NULL,
                    creator_id TEXT NOT NULL,
                    task_id TEXT NOT NULL REFERENCES a2a_tasks(id) ON DELETE CASCADE,
                    config_id TEXT NOT NULL,
                    generation TEXT NOT NULL,
                    state TEXT NOT NULL CHECK (state IN ('active', 'deleting', 'deleted')),
                    ciphertext TEXT,
                    PRIMARY KEY (recipient_id, tenant, task_id, config_id, generation),
                    CHECK ((state = 'deleted') = (ciphertext IS NULL))
                );
                CREATE INDEX IF NOT EXISTS a2a_push_active_tasks
                    ON a2a_push_configs (recipient_id, tenant, task_id)
                    WHERE state = 'active';
                CREATE UNIQUE INDEX IF NOT EXISTS a2a_push_current_config
                    ON a2a_push_configs (recipient_id, tenant, task_id, config_id)
                    WHERE state = 'active';
                CREATE INDEX IF NOT EXISTS a2a_push_retiring_configs
                    ON a2a_push_configs (recipient_id, tenant, task_id, config_id, generation)
                    WHERE state = 'deleting'
            """)

            await self.backend.execute_script("""
                CREATE TABLE IF NOT EXISTS a2a_push_delivery_attempts (
                    recipient_id TEXT NOT NULL,
                    tenant TEXT NOT NULL,
                    task_id TEXT NOT NULL,
                    config_id TEXT NOT NULL,
                    creator_id TEXT NOT NULL,
                    generation TEXT NOT NULL,
                    attempt_id TEXT PRIMARY KEY,
                    executor_id TEXT NOT NULL,
                    retired INTEGER NOT NULL DEFAULT 0 CHECK (retired IN (0, 1)),
                    FOREIGN KEY (recipient_id, tenant, task_id, config_id, generation)
                        REFERENCES a2a_push_configs(recipient_id, tenant, task_id, config_id, generation)
                );
                CREATE INDEX IF NOT EXISTS a2a_push_live_attempts
                    ON a2a_push_delivery_attempts (recipient_id, tenant, task_id, config_id, generation, retired)
            """)

            await self._initialize_task_cleanup_guard()

    async def _initialize_task_cleanup_guard(self) -> None:
        # A live configuration owns a durable consumer even before its first
        # HTTP attempt. Task cleanup must not orphan that consumer or erase its
        # credentials behind the public Delete retirement contract.
        if self.is_sqlite:
            await self.backend.execute_script("""
                CREATE TRIGGER IF NOT EXISTS a2a_push_task_cleanup_v1
                BEFORE DELETE ON a2a_tasks
                WHEN EXISTS (SELECT 1 FROM a2a_push_configs
                             WHERE task_id = OLD.id AND state != 'deleted')
                BEGIN
                    SELECT RAISE(ABORT, 'task has an active push configuration');
                END
            """)
        elif self.is_postgres:
            await self.backend.execute_script("""
                CREATE OR REPLACE FUNCTION a2a_push_task_cleanup_v1()
                RETURNS trigger LANGUAGE plpgsql AS $$
                BEGIN
                    IF EXISTS (SELECT 1 FROM a2a_push_configs
                               WHERE task_id = OLD.id AND state != 'deleted') THEN
                        RAISE EXCEPTION 'task has an active push configuration'
                            USING ERRCODE = '23503';
                    END IF;
                    RETURN OLD;
                END;
                $$;
                CREATE OR REPLACE TRIGGER a2a_push_task_cleanup_v1
                    BEFORE DELETE ON a2a_tasks
                    FOR EACH ROW EXECUTE FUNCTION a2a_push_task_cleanup_v1()
            """)

    @property
    def scope(self) -> NativeRecipientScope:
        return self.task_store.scope

    @property
    def _scope_parameters(self) -> tuple[str, str]:
        return self.scope.recipient_id, self.scope.tenant

    @staticmethod
    def _require_creator(creator_id: str) -> None:
        if not isinstance(creator_id, str) or not creator_id.strip():
            raise ValueError("Native push requires a verified creator identity")

    async def _require_task(self, task_id: str, creator_id: str) -> None:
        self._require_creator(creator_id)
        # The same authority used by native GetTask, never task metadata.
        task = await self.task_store.get_for_creator(
            task_id, creator_id, recipient_agent_id=self.scope.recipient_id
        )
        if task is None:
            raise PushTaskNotFound("Native task is not available to this creator")

    def _aad(
        self, config: TaskPushNotificationConfig, creator_id: str, generation: str
    ) -> bytes:
        return json_dumps(
            [
                "a2a.push-config.v1",
                *self._scope_parameters,
                creator_id,
                config.task_id,
                config.id,
                generation,
            ]
        ).encode("utf-8")

    def _cipher(self) -> AEADCipher:
        # Reuse the shared purpose-separated key hierarchy. Missing keys refuse
        # configuration writes and reads; there is no plaintext storage mode.
        return AEADCipher(get_agent_key(self.scope.recipient_id, "service-keys"))

    def require_encryption_ready(self) -> None:
        """Refuse service startup before any schema work when keys are absent."""
        self._cipher()

    def _decode(self, row) -> PushConfiguration:
        task_id, config_id, creator_id, generation, ciphertext = row
        locator = TaskPushNotificationConfig(
            tenant=self.scope.tenant, task_id=task_id, id=config_id
        )
        plaintext = self._cipher().decrypt(
            ciphertext, aad=self._aad(locator, creator_id, generation)
        )
        config = TaskPushNotificationConfig.FromString(plaintext)
        if (config.tenant, config.task_id, config.id) != (
            self.scope.tenant,
            task_id,
            config_id,
        ):
            raise PushConfigConflict(
                "Stored push configuration differs from its ownership"
            )
        return PushConfiguration.from_config(creator_id, generation, config)

    async def create(
        self, config: TaskPushNotificationConfig, *, creator_id: str
    ) -> PushConfiguration:
        """Set a logical configuration, preserving immutable delivery generations.

        An identical current value is idempotent. A changed value revokes that
        generation and inserts a new one; its original credentials and attempts
        remain until physical retirement. The caller's enclosing transaction
        also replaces the durable subscriber. The task lock orders all setters
        and deletion inventories, including across independent connections.
        """
        if not isinstance(config, TaskPushNotificationConfig):
            raise TypeError(
                "Native push requires an official TaskPushNotificationConfig"
            )
        self._require_creator(creator_id)
        validate_proto_required_fields(config)
        if config.tenant != self.scope.tenant:
            raise PushConfigConflict(
                "Push configuration tenant differs from its recipient"
            )
        value = TaskPushNotificationConfig.FromString(config.SerializeToString())
        value.id = value.id or value.task_id
        cipher = self._cipher()
        async with self.write_transaction():
            await self._require_task(value.task_id, creator_id)
            await self.task_store.changes.lock_task_boundary(
                value.task_id, creator_id=creator_id
            )
            current = await self.get(value.task_id, value.id, creator_id=creator_id)
            if current is not None:
                if current.config == value:
                    return current
                changed = await self.backend.execute(
                    "UPDATE a2a_push_configs SET state = 'deleting' "
                    "WHERE recipient_id = ? AND tenant = ? AND task_id = ? "
                    "AND config_id = ? AND creator_id = ? AND generation = ? AND state = 'active'",
                    (*self._scope_parameters, value.task_id, value.id, creator_id, current.generation),
                )
                if changed != 1:
                    raise PushConfigConflict("Push replacement lost its original generation")
            generation = uuid4().hex
            ciphertext = cipher.encrypt(
                value.SerializeToString(deterministic=True),
                aad=self._aad(value, creator_id, generation),
            ).decode("ascii")
            await self.backend.execute(
                "INSERT INTO a2a_push_configs "
                "(recipient_id, tenant, creator_id, task_id, config_id, generation, state, ciphertext) "
                "VALUES (?, ?, ?, ?, ?, ?, 'active', ?)",
                (
                    *self._scope_parameters,
                    creator_id,
                    value.task_id,
                    value.id,
                    generation,
                    ciphertext,
                ),
            )
        return PushConfiguration.from_config(creator_id, generation, value)

    async def get(
        self, task_id: str, config_id: str, *, creator_id: str
    ) -> PushConfiguration | None:
        await self._require_task(task_id, creator_id)
        row = await self.backend.fetch_one(
            "SELECT task_id, config_id, creator_id, generation, ciphertext "
            "FROM a2a_push_configs WHERE recipient_id = ? AND tenant = ? "
            "AND task_id = ? AND config_id = ? AND creator_id = ? AND state = 'active'",
            (*self._scope_parameters, task_id, config_id, creator_id),
        )
        return self._decode(row) if row else None

    async def list(
        self, task_id: str, *, creator_id: str, limit: int, after: str = ""
    ) -> list[PushConfiguration]:
        """Read an ID-ordered page within one immutable ownership scope."""
        if type(limit) is not int or limit <= 0:
            raise ValueError("Push configuration page size must be positive")
        await self._require_task(task_id, creator_id)
        rows = await self.backend.fetch_all(
            "SELECT task_id, config_id, creator_id, generation, ciphertext "
            "FROM a2a_push_configs WHERE recipient_id = ? AND tenant = ? "
            "AND task_id = ? AND creator_id = ? AND state = 'active' AND config_id > ? "
            "ORDER BY config_id LIMIT ?",
            (*self._scope_parameters, task_id, creator_id, after, limit),
        )
        return [self._decode(row) for row in rows]

    async def begin_delete(
        self, task_id: str, config_id: str, *, creator_id: str
    ) -> tuple[PushConfiguration, ...]:
        """Revoke and inventory every original generation under the task lock.

        A later setter may create a fresh generation. This deletion owns only
        its captured inventory and may never erase that later admission.
        """
        async with self.write_transaction():
            await self._require_task(task_id, creator_id)
            await self.task_store.changes.lock_task_boundary(task_id, creator_id=creator_id)
            await self.backend.execute(
                "UPDATE a2a_push_configs SET state = 'deleting' "
                "WHERE recipient_id = ? AND tenant = ? AND task_id = ? "
                "AND config_id = ? AND creator_id = ? AND state = 'active'",
                (*self._scope_parameters, task_id, config_id, creator_id),
            )
            rows = await self.backend.fetch_all(
                "SELECT task_id, config_id, creator_id, generation, ciphertext "
                "FROM a2a_push_configs WHERE recipient_id = ? AND tenant = ? "
                "AND task_id = ? AND config_id = ? AND creator_id = ? AND state = 'deleting' "
                "ORDER BY generation",
                (*self._scope_parameters, task_id, config_id, creator_id),
            )
        return tuple(self._decode(row) for row in rows)

    async def finish_delete(self, configuration: PushConfiguration) -> bool:
        """Forget credentials only after the service proves this generation retired."""
        value = configuration.config
        if value.tenant != self.scope.tenant:
            raise PushConfigConflict("Push retirement belongs to another tenant")
        async with self.write_transaction():
            changed = (
                await self.backend.execute(
                    "UPDATE a2a_push_configs SET state = 'deleted', ciphertext = NULL "
                    "WHERE recipient_id = ? AND tenant = ? AND task_id = ? AND config_id = ? "
                    "AND creator_id = ? AND generation = ? AND state = 'deleting' "
                    "AND NOT EXISTS (SELECT 1 FROM a2a_push_delivery_attempts AS attempt "
                    "WHERE attempt.recipient_id = a2a_push_configs.recipient_id "
                    "AND attempt.tenant = a2a_push_configs.tenant "
                    "AND attempt.task_id = a2a_push_configs.task_id "
                    "AND attempt.config_id = a2a_push_configs.config_id "
                    "AND attempt.generation = a2a_push_configs.generation AND attempt.retired = 0)",
                    (
                        *self._scope_parameters,
                        value.task_id,
                        value.id,
                        configuration.creator_id,
                        configuration.generation,
                    ),
                )
                == 1
            )
            if changed:
                # The tombstone retains the deleted generation. Completed
                # request receipts no longer need foreign-key custody, so
                # retire their rows atomically with credential destruction.
                await self.backend.execute(
                    "DELETE FROM a2a_push_delivery_attempts WHERE recipient_id = ? "
                    "AND tenant = ? AND task_id = ? AND config_id = ? "
                    "AND creator_id = ? AND generation = ? AND retired = 1",
                    (
                        *self._scope_parameters,
                        value.task_id,
                        value.id,
                        configuration.creator_id,
                        configuration.generation,
                    ),
                )
            elif await self.backend.fetch_one(
                "SELECT state FROM a2a_push_configs WHERE recipient_id = ? AND tenant = ? "
                "AND task_id = ? AND config_id = ? AND creator_id = ? AND generation = ?",
                (*self._scope_parameters, value.task_id, value.id,
                 configuration.creator_id, configuration.generation),
            ) == ("deleted",):
                return True
        return changed

    async def retire_ready(self, *, limit: int) -> int:
        """Forget a finite batch of revoked generations whose HTTP owners retired.

        Replacement and Delete share the same physical retirement predicate.
        Neither a lease clock nor the presence of a newer configuration proves
        an older request retired.
        """
        if type(limit) is not int or limit <= 0:
            raise ValueError("Push retirement batch size must be positive")
        async with self.write_transaction():
            rows = await self.backend.fetch_all(
                "SELECT task_id, config_id, creator_id, generation, ciphertext "
                "FROM a2a_push_configs config WHERE recipient_id = ? AND tenant = ? "
                "AND state = 'deleting' AND NOT EXISTS ("
                "SELECT 1 FROM a2a_push_delivery_attempts attempt "
                "WHERE attempt.recipient_id = config.recipient_id AND attempt.tenant = config.tenant "
                "AND attempt.task_id = config.task_id AND attempt.config_id = config.config_id "
                "AND attempt.generation = config.generation AND attempt.retired = 0) "
                "ORDER BY task_id, config_id, generation LIMIT ?",
                (*self._scope_parameters, limit),
            )
            retired = 0
            for row in rows:
                retired += await self.finish_delete(self._decode(row))
        return retired

    def prepare_delivery(
        self, configuration: PushConfiguration, *, executor_id: str
    ) -> PushDeliveryAttempt:
        """Inventory the immutable attempt before an admission COMMIT can be lost."""
        self._require_creator(configuration.creator_id)
        if not isinstance(executor_id, str) or not executor_id.strip():
            raise ValueError("Push delivery requires its runtime executor identity")
        value = configuration.config
        if value.tenant != self.scope.tenant:
            raise PushConfigConflict("Push delivery belongs to another tenant")
        return PushDeliveryAttempt(
            value.task_id,
            value.id,
            configuration.creator_id,
            configuration.generation,
            uuid4().hex,
            executor_id,
        )

    async def admit_delivery(self, attempt: PushDeliveryAttempt) -> bool:
        """Admit an inventoried external request under the active generation lock.

        The attempt is not a time-based lease. A clock cannot prove physical
        retirement. The caller retains it through admission failure, so a lost
        COMMIT acknowledgement cannot strand a record before HTTP even starts.
        """
        if not isinstance(attempt, PushDeliveryAttempt):
            raise TypeError("Push admission requires its inventoried delivery attempt")
        async with self.write_transaction():
            # Serialize with begin_delete on the same configuration row. An
            # INSERT/SELECT alone could observe an older PostgreSQL snapshot
            # while deletion commits on another connection.
            suffix = " FOR UPDATE" if self.is_postgres else ""
            row = await self.backend.fetch_one(
                "SELECT state FROM a2a_push_configs WHERE recipient_id = ? AND tenant = ? "
                "AND task_id = ? AND config_id = ? AND creator_id = ? AND generation = ?"
                + suffix,
                (
                    *self._scope_parameters,
                    attempt.task_id,
                    attempt.config_id,
                    attempt.creator_id,
                    attempt.generation,
                ),
            )
            if row != ("active",):
                return False
            await self.backend.execute(
                "INSERT INTO a2a_push_delivery_attempts "
                "(recipient_id, tenant, task_id, config_id, creator_id, generation, attempt_id, executor_id) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    *self._scope_parameters,
                    attempt.task_id,
                    attempt.config_id,
                    attempt.creator_id,
                    attempt.generation,
                    attempt.attempt_id,
                    attempt.executor_id,
                ),
            )
        return True

    async def delivery_is_current(self, attempt: PushDeliveryAttempt) -> bool:
        """A worker's revocation monitor reads its exact durable admission."""
        row = await self.backend.fetch_one(
            "SELECT 1 FROM a2a_push_configs AS config "
            "JOIN a2a_push_delivery_attempts AS attempt "
            "ON attempt.recipient_id = config.recipient_id AND attempt.tenant = config.tenant "
            "AND attempt.task_id = config.task_id AND attempt.config_id = config.config_id "
            "AND attempt.creator_id = config.creator_id AND attempt.generation = config.generation "
            "WHERE config.recipient_id = ? AND config.tenant = ? AND config.state = 'active' "
            "AND attempt.task_id = ? AND attempt.config_id = ? AND attempt.creator_id = ? "
            "AND attempt.generation = ? AND attempt.attempt_id = ? AND attempt.executor_id = ? "
            "AND attempt.retired = 0",
            (
                *self._scope_parameters,
                attempt.task_id,
                attempt.config_id,
                attempt.creator_id,
                attempt.generation,
                attempt.attempt_id,
                attempt.executor_id,
            ),
        )
        return row is not None

    async def retire_delivery(self, attempt: PushDeliveryAttempt) -> bool:
        """Record the original owner's completed HTTP/response/client retirement.

        Call only after resource retirement, including on failed HTTP delivery.
        The attempt remains until configuration deletion; retries are idempotent.
        """
        return (
            await self.backend.execute(
                "UPDATE a2a_push_delivery_attempts SET retired = 1 "
                "WHERE recipient_id = ? AND tenant = ? AND task_id = ? AND config_id = ? "
                "AND creator_id = ? AND generation = ? AND attempt_id = ? AND executor_id = ?",
                (
                    *self._scope_parameters,
                    attempt.task_id,
                    attempt.config_id,
                    attempt.creator_id,
                    attempt.generation,
                    attempt.attempt_id,
                    attempt.executor_id,
                ),
            )
            == 1
        )
