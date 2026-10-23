"""
Unified TaskStore - Backend-Agnostic Task Persistence.

Manages task lifecycle from submission to completion/failure.
Works with both SQLite and PostgreSQL backends.
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from kestrel_sovereign._vendor.a2a.helpers.proto_helpers import new_text_message
from kestrel_sovereign._vendor.a2a.server.tasks.task_manager import append_artifact_to_task
from kestrel_sovereign._vendor.a2a.types import (
    Artifact,
    ListTasksRequest,
    ListTasksResponse,
    Role,
    SendMessageRequest,
    Task,
    TaskArtifactUpdateEvent,
    TaskState,
    TaskStatus,
)
from kestrel_sovereign._vendor.a2a.utils.constants import DEFAULT_LIST_TASKS_PAGE_SIZE
from kestrel_sovereign._vendor.a2a.utils.errors import InvalidAgentResponseError, InvalidParamsError
from kestrel_sovereign._vendor.a2a.utils.proto_utils import validate_proto_required_fields
from kestrel_sovereign._vendor.a2a.utils.task import (
    ListTasksCursor,
    apply_history_length,
    decode_list_tasks_cursor,
    encode_list_tasks_cursor,
    validate_history_length,
    validate_page_size,
)
from google.protobuf.json_format import MessageToDict, MessageToJson, ParseDict
from google.protobuf.timestamp_pb2 import Timestamp

from kestrel_sovereign.a2a.lifecycle import TERMINAL_STATES
from kestrel_sovereign.a2a.lifecycle import task_state_from_label as _native_state
from kestrel_sovereign.a2a.lifecycle import task_state_label as _stored_state
from kestrel_sovereign.a2a.native_input import (
    NativeTaskInput,
    prepare_native_task_input,
)
from kestrel_sovereign.a2a.outcomes import TaskOutcomeConflictError, merge_task_outcome
from kestrel_sovereign.a2a.scope import NativeRecipientScope
from kestrel_sovereign.a2a.stores.base import json_dumps, json_loads
from kestrel_sovereign.a2a.stores.unified.base import (
    NativeStoreConflict,
    UnifiedStoreBase,
)
from kestrel_sovereign.a2a.stores.unified.context_store import (
    ContextOwnership,
    ContextOwnershipError,
    ContextStore,
)
from kestrel_sovereign.a2a.stores.unified.task_change_store import TaskChangeStore
from kestrel_sovereign.storage.db.interface import DatabaseBackend, QueryError

logger = logging.getLogger(__name__)

# Keep the v2 lock identity through the mixed-version window. A v2 worker uses
# this exact advisory lock before probing/installing its compatibility trigger;
# changing the key would let v2 and v3 replace database functions concurrently.
_CANCELLATION_SCHEMA_LOCK = "a2a_tasks_cancel_authority_v2"

# Explicit projection keeps row decoding independent of migration column order.
_TASK_COLUMNS = (
    "id, session_id, user_id, task_type, status, message, artifacts, history, "
    "metadata, created_at, updated_at, creator_agent_id, recipient_agent_id, "
    "canceled_by, cancel_reason, cancel_previous_status, cancel_operation_id, "
    "terminal_operation_id, lifecycle_revision, lifecycle_updated_at, "
    "status_timestamp_seconds, status_timestamp_nanos, tenant, input_revision, cognition_wake_eligible, input_wake_eligible"
)


_TASK_SCOPE = "recipient_agent_id = ? AND tenant = ?"
_SNAPSHOT_COLUMNS = (
    _TASK_COLUMNS + ", (SELECT request_json FROM a2a_task_inputs AS native_input "
    "WHERE native_input.task_id = a2a_tasks.id AND native_input.input_revision = a2a_tasks.input_revision), "
    "(SELECT request_digest FROM a2a_task_inputs AS native_input WHERE native_input.task_id = a2a_tasks.id "
    "AND native_input.input_revision = a2a_tasks.input_revision), "
    "(SELECT wire_request_digest FROM a2a_task_inputs AS native_input WHERE native_input.task_id = a2a_tasks.id "
    "AND native_input.input_revision = a2a_tasks.input_revision)"
)


class TaskInputConflictError(NativeStoreConflict):
    """An input would alter an existing request or cross a context boundary."""


class TaskInputUnsupportedError(TaskInputConflictError):
    """This task cannot execute subsequent incoming cognition."""


class TaskInputTerminalError(TaskInputUnsupportedError):
    """A terminal task cannot accept a subsequent message."""


class TaskAlreadyExistsError(ValueError):
    """A caller attempted to create a second task under an occupied ID."""


class TaskMutationAuthorizationError(PermissionError):
    """A responder mutation did not match the durable task recipient."""


@dataclass(frozen=True)
class TaskCancellationSnapshot:
    """Minimal durable state needed by live cancellation monitors."""

    state: str
    actor_agent_id: str | None
    input_revision: int = 0
    cognition_wake_eligible: bool = True
    input_wake_eligible: bool = True


@dataclass(frozen=True)
class NativeTaskSnapshot:
    """Native task snapshot and server-owned admission facts for its wake.

    Metadata is authored by the caller. It cannot establish the creator,
    recipient, context ownership, or the trust of a cognition wake.
    """

    task: Task
    lifecycle_revision: int
    ownership: ContextOwnership
    input_revision: int = 0
    cognition_wake_eligible: bool = True
    input_wake_eligible: bool = True
    input_request_json: str | None = None
    input_request_digest: str | None = None
    input_wire_request_digest: str | None = None

    @property
    def input_request(self) -> SendMessageRequest | None:
        """Rebuild the immutable admitted request, verifying its stored digest."""
        if self.input_request_json is None:
            if (
                self.input_request_digest is not None
                or self.input_wire_request_digest is not None
            ):
                raise TaskInputConflictError(
                    "Native input has a digest without its request"
                )
            return None
        request = ParseDict(json_loads(self.input_request_json), SendMessageRequest())
        if request.configuration.HasField("task_push_notification_config"):
            raise TaskInputConflictError(
                "Callback credentials are not executable task input"
            )
        wire_digest = self.input_wire_request_digest
        if (
            not isinstance(wire_digest, str)
            or len(wire_digest) != 64
            or any(char not in "0123456789abcdef" for char in wire_digest)
        ):
            raise TaskInputConflictError(
                "Native input has no valid original wire request digest"
            )
        digest = hashlib.sha256(
            request.SerializeToString(deterministic=True)
        ).hexdigest()
        if digest != self.input_request_digest:
            raise TaskInputConflictError(
                "Stored native input differs from its admitted request digest"
            )
        return request

    def require_recipient(self, recipient_agent_id: str) -> None:
        """Reject a misrouted snapshot before it becomes a trusted signal."""
        if not isinstance(self.task, Task) or not isinstance(
            self.ownership, ContextOwnership
        ):
            raise TypeError("A native task wake requires a task and durable ownership")
        scope = NativeRecipientScope(
            self.ownership.recipient_agent_id, self.ownership.tenant
        )
        if scope.recipient_id != recipient_agent_id:
            raise TaskMutationAuthorizationError(
                "Task wake does not belong to this recipient"
            )
        if (
            not isinstance(self.ownership.creator_agent_id, str)
            or not self.ownership.creator_agent_id.strip()
        ):
            raise TaskMutationAuthorizationError(
                "Task wake has no authoritative creator"
            )
        if self.task.context_id != self.ownership.context_id:
            raise TaskMutationAuthorizationError(
                "Task wake context does not match its durable ownership"
            )
        if (
            isinstance(self.lifecycle_revision, bool)
            or not isinstance(self.lifecycle_revision, int)
            or self.lifecycle_revision < 0
        ):
            raise ValueError("Task wake requires a non-negative lifecycle revision")
        if (
            isinstance(self.input_revision, bool)
            or not isinstance(self.input_revision, int)
            or self.input_revision < 0
        ):
            raise ValueError("Task wake requires a non-negative input revision")
        if not isinstance(self.cognition_wake_eligible, bool) or not isinstance(
            self.input_wake_eligible, bool
        ):
            raise TypeError(
                "Task wake eligibility requires server-owned boolean policy"
            )


def require_native_task_snapshot(
    candidate: object, recipient_agent_id: str
) -> NativeTaskSnapshot:
    """Check the one native provenance contract before constructing a signal."""
    if not isinstance(candidate, NativeTaskSnapshot):
        raise TypeError("Native cognition wakes require durable task provenance")
    candidate.require_recipient(recipient_agent_id)
    return candidate


def without_reserved_cancellation_receipt(
    metadata: Mapping[str, object] | None,
) -> dict:
    """Copy caller metadata without authority minted by cancellation."""

    sanitized = dict(metadata or {})
    sanitized.pop("cancellation_receipt", None)
    return sanitized


def _task_payload(task: Task) -> tuple[str | None, str, str, str | None]:
    """Serialize one native payload; preserve optional message/map presence."""
    validate_proto_required_fields(task)
    message = (
        MessageToJson(task.status.message) if task.status.HasField("message") else None
    )
    artifacts = json_dumps([MessageToDict(item) for item in task.artifacts])
    history = json_dumps([MessageToDict(item) for item in task.history])
    metadata = (
        json_dumps(without_reserved_cancellation_receipt(MessageToDict(task.metadata)))
        if task.HasField("metadata")
        else None
    )
    return message, artifacts, history, metadata


def _status_timestamp(status: TaskStatus) -> tuple[int | None, int | None]:
    if not status.HasField("timestamp"):
        return None, None
    # Validate the full protobuf range and nanoseconds before writing SQL.
    status.timestamp.ToJsonString()
    return status.timestamp.seconds, status.timestamp.nanos


class TaskStore(UnifiedStoreBase):
    """
    Backend-agnostic task store.

    Replaces both SQLiteTaskStore and PostgresTaskStore with a single
    implementation that works with any DatabaseBackend.
    """

    def __init__(
        self,
        backend: DatabaseBackend,
        *,
        scope: NativeRecipientScope,
        owns_backend: bool = True,
    ):
        """
        Initialize task store with database backend.

        Args:
            backend: DatabaseBackend instance (SQLite or PostgreSQL)
        """
        if not isinstance(scope, NativeRecipientScope):
            raise TypeError("TaskStore requires a server-owned native recipient scope")
        super().__init__(backend, owns_backend=owns_backend)
        self._scope = scope
        self.changes = TaskChangeStore(backend, scope=scope)

    @property
    def scope(self) -> NativeRecipientScope:
        """Immutable host registration, never selected by task metadata."""
        return self._scope

    @property
    def _scope_parameters(self) -> tuple[str, str]:
        return self.scope.recipient_id, self.scope.tenant

    async def initialize(self) -> None:
        """Create native tables without interpreting or settling old rows."""
        from kestrel_sovereign.a2a.native_schema import (
            native_schema_initialization,
            publish_native_a2a_schema,
        )

        async with native_schema_initialization(self.backend):
            await self._initialize_native_objects()
            await publish_native_a2a_schema(self.backend)
        logger.info("TaskStore initialized (%s)", self.backend.backend_type)

    async def _initialize_native_objects(self) -> None:
        """Canonical DDL, also used inside the exclusive offline cutover."""
        ts_type = self.timestamp_type()
        ts_default = self.now_default()
        json_type = self.json_type()

        # Create table with backend-appropriate types
        await self._backend.execute_script(f"""
            CREATE TABLE IF NOT EXISTS a2a_tasks (
                id TEXT PRIMARY KEY,
                session_id TEXT,
                user_id TEXT,
                task_type TEXT NOT NULL,
                status TEXT DEFAULT 'submitted',
                message {json_type},
                artifacts {json_type} DEFAULT '[]',
                history {json_type} DEFAULT '[]',
                metadata {json_type} DEFAULT '{{}}',
                created_at {ts_type} {ts_default},
                updated_at {ts_type} {ts_default},
                creator_agent_id TEXT,
                recipient_agent_id TEXT,
                canceled_by TEXT,
                cancel_reason TEXT,
                cancel_previous_status TEXT,
                cancel_operation_id TEXT,
                terminal_operation_id TEXT,
                lifecycle_revision BIGINT NOT NULL DEFAULT 0,
                lifecycle_updated_at {ts_type},
                status_timestamp_seconds BIGINT,
                status_timestamp_nanos INTEGER,
                tenant TEXT NOT NULL DEFAULT '',
                cognition_wake_eligible INTEGER NOT NULL DEFAULT 1 CHECK (cognition_wake_eligible IN (0,1)),
                input_revision BIGINT NOT NULL DEFAULT 0 CHECK (input_revision >= 0),
                input_wake_eligible INTEGER NOT NULL DEFAULT 1 CHECK (input_wake_eligible IN (0,1))
            )
        """)

        await self.backend.execute_script("""
            CREATE TABLE IF NOT EXISTS a2a_task_inputs (
                task_id TEXT NOT NULL REFERENCES a2a_tasks(id) ON DELETE CASCADE,
                input_revision BIGINT NOT NULL CHECK (input_revision >= 0),
                message_id TEXT NOT NULL, request_json TEXT NOT NULL, request_digest TEXT NOT NULL,
                wire_request_digest TEXT NOT NULL CHECK (length(wire_request_digest) = 64),
                PRIMARY KEY (task_id, input_revision), UNIQUE (task_id, message_id)
            )
        """)

        # Durable task authority and cancellation receipt (#3134). PostgreSQL
        # hosts initialize several agents concurrently, so serialize the whole
        # schema bundle and re-probe after acquiring the database-wide lock.
        # Without that second probe, every waiter would still replay DDL based
        # on the stale pre-lock schema it observed.
        if self.is_postgres:
            async with self._backend.transaction():
                await self._backend.execute(
                    "SELECT pg_advisory_xact_lock(hashtextextended(?, 0))",
                    (_CANCELLATION_SCHEMA_LOCK,),
                )
                if not await self._postgres_cancellation_schema_ready():
                    await self._ensure_cancellation_schema_objects()
        else:
            await self._ensure_cancellation_schema_objects()
        await self.changes.initialize()

    async def _postgres_cancellation_schema_ready(self) -> bool:
        """Re-probe every PostgreSQL object protected by the migration lock."""

        row = await self._backend.fetch_one("""
            SELECT
                (
                    SELECT COUNT(*) = 15
                    FROM information_schema.columns
                    WHERE table_schema = current_schema()
                      AND table_name = 'a2a_tasks'
                      AND column_name IN (
                          'creator_agent_id',
                          'recipient_agent_id',
                          'canceled_by',
                          'cancel_reason',
                          'cancel_previous_status',
                          'cancel_operation_id',
                          'terminal_operation_id',
                          'lifecycle_revision',
                          'lifecycle_updated_at',
                          'status_timestamp_seconds',
                          'status_timestamp_nanos',
                          'tenant',
                          'cognition_wake_eligible',
                          'input_revision',
                          'input_wake_eligible'
                      )
                )
                AND EXISTS (
                    SELECT 1
                    FROM pg_trigger trigger
                    JOIN pg_class relation
                      ON relation.oid = trigger.tgrelid
                    JOIN pg_namespace namespace
                      ON namespace.oid = relation.relnamespace
                    JOIN pg_proc procedure
                      ON procedure.oid = trigger.tgfoid
                    JOIN pg_namespace procedure_namespace
                      ON procedure_namespace.oid = procedure.pronamespace
                    WHERE namespace.nspname = current_schema()
                      AND relation.relname = 'a2a_tasks'
                      AND trigger.tgname = 'a2a_tasks_authority_fence_v5'
                      AND NOT trigger.tgisinternal
                      AND procedure_namespace.nspname = current_schema()
                      AND procedure.proname =
                          'a2a_tasks_enforce_authority_fence_v5'
                      AND pg_get_function_identity_arguments(procedure.oid) = ''
                      AND position(
                          'lifecycle_updated_at'
                          IN pg_get_functiondef(procedure.oid)
                      ) > 0
                )
                AND (
                    SELECT COUNT(*) = 5
                    FROM pg_class index_relation
                    JOIN pg_namespace namespace
                      ON namespace.oid = index_relation.relnamespace
                    WHERE namespace.nspname = current_schema()
                      AND index_relation.relkind = 'i'
                      AND index_relation.relname IN (
                          'idx_tasks_status',
                          'idx_tasks_session',
                          'idx_tasks_user',
                          'idx_tasks_creator',
                          'idx_tasks_recipient'
                      )
                )
        """)
        return bool(row and row[0])

    async def _ensure_cancellation_schema_objects(self) -> None:
        """Install the authority columns, terminal fence, and query indexes."""

        # Added under the same PostgreSQL migration lock as the authority
        # schema. Preserve absence independently from an explicit Unix epoch.
        for column, sql_type in (
            ("user_id", "TEXT"),
            ("status_timestamp_seconds", "BIGINT"),
            ("status_timestamp_nanos", "INTEGER"),
            ("tenant", "TEXT NOT NULL DEFAULT ''"),
            ("input_revision", "BIGINT NOT NULL DEFAULT 0 CHECK (input_revision >= 0)"),
            (
                "input_wake_eligible",
                "INTEGER NOT NULL DEFAULT 1 CHECK (input_wake_eligible IN (0,1))",
            ),
            (
                "cognition_wake_eligible",
                "INTEGER NOT NULL DEFAULT 1 CHECK (cognition_wake_eligible IN (0,1))",
            ),
        ):
            await self.add_column_if_missing("a2a_tasks", column, sql_type)

        # Separate columns rather than caller-controlled metadata give a
        # shared PostgreSQL table the same security boundary as per-agent
        # SQLite.
        authority_columns = (
            "creator_agent_id",
            "recipient_agent_id",
            "canceled_by",
            "cancel_reason",
            "cancel_previous_status",
            "cancel_operation_id",
            "terminal_operation_id",
        )
        for column in authority_columns:
            await self.add_column_if_missing("a2a_tasks", column, "TEXT")
        await self.add_column_if_missing(
            "a2a_tasks",
            "lifecycle_revision",
            "BIGINT NOT NULL DEFAULT 0",
        )
        # When the current revision began. ``updated_at`` cannot stand in:
        # appending an artifact moves it without a new revision, which would
        # keep an old wake identity inside the reconciliation window long
        # after retention forgot it. Rows written before this column existed
        # stay NULL and are read through ``updated_at``.
        await self.add_column_if_missing(
            "a2a_tasks",
            "lifecycle_updated_at",
            self.timestamp_type(),
        )

        # Cancellation is terminal at the storage boundary, including for
        # pre-upgrade writers still running during a PostgreSQL rollout. The
        # application predicates below protect current code, while this fence
        # prevents an older unconditional UPDATE/upsert from resurrecting a
        # row after a new worker has committed its cancellation receipt.
        await self._install_canceled_terminal_fence()

        # Create indexes
        await self._backend.execute(
            "CREATE INDEX IF NOT EXISTS idx_tasks_status ON a2a_tasks(status)"
        )
        await self._backend.execute(
            "CREATE INDEX IF NOT EXISTS idx_tasks_session ON a2a_tasks(session_id)"
        )
        if self.is_postgres:
            # PostgreSQL: user_id column for multi-tenant isolation
            await self._backend.execute(
                "CREATE INDEX IF NOT EXISTS idx_tasks_user ON a2a_tasks(user_id)"
            )
        await self._backend.execute(
            "CREATE INDEX IF NOT EXISTS idx_tasks_creator "
            "ON a2a_tasks(creator_agent_id)"
        )
        await self._backend.execute(
            "CREATE INDEX IF NOT EXISTS idx_tasks_recipient "
            "ON a2a_tasks(recipient_agent_id)"
        )

    async def _install_canceled_terminal_fence(self) -> None:
        """Fence legacy writers from live authority gaps and canceled mutation."""

        if self.is_postgres:
            await self._backend.execute_script("""
                CREATE OR REPLACE FUNCTION a2a_tasks_enforce_authority_fence_v5()
                RETURNS trigger AS $a2a_fence_function$
                BEGIN
                    IF TG_OP = 'UPDATE'
                       AND OLD.status IN ('completed', 'failed', 'canceled', 'rejected') THEN
                        RAISE EXCEPTION 'terminal A2A task cannot be replaced'
                            USING ERRCODE = 'check_violation';
                    END IF;
                    IF TG_OP = 'INSERT'
                       AND (NEW.creator_agent_id IS NULL
                            OR NEW.recipient_agent_id IS NULL) THEN
                        RAISE EXCEPTION 'A2A task requires durable authority'
                            USING ERRCODE = 'check_violation';
                    END IF;
                    IF NEW.status IN ('submitted', 'working', 'input-required', 'auth-required')
                       AND (NEW.creator_agent_id IS NULL
                            OR NEW.recipient_agent_id IS NULL) THEN
                        RAISE EXCEPTION 'live A2A task requires durable authority'
                            USING ERRCODE = 'check_violation';
                    END IF;
                    -- A v3/v4 worker does not know about lifecycle_revision.
                    -- Advance it at the shared database boundary so a status
                    -- transition by that worker cannot reuse a cognition-wake
                    -- identity minted for an earlier state of the same task.
                    IF TG_OP = 'UPDATE'
                       AND NEW.status IS DISTINCT FROM OLD.status THEN
                        NEW.lifecycle_revision := OLD.lifecycle_revision + 1;
                        NEW.lifecycle_updated_at := now();
                    END IF;
                    IF TG_OP = 'INSERT' AND NEW.lifecycle_updated_at IS NULL THEN
                        NEW.lifecycle_updated_at := now();
                    END IF;
                    RETURN NEW;
                END;
                $a2a_fence_function$ LANGUAGE plpgsql;

                DROP TRIGGER IF EXISTS a2a_tasks_canceled_terminal_v1
                    ON a2a_tasks;
                DROP TRIGGER IF EXISTS a2a_tasks_authority_fence_v2
                    ON a2a_tasks;
                DROP TRIGGER IF EXISTS a2a_tasks_authority_fence_v5
                    ON a2a_tasks;
                CREATE TRIGGER a2a_tasks_authority_fence_v5
                BEFORE INSERT OR UPDATE ON a2a_tasks
                FOR EACH ROW
                EXECUTE FUNCTION a2a_tasks_enforce_authority_fence_v5();

                DROP TRIGGER IF EXISTS a2a_tasks_authority_fence_v3
                    ON a2a_tasks;
                DROP TRIGGER IF EXISTS a2a_tasks_authority_fence_v4
                    ON a2a_tasks;
                DROP FUNCTION IF EXISTS a2a_tasks_enforce_authority_fence_v4();

                -- The old functions are no longer load-bearing. Drop them only
                -- after their triggers; the v4 trigger is already present and
                -- continues to fence terminal mutation and authority-less
                -- inserts throughout a mixed-version rollout.
                DROP FUNCTION IF EXISTS a2a_tasks_enforce_authority_fence_v3();
                DROP FUNCTION IF EXISTS a2a_tasks_enforce_authority_fence();
            """)
            return

        await self._backend.execute_script("""
            -- SQLite executes each executescript statement visibly unless the
            -- script opens its own transaction. Install every replacement
            -- before retiring its predecessor so another connection never
            -- observes a terminal row without an update/replace fence.
            CREATE TRIGGER IF NOT EXISTS a2a_tasks_terminal_update_v5
            BEFORE UPDATE ON a2a_tasks
            FOR EACH ROW
            WHEN OLD.status IN ('completed', 'failed', 'canceled', 'rejected')
            BEGIN
                SELECT RAISE(ABORT, 'terminal A2A task cannot be replaced');
            END;

            CREATE TRIGGER IF NOT EXISTS a2a_tasks_terminal_replace_v5
            BEFORE INSERT ON a2a_tasks
            FOR EACH ROW
            WHEN EXISTS (
                SELECT 1
                 FROM a2a_tasks
                 WHERE id = NEW.id
                   AND status IN ('completed', 'failed', 'canceled', 'rejected')
            )
            BEGIN
                -- IGNORE runs before SQLite's REPLACE conflict action deletes
                -- the occupied row. It therefore fences legacy INSERT OR
                -- REPLACE while allowing modern INSERT ... ON CONFLICT DO
                -- NOTHING to report its normal zero-row duplicate outcome.
                SELECT RAISE(IGNORE);
            END;

            CREATE TRIGGER IF NOT EXISTS a2a_tasks_authority_insert_v5
            BEFORE INSERT ON a2a_tasks
            FOR EACH ROW
            WHEN NEW.creator_agent_id IS NULL
              OR NEW.recipient_agent_id IS NULL
            BEGIN
                SELECT RAISE(ABORT, 'A2A task requires durable authority');
            END;

            CREATE TRIGGER IF NOT EXISTS a2a_tasks_live_authority_update_v5
            BEFORE UPDATE ON a2a_tasks
            FOR EACH ROW
            WHEN NEW.status IN ('submitted', 'working', 'input-required', 'auth-required')
              AND (NEW.creator_agent_id IS NULL
                   OR NEW.recipient_agent_id IS NULL)
            BEGIN
                SELECT RAISE(ABORT, 'live A2A task requires durable authority');
            END;

            DROP TRIGGER IF EXISTS a2a_tasks_terminal_update_v3;
            DROP TRIGGER IF EXISTS a2a_tasks_terminal_replace_v4;
            DROP TRIGGER IF EXISTS a2a_tasks_authority_insert_v3;
            DROP TRIGGER IF EXISTS a2a_tasks_live_authority_update_v2;
            DROP TRIGGER IF EXISTS a2a_tasks_canceled_terminal_v1;
            DROP TRIGGER IF EXISTS a2a_tasks_canceled_terminal_v2;
            DROP TRIGGER IF EXISTS a2a_tasks_canceled_replace_v3;
            DROP TRIGGER IF EXISTS a2a_tasks_terminal_replace_v3;
            DROP TRIGGER IF EXISTS a2a_tasks_live_authority_v2;
        """)

    async def save(
        self,
        task: Task,
        *,
        creator_agent_id: str | None = None,
        recipient_agent_id: str | None = None,
    ) -> bool:
        """Create an authoritative task.

        Lifecycle persistence is deliberately a separate recipient-owned seam;
        accepting an unscoped update here would let a shared-store worker mutate
        any task whose identifier it learns.
        """
        if creator_agent_id is not None or recipient_agent_id is not None:
            if creator_agent_id is None or recipient_agent_id is None:
                raise ValueError("Task creation requires both authority identities")
            await self.create(
                task,
                creator_agent_id=creator_agent_id,
                recipient_agent_id=recipient_agent_id,
            )
            return True

        raise ValueError("Task lifecycle saves require save_recipient_lifecycle")

    @staticmethod
    def _input_revision_predicate(
        task_id: str, recipient_agent_id: str
    ) -> tuple[str, tuple]:
        """Fence every recipient write made by an input-driven cognition turn.

        The dispatch context is server-owned and follows tools across runner
        tasks. Peer metadata is never consulted. Reads before this predicate
        cannot authorize a write after a newer input wins the row lock.
        """
        from kestrel_sovereign.signals.context import get_current_signal

        signal = get_current_signal()
        if signal is None or signal.source != "a2a.task_submitted":
            return "", ()
        payload = signal.payload
        if (
            signal.target_agent != recipient_agent_id
            or payload.get("task_id") != task_id
        ):
            return "", ()
        revision = payload.get("input_revision")
        if type(revision) is not int or revision < 0:
            raise ValueError("Task mutation requires its admitted input revision")
        return (
            " AND input_revision = ? AND cognition_wake_eligible = 1 AND input_wake_eligible = 1",
            (revision,),
        )

    async def save_recipient_lifecycle(
        self,
        task: Task,
        *,
        baseline: Task,
        recipient_agent_id: str,
        expected_state: TaskState,
    ) -> Task | None:
        """CAS a worker result from one expected live recipient-owned state."""

        if not isinstance(recipient_agent_id, str) or not recipient_agent_id.strip():
            raise ValueError("Task lifecycle recipient must be a concrete identity")

        if task.status.state == TaskState.TASK_STATE_CANCELED:
            raise ValueError(
                "CANCELED is an authorized transition; use cancel_if_authorized"
            )
        if expected_state in {
            TaskState.TASK_STATE_COMPLETED,
            TaskState.TASK_STATE_FAILED,
            TaskState.TASK_STATE_CANCELED,
            TaskState.TASK_STATE_REJECTED,
        }:
            raise ValueError("A terminal A2A lifecycle cannot be replaced")

        return await self._save_recipient_lifecycle_from_states(
            task,
            baseline=baseline,
            recipient_agent_id=recipient_agent_id,
            expected_states=(expected_state,),
        )

    async def save_recipient_terminal_outcome(
        self,
        task: Task,
        *,
        baseline: Task,
        recipient_agent_id: str,
        operation_id: str,
    ) -> Task | None:
        """Commit a handler's terminal result from whichever live state remains.

        A recipient may legitimately publish WORKING while its handler is still
        running. The handler result must therefore CAS against the live-state
        set rather than the stale state in its in-memory snapshot. Terminal
        states are deliberately absent from the predicate, so a cancellation
        or competing completion remains immutable.
        """

        if task.status.state not in {
            TaskState.TASK_STATE_COMPLETED,
            TaskState.TASK_STATE_FAILED,
            TaskState.TASK_STATE_REJECTED,
        }:
            raise ValueError(
                "Recipient execution outcomes must be completed, failed, or rejected"
            )
        if not isinstance(operation_id, str) or not operation_id:
            raise ValueError(
                "Recipient terminal operation ID must be a concrete string"
            )
        return await self._save_recipient_lifecycle_from_states(
            task,
            baseline=baseline,
            recipient_agent_id=recipient_agent_id,
            expected_states=(
                TaskState.TASK_STATE_SUBMITTED,
                TaskState.TASK_STATE_WORKING,
                TaskState.TASK_STATE_INPUT_REQUIRED,
                TaskState.TASK_STATE_AUTH_REQUIRED,
            ),
            terminal_operation_id=operation_id,
        )

    async def _save_recipient_lifecycle_from_states(
        self,
        task: Task,
        *,
        baseline: Task,
        recipient_agent_id: str,
        expected_states: tuple[TaskState, ...],
        terminal_operation_id: str | None = None,
    ) -> Task | None:
        """Apply an owned delta under the row lock and return what committed."""

        if not isinstance(recipient_agent_id, str) or not recipient_agent_id.strip():
            raise ValueError("Task lifecycle recipient must be a concrete identity")
        if not expected_states:
            raise ValueError("Task lifecycle CAS requires an expected state")

        if not isinstance(task, Task) or not isinstance(baseline, Task):
            raise TypeError("Task lifecycle writes require official A2A Task values")
        validate_proto_required_fields(task)
        validate_proto_required_fields(baseline)
        timestamp_seconds, timestamp_nanos = _status_timestamp(task.status)
        input_predicate, input_values = self._input_revision_predicate(
            task.id, recipient_agent_id
        )
        predicate = f"""
            id = ? AND {_TASK_SCOPE}
            AND session_id = ? AND recipient_agent_id = ?
            AND status IN ({", ".join("?" for _ in expected_states)})
            {input_predicate}
        """
        authority_values = (
            task.id,
            *self._scope_parameters,
            task.context_id,
            recipient_agent_id,
            *(_stored_state(state) for state in expected_states),
            *input_values,
        )
        transaction = self.write_transaction()
        conflict = None
        committed = None
        async with transaction:
            lock_suffix = " FOR UPDATE" if self.is_postgres else ""
            row = await self._backend.fetch_one(
                f"SELECT {_TASK_COLUMNS} FROM a2a_tasks WHERE {predicate}{lock_suffix}",
                authority_values,
            )
            if row is not None:
                try:
                    merged = merge_task_outcome(self._row_to_task(row), baseline, task)
                except TaskOutcomeConflictError as error:
                    # The backend wraps application exceptions raised inside a
                    # transaction. Keep a protocol conflict distinct from SQL.
                    conflict = error
                else:
                    message_json, artifacts_json, history_json, metadata_json = (
                        _task_payload(merged)
                    )
                    rows_affected = await self._backend.execute(
                        f"""
            UPDATE a2a_tasks SET
                status = ?,
                status_timestamp_seconds = ?,
                status_timestamp_nanos = ?,
                message = ?,
                artifacts = ?,
                history = ?,
                metadata = ?,
                terminal_operation_id = ?,
                lifecycle_revision = lifecycle_revision + 1,
                lifecycle_updated_at = {self.now_sql()},
                updated_at = {self.now_sql()}
            WHERE {predicate}
            """,
                        (
                            _stored_state(task.status.state),
                            timestamp_seconds,
                            timestamp_nanos,
                            message_json,
                            artifacts_json,
                            history_json,
                            metadata_json,
                            terminal_operation_id,
                            *authority_values,
                        ),
                    )
                    if rows_affected == 1:
                        committed_row = list(row)
                        committed_row[4:9] = [
                            _stored_state(task.status.state),
                            message_json,
                            artifacts_json,
                            history_json,
                            metadata_json,
                        ]
                        committed_row[20:22] = [timestamp_seconds, timestamp_nanos]
                        committed = self._row_to_task(tuple(committed_row))
        if conflict is not None:
            raise conflict
        return committed

    async def create(
        self,
        task: Task,
        *,
        creator_agent_id: str,
        recipient_agent_id: str,
        cognition_wake_eligible: bool = True,
        input_wake_eligible: bool = True,
    ) -> None:
        """Insert one new task with explicit server-owned wake policy."""

        if not isinstance(task, Task):
            raise TypeError("TaskStore accepts only official A2A protobuf tasks")
        if (
            type(cognition_wake_eligible) is not bool
            or type(input_wake_eligible) is not bool
        ):
            raise TypeError(
                "Task wake eligibility requires server-owned boolean policy"
            )
        if recipient_agent_id != self.scope.recipient_id:
            raise TaskMutationAuthorizationError(
                "Task recipient does not match the selected store"
            )
        if not creator_agent_id or not recipient_agent_id:
            raise ValueError("Task creation requires concrete authority identities")
        if task.status.state == TaskState.TASK_STATE_CANCELED:
            raise ValueError(
                "CANCELED is an authorized transition; use cancel_if_authorized"
            )
        message_json, artifacts_json, history_json, metadata_json = _task_payload(task)
        timestamp_seconds, timestamp_nanos = _status_timestamp(task.status)
        metadata = MessageToDict(task.metadata)
        task_type = metadata.get("task_type", "generic")
        user_id = metadata.get("user_id")

        try:
            rows_affected = await self._backend.execute(
                f"""
                INSERT INTO a2a_tasks
                (id, session_id, user_id, task_type, status, message, artifacts,
                 history, metadata, updated_at, lifecycle_updated_at,
                 creator_agent_id, recipient_agent_id,
                 status_timestamp_seconds, status_timestamp_nanos, tenant,
                 cognition_wake_eligible, input_wake_eligible)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, {self.now_sql()},
                        {self.now_sql()}, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (id) DO NOTHING
                """,
                (
                    task.id,
                    task.context_id,
                    user_id,
                    task_type,
                    _stored_state(task.status.state),
                    message_json,
                    artifacts_json,
                    history_json,
                    metadata_json,
                    creator_agent_id,
                    recipient_agent_id,
                    timestamp_seconds,
                    timestamp_nanos,
                    self.scope.tenant,
                    int(cognition_wake_eligible),
                    int(input_wake_eligible),
                ),
            )
        except QueryError as insert_error:
            # SQLite fires BEFORE INSERT triggers before applying this
            # statement's ON CONFLICT policy. The terminal canceled-row fence
            # must continue blocking legacy INSERT OR REPLACE writers, but an
            # ordinary duplicate submission still has the backend-independent
            # TaskAlreadyExists contract. Re-probe the canonical key after the
            # failed statement; do not translate unrelated storage failures.
            try:
                occupied = await self._backend.fetch_one(
                    f"SELECT 1 FROM a2a_tasks WHERE id = ? AND {_TASK_SCOPE}",
                    (task.id, *self._scope_parameters),
                )
            except QueryError:
                raise insert_error
            if occupied is not None:
                raise TaskAlreadyExistsError(
                    f"Task already exists: {task.id}"
                ) from insert_error
            raise
        if rows_affected != 1:
            raise TaskAlreadyExistsError(f"Task already exists: {task.id}")

    async def _get_unscoped(self, task_id: str) -> Task | None:
        """Principal-unscoped canonical read within the configured recipient/tenant."""
        row = await self._backend.fetch_one(
            f"SELECT {_TASK_COLUMNS} FROM a2a_tasks WHERE id = ? AND {_TASK_SCOPE}",
            (task_id, *self._scope_parameters),
        )
        if not row:
            return None
        return self._row_to_task(row)

    @staticmethod
    def _input_payload(
        request: SendMessageRequest | NativeTaskInput,
    ) -> NativeTaskInput:
        prepared = (
            request
            if isinstance(request, NativeTaskInput)
            else prepare_native_task_input(request)
        )
        if prepared.request.message.role != Role.ROLE_USER:
            raise TaskInputConflictError("Native task input requires a user message")
        return prepared

    async def _write_input(
        self,
        task_id: str,
        revision: int,
        request_json: str,
        digest: str,
        message_id: str,
        wire_digest: str,
    ) -> None:
        await self.backend.execute(
            "INSERT INTO a2a_task_inputs (task_id, input_revision, message_id, request_json, request_digest, wire_request_digest) VALUES (?, ?, ?, ?, ?, ?)",
            (task_id, revision, message_id, request_json, digest, wire_digest),
        )

    async def record_initial_input(
        self,
        task: Task,
        request: SendMessageRequest | NativeTaskInput,
        *,
        creator_agent_id: str,
    ) -> None:
        """Record request0 in the manager's context/task admission transaction."""
        prepared = self._input_payload(request)
        encoded, digest = prepared.request_json, prepared.execution_digest
        request = prepared.request
        if (
            request.tenant != self.scope.tenant
            or request.message.task_id
            or not task.history
            or task.history[0] != request.message
        ):
            raise TaskInputConflictError(
                "Initial request differs from its admitted task"
            )
        conflict = None
        async with self.write_transaction():
            current = await self.backend.fetch_one(
                f"SELECT input_revision FROM a2a_tasks WHERE id = ? AND {_TASK_SCOPE} AND creator_agent_id = ?",
                (task.id, *self._scope_parameters, creator_agent_id),
            )
            if current != (0,):
                conflict = TaskInputConflictError(
                    "Initial request has no matching creator-owned task"
                )
            else:
                await self._write_input(
                    task.id,
                    0,
                    encoded,
                    digest,
                    request.message.message_id,
                    prepared.wire_request_digest,
                )
        if conflict is not None:
            raise conflict

    async def append_creator_input(
        self, request: SendMessageRequest | NativeTaskInput, *, creator_agent_id: str
    ) -> tuple[NativeTaskSnapshot | None, bool]:
        """Append an input under creator/context authority and the task row lock."""
        from kestrel_sovereign.a2a.extensions import (
            DELIVERY_INTENT_URI,
            QUESTION_CORRELATION_URI,
        )

        prepared = self._input_payload(request)
        encoded, digest = prepared.request_json, prepared.execution_digest
        request = prepared.request
        if not isinstance(creator_agent_id, str) or not creator_agent_id.strip():
            raise ValueError("Task continuation requires a verified creator")
        if not request.message.task_id or request.tenant != self.scope.tenant:
            raise TaskInputConflictError(
                "Task continuation requires its routed task and tenant"
            )
        current, appended, conflict = None, False, None
        async with self.write_transaction():
            suffix = " FOR UPDATE" if self.is_postgres else ""
            row = await self.backend.fetch_one(
                f"SELECT {_SNAPSHOT_COLUMNS} FROM a2a_tasks WHERE id = ? AND {_TASK_SCOPE} AND creator_agent_id = ?{suffix}",
                (request.message.task_id, *self._scope_parameters, creator_agent_id),
            )
            if row is not None:
                current = self._row_to_snapshot(row)
                task = current.task
                if task.status.state in TERMINAL_STATES:
                    conflict = TaskInputTerminalError(
                        "A terminal task cannot accept additional messages"
                    )
                elif (
                    not current.cognition_wake_eligible
                    or not current.input_wake_eligible
                ):
                    conflict = TaskInputUnsupportedError(
                        "This task does not accept incoming cognition inputs"
                    )
                elif (
                    request.message.context_id
                    and request.message.context_id != task.context_id
                ):
                    conflict = TaskInputConflictError(
                        "Task and message context identifiers disagree"
                    )
                else:
                    try:
                        await ContextStore(
                            self.backend, owns_backend=False
                        ).require_owned(
                            task.context_id,
                            self.scope,
                            creator_agent_id=creator_agent_id,
                        )
                    except ContextOwnershipError as error:
                        conflict = error
                    if conflict is None:
                        prior = await self.backend.fetch_one(
                            "SELECT wire_request_digest FROM a2a_task_inputs WHERE task_id = ? AND message_id = ?",
                            (task.id, request.message.message_id),
                        )
                        if prior is not None:
                            if prior != (prepared.wire_request_digest,):
                                conflict = TaskInputConflictError(
                                    "Message ID already belongs to another native request"
                                )
                        else:
                            metadata, original = (
                                MessageToDict(request.metadata),
                                MessageToDict(task.metadata),
                            )
                            if any(
                                uri in metadata and metadata[uri] != original.get(uri)
                                for uri in (
                                    DELIVERY_INTENT_URI,
                                    QUESTION_CORRELATION_URI,
                                )
                            ):
                                conflict = TaskInputConflictError(
                                    "Continuation cannot change the task's question or delivery identity"
                                )
                            else:
                                task.history.append(request.message)
                                stamp = Timestamp()
                                stamp.FromDatetime(self.now_utc())
                                task.status.CopyFrom(
                                    TaskStatus(
                                        state=_native_state("submitted"),
                                        timestamp=stamp,
                                    )
                                )
                                message, artifacts, history, task_metadata = (
                                    _task_payload(task)
                                )
                                revision = current.input_revision + 1
                                changed = await self.backend.execute(
                                    "UPDATE a2a_tasks SET input_revision = ?, status = 'submitted', "
                                    "status_timestamp_seconds = ?, status_timestamp_nanos = ?, message = ?, artifacts = ?, history = ?, metadata = ?, "
                                    f"updated_at = {self.now_sql()}, lifecycle_updated_at = {self.now_sql()} "
                                    f"WHERE id = ? AND {_TASK_SCOPE} AND creator_agent_id = ? AND input_revision = ?",
                                    (
                                        revision,
                                        stamp.seconds,
                                        stamp.nanos,
                                        message,
                                        artifacts,
                                        history,
                                        task_metadata,
                                        task.id,
                                        *self._scope_parameters,
                                        creator_agent_id,
                                        current.input_revision,
                                    ),
                                )
                                if changed != 1:
                                    raise RuntimeError(
                                        "Locked native task input lost its revision"
                                    )
                                await self._write_input(
                                    task.id,
                                    revision,
                                    encoded,
                                    digest,
                                    request.message.message_id,
                                    prepared.wire_request_digest,
                                )
                                row = await self.backend.fetch_one(
                                    f"SELECT {_SNAPSHOT_COLUMNS} FROM a2a_tasks WHERE id = ? AND {_TASK_SCOPE}",
                                    (task.id, *self._scope_parameters),
                                )
                                if row is None:
                                    raise RuntimeError(
                                        "Admitted native task input was not readable"
                                    )
                                current, appended = self._row_to_snapshot(row), True
        if conflict is not None:
            raise conflict
        return current, appended

    async def get_task_snapshot(
        self,
        task_id: str,
        *,
        recipient_agent_id: str,
    ) -> NativeTaskSnapshot | None:
        """Read one native wake from the durable recipient/tenant scope."""
        if not isinstance(recipient_agent_id, str) or not recipient_agent_id.strip():
            raise ValueError("Task wake requires a concrete recipient")
        row = await self._backend.fetch_one(
            f"SELECT {_SNAPSHOT_COLUMNS} FROM a2a_tasks WHERE id = ? AND {_TASK_SCOPE} AND recipient_agent_id = ?",
            (task_id, *self._scope_parameters, recipient_agent_id),
        )
        return self._row_to_snapshot(row) if row is not None else None

    def _row_to_snapshot(self, row: tuple) -> NativeTaskSnapshot:
        """Use principal columns, never payload metadata, for wake ownership."""
        candidate = NativeTaskSnapshot(
            task=self._row_to_task(row),
            lifecycle_revision=row[18],
            input_revision=row[23],
            cognition_wake_eligible=bool(row[24]),
            input_wake_eligible=bool(row[25]),
            input_request_json=row[26],
            input_request_digest=row[27],
            input_wire_request_digest=row[28],
            ownership=ContextOwnership(
                context_id=row[1],
                tenant=row[22],
                recipient_agent_id=row[12],
                creator_agent_id=row[11],
            ),
        )
        candidate.require_recipient(self.scope.recipient_id)
        return candidate

    async def get_for_creator(
        self,
        task_id: str,
        creator_agent_id: str,
        *,
        recipient_agent_id: str,
    ) -> Task | None:
        """Retrieve a sender-owned result through its exact routed peer."""

        if not isinstance(creator_agent_id, str) or not creator_agent_id.strip():
            raise ValueError("Task creator lookup requires a concrete identity")
        if not isinstance(recipient_agent_id, str) or not recipient_agent_id.strip():
            raise ValueError("Task creator lookup requires a concrete recipient")
        row = await self._backend.fetch_one(
            f"SELECT {_TASK_COLUMNS} FROM a2a_tasks WHERE id = ? AND {_TASK_SCOPE} AND creator_agent_id = ? "
            "AND recipient_agent_id = ?",
            (task_id, *self._scope_parameters, creator_agent_id, recipient_agent_id),
        )
        if not row:
            return None
        return self._row_to_task(row)

    async def get_for_principal(
        self,
        task_id: str,
        principal_agent_id: str,
    ) -> Task | None:
        """Retrieve a task visible to either one of its durable principals."""

        if not isinstance(principal_agent_id, str) or not principal_agent_id.strip():
            raise ValueError("Task principal lookup requires a concrete identity")
        row = await self._backend.fetch_one(
            f"SELECT {_TASK_COLUMNS} FROM a2a_tasks WHERE id = ? AND {_TASK_SCOPE} "
            "AND (creator_agent_id = ? OR recipient_agent_id = ?)",
            (task_id, *self._scope_parameters, principal_agent_id, principal_agent_id),
        )
        if not row:
            return None
        return self._row_to_task(row)

    async def get_for_recipient(
        self,
        task_id: str,
        recipient_agent_id: str,
    ) -> Task | None:
        """Retrieve a task only through its durable responder principal."""

        if not isinstance(recipient_agent_id, str) or not recipient_agent_id.strip():
            raise ValueError("Task recipient lookup requires a concrete identity")
        row = await self._backend.fetch_one(
            f"SELECT {_TASK_COLUMNS} FROM a2a_tasks WHERE id = ? AND {_TASK_SCOPE} AND recipient_agent_id = ?",
            (task_id, *self._scope_parameters, recipient_agent_id),
        )
        if not row:
            return None
        return self._row_to_task(row)

    async def get_cancellation_snapshot(
        self,
        task_id: str,
        *,
        recipient_agent_id: str,
    ) -> TaskCancellationSnapshot | None:
        """Read cancellation authority without loading task payload columns."""

        if not isinstance(recipient_agent_id, str) or not recipient_agent_id:
            raise ValueError("Cancellation snapshot requires a concrete recipient")

        row = await self._backend.fetch_one(
            "SELECT status, canceled_by, input_revision, cognition_wake_eligible, input_wake_eligible FROM a2a_tasks "
            f"WHERE id = ? AND {_TASK_SCOPE} AND recipient_agent_id = ?",
            (task_id, *self._scope_parameters, recipient_agent_id),
        )
        if not row:
            return None
        return TaskCancellationSnapshot(
            state=str(row[0]),
            actor_agent_id=row[1],
            input_revision=row[2],
            cognition_wake_eligible=bool(row[3]),
            input_wake_eligible=bool(row[4]),
        )

    async def get_cancellation_operation_id(self, task_id: str) -> str | None:
        """Return the private attempt token proving which cancel committed."""

        row = await self._backend.fetch_one(
            f"SELECT cancel_operation_id FROM a2a_tasks WHERE id = ? AND {_TASK_SCOPE}",
            (task_id, *self._scope_parameters),
        )
        return row[0] if row else None

    async def get_recipient_terminal_operation_id(
        self,
        task_id: str,
        recipient_agent_id: str,
    ) -> str | None:
        """Return the private token proving which recipient terminal write won."""

        if not isinstance(recipient_agent_id, str) or not recipient_agent_id.strip():
            raise ValueError("Task recipient lookup requires a concrete identity")
        row = await self._backend.fetch_one(
            "SELECT terminal_operation_id FROM a2a_tasks "
            f"WHERE id = ? AND {_TASK_SCOPE} AND recipient_agent_id = ?",
            (task_id, *self._scope_parameters, recipient_agent_id),
        )
        return row[0] if row else None

    async def is_task_recipient(self, task_id: str, agent_id: str) -> bool:
        """Whether ``agent_id`` is the durable execution recipient of ``task_id``."""

        if not isinstance(agent_id, str) or not agent_id:
            raise ValueError("Task recipient lookup requires a concrete identity")
        row = await self._backend.fetch_one(
            f"SELECT 1 FROM a2a_tasks WHERE id = ? AND {_TASK_SCOPE} AND recipient_agent_id = ?",
            (task_id, *self._scope_parameters, agent_id),
        )
        return row is not None

    async def get_pending_tasks(
        self,
        *,
        recipient_agent_id: str,
        limit: int = 10,
    ) -> list[Task]:
        """Get SUBMITTED work addressed to one durable recipient."""
        if not isinstance(recipient_agent_id, str) or not recipient_agent_id.strip():
            raise ValueError("Pending-task reads require a concrete recipient")
        rows = await self._backend.fetch_all(
            f"""
            SELECT {_TASK_COLUMNS} FROM a2a_tasks
            WHERE {_TASK_SCOPE} AND status = 'submitted' AND recipient_agent_id = ?
            ORDER BY created_at ASC
            LIMIT ?
            """,
            (*self._scope_parameters, recipient_agent_id, limit),
        )
        return [self._row_to_task(row) for row in rows]

    async def update_status(
        self,
        task_id: str,
        status: TaskStatus,
        *,
        recipient_agent_id: str,
        expected_state: TaskState,
    ) -> bool:
        """Update a live task only when the durable recipient matches."""
        if not isinstance(recipient_agent_id, str) or not recipient_agent_id.strip():
            raise ValueError("Task status recipient must be a concrete identity")
        if status.state == TaskState.TASK_STATE_CANCELED:
            raise ValueError(
                "CANCELED is an authorized transition; use cancel_if_authorized"
            )
        if expected_state in {
            TaskState.TASK_STATE_COMPLETED,
            TaskState.TASK_STATE_FAILED,
            TaskState.TASK_STATE_CANCELED,
            TaskState.TASK_STATE_REJECTED,
        }:
            raise ValueError("A terminal A2A lifecycle cannot be replaced")
        validate_proto_required_fields(status)
        message_json = (
            MessageToJson(status.message) if status.HasField("message") else None
        )
        timestamp_seconds, timestamp_nanos = _status_timestamp(status)
        input_predicate, input_values = self._input_revision_predicate(
            task_id, recipient_agent_id
        )
        rows_affected = await self._backend.execute(
            f"""
            UPDATE a2a_tasks
            SET status = ?, message = ?,
                status_timestamp_seconds = ?, status_timestamp_nanos = ?,
                lifecycle_revision = lifecycle_revision + 1,
                lifecycle_updated_at = {self.now_sql()},
                updated_at = {self.now_sql()}
            WHERE id = ? AND {_TASK_SCOPE}
              AND recipient_agent_id = ?
              AND status = ? {input_predicate}
            """,
            (
                _stored_state(status.state),
                message_json,
                timestamp_seconds,
                timestamp_nanos,
                task_id,
                *self._scope_parameters,
                recipient_agent_id,
                _stored_state(expected_state),
                *input_values,
            ),
        )
        return rows_affected == 1

    async def cancel_if_authorized(
        self,
        task_id: str,
        *,
        actor_agent_id: str,
        expected_recipient_agent_id: str | None = None,
        reason: str | None = None,
        task_payload: Task | None = None,
        task_baseline: Task | None = None,
        operation_id: str | None = None,
    ) -> Task | None:
        """Atomically cancel one live task owned by or delegated to ``actor``.

        Creator and recipient are the two durable principals in an A2A task:
        the creator assigned the work and the recipient is its execution
        delegate.  The successful authorization decision and terminal state
        transition are one SQL predicate, so neither backend has a check/use
        window.  ``None`` means the predicate did not match; callers may read
        afterward to distinguish absence, a terminal task, and refusal.
        """
        if not isinstance(actor_agent_id, str) or not actor_agent_id:
            raise ValueError("Cancellation actor must be a concrete agent identity")
        if operation_id is not None and (
            not isinstance(operation_id, str) or not operation_id
        ):
            raise ValueError("Cancellation operation ID must be a concrete string")
        if expected_recipient_agent_id is not None and (
            not isinstance(expected_recipient_agent_id, str)
            or not expected_recipient_agent_id
        ):
            raise ValueError(
                "Expected cancellation recipient must be a concrete agent identity"
            )
        if (task_payload is None) != (task_baseline is None):
            raise ValueError(
                "Cancellation payload and baseline must be supplied together"
            )
        if task_payload is not None:
            if not isinstance(task_payload, Task) or not isinstance(
                task_baseline, Task
            ):
                raise TypeError("Cancellation requires official A2A Task values")
            validate_proto_required_fields(task_payload)
            validate_proto_required_fields(task_baseline)
            if (
                task_payload.id != task_id
                or task_payload.status.state != TaskState.TASK_STATE_CANCELED
            ):
                raise ValueError(
                    "Cancellation payload must be the canceled form of the same task"
                )

        timestamp = Timestamp()
        timestamp.FromDatetime(datetime.now(UTC))
        transaction = self.write_transaction()
        recipient_predicate = (
            "AND recipient_agent_id = ?"
            if expected_recipient_agent_id is not None
            else ""
        )
        recipient_values = (
            (expected_recipient_agent_id,)
            if expected_recipient_agent_id is not None
            else ()
        )
        input_predicate, input_values = self._input_revision_predicate(
            task_id, actor_agent_id
        )
        live_authority_predicate = f"""
            id = ? AND {_TASK_SCOPE}
            AND status IN ('submitted', 'working', 'input-required', 'auth-required')
            AND (creator_agent_id = ? OR recipient_agent_id = ?)
            {recipient_predicate} {input_predicate}
        """
        authority_values = (
            task_id,
            *self._scope_parameters,
            actor_agent_id,
            actor_agent_id,
            *recipient_values,
            *input_values,
        )

        conflict = None
        committed = None
        async with transaction:
            lock_suffix = " FOR UPDATE" if self.is_postgres else ""
            current = await self._backend.fetch_one(
                f"""
                SELECT {_TASK_COLUMNS} FROM a2a_tasks
                WHERE {live_authority_predicate}
                {lock_suffix}
                """,
                authority_values,
            )
            if current is None:
                return None

            latest = self._row_to_task(current)
            try:
                merged = (
                    merge_task_outcome(latest, task_baseline, task_payload)
                    if task_payload is not None
                    else latest
                )
            except TaskOutcomeConflictError as error:
                conflict = error
            else:
                message = new_text_message(
                    f"Task canceled by {actor_agent_id}: {reason}"
                    if reason
                    else f"Task canceled by {actor_agent_id}",
                    task_id=task_id,
                    context_id=merged.context_id,
                )
                merged.status.CopyFrom(
                    TaskStatus(
                        state=TaskState.TASK_STATE_CANCELED,
                        message=message,
                        timestamp=timestamp,
                    )
                )
                merged.history.append(message)
                message_json, artifacts_json, history_json, metadata_json = (
                    _task_payload(merged)
                )
                rows_affected = await self._backend.execute(
                    f"""
                    UPDATE a2a_tasks
                    SET cancel_previous_status = status,
                        status = 'canceled',
                        status_timestamp_seconds = ?,
                        status_timestamp_nanos = ?,
                        message = ?,
                        artifacts = ?,
                        history = ?,
                        metadata = ?,
                        canceled_by = ?,
                        cancel_reason = ?,
                        cancel_operation_id = ?,
                        lifecycle_revision = lifecycle_revision + 1,
                        lifecycle_updated_at = {self.now_sql()},
                        updated_at = {self.now_sql()}
                    WHERE {live_authority_predicate}
                    """,
                    (
                        timestamp.seconds,
                        timestamp.nanos,
                        message_json,
                        artifacts_json,
                        history_json,
                        metadata_json,
                        actor_agent_id,
                        reason,
                        operation_id,
                        *authority_values,
                    ),
                )
                if rows_affected != 1:
                    return None
                # Materialize the exact committed payload from the locked row and
                # the values written above.  A second read after commit can fail or
                # be canceled after authority has already changed, which would
                # wrongly roll back local intent and abandon every projection.
                canceled_row = list(current)
                canceled_row[4] = _stored_state(TaskState.TASK_STATE_CANCELED)
                canceled_row[5] = message_json
                canceled_row[6] = artifacts_json
                canceled_row[7] = history_json
                canceled_row[8] = metadata_json
                canceled_row[13] = actor_agent_id
                canceled_row[14] = reason
                canceled_row[15] = current[4]
                canceled_row[16] = operation_id
                canceled_row[20] = timestamp.seconds
                canceled_row[21] = timestamp.nanos
                committed = self._row_to_task(tuple(canceled_row))
        if conflict is not None:
            raise conflict
        return committed

    async def add_artifact(
        self,
        task_id: str,
        artifact: Artifact,
        *,
        recipient_agent_id: str,
        append: bool = False,
    ) -> None:
        """Append an artifact without crossing a terminal-state transition.

        Cancellation may merge a handler's partial payload into this same JSON
        column.  The artifact writer therefore takes the same row lock (or the
        SQLite immediate-writer lease) and predicates its update on a live
        status.  Whichever operation wins first is preserved: cancellation
        sees an already-appended artifact, or the later append is refused.
        """

        if not isinstance(artifact, Artifact):
            raise TypeError("Task artifact writes require an official A2A Artifact")
        validate_proto_required_fields(artifact)
        if not isinstance(append, bool):
            raise TypeError("Artifact append must be a boolean")
        if not isinstance(recipient_agent_id, str) or not recipient_agent_id.strip():
            raise ValueError("Task artifact recipient must be a concrete identity")

        input_predicate, input_values = self._input_revision_predicate(
            task_id, recipient_agent_id
        )
        transaction = self.write_transaction()
        unauthorized = False
        terminal_status: str | None = None
        update_conflict = False
        protocol_error: InvalidAgentResponseError | None = None
        async with transaction:
            lock_suffix = " FOR UPDATE" if self.is_postgres else ""
            row = await self._backend.fetch_one(
                f"SELECT {_TASK_COLUMNS} FROM a2a_tasks "
                f"WHERE id = ? AND {_TASK_SCOPE} AND recipient_agent_id = ?{input_predicate}{lock_suffix}",
                (task_id, *self._scope_parameters, recipient_agent_id, *input_values),
            )
            if not row:
                unauthorized = True
            elif row[4] not in {
                _stored_state(TaskState.TASK_STATE_SUBMITTED),
                _stored_state(TaskState.TASK_STATE_WORKING),
                _stored_state(TaskState.TASK_STATE_INPUT_REQUIRED),
                _stored_state(TaskState.TASK_STATE_AUTH_REQUIRED),
            }:
                terminal_status = str(row[4])
            else:
                task = self._row_to_task(row)
                try:
                    append_artifact_to_task(
                        task,
                        TaskArtifactUpdateEvent(
                            task_id=task.id,
                            context_id=task.context_id,
                            artifact=artifact,
                            append=append,
                        ),
                    )
                except InvalidAgentResponseError as error:
                    protocol_error = error
                if protocol_error is None:
                    artifacts = [MessageToDict(value) for value in task.artifacts]

                    rows_affected = await self._backend.execute(
                        f"""
                        UPDATE a2a_tasks
                        SET artifacts = ?, updated_at = {self.now_sql()}
                        WHERE id = ? AND {_TASK_SCOPE}
                          AND recipient_agent_id = ?
                          AND status IN ('submitted', 'working', 'input-required', 'auth-required')
                          {input_predicate}
                        """,
                        (
                            json_dumps(artifacts),
                            task_id,
                            *self._scope_parameters,
                            recipient_agent_id,
                            *input_values,
                        ),
                    )
                    update_conflict = rows_affected != 1
        if unauthorized:
            raise TaskMutationAuthorizationError(
                f"Task mutation was not authorized or task was not found: {task_id}"
            )
        if terminal_status is not None:
            raise ValueError(
                f"Cannot add artifact to terminal task {task_id}: {terminal_status}"
            )
        if protocol_error is not None:
            raise protocol_error
        if update_conflict:
            raise ValueError(f"Cannot add artifact to terminal task {task_id}")

    async def list_tasks(
        self,
        *,
        recipient_agent_id: str,
        context_id: str | None = None,
        user_id: str | None = None,
        status: TaskState | None = None,
        limit: int = 100,
    ) -> list[Task]:
        """Project native tasks from the same authoritative recipient query."""
        snapshots = await self.list_task_snapshots(
            recipient_agent_id=recipient_agent_id,
            context_id=context_id,
            user_id=user_id,
            status=status,
            limit=limit,
        )
        return [snapshot.task for snapshot in snapshots]

    async def list_task_snapshots(
        self,
        *,
        recipient_agent_id: str,
        context_id: str | None = None,
        user_id: str | None = None,
        status: TaskState | None = None,
        limit: int = 100,
    ) -> list[NativeTaskSnapshot]:
        """List only tasks addressed to one durable recipient."""
        if not isinstance(recipient_agent_id, str) or not recipient_agent_id.strip():
            raise ValueError("Task listing requires a concrete recipient")
        conditions = [_TASK_SCOPE, "recipient_agent_id = ?"]
        params = [*self._scope_parameters, recipient_agent_id]

        if context_id:
            conditions.append("session_id = ?")
            params.append(context_id)
        if user_id:
            conditions.append("user_id = ?")
            params.append(user_id)
        if status:
            conditions.append("status = ?")
            params.append(_stored_state(status))

        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        params.append(limit)

        rows = await self._backend.fetch_all(
            f"""
            SELECT {_SNAPSHOT_COLUMNS} FROM a2a_tasks
            {where}
            ORDER BY created_at DESC
            LIMIT ?
            """,
            tuple(params),
        )
        return [self._row_to_snapshot(row) for row in rows]

    async def list_for_creator(
        self,
        request: ListTasksRequest,
        *,
        creator_agent_id: str,
        recipient_agent_id: str,
    ) -> ListTasksResponse:
        """List one authenticated creator's tasks using native keyset cursors."""
        if not isinstance(request, ListTasksRequest):
            raise TypeError("Native listing requires an official ListTasksRequest")
        if not all(
            isinstance(value, str) and value.strip()
            for value in (creator_agent_id, recipient_agent_id)
        ):
            raise ValueError(
                "Native listing requires concrete creator and recipient identities"
            )
        if request.tenant != self.scope.tenant:
            raise InvalidParamsError(
                message="A2A tenant does not match the selected recipient"
            )
        validate_history_length(request)
        page_size = (
            request.page_size
            if request.HasField("page_size")
            else DEFAULT_LIST_TASKS_PAGE_SIZE
        )
        validate_page_size(page_size)
        conditions = [_TASK_SCOPE, "recipient_agent_id = ?", "creator_agent_id = ?"]
        values = [*self._scope_parameters, recipient_agent_id, creator_agent_id]
        if request.context_id:
            conditions.append("session_id = ?")
            values.append(request.context_id)
        if request.status:
            try:
                state = _stored_state(request.status)
            except ValueError as error:
                raise InvalidParamsError(
                    message="Unknown native task status filter"
                ) from error
            conditions.append("status = ?")
            values.append(state)
        if request.HasField("status_timestamp_after"):
            try:
                request.status_timestamp_after.ToJsonString()
            except ValueError as error:
                raise InvalidParamsError(
                    message="Invalid status timestamp filter"
                ) from error
            seconds, nanos = (
                request.status_timestamp_after.seconds,
                request.status_timestamp_after.nanos,
            )
            conditions.append(
                "(status_timestamp_seconds > ? OR (status_timestamp_seconds = ? AND status_timestamp_nanos >= ?))"
            )
            values.extend((seconds, seconds, nanos))
        where = " AND ".join(conditions)
        total = await self._backend.fetch_one(
            f"SELECT COUNT(*) FROM a2a_tasks WHERE {where}",
            tuple(values),
        )
        if total is None:
            raise RuntimeError("Native task count query returned no result")
        if request.page_token:
            cursor = decode_list_tasks_cursor(request.page_token)
            if cursor is None:
                raise InvalidParamsError(message="Invalid native task page token")
            if cursor.timestamp_ns is None:
                where += " AND status_timestamp_seconds IS NULL AND id < ?"
                values.append(cursor.task_id)
            else:
                timestamp = Timestamp()
                try:
                    timestamp.FromNanoseconds(cursor.timestamp_ns)
                    timestamp.ToJsonString()
                except (ValueError, OverflowError) as error:
                    raise InvalidParamsError(
                        message="Invalid native task page timestamp"
                    ) from error
                seconds, nanos = timestamp.seconds, timestamp.nanos
                where += """ AND (
                    status_timestamp_seconds IS NULL OR status_timestamp_seconds < ? OR
                    (status_timestamp_seconds = ? AND status_timestamp_nanos < ?) OR
                    (status_timestamp_seconds = ? AND status_timestamp_nanos = ? AND id < ?)
                )"""
                values.extend((seconds, seconds, nanos, seconds, nanos, cursor.task_id))
        rows = await self._backend.fetch_all(
            f"""SELECT {_TASK_COLUMNS} FROM a2a_tasks WHERE {where}
                ORDER BY CASE WHEN status_timestamp_seconds IS NULL THEN 1 ELSE 0 END ASC,
                         status_timestamp_seconds DESC, status_timestamp_nanos DESC, id DESC
                LIMIT ?""",
            (*values, page_size + 1),
        )
        tasks = [self._row_to_task(row) for row in rows[:page_size]]
        next_page_token = ""
        if len(rows) > page_size:
            last = tasks[-1]
            next_page_token = encode_list_tasks_cursor(
                ListTasksCursor(
                    timestamp_ns=last.status.timestamp.ToNanoseconds()
                    if last.status.HasField("timestamp")
                    else None,
                    task_id=last.id,
                )
            )
        projected = []
        for task in tasks:
            if not request.include_artifacts:
                task.ClearField("artifacts")
            projected.append(apply_history_length(task, request))
        return ListTasksResponse(
            tasks=projected,
            total_size=total[0],
            page_size=page_size,
            next_page_token=next_page_token,
        )

    async def list_cognition_wake_candidates(
        self,
        *,
        recipient_agent_id: str,
        live_changed_since: datetime,
        terminal_changed_since: datetime,
    ) -> list[NativeTaskSnapshot]:
        """List authoritative rows whose one-shot cognition wake may be absent.

        Every row is bounded by when its current lifecycle revision began,
        against the retention horizon of the source that owns its wake: live
        SUBMITTED/WORKING rows by the submission source's, terminal rows by
        the completion source's. A row past its horizon has outlived the
        durable row that deduplicates its wake, so listing it would recreate
        that wake once per retention period.
        """

        if not isinstance(recipient_agent_id, str) or not recipient_agent_id.strip():
            raise ValueError("Task wake reconciliation requires a concrete recipient")
        changed_at = "COALESCE(lifecycle_updated_at, updated_at)"
        if self.is_sqlite:
            # Task writes use SQLite's ``datetime('now')`` (space separator),
            # while Python's canonical ISO form contains ``T`` and an offset.
            # TEXT comparison orders those representations incorrectly around
            # the retention boundary; SQLite's datetime parser normalizes both.
            changed_predicate = f"datetime({changed_at}) >= datetime(?)"
        else:
            changed_predicate = f"{changed_at} >= ?"
        rows = await self._backend.fetch_all(
            f"""
            SELECT {_SNAPSHOT_COLUMNS} FROM a2a_tasks
            WHERE {_TASK_SCOPE} AND recipient_agent_id = ?
              AND cognition_wake_eligible = 1
              AND (
                    (
                        status IN ('submitted', 'working')
                        AND input_wake_eligible = 1
                        AND {changed_predicate}
                    )
                    OR (
                        status IN ('completed', 'failed', 'canceled', 'rejected')
                        AND {changed_predicate}
                    )
                  )
            ORDER BY created_at ASC
            """,
            (
                *self._scope_parameters,
                recipient_agent_id,
                self.to_timestamp_param(live_changed_since),
                self.to_timestamp_param(terminal_changed_since),
            ),
        )
        return [self._row_to_snapshot(row) for row in rows]

    async def delete(self, task_id: str) -> bool:
        """Delete a task. Returns True if deleted."""
        rows_affected = await self._backend.execute(
            f"DELETE FROM a2a_tasks WHERE id = ? AND {_TASK_SCOPE}",
            (task_id, *self._scope_parameters),
        )
        return rows_affected > 0

    async def cleanup_old(
        self,
        older_than_days: int = 30,
        terminal_states: tuple = ("completed", "failed", "canceled", "rejected"),
    ) -> int:
        """
        Delete old tasks in terminal states.

        Args:
            older_than_days: Delete tasks updated before this many days ago.
            terminal_states: Task states eligible for cleanup.

        Returns:
            Number of tasks deleted.
        """
        cutoff = datetime.now(UTC) - timedelta(days=older_than_days)
        cutoff_param = self.to_timestamp_param(cutoff)

        placeholders = ", ".join("?" for _ in terminal_states)
        params = [*self._scope_parameters, *terminal_states, cutoff_param]

        rows_affected = await self._backend.execute(
            f"DELETE FROM a2a_tasks WHERE {_TASK_SCOPE} AND status IN ({placeholders}) AND updated_at < ?",
            tuple(params),
        )
        logger.info(
            f"Cleaned up {rows_affected} old tasks (older than {older_than_days} days)"
        )
        return rows_affected

    def _row_to_task(self, row: tuple) -> Task:
        """
        Convert database row to Task object.

        Row columns (in order):
        0: id, 1: session_id, 2: user_id, 3: task_type, 4: status,
        5: message, 6: artifacts, 7: history, 8: metadata,
        9: created_at, 10: updated_at, 11: creator_agent_id,
        12: recipient_agent_id, 13: canceled_by, 14: cancel_reason,
        15: cancel_previous_status, 16: cancel_operation_id,
        17: terminal_operation_id, 18: lifecycle_revision, 19: lifecycle_updated_at,
        20: status_timestamp_seconds, 21: status_timestamp_nanos
        """
        artifacts_data = json_loads(row[6]) if row[6] else []
        history_data = json_loads(row[7]) if row[7] else []
        metadata = without_reserved_cancellation_receipt(
            json_loads(row[8]) if row[8] else {}
        )
        if len(row) > 13 and row[13]:
            metadata = dict(metadata)
            metadata["cancellation_receipt"] = {
                "actor_agent_id": row[13],
                "reason": row[14] if len(row) > 14 else None,
                "status_before": row[15] if len(row) > 15 else None,
            }

        data = {
            "id": row[0],
            "contextId": row[1],
            "status": {"state": _native_state(row[4])},
            "artifacts": artifacts_data,
            "history": history_data,
        }
        if row[5] is not None:
            data["status"]["message"] = json_loads(row[5])
        if row[8] is not None or row[13]:
            data["metadata"] = metadata
        task = ParseDict(data, Task())
        seconds, nanos = row[20], row[21]
        if (seconds is None) != (nanos is None):
            raise ValueError("Stored A2A status timestamp is incomplete")
        if seconds is not None:
            task.status.timestamp.CopyFrom(Timestamp(seconds=seconds, nanos=nanos))
            task.status.timestamp.ToJsonString()
        return task
