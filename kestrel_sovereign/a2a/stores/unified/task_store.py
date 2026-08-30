"""
Unified TaskStore - Backend-Agnostic Task Persistence.

Manages task lifecycle from submission to completion/failure.
Works with both SQLite and PostgreSQL backends.
"""

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional

from kestrel_sovereign.a2a.types import (
    Artifact,
    Message,
    Task,
    TaskState,
    TaskStatus,
    TextPart,
)
from kestrel_sovereign.a2a.stores.base import json_dumps, json_loads
from kestrel_sovereign.a2a.stores.unified.base import UnifiedStoreBase
from kestrel_sovereign.storage.db.interface import DatabaseBackend

logger = logging.getLogger(__name__)

_CANCELLATION_SCHEMA_LOCK = "a2a_tasks_cancel_authority_v2"


class TaskAlreadyExistsError(ValueError):
    """A caller attempted to create a second task under an occupied ID."""


class TaskMutationAuthorizationError(PermissionError):
    """A responder mutation did not match the durable task recipient."""


@dataclass(frozen=True)
class TaskCancellationSnapshot:
    """Minimal durable state needed by live cancellation monitors."""

    state: str
    actor_agent_id: Optional[str]


def without_reserved_cancellation_receipt(
    metadata: Mapping[str, object] | None,
) -> dict:
    """Copy caller metadata without authority minted by cancellation."""

    sanitized = dict(metadata or {})
    sanitized.pop("cancellation_receipt", None)
    return sanitized


class TaskStore(UnifiedStoreBase):
    """
    Backend-agnostic task store.

    Replaces both SQLiteTaskStore and PostgresTaskStore with a single
    implementation that works with any DatabaseBackend.
    """

    def __init__(self, backend: DatabaseBackend):
        """
        Initialize task store with database backend.

        Args:
            backend: DatabaseBackend instance (SQLite or PostgreSQL)
        """
        super().__init__(backend)

    async def initialize(self) -> None:
        """Create tables if not exists."""
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
                cancel_operation_id TEXT
            )
        """)

        # Auto-migrate: add user_id column if missing (for existing databases)
        try:
            if self.is_postgres:
                await self._backend.execute(
                    "ALTER TABLE a2a_tasks ADD COLUMN IF NOT EXISTS user_id TEXT"
                )
            else:
                # SQLite: check if column exists first
                result = await self._backend.fetch_one(
                    "SELECT COUNT(*) FROM pragma_table_info('a2a_tasks') WHERE name='user_id'"
                )
                if result and result[0] == 0:
                    await self._backend.execute("ALTER TABLE a2a_tasks ADD COLUMN user_id TEXT")
                    logger.info("Migrated a2a_tasks: added user_id column")
        except Exception as e:
            logger.debug(f"Migration check for user_id: {e}")

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

        # Pre-#3134 live rows have no trustworthy creator/recipient columns.
        # Metadata was caller-controlled and a shared PostgreSQL table contains
        # multiple recipients, so guessing either principal would mint power.
        # Settle those rows explicitly instead of leaving work permanently live
        # but uncancellable after upgrade. Operators and clients see a terminal
        # failure with the migration reason; terminal legacy rows remain intact.
        legacy_settlement = Message(
            role="agent",
            parts=[
                TextPart(
                    text=(
                        "Task settled during the cancellation-authority upgrade: "
                        "its legacy row has no trustworthy creator/recipient binding"
                    )
                )
            ],
        )
        settled = await self._backend.execute(
            f"""
            UPDATE a2a_tasks
            SET status = 'failed',
                message = ?,
                updated_at = {self.now_sql()}
            WHERE status IN ('submitted', 'working', 'input-required')
              AND (creator_agent_id IS NULL OR recipient_agent_id IS NULL)
            """,
            (legacy_settlement.model_dump_json(),),
        )
        if settled:
            logger.warning(
                "Settled %d live legacy A2A task(s) without durable authority",
                settled,
            )

        logger.info(f"TaskStore initialized ({self._backend.backend_type})")

    async def _postgres_cancellation_schema_ready(self) -> bool:
        """Re-probe every PostgreSQL object protected by the migration lock."""

        row = await self._backend.fetch_one("""
            SELECT
                (
                    SELECT COUNT(*) = 6
                    FROM information_schema.columns
                    WHERE table_schema = current_schema()
                      AND table_name = 'a2a_tasks'
                      AND column_name IN (
                          'creator_agent_id',
                          'recipient_agent_id',
                          'canceled_by',
                          'cancel_reason',
                          'cancel_previous_status',
                          'cancel_operation_id'
                      )
                )
                AND EXISTS (
                    SELECT 1
                    FROM pg_proc procedure
                    JOIN pg_namespace namespace
                      ON namespace.oid = procedure.pronamespace
                    WHERE namespace.nspname = current_schema()
                      AND procedure.proname =
                          'a2a_tasks_enforce_authority_fence'
                      AND pg_get_function_identity_arguments(procedure.oid) = ''
                )
                AND EXISTS (
                    SELECT 1
                    FROM pg_trigger trigger
                    JOIN pg_class relation
                      ON relation.oid = trigger.tgrelid
                    JOIN pg_namespace namespace
                      ON namespace.oid = relation.relnamespace
                    WHERE namespace.nspname = current_schema()
                      AND relation.relname = 'a2a_tasks'
                      AND trigger.tgname = 'a2a_tasks_authority_fence_v2'
                      AND NOT trigger.tgisinternal
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
        )
        for column in authority_columns:
            await self.add_column_if_missing("a2a_tasks", column, "TEXT")

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
                CREATE OR REPLACE FUNCTION a2a_tasks_enforce_authority_fence()
                RETURNS trigger AS $a2a_fence_function$
                BEGIN
                    IF TG_OP = 'UPDATE'
                       AND OLD.status IN ('completed', 'failed', 'canceled') THEN
                        RAISE EXCEPTION 'terminal A2A task cannot be replaced'
                            USING ERRCODE = 'check_violation';
                    END IF;
                    IF NEW.status IN ('submitted', 'working', 'input-required')
                       AND (NEW.creator_agent_id IS NULL
                            OR NEW.recipient_agent_id IS NULL) THEN
                        RAISE EXCEPTION 'live A2A task requires durable authority'
                            USING ERRCODE = 'check_violation';
                    END IF;
                    RETURN NEW;
                END;
                $a2a_fence_function$ LANGUAGE plpgsql;

                DROP TRIGGER IF EXISTS a2a_tasks_canceled_terminal_v1
                    ON a2a_tasks;
                DROP TRIGGER IF EXISTS a2a_tasks_authority_fence_v2
                    ON a2a_tasks;
                CREATE TRIGGER a2a_tasks_authority_fence_v2
                BEFORE INSERT OR UPDATE ON a2a_tasks
                FOR EACH ROW
                EXECUTE FUNCTION a2a_tasks_enforce_authority_fence();
            """)
            return

        await self._backend.execute_script("""
            DROP TRIGGER IF EXISTS a2a_tasks_canceled_terminal_v1;
            DROP TRIGGER IF EXISTS a2a_tasks_canceled_terminal_v2;
            CREATE TRIGGER IF NOT EXISTS a2a_tasks_terminal_update_v3
            BEFORE UPDATE ON a2a_tasks
            FOR EACH ROW
            WHEN OLD.status IN ('completed', 'failed', 'canceled')
            BEGIN
                SELECT RAISE(ABORT, 'terminal A2A task cannot be replaced');
            END;

            CREATE TRIGGER IF NOT EXISTS a2a_tasks_terminal_replace_v3
            BEFORE INSERT ON a2a_tasks
            FOR EACH ROW
            WHEN EXISTS (
                SELECT 1
                 FROM a2a_tasks
                 WHERE id = NEW.id
                   AND status IN ('completed', 'failed', 'canceled')
            )
            BEGIN
                SELECT RAISE(ABORT, 'terminal A2A task cannot be replaced');
            END;

            CREATE TRIGGER IF NOT EXISTS a2a_tasks_live_authority_v2
            BEFORE INSERT ON a2a_tasks
            FOR EACH ROW
            WHEN NEW.status IN ('submitted', 'working', 'input-required')
              AND (NEW.creator_agent_id IS NULL
                   OR NEW.recipient_agent_id IS NULL)
            BEGIN
                SELECT RAISE(ABORT, 'live A2A task requires durable authority');
            END;

            CREATE TRIGGER IF NOT EXISTS a2a_tasks_live_authority_update_v2
            BEFORE UPDATE ON a2a_tasks
            FOR EACH ROW
            WHEN NEW.status IN ('submitted', 'working', 'input-required')
              AND (NEW.creator_agent_id IS NULL
                   OR NEW.recipient_agent_id IS NULL)
            BEGIN
                SELECT RAISE(ABORT, 'live A2A task requires durable authority');
            END;
        """)

    async def save(
        self,
        task: Task,
        *,
        creator_agent_id: Optional[str] = None,
        recipient_agent_id: Optional[str] = None,
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

        raise ValueError(
            "Task lifecycle saves require save_recipient_lifecycle"
        )

    async def save_recipient_lifecycle(
        self,
        task: Task,
        *,
        recipient_agent_id: str,
        expected_state: TaskState,
    ) -> bool:
        """CAS a worker result from one expected live recipient-owned state."""

        if not isinstance(recipient_agent_id, str) or not recipient_agent_id.strip():
            raise ValueError("Task lifecycle recipient must be a concrete identity")

        if task.status.state is TaskState.CANCELED:
            raise ValueError(
                "CANCELED is an authorized transition; use cancel_if_authorized"
            )
        if expected_state in {
            TaskState.COMPLETED,
            TaskState.FAILED,
            TaskState.CANCELED,
        }:
            raise ValueError("A terminal A2A lifecycle cannot be replaced")

        message_json = task.status.message.model_dump_json() if task.status.message else None
        artifacts_json = json_dumps([a.model_dump() for a in (task.artifacts or [])])
        history_json = json_dumps([m.model_dump() for m in (task.history or [])])
        metadata_json = json_dumps(
            without_reserved_cancellation_receipt(task.metadata)
        )

        rows_affected = await self._backend.execute(
            f"""
            UPDATE a2a_tasks SET
                status = ?,
                message = ?,
                artifacts = ?,
                history = ?,
                metadata = ?,
                updated_at = {self.now_sql()}
            WHERE id = ?
              AND recipient_agent_id = ?
              AND status = ?
            """,
            (
                task.status.state.value,
                message_json,
                artifacts_json,
                history_json,
                metadata_json,
                task.id,
                recipient_agent_id,
                expected_state.value,
            ),
        )
        return rows_affected == 1

    async def create(
        self,
        task: Task,
        *,
        creator_agent_id: str,
        recipient_agent_id: str,
    ) -> None:
        """Insert one new task and reject an occupied caller-supplied ID."""

        if not creator_agent_id or not recipient_agent_id:
            raise ValueError("Task creation requires concrete authority identities")
        if task.status.state is TaskState.CANCELED:
            raise ValueError(
                "CANCELED is an authorized transition; use cancel_if_authorized"
            )
        message_json = (
            task.status.message.model_dump_json() if task.status.message else None
        )
        artifacts_json = json_dumps([a.model_dump() for a in (task.artifacts or [])])
        history_json = json_dumps([m.model_dump() for m in (task.history or [])])
        metadata = without_reserved_cancellation_receipt(task.metadata)
        # Cancellation receipts are reserved durable state. A sender may put
        # arbitrary metadata on a new envelope, but only the authorized atomic
        # cancellation transition below may mint this field.
        metadata_json = json_dumps(metadata)
        task_type = (
            metadata.get("task_type", "generic")
        )
        user_id = metadata.get("user_id")

        rows_affected = await self._backend.execute(
            f"""
            INSERT INTO a2a_tasks
            (id, session_id, user_id, task_type, status, message, artifacts,
             history, metadata, updated_at, creator_agent_id, recipient_agent_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, {self.now_sql()}, ?, ?)
            ON CONFLICT (id) DO NOTHING
            """,
            (
                task.id,
                task.sessionId,
                user_id,
                task_type,
                task.status.state.value,
                message_json,
                artifacts_json,
                history_json,
                metadata_json,
                creator_agent_id,
                recipient_agent_id,
            ),
        )
        if rows_affected != 1:
            raise TaskAlreadyExistsError(f"Task already exists: {task.id}")

    async def get(self, task_id: str) -> Optional[Task]:
        """Retrieve a task by ID."""
        row = await self._backend.fetch_one(
            "SELECT * FROM a2a_tasks WHERE id = ?",
            (task_id,),
        )
        if not row:
            return None
        return self._row_to_task(row)

    async def get_for_recipient(
        self,
        task_id: str,
        recipient_agent_id: str,
    ) -> Optional[Task]:
        """Retrieve a task only through its durable responder principal."""

        if not isinstance(recipient_agent_id, str) or not recipient_agent_id.strip():
            raise ValueError("Task recipient lookup requires a concrete identity")
        row = await self._backend.fetch_one(
            "SELECT * FROM a2a_tasks WHERE id = ? AND recipient_agent_id = ?",
            (task_id, recipient_agent_id),
        )
        if not row:
            return None
        return self._row_to_task(row)

    async def get_cancellation_snapshot(
        self,
        task_id: str,
    ) -> Optional[TaskCancellationSnapshot]:
        """Read cancellation authority without loading task payload columns."""

        row = await self._backend.fetch_one(
            "SELECT status, canceled_by FROM a2a_tasks WHERE id = ?",
            (task_id,),
        )
        if not row:
            return None
        return TaskCancellationSnapshot(
            state=str(row[0]),
            actor_agent_id=row[1],
        )

    async def get_cancellation_operation_id(self, task_id: str) -> Optional[str]:
        """Return the private attempt token proving which cancel committed."""

        row = await self._backend.fetch_one(
            "SELECT cancel_operation_id FROM a2a_tasks WHERE id = ?",
            (task_id,),
        )
        return row[0] if row else None

    async def is_task_recipient(self, task_id: str, agent_id: str) -> bool:
        """Whether ``agent_id`` is the durable execution recipient of ``task_id``."""

        if not isinstance(agent_id, str) or not agent_id:
            raise ValueError("Task recipient lookup requires a concrete identity")
        row = await self._backend.fetch_one(
            "SELECT 1 FROM a2a_tasks WHERE id = ? AND recipient_agent_id = ?",
            (task_id, agent_id),
        )
        return row is not None

    async def get_pending_tasks(
        self,
        limit: int = 10,
        *,
        recipient_agent_id: Optional[str] = None,
    ) -> list[Task]:
        """Get submitted work, optionally constrained to one worker recipient."""

        if recipient_agent_id is not None and (
            not isinstance(recipient_agent_id, str)
            or not recipient_agent_id.strip()
        ):
            raise ValueError("Pending-task recipient must be a concrete identity")
        recipient_predicate = (
            " AND recipient_agent_id = ?" if recipient_agent_id is not None else ""
        )
        params = (
            (recipient_agent_id, limit)
            if recipient_agent_id is not None
            else (limit,)
        )
        rows = await self._backend.fetch_all(
            f"""
            SELECT * FROM a2a_tasks
            WHERE status = 'submitted'
            {recipient_predicate}
            ORDER BY created_at ASC
            LIMIT ?
            """,
            params,
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
        if status.state is TaskState.CANCELED:
            raise ValueError(
                "CANCELED is an authorized transition; use cancel_if_authorized"
            )
        if expected_state in {
            TaskState.COMPLETED,
            TaskState.FAILED,
            TaskState.CANCELED,
        }:
            raise ValueError("A terminal A2A lifecycle cannot be replaced")
        message_json = status.message.model_dump_json() if status.message else None
        rows_affected = await self._backend.execute(
            f"""
            UPDATE a2a_tasks
            SET status = ?, message = ?, updated_at = {self.now_sql()}
            WHERE id = ?
              AND recipient_agent_id = ?
              AND status = ?
            """,
            (
                status.state.value,
                message_json,
                task_id,
                recipient_agent_id,
                expected_state.value,
            ),
        )
        return rows_affected == 1

    async def cancel_if_authorized(
        self,
        task_id: str,
        *,
        actor_agent_id: str,
        expected_recipient_agent_id: Optional[str] = None,
        reason: Optional[str] = None,
        task_payload: Optional[Task] = None,
        operation_id: Optional[str] = None,
    ) -> Optional[Task]:
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
        if task_payload is not None and (
            task_payload.id != task_id
            or task_payload.status.state is not TaskState.CANCELED
        ):
            raise ValueError(
                "Cancellation payload must be the canceled form of the same task"
            )

        message = Message(
            role="agent",
            parts=[
                # Actor and reason are also stored in dedicated receipt columns;
                # keeping them in the canonical status message makes the A2A
                # envelope self-describing to existing clients.
                TextPart(
                    text=(
                        f"Task canceled by {actor_agent_id}: {reason}"
                        if reason
                        else f"Task canceled by {actor_agent_id}"
                    )
                )
            ],
        )
        transaction = (
            self._backend.transaction(immediate=True)
            if self.is_sqlite
            else self._backend.transaction()
        )
        recipient_predicate = (
            " AND recipient_agent_id = ?"
            if expected_recipient_agent_id is not None
            else ""
        )
        recipient_values = (
            (expected_recipient_agent_id,)
            if expected_recipient_agent_id is not None
            else ()
        )
        def decode_json(value, default):
            if value is None:
                return default
            if isinstance(value, (list, dict)):
                return value
            return json_loads(value) or default

        def merge_sequence(current_items, payload_items):
            merged = list(current_items)
            for item in payload_items:
                if item not in merged:
                    merged.append(item)
            return merged

        async with transaction:
            lock_suffix = " FOR UPDATE" if self.is_postgres else ""
            current = await self._backend.fetch_one(
                f"SELECT * FROM a2a_tasks WHERE id = ?{lock_suffix}",
                (task_id,),
            )
            if current is None:
                return None

            current_artifacts = decode_json(current[6], [])
            current_history = decode_json(current[7], [])
            current_metadata = decode_json(current[8], {})
            if task_payload is not None:
                payload_artifacts = [
                    artifact.model_dump()
                    for artifact in (task_payload.artifacts or [])
                ]
                payload_history = [
                    item.model_dump() for item in (task_payload.history or [])
                ]
                payload_metadata = dict(task_payload.metadata or {})
                payload_metadata.pop("cancellation_receipt", None)
                merged_artifacts = merge_sequence(
                    current_artifacts, payload_artifacts
                )
                merged_history = merge_sequence(current_history, payload_history)
                merged_metadata = {**current_metadata, **payload_metadata}
            else:
                merged_artifacts = current_artifacts
                merged_history = list(current_history)
                merged_metadata = current_metadata
            merged_history.append(message.model_dump())
            artifacts_json = json_dumps(merged_artifacts)
            history_json = json_dumps(merged_history)
            metadata_json = json_dumps(merged_metadata)
            message_json = message.model_dump_json()
            rows_affected = await self._backend.execute(
                f"""
                UPDATE a2a_tasks
                SET cancel_previous_status = status,
                    status = 'canceled',
                    message = ?,
                    artifacts = ?,
                    history = ?,
                    metadata = ?,
                    canceled_by = ?,
                    cancel_reason = ?,
                    cancel_operation_id = ?,
                    updated_at = {self.now_sql()}
                WHERE id = ?
                  AND status IN ('submitted', 'working', 'input-required')
                  AND (creator_agent_id = ? OR recipient_agent_id = ?)
                  {recipient_predicate}
                """,
                (
                    message_json,
                    artifacts_json,
                    history_json,
                    metadata_json,
                    actor_agent_id,
                    reason,
                    operation_id,
                    task_id,
                    actor_agent_id,
                    actor_agent_id,
                    *recipient_values,
                ),
            )
            if rows_affected != 1:
                return None

            # Materialize the exact committed payload from the locked row and
            # the values written above.  A second read after commit can fail or
            # be canceled after authority has already changed, which would
            # wrongly roll back local intent and abandon every projection.
            canceled_row = list(current)
            canceled_row[4] = TaskState.CANCELED.value
            canceled_row[5] = message_json
            canceled_row[6] = artifacts_json
            canceled_row[7] = history_json
            canceled_row[8] = metadata_json
            canceled_row[13] = actor_agent_id
            canceled_row[14] = reason
            canceled_row[15] = current[4]
            canceled_row[16] = operation_id
            return self._row_to_task(tuple(canceled_row))

    async def add_artifact(
        self,
        task_id: str,
        artifact: Artifact,
        *,
        recipient_agent_id: str,
    ) -> None:
        """Append an artifact without crossing a terminal-state transition.

        Cancellation may merge a handler's partial payload into this same JSON
        column.  The artifact writer therefore takes the same row lock (or the
        SQLite immediate-writer lease) and predicates its update on a live
        status.  Whichever operation wins first is preserved: cancellation
        sees an already-appended artifact, or the later append is refused.
        """

        if not isinstance(recipient_agent_id, str) or not recipient_agent_id.strip():
            raise ValueError("Task artifact recipient must be a concrete identity")

        transaction = (
            self._backend.transaction(immediate=True)
            if self.is_sqlite
            else self._backend.transaction()
        )
        unauthorized = False
        terminal_status: Optional[str] = None
        update_conflict = False
        async with transaction:
            lock_suffix = " FOR UPDATE" if self.is_postgres else ""
            row = await self._backend.fetch_one(
                "SELECT artifacts, status FROM a2a_tasks "
                f"WHERE id = ? AND recipient_agent_id = ?{lock_suffix}",
                (task_id, recipient_agent_id),
            )
            if not row:
                unauthorized = True
            elif row[1] not in {
                TaskState.SUBMITTED.value,
                TaskState.WORKING.value,
                TaskState.INPUT_REQUIRED.value,
            }:
                terminal_status = str(row[1])
            else:
                artifacts = (
                    list(row[0])
                    if isinstance(row[0], list)
                    else json_loads(row[0]) or []
                )
                artifacts.append(artifact.model_dump())

                rows_affected = await self._backend.execute(
                    f"""
                    UPDATE a2a_tasks
                    SET artifacts = ?, updated_at = {self.now_sql()}
                    WHERE id = ?
                      AND recipient_agent_id = ?
                      AND status IN ('submitted', 'working', 'input-required')
                    """,
                    (json_dumps(artifacts), task_id, recipient_agent_id),
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
        if update_conflict:
            raise ValueError(f"Cannot add artifact to terminal task {task_id}")

    async def list_tasks(
        self,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        status: Optional[TaskState] = None,
        limit: int = 100,
    ) -> list[Task]:
        """List tasks with optional filters."""
        conditions = []
        params = []

        if session_id:
            conditions.append("session_id = ?")
            params.append(session_id)
        if user_id:
            conditions.append("user_id = ?")
            params.append(user_id)
        if status:
            conditions.append("status = ?")
            params.append(status.value)

        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        params.append(limit)

        rows = await self._backend.fetch_all(
            f"""
            SELECT * FROM a2a_tasks
            {where}
            ORDER BY created_at DESC
            LIMIT ?
            """,
            tuple(params),
        )
        return [self._row_to_task(row) for row in rows]

    async def delete(self, task_id: str) -> bool:
        """Delete a task. Returns True if deleted."""
        rows_affected = await self._backend.execute(
            "DELETE FROM a2a_tasks WHERE id = ?",
            (task_id,),
        )
        return rows_affected > 0

    async def cleanup_old(
        self,
        older_than_days: int = 30,
        terminal_states: tuple = ("completed", "failed", "canceled"),
    ) -> int:
        """
        Delete old tasks in terminal states.

        Args:
            older_than_days: Delete tasks updated before this many days ago.
            terminal_states: Task states eligible for cleanup.

        Returns:
            Number of tasks deleted.
        """
        cutoff = datetime.now(timezone.utc) - timedelta(days=older_than_days)
        cutoff_str = cutoff.strftime("%Y-%m-%dT%H:%M:%S")

        placeholders = ", ".join("?" for _ in terminal_states)
        params = list(terminal_states) + [cutoff_str]

        rows_affected = await self._backend.execute(
            f"DELETE FROM a2a_tasks WHERE status IN ({placeholders}) AND updated_at < ?",
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
        15: cancel_previous_status, 16: cancel_operation_id
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

        message = None
        if row[5]:
            # Handle both string (SQLite) and dict (PostgreSQL JSONB)
            if isinstance(row[5], dict):
                message = Message.model_validate(row[5])
            else:
                message = Message.model_validate_json(row[5])

        return Task(
            id=row[0],
            sessionId=row[1],
            status=TaskStatus(
                state=TaskState(row[4]),
                message=message,
            ),
            artifacts=[Artifact.model_validate(a) for a in artifacts_data] if artifacts_data else None,
            history=[Message.model_validate(m) for m in history_data] if history_data else None,
            metadata=metadata,
        )
