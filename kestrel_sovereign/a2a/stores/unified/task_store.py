"""
Unified TaskStore - Backend-Agnostic Task Persistence.

Manages task lifecycle from submission to completion/failure.
Works with both SQLite and PostgreSQL backends.
"""

import logging
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


class TaskAlreadyExistsError(ValueError):
    """A caller attempted to create a second task under an occupied ID."""


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
                cancel_previous_status TEXT
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

        # Durable task authority and cancellation receipt (#3134).  These are
        # separate columns rather than caller-controlled metadata so a shared
        # PostgreSQL table has the same security boundary as per-agent SQLite.
        authority_columns = (
            "creator_agent_id",
            "recipient_agent_id",
            "canceled_by",
            "cancel_reason",
            "cancel_previous_status",
        )
        for column in authority_columns:
            await self.add_column_if_missing("a2a_tasks", column, "TEXT")

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

        logger.info(f"TaskStore initialized ({self._backend.backend_type})")

    async def save(
        self,
        task: Task,
        *,
        creator_agent_id: Optional[str] = None,
        recipient_agent_id: Optional[str] = None,
    ) -> bool:
        """Create an authoritative task or persist a lifecycle update.

        Supplying either authority argument denotes initial creation and is
        insert-only. Lifecycle saves omit both arguments and may update the
        existing payload without ever touching authority columns.
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

        if task.status.state is TaskState.CANCELED:
            raise ValueError(
                "CANCELED is an authorized transition; use cancel_if_authorized"
            )

        message_json = task.status.message.model_dump_json() if task.status.message else None
        artifacts_json = json_dumps([a.model_dump() for a in (task.artifacts or [])])
        history_json = json_dumps([m.model_dump() for m in (task.history or [])])
        metadata_json = json_dumps(task.metadata or {})

        rows_affected = await self._backend.execute(
            f"""
            UPDATE a2a_tasks SET
                status = ?,
                message = ?,
                artifacts = ?,
                history = ?,
                metadata = ?,
                updated_at = {self.now_sql()}
            WHERE id = ? AND status <> 'canceled'
            """,
            (
                task.status.state.value,
                message_json,
                artifacts_json,
                history_json,
                metadata_json,
                task.id,
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
        message_json = (
            task.status.message.model_dump_json() if task.status.message else None
        )
        artifacts_json = json_dumps([a.model_dump() for a in (task.artifacts or [])])
        history_json = json_dumps([m.model_dump() for m in (task.history or [])])
        metadata = dict(task.metadata or {})
        # Cancellation receipts are reserved durable state. A sender may put
        # arbitrary metadata on a new envelope, but only the authorized atomic
        # cancellation transition below may mint this field.
        metadata.pop("cancellation_receipt", None)
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

    async def is_task_recipient(self, task_id: str, agent_id: str) -> bool:
        """Whether ``agent_id`` is the durable execution recipient of ``task_id``."""

        if not isinstance(agent_id, str) or not agent_id:
            raise ValueError("Task recipient lookup requires a concrete identity")
        row = await self._backend.fetch_one(
            "SELECT 1 FROM a2a_tasks WHERE id = ? AND recipient_agent_id = ?",
            (task_id, agent_id),
        )
        return row is not None

    async def get_pending_tasks(self, limit: int = 10) -> list[Task]:
        """Get tasks ready for processing (SUBMITTED state)."""
        rows = await self._backend.fetch_all(
            """
            SELECT * FROM a2a_tasks
            WHERE status = 'submitted'
            ORDER BY created_at ASC
            LIMIT ?
            """,
            (limit,),
        )
        return [self._row_to_task(row) for row in rows]

    async def update_status(self, task_id: str, status: TaskStatus) -> bool:
        """Update a live task status without overwriting cancellation."""
        if status.state is TaskState.CANCELED:
            raise ValueError(
                "CANCELED is an authorized transition; use cancel_if_authorized"
            )
        message_json = status.message.model_dump_json() if status.message else None
        rows_affected = await self._backend.execute(
            f"""
            UPDATE a2a_tasks
            SET status = ?, message = ?, updated_at = {self.now_sql()}
            WHERE id = ? AND status <> 'canceled'
            """,
            (status.state.value, message_json, task_id),
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
        if task_payload is not None:
            transaction = (
                self._backend.transaction(immediate=True)
                if self.is_sqlite
                else self._backend.transaction()
            )
            async with transaction:
                lock_suffix = " FOR UPDATE" if self.is_postgres else ""
                current = await self._backend.fetch_one(
                    "SELECT artifacts, history, metadata FROM a2a_tasks "
                    f"WHERE id = ?{lock_suffix}",
                    (task_id,),
                )
                if current is None:
                    return None

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

                current_artifacts = decode_json(current[0], [])
                current_history = decode_json(current[1], [])
                current_metadata = decode_json(current[2], {})
                payload_artifacts = [
                    artifact.model_dump()
                    for artifact in (task_payload.artifacts or [])
                ]
                payload_history = [
                    item.model_dump() for item in (task_payload.history or [])
                ]
                payload_metadata = dict(task_payload.metadata or {})
                payload_metadata.pop("cancellation_receipt", None)
                merged_metadata = {
                    **current_metadata,
                    **payload_metadata,
                }
                merged_history = merge_sequence(
                    current_history,
                    payload_history,
                )
                merged_history.append(message.model_dump())
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
                        updated_at = {self.now_sql()}
                    WHERE id = ?
                      AND status IN ('submitted', 'working', 'input-required')
                      AND (creator_agent_id = ? OR recipient_agent_id = ?)
                      {recipient_predicate}
                    """,
                    (
                        message.model_dump_json(),
                        json_dumps(
                            merge_sequence(current_artifacts, payload_artifacts)
                        ),
                        json_dumps(merged_history),
                        json_dumps(merged_metadata),
                        actor_agent_id,
                        reason,
                        task_id,
                        actor_agent_id,
                        actor_agent_id,
                        *recipient_values,
                    ),
                )
            if rows_affected != 1:
                return None
            return await self.get(task_id)

        if self.is_postgres:
            payload_assignment = (
                "history = COALESCE(history, '[]'::jsonb) || ?::jsonb,"
            )
            payload_values = (json_dumps([message.model_dump()]),)
        else:
            payload_assignment = (
                "history = json_insert(COALESCE(history, '[]'), '$[#]', json(?)),"
            )
            payload_values = (message.model_dump_json(),)
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
        rows_affected = await self._backend.execute(
            f"""
            UPDATE a2a_tasks
            SET cancel_previous_status = status,
                status = 'canceled',
                message = ?,
                {payload_assignment}
                canceled_by = ?,
                cancel_reason = ?,
                updated_at = {self.now_sql()}
            WHERE id = ?
              AND status IN ('submitted', 'working', 'input-required')
              AND (creator_agent_id = ? OR recipient_agent_id = ?)
              {recipient_predicate}
            """,
            (
                message.model_dump_json(),
                *payload_values,
                actor_agent_id,
                reason,
                task_id,
                actor_agent_id,
                actor_agent_id,
                *recipient_values,
            ),
        )
        if rows_affected != 1:
            return None
        return await self.get(task_id)

    async def add_artifact(self, task_id: str, artifact: Artifact) -> None:
        """Append an artifact without crossing a terminal-state transition.

        Cancellation may merge a handler's partial payload into this same JSON
        column.  The artifact writer therefore takes the same row lock (or the
        SQLite immediate-writer lease) and predicates its update on a live
        status.  Whichever operation wins first is preserved: cancellation
        sees an already-appended artifact, or the later append is refused.
        """

        transaction = (
            self._backend.transaction(immediate=True)
            if self.is_sqlite
            else self._backend.transaction()
        )
        terminal_status: Optional[str] = None
        update_conflict = False
        async with transaction:
            lock_suffix = " FOR UPDATE" if self.is_postgres else ""
            row = await self._backend.fetch_one(
                "SELECT artifacts, status FROM a2a_tasks "
                f"WHERE id = ?{lock_suffix}",
                (task_id,),
            )
            if not row:
                raise ValueError(f"Task not found: {task_id}")
            if row[1] not in {
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
                      AND status IN ('submitted', 'working', 'input-required')
                    """,
                    (json_dumps(artifacts), task_id),
                )
                update_conflict = rows_affected != 1
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
        15: cancel_previous_status
        """
        artifacts_data = json_loads(row[6]) if row[6] else []
        history_data = json_loads(row[7]) if row[7] else []
        metadata = json_loads(row[8]) if row[8] else {}
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
