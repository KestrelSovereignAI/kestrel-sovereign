"""
Unified TaskStore - Backend-Agnostic Task Persistence.

Manages task lifecycle from submission to completion/failure.
Works with both SQLite and PostgreSQL backends.
"""

import logging
from typing import Optional

from kestrel_sovereign.a2a.types import Task, TaskStatus, TaskState, Artifact, Message
from kestrel_sovereign.a2a.stores.base import json_dumps, json_loads
from kestrel_sovereign.a2a.stores.unified.base import UnifiedStoreBase
from kestrel_sovereign.storage.db.interface import DatabaseBackend

logger = logging.getLogger(__name__)


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
                updated_at {ts_type} {ts_default}
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

        logger.info(f"TaskStore initialized ({self._backend.backend_type})")

    async def save(self, task: Task) -> None:
        """Save or update a task."""
        message_json = task.status.message.model_dump_json() if task.status.message else None
        artifacts_json = json_dumps([a.model_dump() for a in (task.artifacts or [])])
        history_json = json_dumps([m.model_dump() for m in (task.history or [])])
        metadata_json = json_dumps(task.metadata or {})
        task_type = task.metadata.get("task_type", "generic") if task.metadata else "generic"
        user_id = task.metadata.get("user_id") if task.metadata else None

        if self.is_postgres:
            # PostgreSQL: Use ON CONFLICT for upsert
            await self._backend.execute(
                f"""
                INSERT INTO a2a_tasks
                (id, session_id, user_id, task_type, status, message, artifacts, history, metadata, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, {self.now_sql()})
                ON CONFLICT (id) DO UPDATE SET
                    status = EXCLUDED.status,
                    message = EXCLUDED.message,
                    artifacts = EXCLUDED.artifacts,
                    history = EXCLUDED.history,
                    metadata = EXCLUDED.metadata,
                    updated_at = {self.now_sql()}
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
                ),
            )
        else:
            # SQLite: Use INSERT OR REPLACE
            await self._backend.execute(
                f"""
                INSERT OR REPLACE INTO a2a_tasks
                (id, session_id, user_id, task_type, status, message, artifacts, history, metadata, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, {self.now_sql()})
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
                ),
            )

    async def get(self, task_id: str) -> Optional[Task]:
        """Retrieve a task by ID."""
        row = await self._backend.fetch_one(
            "SELECT * FROM a2a_tasks WHERE id = ?",
            (task_id,),
        )
        if not row:
            return None
        return self._row_to_task(row)

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

    async def update_status(self, task_id: str, status: TaskStatus) -> None:
        """Update task status."""
        message_json = status.message.model_dump_json() if status.message else None
        await self._backend.execute(
            f"""
            UPDATE a2a_tasks
            SET status = ?, message = ?, updated_at = {self.now_sql()}
            WHERE id = ?
            """,
            (status.state.value, message_json, task_id),
        )

    async def add_artifact(self, task_id: str, artifact: Artifact) -> None:
        """Add artifact to task."""
        # Get current artifacts
        row = await self._backend.fetch_one(
            "SELECT artifacts FROM a2a_tasks WHERE id = ?",
            (task_id,),
        )
        if not row:
            raise ValueError(f"Task not found: {task_id}")

        artifacts = json_loads(row[0]) or []
        artifacts.append(artifact.model_dump())

        await self._backend.execute(
            f"""
            UPDATE a2a_tasks
            SET artifacts = ?, updated_at = {self.now_sql()}
            WHERE id = ?
            """,
            (json_dumps(artifacts), task_id),
        )

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

    def _row_to_task(self, row: tuple) -> Task:
        """
        Convert database row to Task object.

        Row columns (in order):
        0: id, 1: session_id, 2: user_id, 3: task_type, 4: status,
        5: message, 6: artifacts, 7: history, 8: metadata,
        9: created_at, 10: updated_at
        """
        artifacts_data = json_loads(row[6]) if row[6] else []
        history_data = json_loads(row[7]) if row[7] else []
        metadata = json_loads(row[8]) if row[8] else {}

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
