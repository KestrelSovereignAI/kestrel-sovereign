"""
Unified OrchestrationStore - Backend-Agnostic Workflow Coordination.

Enables complex multi-agent workflows:
- Task delegation between agents
- Workflow step tracking
- Parent-child task relationships
- Status propagation across workflow

Works with both SQLite and PostgreSQL backends.
"""

import logging
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel

from kestrel_sovereign.a2a.stores.base import generate_id, json_dumps, json_loads
from kestrel_sovereign.a2a.stores.unified.base import UnifiedStoreBase
from kestrel_sovereign.storage.db.interface import DatabaseBackend

logger = logging.getLogger(__name__)


class OrchestrationStatus(str, Enum):
    """Status of an orchestration task."""

    PENDING = "pending"
    DELEGATED = "delegated"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELED = "canceled"


class OrchestrationTask(BaseModel):
    """A task within a multi-agent workflow."""

    task_id: str
    workflow_id: str
    step_number: int
    agent_name: str
    delegated_to: Optional[str]  # Agent this was delegated to
    status: OrchestrationStatus
    input_data: dict[str, Any]
    output_data: Optional[dict[str, Any]]
    parent_task_id: Optional[str]  # For sub-task hierarchies
    created_at: Any  # datetime
    updated_at: Any  # datetime
    completed_at: Optional[Any]  # datetime
    metadata: dict[str, Any]


class OrchestrationStore(UnifiedStoreBase):
    """
    Backend-agnostic orchestration store.

    Replaces both SQLiteOrchestrationStore and PostgresOrchestrationStore
    with a single implementation that works with any DatabaseBackend.
    """

    def __init__(self, backend: DatabaseBackend):
        """
        Initialize orchestration store with database backend.

        Args:
            backend: DatabaseBackend instance (SQLite or PostgreSQL)
        """
        super().__init__(backend)

    async def initialize(self) -> None:
        """Create tables if not exists."""
        ts_type = self.timestamp_type()
        ts_default = self.now_default()
        json_type = self.json_type()

        # Workflows table
        await self._backend.execute_script(f"""
            CREATE TABLE IF NOT EXISTS a2a_workflows (
                id TEXT PRIMARY KEY,
                created_at {ts_type} {ts_default},
                metadata {json_type} DEFAULT '{{}}'
            )
        """)

        # Orchestration tasks table
        # Note: FK syntax is the same for both backends
        await self._backend.execute_script(f"""
            CREATE TABLE IF NOT EXISTS a2a_orchestration (
                id TEXT PRIMARY KEY,
                workflow_id TEXT NOT NULL,
                step_number INTEGER NOT NULL,
                agent_name TEXT NOT NULL,
                delegated_to TEXT,
                status TEXT DEFAULT 'pending',
                input_data {json_type} DEFAULT '{{}}',
                output_data {json_type},
                parent_task_id TEXT,
                created_at {ts_type} {ts_default},
                updated_at {ts_type} {ts_default},
                completed_at {ts_type},
                metadata {json_type} DEFAULT '{{}}',
                FOREIGN KEY (workflow_id) REFERENCES a2a_workflows(id) ON DELETE CASCADE,
                FOREIGN KEY (parent_task_id) REFERENCES a2a_orchestration(id)
            )
        """)

        await self._backend.execute(
            "CREATE INDEX IF NOT EXISTS idx_orch_workflow ON a2a_orchestration(workflow_id)"
        )
        await self._backend.execute(
            "CREATE INDEX IF NOT EXISTS idx_orch_agent ON a2a_orchestration(agent_name, status)"
        )
        await self._backend.execute(
            "CREATE INDEX IF NOT EXISTS idx_orch_delegated ON a2a_orchestration(delegated_to, status)"
        )
        await self._backend.execute(
            "CREATE INDEX IF NOT EXISTS idx_orch_parent ON a2a_orchestration(parent_task_id)"
        )

        logger.info(f"OrchestrationStore initialized ({self._backend.backend_type})")

    async def create_workflow(
        self,
        workflow_id: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> str:
        """Create a new workflow. Returns workflow_id."""
        wf_id = workflow_id or generate_id()
        now = self.now_utc_param()

        await self._backend.execute(
            """
            INSERT INTO a2a_workflows (id, created_at, metadata)
            VALUES (?, ?, ?)
            """,
            (wf_id, now, json_dumps(metadata or {})),
        )

        return wf_id

    async def add_task(
        self,
        workflow_id: str,
        agent_name: str,
        step_number: int,
        input_data: dict[str, Any],
        parent_task_id: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> str:
        """Add a task to a workflow. Returns task_id."""
        task_id = generate_id()
        now = self.now_utc_param()

        await self._backend.execute(
            """
            INSERT INTO a2a_orchestration
            (id, workflow_id, step_number, agent_name, input_data, parent_task_id, created_at, updated_at, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                task_id,
                workflow_id,
                step_number,
                agent_name,
                json_dumps(input_data),
                parent_task_id,
                now,
                now,
                json_dumps(metadata or {}),
            ),
        )

        return task_id

    async def delegate_task(
        self,
        task_id: str,
        delegated_to: str,
    ) -> None:
        """Delegate a task to another agent."""
        await self._backend.execute(
            f"""
            UPDATE a2a_orchestration
            SET delegated_to = ?, status = 'delegated', updated_at = {self.now_sql()}
            WHERE id = ?
            """,
            (delegated_to, task_id),
        )

    async def update_task_status(
        self,
        task_id: str,
        status: OrchestrationStatus,
        output_data: Optional[dict[str, Any]] = None,
    ) -> None:
        """Update task status and optionally set output."""
        is_terminal = status in (
            OrchestrationStatus.COMPLETED,
            OrchestrationStatus.FAILED,
            OrchestrationStatus.CANCELED,
        )

        if output_data is not None:
            if is_terminal:
                await self._backend.execute(
                    f"""
                    UPDATE a2a_orchestration
                    SET status = ?, output_data = ?, updated_at = {self.now_sql()}, completed_at = {self.now_sql()}
                    WHERE id = ?
                    """,
                    (status.value, json_dumps(output_data), task_id),
                )
            else:
                await self._backend.execute(
                    f"""
                    UPDATE a2a_orchestration
                    SET status = ?, output_data = ?, updated_at = {self.now_sql()}
                    WHERE id = ?
                    """,
                    (status.value, json_dumps(output_data), task_id),
                )
        else:
            if is_terminal:
                await self._backend.execute(
                    f"""
                    UPDATE a2a_orchestration
                    SET status = ?, updated_at = {self.now_sql()}, completed_at = {self.now_sql()}
                    WHERE id = ?
                    """,
                    (status.value, task_id),
                )
            else:
                await self._backend.execute(
                    f"""
                    UPDATE a2a_orchestration
                    SET status = ?, updated_at = {self.now_sql()}
                    WHERE id = ?
                    """,
                    (status.value, task_id),
                )

    async def get_task(self, task_id: str) -> Optional[OrchestrationTask]:
        """Get a specific task by ID."""
        row = await self._backend.fetch_one(
            "SELECT * FROM a2a_orchestration WHERE id = ?",
            (task_id,),
        )
        if not row:
            return None
        return self._row_to_task(row)

    async def get_workflow_tasks(
        self,
        workflow_id: str,
        status: Optional[OrchestrationStatus] = None,
    ) -> list[OrchestrationTask]:
        """Get all tasks in a workflow, optionally filtered by status."""
        if status:
            rows = await self._backend.fetch_all(
                """
                SELECT * FROM a2a_orchestration
                WHERE workflow_id = ? AND status = ?
                ORDER BY step_number
                """,
                (workflow_id, status.value),
            )
        else:
            rows = await self._backend.fetch_all(
                """
                SELECT * FROM a2a_orchestration
                WHERE workflow_id = ?
                ORDER BY step_number
                """,
                (workflow_id,),
            )
        return [self._row_to_task(row) for row in rows]

    async def get_agent_tasks(
        self,
        agent_name: str,
        status: Optional[OrchestrationStatus] = None,
        limit: int = 100,
    ) -> list[OrchestrationTask]:
        """Get tasks assigned to or delegated to an agent."""
        conditions = ["(agent_name = ? OR delegated_to = ?)"]
        params: list[Any] = [agent_name, agent_name]

        if status:
            conditions.append("status = ?")
            params.append(status.value)

        params.append(limit)

        rows = await self._backend.fetch_all(
            f"""
            SELECT * FROM a2a_orchestration
            WHERE {' AND '.join(conditions)}
            ORDER BY created_at DESC
            LIMIT ?
            """,
            tuple(params),
        )
        return [self._row_to_task(row) for row in rows]

    async def get_pending_delegations(
        self,
        agent_name: str,
    ) -> list[OrchestrationTask]:
        """Get tasks delegated to this agent that are pending."""
        rows = await self._backend.fetch_all(
            """
            SELECT * FROM a2a_orchestration
            WHERE delegated_to = ? AND status = 'delegated'
            ORDER BY created_at
            """,
            (agent_name,),
        )
        return [self._row_to_task(row) for row in rows]

    async def get_child_tasks(
        self,
        parent_task_id: str,
    ) -> list[OrchestrationTask]:
        """Get all child tasks of a parent task."""
        rows = await self._backend.fetch_all(
            """
            SELECT * FROM a2a_orchestration
            WHERE parent_task_id = ?
            ORDER BY step_number
            """,
            (parent_task_id,),
        )
        return [self._row_to_task(row) for row in rows]

    async def is_workflow_complete(self, workflow_id: str) -> bool:
        """Check if all tasks in a workflow are complete (or failed/canceled)."""
        row = await self._backend.fetch_one(
            """
            SELECT COUNT(*) as total,
                   SUM(CASE WHEN status IN ('completed', 'failed', 'canceled') THEN 1 ELSE 0 END) as done
            FROM a2a_orchestration
            WHERE workflow_id = ?
            """,
            (workflow_id,),
        )
        if not row or row[0] == 0:
            return True  # Empty workflow is complete
        return row[0] == row[1]

    async def delete_workflow(self, workflow_id: str) -> int:
        """Delete a workflow and all its tasks. Returns count deleted."""
        # Delete tasks first (FK constraint)
        rows_affected = await self._backend.execute(
            "DELETE FROM a2a_orchestration WHERE workflow_id = ?",
            (workflow_id,),
        )
        # Delete workflow
        await self._backend.execute(
            "DELETE FROM a2a_workflows WHERE id = ?",
            (workflow_id,),
        )
        return rows_affected

    def _row_to_task(self, row: tuple) -> OrchestrationTask:
        """
        Convert database row to OrchestrationTask object.

        Row columns (in order):
        0: id, 1: workflow_id, 2: step_number, 3: agent_name, 4: delegated_to,
        5: status, 6: input_data, 7: output_data, 8: parent_task_id,
        9: created_at, 10: updated_at, 11: completed_at, 12: metadata
        """
        return OrchestrationTask(
            task_id=row[0],
            workflow_id=row[1],
            step_number=row[2],
            agent_name=row[3],
            delegated_to=row[4],
            status=OrchestrationStatus(row[5]),
            input_data=json_loads(row[6]) if row[6] else {},
            output_data=json_loads(row[7]) if row[7] else None,
            parent_task_id=row[8],
            created_at=self.from_timestamp_field(row[9]),
            updated_at=self.from_timestamp_field(row[10]),
            completed_at=self.from_timestamp_field(row[11]) if row[11] else None,
            metadata=json_loads(row[12]) if row[12] else {},
        )
