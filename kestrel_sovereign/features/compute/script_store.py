"""
Kestrel Compute Feature - Script Store.

SQLite-backed persistence for compute scripts with full lifecycle tracking.
"""

import aiosqlite
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from .models import ComputeScript, ExecutionRecord, ScriptState, SecurityFinding

logger = logging.getLogger(__name__)


class ScriptStore:
    """
    SQLite-backed storage for compute scripts.
    
    Provides CRUD operations and queries for ComputeScript and ExecutionRecord.
    Scripts persist across agent restarts.
    
    Example:
        store = ScriptStore("kestrel_prime.db")
        await store.initialize()
        
        # Save a new script
        await store.save(script)
        
        # Find by ID prefix
        script = await store.find_by_id_prefix("abc123")
        
        # List pending scripts
        pending = await store.list_by_state(ScriptState.PENDING_REVIEW)
    """
    
    def __init__(self, db_path: str):
        """
        Initialize the script store.
        
        Args:
            db_path: Path to the SQLite database file
        """
        self.db_path = db_path
        self._initialized = False
    
    async def initialize(self) -> None:
        """Create database tables if they don't exist."""
        if self._initialized:
            return
        
        # Ensure directory exists
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        
        async with aiosqlite.connect(self.db_path) as db:
            # Scripts table
            await db.execute("""
                CREATE TABLE IF NOT EXISTS compute_scripts (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    language TEXT NOT NULL,
                    content TEXT NOT NULL,
                    purpose TEXT NOT NULL,
                    state TEXT NOT NULL DEFAULT 'draft',
                    signature TEXT,
                    signed_by TEXT,
                    signed_at TIMESTAMP,
                    security_findings TEXT,
                    risk_score INTEGER DEFAULT 0,
                    review_notes TEXT,
                    execution_id TEXT,
                    timeout_seconds INTEGER DEFAULT 300,
                    environment TEXT,
                    requirements TEXT,
                    parent_task_id TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Execution records table
            await db.execute("""
                CREATE TABLE IF NOT EXISTS compute_executions (
                    id TEXT PRIMARY KEY,
                    script_id TEXT NOT NULL,
                    started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    completed_at TIMESTAMP,
                    exit_code INTEGER,
                    stdout TEXT,
                    stderr TEXT,
                    executor TEXT DEFAULT 'uv',
                    container_id TEXT,
                    resource_usage TEXT,
                    dry_run INTEGER DEFAULT 0,
                    workdir TEXT,
                    FOREIGN KEY (script_id) REFERENCES compute_scripts(id)
                )
            """)
            
            # Indexes for efficient queries
            await db.execute("""
                CREATE INDEX IF NOT EXISTS idx_scripts_state
                ON compute_scripts(state)
            """)
            await db.execute("""
                CREATE INDEX IF NOT EXISTS idx_scripts_created
                ON compute_scripts(created_at DESC)
            """)
            await db.execute("""
                CREATE INDEX IF NOT EXISTS idx_executions_script
                ON compute_executions(script_id)
            """)
            await db.execute("""
                CREATE INDEX IF NOT EXISTS idx_executions_started
                ON compute_executions(started_at DESC)
            """)
            
            await db.commit()
        
        self._initialized = True
        logger.info("ScriptStore initialized")
    
    async def save(self, script: ComputeScript) -> None:
        """
        Save a new script to the database.
        
        Args:
            script: The ComputeScript to save
        """
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                INSERT INTO compute_scripts (
                    id, name, language, content, purpose, state,
                    signature, signed_by, signed_at,
                    security_findings, risk_score, review_notes,
                    execution_id, timeout_seconds, environment, requirements,
                    parent_task_id, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                script.id,
                script.name,
                script.language,
                script.content,
                script.purpose,
                script.state.value,
                script.signature,
                script.signed_by,
                script.signed_at.isoformat() if script.signed_at else None,
                json.dumps([f.to_dict() for f in script.security_findings]),
                script.risk_score,
                script.review_notes,
                script.execution_id,
                script.timeout_seconds,
                json.dumps(script.environment),
                json.dumps(script.requirements),
                script.parent_task_id,
                script.created_at.isoformat(),
                script.updated_at.isoformat(),
            ))
            await db.commit()
        
        logger.debug(f"Saved script {script.id[:8]}... ({script.name})")
    
    async def update(self, script: ComputeScript) -> None:
        """
        Update an existing script in the database.
        
        Args:
            script: The ComputeScript to update
        """
        script.updated_at = datetime.now()
        
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                UPDATE compute_scripts SET
                    name = ?, language = ?, content = ?, purpose = ?, state = ?,
                    signature = ?, signed_by = ?, signed_at = ?,
                    security_findings = ?, risk_score = ?, review_notes = ?,
                    execution_id = ?, timeout_seconds = ?, environment = ?, requirements = ?,
                    parent_task_id = ?, updated_at = ?
                WHERE id = ?
            """, (
                script.name,
                script.language,
                script.content,
                script.purpose,
                script.state.value,
                script.signature,
                script.signed_by,
                script.signed_at.isoformat() if script.signed_at else None,
                json.dumps([f.to_dict() for f in script.security_findings]),
                script.risk_score,
                script.review_notes,
                script.execution_id,
                script.timeout_seconds,
                json.dumps(script.environment),
                json.dumps(script.requirements),
                script.parent_task_id,
                script.updated_at.isoformat(),
                script.id,
            ))
            await db.commit()
        
        logger.debug(f"Updated script {script.id[:8]}... (state={script.state.value})")
    
    async def get(self, script_id: str) -> Optional[ComputeScript]:
        """
        Get a script by exact ID.
        
        Args:
            script_id: The full script ID
            
        Returns:
            ComputeScript if found, None otherwise
        """
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                "SELECT * FROM compute_scripts WHERE id = ?",
                (script_id,)
            )
            row = await cursor.fetchone()
            
            if row:
                return self._row_to_script(row)
            return None
    
    async def find_by_id_prefix(self, prefix: str) -> Optional[ComputeScript]:
        """
        Find a script by ID prefix (for convenience).
        
        Args:
            prefix: ID prefix (e.g., first 8 characters)
            
        Returns:
            ComputeScript if found and unique, None otherwise
        """
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                "SELECT * FROM compute_scripts WHERE id LIKE ?",
                (f"{prefix}%",)
            )
            rows = await cursor.fetchall()
            
            if len(rows) == 1:
                return self._row_to_script(rows[0])
            elif len(rows) > 1:
                logger.warning(f"Multiple scripts match prefix '{prefix}'")
            return None
    
    async def list_by_state(
        self,
        state: ScriptState,
        limit: int = 100,
    ) -> List[ComputeScript]:
        """
        List scripts by state.
        
        Args:
            state: The script state to filter by
            limit: Maximum number of results
            
        Returns:
            List of matching ComputeScripts
        """
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                "SELECT * FROM compute_scripts WHERE state = ? ORDER BY created_at DESC LIMIT ?",
                (state.value, limit)
            )
            rows = await cursor.fetchall()
            return [self._row_to_script(row) for row in rows]
    
    async def list_recent(self, limit: int = 50) -> List[ComputeScript]:
        """
        List recent scripts regardless of state.
        
        Args:
            limit: Maximum number of results
            
        Returns:
            List of recent ComputeScripts
        """
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                "SELECT * FROM compute_scripts ORDER BY created_at DESC LIMIT ?",
                (limit,)
            )
            rows = await cursor.fetchall()
            return [self._row_to_script(row) for row in rows]
    
    async def delete(self, script_id: str) -> bool:
        """
        Delete a script (and its execution records).
        
        Args:
            script_id: The script ID to delete
            
        Returns:
            True if deleted, False if not found
        """
        async with aiosqlite.connect(self.db_path) as db:
            # Delete execution records first
            await db.execute(
                "DELETE FROM compute_executions WHERE script_id = ?",
                (script_id,)
            )
            
            cursor = await db.execute(
                "DELETE FROM compute_scripts WHERE id = ?",
                (script_id,)
            )
            await db.commit()
            
            deleted = cursor.rowcount > 0
            if deleted:
                logger.info(f"Deleted script {script_id[:8]}...")
            return deleted
    
    # === Execution Records ===
    
    async def save_execution(self, record: ExecutionRecord) -> None:
        """
        Save an execution record.
        
        Args:
            record: The ExecutionRecord to save
        """
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                INSERT INTO compute_executions (
                    id, script_id, started_at, completed_at,
                    exit_code, stdout, stderr, executor,
                    container_id, resource_usage, dry_run, workdir
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                record.id,
                record.script_id,
                record.started_at.isoformat(),
                record.completed_at.isoformat() if record.completed_at else None,
                record.exit_code,
                record.stdout,
                record.stderr,
                record.executor,
                record.container_id,
                json.dumps(record.resource_usage),
                1 if record.dry_run else 0,
                record.workdir,
            ))
            await db.commit()
        
        logger.debug(f"Saved execution {record.id[:8]}... for script {record.script_id[:8]}...")
    
    async def update_execution(self, record: ExecutionRecord) -> None:
        """
        Update an execution record.
        
        Args:
            record: The ExecutionRecord to update
        """
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                UPDATE compute_executions SET
                    completed_at = ?, exit_code = ?,
                    stdout = ?, stderr = ?,
                    container_id = ?, resource_usage = ?
                WHERE id = ?
            """, (
                record.completed_at.isoformat() if record.completed_at else None,
                record.exit_code,
                record.stdout,
                record.stderr,
                record.container_id,
                json.dumps(record.resource_usage),
                record.id,
            ))
            await db.commit()
    
    async def get_execution(self, execution_id: str) -> Optional[ExecutionRecord]:
        """
        Get an execution record by ID.
        
        Args:
            execution_id: The execution ID
            
        Returns:
            ExecutionRecord if found, None otherwise
        """
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                "SELECT * FROM compute_executions WHERE id = ?",
                (execution_id,)
            )
            row = await cursor.fetchone()
            
            if row:
                return self._row_to_execution(row)
            return None
    
    async def get_executions_for_script(
        self,
        script_id: str,
        limit: int = 10,
    ) -> List[ExecutionRecord]:
        """
        Get execution history for a script.
        
        Args:
            script_id: The script ID
            limit: Maximum number of results
            
        Returns:
            List of ExecutionRecords, newest first
        """
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                """SELECT * FROM compute_executions 
                   WHERE script_id = ? 
                   ORDER BY started_at DESC 
                   LIMIT ?""",
                (script_id, limit)
            )
            rows = await cursor.fetchall()
            return [self._row_to_execution(row) for row in rows]
    
    async def list_recent_executions(self, limit: int = 50) -> List[ExecutionRecord]:
        """
        List recent executions across all scripts.
        
        Args:
            limit: Maximum number of results
            
        Returns:
            List of recent ExecutionRecords
        """
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                "SELECT * FROM compute_executions ORDER BY started_at DESC LIMIT ?",
                (limit,)
            )
            rows = await cursor.fetchall()
            return [self._row_to_execution(row) for row in rows]
    
    # === Helper Methods ===
    
    def _row_to_script(self, row: aiosqlite.Row) -> ComputeScript:
        """Convert a database row to a ComputeScript."""
        findings_data = json.loads(row["security_findings"] or "[]")
        findings = [SecurityFinding.from_dict(f) for f in findings_data]
        
        return ComputeScript(
            id=row["id"],
            name=row["name"],
            language=row["language"],
            content=row["content"],
            purpose=row["purpose"],
            state=ScriptState(row["state"]),
            signature=row["signature"],
            signed_by=row["signed_by"],
            signed_at=datetime.fromisoformat(row["signed_at"]) if row["signed_at"] else None,
            security_findings=findings,
            risk_score=row["risk_score"],
            review_notes=row["review_notes"],
            execution_id=row["execution_id"],
            timeout_seconds=row["timeout_seconds"],
            environment=json.loads(row["environment"] or "{}"),
            requirements=json.loads(row["requirements"] or "[]"),
            parent_task_id=row["parent_task_id"],
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
        )
    
    def _row_to_execution(self, row: aiosqlite.Row) -> ExecutionRecord:
        """Convert a database row to an ExecutionRecord."""
        return ExecutionRecord(
            id=row["id"],
            script_id=row["script_id"],
            started_at=datetime.fromisoformat(row["started_at"]),
            completed_at=datetime.fromisoformat(row["completed_at"]) if row["completed_at"] else None,
            exit_code=row["exit_code"],
            stdout=row["stdout"] or "",
            stderr=row["stderr"] or "",
            executor=row["executor"],
            container_id=row["container_id"],
            resource_usage=json.loads(row["resource_usage"] or "{}"),
            dry_run=bool(row["dry_run"]),
            workdir=row["workdir"],
        )
