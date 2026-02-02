"""
Unified SessionService - Backend-Agnostic Session Management.

Implements the Google ADK pattern for session management:
- Session state persists across requests
- Event history tracks all interactions
- Supports session metadata for user context

Works with both SQLite and PostgreSQL backends.
"""

import logging
from typing import Any, Optional

from pydantic import BaseModel

from kestrel_sovereign.a2a.stores.base import generate_id, json_dumps, json_loads
from kestrel_sovereign.a2a.stores.unified.base import UnifiedStoreBase
from kestrel_sovereign.storage.db.interface import DatabaseBackend

logger = logging.getLogger(__name__)


class SessionState(BaseModel):
    """A session with its state and event history."""

    session_id: str
    agent_name: str
    user_id: Optional[str]
    state: dict[str, Any]  # Persistent state across requests
    events: list[dict[str, Any]]  # Chronological event history
    created_at: Any  # datetime
    updated_at: Any  # datetime
    metadata: dict[str, Any]


class SessionService(UnifiedStoreBase):
    """
    Backend-agnostic session service.

    Replaces both SQLiteSessionService and PostgresSessionService with a single
    implementation that works with any DatabaseBackend.
    """

    def __init__(self, backend: DatabaseBackend):
        """
        Initialize session service with database backend.

        Args:
            backend: DatabaseBackend instance (SQLite or PostgreSQL)
        """
        super().__init__(backend)

    async def initialize(self) -> None:
        """Create tables if not exists."""
        ts_type = self.timestamp_type()
        ts_default = self.now_default()
        json_type = self.json_type()

        await self._backend.execute_script(f"""
            CREATE TABLE IF NOT EXISTS a2a_sessions (
                id TEXT PRIMARY KEY,
                agent_name TEXT NOT NULL,
                user_id TEXT,
                state {json_type} DEFAULT '{{}}',
                events {json_type} DEFAULT '[]',
                metadata {json_type} DEFAULT '{{}}',
                created_at {ts_type} {ts_default},
                updated_at {ts_type} {ts_default}
            )
        """)

        await self._backend.execute(
            "CREATE INDEX IF NOT EXISTS idx_sessions_agent ON a2a_sessions(agent_name)"
        )
        await self._backend.execute(
            "CREATE INDEX IF NOT EXISTS idx_sessions_user ON a2a_sessions(user_id)"
        )

        logger.info(f"SessionService initialized ({self._backend.backend_type})")

    async def create_session(
        self,
        agent_name: str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> str:
        """Create a new session. Returns session_id."""
        sid = session_id or generate_id()
        now = self.now_utc_param()

        await self._backend.execute(
            f"""
            INSERT INTO a2a_sessions
            (id, agent_name, user_id, metadata, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (sid, agent_name, user_id, json_dumps(metadata or {}), now, now),
        )

        logger.info(f"Session created: {sid} for agent {agent_name}")
        return sid

    async def get_session(self, session_id: str) -> Optional[SessionState]:
        """Get session by ID."""
        row = await self._backend.fetch_one(
            "SELECT * FROM a2a_sessions WHERE id = ?",
            (session_id,),
        )
        if not row:
            return None
        return self._row_to_session(row)

    async def list_sessions(
        self,
        agent_name: Optional[str] = None,
        user_id: Optional[str] = None,
        limit: int = 100,
    ) -> list[SessionState]:
        """List sessions with optional filters."""
        conditions = []
        params: list[Any] = []

        if agent_name:
            conditions.append("agent_name = ?")
            params.append(agent_name)
        if user_id:
            conditions.append("user_id = ?")
            params.append(user_id)

        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        params.append(limit)

        rows = await self._backend.fetch_all(
            f"""
            SELECT * FROM a2a_sessions
            {where}
            ORDER BY updated_at DESC
            LIMIT ?
            """,
            tuple(params),
        )
        return [self._row_to_session(row) for row in rows]

    async def update_session(
        self,
        session_id: str,
        state: Optional[dict[str, Any]] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        """Update session state and/or metadata."""
        # Get current session
        row = await self._backend.fetch_one(
            "SELECT state, metadata FROM a2a_sessions WHERE id = ?",
            (session_id,),
        )
        if not row:
            raise ValueError(f"Session not found: {session_id}")

        # Merge updates
        current_state = json_loads(row[0]) if row[0] else {}
        current_metadata = json_loads(row[1]) if row[1] else {}

        if state:
            current_state.update(state)
        if metadata:
            current_metadata.update(metadata)

        await self._backend.execute(
            f"""
            UPDATE a2a_sessions
            SET state = ?, metadata = ?, updated_at = {self.now_sql()}
            WHERE id = ?
            """,
            (json_dumps(current_state), json_dumps(current_metadata), session_id),
        )

    async def delete_session(self, session_id: str) -> bool:
        """Delete a session. Returns True if deleted."""
        rows_affected = await self._backend.execute(
            "DELETE FROM a2a_sessions WHERE id = ?",
            (session_id,),
        )
        return rows_affected > 0

    async def append_event(
        self,
        session_id: str,
        event_type: str,
        data: dict[str, Any],
    ) -> None:
        """Append an event to session history."""
        event = {
            "event_type": event_type,
            "timestamp": self.now_utc().isoformat(),
            "data": data,
        }

        if self.is_postgres:
            # PostgreSQL: Use JSONB array concatenation
            await self._backend.execute(
                f"""
                UPDATE a2a_sessions
                SET events = events || ?::jsonb,
                    updated_at = {self.now_sql()}
                WHERE id = ?
                """,
                (json_dumps([event]), session_id),
            )
        else:
            # SQLite: Fetch, modify, save
            row = await self._backend.fetch_one(
                "SELECT events FROM a2a_sessions WHERE id = ?",
                (session_id,),
            )
            if not row:
                raise ValueError(f"Session not found: {session_id}")

            events = json_loads(row[0]) or []
            events.append(event)

            await self._backend.execute(
                f"""
                UPDATE a2a_sessions
                SET events = ?, updated_at = {self.now_sql()}
                WHERE id = ?
                """,
                (json_dumps(events), session_id),
            )

    def _row_to_session(self, row: tuple) -> SessionState:
        """
        Convert database row to SessionState object.

        Row columns (in order):
        0: id, 1: agent_name, 2: user_id, 3: state, 4: events,
        5: metadata, 6: created_at, 7: updated_at
        """
        return SessionState(
            session_id=row[0],
            agent_name=row[1],
            user_id=row[2],
            state=json_loads(row[3]) if row[3] else {},
            events=json_loads(row[4]) if row[4] else [],
            metadata=json_loads(row[5]) if row[5] else {},
            created_at=self.from_timestamp_field(row[6]),
            updated_at=self.from_timestamp_field(row[7]),
        )
