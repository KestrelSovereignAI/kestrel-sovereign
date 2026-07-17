"""Durable constitutional runtime state.

Safe Mode is a security boundary, not a session preference.  This store keeps
the boundary and the periodic-audit deadline in the agent's primary database
so a process restart cannot clear either one.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from kestrel_sovereign.storage.db.interface import DatabaseBackend


@dataclass(frozen=True)
class ConstitutionRuntimeState:
    """Latest authoritative constitutional state for one agent."""

    agent_id: str
    safe_mode: bool
    safe_mode_reason: Optional[str]
    safe_mode_entered_at: Optional[datetime]
    safe_mode_exited_at: Optional[datetime]
    safe_mode_exit_authorization: Optional[str]
    last_successful_audit_at: Optional[datetime]
    interaction_count: int
    updated_at: datetime
    # Distinguishes an interrupted first-ever identity bootstrap from a legacy
    # identity whose anchor is missing. Only the former may establish its
    # initial anchor automatically before the mandatory full startup audit.
    bootstrap_pending: bool = False


class ConstitutionRuntimeStateStore:
    """SQLite/PostgreSQL store for Safe Mode and audit-deadline state."""

    SCHEMA_VERSION = 1

    def __init__(self, backend: DatabaseBackend):
        self._backend = backend

    @property
    def _is_postgres(self) -> bool:
        return self._backend.backend_type == "postgres"

    def _timestamp_type(self) -> str:
        return "TIMESTAMPTZ" if self._is_postgres else "TEXT"

    def _boolean_type(self) -> str:
        return "BOOLEAN" if self._is_postgres else "INTEGER"

    def _integer_primary_key_type(self) -> str:
        if self._is_postgres:
            return "BIGSERIAL PRIMARY KEY"
        return "INTEGER PRIMARY KEY AUTOINCREMENT"

    def _timestamp_param(self, value: Optional[datetime]):
        if value is None or self._is_postgres:
            return value
        return self._as_utc(value).isoformat()

    def _boolean_param(self, value: bool):
        return value if self._is_postgres else int(value)

    @staticmethod
    def _as_utc(value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)

    @classmethod
    def _timestamp_value(cls, value) -> Optional[datetime]:
        if value is None:
            return None
        if not isinstance(value, datetime):
            value = datetime.fromisoformat(str(value))
        return cls._as_utc(value)

    async def initialize(self) -> None:
        """Create the state and append-only transition-event tables."""
        timestamp_type = self._timestamp_type()
        boolean_type = self._boolean_type()
        event_pk = self._integer_primary_key_type()
        await self._backend.execute_script(
            f"""
            CREATE TABLE IF NOT EXISTS constitution_runtime_state (
                agent_id TEXT PRIMARY KEY,
                safe_mode {boolean_type} NOT NULL,
                safe_mode_reason TEXT,
                safe_mode_entered_at {timestamp_type},
                safe_mode_exited_at {timestamp_type},
                safe_mode_exit_authorization TEXT,
                last_successful_audit_at {timestamp_type},
                interaction_count INTEGER NOT NULL DEFAULT 0
                    CHECK (interaction_count >= 0),
                bootstrap_pending {boolean_type} NOT NULL,
                schema_version INTEGER NOT NULL,
                updated_at {timestamp_type} NOT NULL
            );

            CREATE TABLE IF NOT EXISTS constitution_runtime_events (
                id {event_pk},
                agent_id TEXT NOT NULL,
                event_type TEXT NOT NULL,
                reason TEXT,
                authorization_detail TEXT,
                occurred_at {timestamp_type} NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_constitution_runtime_events_agent
                ON constitution_runtime_events(agent_id, id);
            """
        )

    async def load(self, agent_id: str) -> Optional[ConstitutionRuntimeState]:
        """Load one agent's state, returning ``None`` for a legacy agent."""
        row = await self._backend.fetch_one(
            """
            SELECT agent_id, safe_mode, safe_mode_reason,
                   safe_mode_entered_at, safe_mode_exited_at,
                   safe_mode_exit_authorization, last_successful_audit_at,
                   interaction_count, bootstrap_pending, schema_version,
                   updated_at
              FROM constitution_runtime_state
             WHERE agent_id = ?
            """,
            (agent_id,),
        )
        if row is None:
            return None
        if int(row[9]) != self.SCHEMA_VERSION:
            raise ValueError(
                "Unsupported constitution runtime-state schema version"
            )
        return ConstitutionRuntimeState(
            agent_id=str(row[0]),
            safe_mode=bool(row[1]),
            safe_mode_reason=row[2],
            safe_mode_entered_at=self._timestamp_value(row[3]),
            safe_mode_exited_at=self._timestamp_value(row[4]),
            safe_mode_exit_authorization=row[5],
            last_successful_audit_at=self._timestamp_value(row[6]),
            interaction_count=max(0, int(row[7])),
            bootstrap_pending=bool(row[8]),
            updated_at=self._timestamp_value(row[10]),
        )

    async def write(
        self,
        state: ConstitutionRuntimeState,
        *,
        event_type: Optional[str] = None,
        event_reason: Optional[str] = None,
        event_authorization: Optional[str] = None,
    ) -> None:
        """Atomically replace current state and optionally append an event."""
        values = (
            state.agent_id,
            self._boolean_param(state.safe_mode),
            state.safe_mode_reason,
            self._timestamp_param(state.safe_mode_entered_at),
            self._timestamp_param(state.safe_mode_exited_at),
            state.safe_mode_exit_authorization,
            self._timestamp_param(state.last_successful_audit_at),
            max(0, int(state.interaction_count)),
            self._boolean_param(state.bootstrap_pending),
            self.SCHEMA_VERSION,
            self._timestamp_param(state.updated_at),
        )
        async with self._backend.transaction():
            await self._backend.execute(
                """
                INSERT INTO constitution_runtime_state
                    (agent_id, safe_mode, safe_mode_reason,
                     safe_mode_entered_at, safe_mode_exited_at,
                     safe_mode_exit_authorization, last_successful_audit_at,
                     interaction_count, bootstrap_pending, schema_version,
                     updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(agent_id) DO UPDATE SET
                    safe_mode = excluded.safe_mode,
                    safe_mode_reason = excluded.safe_mode_reason,
                    safe_mode_entered_at = excluded.safe_mode_entered_at,
                    safe_mode_exited_at = excluded.safe_mode_exited_at,
                    safe_mode_exit_authorization = excluded.safe_mode_exit_authorization,
                    last_successful_audit_at = excluded.last_successful_audit_at,
                    interaction_count = excluded.interaction_count,
                    bootstrap_pending = excluded.bootstrap_pending,
                    schema_version = excluded.schema_version,
                    updated_at = excluded.updated_at
                """,
                values,
            )
            if event_type is not None:
                await self._backend.execute(
                    """
                    INSERT INTO constitution_runtime_events
                        (agent_id, event_type, reason, authorization_detail,
                         occurred_at)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        state.agent_id,
                        event_type,
                        event_reason,
                        event_authorization,
                        self._timestamp_param(state.updated_at),
                    ),
                )

    async def list_events(self, agent_id: str) -> list[dict]:
        """Return transition history in insertion order (operator/test aid)."""
        rows = await self._backend.fetch_all(
            """
            SELECT event_type, reason, authorization_detail, occurred_at
              FROM constitution_runtime_events
             WHERE agent_id = ?
             ORDER BY id
            """,
            (agent_id,),
        )
        return [
            {
                "event_type": row[0],
                "reason": row[1],
                "authorization": row[2],
                "occurred_at": self._timestamp_value(row[3]),
            }
            for row in rows
        ]
