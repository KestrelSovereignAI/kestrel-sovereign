"""Durable constitutional runtime state.

Safe Mode is a security boundary, not a session preference.  This store keeps
the boundary and the periodic-audit deadline in the agent's primary database
so a process restart cannot clear either one.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timezone
from typing import Optional

from kestrel_sovereign.storage.db.interface import DatabaseBackend
from kestrel_sovereign.storage.db.timestamp import TimestamptzParameter


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
    #: Why cognition is restricted, when it is. NULL on rows written before
    #: causes were recorded — which is not the same as no cause, and must not
    #: be read as an integrity finding (#2920 defect 3).
    safe_mode_cause: Optional[str] = None
    # Optimistic database fencing, independent of wall-clock precision.
    # None is an explicit first-creation snapshot; loaded legacy revision zero
    # is an existing record and may ONLY be conditionally updated.
    revision: Optional[int] = None


class ConstitutionStateConflictError(RuntimeError):
    """A stale snapshot or unauthorized restriction-clear was refused."""


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
        if value is None:
            return None
        instant = self._as_utc(value)
        if self._is_postgres:
            # These columns are TIMESTAMPTZ, not legacy naive TIMESTAMP.
            # Preserve the instant through PostgresBackend's shared adapter.
            return TimestamptzParameter(instant)
        return instant.isoformat()

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
                updated_at {timestamp_type} NOT NULL,
                safe_mode_cause TEXT,
                revision INTEGER NOT NULL DEFAULT 0 CHECK (revision >= 0)
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
        # Existing installs already have the table, so CREATE IF NOT EXISTS
        # never adds the column. SCHEMA_VERSION is deliberately NOT bumped:
        # it is nullable, the row shape stays readable, and ``load`` RAISES on
        # a version mismatch — which lands in
        # ``_mark_constitution_state_unavailable`` and would put every
        # existing agent into Safe Mode on upgrade.
        await self._migrate_safe_mode_cause_column()
        await self._migrate_column("revision", "INTEGER NOT NULL DEFAULT 0 CHECK (revision >= 0)")
        await self._ensure_revision_fence()

    async def _ensure_revision_fence(self) -> None:
        """Old binaries cannot overwrite a row without advancing its fence.

        This is enforced in the database because a replica predating revision
        fencing does not execute the new store's CAS statement. Repeated boots
        inspect the catalog, avoiding an unnecessary DDL lock on a live table.
        """
        trigger = "constitution_runtime_revision_fence_v1"
        if self._is_postgres:
            existing = await self._backend.fetch_one(
                "SELECT 1 FROM pg_trigger WHERE tgrelid = to_regclass(?) "
                "AND tgname = ? AND NOT tgisinternal",
                ("constitution_runtime_state", trigger),
            )
            if existing is not None:
                return
            # Serialize initial installation only. The catalog recheck after
            # the advisory lock also handles concurrently starting replicas.
            async with self._backend.transaction():
                await self._backend.fetch_one(
                    "SELECT pg_advisory_xact_lock(hashtextextended(current_schema() || ?, 0))",
                    (f":{trigger}",),
                )
                existing = await self._backend.fetch_one(
                    "SELECT 1 FROM pg_trigger WHERE tgrelid = to_regclass(?) "
                    "AND tgname = ? AND NOT tgisinternal",
                    ("constitution_runtime_state", trigger),
                )
                if existing is None:
                    await self._backend.execute_script(
                        f"""
                        CREATE OR REPLACE FUNCTION {trigger}() RETURNS trigger AS $fence$
                        BEGIN
                            IF NEW.revision <> OLD.revision + 1 THEN
                                RAISE EXCEPTION 'constitution runtime revision fence refused old writer';
                            END IF;
                            RETURN NEW;
                        END;
                        $fence$ LANGUAGE plpgsql;
                        CREATE TRIGGER {trigger} BEFORE UPDATE ON constitution_runtime_state
                        FOR EACH ROW EXECUTE FUNCTION {trigger}();
                        """
                    )
        else:
            existing = await self._backend.fetch_one(
                "SELECT 1 FROM sqlite_master WHERE type = 'trigger' AND name = ?",
                (trigger,),
            )
            if existing is None:
                await self._backend.execute_script(
                    f"""
                    CREATE TRIGGER IF NOT EXISTS {trigger}
                    BEFORE UPDATE ON constitution_runtime_state
                    FOR EACH ROW WHEN NEW.revision <> OLD.revision + 1
                    BEGIN
                        SELECT RAISE(ABORT, 'constitution runtime revision fence refused old writer');
                    END;
                    """
                )

    async def _migrate_safe_mode_cause_column(self) -> None:
        """Add ``safe_mode_cause`` to a table created before it existed.

        The metadata is read first rather than issuing the ALTER and treating
        every failure as "already there". On a fresh database the CREATE above
        already includes the column, so the ALTER was pure cost — and on
        hosted PostgreSQL it takes an ACCESS EXCLUSIVE lock, so agents
        starting concurrently against one database serialized on it and could
        block live state writes. Swallowing every exception also hid a real
        migration failure behind the same silence.
        """
        await self._migrate_column("safe_mode_cause", "TEXT")

    async def _migrate_column(self, name: str, declaration: str) -> None:
        """Inspect cheaply on normal boots; reserve and recheck on upgrade."""
        if await self._has_column(name):
            return
        if self._is_postgres:
            await self._backend.execute(
                f"ALTER TABLE constitution_runtime_state ADD COLUMN IF NOT EXISTS {name} {declaration}"
            )
        else:
            # Independent SQLite connections must reserve the writer slot
            # BEFORE reading the upgrade metadata. Deferred read-then-ALTER
            # races either duplicate the column or fail to promote a snapshot.
            async with self._backend.transaction(immediate=True):
                if not await self._has_column(name):
                    await self._backend.execute(
                        f"ALTER TABLE constitution_runtime_state ADD COLUMN {name} {declaration}"
                    )

    async def _has_safe_mode_cause_column(self) -> bool:
        """Whether the column is already present, per the backend's catalog."""
        return await self._has_column("safe_mode_cause")

    async def _has_column(self, name: str) -> bool:
        if getattr(self._backend, "backend_type", "sqlite") == "postgres":
            row = await self._backend.fetch_one(
                """
                SELECT 1 FROM information_schema.columns
                 WHERE table_name = 'constitution_runtime_state'
                   AND table_schema = current_schema()
                   AND column_name = ?
                """,
                (name,),
            )
            return row is not None
        rows = await self._backend.fetch_all(
            "PRAGMA table_info(constitution_runtime_state)"
        )
        return any(
            (r[1] if not isinstance(r, dict) else r.get("name")) == name
            for r in (rows or ())
        )

    async def load(self, agent_id: str) -> Optional[ConstitutionRuntimeState]:
        """Load one agent's state, returning ``None`` for a legacy agent."""
        row = await self._backend.fetch_one(
            """
            SELECT agent_id, safe_mode, safe_mode_reason,
                   safe_mode_entered_at, safe_mode_exited_at,
                   safe_mode_exit_authorization, last_successful_audit_at,
                   interaction_count, bootstrap_pending, schema_version,
                   updated_at, safe_mode_cause, revision
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
            safe_mode_cause=row[11],
            revision=int(row[12]),
        )

    async def write(
        self,
        state: ConstitutionRuntimeState,
        *,
        event_type: Optional[str] = None,
        event_reason: Optional[str] = None,
        event_authorization: Optional[str] = None,
    ) -> ConstitutionRuntimeState:
        """CAS-replace state and event; ordinary writes cannot clear SafeMode.

        Return the persisted revision. Callers must carry it into their next
        snapshot. A failed CAS appends no event and must restrict the caller,
        not silently overwrite a newer replica's latch or audit due marker.
        """
        authorized_exit = (
            event_type == "safe_mode_exited"
            and bool(event_authorization)
            and state.safe_mode_exit_authorization == event_authorization
            and state.safe_mode_exited_at is not None
            and state.last_successful_audit_at == state.safe_mode_exited_at
            and not state.safe_mode
        )
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
            state.safe_mode_cause,
            1 if state.revision is None else state.revision + 1,
        )
        async with self._backend.transaction():
            if state.revision is None:
                written = await self._backend.fetch_one(
                    """
                    INSERT INTO constitution_runtime_state
                    (agent_id, safe_mode, safe_mode_reason,
                     safe_mode_entered_at, safe_mode_exited_at,
                     safe_mode_exit_authorization, last_successful_audit_at,
                     interaction_count, bootstrap_pending, schema_version,
                     updated_at, safe_mode_cause, revision)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(agent_id) DO NOTHING
                    RETURNING revision
                    """,
                    values,
                )
            else:
                # UPDATE cannot turn into INSERT after a concurrently deleted
                # unique row disappears. In PostgreSQL a conditional UPSERT's
                # initial EXISTS snapshot does not provide that guarantee.
                written = await self._backend.fetch_one(
                    """
                    UPDATE constitution_runtime_state SET
                        safe_mode = ?, safe_mode_reason = ?,
                        safe_mode_entered_at = ?, safe_mode_exited_at = ?,
                        safe_mode_exit_authorization = ?, last_successful_audit_at = ?,
                        interaction_count = ?, bootstrap_pending = ?, schema_version = ?,
                        updated_at = ?, safe_mode_cause = ?, revision = ?
                    WHERE agent_id = ? AND revision = ?
                      AND (NOT safe_mode OR ? OR ?)
                      AND (interaction_count <= ? OR ?)
                      AND (NOT bootstrap_pending OR ? OR ?)
                    RETURNING revision
                    """,
                    values[1:] + (
                        state.agent_id, state.revision,
                        self._boolean_param(state.safe_mode), self._boolean_param(authorized_exit),
                        max(0, int(state.interaction_count)),
                        self._boolean_param(authorized_exit or event_type == "audit_succeeded"),
                        self._boolean_param(state.bootstrap_pending),
                        self._boolean_param(authorized_exit or event_type == "audit_succeeded"),
                    ),
                )
            if written is not None and event_type is not None:
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
        # Both adapters wrap exceptions raised inside a transaction. A CAS
        # refusal made no mutation/event; report its typed conflict outside
        # that boundary so callers can distinguish it from storage failure.
        if written is None:
            raise ConstitutionStateConflictError(
                "stale constitution state or unauthorized SafeMode clear"
            )
        return replace(state, revision=int(written[0]))

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
