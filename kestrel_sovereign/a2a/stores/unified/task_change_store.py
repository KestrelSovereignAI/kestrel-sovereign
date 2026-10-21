"""Atomic task-change references for the durable native push producer."""

from dataclasses import dataclass

from kestrel_sovereign.a2a.scope import NativeRecipientScope
from kestrel_sovereign.a2a.stores.unified.base import (
    NativeStoreConflict,
    UnifiedStoreBase,
)


@dataclass(frozen=True)
class NativeTaskChange:
    """A primary-store reference, containing neither task text nor credentials."""

    sequence: int
    task_id: str


class TaskChangeStore(UnifiedStoreBase):
    """Journal every task mutation in its original database transaction.

    Database triggers cover all canonical lifecycle, artifact and input writes,
    including writes that do not pass through an in-memory manager. A producer
    forwards these references to the existing durable consumer ledger in a
    second atomic transaction. Retain references until task cleanup so a newly
    registered consumer can distinguish older events from its initial snapshot.
    """

    SQLITE_TRIGGERS = (
        ("INSERT", "a2a_task_changes_insert_v1"),
        ("UPDATE", "a2a_task_changes_update_v1"),
    )
    POSTGRES_TRIGGER = "a2a_task_changes_v1"
    TRIGGER_NAMES = frozenset(name for _, name in SQLITE_TRIGGERS) | {POSTGRES_TRIGGER}

    def __init__(self, backend, *, scope: NativeRecipientScope):
        if not isinstance(scope, NativeRecipientScope):
            raise TypeError("Task changes require the canonical recipient scope")
        super().__init__(backend, owns_backend=False)
        self._scope = scope

    @property
    def scope(self) -> NativeRecipientScope:
        """The same immutable recipient and tenant as the canonical task store."""
        return self._scope

    @property
    def _scope_parameters(self) -> tuple[str, str]:
        return self.scope.recipient_id, self.scope.tenant

    async def initialize(self) -> None:
        """Install inside TaskStore's exclusive native schema transaction."""
        async with self.write_transaction():
            await self.backend.execute_script(f"""
                CREATE TABLE IF NOT EXISTS a2a_task_changes (
                    sequence {self.integer_primary_key_type()},
                    recipient_id TEXT NOT NULL,
                    tenant TEXT NOT NULL,
                    task_id TEXT NOT NULL REFERENCES a2a_tasks(id) ON DELETE CASCADE,
                    forwarded_event_id TEXT
                );
                CREATE INDEX IF NOT EXISTS a2a_task_changes_pending
                    ON a2a_task_changes (recipient_id, tenant, sequence)
                    WHERE forwarded_event_id IS NULL;
                CREATE INDEX IF NOT EXISTS a2a_task_changes_task
                    ON a2a_task_changes (recipient_id, tenant, task_id, sequence)
            """)
            insert = """
                INSERT INTO a2a_task_changes (recipient_id, tenant, task_id)
                    VALUES (NEW.recipient_agent_id, NEW.tenant, NEW.id);
            """
            if self.is_sqlite:
                for event, name in self.SQLITE_TRIGGERS:
                    await self.backend.execute_script(f"""
                        CREATE TRIGGER IF NOT EXISTS {name}
                        AFTER {event} ON a2a_tasks
                        WHEN NEW.recipient_agent_id IS NOT NULL
                        BEGIN {insert} END
                    """)
            elif self.is_postgres:
                await self.backend.execute_script(f"""
                    CREATE OR REPLACE FUNCTION a2a_task_changes_v1()
                    RETURNS trigger LANGUAGE plpgsql AS $$
                    BEGIN
                        IF NEW.recipient_agent_id IS NOT NULL THEN {insert} END IF;
                        RETURN NEW;
                    END;
                    $$;
                    CREATE OR REPLACE TRIGGER {self.POSTGRES_TRIGGER}
                        AFTER INSERT OR UPDATE ON a2a_tasks
                        FOR EACH ROW EXECUTE FUNCTION a2a_task_changes_v1()
                """)
            else:
                raise ValueError("Native task changes require SQLite or PostgreSQL")

    async def lock_task_boundary(self, task_id: str, *, creator_id: str) -> int:
        """Authorize and lock the task before capturing a registration boundary.

        The caller must hold its enclosing registration transaction until its
        consumer and initial snapshot reference commit. On PostgreSQL this row
        lock orders registration against every original task mutation; SQLite
        has already reserved its writer through ``write_transaction``.
        """
        if not isinstance(creator_id, str) or not creator_id.strip():
            raise ValueError("Task change registration requires a verified creator")
        suffix = " FOR UPDATE" if self.is_postgres else ""
        row = await self.backend.fetch_one(
            "SELECT id FROM a2a_tasks WHERE id = ? AND recipient_agent_id = ? "
            f"AND tenant = ? AND creator_agent_id = ?{suffix}",
            (task_id, *self._scope_parameters, creator_id),
        )
        if row is None:
            raise NativeStoreConflict("Native task is unavailable to this creator")
        return await self.backend.fetch_val(
            "SELECT COALESCE(MAX(sequence), 0) FROM a2a_task_changes "
            "WHERE recipient_id = ? AND tenant = ? AND task_id = ?",
            (*self._scope_parameters, task_id),
        )

    async def enqueue_snapshot(self, task_id: str, *, creator_id: str) -> int:
        """Append a current-state notification under the creator's task lock."""
        async with self.write_transaction():
            await self.lock_task_boundary(task_id, creator_id=creator_id)
            return await self.backend.fetch_val(
                "INSERT INTO a2a_task_changes (recipient_id, tenant, task_id) "
                "VALUES (?, ?, ?) RETURNING sequence",
                (*self._scope_parameters, task_id),
            )

    async def pending(self, *, limit: int) -> list[NativeTaskChange]:
        """Select references while the producer holds its forwarding transaction."""
        if type(limit) is not int or limit <= 0:
            raise ValueError("Task change batch size must be positive")
        suffix = " FOR UPDATE SKIP LOCKED" if self.is_postgres else ""
        rows = await self.backend.fetch_all(
            "SELECT sequence, task_id FROM a2a_task_changes "
            "WHERE recipient_id = ? AND tenant = ? AND forwarded_event_id IS NULL "
            f"ORDER BY sequence LIMIT ?{suffix}",
            (*self._scope_parameters, limit),
        )
        return [NativeTaskChange(*row) for row in rows]

    async def mark_forwarded(self, change: NativeTaskChange, event_id: str) -> None:
        """Commit only with the original durable signal and its initial fan-out."""
        if not isinstance(event_id, str) or not event_id.strip():
            raise ValueError("Task change forwarding requires a durable event ID")
        changed = await self.backend.execute(
            "UPDATE a2a_task_changes SET forwarded_event_id = ? WHERE sequence = ? "
            "AND recipient_id = ? AND tenant = ? AND task_id = ? "
            "AND forwarded_event_id IS NULL",
            (event_id, change.sequence, *self._scope_parameters, change.task_id),
        )
        if changed != 1:
            raise NativeStoreConflict("Task change lost its original forwarding claim")
