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
    """Journal subscribed task mutations in their original database transaction.

    Database triggers cover all canonical lifecycle, artifact and input writes,
    including writes that do not pass through an in-memory manager. A producer
    forwards these references to the existing durable consumer ledger in a
    second atomic transaction. Initial subscriptions own an explicit snapshot.
    References retire atomically with their terminal durable events.
    """

    SQLITE_TRIGGERS = (
        ("INSERT", "a2a_task_changes_insert_v2"),
        ("UPDATE", "a2a_task_changes_update_v2"),
    )
    POSTGRES_TRIGGER = "a2a_task_changes_v2"
    LEGACY_TRIGGER_NAMES = frozenset(
        {"a2a_task_changes_insert_v1", "a2a_task_changes_update_v1", "a2a_task_changes_v1"}
    )
    TRIGGER_NAMES = (
        frozenset(name for _, name in SQLITE_TRIGGERS)
        | {POSTGRES_TRIGGER}
        | LEGACY_TRIGGER_NAMES
    )

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
        """Install references without enabling the optional push producer."""
        async with self.write_transaction():
            await self._refuse_legacy_capture()
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
                    ON a2a_task_changes (recipient_id, tenant, task_id, sequence);
                CREATE INDEX IF NOT EXISTS a2a_task_changes_forwarded
                    ON a2a_task_changes (recipient_id, tenant, forwarded_event_id)
                    WHERE forwarded_event_id IS NOT NULL
            """)

    async def _refuse_legacy_capture(self) -> None:
        """A partial older native bundle cannot silently keep global capture."""
        if self.is_sqlite:
            rows = await self.backend.fetch_all(
                "SELECT name FROM sqlite_master WHERE type = 'trigger' "
                "AND tbl_name = 'a2a_tasks'"
            )
        elif self.is_postgres:
            rows = await self.backend.fetch_all(
                "SELECT t.tgname FROM pg_trigger t JOIN pg_class c ON c.oid = t.tgrelid "
                "JOIN pg_namespace n ON n.oid = c.relnamespace "
                "WHERE n.nspname = current_schema() AND c.relname = 'a2a_tasks' "
                "AND NOT t.tgisinternal"
            )
        else:
            raise ValueError("Native task changes require SQLite or PostgreSQL")
        if any(row[0] in self.LEGACY_TRIGGER_NAMES for row in rows):
            from kestrel_sovereign.a2a.native_schema import NativeA2AMigrationRequired

            raise NativeA2AMigrationRequired(
                "Stored native task-change capture requires offline migration"
            )

    async def enable_push_capture(self) -> None:
        """Enable capture only after configuration, subscription and ledger DDL.

        These are shared schema objects, never per-agent startup switches.
        Every trigger derives eligibility from the exact active configuration
        and consumer rows, so disabled recipients cannot remove another owner's
        capture. An intentionally paused live subscription retains its backlog.
        """
        async with self.write_transaction():
            for table in (
                "a2a_push_configs", "a2a_push_subscriptions", "durable_signal_consumers"
            ):
                if not await self.backend.table_exists(table):
                    raise RuntimeError("Native push capture requires its initialized stores")
            await self._refuse_legacy_capture()
            subscribed = """
                EXISTS (
                    SELECT 1 FROM a2a_push_configs cfg
                    JOIN a2a_push_subscriptions sub
                      ON sub.recipient_id = cfg.recipient_id AND sub.tenant = cfg.tenant
                     AND sub.task_id = cfg.task_id AND sub.config_id = cfg.config_id
                     AND sub.generation = cfg.generation
                    JOIN durable_signal_consumers consumer
                      ON consumer.agent_id = sub.agent_id
                     AND consumer.consumer_id = sub.consumer_id
                    WHERE cfg.recipient_id = NEW.recipient_agent_id
                      AND cfg.tenant = NEW.tenant AND cfg.task_id = NEW.id
                      AND cfg.state = 'active' AND consumer.active = TRUE
                )
            """
            insert = """
                INSERT INTO a2a_task_changes (recipient_id, tenant, task_id)
                    VALUES (NEW.recipient_agent_id, NEW.tenant, NEW.id);
            """
            if self.is_sqlite:
                for event, name in self.SQLITE_TRIGGERS:
                    await self.backend.execute_script(f"""
                        CREATE TRIGGER IF NOT EXISTS {name}
                        AFTER {event} ON a2a_tasks
                        WHEN {subscribed}
                        BEGIN {insert} END
                    """)
            elif self.is_postgres:
                await self.backend.execute_script(f"""
                    CREATE OR REPLACE FUNCTION {self.POSTGRES_TRIGGER}()
                    RETURNS trigger LANGUAGE plpgsql AS $$
                    BEGIN
                        IF {subscribed} THEN {insert} END IF;
                        RETURN NEW;
                    END;
                    $$;
                    CREATE OR REPLACE TRIGGER {self.POSTGRES_TRIGGER}
                        AFTER INSERT OR UPDATE ON a2a_tasks
                        FOR EACH ROW EXECUTE FUNCTION {self.POSTGRES_TRIGGER}()
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

    async def retire_forwarded(self, event_ids: tuple[str, ...]) -> int:
        """Remove only this scope's references to exact already-deleted events.

        The outbox owns the enclosing transaction through both the signal
        deletion and this reference retirement. No clock or history scan can
        substitute for that original deletion result.
        """
        removed = 0
        for event_id in event_ids:
            if not isinstance(event_id, str) or not event_id:
                raise ValueError("Task change retirement requires concrete event IDs")
            removed += await self.backend.execute(
                "DELETE FROM a2a_task_changes WHERE recipient_id = ? AND tenant = ? "
                "AND forwarded_event_id = ?",
                (*self._scope_parameters, event_id),
            )
        return removed
