"""SQLite/PostgreSQL parity for atomic A2A cancellation authority (#3134)."""

import asyncio
import os
import threading
from uuid import uuid4

import pytest

from kestrel_sovereign.a2a.stores.unified.task_store import TaskStore
from kestrel_sovereign.a2a.types import (
    Artifact,
    Message,
    Task,
    TaskState,
    TaskStatus,
    TextPart,
)
from kestrel_sovereign.storage.db import SQLiteBackend


POSTGRES_URL = (
    os.environ.get("TEST_POSTGRES_URL")
    or os.environ.get("KESTREL_DATABASE_URL")
    or os.environ.get("DATABASE_URL")
)


async def _exercise_authorized_cancel(store: TaskStore) -> None:
    task_id = f"cancel-auth-{uuid4().hex}"
    payload_task_id = f"cancel-payload-{uuid4().hex}"
    artifact_race_task_id = f"cancel-artifact-race-{uuid4().hex}"
    creator = f"did:test:creator:{uuid4().hex}"
    recipient = f"did:test:recipient:{uuid4().hex}"
    try:
        await store.initialize()
        await store.save(
            Task(id=task_id, status=TaskStatus(state=TaskState.SUBMITTED)),
            creator_agent_id=creator,
            recipient_agent_id=recipient,
        )

        assert (
            await store.cancel_if_authorized(
                task_id,
                actor_agent_id=f"did:test:peer:{uuid4().hex}",
                reason="not mine",
            )
            is None
        )
        assert (await store._get_unscoped(task_id)).status.state is TaskState.SUBMITTED
        assert (
            await store.cancel_if_authorized(
                task_id,
                actor_agent_id=creator,
                expected_recipient_agent_id=f"did:test:wrong-recipient:{uuid4().hex}",
                reason="misrouted",
            )
            is None
        )
        assert (await store._get_unscoped(task_id)).status.state is TaskState.SUBMITTED

        canceled = await store.cancel_if_authorized(
            task_id,
            actor_agent_id=recipient,
            expected_recipient_agent_id=recipient,
            reason="delegate stopped work",
        )
        assert canceled is not None
        assert canceled.status.state is TaskState.CANCELED
        assert canceled.metadata["cancellation_receipt"] == {
            "actor_agent_id": recipient,
            "reason": "delegate stopped work",
            "status_before": "submitted",
        }
        assert canceled.history[-1].parts[0].text == (
            f"Task canceled by {recipient}: delegate stopped work"
        )
        stale = Task(id=task_id, status=TaskStatus(state=TaskState.COMPLETED))
        assert not await store.save_recipient_lifecycle(
            stale,
            recipient_agent_id=recipient,
            expected_state=TaskState.SUBMITTED,
        )
        assert (await store._get_unscoped(task_id)).status.state is TaskState.CANCELED

        initial = Task(
            id=payload_task_id,
            status=TaskStatus(state=TaskState.SUBMITTED),
            history=[Message(role="user", parts=[TextPart(text="initial")])],
            metadata={"initial": True},
        )
        await store.save(
            initial,
            creator_agent_id=creator,
            recipient_agent_id=recipient,
        )
        concurrent = await store._get_unscoped(payload_task_id)
        concurrent.artifacts = [
            Artifact(name="concurrent", parts=[TextPart(text="keep")])
        ]
        concurrent.history.append(
            Message(role="agent", parts=[TextPart(text="concurrent")])
        )
        concurrent.metadata["concurrent"] = True
        assert await store.save_recipient_lifecycle(
            concurrent,
            recipient_agent_id=recipient,
            expected_state=TaskState.SUBMITTED,
        )

        handler_payload = Task(
            id=payload_task_id,
            status=TaskStatus(state=TaskState.CANCELED),
            artifacts=[Artifact(name="handler", parts=[TextPart(text="partial")])],
            history=[
                initial.history[0],
                Message(role="agent", parts=[TextPart(text="handler")]),
            ],
            metadata={"initial": True, "handler": True},
        )
        merged = await store.cancel_if_authorized(
            payload_task_id,
            actor_agent_id=recipient,
            reason="handler declined",
            task_payload=handler_payload,
        )

        assert merged is not None
        assert {artifact.name for artifact in merged.artifacts} == {
            "concurrent",
            "handler",
        }
        assert merged.metadata["concurrent"] is True
        assert merged.metadata["handler"] is True
        merged_history = [
            part.text for message in merged.history for part in message.parts
        ]
        assert "concurrent" in merged_history
        assert "handler" in merged_history

        await store.save(
            Task(
                id=artifact_race_task_id,
                status=TaskStatus(state=TaskState.WORKING),
            ),
            creator_agent_id=creator,
            recipient_agent_id=recipient,
        )
        racing_artifact = Artifact(
            name="racing-artifact",
            parts=[TextPart(text="keep if append won")],
        )
        append_result, cancel_result = await asyncio.gather(
            store.add_artifact(
                artifact_race_task_id,
                racing_artifact,
                recipient_agent_id=recipient,
            ),
            store.cancel_if_authorized(
                artifact_race_task_id,
                actor_agent_id=creator,
                reason="raced artifact production",
            ),
            return_exceptions=True,
        )
        assert cancel_result is not None and not isinstance(cancel_result, Exception)
        raced = await store._get_unscoped(artifact_race_task_id)
        assert raced.status.state is TaskState.CANCELED
        if isinstance(append_result, Exception):
            assert "terminal task" in str(append_result)
            assert not raced.artifacts
        else:
            assert [artifact.name for artifact in raced.artifacts] == [
                "racing-artifact"
            ]
    finally:
        await store.delete(task_id)
        await store.delete(payload_task_id)
        await store.delete(artifact_race_task_id)


@pytest.mark.asyncio
async def test_cancel_task_authorization_sqlite(tmp_path):
    backend = SQLiteBackend(str(tmp_path / "cancel-authority.db"))
    await backend.connect()
    try:
        await _exercise_authorized_cancel(TaskStore(backend))
    finally:
        await backend.close()


@pytest.mark.asyncio
async def test_sqlite_legacy_insert_or_replace_cannot_overwrite_cancellation(tmp_path):
    """A mixed-version writer cannot erase a committed cancellation receipt."""

    backend = SQLiteBackend(str(tmp_path / "cancel-replace-fence.db"))
    await backend.connect()
    store = TaskStore(backend)
    task_id = f"cancel-replace-{uuid4().hex}"
    creator = f"did:test:creator:{uuid4().hex}"
    recipient = f"did:test:recipient:{uuid4().hex}"
    try:
        await store.initialize()
        await store.save(
            Task(id=task_id, status=TaskStatus(state=TaskState.WORKING)),
            creator_agent_id=creator,
            recipient_agent_id=recipient,
        )
        assert await store.cancel_if_authorized(
            task_id,
            actor_agent_id=creator,
            reason="committed before stale writer",
        )

        with pytest.raises(Exception, match="requires durable authority"):
            await backend.execute(
                """
                INSERT OR REPLACE INTO a2a_tasks
                    (id, task_type, status, metadata)
                VALUES (?, 'generic', 'completed', '{}')
                """,
                (task_id,),
            )

        canceled = await store._get_unscoped(task_id)
        assert canceled is not None
        assert canceled.status.state is TaskState.CANCELED
        assert canceled.metadata["cancellation_receipt"] == {
            "actor_agent_id": creator,
            "reason": "committed before stale writer",
            "status_before": "working",
        }
    finally:
        await store.delete(task_id)
        await backend.close()


@pytest.mark.asyncio
async def test_sqlite_fence_upgrade_never_exposes_terminal_row_to_legacy_writer(
    tmp_path,
):
    """Replacement fences must exist before an obsolete fence is removed."""

    database = str(tmp_path / "cancel-fence-upgrade-order.db")
    migrating_backend = SQLiteBackend(database)
    legacy_backend = SQLiteBackend(database)
    await migrating_backend.connect()
    await legacy_backend.connect()
    store = TaskStore(migrating_backend)
    task_id = f"cancel-upgrade-{uuid4().hex}"
    at_replacement = threading.Event()
    continue_migration = threading.Event()

    def pause_before_replacement(statement):
        if "CREATE TRIGGER IF NOT EXISTS a2a_tasks_terminal_update_v3" in statement:
            at_replacement.set()
            continue_migration.wait(timeout=5)

    migration = None
    persisted = None
    try:
        await store.initialize()
        await store.create(
            Task(id=task_id, status=TaskStatus(state=TaskState.SUBMITTED)),
            creator_agent_id="did:test:creator",
            recipient_agent_id="did:test:recipient",
        )
        assert await store.cancel_if_authorized(
            task_id,
            actor_agent_id="did:test:creator",
            reason="must remain terminal",
        )

        # Model the last released schema: only the obsolete canceled-update
        # fence remains. The migration pauses immediately before executing the
        # replacement CREATE so a second connection can attempt a legacy write.
        await migrating_backend.execute_script("""
            DROP TRIGGER IF EXISTS a2a_tasks_terminal_update_v3;
            DROP TRIGGER IF EXISTS a2a_tasks_terminal_replace_v4;
            CREATE TRIGGER a2a_tasks_canceled_terminal_v1
            BEFORE UPDATE ON a2a_tasks
            FOR EACH ROW
            WHEN OLD.status = 'canceled'
            BEGIN
                SELECT RAISE(ABORT, 'terminal A2A task cannot be replaced');
            END;
        """)
        await migrating_backend._connection.set_trace_callback(
            pause_before_replacement
        )
        migration = asyncio.create_task(store._install_canceled_terminal_fence())
        assert await asyncio.to_thread(at_replacement.wait, 2)

        with pytest.raises(Exception, match="terminal A2A task cannot be replaced"):
            await legacy_backend.execute(
                "UPDATE a2a_tasks SET status = 'completed' WHERE id = ?",
                (task_id,),
            )
    finally:
        continue_migration.set()
        if migration is not None:
            await migration
        await migrating_backend._connection.set_trace_callback(None)
        persisted = await store._get_unscoped(task_id)
        await legacy_backend.close()
        await migrating_backend.close()

    assert persisted is not None
    assert persisted.status.state is TaskState.CANCELED


@pytest.mark.asyncio
async def test_upgrade_settles_live_rows_without_trustworthy_authority(tmp_path):
    backend = SQLiteBackend(str(tmp_path / "legacy-live-task.db"))
    await backend.connect()
    try:
        await backend.execute_script(
            """
            CREATE TABLE a2a_tasks (
                id TEXT PRIMARY KEY,
                session_id TEXT,
                user_id TEXT,
                task_type TEXT NOT NULL,
                status TEXT DEFAULT 'submitted',
                message TEXT,
                artifacts TEXT DEFAULT '[]',
                history TEXT DEFAULT '[]',
                metadata TEXT DEFAULT '{}',
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            );
            INSERT INTO a2a_tasks
                (id, task_type, status, metadata)
            VALUES
                ('legacy-live', 'generic', 'working', '{"sender":"untrusted"}'),
                ('legacy-done', 'generic', 'completed', '{}');
            """
        )
        store = TaskStore(backend)
        await store.initialize()

        live = await store._get_unscoped("legacy-live")
        done = await store._get_unscoped("legacy-done")
        assert live.status.state is TaskState.FAILED
        assert "no trustworthy creator/recipient binding" in (
            live.status.message.parts[0].text
        )
        assert done.status.state is TaskState.COMPLETED
    finally:
        await backend.close()


@pytest.mark.asyncio
async def test_cancel_task_authorization_postgres():
    if not POSTGRES_URL:  # pragma: no cover - environment gate
        pytest.skip(
            "TEST_POSTGRES_URL / KESTREL_DATABASE_URL / DATABASE_URL required"
        )
    from kestrel_sovereign.storage.db.postgres import PostgresBackend

    backend = PostgresBackend(POSTGRES_URL)
    await backend.connect()
    try:
        await _exercise_authorized_cancel(TaskStore(backend))
    finally:
        await backend.close()
