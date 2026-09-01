"""SQLite/PostgreSQL parity for recipient-owned A2A writes (#3144)."""

import asyncio
import os
from uuid import uuid4

import pytest

from kestrel_sovereign.a2a.stores.unified.task_store import (
    TaskAlreadyExistsError,
    TaskMutationAuthorizationError,
    TaskStore,
)
from kestrel_sovereign.a2a.types import (
    Artifact,
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


async def _exercise_recipient_mutations(store: TaskStore) -> None:
    status_id = f"recipient-status-{uuid4().hex}"
    artifact_id = f"recipient-artifact-{uuid4().hex}"
    lifecycle_id = f"recipient-lifecycle-{uuid4().hex}"
    lifecycle_race_id = f"recipient-lifecycle-race-{uuid4().hex}"
    status_race_id = f"recipient-status-race-{uuid4().hex}"
    creator = f"did:test:creator:{uuid4().hex}"
    recipient = f"did:test:recipient:{uuid4().hex}"
    try:
        await store.initialize()
        for task_id in (
            status_id,
            artifact_id,
            lifecycle_id,
            lifecycle_race_id,
            status_race_id,
        ):
            await store.save(
                Task(
                    id=task_id,
                    status=TaskStatus(state=TaskState.SUBMITTED),
                ),
                creator_agent_id=creator,
                recipient_agent_id=recipient,
            )

        allowed, denied = await asyncio.gather(
            store.update_status(
                status_id,
                TaskStatus(state=TaskState.WORKING),
                recipient_agent_id=recipient,
                expected_state=TaskState.SUBMITTED,
            ),
            store.update_status(
                status_id,
                TaskStatus(state=TaskState.FAILED),
                recipient_agent_id=creator,
                expected_state=TaskState.SUBMITTED,
            ),
        )
        assert allowed is True
        assert denied is False
        assert (await store.get(status_id)).status.state is TaskState.WORKING

        artifact = Artifact(name="result", parts=[TextPart(text="payload")])
        with pytest.raises(TaskMutationAuthorizationError):
            await store.add_artifact(
                artifact_id,
                artifact,
                recipient_agent_id=creator,
            )
        await store.add_artifact(
            artifact_id,
            artifact,
            recipient_agent_id=recipient,
        )
        assert [item.name for item in (await store.get(artifact_id)).artifacts] == [
            "result"
        ]

        worker_copy = await store.get(lifecycle_id)
        worker_copy.status = TaskStatus(state=TaskState.WORKING)
        assert not await store.save_recipient_lifecycle(
            worker_copy,
            recipient_agent_id=creator,
            expected_state=TaskState.SUBMITTED,
        )
        assert await store.save_recipient_lifecycle(
            worker_copy,
            recipient_agent_id=recipient,
            expected_state=TaskState.SUBMITTED,
        )
        assert (await store.get(lifecycle_id)).status.state is TaskState.WORKING

        first = await store.get(lifecycle_race_id)
        second = await store.get(lifecycle_race_id)
        first.status = TaskStatus(state=TaskState.COMPLETED)
        second.status = TaskStatus(state=TaskState.FAILED)
        lifecycle_winners = await asyncio.gather(
            store.save_recipient_lifecycle(
                first,
                recipient_agent_id=recipient,
                expected_state=TaskState.SUBMITTED,
            ),
            store.save_recipient_lifecycle(
                second,
                recipient_agent_id=recipient,
                expected_state=TaskState.SUBMITTED,
            ),
        )
        assert lifecycle_winners.count(True) == 1
        assert lifecycle_winners.count(False) == 1
        assert (await store.get(lifecycle_race_id)).status.state in {
            TaskState.COMPLETED,
            TaskState.FAILED,
        }

        status_winners = await asyncio.gather(
            store.update_status(
                status_race_id,
                TaskStatus(state=TaskState.COMPLETED),
                recipient_agent_id=recipient,
                expected_state=TaskState.SUBMITTED,
            ),
            store.update_status(
                status_race_id,
                TaskStatus(state=TaskState.FAILED),
                recipient_agent_id=recipient,
                expected_state=TaskState.SUBMITTED,
            ),
        )
        assert status_winners.count(True) == 1
        assert status_winners.count(False) == 1
        assert (await store.get(status_race_id)).status.state in {
            TaskState.COMPLETED,
            TaskState.FAILED,
        }
    finally:
        for task_id in (
            status_id,
            artifact_id,
            lifecycle_id,
            lifecycle_race_id,
            status_race_id,
        ):
            await store.delete(task_id)


@pytest.mark.asyncio
async def test_recipient_mutation_authority_sqlite(tmp_path):
    backend = SQLiteBackend(str(tmp_path / "recipient-mutations.db"))
    await backend.connect()
    try:
        await _exercise_recipient_mutations(TaskStore(backend))
    finally:
        await backend.close()


@pytest.mark.asyncio
async def test_recipient_mutation_authority_postgres():
    if not POSTGRES_URL:  # pragma: no cover - environment gate
        pytest.skip(
            "TEST_POSTGRES_URL / KESTREL_DATABASE_URL / DATABASE_URL required"
        )
    from kestrel_sovereign.storage.db.postgres import PostgresBackend

    backend = PostgresBackend(POSTGRES_URL)
    await backend.connect()
    try:
        await _exercise_recipient_mutations(TaskStore(backend))
    finally:
        await backend.close()


@pytest.mark.asyncio
async def test_sqlite_legacy_replace_cannot_resurrect_terminal_task(tmp_path):
    backend = SQLiteBackend(str(tmp_path / "recipient-terminal-replace.db"))
    await backend.connect()
    store = TaskStore(backend)
    task_id = f"legacy-replace-{uuid4().hex}"
    creator = f"did:test:creator:{uuid4().hex}"
    recipient = f"did:test:recipient:{uuid4().hex}"
    try:
        await store.initialize()
        await store.save(
            Task(id=task_id, status=TaskStatus(state=TaskState.SUBMITTED)),
            creator_agent_id=creator,
            recipient_agent_id=recipient,
        )
        canceled = await store.cancel_if_authorized(
            task_id,
            actor_agent_id=creator,
            expected_recipient_agent_id=recipient,
        )
        assert canceled is not None

        with pytest.raises(TaskAlreadyExistsError):
            await store.create(
                Task(id=task_id, status=TaskStatus(state=TaskState.SUBMITTED)),
                creator_agent_id=creator,
                recipient_agent_id=recipient,
            )

        rows = await backend.execute(
            """
            INSERT OR REPLACE INTO a2a_tasks
                (id, task_type, status, creator_agent_id, recipient_agent_id)
            VALUES (?, 'generic', 'completed', ?, ?)
            """,
            (task_id, creator, recipient),
        )
        assert rows == 0

        persisted = await store.get(task_id)
        assert persisted.status.state is TaskState.CANCELED
        assert persisted.metadata["cancellation_receipt"]["actor_agent_id"] == creator
    finally:
        await store.delete(task_id)
        await backend.close()


@pytest.mark.asyncio
async def test_sqlite_authorityless_terminal_replace_cannot_claim_live_task(tmp_path):
    """A legacy replacement cannot erase authority while claiming completion."""

    backend = SQLiteBackend(str(tmp_path / "recipient-live-replace.db"))
    await backend.connect()
    store = TaskStore(backend)
    task_id = f"authorityless-terminal-replace-{uuid4().hex}"
    creator = f"did:test:creator:{uuid4().hex}"
    recipient = f"did:test:recipient:{uuid4().hex}"
    try:
        await store.initialize()
        await store.save(
            Task(id=task_id, status=TaskStatus(state=TaskState.SUBMITTED)),
            creator_agent_id=creator,
            recipient_agent_id=recipient,
        )

        with pytest.raises(Exception, match="requires durable authority"):
            await backend.execute(
                """
                INSERT OR REPLACE INTO a2a_tasks (id, task_type, status)
                VALUES (?, 'generic', 'completed')
                """,
                (task_id,),
            )

        persisted = await store.get(task_id)
        assert persisted is not None
        assert persisted.status.state is TaskState.SUBMITTED
        authority = await backend.fetch_one(
            """
            SELECT creator_agent_id, recipient_agent_id
            FROM a2a_tasks
            WHERE id = ?
            """,
            (task_id,),
        )
        assert authority == (creator, recipient)
    finally:
        await store.delete(task_id)
        await backend.close()
