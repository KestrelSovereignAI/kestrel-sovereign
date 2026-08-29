"""SQLite/PostgreSQL parity for recipient-owned A2A writes (#3144)."""

import asyncio
import os
from uuid import uuid4

import pytest

from kestrel_sovereign.a2a.stores.unified.task_store import (
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
    creator = f"did:test:creator:{uuid4().hex}"
    recipient = f"did:test:recipient:{uuid4().hex}"
    try:
        await store.initialize()
        for task_id in (status_id, artifact_id, lifecycle_id):
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
            ),
            store.update_status(
                status_id,
                TaskStatus(state=TaskState.FAILED),
                recipient_agent_id=creator,
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
        )
        assert await store.save_recipient_lifecycle(
            worker_copy,
            recipient_agent_id=recipient,
        )
        assert (await store.get(lifecycle_id)).status.state is TaskState.WORKING
    finally:
        for task_id in (status_id, artifact_id, lifecycle_id):
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

