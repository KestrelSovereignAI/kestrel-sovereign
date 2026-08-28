"""SQLite/PostgreSQL parity for atomic A2A cancellation authority (#3134)."""

import os
from uuid import uuid4

import pytest

from kestrel_sovereign.a2a.stores.unified.task_store import TaskStore
from kestrel_sovereign.a2a.types import Task, TaskState, TaskStatus
from kestrel_sovereign.storage.db import SQLiteBackend


POSTGRES_URL = (
    os.environ.get("TEST_POSTGRES_URL")
    or os.environ.get("KESTREL_DATABASE_URL")
    or os.environ.get("DATABASE_URL")
)


async def _exercise_authorized_cancel(store: TaskStore) -> None:
    task_id = f"cancel-auth-{uuid4().hex}"
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
        assert (await store.get(task_id)).status.state is TaskState.SUBMITTED

        canceled = await store.cancel_if_authorized(
            task_id,
            actor_agent_id=recipient,
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
        assert await store.save(stale) is False
        assert (await store.get(task_id)).status.state is TaskState.CANCELED
    finally:
        await store.delete(task_id)


@pytest.mark.asyncio
async def test_cancel_authorization_sqlite(tmp_path):
    backend = SQLiteBackend(str(tmp_path / "cancel-authority.db"))
    await backend.connect()
    try:
        await _exercise_authorized_cancel(TaskStore(backend))
    finally:
        await backend.close()


@pytest.mark.asyncio
async def test_cancel_authorization_postgres():
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
