"""SQLite/PostgreSQL parity for principal-scoped A2A reads (#3145)."""

from __future__ import annotations

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


async def _exercise_principal_reads(store: TaskStore) -> None:
    suffix = uuid4().hex
    task_a = f"principal-a-{suffix}"
    task_b = f"principal-b-{suffix}"
    creator_a = f"did:test:creator-a:{suffix}"
    creator_b = f"did:test:creator-b:{suffix}"
    recipient_a = f"did:test:recipient-a:{suffix}"
    recipient_b = f"did:test:recipient-b:{suffix}"
    try:
        await store.initialize()
        await store.save(
            Task(id=task_a, status=TaskStatus(state=TaskState.SUBMITTED)),
            creator_agent_id=creator_a,
            recipient_agent_id=recipient_a,
        )
        await store.save(
            Task(id=task_b, status=TaskStatus(state=TaskState.SUBMITTED)),
            creator_agent_id=creator_b,
            recipient_agent_id=recipient_b,
        )

        assert await store.get_for_creator(
            task_a,
            creator_a,
            recipient_agent_id=recipient_a,
        ) is not None
        assert await store.get_for_creator(
            task_a,
            creator_a,
            recipient_agent_id=recipient_b,
        ) is None
        assert await store.get_for_creator(
            task_a,
            creator_b,
            recipient_agent_id=recipient_a,
        ) is None
        assert await store.get_for_recipient(task_a, recipient_a) is not None
        assert await store.get_for_recipient(task_a, recipient_b) is None
        assert [task.id for task in await store.list_tasks(
            recipient_agent_id=recipient_a
        )] == [task_a]
        assert [task.id for task in await store.get_pending_tasks(
            recipient_agent_id=recipient_b
        )] == [task_b]
    finally:
        await store.delete(task_a)
        await store.delete(task_b)
@pytest.mark.asyncio
async def test_principal_read_authority_sqlite(tmp_path):
    backend = SQLiteBackend(str(tmp_path / "principal-reads.db"))
    await backend.connect()
    try:
        await _exercise_principal_reads(TaskStore(backend))
    finally:
        await backend.close()


@pytest.mark.asyncio
async def test_principal_read_authority_postgres():
    if not POSTGRES_URL:  # pragma: no cover - environment gate
        pytest.skip(
            "TEST_POSTGRES_URL / KESTREL_DATABASE_URL / DATABASE_URL required"
        )
    from kestrel_sovereign.storage.db.postgres import PostgresBackend

    backend = PostgresBackend(POSTGRES_URL)
    await backend.connect()
    try:
        await _exercise_principal_reads(TaskStore(backend))
    finally:
        await backend.close()
