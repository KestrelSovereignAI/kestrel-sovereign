"""SQLite/PostgreSQL semantic parity contracts for storage seams."""

from datetime import datetime, timedelta
from uuid import uuid4

import pytest

from kestrel_sovereign.a2a.stores.unified import TaskStore
from kestrel_sovereign.a2a.types import (
    Artifact,
    Message,
    Task,
    TaskState,
    TaskStatus,
    TextPart,
)
from kestrel_sovereign.privacy import PrivacyMode
from kestrel_sovereign.storage.async_storage import AsyncStorage
from kestrel_sovereign.storage.privacy_wrapper import PrivacyEnforcingStorage


def _project_rows(rows):
    return [(row[1], row[2]) for row in rows]


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_conversation_session_queries_are_backend_neutral(db_backend):
    storage = AsyncStorage.from_backend(db_backend)
    await storage.initialize()
    privacy_storage = PrivacyEnforcingStorage(storage, PrivacyMode.NORMAL)

    agent_id = f"did:test:{uuid4()}"
    other_agent_id = f"did:test:{uuid4()}"
    start = datetime(2026, 4, 16, 12, 0, 0)

    await storage.db.execute_many(
        """
        INSERT INTO conversation_history (agent_id, role, content, metadata, created_at)
        VALUES (?, ?, ?, ?, ?)
        """,
        [
            (
                agent_id,
                "system",
                "[New conversation started]",
                '{"new_session": true, "type": "session_marker"}',
                start,
            ),
            (agent_id, "user", "hello", '{"topic": "parity"}', start + timedelta(minutes=1)),
            (agent_id, "assistant", "hi there", "{}", start + timedelta(minutes=2)),
            (other_agent_id, "user", "not yours", "{}", start + timedelta(minutes=3)),
        ],
    )

    inserted = await storage.db.fetchall(
        """
        SELECT id, role, content, metadata, created_at
        FROM conversation_history
        WHERE agent_id = ?
        ORDER BY created_at ASC
        """,
        (agent_id,),
    )
    session_id = str(inserted[0][0])

    listed = await privacy_storage.query_conversations(agent_id, limit=10)
    assert _project_rows(listed) == [
        ("assistant", "hi there"),
        ("user", "hello"),
        ("system", "[New conversation started]"),
    ]

    start_row = await privacy_storage.query_conversation_start(session_id, agent_id)
    assert start_row is not None

    session_rows = await privacy_storage.query_conversation_messages(
        agent_id,
        start_row[0],
        limit=10,
    )
    assert _project_rows(session_rows) == [
        ("system", "[New conversation started]"),
        ("user", "hello"),
        ("assistant", "hi there"),
    ]

    other_rows = await privacy_storage.query_conversation_messages(
        other_agent_id,
        start_row[0],
        limit=10,
    )
    assert _project_rows(other_rows) == [("user", "not yours")]


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_a2a_task_store_filters_and_payloads_are_backend_neutral(db_backend):
    store = TaskStore(db_backend)
    await store.initialize()

    session_a = f"session-{uuid4()}"
    session_b = f"session-{uuid4()}"
    user_a = f"user-{uuid4()}"
    user_b = f"user-{uuid4()}"
    task_a = f"task-{uuid4()}"
    task_b = f"task-{uuid4()}"
    other_task = f"task-{uuid4()}"

    await store.save(
        Task(
            id=task_a,
            sessionId=session_a,
            status=TaskStatus(state=TaskState.SUBMITTED),
            history=[Message(role="user", parts=[TextPart(text="build the thing")])],
            metadata={"task_type": "audit", "user_id": user_a, "marker": "first"},
        )
    )
    await store.save(
        Task(
            id=task_b,
            sessionId=session_a,
            status=TaskStatus(state=TaskState.COMPLETED),
            metadata={"task_type": "audit", "user_id": user_a, "marker": "second"},
        )
    )
    await store.save(
        Task(
            id=other_task,
            sessionId=session_b,
            status=TaskStatus(state=TaskState.SUBMITTED),
            metadata={"task_type": "audit", "user_id": user_b, "marker": "other"},
        )
    )

    await store.update_status(
        task_a,
        TaskStatus(
            state=TaskState.WORKING,
            message=Message(role="agent", parts=[TextPart(text="underway")]),
        ),
    )
    await store.add_artifact(
        task_a,
        Artifact(
            name="result.txt",
            parts=[TextPart(text="semantic parity")],
        ),
    )

    retrieved = await store.get(task_a)
    assert retrieved is not None
    assert retrieved.sessionId == session_a
    assert retrieved.status.state == TaskState.WORKING
    assert retrieved.status.message is not None
    assert retrieved.status.message.parts[0].text == "underway"
    assert retrieved.history is not None
    assert retrieved.history[0].parts[0].text == "build the thing"
    assert retrieved.artifacts is not None
    assert retrieved.artifacts[0].parts[0].text == "semantic parity"
    assert retrieved.metadata == {"task_type": "audit", "user_id": user_a, "marker": "first"}

    session_tasks = await store.list_tasks(session_id=session_a, user_id=user_a, limit=10)
    assert {task.id for task in session_tasks} == {task_a, task_b}
    assert {task.metadata["marker"] for task in session_tasks} == {"first", "second"}

    working_tasks = await store.list_tasks(user_id=user_a, status=TaskState.WORKING, limit=10)
    assert [task.id for task in working_tasks] == [task_a]

    assert await store.delete(task_a) is True
    assert await store.get(task_a) is None
