"""Native task authority and CAS races across independent backend connections."""

import asyncio
from uuid import uuid4

import pytest
from kestrel_sdk.native_a2a.helpers.proto_helpers import new_text_message
from kestrel_sdk.native_a2a.types import (
    Artifact,
    ListTasksRequest,
    Part,
    Role,
    SendMessageRequest,
    TaskState,
    TaskStatus,
)

from kestrel_sovereign.a2a.native_schema import NativeA2AMigrationRequired
from kestrel_sovereign.a2a.scope import NativeRecipientScope
from kestrel_sovereign.a2a.stores.unified.task_store import (
    TaskMutationAuthorizationError,
)
from tests.utils.native_a2a import manager_for
from tests.utils.native_a2a import (
    native_a2a_backend_factory as native_a2a_backend_factory,  # noqa: PLC0414 - pytest fixture export
)


def request(scope, text="Native work", **metadata):
    return SendMessageRequest(
        tenant=scope.tenant, message=new_text_message(text, role=Role.ROLE_USER), metadata=metadata,
    )


async def test_native_principal_reads_never_cross_recipient_tenant_or_creator(native_a2a_backend_factory):
    first, second = native_a2a_backend_factory(), native_a2a_backend_factory()
    await first.connect()
    await second.connect()
    scope = NativeRecipientScope("did:test:recipient-a", "tenant-a")
    other_scope = NativeRecipientScope("did:test:recipient-b", "tenant-b")
    a = manager_for(first, scope, owns_backend=False)
    b = manager_for(second, other_scope, owns_backend=False)
    try:
        await a.initialize()
        await b.initialize()
        task = await a.create_task(
            request(scope, sender="did:test:forged", recipient="did:test:forged"),
            creator_agent_id="did:test:creator-a",
        )
        other = await b.create_task(request(other_scope), creator_agent_id="did:test:creator-b")
        assert await a.task_store.get_for_creator(task.id, "did:test:creator-a", recipient_agent_id=a.did) == task
        assert await a.task_store.get_for_creator(task.id, "did:test:creator-b", recipient_agent_id=a.did) is None
        assert await a.task_store.get_for_creator(task.id, "did:test:creator-a", recipient_agent_id=b.did) is None
        assert await b.task_store.get_for_principal(task.id, "did:test:creator-a") is None
        assert await a.task_store.get_for_recipient(other.id, b.did) is None
        page = await a.task_store.list_for_creator(
            ListTasksRequest(tenant=scope.tenant), creator_agent_id="did:test:creator-a", recipient_agent_id=a.did,
        )
        assert [item.id for item in page.tasks] == [task.id]
        snapshot = await a.get_task_snapshot(task.id, recipient_agent_id=a.did)
        assert snapshot.ownership.creator_agent_id == "did:test:creator-a"
        assert snapshot.ownership.recipient_agent_id == a.did
    finally:
        await a.close()
        await b.close()
        await first.close()
        await second.close()


async def test_native_recipient_mutation_and_cancel_races_have_one_winner(native_a2a_backend_factory):
    first, second = native_a2a_backend_factory(), native_a2a_backend_factory()
    await first.connect()
    await second.connect()
    scope = NativeRecipientScope("did:test:recipient", "tenant")
    a = manager_for(first, scope, owns_backend=False)
    b = manager_for(second, scope, owns_backend=False)
    creator = "did:test:creator"
    try:
        await a.initialize()
        await b.initialize()
        task = await a.create_task(request(scope), creator_agent_id=creator)
        assert not await b.task_store.update_status(
            task.id, TaskStatus(state=TaskState.TASK_STATE_WORKING),
            recipient_agent_id=creator, expected_state=TaskState.TASK_STATE_SUBMITTED,
        )
        artifact = Artifact(artifact_id=uuid4().hex, name="result", parts=[Part(text="durable output")])
        with pytest.raises(TaskMutationAuthorizationError):
            await b.task_store.add_artifact(task.id, artifact, recipient_agent_id=creator)
        await a.task_store.add_artifact(task.id, artifact, recipient_agent_id=a.did)
        winners = await asyncio.gather(*(
            manager.task_store.update_status(
                task.id, TaskStatus(state=state), recipient_agent_id=a.did,
                expected_state=TaskState.TASK_STATE_SUBMITTED,
            ) for manager, state in ((a, TaskState.TASK_STATE_COMPLETED), (b, TaskState.TASK_STATE_FAILED))
        ))
        assert winners.count(True) == winners.count(False) == 1
        stored = await b.get_task_snapshot(task.id, recipient_agent_id=b.did)
        assert list(stored.task.artifacts) == [artifact]
        assert stored.task.status.state in {TaskState.TASK_STATE_COMPLETED, TaskState.TASK_STATE_FAILED}
        pending = await a.create_task(request(scope), creator_agent_id=creator)
        assert await b.task_store.cancel_if_authorized(pending.id, actor_agent_id="did:test:stranger", reason="forged") is None
        canceled, completed = await asyncio.gather(
            a.task_store.cancel_if_authorized(
                pending.id, actor_agent_id=creator, expected_recipient_agent_id=a.did, reason="requested",
            ),
            b.task_store.update_status(
                pending.id, TaskStatus(state=TaskState.TASK_STATE_COMPLETED),
                recipient_agent_id=b.did, expected_state=TaskState.TASK_STATE_SUBMITTED,
            ),
        )
        assert (canceled is not None) != completed
        stored = await b.get_task_snapshot(pending.id, recipient_agent_id=b.did)
        assert stored.ownership.creator_agent_id == creator and stored.ownership.recipient_agent_id == a.did
        assert stored.task.status.state in {TaskState.TASK_STATE_CANCELED, TaskState.TASK_STATE_COMPLETED}
    finally:
        await a.close()
        await b.close()
        await first.close()
        await second.close()


async def test_native_startup_refuses_draft_rows_before_any_conversion(native_a2a_backend_factory):
    backend, observer = native_a2a_backend_factory(), native_a2a_backend_factory()
    await backend.connect()
    await observer.connect()
    manager = manager_for(backend, NativeRecipientScope("did:test:recipient"), owns_backend=False)
    try:
        await backend.execute("CREATE TABLE a2a_tasks(id TEXT PRIMARY KEY, status TEXT NOT NULL)")
        await backend.execute("INSERT INTO a2a_tasks VALUES ('draft-task', 'working')")
        with pytest.raises(NativeA2AMigrationRequired):
            await manager.initialize()
        assert await observer.fetch_one("SELECT id, status FROM a2a_tasks") == ("draft-task", "working")
        assert not await observer.table_exists("a2a_native_store_schema")
        assert not await observer.table_exists("a2a_contexts")
    finally:
        await manager.close()
        await backend.close()
        await observer.close()
