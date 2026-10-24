"""TCK business examples are persisted by the real native task lifecycle."""

from __future__ import annotations

from dataclasses import replace

import pytest
import pytest_asyncio
from kestrel_sdk.native_a2a.helpers.proto_helpers import new_text_message
from kestrel_sdk.native_a2a.types import Role, SendMessageRequest, TaskState
from google.protobuf.json_format import MessageToDict

from kestrel_sovereign.a2a.scope import NativeRecipientScope
from tests.utils.native_a2a import manager_for
from tests.utils.native_tck_business import execute_tck_task


@pytest_asyncio.fixture
async def manager(native_a2a_backend_factory):
    backend = native_a2a_backend_factory()
    await backend.connect()
    instance = manager_for(backend, NativeRecipientScope("did:kestrel:recipient"))
    try:
        await instance.initialize()
        yield instance
    finally:
        await instance.close()


async def admit(manager, prefix):
    request = SendMessageRequest(message=new_text_message(
        "TCK artifact test", role=Role.ROLE_USER,
    ))
    request.message.message_id = f"tck-{prefix}-example"
    task = await manager.create_task(request, creator_agent_id="did:kestrel:foreign")
    snapshot = await manager.get_task_snapshot(
        task.id, recipient_agent_id="did:kestrel:recipient",
    )
    assert snapshot is not None
    return snapshot


@pytest.mark.parametrize(("prefix", "part"), [
    ("artifact-text", {"text": "Generated text content"}),
    ("artifact-file", {
        "raw": "R2VuZXJhdGVkIGZpbGUgY29udGVudA==", "filename": "output.txt",
        "mediaType": "text/plain",
    }),
    ("artifact-file-url", {
        "url": "https://example.com/output.txt", "filename": "output.txt",
        "mediaType": "text/plain",
    }),
    ("artifact-data", {"data": {"key": "value", "count": 42}}),
    ("complete-task", None),
    # A partial prefix must not select another scenario.
    ("artifact-textual", None),
])
async def test_tck_scenario_completes_the_original_task_with_native_artifacts(manager, prefix, part):
    snapshot = await admit(manager, prefix)
    await execute_tck_task(snapshot, manager=manager)
    task = await manager.get_task_for_creator(
        snapshot.task.id, "did:kestrel:foreign", recipient_agent_id="did:kestrel:recipient",
    )
    assert task.id == snapshot.task.id
    assert task.status.state == TaskState.TASK_STATE_COMPLETED
    assert task.status.message.parts[0].text == "Hello from TCK"
    if part is None:
        assert not task.artifacts
    else:
        assert len(task.artifacts) == 1
        artifact = task.artifacts[0]
        assert artifact.artifact_id
        assert len(artifact.parts) == 1
        assert MessageToDict(artifact.parts[0]) == part


async def test_tck_business_refuses_missing_original_input_without_finishing_a_task(manager):
    snapshot = await admit(manager, "artifact-text")
    missing = replace(
        snapshot, input_request_json=None, input_request_digest=None, input_wire_request_digest=None,
    )
    with pytest.raises(ValueError, match="original admitted request"):
        await execute_tck_task(missing, manager=manager)
    task = await manager.get_task_for_recipient(snapshot.task.id, "did:kestrel:recipient")
    assert task.status.state != TaskState.TASK_STATE_COMPLETED
    assert not task.artifacts


@pytest.mark.parametrize(("prefix", "state"), [
    ("working", TaskState.TASK_STATE_WORKING),
    ("input-required", TaskState.TASK_STATE_INPUT_REQUIRED),
    ("reject-task", TaskState.TASK_STATE_REJECTED),
])
async def test_tck_business_preserves_the_required_noncompletion_state(manager, prefix, state):
    snapshot = await admit(manager, prefix)
    await execute_tck_task(snapshot, manager=manager)
    task = await manager.get_task_for_recipient(snapshot.task.id, "did:kestrel:recipient")
    assert task.status.state == state
    assert not task.artifacts


async def test_tck_input_required_task_can_finish_through_the_original_native_continuation(manager):
    original = await admit(manager, "input-required")
    await execute_tck_task(original, manager=manager)
    continuation = SendMessageRequest(message=new_text_message(
        "Here is the next input", role=Role.ROLE_USER,
        task_id=original.task.id, context_id=original.task.context_id,
    ))
    continuation.message.message_id = "tck-complete-task-next-input"
    await manager.continue_task(continuation, creator_agent_id="did:kestrel:foreign")
    snapshot = await manager.get_task_snapshot(
        original.task.id, recipient_agent_id="did:kestrel:recipient",
    )
    await execute_tck_task(snapshot, manager=manager)
    task = await manager.get_task_for_recipient(original.task.id, "did:kestrel:recipient")
    assert task.status.state == TaskState.TASK_STATE_COMPLETED
    assert task.id == original.task.id
    assert task.context_id == original.task.context_id
