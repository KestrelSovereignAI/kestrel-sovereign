"""Native artifact IDs, append semantics, chunk events and durable refusal."""

import asyncio
from types import SimpleNamespace

import pytest
import pytest_asyncio
from kestrel_sovereign._vendor.a2a.helpers.proto_helpers import new_text_artifact, new_text_message
from kestrel_sovereign._vendor.a2a.server.tasks.task_manager import append_artifact_to_task
from kestrel_sovereign._vendor.a2a.types import (
    AgentSkill,
    Artifact,
    Part,
    Role,
    SendMessageRequest,
    Task,
    TaskArtifactUpdateEvent,
    TaskState,
    TaskStatusUpdateEvent,
)
from kestrel_sovereign._vendor.a2a.utils.errors import InvalidAgentResponseError, InvalidParamsError
from google.protobuf.json_format import MessageToDict
from kestrel_sdk.tools.result import ToolResultStatus

from kestrel_sovereign.a2a.scope import NativeRecipientScope
from kestrel_sovereign.features.tasks.feature import TaskFeature
from tests.utils.native_a2a import manager_for


@pytest_asyncio.fixture
async def artifacts(native_a2a_backend_factory):
    backend = native_a2a_backend_factory()
    await backend.connect()
    scope = NativeRecipientScope("did:kestrel:recipient", "tenant")
    manager = manager_for(backend, scope)
    feature = TaskFeature(SimpleNamespace(did=scope.recipient_id))
    feature.set_task_manager(manager)
    try:
        await manager.initialize()
        task = await manager.create_task(
            SendMessageRequest(
                tenant="tenant",
                message=new_text_message("request", role=Role.ROLE_USER),
            ),
            creator_agent_id="did:kestrel:sender",
        )
        yield manager, feature, task
    finally:
        await manager.close()


async def test_tool_chunks_use_native_ids_events_and_one_durable_artifact(artifacts):
    manager, feature, task = artifacts
    stream = manager.subscribe(task.id, recipient_agent_id=manager.did)
    client_task = await anext(stream)
    try:
        first = await feature.attach_artifact_to_a2a_task(
            task.id, "reply", "first", last_chunk=False
        )
        assert first.status is ToolResultStatus.OK
        artifact_id = first.data["artifact_id"]
        event = await anext(stream)
        assert isinstance(event, TaskArtifactUpdateEvent)
        assert event.artifact.artifact_id == artifact_id
        assert not event.append and not event.last_chunk
        append_artifact_to_task(client_task, event)
        second = await feature.attach_artifact_to_a2a_task(
            task.id,
            "reply",
            "second",
            artifact_id=artifact_id,
            last_chunk=True,
        )
        assert second.status is ToolResultStatus.OK
        assert second.data["artifact_id"] == artifact_id
        event = await anext(stream)
        assert event.append and event.last_chunk
        append_artifact_to_task(client_task, event)
        stored = await manager.get_task_for_recipient(task.id, manager.did)
        assert len(stored.artifacts) == 1
        assert list(stored.artifacts) == list(client_task.artifacts)
        assert [part.text for part in stored.artifacts[0].parts] == ["first", "second"]
        assert "index" not in MessageToDict(stored.artifacts[0])
        assert "lastChunk" not in MessageToDict(stored.artifacts[0])
    finally:
        await stream.aclose()


async def test_sdk_append_preserves_json_metadata_and_replace_does_not_duplicate_id(
    artifacts,
):
    manager, _, task = artifacts
    first = new_text_artifact("reply", "first")
    first.metadata.update({"original": {"nested": [None, {}, []]}, "empty": ""})
    await manager.add_artifact(
        task.id, first, recipient_agent_id=manager.did, last_chunk=False
    )
    second = new_text_artifact("reply", "second", artifact_id=first.artifact_id)
    second.metadata.update({"second": [False, 0, None, {}]})
    before = second.SerializeToString(deterministic=True)
    stored = await manager.add_artifact(
        task.id, second, recipient_agent_id=manager.did, append=True
    )
    assert second.SerializeToString(deterministic=True) == before
    assert MessageToDict(stored.artifacts[0].metadata) == {
        "original": {"nested": [None, {}, []]},
        "empty": "",
        "second": [False, 0, None, {}],
    }
    replacement = new_text_artifact(
        "replacement", "whole new body", artifact_id=first.artifact_id
    )
    stored = await manager.add_artifact(
        task.id, replacement, recipient_agent_id=manager.did, append=False
    )
    assert len(stored.artifacts) == 1
    assert stored.artifacts[0] == replacement


async def test_unknown_append_is_a_protocol_error_without_a_write_or_event(artifacts):
    manager, feature, task = artifacts
    stream = manager.subscribe(task.id, recipient_agent_id=manager.did)
    await anext(stream)
    try:
        with pytest.raises(InvalidAgentResponseError, match="nonexistent artifact_id"):
            await manager.add_artifact(
                task.id,
                new_text_artifact("missing", "bad", artifact_id="not-created"),
                recipient_agent_id=manager.did,
                append=True,
            )
        result = await feature.attach_artifact_to_a2a_task(
            task.id, "missing", "bad", artifact_id="not-created"
        )
        assert result.status is ToolResultStatus.ERROR
        stored = await manager.get_task_for_recipient(task.id, manager.did)
        assert stored == task
        assert manager._subscribers[task.id][0].empty()
    finally:
        await stream.aclose()


@pytest.mark.parametrize("field,value", [("append", "true"), ("last_chunk", 1)])
async def test_chunk_flags_fail_before_mutation(artifacts, field, value):
    manager, _, task = artifacts
    with pytest.raises(TypeError, match="booleans"):
        await manager.add_artifact(
            task.id,
            new_text_artifact("reply", "bad"),
            recipient_agent_id=manager.did,
            **{field: value},
        )
    assert await manager.get_task_for_recipient(task.id, manager.did) == task


@pytest.mark.parametrize(
    "value,error",
    [
        (Task(), TypeError),
        (Artifact(parts=[Part(text="missing id")]), InvalidParamsError),
    ],
)
async def test_store_refuses_wrong_or_incomplete_native_artifacts(
    artifacts, value, error
):
    manager, _, task = artifacts
    with pytest.raises(error):
        await manager.task_store.add_artifact(
            task.id, value, recipient_agent_id=manager.did
        )
    assert await manager.get_task_for_recipient(task.id, manager.did) == task


async def test_foreign_or_terminal_task_refuses_new_chunks(artifacts):
    manager, feature, task = artifacts
    foreign = TaskFeature(SimpleNamespace(did="did:kestrel:other"))
    foreign.set_task_manager(manager)
    assert (
        await foreign.attach_artifact_to_a2a_task(task.id, "bad", "bad")
    ).status is ToolResultStatus.ERROR
    assert (
        await feature.respond_to_a2a_task(task.id, "finished")
    ).status is ToolResultStatus.OK
    result = await feature.attach_artifact_to_a2a_task(task.id, "late", "bad")
    assert result.status is ToolResultStatus.ERROR
    assert (
        list(
            (
                await manager.get_task_for_recipient(task.id, manager.did)
            ).artifacts
        )
        == []
    )


async def test_async_handler_artifacts_reach_stream_before_terminal_status(artifacts):
    manager, _, _ = artifacts

    handler = GatedExport()
    manager.register_provider(
        "exporter",
        [
            AgentSkill(
                id="export", name="Export", description="Export a body", tags=["test"]
            )
        ],
        handler,
    )
    task = await manager.execute_skill(
        agent_id="exporter", skill_id="export", args={}, sync=False
    )
    stream = manager.subscribe(task.id, recipient_agent_id=manager.did)
    try:
        await asyncio.wait_for(handler.started.wait(), 5)
        client_task = await anext(stream)
        handler.release.set()
        event = await asyncio.wait_for(anext(stream), 5)
        assert isinstance(event, TaskArtifactUpdateEvent)
        append_artifact_to_task(client_task, event)
        assert event.last_chunk and not event.append
        final = await asyncio.wait_for(anext(stream), 5)
        assert isinstance(final, TaskStatusUpdateEvent)
        assert final.status.state == TaskState.TASK_STATE_COMPLETED
        assert client_task.artifacts[0].parts[0].text == "complete body"
        with pytest.raises(StopAsyncIteration):
            await anext(stream)
    finally:
        handler.release.set()
        await stream.aclose()
        await manager.drain_execution_tasks()


class GatedExport:
    def __init__(self, *, in_place=False):
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.in_place = in_place

    def get_skill_for_command(self, command):
        return None

    async def handle_task(self, task, *, arguments=None):
        self.started.set()
        await self.release.wait()
        result = task if self.in_place else Task.FromString(task.SerializeToString())
        result.artifacts.append(new_text_artifact("export", "complete body"))
        result.status.state = TaskState.TASK_STATE_COMPLETED
        return result


async def test_handler_outcome_preserves_concurrent_artifact_and_history(artifacts):
    manager, _, _ = artifacts
    handler = GatedExport()
    manager.register_provider(
        "exporter",
        [
            AgentSkill(
                id="export", name="Export", description="Export a body", tags=["test"]
            )
        ],
        handler,
    )
    task = await manager.execute_skill(
        agent_id="exporter", skill_id="export", args={}, sync=False
    )
    try:
        await asyncio.wait_for(handler.started.wait(), 5)
        early = new_text_artifact("already committed", "keep this output")
        await manager.add_artifact(
            task.id, early, recipient_agent_id=manager.did
        )
        progress = new_text_message(
            "durable progress",
            role=Role.ROLE_AGENT,
            task_id=task.id,
            context_id=task.context_id,
        )
        await manager.update_status(
            task.id,
            TaskState.TASK_STATE_WORKING,
            message=progress,
            recipient_agent_id=manager.did,
        )
        handler.release.set()
        await manager.drain_execution_tasks()
        stored = await manager.get_task_for_recipient(task.id, manager.did)
        assert {artifact.name for artifact in stored.artifacts} == {
            "already committed",
            "export",
        }
        assert progress in stored.history
        assert stored.status.state == TaskState.TASK_STATE_COMPLETED
    finally:
        handler.release.set()
        await manager.drain_execution_tasks()


async def test_in_place_handler_cannot_mutate_the_accepted_acknowledgement(artifacts):
    manager, _, _ = artifacts
    handler = GatedExport(in_place=True)
    manager.register_provider(
        "exporter",
        [AgentSkill(id="export", name="Export", description="Export", tags=["test"])],
        handler,
    )
    acknowledgement = await manager.execute_skill("exporter", "export", {}, sync=False)
    before = acknowledgement.SerializeToString(deterministic=True)
    try:
        await asyncio.wait_for(handler.started.wait(), 5)
        handler.release.set()
        await manager.drain_execution_tasks()
        assert acknowledgement.SerializeToString(deterministic=True) == before
        assert acknowledgement.status.state == TaskState.TASK_STATE_SUBMITTED
        stored = await manager.get_task_for_recipient(
            acknowledgement.id, manager.did
        )
        assert stored.status.state == TaskState.TASK_STATE_COMPLETED
        assert len(stored.artifacts) == 1
        assert stored.artifacts[0].name == "export"
    finally:
        handler.release.set()
        await manager.drain_execution_tasks()
