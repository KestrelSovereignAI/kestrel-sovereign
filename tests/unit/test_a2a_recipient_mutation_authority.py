"""Recipient-owned A2A response and artifact mutations (#3144)."""

import asyncio
from types import SimpleNamespace

import pytest

from kestrel_sdk.tools.result import ToolResultStatus
from kestrel_sovereign.a2a.stores.unified.task_store import (
    TaskMutationAuthorizationError,
)
from kestrel_sovereign.a2a.task_manager import create_task_manager
from kestrel_sovereign.a2a.types import (
    Artifact,
    Message,
    TaskSendParams,
    TaskState,
    TextPart,
)
from kestrel_sovereign.features.tasks.feature import TaskFeature


CREATOR = "did:test:creator"
RECIPIENT = "did:test:recipient"
PEER = "did:test:unrelated-peer"


def _params(task_id: str) -> TaskSendParams:
    return TaskSendParams(
        id=task_id,
        sessionId=f"session-{task_id}",
        message=Message(role="user", parts=[TextPart(text="do the work")]),
        metadata={
            "sender_agent_id": RECIPIENT,
            "recipient_agent_id": PEER,
            "causation_chain": [{"agent_id": RECIPIENT}],
        },
    )


def _feature(manager, agent_id: str) -> TaskFeature:
    feature = TaskFeature(SimpleNamespace(did=agent_id))
    feature.set_task_manager(manager)
    return feature


@pytest.mark.asyncio
async def test_response_authority_is_recipient_not_creator_or_metadata(tmp_path):
    manager = await create_task_manager(str(tmp_path / "responses.db"))
    try:
        await manager.create_task(
            _params("recipient-response"),
            agent_name=RECIPIENT,
            creator_agent_id=CREATOR,
        )

        creator_result = await _feature(manager, CREATOR).respond_to_a2a_task(
            "recipient-response",
            "creator must not answer",
        )
        peer_result = await _feature(manager, PEER).respond_to_a2a_task(
            "recipient-response",
            "metadata must not delegate",
        )

        assert creator_result.status is ToolResultStatus.ERROR
        assert peer_result.status is ToolResultStatus.ERROR
        assert (await manager.task_store._get_unscoped("recipient-response")).status.state is (
            TaskState.SUBMITTED
        )

        recipient_result = await _feature(manager, RECIPIENT).respond_to_a2a_task(
            "recipient-response",
            "recipient answer",
        )
        assert recipient_result.status is ToolResultStatus.OK
        task = await manager.task_store._get_unscoped("recipient-response")
        assert task.status.state is TaskState.COMPLETED
        assert task.status.message.parts[0].text == "recipient answer"
    finally:
        await manager.close()


@pytest.mark.asyncio
async def test_artifact_authority_is_recipient_and_predicated_in_store(tmp_path):
    manager = await create_task_manager(str(tmp_path / "artifacts.db"))
    try:
        await manager.create_task(
            _params("recipient-artifact"),
            agent_name=RECIPIENT,
            creator_agent_id=CREATOR,
        )

        denied = await _feature(manager, CREATOR).attach_artifact_to_a2a_task(
            "recipient-artifact",
            "creator-output",
            "must not persist",
        )
        assert denied.status is ToolResultStatus.ERROR
        assert not (await manager.task_store._get_unscoped("recipient-artifact")).artifacts

        accepted = await _feature(manager, RECIPIENT).attach_artifact_to_a2a_task(
            "recipient-artifact",
            "recipient-output",
            "persist this",
        )
        assert accepted.status is ToolResultStatus.OK
        task = await manager.task_store._get_unscoped("recipient-artifact")
        assert [artifact.name for artifact in task.artifacts] == [
            "recipient-output"
        ]
    finally:
        await manager.close()


@pytest.mark.asyncio
async def test_worker_lifecycle_has_only_typed_recipient_owned_save(tmp_path):
    manager = await create_task_manager(str(tmp_path / "worker-save.db"))
    try:
        task = await manager.create_task(
            _params("worker-save"),
            agent_name=RECIPIENT,
            creator_agent_id=CREATOR,
        )
        task.status.state = TaskState.WORKING

        with pytest.raises(ValueError, match="save_recipient_lifecycle"):
            await manager.task_store.save(task)
        assert not await manager.task_store.save_recipient_lifecycle(
            task,
            recipient_agent_id=CREATOR,
        )
        assert (await manager.task_store._get_unscoped(task.id)).status.state is TaskState.SUBMITTED

        assert await manager.task_store.save_recipient_lifecycle(
            task,
            recipient_agent_id=RECIPIENT,
        )
        assert (await manager.task_store._get_unscoped(task.id)).status.state is TaskState.WORKING
    finally:
        await manager.close()


@pytest.mark.asyncio
async def test_unauthorized_status_race_cannot_win_or_change_payload(tmp_path):
    manager = await create_task_manager(str(tmp_path / "status-race.db"))
    try:
        await manager.create_task(
            _params("status-race"),
            agent_name=RECIPIENT,
            creator_agent_id=CREATOR,
        )

        accepted, denied = await asyncio.gather(
            manager.update_status(
                "status-race",
                TaskState.WORKING,
                agent_name=RECIPIENT,
                recipient_agent_id=RECIPIENT,
            ),
            manager.update_status(
                "status-race",
                TaskState.FAILED,
                message=Message(
                    role="agent",
                    parts=[TextPart(text="forged failure")],
                ),
                agent_name=RECIPIENT,
                recipient_agent_id=CREATOR,
            ),
            return_exceptions=True,
        )

        assert accepted.status.state is TaskState.WORKING
        assert isinstance(denied, TaskMutationAuthorizationError)
        task = await manager.task_store._get_unscoped("status-race")
        assert task.status.state is TaskState.WORKING
        assert task.status.message is None
    finally:
        await manager.close()


def test_task_feature_binds_mutations_to_runtime_did():
    """Mutation guard: trusted principal comes from the feature's agent."""

    import inspect

    source = inspect.getsource(TaskFeature)
    assert source.count("recipient_agent_id=agent_name") >= 4
