"""Recipient-owned A2A response and artifact mutations (#3144)."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from kestrel_sdk.tools.result import ToolResultStatus
from kestrel_sovereign.a2a.agent_card import (
    AgentCapabilities,
    AgentCard,
    AgentSkill,
)
from kestrel_sovereign.a2a.stores.unified.task_store import (
    TaskMutationAuthorizationError,
)
from kestrel_sovereign.a2a.task_manager import create_task_manager
from kestrel_sovereign.a2a.task_worker import TaskWorker
from kestrel_sovereign.a2a.types import (
    Artifact,
    Message,
    TaskSendParams,
    TaskState,
    TaskStatus,
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
        assert (await manager.get_task("recipient-response")).status.state is (
            TaskState.SUBMITTED
        )

        recipient_result = await _feature(manager, RECIPIENT).respond_to_a2a_task(
            "recipient-response",
            "recipient answer",
        )
        assert recipient_result.status is ToolResultStatus.OK
        task = await manager.get_task("recipient-response")
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
        assert not (await manager.get_task("recipient-artifact")).artifacts

        accepted = await _feature(manager, RECIPIENT).attach_artifact_to_a2a_task(
            "recipient-artifact",
            "recipient-output",
            "persist this",
        )
        assert accepted.status is ToolResultStatus.OK
        task = await manager.get_task("recipient-artifact")
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
            expected_state=TaskState.SUBMITTED,
        )
        assert (await manager.get_task(task.id)).status.state is TaskState.SUBMITTED

        assert await manager.task_store.save_recipient_lifecycle(
            task,
            recipient_agent_id=RECIPIENT,
            expected_state=TaskState.SUBMITTED,
        )
        assert (await manager.get_task(task.id)).status.state is TaskState.WORKING
    finally:
        await manager.close()


@pytest.mark.asyncio
async def test_recipient_poll_is_not_starved_by_older_foreign_rows(tmp_path):
    manager = await create_task_manager(str(tmp_path / "worker-poll.db"))
    try:
        await manager.create_task(
            _params("older-foreign"),
            agent_name=PEER,
            creator_agent_id=CREATOR,
        )
        await manager.create_task(
            _params("later-owned"),
            agent_name=RECIPIENT,
            creator_agent_id=CREATOR,
        )

        pending = await manager.get_pending_tasks(
            limit=1,
            recipient_agent_id=RECIPIENT,
        )

        assert [task.id for task in pending] == ["later-owned"]
    finally:
        await manager.close()


@pytest.mark.asyncio
async def test_worker_wires_its_recipient_into_pending_poll():
    manager = SimpleNamespace(get_pending_tasks=AsyncMock(return_value=[]))
    worker = TaskWorker(
        manager,
        agent_name=RECIPIENT,
        max_concurrent=3,
    )

    await worker._poll_and_process()

    manager.get_pending_tasks.assert_awaited_once_with(
        limit=3,
        recipient_agent_id=RECIPIENT,
    )


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
        task = await manager.get_task("status-race")
        assert task.status.state is TaskState.WORKING
        assert task.status.message is None
    finally:
        await manager.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("sync", [True, False], ids=["sync", "async"])
@pytest.mark.parametrize(
    ("concurrent_state", "expected_state"),
    [
        (TaskState.WORKING, TaskState.COMPLETED),
        (TaskState.INPUT_REQUIRED, TaskState.COMPLETED),
        (TaskState.CANCELED, TaskState.CANCELED),
    ],
    ids=["working-progress", "input-progress", "terminal-winner"],
)
async def test_handler_terminal_outcome_reconciles_live_cas_without_replacing_winner(
    tmp_path,
    sync,
    concurrent_state,
    expected_state,
):
    """A live CAS miss is retryable, but an existing terminal result is final."""

    manager = await create_task_manager(
        str(tmp_path / f"handler-cas-{sync}-{concurrent_state.value}.db"),
        host_agent_id=RECIPIENT,
    )
    entered = asyncio.Event()
    release = asyncio.Event()

    class CompletingHandler:
        task = None

        async def handle_task(self, task):
            self.task = task
            entered.set()
            await release.wait()
            task.status = TaskStatus(
                state=TaskState.COMPLETED,
                message=Message(
                    role="agent",
                    parts=[TextPart(text="handler result")],
                ),
            )
            return task

        def get_skill_for_command(self, command):
            return None

    handler = CompletingHandler()
    manager.register_agent(
        AgentCard(
            name="worker",
            description="concurrent lifecycle worker",
            url="/agents/worker",
            version="1.0.0",
            capabilities=AgentCapabilities(),
            skills=[
                AgentSkill(
                    id="finish",
                    name="finish",
                    description="finish after a concurrent lifecycle write",
                )
            ],
        ),
        handler,
    )
    execution = None
    try:
        if sync:
            execution = asyncio.create_task(
                manager.execute_skill("worker", "finish", {}, sync=True)
            )
        else:
            await manager.execute_skill("worker", "finish", {}, sync=False)

        await entered.wait()
        assert handler.task is not None
        if concurrent_state in {TaskState.WORKING, TaskState.INPUT_REQUIRED}:
            await manager.update_status(
                handler.task.id,
                TaskState.WORKING,
                agent_name=RECIPIENT,
                recipient_agent_id=RECIPIENT,
            )
            if concurrent_state is TaskState.INPUT_REQUIRED:
                await manager.update_status(
                    handler.task.id,
                    TaskState.INPUT_REQUIRED,
                    agent_name=RECIPIENT,
                    recipient_agent_id=RECIPIENT,
                )
        else:
            await manager.cancel_task(
                handler.task.id,
                reason="terminal writer won",
                agent_name=RECIPIENT,
            )

        release.set()
        if execution is not None:
            returned = await execution
            assert returned.status.state is expected_state
        else:
            await manager.drain_execution_tasks()

        persisted = await manager.get_task(handler.task.id)
        assert persisted is not None
        assert persisted.status.state is expected_state
        if expected_state is TaskState.COMPLETED:
            assert persisted.status.message.parts[0].text == "handler result"
        else:
            assert persisted.metadata["cancellation_receipt"]["reason"] == (
                "terminal writer won"
            )
    finally:
        release.set()
        if execution is not None and not execution.done():
            execution.cancel()
            await asyncio.gather(execution, return_exceptions=True)
        await manager.close()


def test_task_feature_binds_mutations_to_runtime_did():
    """Mutation guard: trusted principal comes from the feature's agent."""

    import inspect

    source = inspect.getsource(TaskFeature)
    assert source.count("recipient_agent_id=agent_name") >= 4
