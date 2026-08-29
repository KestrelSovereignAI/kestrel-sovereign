"""Recipient-owned A2A response and artifact mutations (#3144)."""

import asyncio
from datetime import datetime
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
from kestrel_sovereign.a2a.task_worker import SimpleTaskHandler, TaskResult, TaskWorker
from kestrel_sovereign.a2a.types import (
    Artifact,
    DataPart,
    Message,
    Task,
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


def _feature_with_agent_id(manager, agent_id: str) -> TaskFeature:
    feature = TaskFeature(SimpleNamespace(agent_id=agent_id))
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
async def test_unauthorized_failure_does_not_write_victim_observability(tmp_path):
    """Rejected task mutations must not leave attacker-authored projections."""

    manager = await create_task_manager(str(tmp_path / "failure-observability.db"))
    try:
        await manager.create_task(
            _params("unauthorized-failure"),
            agent_name=RECIPIENT,
            creator_agent_id=CREATOR,
        )

        with pytest.raises(TaskMutationAuthorizationError):
            await manager.fail_task(
                "unauthorized-failure",
                "forged failure",
                agent_name=PEER,
                recipient_agent_id=PEER,
            )

        task = await manager.get_task("unauthorized-failure")
        assert task.status.state is TaskState.SUBMITTED
        error_events = await manager.observability_store._backend.fetch_all(
            """
            SELECT agent_name, session_id, error_message
            FROM a2a_observability
            WHERE event_type = 'error'
            """
        )
        assert error_events == []

        failed = await manager.fail_task(
            "unauthorized-failure",
            "recipient failure",
            agent_name=RECIPIENT,
            recipient_agent_id=RECIPIENT,
        )
        assert failed.status.state is TaskState.FAILED
        error_events = await manager.observability_store._backend.fetch_all(
            """
            SELECT agent_name, session_id, error_message
            FROM a2a_observability
            WHERE event_type = 'error'
            """
        )
        assert error_events == [
            (
                RECIPIENT,
                "session-unauthorized-failure",
                "recipient failure",
            )
        ]
    finally:
        await manager.close()


@pytest.mark.asyncio
async def test_agent_id_only_host_can_authorize_response_and_artifact(tmp_path):
    """A supported durable identity must not degrade to the host class name."""

    manager = await create_task_manager(str(tmp_path / "agent-id-only.db"))
    try:
        await manager.create_task(
            _params("agent-id-only-response"),
            agent_name=RECIPIENT,
            creator_agent_id=CREATOR,
        )
        response = await _feature_with_agent_id(
            manager,
            RECIPIENT,
        ).respond_to_a2a_task(
            "agent-id-only-response",
            "recipient answer",
        )

        await manager.create_task(
            _params("agent-id-only-artifact"),
            agent_name=RECIPIENT,
            creator_agent_id=CREATOR,
        )
        artifact = await _feature_with_agent_id(
            manager,
            RECIPIENT,
        ).attach_artifact_to_a2a_task(
            "agent-id-only-artifact",
            "recipient-output",
            "persist this",
        )

        assert response.status is ToolResultStatus.OK
        assert artifact.status is ToolResultStatus.OK
    finally:
        await manager.close()


@pytest.mark.asyncio
async def test_response_and_artifact_fail_closed_without_durable_identity(tmp_path):
    """A display/class name is never promoted to mutation authority."""

    manager = await create_task_manager(str(tmp_path / "missing-identity.db"))
    try:
        for task_id in ("identityless-response", "identityless-artifact"):
            await manager.create_task(
                _params(task_id),
                agent_name="SimpleNamespace",
                creator_agent_id=CREATOR,
            )
        feature = TaskFeature(SimpleNamespace())
        feature.set_task_manager(manager)

        response = await feature.respond_to_a2a_task(
            "identityless-response",
            "must not persist",
        )
        artifact = await feature.attach_artifact_to_a2a_task(
            "identityless-artifact",
            "forged-output",
            "must not persist",
        )

        assert response.status is ToolResultStatus.ERROR
        assert artifact.status is ToolResultStatus.ERROR
        assert "durable identity" in response.error
        assert "durable identity" in artifact.error
        response_task = await manager.get_task("identityless-response")
        artifact_task = await manager.get_task("identityless-artifact")
        assert response_task.status.state is TaskState.SUBMITTED
        assert not artifact_task.artifacts
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
        assert (await manager.task_store._get_unscoped(task.id)).status.state is TaskState.SUBMITTED

        assert await manager.task_store.save_recipient_lifecycle(
            task,
            recipient_agent_id=RECIPIENT,
            expected_state=TaskState.SUBMITTED,
        )
        assert (await manager.task_store._get_unscoped(task.id)).status.state is TaskState.WORKING
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
    manager = SimpleNamespace(
        host_agent_id=RECIPIENT,
        get_pending_tasks=AsyncMock(return_value=[]),
    )
    worker = TaskWorker(
        manager,
        agent_name="Meridian",
        max_concurrent=3,
    )

    await worker._poll_and_process()

    manager.get_pending_tasks.assert_awaited_once_with(
        limit=3,
        recipient_agent_id=RECIPIENT,
    )


def test_worker_fails_closed_without_manager_bound_recipient_identity():
    """A display name, even a DID-shaped one, is not worker authority."""

    manager = SimpleNamespace(
        host_agent_id=None,
        get_pending_tasks=AsyncMock(return_value=[]),
    )

    with pytest.raises(ValueError, match="task_manager.host_agent_id"):
        TaskWorker(manager, agent_name=RECIPIENT)


@pytest.mark.asyncio
async def test_worker_display_name_does_not_replace_durable_recipient_authority(
    tmp_path,
):
    """A display-named worker must still claim and finish its DID-owned work."""

    manager = await create_task_manager(
        str(tmp_path / "worker-display-name.db"),
        host_agent_id=RECIPIENT,
    )
    try:
        await manager.create_task(
            _params("display-named-worker"),
            agent_name=RECIPIENT,
            creator_agent_id=CREATOR,
        )
        worker = TaskWorker(
            manager,
            agent_name="Meridian",
            max_concurrent=1,
        )
        worker.register_handler(
            SimpleTaskHandler(
                lambda _task: TaskResult(
                    success=True,
                    response="finished by the DID recipient",
                )
            )
        )
        worker._semaphore = asyncio.Semaphore(1)

        await worker._poll_and_process()
        await asyncio.gather(*worker._tasks)

        persisted = await manager.get_task("display-named-worker")
        assert persisted.status.state is TaskState.COMPLETED
        assert persisted.status.message.parts[0].text == (
            "finished by the DID recipient"
        )
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


@pytest.mark.asyncio
@pytest.mark.parametrize("sync", [True, False], ids=["sync", "async"])
async def test_handler_nonterminal_outcome_reconciles_live_progress(tmp_path, sync):
    """A handler's requested-input result must survive its WORKING update."""

    manager = await create_task_manager(
        str(tmp_path / f"handler-input-{sync}.db"),
        host_agent_id=RECIPIENT,
    )

    class InputHandler:
        async def handle_task(self, task):
            await manager.update_status(
                task.id,
                TaskState.WORKING,
                agent_name=RECIPIENT,
                recipient_agent_id=RECIPIENT,
            )
            task.status = TaskStatus(
                state=TaskState.INPUT_REQUIRED,
                message=Message(
                    role="agent",
                    parts=[TextPart(text="Which account?")],
                ),
            )
            return task

        def get_skill_for_command(self, command):
            return None

    manager.register_agent(
        AgentCard(
            name="input-worker",
            description="worker that requests input after publishing progress",
            url="/agents/input-worker",
            version="1.0.0",
            capabilities=AgentCapabilities(),
            skills=[
                AgentSkill(
                    id="ask",
                    name="ask",
                    description="request required input",
                )
            ],
        ),
        InputHandler(),
    )
    notify_status = AsyncMock(wraps=manager._notify_status_update)
    manager._notify_status_update = notify_status
    try:
        returned = await manager.execute_skill("input-worker", "ask", {}, sync=sync)
        if not sync:
            await manager.drain_execution_tasks()

        persisted = await manager.get_task(returned.id)
        assert persisted is not None
        assert persisted.status.state is TaskState.INPUT_REQUIRED
        assert persisted.status.message.parts[0].text == "Which account?"
        if sync:
            assert returned.status.state is TaskState.INPUT_REQUIRED
        else:
            input_notifications = [
                call
                for call in notify_status.await_args_list
                if call.args[0].status.state is TaskState.INPUT_REQUIRED
            ]
            assert len(input_notifications) == 1
            assert input_notifications[0].kwargs["final"] is False
    finally:
        await manager.close()


@pytest.mark.asyncio
async def test_async_terminal_commit_lost_ack_still_emits_completion_wake(tmp_path):
    """An uncertain terminal commit must retain ownership of its projections."""

    manager = await create_task_manager(
        str(tmp_path / "terminal-lost-ack.db"),
        host_agent_id=RECIPIENT,
    )

    class CompletingHandler:
        async def handle_task(self, task):
            task.status = TaskStatus(
                state=TaskState.COMPLETED,
                message=Message(
                    role="agent",
                    parts=[TextPart(text="committed result")],
                ),
            )
            return task

        def get_skill_for_command(self, command):
            return None

    manager.register_agent(
        AgentCard(
            name="worker",
            description="uncertain commit worker",
            url="/agents/worker",
            version="1.0.0",
            capabilities=AgentCapabilities(),
            skills=[
                AgentSkill(
                    id="finish",
                    name="finish",
                    description="finish before losing the commit acknowledgement",
                )
            ],
        ),
        CompletingHandler(),
    )
    completions: list[str] = []
    manager._on_task_complete = lambda task: completions.append(task.id)
    canonical_save = manager.task_store.save_recipient_terminal_outcome
    first_write = True

    async def commit_then_lose_ack(*args, **kwargs):
        nonlocal first_write
        committed = await canonical_save(*args, **kwargs)
        assert committed is True
        if first_write:
            first_write = False
            raise RuntimeError("lost PostgreSQL COMMIT acknowledgement")
        return committed

    manager.task_store.save_recipient_terminal_outcome = commit_then_lose_ack
    try:
        submitted = await manager.execute_skill("worker", "finish", {}, sync=False)
        terminal_events = asyncio.Queue()
        manager._subscribers[submitted.id] = [terminal_events]

        await manager.drain_execution_tasks()

        persisted = await manager.get_task(submitted.id)
        assert persisted is not None
        assert persisted.status.state is TaskState.COMPLETED
        assert persisted.status.message.parts[0].text == "committed result"
        assert completions == [submitted.id]
        event = terminal_events.get_nowait()
        assert event["event"] == "status"
        assert event["final"] is True
        assert terminal_events.empty()
    finally:
        await manager.close()


@pytest.mark.asyncio
async def test_matching_terminal_token_retains_wake_when_canonical_reread_fails(
    tmp_path,
):
    """A proven terminal commit keeps its wake through a transient row outage."""

    manager = await create_task_manager(
        str(tmp_path / "terminal-lost-ack-read-outage.db"),
        host_agent_id=RECIPIENT,
    )

    class CompletingHandler:
        async def handle_task(self, task):
            task.status = TaskStatus(
                state=TaskState.COMPLETED,
                message=Message(
                    role="agent",
                    parts=[TextPart(text="committed through read outage")],
                ),
            )
            return task

        def get_skill_for_command(self, command):
            return None

    manager.register_agent(
        AgentCard(
            name="worker",
            description="uncertain commit read-outage worker",
            url="/agents/worker",
            version="1.0.0",
            capabilities=AgentCapabilities(),
            skills=[
                AgentSkill(
                    id="finish",
                    name="finish",
                    description="finish while canonical reread is unavailable",
                )
            ],
        ),
        CompletingHandler(),
    )
    completions: list[str] = []
    manager._on_task_complete = lambda task: completions.append(task.id)
    canonical_save = manager.task_store.save_recipient_terminal_outcome
    canonical_read = manager.task_store.get_for_recipient
    first_write = True
    first_read = True

    async def commit_then_lose_ack(*args, **kwargs):
        nonlocal first_write
        committed = await canonical_save(*args, **kwargs)
        assert committed is True
        if first_write:
            first_write = False
            raise RuntimeError("lost terminal COMMIT acknowledgement")
        return committed

    async def lose_first_canonical_reread(*args, **kwargs):
        nonlocal first_read
        if first_read:
            first_read = False
            raise RuntimeError("transient canonical row read outage")
        return await canonical_read(*args, **kwargs)

    manager.task_store.save_recipient_terminal_outcome = commit_then_lose_ack
    manager.task_store.get_for_recipient = lose_first_canonical_reread
    try:
        submitted = await manager.execute_skill("worker", "finish", {}, sync=False)
        terminal_events = asyncio.Queue()
        manager._subscribers[submitted.id] = [terminal_events]

        await manager.drain_execution_tasks()

        persisted = await manager.get_task(submitted.id)
        assert persisted is not None
        assert persisted.status.state is TaskState.COMPLETED
        assert persisted.status.message.parts[0].text == (
            "committed through read outage"
        )
        assert completions == [submitted.id]
        event = terminal_events.get_nowait()
        assert event["event"] == "status"
        assert event["final"] is True
        assert terminal_events.empty()
    finally:
        await manager.close()


@pytest.mark.asyncio
async def test_lost_ack_with_normalized_terminal_payload_still_emits_wake(tmp_path):
    """Persistence normalization cannot veto a matching terminal attempt token."""

    manager = await create_task_manager(
        str(tmp_path / "terminal-normalized-lost-ack.db"),
        host_agent_id=RECIPIENT,
    )

    class CompletingHandler:
        async def handle_task(self, task):
            task.status = TaskStatus(
                state=TaskState.COMPLETED,
                message=Message(role="agent", parts=[TextPart(text="done")]),
            )
            task.artifacts = [
                Artifact(
                    name="normalized-output",
                    parts=[
                        DataPart(data={"when": datetime(2020, 1, 1)})
                    ],
                )
            ]
            return task

        def get_skill_for_command(self, command):
            return None

    manager.register_agent(
        AgentCard(
            name="worker",
            description="normalizing uncertain commit worker",
            url="/agents/worker",
            version="1.0.0",
            capabilities=AgentCapabilities(),
            skills=[
                AgentSkill(
                    id="finish",
                    name="finish",
                    description="finish with a normalized artifact payload",
                )
            ],
        ),
        CompletingHandler(),
    )
    completions: list[str] = []
    manager._on_task_complete = lambda task: completions.append(task.id)
    canonical_save = manager.task_store.save_recipient_terminal_outcome
    first_write = True

    async def commit_then_lose_ack(*args, **kwargs):
        nonlocal first_write
        committed = await canonical_save(*args, **kwargs)
        if first_write:
            first_write = False
            assert committed is True
            raise RuntimeError("lost terminal COMMIT acknowledgement")
        return committed

    manager.task_store.save_recipient_terminal_outcome = commit_then_lose_ack
    try:
        submitted = await manager.execute_skill("worker", "finish", {}, sync=False)
        terminal_events = asyncio.Queue()
        manager._subscribers[submitted.id] = [terminal_events]

        await manager.drain_execution_tasks()

        persisted = await manager.get_task(submitted.id)
        assert persisted.status.state is TaskState.COMPLETED
        assert persisted.artifacts[0].parts[0].data == {
            "when": "2020-01-01 00:00:00"
        }
        assert completions == [submitted.id]
        assert terminal_events.get_nowait()["final"] is True
        assert terminal_events.empty()
    finally:
        await manager.close()


@pytest.mark.asyncio
async def test_uncertain_terminal_write_does_not_claim_different_payload(tmp_path):
    """A competing terminal payload remains the sole notification owner."""

    manager = await create_task_manager(
        str(tmp_path / "terminal-lost-ack-winner.db"),
        host_agent_id=RECIPIENT,
    )

    class CompletingHandler:
        async def handle_task(self, task):
            task.status = TaskStatus(
                state=TaskState.COMPLETED,
                message=Message(role="agent", parts=[TextPart(text="our result")]),
            )
            return task

        def get_skill_for_command(self, command):
            return None

    manager.register_agent(
        AgentCard(
            name="worker",
            description="competing terminal writer",
            url="/agents/worker",
            version="1.0.0",
            capabilities=AgentCapabilities(),
            skills=[
                AgentSkill(
                    id="finish",
                    name="finish",
                    description="lose to a different terminal payload",
                )
            ],
        ),
        CompletingHandler(),
    )
    completions: list[str] = []
    manager._on_task_complete = lambda task: completions.append(task.id)
    canonical_save = manager.task_store.save_recipient_terminal_outcome
    first_write = True

    async def competing_commit_then_error(task, **kwargs):
        nonlocal first_write
        if first_write:
            first_write = False
            attempted_operation_id = kwargs.pop("operation_id", None)
            winner = task.model_copy(deep=True)
            winner.status.message = Message(
                role="agent",
                parts=[TextPart(text="competing result")],
            )
            winner_kwargs = dict(kwargs)
            if attempted_operation_id is not None:
                winner_kwargs["operation_id"] = (
                    f"competing-{attempted_operation_id}"
                )
            assert await canonical_save(winner, **winner_kwargs) is True
            raise RuntimeError("write outcome was uncertain")
        return await canonical_save(task, **kwargs)

    manager.task_store.save_recipient_terminal_outcome = competing_commit_then_error
    try:
        submitted = await manager.execute_skill("worker", "finish", {}, sync=False)
        terminal_events = asyncio.Queue()
        manager._subscribers[submitted.id] = [terminal_events]

        await manager.drain_execution_tasks()

        persisted = await manager.get_task(submitted.id)
        assert persisted.status.state is TaskState.COMPLETED
        assert persisted.status.message.parts[0].text == "competing result"
        assert completions == []
        assert terminal_events.empty()
    finally:
        await manager.close()


@pytest.mark.asyncio
async def test_uncertain_terminal_write_does_not_claim_identical_competing_payload(
    tmp_path,
):
    """Byte-identical output does not prove which terminal attempt committed."""

    manager = await create_task_manager(
        str(tmp_path / "terminal-identical-winner.db"),
        host_agent_id=RECIPIENT,
    )

    class CompletingHandler:
        async def handle_task(self, task):
            task.status = TaskStatus(
                state=TaskState.COMPLETED,
                message=Message(role="agent", parts=[TextPart(text="same result")]),
            )
            return task

        def get_skill_for_command(self, command):
            return None

    manager.register_agent(
        AgentCard(
            name="worker",
            description="identical competing terminal writer",
            url="/agents/worker",
            version="1.0.0",
            capabilities=AgentCapabilities(),
            skills=[
                AgentSkill(
                    id="finish",
                    name="finish",
                    description="lose to an identical terminal payload",
                )
            ],
        ),
        CompletingHandler(),
    )
    completions: list[str] = []
    manager._on_task_complete = lambda task: completions.append(task.id)
    canonical_save = manager.task_store.save_recipient_terminal_outcome
    first_write = True

    async def competing_identical_commit_then_error(task, **kwargs):
        nonlocal first_write
        if first_write:
            first_write = False
            attempted_operation_id = kwargs.pop("operation_id", None)
            winner_kwargs = dict(kwargs)
            if attempted_operation_id is not None:
                winner_kwargs["operation_id"] = (
                    f"competing-{attempted_operation_id}"
                )
            assert await canonical_save(
                task.model_copy(deep=True),
                **winner_kwargs,
            ) is True
            raise RuntimeError("our write failed before commit")
        return await canonical_save(task, **kwargs)

    manager.task_store.save_recipient_terminal_outcome = (
        competing_identical_commit_then_error
    )
    try:
        submitted = await manager.execute_skill("worker", "finish", {}, sync=False)
        terminal_events = asyncio.Queue()
        manager._subscribers[submitted.id] = [terminal_events]

        await manager.drain_execution_tasks()

        persisted = await manager.get_task(submitted.id)
        assert persisted.status.state is TaskState.COMPLETED
        assert persisted.status.message.parts[0].text == "same result"
        assert completions == []
        assert terminal_events.empty()
    finally:
        await manager.close()


@pytest.mark.asyncio
async def test_terminal_commit_reconciliation_propagates_cancellation(tmp_path):
    """Stop during reconciliation must not become a generic execution failure."""

    manager = await create_task_manager(
        str(tmp_path / "terminal-reconcile-cancel.db"),
        host_agent_id=RECIPIENT,
    )
    attempted = Task(
        id="terminal-reconcile-cancel",
        status=TaskStatus(state=TaskState.COMPLETED),
    )
    manager._save_recipient_execution_result = AsyncMock(
        side_effect=RuntimeError("terminal write failed")
    )
    manager.task_store.get_recipient_terminal_operation_id = AsyncMock(
        side_effect=asyncio.CancelledError()
    )
    try:
        with pytest.raises(asyncio.CancelledError):
            await manager._persist_execution_outcome(
                attempted,
                authority_agent_id=RECIPIENT,
                expected_state=TaskState.SUBMITTED,
            )
    finally:
        await manager.close()


def test_task_feature_binds_mutations_to_durable_runtime_identity():
    """Mutation guard: trusted principal comes from the feature's agent."""

    import inspect

    source = inspect.getsource(TaskFeature)
    assert source.count("recipient_agent_id=actor_agent_id") >= 4
    assert 'for attribute in ("did", "agent_id")' in source
    assert "type(self.agent).__name__" not in source
