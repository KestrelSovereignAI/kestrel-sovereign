"""Durable-principal scoping for A2A reads and subscriptions (#3145)."""

from __future__ import annotations

import pytest
from types import SimpleNamespace
from unittest.mock import AsyncMock

from kestrel_sdk.tools.result import ToolResultStatus
from kestrel_sovereign.a2a.task_manager import create_task_manager
from kestrel_sovereign.a2a.types import Message, TaskSendParams, TextPart
from kestrel_sovereign.features.tasks.feature import TaskFeature
from kestrel_sovereign.features.peers.directory import (
    PeerIdentity,
    PeerNotFoundError,
    PeerRequester,
)
from kestrel_sovereign.multi_agent.agent_manager import AgentManager


CREATOR_A = "did:test:creator-a"
CREATOR_B = "did:test:creator-b"
RECIPIENT_A = "did:test:recipient-a"
RECIPIENT_B = "did:test:recipient-b"


def _params(task_id: str) -> TaskSendParams:
    return TaskSendParams(
        id=task_id,
        sessionId=f"session-{task_id}",
        message=Message(role="user", parts=[TextPart(text=f"secret {task_id}")]),
    )


class _Agent:
    def __init__(self, did: str):
        self.did = did


def _feature(manager, did: str) -> TaskFeature:
    feature = TaskFeature(_Agent(did))
    feature.set_task_manager(manager)
    return feature


@pytest.mark.asyncio
async def test_point_and_list_reads_are_scoped_by_durable_role(tmp_path):
    manager = await create_task_manager(str(tmp_path / "principal-reads.db"))
    try:
        await manager.create_task(
            _params("task-a"),
            agent_name=RECIPIENT_A,
            creator_agent_id=CREATOR_A,
        )
        await manager.create_task(
            _params("task-b"),
            agent_name=RECIPIENT_B,
            creator_agent_id=CREATOR_B,
        )

        assert await manager.get_task_for_creator("task-a", CREATOR_A) is not None
        assert await manager.get_task_for_creator("task-a", CREATOR_B) is None
        assert await manager.get_task_for_recipient("task-a", RECIPIENT_A) is not None
        assert await manager.get_task_for_recipient("task-a", RECIPIENT_B) is None

        inbox_a = await manager.list_tasks(recipient_agent_id=RECIPIENT_A)
        pending_b = await manager.get_pending_tasks(
            recipient_agent_id=RECIPIENT_B
        )
        assert [task.id for task in inbox_a] == ["task-a"]
        assert [task.id for task in pending_b] == ["task-b"]
    finally:
        await manager.close()


@pytest.mark.asyncio
async def test_agent_tools_cannot_probe_another_recipient_in_shared_store(tmp_path):
    manager = await create_task_manager(str(tmp_path / "tool-reads.db"))
    try:
        await manager.create_task(
            _params("private-a"),
            agent_name=RECIPIENT_A,
            creator_agent_id=CREATOR_A,
        )
        await manager.create_task(
            _params("private-b"),
            agent_name=RECIPIENT_B,
            creator_agent_id=CREATOR_B,
        )

        denied = await _feature(manager, RECIPIENT_B).check_task_status("private-a")
        own = await _feature(manager, RECIPIENT_A).check_task_status("private-a")
        listing = await _feature(manager, RECIPIENT_A).list_my_tasks()

        assert denied.status is ToolResultStatus.ERROR
        assert own.status is ToolResultStatus.OK
        assert "secret private-a" in own.data["request_content"]
        assert [item["task_id"] for item in listing.data["tasks"]] == ["private-a"]
    finally:
        await manager.close()


@pytest.mark.asyncio
async def test_subscription_admission_is_creator_scoped(tmp_path):
    manager = await create_task_manager(str(tmp_path / "subscription-reads.db"))
    try:
        await manager.create_task(
            _params("creator-stream"),
            agent_name=RECIPIENT_A,
            creator_agent_id=CREATOR_A,
        )

        denied = manager.subscribe(
            "creator-stream",
            creator_agent_id=CREATOR_B,
        )
        with pytest.raises(StopAsyncIteration):
            await anext(denied)

        allowed = manager.subscribe(
            "creator-stream",
            creator_agent_id=CREATOR_A,
        )
        first = await anext(allowed)
        assert first["event"] == "status"
        assert first["final"] is False
        await allowed.aclose()
    finally:
        await manager.close()


@pytest.mark.asyncio
async def test_host_attested_result_read_binds_live_sender_and_recipient(tmp_path):
    task_manager = await create_task_manager(str(tmp_path / "host-read.db"))
    host = AgentManager()
    sender = SimpleNamespace(
        did=CREATOR_A,
        agent_id=CREATOR_A,
        identity=None,
    )
    intruder = SimpleNamespace(
        did=CREATOR_B,
        agent_id=CREATOR_B,
        identity=None,
    )
    recipient = SimpleNamespace(
        did=RECIPIENT_A,
        agent_id=RECIPIENT_A,
        task_manager=task_manager,
    )
    host._register_agent("sender", sender)
    host._register_agent("intruder", intruder)
    host._register_agent("recipient", recipient)
    recipient_router = SimpleNamespace(
        authorize_inbound_sender=AsyncMock(return_value=True)
    )
    recipient_requester = PeerRequester(RECIPIENT_A, object())

    def install(agent):
        requester = PeerRequester(agent.did, object())
        host.install_a2a_hosted_policy(
            agent,
            resolver=object(),
            authorizer=object(),
            router=SimpleNamespace(),
            requester=requester,
        )
        return requester

    sender_requester = install(sender)
    intruder_requester = install(intruder)
    host.install_a2a_hosted_policy(
        recipient,
        resolver=object(),
        authorizer=object(),
        router=recipient_router,
        requester=recipient_requester,
    )
    peer = PeerIdentity(
        agent_id=RECIPIENT_A,
        slug="recipient",
        routing_key="recipient",
    )
    try:
        await task_manager.create_task(
            _params("host-private"),
            agent_name=RECIPIENT_A,
            creator_agent_id=CREATOR_A,
        )

        result = await host.get_host_attested_local_a2a_task(
            sender=sender,
            requester=sender_requester,
            peer=peer,
            task_id="host-private",
        )
        assert result["id"] == "host-private"

        with pytest.raises(PeerNotFoundError):
            await host.get_host_attested_local_a2a_task(
                sender=intruder,
                requester=intruder_requester,
                peer=peer,
                task_id="host-private",
            )
    finally:
        await task_manager.close()
