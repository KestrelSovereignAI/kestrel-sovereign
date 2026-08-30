"""Durable-principal scoping for A2A reads and subscriptions (#3145)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from slowapi import Limiter
from slowapi.util import get_remote_address

from kestrel_sdk.tools.result import ToolResultStatus
import kestrel_sovereign.endpoints.agent as agent_endpoint
from kestrel_sovereign.a2a.task_manager import create_task_manager
from kestrel_sovereign.a2a.types import (
    Message,
    TaskSendParams,
    TaskState,
    TaskStatus,
    TextPart,
)
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


def _principal_action_body(task_id: str, verb: str) -> dict:
    return {
        "id": task_id,
        "sessionId": f"a2a-{verb}:{task_id}",
        "message": {
            "role": "user",
            "parts": [{"type": "text", "text": f"{verb}:{task_id}"}],
        },
        "metadata": {
            "sender": CREATOR_A,
            "a2a_verb": verb,
            "signature": {"proof": "test"},
        },
    }


def _principal_endpoint_app(monkeypatch, agent) -> FastAPI:
    limiter = Limiter(key_func=get_remote_address)
    monkeypatch.setattr(agent_endpoint, "limiter", limiter)
    app = FastAPI()
    app.state.limiter = limiter

    @app.middleware("http")
    async def attach_agent(request, call_next):
        request.state.agent = agent
        return await call_next(request)

    app.include_router(agent_endpoint.router)
    return app


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
async def test_subscription_rereads_after_registration_to_close_terminal_race(
    tmp_path, monkeypatch
):
    manager = await create_task_manager(str(tmp_path / "subscription-race.db"))
    try:
        working = await manager.create_task(
            _params("fast-terminal"),
            agent_name=RECIPIENT_A,
            creator_agent_id=CREATOR_A,
        )
        terminal = working.model_copy(deep=True)
        terminal.status = TaskStatus(state=TaskState.COMPLETED)
        scoped_read = AsyncMock(side_effect=[working, terminal])
        monkeypatch.setattr(manager.task_store, "get_for_creator", scoped_read)

        stream = manager.subscribe(
            "fast-terminal",
            creator_agent_id=CREATOR_A,
        )
        event = await anext(stream)

        assert event["final"] is True
        assert '"state":"completed"' in event["data"]
        assert scoped_read.await_count == 2
        await stream.aclose()
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


def test_signed_http_read_uses_verified_creator_principal(monkeypatch):
    task = SimpleNamespace(
        id="wire-read",
        status=SimpleNamespace(state=TaskState.COMPLETED, message=None),
        artifacts=[],
        metadata={"private": "creator-only"},
    )
    manager = SimpleNamespace(
        get_task_for_creator=AsyncMock(return_value=task),
    )
    agent = SimpleNamespace(
        agent_id=RECIPIENT_A,
        did=RECIPIENT_A,
        task_manager=manager,
    )

    async def verify(_agent, params, _parts, _raw, _artifacts, commit=None):
        assert params.metadata["a2a_verb"] == "read_task"
        assert params.metadata["signature"] == {"proof": "test"}
        return await commit(CREATOR_A)

    monkeypatch.setattr(agent_endpoint, "_create_verified_a2a_task", verify)
    app = _principal_endpoint_app(monkeypatch, agent)

    with TestClient(app) as client:
        response = client.post(
            "/api/agent/tasks/wire-read/read",
            json=_principal_action_body("wire-read", "read_task"),
        )

    assert response.status_code == 200
    assert response.json()["metadata"] == {"private": "creator-only"}
    manager.get_task_for_creator.assert_awaited_once_with(
        "wire-read",
        CREATOR_A,
    )


def test_signed_http_subscription_uses_verified_creator_principal(monkeypatch):
    task = SimpleNamespace(
        id="wire-subscribe",
        status=SimpleNamespace(state=TaskState.SUBMITTED, message=None),
        artifacts=[],
        metadata={},
    )
    subscribe_calls = []

    async def subscribe(task_id, **kwargs):
        subscribe_calls.append((task_id, kwargs))
        yield {
            "event": "status",
            "data": '{"id":"wire-subscribe","final":true}',
            "final": True,
        }

    manager = SimpleNamespace(
        get_task_for_creator=AsyncMock(return_value=task),
        subscribe=subscribe,
    )
    agent = SimpleNamespace(
        agent_id=RECIPIENT_A,
        did=RECIPIENT_A,
        task_manager=manager,
    )

    async def verify(_agent, params, _parts, _raw, _artifacts, commit=None):
        assert params.metadata["a2a_verb"] == "subscribe_task"
        return await commit(CREATOR_A)

    monkeypatch.setattr(agent_endpoint, "_create_verified_a2a_task", verify)
    app = _principal_endpoint_app(monkeypatch, agent)

    with TestClient(app) as client:
        with client.stream(
            "POST",
            "/api/agent/tasks/wire-subscribe/subscribe",
            json=_principal_action_body("wire-subscribe", "subscribe_task"),
        ) as response:
            body = "".join(response.iter_text())

    assert response.status_code == 200
    assert "wire-subscribe" in body
    assert subscribe_calls == [
        ("wire-subscribe", {"creator_agent_id": CREATOR_A})
    ]
