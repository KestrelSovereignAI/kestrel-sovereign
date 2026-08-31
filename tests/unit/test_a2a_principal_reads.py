"""Durable-principal scoping for A2A reads and subscriptions (#3145)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
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

        assert await manager.get_task_for_creator(
            "task-a", CREATOR_A, recipient_agent_id=RECIPIENT_A
        ) is not None
        assert await manager.get_task_for_creator(
            "task-a", CREATOR_B, recipient_agent_id=RECIPIENT_A
        ) is None
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
async def test_creator_read_is_also_bound_to_routed_recipient(tmp_path):
    """A creator cannot reuse another recipient route in a shared store."""

    manager = await create_task_manager(str(tmp_path / "recipient-route.db"))
    try:
        await manager.create_task(
            _params("recipient-bound"),
            agent_name=RECIPIENT_A,
            creator_agent_id=CREATOR_A,
        )

        assert await manager.get_task_for_creator(
            "recipient-bound",
            CREATOR_A,
            recipient_agent_id=RECIPIENT_A,
        ) is not None
        assert await manager.get_task_for_creator(
            "recipient-bound",
            CREATOR_A,
            recipient_agent_id=RECIPIENT_B,
        ) is None
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
            recipient_agent_id=RECIPIENT_A,
        )
        with pytest.raises(StopAsyncIteration):
            await anext(denied)

        wrong_route = manager.subscribe(
            "creator-stream",
            creator_agent_id=CREATOR_A,
            recipient_agent_id=RECIPIENT_B,
        )
        with pytest.raises(StopAsyncIteration):
            await anext(wrong_route)

        allowed = manager.subscribe(
            "creator-stream",
            creator_agent_id=CREATOR_A,
            recipient_agent_id=RECIPIENT_A,
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
            recipient_agent_id=RECIPIENT_A,
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
    wrong_recipient = SimpleNamespace(
        did=RECIPIENT_B,
        agent_id=RECIPIENT_B,
        task_manager=task_manager,
    )
    host._register_agent("sender", sender)
    host._register_agent("intruder", intruder)
    host._register_agent("recipient", recipient)
    host._register_agent("wrong-recipient", wrong_recipient)
    recipient_router = SimpleNamespace(
        authorize_inbound_sender=AsyncMock(return_value=True)
    )
    recipient_requester = PeerRequester(RECIPIENT_A, object())
    wrong_recipient_router = SimpleNamespace(
        authorize_inbound_sender=AsyncMock(return_value=True)
    )
    wrong_recipient_requester = PeerRequester(RECIPIENT_B, object())

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
    host.install_a2a_hosted_policy(
        wrong_recipient,
        resolver=object(),
        authorizer=object(),
        router=wrong_recipient_router,
        requester=wrong_recipient_requester,
    )
    peer = PeerIdentity(
        agent_id=RECIPIENT_A,
        slug="recipient",
        routing_key="recipient",
    )
    wrong_peer = PeerIdentity(
        agent_id=RECIPIENT_B,
        slug="wrong-recipient",
        routing_key="wrong-recipient",
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
                sender=sender,
                requester=sender_requester,
                peer=wrong_peer,
                task_id="host-private",
            )

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
        recipient_agent_id=RECIPIENT_A,
    )


def test_process_resolver_verifies_signed_http_result_read(
    monkeypatch,
    tmp_path,
):
    import json
    from datetime import datetime, timezone

    from kestrel_sovereign.a2a.did_registry import ProcessA2ADidResolver
    from kestrel_sovereign.a2a.envelope_signing import (
        bound_envelope_fields,
        canonical_message,
        sign_envelope,
    )
    from kestrel_sovereign.identity.did_web import build_verification_methods
    from kestrel_sovereign.identity.hybrid_keypair import generate_hybrid_keypair

    sender_did = "did:web:example.com:process-sender"
    sender_keypair = generate_hybrid_keypair()
    sender_root = tmp_path / "process-sender"
    sender_root.mkdir()
    (sender_root / "sender_did.json").write_text(
        json.dumps(
            {
                "id": sender_did,
                "verificationMethod": build_verification_methods(
                    sender_did,
                    sender_keypair.public_keys(),
                ),
            }
        ),
        encoding="utf-8",
    )
    task = SimpleNamespace(
        id="process-read",
        status=SimpleNamespace(state=TaskState.COMPLETED, message=None),
        artifacts=[],
        metadata={"private": "creator-only"},
    )
    manager = SimpleNamespace(
        get_task_for_creator=AsyncMock(return_value=task),
    )
    resolver = ProcessA2ADidResolver((sender_root,))
    agent = SimpleNamespace(
        agent_id=RECIPIENT_A,
        did=RECIPIENT_A,
        task_manager=manager,
        a2a_did_resolver=resolver.resolve,
    )
    app = _principal_endpoint_app(monkeypatch, agent)
    body = _principal_action_body("process-read", "read_task")
    body["metadata"]["sender"] = sender_did
    body["metadata"]["signature"] = sign_envelope(
        sender_keypair,
        sender=sender_did,
        task_id="process-read",
        message=canonical_message(["read_task:process-read"]),
        timestamp=datetime.now(timezone.utc).isoformat(),
        session_id=body["sessionId"],
        bound=bound_envelope_fields(body["metadata"]),
    )

    with TestClient(app) as client:
        response = client.post(
            "/api/agent/tasks/process-read/read",
            json=body,
        )

    assert response.status_code == 200
    manager.get_task_for_creator.assert_awaited_once_with(
        "process-read",
        sender_did,
        recipient_agent_id=RECIPIENT_A,
    )


@pytest.mark.parametrize(
    "malformed_message",
    [
        "not-an-object",
        [],
        {"role": "user", "parts": "not-a-list"},
        {"role": "user", "parts": ""},
        {"role": "user", "parts": 7},
        {"role": "user", "parts": [7]},
    ],
)
def test_principal_action_rejects_malformed_message_shape(
    monkeypatch,
    malformed_message,
):
    manager = SimpleNamespace(
        get_task_for_creator=AsyncMock(return_value=None),
    )
    agent = SimpleNamespace(
        agent_id=RECIPIENT_A,
        did=RECIPIENT_A,
        task_manager=manager,
    )
    app = _principal_endpoint_app(monkeypatch, agent)
    body = _principal_action_body("wire-read", "read_task")
    body["message"] = malformed_message

    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post(
            "/api/agent/tasks/wire-read/read",
            json=body,
        )

    assert response.status_code == 400
    assert "message" in response.json()["detail"]
    manager.get_task_for_creator.assert_not_awaited()


def test_principal_action_rejects_invalid_message_role_as_client_error(
    monkeypatch,
):
    manager = SimpleNamespace(
        get_task_for_creator=AsyncMock(return_value=None),
    )
    agent = SimpleNamespace(
        agent_id=RECIPIENT_A,
        did=RECIPIENT_A,
        task_manager=manager,
    )
    app = _principal_endpoint_app(monkeypatch, agent)
    body = _principal_action_body("wire-read", "read_task")
    body["message"]["role"] = "operator"

    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post(
            "/api/agent/tasks/wire-read/read",
            json=body,
        )

    assert response.status_code == 400
    assert "Invalid A2A principal action" in response.json()["detail"]
    manager.get_task_for_creator.assert_not_awaited()


def _auth_request(method: str, path: str, headers: dict[str, str]) -> Request:
    return Request(
        {
            "type": "http",
            "http_version": "1.1",
            "method": method,
            "scheme": "http",
            "path": path,
            "raw_path": path.encode("ascii"),
            "query_string": b"",
            "headers": [
                (key.lower().encode("ascii"), value.encode("ascii"))
                for key, value in headers.items()
            ],
            "client": ("127.0.0.1", 10000),
            "server": ("testserver", 80),
        }
    )


@pytest.mark.asyncio
async def test_peer_transport_auth_is_distinct_from_sovereign_task_views(
    monkeypatch,
):
    """Peer transport reaches signed A2A POSTs but never legacy UI reads."""

    from kestrel_sovereign import server

    monkeypatch.setenv("KESTREL_API_KEY", "sovereign-key")
    monkeypatch.setenv("KESTREL_A2A_TRANSPORT_KEY", "peer-transport-key")
    admitted = []

    async def observe_caller(request):
        caller = request.state.caller
        admitted.append(caller)
        return JSONResponse(
            {
                "role": caller.role.value,
                "auth_method": caller.auth_method.value,
            }
        )

    peer_post = await server.auth_middleware(
        _auth_request(
            "POST",
            "/api/agents/recipient/api/agent/tasks/team/queue/task-1/read",
            {"X-Kestrel-A2A-Key": "peer-transport-key"},
        ),
        observe_caller,
    )
    peer_get = await server.auth_middleware(
        _auth_request(
            "GET",
            "/api/agents/recipient/api/agent/tasks/task-1",
            {
                "X-Kestrel-A2A-Key": "peer-transport-key",
                "X-API-Key": "sovereign-key",
            },
        ),
        observe_caller,
    )
    operator_get = await server.auth_middleware(
        _auth_request(
            "GET",
            "/api/agents/recipient/api/agent/tasks/task-1",
            {"X-API-Key": "sovereign-key"},
        ),
        observe_caller,
    )

    assert peer_get.status_code == 403
    assert peer_post.status_code == 200
    assert peer_post.body == b'{"role":"authenticated","auth_method":"a2a_transport"}'
    assert operator_get.status_code == 200
    assert operator_get.body == b'{"role":"sovereign","auth_method":"api_key"}'
    assert len(admitted) == 2


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
        (
            "wire-subscribe",
            {
                "creator_agent_id": CREATOR_A,
                "recipient_agent_id": RECIPIENT_A,
            },
        )
    ]
