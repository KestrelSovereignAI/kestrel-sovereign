"""``POST /api/agent/tasks/send`` — send-side artifact ingress (#1525).

A sender may attach durable handoff payload (planning docs, evidence
bundles, saved-memory/recall references, logs, diffs) to an outgoing
A2A task at creation time. The receiving endpoint validates the
``artifacts`` list and persists them on the task at SUBMITTED so the
recipient can retrieve them from the task store before producing any
response. This is the send-side mirror of the responder-side
``attach_artifact_to_a2a_task`` flow.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from slowapi import Limiter
from slowapi.util import get_remote_address

import kestrel_sovereign.endpoints.agent as agent_endpoint
from kestrel_sovereign.a2a.types import (
    Task,
    TaskState,
    TaskStatus,
)


@pytest.fixture
def app_with_send(monkeypatch):
    limiter = Limiter(key_func=get_remote_address)
    monkeypatch.setattr(agent_endpoint, "limiter", limiter)

    app = FastAPI()
    app.state.limiter = limiter
    app.include_router(agent_endpoint.router)
    return app


def _stub_agent():
    """Agent whose ``create_task`` echoes a real Task built from the
    params + artifacts it was handed, so the test can assert both the
    call arguments and the serialized response envelope."""
    agent = MagicMock()
    agent.did = "did:test:recipient"
    agent._agent_name = "recipient"
    agent.task_manager = MagicMock()

    async def fake_create_task(*, params, agent_name, artifacts=None):
        return Task(
            id=params.id,
            sessionId=params.sessionId,
            status=TaskStatus(state=TaskState.SUBMITTED),
            history=[params.message],
            metadata=params.metadata or {},
            artifacts=list(artifacts) if artifacts else None,
        )

    agent.task_manager.create_task = AsyncMock(side_effect=fake_create_task)
    return agent


def _attach(app: FastAPI, agent) -> None:
    @app.middleware("http")
    async def _attach_agent(request, call_next):
        request.state.agent = agent
        return await call_next(request)


def _body(**overrides):
    body = {
        "id": "task-1",
        "sessionId": "sess-1",
        "message": {"role": "user", "parts": [{"type": "text", "text": "do it"}]},
        "metadata": {"sender": "emma"},
    }
    body.update(overrides)
    return body


def test_send_task_persists_sender_artifacts(app_with_send):
    agent = _stub_agent()
    _attach(app_with_send, agent)

    body = _body(
        artifacts=[
            {
                "name": "plan",
                "parts": [{"type": "text", "text": "step one"}],
                "index": 0,
                "lastChunk": True,
                "metadata": {"origin": "saved_item"},
            },
            {
                "name": "references",
                "parts": [{"type": "data", "data": {"ref_type": "memory", "id": "m1"}}],
                "index": 0,
                "metadata": {"kind": "reference"},
            },
        ]
    )

    with TestClient(app_with_send) as client:
        resp = client.post("/api/agent/tasks/send", json=body)

    assert resp.status_code == 200
    # create_task received parsed Artifact objects in order.
    call = agent.task_manager.create_task.await_args
    artifacts = call.kwargs["artifacts"]
    assert artifacts is not None and len(artifacts) == 2
    assert artifacts[0].name == "plan"
    assert artifacts[0].metadata == {"origin": "saved_item"}
    assert artifacts[0].parts[0].text == "step one"
    assert artifacts[1].name == "references"
    assert artifacts[1].parts[0].data == {"ref_type": "memory", "id": "m1"}

    # Recipient retrieval: the serialized task envelope carries them in
    # order with structured metadata intact.
    payload = resp.json()
    assert [a["name"] for a in payload["artifacts"]] == ["plan", "references"]
    assert payload["artifacts"][0]["lastChunk"] is True
    assert payload["artifacts"][1]["parts"][0]["data"] == {
        "ref_type": "memory",
        "id": "m1",
    }


def test_send_task_without_artifacts_passes_none(app_with_send):
    agent = _stub_agent()
    _attach(app_with_send, agent)

    with TestClient(app_with_send) as client:
        resp = client.post("/api/agent/tasks/send", json=_body())

    assert resp.status_code == 200
    assert agent.task_manager.create_task.await_args.kwargs["artifacts"] is None


def test_send_task_rejects_non_list_artifacts(app_with_send):
    agent = _stub_agent()
    _attach(app_with_send, agent)

    with TestClient(app_with_send) as client:
        resp = client.post("/api/agent/tasks/send", json=_body(artifacts="nope"))

    assert resp.status_code == 400
    agent.task_manager.create_task.assert_not_awaited()


def test_send_task_rejects_malformed_artifact(app_with_send):
    agent = _stub_agent()
    _attach(app_with_send, agent)

    # parts is required on an Artifact — a missing body must 400, not 500.
    with TestClient(app_with_send) as client:
        resp = client.post(
            "/api/agent/tasks/send", json=_body(artifacts=[{"name": "bad"}])
        )

    assert resp.status_code == 400
    agent.task_manager.create_task.assert_not_awaited()
