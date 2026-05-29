"""``GET /api/agent/api/agent/tasks/{task_id}/subscribe`` — sender-side push ingress
for the async ``send_a2a_question`` resumption design (#1444).

The receiver-side endpoint wraps ``TaskManager.subscribe(task_id)`` as an
SSE stream so a sender can wait on a question's terminal state without
polling. The first frame is the current state snapshot (so a late
subscriber doesn't miss a terminal that already fired), subsequent
frames stream live updates, the stream closes on the first final event.

Pre-#1444 the only sender-side mechanism was an adaptive backoff polling
loop in ``PeersFeature.send_a2a_question``. The polling burn was the
root cause of the multi-hop chain failure (see #1444 description).
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from slowapi import Limiter
from slowapi.util import get_remote_address

import kestrel_sovereign.endpoints.agent as agent_endpoint


@pytest.fixture
def app_with_subscribe(monkeypatch):
    """Wire just enough of the FastAPI surface to drive the subscribe
    handler against a stub TaskManager. We intentionally do NOT spin up
    the full lifespan / multi-agent host — the handler under test is
    pure: it reads ``request.state.agent``, calls ``task_manager.subscribe``,
    re-emits SSE frames."""
    # Use a fresh limiter so the test process doesn't share rate-limit
    # state with whatever else is using slowapi.
    limiter = Limiter(key_func=get_remote_address)
    monkeypatch.setattr(agent_endpoint, "limiter", limiter)

    app = FastAPI()
    app.state.limiter = limiter
    app.include_router(agent_endpoint.router)
    return app


def _stub_agent_with_task(*, task_present: bool, sse_frames):
    """Build a stub agent whose TaskManager yields ``sse_frames`` from
    subscribe() and ``task_store.get`` returns either a truthy stand-in
    or None per ``task_present``. ``sse_frames`` is an iterable of dicts
    matching the shape ``TaskManager.subscribe`` emits."""
    agent = MagicMock()
    agent.agent_id = "did:test:agent"
    agent.task_manager = MagicMock()
    agent.task_manager.task_store = MagicMock()

    async def fake_get(task_id):
        return MagicMock() if task_present else None

    agent.task_manager.task_store.get = fake_get

    async def fake_subscribe(task_id):
        for frame in sse_frames:
            yield frame

    agent.task_manager.subscribe = fake_subscribe
    return agent


def _set_scope_state_agent(app: FastAPI, agent) -> None:
    """The agent-routing middleware normally attaches the resolved
    agent to ``scope['state']['agent']``. For these handler-level tests
    we bypass by attaching the agent directly to request state via a
    middleware shim. (FastAPI ``dependency_overrides`` won't work here
    because the handler reads ``get_agent(request)`` directly — not as
    a ``Depends(...)`` parameter.)"""

    @app.middleware("http")
    async def _attach_agent(request, call_next):
        request.state.agent = agent
        return await call_next(request)


def test_subscribe_returns_404_when_task_manager_missing(app_with_subscribe):
    agent = MagicMock()
    agent.agent_id = "did:test:agent"
    agent.task_manager = None
    _set_scope_state_agent(app_with_subscribe, agent)

    with TestClient(app_with_subscribe) as client:
        resp = client.get("/api/agent/tasks/abc123/subscribe")
    assert resp.status_code == 404
    assert "TaskManager" in resp.json()["detail"]


def test_subscribe_returns_404_for_unknown_task(app_with_subscribe):
    agent = _stub_agent_with_task(task_present=False, sse_frames=[])
    _set_scope_state_agent(app_with_subscribe, agent)

    with TestClient(app_with_subscribe) as client:
        resp = client.get("/api/agent/tasks/unknown-task-id/subscribe")
    assert resp.status_code == 404
    assert "not found" in resp.json()["detail"]


def test_subscribe_streams_snapshot_then_terminal(app_with_subscribe):
    """Happy path: TaskManager.subscribe yields a current-state snapshot,
    then a terminal status event, then the loop breaks. The endpoint
    must forward both as SSE frames and close the stream cleanly."""
    frames = [
        {
            "event": "status",
            "data": json.dumps(
                {"id": "t1", "status": {"state": "submitted"}, "final": False}
            ),
        },
        {
            "event": "status",
            "data": json.dumps(
                {"id": "t1", "status": {"state": "completed"}, "final": True}
            ),
            "final": True,
        },
    ]
    agent = _stub_agent_with_task(task_present=True, sse_frames=frames)
    _set_scope_state_agent(app_with_subscribe, agent)

    with TestClient(app_with_subscribe) as client:
        with client.stream("GET", "/api/agent/tasks/t1/subscribe") as resp:
            assert resp.status_code == 200
            assert "text/event-stream" in resp.headers["content-type"]
            body = "".join(chunk for chunk in resp.iter_text())

    # Both frames must appear, in order.
    assert "event: status" in body
    submitted_idx = body.index('"state": "submitted"')
    completed_idx = body.index('"state": "completed"')
    assert submitted_idx < completed_idx, (
        "Current-state snapshot must precede the terminal event so a "
        "late subscriber sees the existing state before fresh updates."
    )
    # End-of-stream marker comment fires after the terminal frame.
    eos_idx = body.rfind(": end-of-stream")
    assert eos_idx > completed_idx, (
        "Endpoint must signal end-of-stream cleanly after the final "
        "event so the SSE client knows the subscription is closed."
    )


def test_subscribe_forwards_keepalive_frames(app_with_subscribe):
    """``TaskManager.subscribe`` emits its own ``keepalive`` events when
    no update arrives within its internal timeout. The endpoint must
    forward those as SSE frames so HTTP intermediaries don't idle-close
    the connection between updates."""
    frames = [
        {"event": "keepalive", "data": ""},
        {
            "event": "status",
            "data": json.dumps(
                {"id": "t2", "status": {"state": "completed"}, "final": True}
            ),
            "final": True,
        },
    ]
    agent = _stub_agent_with_task(task_present=True, sse_frames=frames)
    _set_scope_state_agent(app_with_subscribe, agent)

    with TestClient(app_with_subscribe) as client:
        with client.stream("GET", "/api/agent/tasks/t2/subscribe") as resp:
            assert resp.status_code == 200
            body = "".join(chunk for chunk in resp.iter_text())

    assert "event: keepalive" in body, (
        "Keepalive frames from the underlying generator must be forwarded "
        "so the connection stays warm during long sub-task waits."
    )


def test_subscribe_response_headers_match_sse_contract(app_with_subscribe):
    frames = [
        {
            "event": "status",
            "data": json.dumps(
                {"id": "t3", "status": {"state": "completed"}, "final": True}
            ),
            "final": True,
        },
    ]
    agent = _stub_agent_with_task(task_present=True, sse_frames=frames)
    _set_scope_state_agent(app_with_subscribe, agent)

    with TestClient(app_with_subscribe) as client:
        with client.stream("GET", "/api/agent/tasks/t3/subscribe") as resp:
            assert resp.status_code == 200
            assert resp.headers["cache-control"] == "no-cache"
            assert resp.headers["connection"] == "keep-alive"
            assert resp.headers["x-accel-buffering"] == "no", (
                "X-Accel-Buffering: no required so nginx-style proxies "
                "don't buffer SSE frames into batched flushes — the "
                "sender's wake latency depends on terminal events "
                "arriving as they fire, not at proxy flush boundaries."
            )
            # Drain the stream so the test client closes cleanly.
            list(resp.iter_text())
