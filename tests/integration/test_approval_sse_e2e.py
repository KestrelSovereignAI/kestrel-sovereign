"""
End-to-end test for the approval-request SSE path with real infrastructure.

Uses the real `ApprovalQueue`, real `EventManagerMixin`, real FastAPI
routing through starlette's ASGI machinery, and the real `POST
/api/security/approve` endpoint. The only thing that differs from a
standalone server process is that we drive the notifications_sse
generator directly instead of going through uvicorn's HTTP layer —
every line of application logic under test is the exact code that
runs in production.

Regression for #748. Before the fix, request_approval() would queue
correctly and fire _emit_approval → emit_event → iterate
_event_listeners (empty) → event dropped. This test fails if the SSE
stream fails to deliver the approval_request event or if the approve
endpoint fails to resolve the caller.
"""

import asyncio
import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.requests import Request

from endpoints.agent import _sse_connections, notifications_sse, router as agent_router
from endpoints.security import router as security_router
from kestrel_sovereign.agent.event_manager import EventManagerMixin
from kestrel_sovereign.features.security.approval_queue import ApprovalQueue


class _ApprovalAgent(EventManagerMixin):
    """Minimal KestrelAgent stand-in with the real event bus + approval queue."""

    def __init__(self):
        self.agent_id = "did:test:approval-e2e"
        self._event_listeners = []
        self._pending_task_notifications = []
        self.approval_queue = ApprovalQueue(on_request_added=self._emit_approval)
        self.features = {}

    async def _emit_approval(self, request):
        await self.emit_event(
            "approval_request",
            {
                "id": request.id,
                "feature": request.feature_name,
                "tool": request.tool_name,
                "args": request.tool_args,
                "timestamp": request.created_at.isoformat(),
            },
        )

    def get_pending_notifications(self):
        return []


class _SecurityFeatureShim:
    """Minimal shim exposing the attributes endpoints/security reads."""

    def __init__(self, agent: _ApprovalAgent):
        self.approval_queue = agent.approval_queue
        self.permission_store = None


@pytest.fixture(autouse=True)
def clean_sse_connections():
    _sse_connections.clear()
    yield
    _sse_connections.clear()


def _build_request(app, disconnect_event: asyncio.Event):
    """Build a real starlette Request whose receive() returns http.disconnect
    once `disconnect_event` is set."""

    async def receive():
        if disconnect_event.is_set():
            return {"type": "http.disconnect"}
        await asyncio.sleep(0.05)
        if disconnect_event.is_set():
            return {"type": "http.disconnect"}
        return {"type": "http.request", "body": b"", "more_body": False}

    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "GET",
        "scheme": "http",
        "path": "/api/agent/notifications/sse",
        "raw_path": b"/api/agent/notifications/sse",
        "query_string": b"",
        "root_path": "",
        "headers": [(b"host", b"testserver")],
        "client": ("testclient", 12345),
        "server": ("testserver", 80),
        "app": app,
        "state": {},
    }
    return Request(scope, receive=receive)


@pytest.mark.asyncio
async def test_approval_request_reaches_sse_and_decision_resolves_caller():
    app = FastAPI()
    app.include_router(agent_router)
    app.include_router(security_router)

    agent = _ApprovalAgent()
    agent.features = {"SecurityFeature": _SecurityFeatureShim(agent)}
    app.state.agent = agent

    disconnect = asyncio.Event()
    request = _build_request(app, disconnect)

    response = await notifications_sse(request)

    async def stream_reader():
        """Accumulate SSE payload until we see approval_request."""
        buffer = ""
        current_event = None
        async for chunk in response.body_iterator:
            if isinstance(chunk, bytes):
                chunk = chunk.decode("utf-8")
            buffer += chunk
            # Parse SSE events as they arrive
            while "\n\n" in buffer:
                block, buffer = buffer.split("\n\n", 1)
                event_type = None
                data_line = None
                for line in block.splitlines():
                    if line.startswith("event: "):
                        event_type = line[len("event: "):].strip()
                    elif line.startswith("data: "):
                        data_line = line[len("data: "):]
                if event_type == "approval_request" and data_line:
                    return json.loads(data_line)

    reader_task = asyncio.create_task(stream_reader())

    # Let the reader start and register the listener.
    await asyncio.sleep(0.1)

    # Fire a real approval request through the queue.
    async def caller():
        return await agent.approval_queue.request_approval(
            feature_name="ComputeFeature",
            tool_name="run_script",
            tool_args={"script_id": "s-test-748"},
            timeout=5.0,
        )

    caller_task = asyncio.create_task(caller())

    # SSE must deliver the event.
    payload = await asyncio.wait_for(reader_task, timeout=5.0)
    assert payload["feature"] == "ComputeFeature"
    assert payload["tool"] == "run_script"
    assert payload["args"] == {"script_id": "s-test-748"}
    request_id = payload["id"]

    # Simulate the user clicking "Approve (session)" in the modal.
    # TestClient drives a sync request through the same FastAPI app.
    with TestClient(app) as client:
        approve_response = client.post(
            "/api/security/approve",
            json={"approval_id": request_id, "approved": True, "scope": "session"},
        )
        assert approve_response.status_code == 200, approve_response.text
        assert approve_response.json()["approved"] is True

    # Caller blocking on request_approval must unblock with the approval.
    approved, scope = await asyncio.wait_for(caller_task, timeout=2.0)
    assert approved is True
    assert scope == "session"

    # Close the SSE stream cleanly.
    disconnect.set()
    # Drain any remaining body_iterator frames so cleanup runs.
    try:
        async for _ in response.body_iterator:
            pass
    except Exception:
        pass
    # Listener should have been removed by the finally block.
    assert agent._event_listeners == []
