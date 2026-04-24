"""
Live-server integration test for the approval SSE path (#748).

Boots real uvicorn on a random port, real FastAPI app, real SecurityFeature
with a real ApprovalQueue, then:

    1. Connects a real HTTP client (httpx over a real TCP socket) to the
       `/agent/notifications/sse` endpoint.
    2. Concurrently fires `approval_queue.request_approval()` on the agent
       (same Python process, so we have the object directly — this simulates
       what SecurityHook.execute does on any APPROVAL_REQUIRED tool call).
    3. Reads the real `approval_request` event off the SSE wire.
    4. POSTs the decision to `/api/security/approve` via real HTTP.
    5. Verifies the original `request_approval()` call unblocks with the
       approved decision.

Every layer except the browser EventSource is real production code on real
sockets. The browser layer is covered separately by `tests/e2e/
test_approval_popup.spec.cjs`.
"""

import asyncio
import contextlib
import json
import socket
import threading
import time

import httpx
import pytest
import uvicorn
from fastapi import FastAPI

from endpoints.agent import router as agent_router, _sse_connections
from endpoints.security import router as security_router
from kestrel_sovereign.agent.event_manager import EventManagerMixin
from kestrel_sovereign.features.security.approval_queue import ApprovalQueue


class _ApprovalAgent(EventManagerMixin):
    def __init__(self):
        self.agent_id = "did:test:live-server"
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
    def __init__(self, agent: _ApprovalAgent):
        self.approval_queue = agent.approval_queue
        self.permission_store = None


def _free_port() -> int:
    with contextlib.closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(autouse=True)
def clean_sse_connections():
    _sse_connections.clear()
    yield
    _sse_connections.clear()


@pytest.fixture
def live_server():
    app = FastAPI()
    app.include_router(agent_router)
    app.include_router(security_router)
    agent = _ApprovalAgent()
    agent.features = {"SecurityFeature": _SecurityFeatureShim(agent)}
    app.state.agent = agent

    port = _free_port()
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    server = uvicorn.Server(config)

    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    # Wait for socket to accept.
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                break
        except OSError:
            time.sleep(0.05)
    else:
        raise RuntimeError(f"uvicorn failed to bind on port {port}")

    yield {"url": f"http://127.0.0.1:{port}", "agent": agent}

    server.should_exit = True
    thread.join(timeout=5.0)


async def _read_approval_event(client: httpx.AsyncClient, url: str, started: asyncio.Event):
    """Open an SSE stream, yield once the connection is up, return the first approval_request payload."""
    async with client.stream("GET", url, headers={"Accept": "text/event-stream"}) as response:
        assert response.status_code == 200, f"SSE status {response.status_code}"
        buffer = ""
        current_event = None
        async for chunk in response.aiter_bytes():
            if not started.is_set() and chunk:
                # First byte back means the connection is live and the server
                # has registered the event-bus listener.
                started.set()
            buffer += chunk.decode("utf-8", errors="replace")
            while "\n\n" in buffer:
                block, buffer = buffer.split("\n\n", 1)
                event_type = None
                data = None
                for line in block.splitlines():
                    if line.startswith("event: "):
                        event_type = line[len("event: "):].strip()
                    elif line.startswith("data: "):
                        data = line[len("data: "):]
                if event_type == "approval_request" and data:
                    return json.loads(data)


@pytest.mark.asyncio
async def test_real_http_sse_delivers_approval_and_decision_resolves_caller(live_server):
    base_url = live_server["url"]
    agent = live_server["agent"]

    async with httpx.AsyncClient(timeout=10.0) as client:
        started = asyncio.Event()
        reader_task = asyncio.create_task(
            _read_approval_event(client, f"{base_url}/agent/notifications/sse", started)
        )

        # Wait for the SSE stream to be live (ensures the server's listener is
        # registered before we fire the approval).
        await asyncio.wait_for(started.wait(), timeout=5.0)

        async def caller():
            return await agent.approval_queue.request_approval(
                feature_name="ComputeFeature",
                tool_name="run_script",
                tool_args={"script_id": "live-test-748"},
                timeout=10.0,
            )

        caller_task = asyncio.create_task(caller())

        payload = await asyncio.wait_for(reader_task, timeout=5.0)
        assert payload["feature"] == "ComputeFeature"
        assert payload["tool"] == "run_script"
        assert payload["args"] == {"script_id": "live-test-748"}
        approval_id = payload["id"]

        # Submit the decision via real HTTP POST.
        approve_response = await client.post(
            f"{base_url}/api/security/approve",
            json={"approval_id": approval_id, "approved": True, "scope": "session"},
        )
        assert approve_response.status_code == 200, approve_response.text
        body = approve_response.json()
        assert body["approved"] is True
        assert body["scope"] == "session"

        # The server-side request_approval call must unblock with the decision.
        approved, scope = await asyncio.wait_for(caller_task, timeout=2.0)
        assert approved is True
        assert scope == "session"


@pytest.mark.asyncio
async def test_real_http_denying_posts_404_after_request_times_out(live_server):
    """After request_approval times out, submit_decision returns 404 — same
    path the user reported ('Request not found or expired'). Guards against
    regressions that would mask this as a different error."""
    base_url = live_server["url"]
    agent = live_server["agent"]

    # Fire a request with a very short timeout so it expires before we submit.
    async def caller():
        return await agent.approval_queue.request_approval(
            feature_name="ComputeFeature",
            tool_name="run_script",
            tool_args={},
            timeout=0.2,
        )

    approved, scope = await caller()
    assert approved is False
    assert scope == "timeout"

    # Now try to submit a decision for any ID — the queue is empty.
    async with httpx.AsyncClient(timeout=5.0) as client:
        response = await client.post(
            f"{base_url}/api/security/approve",
            json={"approval_id": "ghost-id", "approved": True, "scope": "once"},
        )
        assert response.status_code == 404
        assert "not found or expired" in response.text
