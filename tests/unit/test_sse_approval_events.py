"""
Tests that the /agent/notifications/sse endpoint forwards events from the
agent's event bus (e.g. approval_request) to the client stream.

Regression for #748 — before the fix, the SSE generator only polled
agent.get_pending_notifications() and never subscribed to emit_event,
so SecurityFeature approval popups never reached the browser.
"""

import asyncio
import json

import pytest
from fastapi import FastAPI
from starlette.requests import Request

from endpoints.agent import (
    _sse_connections,
    notifications_sse,
    router,
)


@pytest.fixture(autouse=True)
def clean_sse_connections():
    _sse_connections.clear()
    yield
    _sse_connections.clear()


class _BusAgent:
    """Minimal stand-in for KestrelAgent that implements the event bus surface."""

    def __init__(self):
        self.agent_id = "did:test:bus-agent"
        self._listeners = []
        self.add_calls = 0
        self.remove_calls = 0

    def add_event_listener(self, listener):
        self.add_calls += 1
        self._listeners.append(listener)

    def remove_event_listener(self, listener):
        self.remove_calls += 1
        if listener in self._listeners:
            self._listeners.remove(listener)

    async def emit_event(self, event_type, data):
        for listener in list(self._listeners):
            await listener(event_type, data)

    def get_pending_notifications(self):
        return []


def _make_request(agent, disconnect_after_polls=3):
    """Build a real starlette Request wired to an app that holds the agent."""
    app = FastAPI()
    app.include_router(router)
    app.state.agent = agent

    call_count = {"n": 0}

    async def receive():
        # Simulate disconnect after N calls so request.is_disconnected flips True.
        call_count["n"] += 1
        if call_count["n"] >= disconnect_after_polls:
            return {"type": "http.disconnect"}
        await asyncio.sleep(0.05)
        return {"type": "http.request", "body": b"", "more_body": False}

    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "GET",
        "scheme": "http",
        "path": "/agent/notifications/sse",
        "raw_path": b"/agent/notifications/sse",
        "query_string": b"",
        "root_path": "",
        "headers": [(b"host", b"testserver")],
        "client": ("testclient", 12345),
        "server": ("testserver", 80),
        "app": app,
        "state": {},
    }
    return Request(scope, receive=receive)


async def _collect_sse_body(response, emit_coro=None, max_chunks=20):
    """Drain the StreamingResponse body, optionally kicking off an emit mid-stream."""
    chunks = []
    emit_task = None
    async for chunk in response.body_iterator:
        if isinstance(chunk, bytes):
            chunk = chunk.decode("utf-8")
        chunks.append(chunk)
        # Kick off the emit once we've seen the initial `connected` event so the
        # listener is already registered.
        if emit_coro and emit_task is None and "connected" in chunk:
            emit_task = asyncio.create_task(emit_coro())
        if len(chunks) >= max_chunks:
            break
    if emit_task:
        await emit_task
    return "".join(chunks)


@pytest.mark.asyncio
async def test_approval_request_event_is_forwarded_to_sse_stream():
    """An emit_event('approval_request', ...) on the agent bus reaches the SSE client."""
    agent = _BusAgent()
    request = _make_request(agent, disconnect_after_polls=10)

    response = await notifications_sse(request)

    async def emit():
        # Wait a tick so the generator has registered the listener.
        await asyncio.sleep(0.1)
        await agent.emit_event(
            "approval_request",
            {
                "id": "req-abc",
                "feature": "ComputeFeature",
                "tool": "run_script",
                "args": {"script_id": "s1"},
                "timestamp": "2026-04-24T00:00:00",
            },
        )

    body = await _collect_sse_body(response, emit_coro=emit, max_chunks=30)

    assert "event: connected" in body
    assert "event: approval_request" in body, (
        f"approval_request event was not forwarded. Got:\n{body}"
    )

    # Payload preserved
    lines = [line for line in body.splitlines() if line.startswith("data: ")]
    approval_payloads = [
        json.loads(line[len("data: "):])
        for line in lines
        if '"id": "req-abc"' in line
    ]
    assert approval_payloads, "approval_request payload missing from stream"
    assert approval_payloads[0]["feature"] == "ComputeFeature"
    assert approval_payloads[0]["tool"] == "run_script"


@pytest.mark.asyncio
async def test_listener_is_registered_and_cleaned_up():
    """The SSE generator registers a listener on connect and removes it on disconnect."""
    agent = _BusAgent()
    request = _make_request(agent, disconnect_after_polls=2)

    response = await notifications_sse(request)
    await _collect_sse_body(response, max_chunks=10)

    assert agent.add_calls == 1, "listener should be registered once"
    assert agent.remove_calls == 1, "listener should be removed on disconnect"
    assert agent._listeners == [], "no residual listeners after stream ends"


@pytest.mark.asyncio
async def test_agent_without_event_bus_does_not_crash():
    """An agent without add_event_listener still serves task notifications."""
    # MagicMock that explicitly lacks add_event_listener would still satisfy
    # hasattr() because MagicMock auto-creates attributes. Use a bare object.
    class MinimalAgent:
        agent_id = "did:test:minimal"

        def get_pending_notifications(self):
            return []

    agent = MinimalAgent()
    request = _make_request(agent, disconnect_after_polls=2)

    response = await notifications_sse(request)
    body = await _collect_sse_body(response, max_chunks=10)

    # Should still send the connected event and not raise
    assert "event: connected" in body
