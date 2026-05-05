"""
Tests for the ASGI-level multi_agent routing middleware.

The HTTP-only `agent_routing_middleware` (decorated with
`@app.middleware("http")`) does NOT fire on WebSocket scope, so prior
to fix/voice-pipeline-ws-multi_agent the Pipeline voice client connecting to
`/api/agents/<name>/voice/chat` 4503'd because the WS handler couldn't
find the agent. The replacement `MultiAgentAgentRoutingMiddleware` is a
class-based ASGI middleware that handles both `http` and `websocket`
scopes — these tests exercise it directly against an ASGI scope dict so
the WebSocket path is verified, not assumed.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest


def _make_middleware(agent_for: dict[str, Any] | None = None):
    """Build the middleware with a stubbed agent_manager.

    Imports server.py late to avoid loading the whole app at module-import
    time. Patches the module-level `app.state.agent_manager` so the
    middleware sees the stub.
    """
    import server as server_module

    inner_calls: list[dict] = []

    async def _inner(scope, receive, send):
        # Capture the scope at the point the inner app would run, so tests
        # can assert path-rewriting + state attachment without standing up
        # a real FastAPI route.
        inner_calls.append({"scope": dict(scope), "called": True})

    mw = server_module.MultiAgentAgentRoutingMiddleware(_inner)

    fake_manager = MagicMock()
    fake_manager.get_agent = MagicMock(side_effect=lambda name: (agent_for or {}).get(name))

    # Patch the module-level `app.state.agent_manager` reference the
    # middleware reads. Using direct attribute set instead of monkeypatch
    # so the change persists across the await chain.
    server_module.app.state.agent_manager = fake_manager
    return mw, inner_calls, fake_manager


@pytest.mark.asyncio
async def test_http_request_with_known_agent_strips_prefix_and_attaches_agent():
    fake_agent = MagicMock(name="Nellie")
    mw, inner, _ = _make_middleware({"Nellie": fake_agent})

    scope = {
        "type": "http",
        "path": "/api/agents/Nellie/voice/realtime/session",
        "raw_path": b"/api/agents/Nellie/voice/realtime/session",
    }
    sent: list[Any] = []

    async def _recv():
        return {}

    async def _send(msg):
        sent.append(msg)

    await mw(scope, _recv, _send)

    assert len(inner) == 1, "inner ASGI app should be invoked exactly once"
    rewritten = inner[0]["scope"]
    assert rewritten["path"] == "/voice/realtime/session"
    assert rewritten["raw_path"] == b"/voice/realtime/session"
    assert rewritten["state"]["agent"] is fake_agent


@pytest.mark.asyncio
async def test_websocket_with_known_agent_strips_prefix_and_attaches_agent():
    """The bug this middleware fixes: WebSocket scope used to bypass agent
    resolution entirely because @app.middleware('http') doesn't fire on it.
    """
    fake_agent = MagicMock(name="Nellie")
    mw, inner, _ = _make_middleware({"Nellie": fake_agent})

    scope = {
        "type": "websocket",
        "path": "/api/agents/Nellie/voice/chat",
        "raw_path": b"/api/agents/Nellie/voice/chat",
    }
    sent: list[Any] = []

    async def _recv():
        return {}

    async def _send(msg):
        sent.append(msg)

    await mw(scope, _recv, _send)

    assert len(inner) == 1, "inner ASGI app should be invoked exactly once"
    rewritten = inner[0]["scope"]
    assert rewritten["path"] == "/voice/chat"
    assert rewritten["state"]["agent"] is fake_agent


@pytest.mark.asyncio
async def test_websocket_with_unknown_agent_closes_with_4404():
    """Don't accept a WS for a missing agent — emit a clear close code so the
    browser console has something actionable instead of a generic disconnect.
    """
    mw, inner, _ = _make_middleware({})  # no agents

    scope = {
        "type": "websocket",
        "path": "/api/agents/Ghost/voice/chat",
        "raw_path": b"/api/agents/Ghost/voice/chat",
    }
    sent: list[Any] = []

    async def _recv():
        return {}

    async def _send(msg):
        sent.append(msg)

    await mw(scope, _recv, _send)

    assert len(inner) == 0, "inner app must NOT be invoked on unknown agent"
    assert any(m.get("type") == "websocket.close" and m.get("code") == 4404 for m in sent), (
        f"expected websocket.close with code 4404, got {sent}"
    )


@pytest.mark.asyncio
async def test_http_with_unknown_agent_returns_404_json():
    mw, inner, _ = _make_middleware({})

    scope = {
        "type": "http",
        "path": "/api/agents/Ghost/voice/voices",
        "raw_path": b"/api/agents/Ghost/voice/voices",
        "method": "GET",
        "headers": [],
        "query_string": b"",
    }
    sent: list[Any] = []

    async def _recv():
        return {}

    async def _send(msg):
        sent.append(msg)

    await mw(scope, _recv, _send)

    assert len(inner) == 0
    # Starlette JSONResponse sends a `http.response.start` with status=404.
    starts = [m for m in sent if m.get("type") == "http.response.start"]
    assert starts, f"no http.response.start sent, got {sent}"
    assert starts[0]["status"] == 404


@pytest.mark.asyncio
async def test_non_agent_path_passes_through_unchanged():
    """Routes outside /api/agents/{name}/... must not be touched."""
    mw, inner, _ = _make_middleware({"Nellie": MagicMock()})

    scope = {
        "type": "http",
        "path": "/api/auth/key",
        "raw_path": b"/api/auth/key",
    }

    async def _recv():
        return {}

    async def _send(msg):
        pass

    await mw(scope, _recv, _send)

    assert len(inner) == 1
    assert inner[0]["scope"]["path"] == "/api/auth/key", (
        "non-agent paths must not have their path rewritten"
    )
    # State must not have an `agent` attached for a non-agent path.
    assert "agent" not in inner[0]["scope"].get("state", {})


@pytest.mark.asyncio
async def test_no_agent_manager_passes_through():
    """Single-agent mode (no agent_manager on app.state) — middleware no-op."""
    import server as server_module

    inner_calls: list[dict] = []

    async def _inner(scope, receive, send):
        inner_calls.append({"scope": dict(scope), "called": True})

    mw = server_module.MultiAgentAgentRoutingMiddleware(_inner)
    # Force agent_manager to None.
    server_module.app.state.agent_manager = None

    scope = {
        "type": "http",
        "path": "/api/agents/Nellie/voice/voices",
        "raw_path": b"/api/agents/Nellie/voice/voices",
    }

    async def _recv():
        return {}

    async def _send(msg):
        pass

    await mw(scope, _recv, _send)

    # Pass-through: inner sees the original path (single-agent mode handles
    # the path itself or the route doesn't exist — that's the legacy
    # behavior; the middleware just doesn't try to rewrite).
    assert len(inner_calls) == 1
    assert inner_calls[0]["scope"]["path"] == "/api/agents/Nellie/voice/voices"


@pytest.mark.asyncio
async def test_lifespan_scope_passes_through():
    mw, inner, _ = _make_middleware({"Nellie": MagicMock()})

    scope = {"type": "lifespan"}

    async def _recv():
        return {}

    async def _send(msg):
        pass

    await mw(scope, _recv, _send)

    assert len(inner) == 1
    # Lifespan scopes don't have a path — middleware shouldn't crash.
    assert inner[0]["scope"]["type"] == "lifespan"
