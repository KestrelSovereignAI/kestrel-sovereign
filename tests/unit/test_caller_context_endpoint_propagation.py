"""Regression suite for #736 — every authenticated endpoint must pass a
CallerContext into agent.process_input/process_input_streaming so that
sovereign-command authorization is consistent regardless of which path the
caller reached.

These tests don't re-verify the gate itself (that's
test_caller_context_auth.py).  They pin the invariant that the *endpoint
layer* never hands off to the agent without `caller=`.  If it does, a
sovereign API-key holder can be rejected through one endpoint and accepted
through another — which is exactly the bug #736 fixes.
"""

from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from kestrel_sovereign.auth import AuthMethod, CallerRole


# ---------------------------------------------------------------------------
# Shared fixture helpers (same pattern as test_endpoint_contract_suite.py)
# ---------------------------------------------------------------------------


def _prepare_app(agent):
    from server import app

    @asynccontextmanager
    async def noop_lifespan(_app):
        yield

    original = {
        "lifespan": app.router.lifespan_context,
        "agent": getattr(app.state, "agent", None),
        "manager": getattr(app.state, "agent_manager", None),
    }
    app.router.lifespan_context = noop_lifespan
    app.state.agent = agent
    app.state.agent_manager = None
    return app, original


def _restore_app(app, original):
    app.router.lifespan_context = original["lifespan"]
    app.state.agent = original["agent"]
    app.state.agent_manager = original["manager"]


def _capturing_agent():
    """Return an agent mock that records the `caller=` kwarg on every
    process_input call.  Returns (agent, captured) where `captured` is a
    list populated by `process_input`/`process_input_streaming`.
    """
    captured = []

    async def _record(user_input, **kwargs):  # noqa: ARG001
        captured.append(kwargs.get("caller"))
        return "ok"

    async def _record_streaming(user_input, **kwargs):  # noqa: ARG001
        captured.append(kwargs.get("caller"))
        yield "ok"

    llm_service = MagicMock()
    llm_service.providers = [{"model": "gpt-5-mini"}]
    llm_service.get_active_model_id = MagicMock(return_value="gpt-5-mini")
    agent = MagicMock(llm_service=llm_service)
    agent.process_input = AsyncMock(side_effect=_record)
    agent.process_input_streaming = _record_streaming
    return agent, captured


# ---------------------------------------------------------------------------
# /v1/chat/completions  (endpoints/models.py)
# ---------------------------------------------------------------------------


def test_chat_completions_propagates_sovereign_caller_from_api_key():
    agent, captured = _capturing_agent()

    app, original = _prepare_app(agent)
    try:
        with patch.dict("os.environ", {"KESTREL_API_KEY": "test-key"}):
            with TestClient(app) as client:
                resp = client.post(
                    "/v1/chat/completions",
                    headers={"X-API-Key": "test-key"},
                    json={
                        "model": "gpt-5-mini",
                        "messages": [{"role": "user", "content": "!safe-mode exit"}],
                    },
                )
        assert resp.status_code == 200
        assert len(captured) == 1, "agent.process_input must have been called exactly once"
        caller = captured[0]
        assert caller is not None, (
            "chat-completions handed the prompt to the agent without a caller — "
            "sovereign commands issued via OpenAI-compatible clients would go "
            "through ungated. See issue #736."
        )
        assert caller.is_sovereign
        assert caller.auth_method == AuthMethod.API_KEY
    finally:
        _restore_app(app, original)


# ---------------------------------------------------------------------------
# /api/bridge/invoke and /api/bridge/stream  (features/bridge/router.py)
# ---------------------------------------------------------------------------


def _wire_bridge_feature(agent):
    """Populate agent with a minimal BridgeFeature stub so the router can
    resolve a session without hitting storage.  Returns the fake session id.
    """
    bridge = MagicMock()
    session = MagicMock(id="bridge-session-1")
    bridge.get_or_create_session = AsyncMock(return_value=session)
    bridge.log_invocation = AsyncMock()
    agent.features = {"BridgeFeature": bridge}
    return session.id


def test_bridge_invoke_propagates_sovereign_caller_from_api_key():
    agent, captured = _capturing_agent()
    _wire_bridge_feature(agent)

    app, original = _prepare_app(agent)
    try:
        # Bridge router is registered per-agent at runtime; pull it in here.
        from kestrel_sovereign.features.bridge.router import get_router
        app.include_router(get_router())

        with patch.dict("os.environ", {"KESTREL_API_KEY": "test-key"}):
            with TestClient(app) as client:
                resp = client.post(
                    "/api/bridge/invoke",
                    headers={"X-API-Key": "test-key"},
                    json={
                        "message": "!safe-mode exit",
                        "channel_type": "api",
                        "sender_id": "caller-1",
                    },
                )
        assert resp.status_code == 200, resp.text
        assert len(captured) == 1
        caller = captured[0]
        assert caller is not None, (
            "bridge invoke handed off to agent without a caller. "
            "Sovereign commands through the bridge would bypass the gate. See #736."
        )
        assert caller.is_sovereign
    finally:
        _restore_app(app, original)


def test_bridge_stream_propagates_sovereign_caller_from_api_key():
    agent, captured = _capturing_agent()
    _wire_bridge_feature(agent)

    app, original = _prepare_app(agent)
    try:
        from kestrel_sovereign.features.bridge.router import get_router
        app.include_router(get_router())

        with patch.dict("os.environ", {"KESTREL_API_KEY": "test-key"}):
            with TestClient(app) as client:
                # SSE endpoint — read fully to trigger the async for loop.
                with client.stream(
                    "POST",
                    "/api/bridge/stream",
                    headers={"X-API-Key": "test-key"},
                    json={
                        "message": "!safe-mode exit",
                        "channel_type": "api",
                        "sender_id": "caller-1",
                    },
                ) as resp:
                    assert resp.status_code == 200, resp.read()
                    # Drain stream so the generator runs through process_input_streaming.
                    for _chunk in resp.iter_text():
                        pass
        assert len(captured) == 1
        caller = captured[0]
        assert caller is not None, (
            "bridge stream handed off without a caller. See #736."
        )
        assert caller.is_sovereign
    finally:
        _restore_app(app, original)


# ---------------------------------------------------------------------------
# Voice WebSocket  (endpoints/voice.py)
# ---------------------------------------------------------------------------


def test_voice_websocket_auth_returns_sovereign_caller_for_valid_api_key():
    """The websocket-level auth helper must return a CallerContext
    (not just a bool) so that the caller can be threaded into
    agent.process_input_streaming for governance-command gating.
    """
    from endpoints.voice import _ws_authenticate

    ws = MagicMock()
    ws.query_params = {"api_key": "test-key"}

    with patch.dict("os.environ", {"KESTREL_API_KEY": "test-key"}):
        caller = _ws_authenticate(ws)

    assert caller is not None
    assert caller.role == CallerRole.SOVEREIGN
    assert caller.auth_method == AuthMethod.API_KEY


def test_voice_websocket_auth_returns_none_for_bad_api_key():
    from endpoints.voice import _ws_authenticate

    ws = MagicMock()
    ws.query_params = {"api_key": "wrong"}
    # Session cookie path must also be absent for this case.
    del ws.session

    with patch.dict("os.environ", {"KESTREL_API_KEY": "test-key"}):
        caller = _ws_authenticate(ws)

    assert caller is None


def test_voice_websocket_auth_returns_authenticated_caller_for_session_cookie():
    from endpoints.voice import _ws_authenticate

    ws = MagicMock()
    ws.query_params = {}
    ws.session = {"user_email": "user@example.com"}

    with patch.dict("os.environ", {"KESTREL_API_KEY": "test-key"}):
        caller = _ws_authenticate(ws)

    assert caller is not None
    assert caller.role == CallerRole.AUTHENTICATED
    assert caller.identity == "user@example.com"
    assert caller.auth_method == AuthMethod.OAUTH_SESSION


def test_voice_websocket_auth_no_key_configured_is_sovereign():
    """When KESTREL_API_KEY is not set (local dev), behavior matches the HTTP
    middleware: no auth required, caller is sovereign by default."""
    from endpoints.voice import _ws_authenticate

    ws = MagicMock()
    ws.query_params = {}
    del ws.session

    with patch.dict("os.environ", {}, clear=False):
        # Ensure the key is absent for this test regardless of ambient env.
        import os
        os.environ.pop("KESTREL_API_KEY", None)
        caller = _ws_authenticate(ws)

    assert caller is not None
    assert caller.role == CallerRole.SOVEREIGN
