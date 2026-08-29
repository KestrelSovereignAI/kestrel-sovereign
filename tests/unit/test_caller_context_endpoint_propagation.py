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
        "route_count": len(app.routes),
    }
    app.router.lifespan_context = noop_lifespan
    app.state.agent = agent
    app.state.agent_manager = None
    return app, original


def _restore_app(app, original):
    app.router.lifespan_context = original["lifespan"]
    app.state.agent = original["agent"]
    app.state.agent_manager = original["manager"]
    # The /api/bridge/* cases pull the per-agent bridge router into the shared
    # ``server.app`` singleton with ``app.include_router(get_router())``. Drop
    # any routes the test appended so they don't leak into later tests that
    # assert the app's route set (e.g. the #2522 route-gate suite counting a
    # single mounted /api/bridge/health).
    del app.routes[original["route_count"]:]


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


@pytest.mark.parametrize("path", ["invoke", "stream"])
def test_bridge_lifecycle_precedes_session_and_log_side_effects(path):
    """A concurrent exact Stop must see the bridge turn as live."""

    events = []
    agent, _captured = _capturing_agent()
    agent.register_active_request = MagicMock(
        side_effect=lambda request_id: events.append(("register", request_id))
    )
    agent._cleanup_cancelled_request = MagicMock()
    agent.is_request_cancelled = MagicMock(return_value=False)
    session = MagicMock(id="bridge-session-1")

    async def _session(**_kwargs):
        events.append(("session", None))
        assert events[0] == ("register", "bridge-ordering")
        return session

    async def _log(**_kwargs):
        events.append(("log", None))
        assert events[0] == ("register", "bridge-ordering")

    bridge = MagicMock()
    bridge.get_or_create_session = AsyncMock(side_effect=_session)
    bridge.log_invocation = AsyncMock(side_effect=_log)
    agent.features = {"BridgeFeature": bridge}

    app, original = _prepare_app(agent)
    try:
        from kestrel_sovereign.features.bridge.router import get_router

        app.include_router(get_router())
        with patch.dict("os.environ", {"KESTREL_API_KEY": "test-key"}):
            with TestClient(app) as client:
                if path == "stream":
                    with client.stream(
                        "POST",
                        "/api/bridge/stream",
                        headers={"X-API-Key": "test-key"},
                        json={
                            "message": "work",
                            "channel_type": "api",
                            "request_id": "bridge-ordering",
                        },
                    ) as response:
                        assert response.status_code == 200, response.read()
                        list(response.iter_text())
                else:
                    response = client.post(
                        "/api/bridge/invoke",
                        headers={"X-API-Key": "test-key"},
                        json={
                            "message": "work",
                            "channel_type": "api",
                            "request_id": "bridge-ordering",
                        },
                    )
                    assert response.status_code == 200, response.text

        assert events[0] == ("register", "bridge-ordering")
        agent._cleanup_cancelled_request.assert_called_once_with(
            "bridge-ordering"
        )
    finally:
        _restore_app(app, original)


def test_bridge_invoke_forwards_header_retry_id_and_trusted_provenance():
    """Bridge redeliveries must not mint a new canonical tool operation."""
    from kestrel_sovereign.agent.invocation import invocation_id_response_header

    agent = MagicMock()
    agent.process_input = AsyncMock(return_value="ok")
    _wire_bridge_feature(agent)
    request_id = "bridge ☃ / 100% %E2%98%83?redelivery=1"
    header_echo = invocation_id_response_header(request_id)

    app, original = _prepare_app(agent)
    try:
        from kestrel_sovereign.features.bridge.router import get_router
        app.include_router(get_router())

        with patch.dict("os.environ", {"KESTREL_API_KEY": "test-key"}):
            with TestClient(app) as client:
                response = client.post(
                    "/api/bridge/invoke",
                    headers={
                        "X-API-Key": "test-key",
                        "X-Request-ID": header_echo,
                    },
                    json={"message": "teach this", "channel_type": "api"},
                )

        assert response.status_code == 200, response.text
        assert response.headers["X-Request-ID"] == header_echo
        _, kwargs = agent.process_input.await_args
        assert kwargs["invocation_id"] == request_id
        provenance = kwargs["invocation_provenance"]
        assert provenance.actor == "api_key"
        assert provenance.source_locator == "POST:/api/bridge/invoke"
    finally:
        _restore_app(app, original)


def test_bridge_invoke_failure_logs_no_message_or_exception_text():
    """Synchronous bridge failures must not serialize request content to logs."""
    marker = "BRIDGE_SYNC_PRIVATE_FACT_DO_NOT_LOG"
    agent = MagicMock()
    agent.process_input = AsyncMock(
        side_effect=RuntimeError(f"provider rejected {marker}")
    )
    _wire_bridge_feature(agent)

    app, original = _prepare_app(agent)
    try:
        from kestrel_sovereign.features.bridge.router import get_router

        app.include_router(get_router())
        with patch.dict("os.environ", {"KESTREL_API_KEY": "test-key"}):
            with patch(
                "kestrel_sovereign.features.bridge.router.logger.error"
            ) as error_log:
                with TestClient(app) as client:
                    response = client.post(
                        "/api/bridge/invoke",
                        headers={"X-API-Key": "test-key"},
                        json={"message": marker, "channel_type": "api"},
                    )

        assert response.status_code == 500
        logged = " ".join(
            str(item)
            for call in error_log.call_args_list
            for item in (*call.args, *call.kwargs.values())
        )
        assert marker not in logged
        assert "provider rejected" not in logged
        assert error_log.call_args.kwargs.get("exc_info") is None
    finally:
        _restore_app(app, original)


def test_bridge_invoke_reports_cooperative_stop_as_conflict():
    from kestrel_sovereign.agent.invocation import InvocationCancelledError

    agent = MagicMock()
    agent.process_input = AsyncMock(
        side_effect=InvocationCancelledError("isolated turn stopped")
    )
    _wire_bridge_feature(agent)

    app, original = _prepare_app(agent)
    try:
        from kestrel_sovereign.features.bridge.router import get_router

        app.include_router(get_router())
        with patch.dict("os.environ", {"KESTREL_API_KEY": "test-key"}):
            with TestClient(app) as client:
                response = client.post(
                    "/api/bridge/invoke",
                    headers={
                        "X-API-Key": "test-key",
                        "X-Request-ID": "bridge-stopped-turn",
                    },
                    json={"message": "stop this", "channel_type": "api"},
                )

        assert response.status_code == 409
        assert response.json()["detail"] == "Request stopped during execution."
        assert response.headers["X-Request-ID"] == "bridge-stopped-turn"
    finally:
        _restore_app(app, original)


def test_bridge_invoke_rechecks_stop_after_outbound_audit():
    """An outbound audit await cannot reopen the response publication race."""
    stopped = False
    agent = MagicMock()
    agent.process_input = AsyncMock(return_value="must not publish")
    agent.is_request_cancelled = MagicMock(
        side_effect=lambda _request_id: stopped
    )
    bridge = MagicMock()
    bridge.get_or_create_session = AsyncMock(
        return_value=MagicMock(id="bridge-session-1")
    )

    async def _log_invocation(**kwargs):
        nonlocal stopped
        if kwargs["direction"] == "outbound":
            stopped = True

    bridge.log_invocation = AsyncMock(side_effect=_log_invocation)
    agent.features = {"BridgeFeature": bridge}

    app, original = _prepare_app(agent)
    try:
        from kestrel_sovereign.features.bridge.router import get_router

        app.include_router(get_router())
        with patch.dict("os.environ", {"KESTREL_API_KEY": "test-key"}):
            with TestClient(app) as client:
                response = client.post(
                    "/api/bridge/invoke",
                    headers={"X-API-Key": "test-key"},
                    json={
                        "message": "stop at audit boundary",
                        "channel_type": "api",
                        "request_id": "bridge-outbound-stop",
                    },
                )

        assert response.status_code == 409
        assert response.json()["detail"] == "Request stopped during execution."
    finally:
        _restore_app(app, original)


def test_bridge_stream_setup_failure_completes_owned_lifecycle():
    """Terminal setup errors are completed, not abandoned generations."""
    agent = MagicMock()
    agent.process_input_streaming = MagicMock()
    agent.is_request_cancelled = MagicMock(return_value=False)
    bridge = MagicMock()
    bridge.get_or_create_session = AsyncMock(
        side_effect=RuntimeError("session setup failed")
    )
    bridge.log_invocation = AsyncMock()
    agent.features = {"BridgeFeature": bridge}

    app, original = _prepare_app(agent)
    try:
        from kestrel_sovereign.features.bridge.router import get_router

        app.include_router(get_router())
        with patch.dict("os.environ", {"KESTREL_API_KEY": "test-key"}):
            with TestClient(app, raise_server_exceptions=False) as client:
                response = client.post(
                    "/api/bridge/stream",
                    headers={"X-API-Key": "test-key"},
                    json={
                        "message": "setup",
                        "channel_type": "api",
                        "request_id": "bridge-setup-failure",
                    },
                )

        assert response.status_code == 500
        agent._cleanup_cancelled_request.assert_called_once_with(
            "bridge-setup-failure"
        )
    finally:
        _restore_app(app, original)


def test_bridge_stream_response_failure_completes_owned_lifecycle(monkeypatch):
    """Response construction is still endpoint-owned terminal setup work."""
    agent, _captured = _capturing_agent()
    agent.is_request_cancelled = MagicMock(return_value=False)
    _wire_bridge_feature(agent)

    class ResponseConstructionFailure:
        def __init__(self, *_args, **_kwargs):
            raise RuntimeError("response construction failed")

    app, original = _prepare_app(agent)
    try:
        import kestrel_sovereign.features.bridge.router as bridge_router

        monkeypatch.setattr(
            bridge_router,
            "StreamingResponse",
            ResponseConstructionFailure,
        )
        app.include_router(bridge_router.get_router())
        with patch.dict("os.environ", {"KESTREL_API_KEY": "test-key"}):
            with TestClient(app, raise_server_exceptions=False) as client:
                response = client.post(
                    "/api/bridge/stream",
                    headers={"X-API-Key": "test-key"},
                    json={
                        "message": "setup",
                        "channel_type": "api",
                        "request_id": "bridge-response-failure",
                    },
                )

        assert response.status_code == 500
        agent._cleanup_cancelled_request.assert_called_once_with(
            "bridge-response-failure"
        )
    finally:
        _restore_app(app, original)


def test_bridge_stream_forwards_body_retry_id_and_trusted_provenance():
    """The bridge SSE route shares the same invocation identity contract."""
    captured = {}

    async def _stream(_input, **kwargs):
        captured.update(kwargs)
        yield "ok"

    agent = MagicMock()
    agent.process_input_streaming = _stream
    _wire_bridge_feature(agent)

    app, original = _prepare_app(agent)
    try:
        from kestrel_sovereign.features.bridge.router import get_router
        app.include_router(get_router())

        with patch.dict("os.environ", {"KESTREL_API_KEY": "test-key"}):
            with TestClient(app) as client:
                with client.stream(
                    "POST",
                    "/api/bridge/stream",
                    headers={"X-API-Key": "test-key"},
                    json={
                        "message": "teach this",
                        "channel_type": "api",
                        "request_id": "bridge-stream-retry-2765",
                    },
                ) as response:
                    assert response.status_code == 200, response.read()
                    assert response.headers["X-Request-ID"] == "bridge-stream-retry-2765"
                    for _chunk in response.iter_text():
                        pass

        assert captured["request_id"] == "bridge-stream-retry-2765"
        provenance = captured["invocation_provenance"]
        assert provenance.actor == "api_key"
        assert provenance.source_locator == "POST:/api/bridge/stream"
    finally:
        _restore_app(app, original)


def test_bridge_stream_registers_and_releases_the_request_lifecycle():
    """Bridge SSE uses the same cancellable counted lifecycle as chat SSE."""
    async def _stream(_input, **_kwargs):
        yield "ok"

    agent = MagicMock()
    agent.process_input_streaming = _stream
    _wire_bridge_feature(agent)

    app, original = _prepare_app(agent)
    try:
        from kestrel_sovereign.features.bridge.router import get_router
        app.include_router(get_router())

        with patch.dict("os.environ", {"KESTREL_API_KEY": "test-key"}):
            with TestClient(app) as client:
                with client.stream(
                    "POST",
                    "/api/bridge/stream",
                    headers={"X-API-Key": "test-key"},
                    json={
                        "message": "stream this",
                        "channel_type": "api",
                        "request_id": "bridge-lifecycle-2765",
                    },
                ) as response:
                    assert response.status_code == 200, response.read()
                    for _chunk in response.iter_text():
                        pass

        agent.register_active_request.assert_called_once_with("bridge-lifecycle-2765")
        agent._cleanup_cancelled_request.assert_called_once_with(
            "bridge-lifecycle-2765"
        )
    finally:
        _restore_app(app, original)


@pytest.mark.asyncio
async def test_bridge_unstarted_response_body_releases_request_lifecycle():
    """A disconnect before first body pull cannot strand Stop registration."""

    from fastapi import FastAPI
    from starlette.requests import Request

    from kestrel_sovereign.features.bridge.protocol import BridgeRequest
    from kestrel_sovereign.features.bridge.router import get_router

    entered_stream = False

    async def _stream(_input, **_kwargs):
        nonlocal entered_stream
        entered_stream = True
        yield "unreachable"

    agent = MagicMock()
    agent.process_input_streaming = _stream
    agent.is_request_cancelled = MagicMock(return_value=False)
    _wire_bridge_feature(agent)
    app = FastAPI()
    app.state.agent = agent
    app.state.agent_manager = None
    request = Request(
        {
            "type": "http",
            "http_version": "1.1",
            "method": "POST",
            "scheme": "http",
            "path": "/api/bridge/stream",
            "raw_path": b"/api/bridge/stream",
            "query_string": b"",
            "headers": [],
            "client": ("test", 1),
            "server": ("test", 80),
            "app": app,
        }
    )
    route = next(
        route for route in get_router().routes if route.path == "/api/bridge/stream"
    )
    endpoint = getattr(route.endpoint, "__wrapped__", route.endpoint)

    response = await endpoint(
        request,
        BridgeRequest(message="work", request_id="bridge-unstarted-body"),
    )
    assert entered_stream is False
    agent.register_active_request.assert_called_once_with("bridge-unstarted-body")

    await response.body_iterator.aclose()

    assert entered_stream is False
    agent._cleanup_cancelled_request.assert_called_once_with(
        "bridge-unstarted-body"
    )
