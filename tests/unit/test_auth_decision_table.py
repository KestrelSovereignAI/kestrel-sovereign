"""Decision-table tests for auth classes in server.py."""

from contextlib import asynccontextmanager
import os
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
import pytest
from starlette.middleware.sessions import SessionMiddleware

from kestrel_sovereign.server import auth_middleware


class _SessionAgentManager:
    def __init__(self, agent):
        self._agent = agent

    def get_agent(self, name):
        if name.lower() == "claw":
            return self._agent
        return None

    def list_agents(self):
        return {"Claw": self._agent}


def _make_agent():
    agent = MagicMock()
    agent.agent_id = "did:pkh:eip155:1:0xabc"
    agent.privacy_mode = MagicMock()
    agent.privacy_mode.value = "NORMAL"
    agent.features = {}
    agent.process_input = AsyncMock(return_value="ok")
    agent.process_input_streaming = AsyncMock()
    agent.register_active_request = MagicMock()
    agent.is_request_cancelled = MagicMock(return_value=False)
    agent._cleanup_cancelled_request = MagicMock()
    agent.cancel_current_request = MagicMock(return_value=False)
    return agent


def _prepare_app():
    from server import app

    @asynccontextmanager
    async def noop_lifespan(_app):
        yield

    original = {
        "lifespan": app.router.lifespan_context,
        "agent": getattr(app.state, "agent", None),
        "manager": getattr(app.state, "agent_manager", None),
        "config": getattr(app.state, "multi_agent_config", None),
    }
    app.router.lifespan_context = noop_lifespan
    return app, original


def _restore_app(app, original):
    app.router.lifespan_context = original["lifespan"]
    app.state.agent = original["agent"]
    app.state.agent_manager = original["manager"]
    app.state.multi_agent_config = original["config"]


def test_root_html_is_public_when_oauth_not_required():
    app, original = _prepare_app()
    try:
        with patch.dict("os.environ", {"KESTREL_API_KEY": "test-key", "KESTREL_REQUIRE_OAUTH": "false"}):
            with TestClient(app) as client:
                response = client.get("/", headers={"accept": "text/html"})
        assert response.status_code == 200
    finally:
        _restore_app(app, original)


def test_root_html_redirects_when_oauth_required():
    from kestrel_sovereign.endpoints.auth_oauth import oauth

    app, original = _prepare_app()
    try:
        with patch.dict("os.environ", {"KESTREL_API_KEY": "test-key", "KESTREL_REQUIRE_OAUTH": "true"}):
            oauth._clients["google"] = MagicMock()
            with TestClient(app) as client:
                response = client.get("/", headers={"accept": "text/html"}, follow_redirects=False)
        assert response.status_code == 302
        assert response.headers["location"] == "/auth/login"
    finally:
        oauth._clients.pop("google", None)
        _restore_app(app, original)


def test_bootstrap_key_is_localhost_only_when_enabled():
    app, original = _prepare_app()
    try:
        with patch.dict("os.environ", {"KESTREL_API_KEY": "bootstrap-key", "KESTREL_REQUIRE_OAUTH": "false"}):
            with TestClient(app, client=("127.0.0.1", 55000)) as client:
                ok_response = client.get("/api/auth/key")
            with TestClient(app, client=("203.0.113.10", 55000)) as client:
                denied_response = client.get("/api/auth/key")
        assert ok_response.status_code == 200
        assert ok_response.json()["key"] == "bootstrap-key"
        assert denied_response.status_code == 403
    finally:
        _restore_app(app, original)


def test_managed_peer_process_cannot_bootstrap_or_use_sovereign_key():
    """A transport-only child has no recoverable local operator lane."""

    app, original = _prepare_app()
    try:
        with patch.dict(
            "os.environ",
            {
                "KESTREL_A2A_TRANSPORT_ONLY": "true",
                "KESTREL_API_KEY": "",
                "KESTREL_REQUIRE_OAUTH": "false",
            },
        ):
            with TestClient(app, client=("127.0.0.1", 55000)) as client:
                bootstrap = client.get("/api/auth/key")
                assert bootstrap.status_code == 404
                assert os.environ["KESTREL_API_KEY"] == ""

                # Even an accidentally configured child-local key must not
                # recreate operator authority inside the managed process.
                os.environ["KESTREL_API_KEY"] = "child-local-sovereign-key"
                protected = client.get(
                    "/api/agent/tasks",
                    headers={"X-API-Key": "child-local-sovereign-key"},
                )
                assert protected.status_code == 401
                assert protected.json()["error"]["code"] == (
                    "authentication_required"
                )
    finally:
        _restore_app(app, original)


@pytest.mark.parametrize("host_state", ["manager", "config"])
def test_multi_agent_host_never_bootstraps_sovereign_key_to_local_peer(host_state):
    """Loopback cannot distinguish a browser from a managed peer process."""

    app, original = _prepare_app()
    try:
        app.state.agent = None
        app.state.agent_manager = (
            _SessionAgentManager(_make_agent()) if host_state == "manager" else None
        )
        app.state.multi_agent_config = object() if host_state == "config" else None
        with patch.dict(
            "os.environ",
            {
                "KESTREL_API_KEY": "host-sovereign-key",
                "KESTREL_REQUIRE_OAUTH": "false",
            },
        ):
            with TestClient(app, client=("127.0.0.1", 55000)) as client:
                response = client.get("/api/auth/key")

        assert response.status_code == 404
        assert "host-sovereign-key" not in response.text
    finally:
        _restore_app(app, original)


def test_multi_agent_server_rejects_missing_out_of_band_host_key(monkeypatch):
    """A fleet must never fall back to a peer-recoverable ephemeral key."""
    from kestrel_sovereign import server

    monkeypatch.setenv("KESTREL_MULTI_AGENT", "true")
    monkeypatch.delenv("KESTREL_API_KEY", raising=False)

    with pytest.raises(RuntimeError, match="kestrel setup keys"):
        server._require_multi_agent_host_api_key(os.environ)


@pytest.mark.asyncio
async def test_multi_agent_lifespan_checks_host_key_before_starting_resources(
    monkeypatch,
):
    """The direct-uvicorn path must invoke the same fleet credential guard."""
    from kestrel_sovereign import server

    class GuardReached(RuntimeError):
        pass

    monkeypatch.setenv("KESTREL_MULTI_AGENT", "true")
    monkeypatch.setattr(
        server,
        "_require_multi_agent_host_api_key",
        lambda _environ: (_ for _ in ()).throw(GuardReached("guard reached")),
    )

    with pytest.raises(GuardReached, match="guard reached"):
        async with server._lifespan_startup(FastAPI()):
            pass


def test_auth_me_rejects_api_key_without_session():
    app, original = _prepare_app()
    try:
        with patch.dict("os.environ", {"KESTREL_API_KEY": "test-key"}):
            with TestClient(app) as client:
                response = client.get("/auth/me", headers={"X-API-Key": "test-key"})
        assert response.status_code == 401
        assert response.json()["detail"] == "Not authenticated"
        assert response.json()["error"]["code"] == "authentication_required"
        assert response.json()["error"]["correlation_id"] == response.headers[
            "X-Correlation-ID"
        ]
    finally:
        _restore_app(app, original)


def test_auth_me_returns_session_payload_when_session_present():
    app, original = _prepare_app()
    route = None

    @app.get("/_test/session")
    async def _test_session(request: Request):
        request.session["user_email"] = "user@example.com"
        request.session["user_name"] = "User"
        return {"ok": True}

    route = app.router.routes[-1]
    try:
        with patch.dict("os.environ", {"KESTREL_API_KEY": "test-key"}):
            with TestClient(app) as client:
                session_response = client.get("/_test/session", headers={"X-API-Key": "test-key"})
                response = client.get("/auth/me")
        assert session_response.status_code == 200
        assert response.status_code == 200
        assert response.json()["email"] == "user@example.com"
    finally:
        app.router.routes.remove(route)
        _restore_app(app, original)


def test_sse_query_param_auth_reaches_stream_endpoint_and_preserves_400():
    app, original = _prepare_app()
    app.state.agent = _make_agent()
    try:
        with patch.dict("os.environ", {"KESTREL_API_KEY": "test-key"}):
            with TestClient(app) as client:
                response = client.post("/api/agent/stream?api_key=test-key", json={})
        assert response.status_code == 400
        assert response.json()["detail"] == "Input not provided."
    finally:
        _restore_app(app, original)


@pytest.mark.parametrize("path", ["/api/agent/invoke", "/api/agent/stream"])
def test_agent_endpoints_reject_non_object_json_with_typed_400(path):
    app, original = _prepare_app()
    app.state.agent = _make_agent()
    try:
        with patch.dict("os.environ", {"KESTREL_API_KEY": "test-key"}):
            with TestClient(app) as client:
                response = client.post(
                    path,
                    json=[{"input": "must-not-be-treated-as-an-object"}],
                    headers={"X-API-Key": "test-key"},
                )
        assert response.status_code == 400
        assert response.json()["error"]["code"] == "invalid_request_body"
        assert response.json()["detail"] == "JSON request body must be an object."
    finally:
        _restore_app(app, original)


@pytest.mark.parametrize("path", ["/api/agent/invoke", "/api/agent/stream"])
def test_agent_endpoints_do_not_echo_malformed_json_content(path):
    app, original = _prepare_app()
    app.state.agent = _make_agent()
    try:
        with patch.dict("os.environ", {"KESTREL_API_KEY": "test-key"}):
            with TestClient(app) as client:
                response = client.post(
                    path,
                    content=b'{"password":"do-not-echo",',
                    headers={
                        "Content-Type": "application/json",
                        "X-API-Key": "test-key",
                    },
                )
        assert response.status_code == 400
        assert response.json()["error"]["code"] == "invalid_json"
        assert response.json()["detail"].startswith("Invalid JSON at line ")
        assert "do-not-echo" not in response.text
    finally:
        _restore_app(app, original)


def test_sse_query_param_auth_rejects_wrong_key():
    app, original = _prepare_app()
    app.state.agent = _make_agent()
    try:
        with patch.dict("os.environ", {"KESTREL_API_KEY": "test-key"}):
            with TestClient(app) as client:
                response = client.post(
                    "/api/agent/stream?api_key=wrong-key", json={}
                )
        assert response.status_code == 401
        assert response.json()["detail"] == "Invalid or missing API Key"
    finally:
        _restore_app(app, original)


@pytest.mark.parametrize(
    ("suffix", "query_key", "expected_status"),
    [
        pytest.param(
            "/api/agent/notifications/sse", "test-key", 200,
            id="notifications-valid",
        ),
        pytest.param(
            "/api/agent/stream", "test-key", 200,
            id="stream-valid",
        ),
        pytest.param(
            "/api/agent/notifications/sse", "wrong-key", 401,
            id="notifications-wrong-key",
        ),
        pytest.param(
            "/api/agent/stream", "wrong-key", 401,
            id="stream-wrong-key",
        ),
    ],
)
def test_prefixed_sse_query_auth_uses_real_middleware(
    monkeypatch, suffix, query_key, expected_status,
):
    """The deployed multi-agent URL shape uses canonical server auth."""
    monkeypatch.setenv("KESTREL_API_KEY", "test-key")
    test_app = FastAPI()
    test_app.middleware("http")(auth_middleware)
    test_app.add_middleware(
        SessionMiddleware,
        secret_key="test-session-secret",
        session_cookie="kestrel_session",
    )

    @test_app.get("/api/agents/{agent}/api/agent/notifications/sse")
    @test_app.get("/api/agents/{agent}/api/agent/stream")
    def prefixed_sse(agent: str):
        return {"agent": agent}

    response = TestClient(test_app).get(
        f"/api/agents/Kite{suffix}", params={"api_key": query_key}
    )

    assert response.status_code == expected_status
    if expected_status == 200:
        assert response.json() == {"agent": "Kite"}
    else:
        assert response.json()["detail"] == "Invalid or missing API Key"


def test_non_sse_query_param_auth_is_rejected():
    app, original = _prepare_app()
    try:
        with patch.dict("os.environ", {"KESTREL_API_KEY": "test-key"}):
            with TestClient(app) as client:
                response = client.get("/api/memories?api_key=test-key")
        assert response.status_code == 401
    finally:
        _restore_app(app, original)


def test_protected_agent_route_accepts_api_key_and_multi_agent_rewrite_matches():
    app, original = _prepare_app()
    agent = _make_agent()
    app.state.agent = agent
    app.state.agent_manager = _SessionAgentManager(agent)
    try:
        with patch.dict("os.environ", {"KESTREL_API_KEY": "test-key"}):
            with TestClient(app) as client:
                direct_response = client.get("/api/agent/info", headers={"X-API-Key": "test-key"})
                rewritten_response = client.get(
                    "/api/agents/Claw/api/agent/info",
                    headers={"X-API-Key": "test-key"},
                )
        assert direct_response.status_code == 200
        assert rewritten_response.status_code == 200
        assert rewritten_response.json()["agent_id"] == direct_response.json()["agent_id"]
    finally:
        _restore_app(app, original)
