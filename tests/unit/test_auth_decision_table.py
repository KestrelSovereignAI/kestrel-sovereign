"""Decision-table tests for auth classes in server.py."""

from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi import Request
from fastapi.testclient import TestClient


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
    }
    app.router.lifespan_context = noop_lifespan
    return app, original


def _restore_app(app, original):
    app.router.lifespan_context = original["lifespan"]
    app.state.agent = original["agent"]
    app.state.agent_manager = original["manager"]


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
    from endpoints.auth_oauth import oauth

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


def test_auth_me_rejects_api_key_without_session():
    app, original = _prepare_app()
    try:
        with patch.dict("os.environ", {"KESTREL_API_KEY": "test-key"}):
            with TestClient(app) as client:
                response = client.get("/auth/me", headers={"X-API-Key": "test-key"})
        assert response.status_code == 401
        assert response.json()["detail"] == "Not authenticated"
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
                response = client.post("/agent/stream?api_key=test-key", json={})
        assert response.status_code == 400
        assert response.json()["detail"] == "Input not provided."
    finally:
        _restore_app(app, original)


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
                direct_response = client.get("/agent/info", headers={"X-API-Key": "test-key"})
                rewritten_response = client.get(
                    "/api/agents/Claw/agent/info",
                    headers={"X-API-Key": "test-key"},
                )
        assert direct_response.status_code == 200
        assert rewritten_response.status_code == 200
        assert rewritten_response.json()["agent_id"] == direct_response.json()["agent_id"]
    finally:
        _restore_app(app, original)
