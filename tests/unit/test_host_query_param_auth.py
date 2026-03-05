"""
Unit tests for host.py API key query parameter restriction (GitHub issue #160).

Verifies that API key authentication via ?api_key= query parameter
is only accepted on SSE/streaming proxy paths in the host, not on all endpoints.
Keys in URLs get logged in access logs, proxy logs, and browser history,
so query param auth must be restricted to endpoints that require it
(EventSource/SSE cannot send custom headers).

This mirrors the fix from #149 applied to server.py but for host.py.
"""

import secrets as secrets_mod

import pytest
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

API_KEY = "test-host-api-key-for-sse-restriction"

# Must match host.py definition
SSE_PATH_SUFFIXES = ("/agent/notifications/sse", "/agent/stream")


@pytest.fixture
def app():
    """Create a minimal FastAPI app that mirrors host.py auth middleware."""
    test_app = FastAPI()

    @test_app.middleware("http")
    async def auth_middleware(request, call_next):
        """Mirrors the auth_middleware in host.py."""
        public_paths = {"/health", "/", "/favicon.ico", "/api/auth/key"}
        static_prefixes = ("/static", "/js/", "/shared/", "/utils/")

        if request.url.path in public_paths or any(
            request.url.path.startswith(p) for p in static_prefixes
        ):
            return await call_next(request)

        expected_key = API_KEY

        # Check X-API-Key header
        api_key_header = request.headers.get("X-API-Key")
        if api_key_header and secrets_mod.compare_digest(api_key_header, expected_key):
            return await call_next(request)

        # Check Bearer token
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            token = auth_header[7:]
            if secrets_mod.compare_digest(token, expected_key):
                return await call_next(request)

        # Check query parameter for SSE endpoints only
        api_key_query = request.query_params.get("api_key")
        if api_key_query and any(
            request.url.path.endswith(s) for s in SSE_PATH_SUFFIXES
        ):
            if secrets_mod.compare_digest(api_key_query, expected_key):
                return await call_next(request)

        return JSONResponse(
            content={"detail": "Invalid or missing API Key"},
            status_code=401,
        )

    # Proxied SSE endpoints (through the host proxy)
    @test_app.get("/api/agents/{agent_id}/agent/notifications/sse")
    def proxied_notifications_sse(agent_id: str):
        return {"type": "sse", "agent": agent_id}

    @test_app.post("/api/agents/{agent_id}/agent/stream")
    def proxied_agent_stream(agent_id: str):
        return {"type": "stream", "agent": agent_id}

    # Non-SSE host endpoints
    @test_app.get("/api/agents")
    def list_agents():
        return {"agents": []}

    @test_app.post("/api/agents/{agent_id}/start")
    def start_agent(agent_id: str):
        return {"status": "started"}

    @test_app.post("/api/agents/{agent_id}/stop")
    def stop_agent(agent_id: str):
        return {"status": "stopped"}

    @test_app.get("/api/agents/{agent_id}/status")
    def agent_status(agent_id: str):
        return {"status": "running"}

    @test_app.get("/api/agents/{agent_id}/logs")
    def agent_logs(agent_id: str):
        return {"logs": ""}

    # Non-SSE proxied endpoint
    @test_app.get("/api/agents/{agent_id}/api/conversations")
    def proxied_conversations(agent_id: str):
        return {"conversations": []}

    @test_app.get("/health")
    def health():
        return {"status": "ok"}

    yield test_app


@pytest.fixture
def client(app):
    return TestClient(app)


class TestQueryParamAuthOnSSEProxyPaths:
    """Query param auth should work on proxied SSE/streaming endpoints."""

    def test_query_param_auth_on_proxied_notifications_sse(self, client):
        """api_key query param should authenticate on proxied /agent/notifications/sse."""
        resp = client.get(f"/api/agents/claw/agent/notifications/sse?api_key={API_KEY}")
        assert resp.status_code == 200
        assert resp.json()["type"] == "sse"

    def test_query_param_auth_on_proxied_agent_stream(self, client):
        """api_key query param should authenticate on proxied /agent/stream."""
        resp = client.post(f"/api/agents/claw/agent/stream?api_key={API_KEY}")
        assert resp.status_code == 200
        assert resp.json()["type"] == "stream"

    def test_query_param_auth_works_for_different_agent_ids(self, client):
        """api_key query param should work for any agent_id on SSE paths."""
        resp = client.get(
            f"/api/agents/other-agent/agent/notifications/sse?api_key={API_KEY}"
        )
        assert resp.status_code == 200
        assert resp.json()["agent"] == "other-agent"


class TestQueryParamAuthRejectedOnNonSSEPaths:
    """Query param auth must be rejected on non-SSE endpoints."""

    def test_query_param_rejected_on_list_agents(self, client):
        """api_key query param should NOT authenticate on /api/agents."""
        resp = client.get(f"/api/agents?api_key={API_KEY}")
        assert resp.status_code == 401

    def test_query_param_rejected_on_start_agent(self, client):
        """api_key query param should NOT authenticate on /api/agents/{id}/start."""
        resp = client.post(f"/api/agents/claw/start?api_key={API_KEY}")
        assert resp.status_code == 401

    def test_query_param_rejected_on_stop_agent(self, client):
        """api_key query param should NOT authenticate on /api/agents/{id}/stop."""
        resp = client.post(f"/api/agents/claw/stop?api_key={API_KEY}")
        assert resp.status_code == 401

    def test_query_param_rejected_on_agent_status(self, client):
        """api_key query param should NOT authenticate on /api/agents/{id}/status."""
        resp = client.get(f"/api/agents/claw/status?api_key={API_KEY}")
        assert resp.status_code == 401

    def test_query_param_rejected_on_agent_logs(self, client):
        """api_key query param should NOT authenticate on /api/agents/{id}/logs."""
        resp = client.get(f"/api/agents/claw/logs?api_key={API_KEY}")
        assert resp.status_code == 401

    def test_query_param_rejected_on_proxied_conversations(self, client):
        """api_key query param should NOT authenticate on proxied non-SSE paths."""
        resp = client.get(f"/api/agents/claw/api/conversations?api_key={API_KEY}")
        assert resp.status_code == 401


class TestHeaderAuthStillWorks:
    """Header-based auth should continue to work on all endpoints."""

    def test_header_auth_on_list_agents(self, client):
        """X-API-Key header should work on /api/agents."""
        resp = client.get("/api/agents", headers={"X-API-Key": API_KEY})
        assert resp.status_code == 200

    def test_bearer_auth_on_start_agent(self, client):
        """Bearer token should work on /api/agents/{id}/start."""
        resp = client.post(
            "/api/agents/claw/start",
            headers={"Authorization": f"Bearer {API_KEY}"},
        )
        assert resp.status_code == 200

    def test_header_auth_on_proxied_sse(self, client):
        """X-API-Key header should also work on proxied SSE endpoints."""
        resp = client.get(
            "/api/agents/claw/agent/notifications/sse",
            headers={"X-API-Key": API_KEY},
        )
        assert resp.status_code == 200

    def test_header_auth_on_proxied_conversations(self, client):
        """X-API-Key header should work on proxied non-SSE endpoints."""
        resp = client.get(
            "/api/agents/claw/api/conversations",
            headers={"X-API-Key": API_KEY},
        )
        assert resp.status_code == 200

    def test_no_auth_returns_401(self, client):
        """Requests without any auth should return 401."""
        resp = client.get("/api/agents")
        assert resp.status_code == 401


class TestWrongKeyRejected:
    """Wrong API key should be rejected everywhere."""

    def test_wrong_query_param_on_proxied_sse(self, client):
        """Wrong api_key query param should be rejected even on SSE paths."""
        resp = client.get(
            "/api/agents/claw/agent/notifications/sse?api_key=wrong-key"
        )
        assert resp.status_code == 401

    def test_wrong_header_key(self, client):
        """Wrong X-API-Key header should be rejected."""
        resp = client.get("/api/agents", headers={"X-API-Key": "wrong-key"})
        assert resp.status_code == 401


class TestSSEPathSuffixesConstant:
    """Test that SSE_PATH_SUFFIXES is correctly defined in host.py."""

    def test_host_sse_path_suffixes_defined(self):
        """host.py should define SSE_PATH_SUFFIXES."""
        from host import SSE_PATH_SUFFIXES

        assert "/agent/notifications/sse" in SSE_PATH_SUFFIXES
        assert "/agent/stream" in SSE_PATH_SUFFIXES

    def test_host_sse_path_suffixes_is_tuple(self):
        """SSE_PATH_SUFFIXES should be a tuple."""
        from host import SSE_PATH_SUFFIXES

        assert isinstance(SSE_PATH_SUFFIXES, tuple)
