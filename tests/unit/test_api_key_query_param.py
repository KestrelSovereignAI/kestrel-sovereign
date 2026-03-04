"""
Unit tests for API key query parameter restriction (GitHub issue #149).

Verifies that API key authentication via ?api_key= query parameter
is only accepted on SSE/streaming endpoints, not on all endpoints.
Keys in URLs get logged in access logs, proxy logs, and browser history,
so query param auth must be restricted to endpoints that require it
(EventSource/SSE cannot send custom headers).
"""

import os
import pytest
from unittest.mock import patch
from fastapi import FastAPI
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.testclient import TestClient
from starlette.middleware.sessions import SessionMiddleware

API_KEY = "test-api-key-for-sse-restriction"


@pytest.fixture
def app():
    """Create a minimal FastAPI app that mirrors server.py auth middleware."""
    import secrets as secrets_mod

    env = {
        "KESTREL_API_KEY": API_KEY,
        "KESTREL_SESSION_SECRET": "test-session-secret",
    }
    with patch.dict(os.environ, env):
        from server import SSE_PATHS

        test_app = FastAPI()

        @test_app.middleware("http")
        async def auth_middleware(request, call_next):
            """Mirrors the auth_middleware in server.py."""
            public_paths = ["/health"]

            if request.url.path in public_paths:
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
            if api_key_query and request.url.path in SSE_PATHS:
                if secrets_mod.compare_digest(api_key_query, expected_key):
                    return await call_next(request)

            return JSONResponse(
                content={"detail": "Invalid or missing API Key"},
                status_code=401,
            )

        test_app.add_middleware(
            SessionMiddleware,
            secret_key="test-session-secret",
            session_cookie="kestrel_session",
        )

        # Register test endpoints that mirror the real route paths
        @test_app.get("/agent/notifications/sse")
        def notifications_sse():
            return {"type": "sse"}

        @test_app.post("/agent/stream")
        def agent_stream():
            return {"type": "stream"}

        @test_app.get("/api/conversations")
        def conversations():
            return {"conversations": []}

        @test_app.post("/agent/invoke")
        def agent_invoke():
            return {"response": "ok"}

        @test_app.get("/api/memories")
        def memories():
            return {"memories": []}

        @test_app.get("/health")
        def health():
            return {"status": "ok"}

        yield test_app


@pytest.fixture
def client(app):
    return TestClient(app)


class TestQueryParamAuthOnSSEPaths:
    """Query param auth should work on SSE/streaming endpoints."""

    def test_query_param_auth_on_notifications_sse(self, client):
        """api_key query param should authenticate on /agent/notifications/sse."""
        resp = client.get(f"/agent/notifications/sse?api_key={API_KEY}")
        assert resp.status_code == 200
        assert resp.json() == {"type": "sse"}

    def test_query_param_auth_on_agent_stream(self, client):
        """api_key query param should authenticate on /agent/stream."""
        resp = client.post(f"/agent/stream?api_key={API_KEY}")
        assert resp.status_code == 200
        assert resp.json() == {"type": "stream"}


class TestQueryParamAuthRejectedOnOtherPaths:
    """Query param auth must be rejected on non-SSE endpoints."""

    def test_query_param_rejected_on_conversations(self, client):
        """api_key query param should NOT authenticate on /api/conversations."""
        resp = client.get(f"/api/conversations?api_key={API_KEY}")
        assert resp.status_code == 401

    def test_query_param_rejected_on_agent_invoke(self, client):
        """api_key query param should NOT authenticate on /agent/invoke."""
        resp = client.post(f"/agent/invoke?api_key={API_KEY}")
        assert resp.status_code == 401

    def test_query_param_rejected_on_memories(self, client):
        """api_key query param should NOT authenticate on /api/memories."""
        resp = client.get(f"/api/memories?api_key={API_KEY}")
        assert resp.status_code == 401


class TestHeaderAuthStillWorks:
    """Header-based auth should continue to work on all endpoints."""

    def test_header_auth_on_conversations(self, client):
        """X-API-Key header should still work on non-SSE endpoints."""
        resp = client.get(
            "/api/conversations",
            headers={"X-API-Key": API_KEY},
        )
        assert resp.status_code == 200

    def test_bearer_auth_on_conversations(self, client):
        """Bearer token should still work on non-SSE endpoints."""
        resp = client.get(
            "/api/conversations",
            headers={"Authorization": f"Bearer {API_KEY}"},
        )
        assert resp.status_code == 200

    def test_header_auth_on_sse(self, client):
        """X-API-Key header should also work on SSE endpoints."""
        resp = client.get(
            "/agent/notifications/sse",
            headers={"X-API-Key": API_KEY},
        )
        assert resp.status_code == 200

    def test_no_auth_returns_401(self, client):
        """Requests without any auth should return 401."""
        resp = client.get("/api/conversations")
        assert resp.status_code == 401


class TestWrongKeyRejected:
    """Wrong API key should be rejected everywhere."""

    def test_wrong_query_param_on_sse(self, client):
        """Wrong api_key query param should be rejected even on SSE paths."""
        resp = client.get("/agent/notifications/sse?api_key=wrong-key")
        assert resp.status_code == 401

    def test_wrong_header_key(self, client):
        """Wrong X-API-Key header should be rejected."""
        resp = client.get(
            "/api/conversations",
            headers={"X-API-Key": "wrong-key"},
        )
        assert resp.status_code == 401


class TestSSEPathsConstant:
    """Test that SSE_PATHS is correctly defined in server.py."""

    def test_sse_paths_contains_expected_paths(self):
        """SSE_PATHS should contain the notification SSE and stream endpoints."""
        from server import SSE_PATHS

        assert "/agent/notifications/sse" in SSE_PATHS
        assert "/agent/stream" in SSE_PATHS

    def test_sse_paths_is_a_set(self):
        """SSE_PATHS should be a set for O(1) lookup."""
        from server import SSE_PATHS

        assert isinstance(SSE_PATHS, set)
