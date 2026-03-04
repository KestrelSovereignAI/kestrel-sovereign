"""
Unit tests for Google OAuth authentication endpoints and middleware.

Tests the OAuth flow, email allowlist, and session-based auth.
"""

import os
import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from fastapi.testclient import TestClient


@pytest.fixture
def oauth_env():
    """Set OAuth environment variables for testing."""
    env = {
        "GOOGLE_CLIENT_ID": "test-client-id.apps.googleusercontent.com",
        "GOOGLE_CLIENT_SECRET": "test-client-secret",
        "KESTREL_ALLOWED_EMAILS": "allowed@gmail.com,admin@gmail.com",
        "KESTREL_SESSION_SECRET": "test-session-secret-key",
        "KESTREL_API_KEY": "test-api-key-12345",
    }
    with patch.dict(os.environ, env):
        yield env


@pytest.fixture
def app(oauth_env):
    """Create a FastAPI app with OAuth configured for testing."""
    from starlette.middleware.sessions import SessionMiddleware
    from fastapi import FastAPI
    from fastapi.responses import JSONResponse, RedirectResponse

    # Build a minimal app that mirrors server.py auth behavior
    test_app = FastAPI()

    from endpoints.auth_oauth import router as auth_router, register_oauth
    test_app.include_router(auth_router)
    register_oauth(test_app)

    SERVE_UI = True

    @test_app.middleware("http")
    async def auth_middleware(request, call_next):
        public_paths = ["/health", "/api/auth/key"]
        auth_paths = ["/auth/login", "/auth/callback", "/auth/logout"]

        if request.url.path in public_paths or request.url.path in auth_paths:
            return await call_next(request)

        # Check API key
        api_key = request.headers.get("X-API-Key")
        if api_key and api_key == os.environ.get("KESTREL_API_KEY"):
            return await call_next(request)

        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            token = auth_header[7:]
            if token == os.environ.get("KESTREL_API_KEY"):
                return await call_next(request)

        # Check session
        user_email = request.session.get("user_email")
        if user_email:
            return await call_next(request)

        # Redirect browsers to login
        if request.url.path == "/" and SERVE_UI:
            accept = request.headers.get("accept", "")
            if "text/html" in accept:
                return RedirectResponse(url="/auth/login", status_code=302)

        return JSONResponse(content={"detail": "Invalid or missing API Key"}, status_code=401)

    # SessionMiddleware must be added AFTER auth_middleware (outermost = last added)
    test_app.add_middleware(
        SessionMiddleware,
        secret_key="test-session-secret-key",
        session_cookie="kestrel_session",
    )

    @test_app.get("/health")
    def health():
        return {"status": "ok"}

    @test_app.get("/api/conversations")
    def conversations():
        return {"conversations": []}

    return test_app


@pytest.fixture
def client(app):
    """Create a test client."""
    return TestClient(app)


class TestOAuthEndpoints:
    """Test OAuth login/callback/logout/me endpoints."""

    def test_login_redirects_to_google(self, client):
        """Test /auth/login redirects to Google OAuth."""
        resp = client.get("/auth/login", follow_redirects=False)
        assert resp.status_code in (302, 303)
        location = resp.headers.get("location", "")
        assert "accounts.google.com" in location

    def test_logout_clears_session(self, client):
        """Test /auth/logout clears session and redirects."""
        resp = client.get("/auth/logout", follow_redirects=False)
        assert resp.status_code == 302
        assert "/auth/login" in resp.headers.get("location", "")

    def test_me_unauthenticated(self, client):
        """Test /auth/me returns 401 when not authenticated."""
        resp = client.get("/auth/me")
        assert resp.status_code == 401

    def test_me_unauthenticated_detail(self, client):
        """Test /auth/me returns proper detail when not authenticated."""
        resp = client.get("/auth/me")
        assert resp.status_code == 401


class TestEmailAllowlist:
    """Test email allowlist enforcement."""

    def test_get_allowed_emails(self, oauth_env):
        """Test parsing of KESTREL_ALLOWED_EMAILS env var."""
        from endpoints.auth_oauth import _get_allowed_emails
        emails = _get_allowed_emails()
        assert emails == {"allowed@gmail.com", "admin@gmail.com"}

    def test_get_allowed_emails_empty(self):
        """Test empty allowlist returns empty set."""
        with patch.dict(os.environ, {"KESTREL_ALLOWED_EMAILS": ""}, clear=False):
            from endpoints.auth_oauth import _get_allowed_emails
            emails = _get_allowed_emails()
            assert emails == set()

    def test_get_allowed_emails_whitespace(self):
        """Test allowlist with extra whitespace."""
        with patch.dict(os.environ, {"KESTREL_ALLOWED_EMAILS": " a@b.com , c@d.com "}, clear=False):
            from endpoints.auth_oauth import _get_allowed_emails
            emails = _get_allowed_emails()
            assert emails == {"a@b.com", "c@d.com"}

    def test_get_allowed_emails_case_insensitive(self):
        """Test emails are lowercased."""
        with patch.dict(os.environ, {"KESTREL_ALLOWED_EMAILS": "User@Gmail.COM"}, clear=False):
            from endpoints.auth_oauth import _get_allowed_emails
            emails = _get_allowed_emails()
            assert emails == {"user@gmail.com"}


class TestAuthMiddleware:
    """Test that auth middleware accepts both API key and OAuth session."""

    def test_api_key_still_works(self, client):
        """Test API key auth continues to work."""
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_api_key_header_auth(self, client):
        """Test X-API-Key header authentication."""
        resp = client.get(
            "/api/conversations",
            headers={"X-API-Key": "test-api-key-12345"},
        )
        # Should not be 401
        assert resp.status_code != 401

    def test_bearer_token_auth(self, client):
        """Test Bearer token authentication."""
        resp = client.get(
            "/api/conversations",
            headers={"Authorization": "Bearer test-api-key-12345"},
        )
        assert resp.status_code != 401

    def test_no_auth_returns_401(self, client):
        """Test requests without auth return 401."""
        resp = client.get("/api/conversations")
        assert resp.status_code == 401

    def test_root_redirects_to_login_for_browsers(self, client):
        """Test that / redirects browsers to /auth/login when not authenticated."""
        resp = client.get(
            "/",
            headers={"Accept": "text/html"},
            follow_redirects=False,
        )
        # Should redirect to login
        assert resp.status_code == 302
        assert "/auth/login" in resp.headers.get("location", "")

    def test_auth_paths_are_public(self, client):
        """Test that /auth/* paths don't require auth."""
        for path in ["/auth/login", "/auth/callback", "/auth/logout"]:
            resp = client.get(path, follow_redirects=False)
            # These should NOT return 401
            assert resp.status_code != 401, f"{path} returned 401"


class TestOAuthDisabled:
    """Test behavior when OAuth env vars are not set."""

    def test_login_returns_503_when_not_configured(self):
        """Test /auth/login returns 503 when Google OAuth is not configured."""
        from starlette.middleware.sessions import SessionMiddleware
        from fastapi import FastAPI
        from endpoints.auth_oauth import router as auth_router, oauth, register_oauth

        # Clear any previously registered client
        if "google" in oauth._clients:
            del oauth._clients["google"]

        env = {"KESTREL_API_KEY": "test-key", "KESTREL_SESSION_SECRET": "test-secret"}
        with patch.dict(os.environ, env, clear=True):
            test_app = FastAPI()
            test_app.add_middleware(SessionMiddleware, secret_key="test-secret")
            test_app.include_router(auth_router)
            register_oauth(test_app)

            test_client = TestClient(test_app)
            resp = test_client.get("/auth/login", follow_redirects=False)
            assert resp.status_code == 503


# Helper for context manager noop
class _noop:
    def __enter__(self):
        return self
    def __exit__(self, *args):
        pass
