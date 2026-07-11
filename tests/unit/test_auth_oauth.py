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
    """Mount the production OAuth router without copying server middleware."""
    from starlette.middleware.sessions import SessionMiddleware
    from fastapi import FastAPI

    test_app = FastAPI()

    from kestrel_sovereign.endpoints.auth_oauth import router as auth_router, register_oauth
    test_app.include_router(auth_router)
    register_oauth(test_app)
    test_app.add_middleware(
        SessionMiddleware,
        secret_key="test-session-secret-key",
        session_cookie="kestrel_session",
    )
    return test_app


@pytest.fixture
def client(app):
    """Create a test client."""
    return TestClient(app)


class TestOAuthEndpoints:
    """Test OAuth login/callback/logout/me endpoints."""

    def test_login_redirects_to_google(self, client):
        """Test /auth/login redirects to Google OAuth."""
        from fastapi.responses import RedirectResponse

        # Mock the OAuth redirect to avoid real network calls
        mock_redirect = RedirectResponse(
            url="https://accounts.google.com/o/oauth2/auth?client_id=test&redirect_uri=http://testserver/auth/callback",
            status_code=302
        )

        with patch("kestrel_sovereign.endpoints.auth_oauth.oauth") as mock_oauth:
            mock_oauth._clients = {"google": MagicMock()}
            mock_oauth.google.authorize_redirect = AsyncMock(return_value=mock_redirect)

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

class TestEmailAllowlist:
    """Test email allowlist enforcement."""

    def test_get_allowed_emails(self, oauth_env):
        """Test parsing of KESTREL_ALLOWED_EMAILS env var."""
        from kestrel_sovereign.endpoints.auth_oauth import _get_allowed_emails
        emails = _get_allowed_emails()
        assert emails == {"allowed@gmail.com", "admin@gmail.com"}

    def test_get_allowed_emails_empty(self):
        """Test empty allowlist returns empty set."""
        with patch.dict(os.environ, {"KESTREL_ALLOWED_EMAILS": ""}, clear=False):
            from kestrel_sovereign.endpoints.auth_oauth import _get_allowed_emails
            emails = _get_allowed_emails()
            assert emails == set()

    def test_get_allowed_emails_whitespace(self):
        """Test allowlist with extra whitespace."""
        with patch.dict(os.environ, {"KESTREL_ALLOWED_EMAILS": " a@b.com , c@d.com "}, clear=False):
            from kestrel_sovereign.endpoints.auth_oauth import _get_allowed_emails
            emails = _get_allowed_emails()
            assert emails == {"a@b.com", "c@d.com"}

    def test_get_allowed_emails_case_insensitive(self):
        """Test emails are lowercased."""
        with patch.dict(os.environ, {"KESTREL_ALLOWED_EMAILS": "User@Gmail.COM"}, clear=False):
            from kestrel_sovereign.endpoints.auth_oauth import _get_allowed_emails
            emails = _get_allowed_emails()
            assert emails == {"user@gmail.com"}

    def test_get_allowed_emails_semicolon_separator(self):
        """`;` works as separator (Cloud Run deploys can't use `,`)."""
        with patch.dict(os.environ, {"KESTREL_ALLOWED_EMAILS": "a@b.com;c@d.com"}, clear=False):
            from kestrel_sovereign.endpoints.auth_oauth import _get_allowed_emails
            emails = _get_allowed_emails()
            assert emails == {"a@b.com", "c@d.com"}

    def test_get_allowed_emails_mixed_separators(self):
        """Mixed `,` and `;` separators both work."""
        with patch.dict(os.environ, {"KESTREL_ALLOWED_EMAILS": "a@b.com, c@d.com;e@f.com"}, clear=False):
            from kestrel_sovereign.endpoints.auth_oauth import _get_allowed_emails
            emails = _get_allowed_emails()
            assert emails == {"a@b.com", "c@d.com", "e@f.com"}


class TestOAuthDisabled:
    """Test behavior when OAuth env vars are not set."""

    def test_login_returns_503_when_not_configured(self):
        """Test /auth/login returns 503 when Google OAuth is not configured."""
        from starlette.middleware.sessions import SessionMiddleware
        from fastapi import FastAPI
        from kestrel_sovereign.endpoints.auth_oauth import router as auth_router, oauth, register_oauth

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
