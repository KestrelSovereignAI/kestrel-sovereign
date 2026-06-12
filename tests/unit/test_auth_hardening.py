"""Auth hardening (#1724): fail-closed allowlist, JWT/API-key decoupling,
/auth/token rate limit, and narrowed bootstrap-key host gating."""
from __future__ import annotations

import pytest


# ---------------------------------------------------------------------------
# _email_authorized — fail closed on a missing allowlist
# ---------------------------------------------------------------------------
class TestEmailAuthorized:
    def _auth(self):
        from kestrel_sovereign.endpoints.auth_oauth import _email_authorized
        return _email_authorized

    def test_allowlisted_email_admitted(self, monkeypatch):
        monkeypatch.setenv("KESTREL_ALLOWED_EMAILS", "a@x.com,b@x.com")
        assert self._auth()("a@x.com") is True
        assert self._auth()("c@x.com") is False

    def test_empty_allowlist_with_require_oauth_denies(self, monkeypatch):
        """The headline fix: REQUIRE_OAUTH + no allowlist must NOT admit everyone."""
        monkeypatch.delenv("KESTREL_ALLOWED_EMAILS", raising=False)
        monkeypatch.setenv("KESTREL_REQUIRE_OAUTH", "true")
        assert self._auth()("anyone@gmail.com") is False

    def test_empty_allowlist_without_require_oauth_admits(self, monkeypatch):
        """Dev/local (OAuth not required) keeps the permissive default."""
        monkeypatch.delenv("KESTREL_ALLOWED_EMAILS", raising=False)
        monkeypatch.delenv("KESTREL_REQUIRE_OAUTH", raising=False)
        assert self._auth()("anyone@gmail.com") is True

    def test_wildcard_allows_any_authenticated(self, monkeypatch):
        monkeypatch.setenv("KESTREL_ALLOWED_EMAILS", "*")
        monkeypatch.setenv("KESTREL_REQUIRE_OAUTH", "true")
        assert self._auth()("anyone@gmail.com") is True


# ---------------------------------------------------------------------------
# _get_jwt_secret — decouple signing key from the API key
# ---------------------------------------------------------------------------
class TestJwtSecret:
    def _secret(self):
        from kestrel_sovereign.endpoints.auth_oauth import _get_jwt_secret
        return _get_jwt_secret

    def test_prefers_dedicated_secret(self, monkeypatch):
        monkeypatch.setenv("JWT_SECRET_KEY", "dedicated")
        monkeypatch.setenv("KESTREL_API_KEY", "api-key")
        assert self._secret()() == "dedicated"

    def test_derives_distinct_secret_from_api_key(self, monkeypatch):
        monkeypatch.delenv("JWT_SECRET_KEY", raising=False)
        monkeypatch.delenv("KESTREL_SESSION_SECRET", raising=False)
        monkeypatch.setenv("KESTREL_API_KEY", "super-secret-api-key")
        derived = self._secret()()
        # The signing key is NOT the raw API key (decoupled), and is stable.
        assert derived != "super-secret-api-key"
        assert derived == self._secret()()

    def test_raises_when_nothing_configured(self, monkeypatch):
        for k in ("JWT_SECRET_KEY", "KESTREL_SESSION_SECRET", "KESTREL_API_KEY"):
            monkeypatch.delenv(k, raising=False)
        with pytest.raises(RuntimeError):
            self._secret()()


# ---------------------------------------------------------------------------
# bootstrap-key host gating — narrowed from 172.16.0.0/12
# ---------------------------------------------------------------------------
class TestBootstrapHostGating:
    def _allowed(self):
        from kestrel_sovereign.security.bootstrap_access import is_bootstrap_host_allowed
        return is_bootstrap_host_allowed

    def test_loopback_and_gateway_allowed(self, monkeypatch):
        monkeypatch.delenv("KESTREL_BOOTSTRAP_ALLOWED_HOSTS", raising=False)
        for h in ("127.0.0.1", "::1", "172.17.0.1"):
            assert self._allowed()(h) is True

    def test_arbitrary_bridge_container_rejected(self, monkeypatch):
        """A sibling container at e.g. 172.18.0.5 (was inside the old /12) is now
        rejected — only the gateway is trusted."""
        monkeypatch.delenv("KESTREL_BOOTSTRAP_ALLOWED_HOSTS", raising=False)
        assert self._allowed()("172.18.0.5") is False
        assert self._allowed()("172.20.10.10") is False
        assert self._allowed()(None) is False

    def test_explicit_allowlist_extends(self, monkeypatch):
        monkeypatch.setenv("KESTREL_BOOTSTRAP_ALLOWED_HOSTS", "10.1.2.3")
        assert self._allowed()("10.1.2.3") is True


# ---------------------------------------------------------------------------
# /auth/token is rate-limited
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_auth_token_is_rate_limited(monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from slowapi import _rate_limit_exceeded_handler
    from slowapi.errors import RateLimitExceeded
    from slowapi.middleware import SlowAPIMiddleware
    from kestrel_sovereign.rate_limit import limiter
    from kestrel_sovereign.endpoints.auth_oauth import router

    monkeypatch.delenv("KESTREL_ALLOWED_EMAILS", raising=False)
    monkeypatch.delenv("KESTREL_REQUIRE_OAUTH", raising=False)

    app = FastAPI()
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
    app.add_middleware(SlowAPIMiddleware)
    app.include_router(router)

    client = TestClient(app)
    body = {"email": "x@y.com", "password": "wrong"}
    statuses = [client.post("/auth/token", json=body).status_code for _ in range(7)]
    # 5/minute → at least one 429 within 7 attempts.
    assert 429 in statuses
