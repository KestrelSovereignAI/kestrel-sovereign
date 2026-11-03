"""Actual middleware credentials remain bound to their original request."""

from base64 import b64encode
import asyncio
from contextvars import Context
from dataclasses import replace
from datetime import UTC, datetime
import json
import os

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient
from itsdangerous import TimestampSigner
import jwt
import httpx2
import pytest

from kestrel_sovereign import server
from kestrel_sovereign.endpoints.auth_oauth import _create_jwt
from kestrel_sovereign.security.capability_authority import HostCapabilityAccessDenied
from kestrel_sovereign.security.csrf import CSRFError
from kestrel_sovereign.security.request_auth import (
    AuthenticatedSessionMiddleware, InteractiveAuthenticationLifetimeMiddleware,
    RequestAuthenticationService, current_request_authentication,
)
from kestrel_sovereign.turn_scope import capture_turn_scope


API_KEY = "owned-private-request-api-key"
SESSION_KEY = "owned-private-request-session-key"
JWT_KEY = "owned-private-request-jwt-key-with-enough-entropy"
ALICE = "alice@example.test"


@pytest.fixture
def authenticated_app(monkeypatch):
    monkeypatch.setenv("KESTREL_API_KEY", API_KEY)
    monkeypatch.setenv("JWT_SECRET_KEY", JWT_KEY)
    monkeypatch.setenv("KESTREL_REQUIRE_OAUTH", "1")
    monkeypatch.setenv("KESTREL_ALLOWED_EMAILS", ALICE)
    monkeypatch.setenv("KESTREL_A2A_TRANSPORT_KEY", "owned-peer-directory-key")
    app = FastAPI()
    app.state.request_authentication = RequestAuthenticationService(
        api_key=lambda: server.get_api_key(generate_if_missing=False))
    app.add_middleware(InteractiveAuthenticationLifetimeMiddleware)
    app.middleware("http")(server.auth_middleware)
    app.add_middleware(AuthenticatedSessionMiddleware, secret_key=SESSION_KEY,
                       session_cookie="kestrel_session", max_age=600)
    bindings = []
    producers = []
    carried_scopes = []

    @app.get("/api/owned-carried")
    async def carried():
        original = current_request_authentication()
        snapshot = capture_turn_scope(None)
        carried_scopes.append(snapshot)
        async def foreign_reader():
            assert current_request_authentication() is None
            with snapshot.bind():
                assert current_request_authentication() is original
                original.require_current()
        await asyncio.create_task(foreign_reader(), context=Context())
        return {"carried": True}

    @app.get("/api/agents")
    async def roster(request: Request):
        assert current_request_authentication() is None
        return {"lane": request.state.caller.auth_method.value}

    @app.get("/api/agent/notifications/sse")
    async def eventsource(request: Request):
        assert current_request_authentication() is None
        return {"lane": request.state.caller.auth_method.value}

    @app.get("/api/owned-nested")
    async def nested():
        original = current_request_authentication()
        assert original is not None
        child = FastAPI()
        child.add_middleware(InteractiveAuthenticationLifetimeMiddleware)
        @child.get("/public")
        async def public():
            assert current_request_authentication() is None
            return {"public": True}
        async with httpx2.AsyncClient(transport=httpx2.ASGITransport(app=child), base_url="http://nested") as client:
            response = await client.get("/public")
            assert response.status_code == 200
        assert current_request_authentication() is original
        original.require_current()
        return {"isolated": True}

    @app.get("/api/owned-login")
    async def login(request: Request):
        request.session["user_email"] = ALICE
        return {"signed_in": True}

    @app.api_route("/api/private", methods=["GET", "POST"])
    async def private(request: Request):
        service = request.app.state.request_authentication
        try:
            with service.bind(request) as binding:
                producer = current_request_authentication()
                assert producer is not None and producer.caller is binding.caller
                producer.require_current()
                producers.append(producer)
                bindings.append(binding)
                probe = request.headers.get("x-owned-probe")
                if probe == "rotate-api":
                    os.environ["KESTREL_API_KEY"] = "rotated-owned-key"
                elif probe == "remove-api":
                    os.environ.pop("KESTREL_API_KEY")
                elif probe == "rotate-jwt":
                    os.environ["JWT_SECRET_KEY"] = "rotated-owned-jwt-key-with-enough-entropy"
                elif probe == "revoke-email":
                    os.environ["KESTREL_ALLOWED_EMAILS"] = "another@example.test"
                elif probe == "replace-caller":
                    request.state.caller = replace(request.state.caller)
                elif probe == "replace-service":
                    request.app.state.request_authentication = RequestAuthenticationService(api_key=lambda: API_KEY)
                elif probe == "replace-session":
                    request.scope["session"] = type(request.session)(request.session)
                binding.require_current()
                return {"private": "original", "expires_at": binding.expires_at.isoformat() if binding.expires_at else None}
        except (HostCapabilityAccessDenied, CSRFError):
            return JSONResponse({"error": "private_access_denied"}, status_code=403)

    with TestClient(app, raise_server_exceptions=True) as client:
        app.state.producer_bindings = producers
        app.state.carried_scopes = carried_scopes
        yield client, bindings


def credentials(client, lane):
    if lane == "header":
        return {"X-API-Key": API_KEY}
    if lane == "bearer":
        return {"Authorization": "Bearer " + API_KEY}
    if lane == "jwt":
        return {"Authorization": "Bearer " + _create_jwt(ALICE)}
    assert lane == "cookie"
    response = client.get("/api/owned-login", headers={"X-API-Key": API_KEY})
    assert response.status_code == 200
    return {}


@pytest.mark.parametrize("lane", ["header", "bearer", "jwt", "cookie"])
def test_native_interactive_credential_and_request_lifetime(authenticated_app, lane):
    client, bindings = authenticated_app
    response = client.get("/api/private", headers=credentials(client, lane))
    assert response.status_code == 200 and response.json()["private"] == "original"
    if lane in {"jwt", "cookie"}:
        assert datetime.fromisoformat(response.json()["expires_at"]) > datetime.now(UTC)
    assert len(bindings) == 1
    with pytest.raises(HostCapabilityAccessDenied, match="ended"):
        bindings[0].require_current()
    with pytest.raises(HostCapabilityAccessDenied, match="ended"):
        client.app.state.producer_bindings[0].require_current()


@pytest.mark.parametrize(("lane", "probe"), [
    ("header", "rotate-api"), ("bearer", "remove-api"),
    ("jwt", "rotate-jwt"), ("jwt", "revoke-email"),
    ("cookie", "revoke-email"), ("cookie", "replace-session"),
    ("header", "replace-caller"), ("header", "replace-service"),
])
def test_original_credentials_rechecked_inside_same_request(authenticated_app, lane, probe):
    client, bindings = authenticated_app
    headers = credentials(client, lane)
    response = client.get("/api/private", headers={**headers, "X-Owned-Probe": probe})
    assert len(bindings) == 1 and response.status_code == 403
    assert "private" not in response.json() and API_KEY not in response.text
    if probe == "remove-api":
        assert os.environ.get("KESTREL_API_KEY") is None, "checking authority minted another credential"


def test_cookie_mutations_use_the_canonical_csrf_check(authenticated_app):
    client, bindings = authenticated_app
    credentials(client, "cookie")
    assert client.post("/api/private").status_code == 403
    assert not bindings
    client.cookies.set("kestrel_csrf", "owned-token")
    response = client.post("/api/private", headers={"X-CSRF-Token": "owned-token", "Origin": "http://testserver"})
    assert response.status_code == 200
    assert client.post("/api/private", headers={"X-CSRF-Token": "owned-token", "Origin": "https://foreign.test"}).status_code == 403


def test_jwt_without_expiry_cannot_acquire_private_delivery_lease(authenticated_app):
    client, bindings = authenticated_app
    token = jwt.encode({"sub": ALICE}, JWT_KEY, algorithm="HS256")
    response = client.get("/api/private", headers={"Authorization": "Bearer " + token})
    assert response.status_code == 403 and not bindings


def test_caller_descriptor_and_query_key_are_not_private_credentials(authenticated_app):
    client, bindings = authenticated_app
    response = client.post("/api/private", params={"api_key": API_KEY},
        json={"caller": {"role": "sovereign", "identity": ALICE}, "agent_id": "guessed"})
    assert response.status_code == 401 and not bindings


def test_expired_native_session_is_rejected_before_private_access(authenticated_app):
    client, bindings = authenticated_app
    class OldSigner(TimestampSigner):
        def get_timestamp(self):
            return super().get_timestamp() - 700
    raw = OldSigner(SESSION_KEY).sign(b64encode(json.dumps({"user_email": ALICE}).encode())).decode()
    client.cookies.set("kestrel_session", raw)
    assert client.get("/api/private").status_code == 401 and not bindings


def test_peer_and_eventsource_lanes_do_not_gain_interactive_authority(authenticated_app):
    client, bindings = authenticated_app
    peer = client.get("/api/agents", headers={"X-Kestrel-A2A-Key": "owned-peer-directory-key", "X-API-Key": API_KEY})
    assert peer.status_code == 200 and peer.json()["lane"] == "a2a_transport"
    sse = client.get("/api/agent/notifications/sse", params={"api_key": API_KEY})
    assert sse.status_code == 200 and sse.json()["lane"] == "api_key"
    assert not bindings


def test_nested_public_asgi_request_cannot_borrow_interactive_owner(authenticated_app):
    client, _ = authenticated_app
    assert client.get("/api/owned-nested", headers={"X-API-Key": API_KEY}).json() == {"isolated": True}


def test_original_http_authentication_follows_foreign_turn_callbacks_and_expires_with_its_owner(authenticated_app):
    client, _ = authenticated_app
    response = client.get("/api/owned-carried", headers={"X-API-Key": API_KEY})
    assert response.status_code == 200 and response.json() == {"carried": True}
    with client.app.state.carried_scopes[0].bind():
        with pytest.raises(HostCapabilityAccessDenied, match="ended"):
            current_request_authentication()
