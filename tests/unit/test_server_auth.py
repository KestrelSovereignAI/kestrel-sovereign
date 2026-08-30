"""Auth middleware status-correctness tests (#2490).

Authentication failures alone produce 401. Once authentication succeeds,
downstream failures follow FastAPI error handling: an unhandled route
exception yields the framework's generic 500 (never 401, never a
secret-bearing traceback body) and controlled ``HTTPException`` statuses
pass through intact.
"""

import os

import pytest
from fastapi import FastAPI, HTTPException, Request
from fastapi.testclient import TestClient
from starlette.middleware.sessions import SessionMiddleware

from kestrel_sovereign import server as server_module
from kestrel_sovereign.features.peers.directory import LocalHostPeerDirectory

API_KEY = "unit-test-key-2490"
PEER_KEY = "unit-test-peer-key-3148"
SENTINEL = "sentinel downstream failure"

# Lanes the middleware accepts credentials on. The query-param lane is only
# honored on SSE-suffixed paths, so lane-specific tests pick their path.
AUTH_LANES = ["header", "bearer", "query", "session"]


def _build_app():
    """Fresh app wired with the canonical Kestrel auth middleware."""
    app = FastAPI()
    app.middleware("http")(server_module.auth_middleware)
    app.add_middleware(
        SessionMiddleware,
        secret_key="test-session-secret",
        session_cookie="kestrel_session",
    )

    @app.get("/api/agent/info")
    async def ok_route():
        return {"ok": True}

    # SSE-suffixed paths so the query-param lane is exercised on both the
    # success route and the downstream-failure route.
    @app.get("/api/agent/notifications/sse")
    async def sse_ok_route():
        return {"ok": True}

    @app.get("/api/agent/stream")
    async def failing_route():
        raise RuntimeError(SENTINEL)

    @app.get("/api/agent/teapot")
    async def controlled_route():
        raise HTTPException(status_code=418, detail="short and stout")

    @app.get("/api/test/login")
    async def login(request: Request):
        request.session["user_email"] = "operator@example.com"
        return {"ok": True}

    @app.get("/api/test/caller")
    async def caller_route(request: Request):
        caller = request.state.caller
        return {
            "role": caller.role.value,
            "auth_method": caller.auth_method.value,
            "is_sovereign": caller.is_sovereign,
        }

    @app.get("/api/agents")
    async def peer_directory_route(request: Request):
        caller = request.state.caller
        return {
            "role": caller.role.value,
            "auth_method": caller.auth_method.value,
            "is_sovereign": caller.is_sovereign,
        }

    return app


@pytest.fixture()
def client(monkeypatch):
    monkeypatch.setenv("KESTREL_API_KEY", API_KEY)
    monkeypatch.setenv("KESTREL_PEER_API_KEY", PEER_KEY)
    with TestClient(_build_app(), raise_server_exceptions=False) as test_client:
        yield test_client


def _lane_kwargs(client, lane, key=API_KEY):
    """Request kwargs for an authentication lane (logs in for the session lane)."""
    if lane == "header":
        return {"headers": {"X-API-Key": key}}
    if lane == "bearer":
        return {"headers": {"Authorization": f"Bearer {key}"}}
    if lane == "query":
        return {"params": {"api_key": key}}
    if lane == "session":
        login = client.get("/api/test/login", headers={"X-API-Key": API_KEY})
        assert login.status_code == 200
        return {}
    raise AssertionError(f"unknown lane {lane}")


@pytest.mark.parametrize("lane", AUTH_LANES)
def test_valid_credentials_reach_route(client, lane):
    path = "/api/agent/notifications/sse" if lane == "query" else "/api/agent/info"
    response = client.get(path, **_lane_kwargs(client, lane))
    assert response.status_code == 200
    assert response.json() == {"ok": True}


def test_local_peer_transport_key_is_authenticated_but_not_sovereign(client):
    headers = LocalHostPeerDirectory(
        "http://localhost:8888",
        peer_api_key=PEER_KEY,
    )._headers()

    response = client.get("/api/agents", headers=headers)

    assert response.status_code == 200
    assert response.json() == {
        "role": "authenticated",
        "auth_method": "internal",
        "is_sovereign": False,
    }


@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("GET", "/api/agents"),
        ("POST", "/api/agents/Claw/api/agent/invoke"),
        ("POST", "/api/agents/Claw/api/agent/tasks/send"),
        ("GET", "/api/agents/Claw/api/agent/tasks/task-1"),
        ("GET", "/api/agents/Claw/api/agent/tasks/task-1/subscribe"),
    ],
)
def test_local_peer_transport_route_allowlist_covers_directory_contract(
    method, path,
):
    assert server_module._is_local_peer_transport_route(method, path) is True


@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("get", "/api/test/caller"),
        ("post", "/api/agents"),
        ("delete", "/api/agents/Claw"),
        ("post", "/api/features/PeersFeature/disable"),
    ],
)
def test_local_peer_transport_key_is_rejected_outside_peer_routes(
    client, method, path,
):
    response = getattr(client, method)(
        path,
        headers={"X-Kestrel-Peer-Key": PEER_KEY},
    )

    assert response.status_code == 401


def test_peer_key_is_not_accepted_on_sovereign_header(client):
    response = client.get(
        "/api/test/caller",
        headers={"X-API-Key": PEER_KEY},
    )

    assert response.status_code == 401


def test_legacy_peer_marker_cannot_downgrade_or_upgrade_sovereign_key(client):
    response = client.get(
        "/api/test/caller",
        headers={
            "X-API-Key": API_KEY,
            "X-Kestrel-Peer-Transport": "local-host-v1",
        },
    )

    assert response.status_code == 200
    assert response.json()["is_sovereign"] is True


def test_peer_and_sovereign_keys_must_be_distinct(monkeypatch):
    monkeypatch.setenv("KESTREL_API_KEY", "same-key")
    monkeypatch.setenv("KESTREL_PEER_API_KEY", "same-key")

    with pytest.raises(RuntimeError, match="must be distinct"):
        server_module.get_peer_api_key()


def test_generated_peer_key_is_stable_and_distinct(monkeypatch):
    monkeypatch.setenv("KESTREL_API_KEY", "sovereign-key")
    monkeypatch.delenv("KESTREL_PEER_API_KEY", raising=False)

    first = server_module.get_peer_api_key()
    second = server_module.get_peer_api_key()

    assert first == second == os.environ["KESTREL_PEER_API_KEY"]
    assert first != "sovereign-key"


@pytest.mark.parametrize(
    ("lane", "path"),
    [
        ("header", "/api/agent/info"),
        ("bearer", "/api/agent/info"),
        ("query", "/api/agent/notifications/sse"),
    ],
)
def test_invalid_credentials_are_401(client, lane, path):
    response = client.get(path, **_lane_kwargs(client, lane, key="wrong-key-2490"))
    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid or missing API Key"


def test_missing_credentials_are_401(client):
    response = client.get("/api/agent/info")
    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid or missing API Key"
    assert response.json()["error"]["code"] == "authentication_required"
    assert response.json()["error"]["correlation_id"] == response.headers[
        "X-Correlation-ID"
    ]


def test_query_param_lane_stays_restricted_to_sse_paths(client):
    # A valid key via query param on a non-SSE path must not authenticate.
    response = client.get("/api/agent/info", params={"api_key": API_KEY})
    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid or missing API Key"


@pytest.mark.parametrize("lane", AUTH_LANES)
def test_authenticated_downstream_failure_is_generic_500_not_401(client, lane):
    response = client.get("/api/agent/stream", **_lane_kwargs(client, lane))
    assert response.status_code == 500
    # Framework generic body — no auth translation, no traceback, no secrets.
    assert response.text == "Internal Server Error"
    assert SENTINEL not in response.text
    assert API_KEY not in response.text
    assert "Authentication failed" not in response.text


@pytest.mark.parametrize("lane", ["header", "bearer", "session"])
def test_authenticated_downstream_http_exception_status_is_preserved(client, lane):
    response = client.get("/api/agent/teapot", **_lane_kwargs(client, lane))
    assert response.status_code == 418
    assert response.json()["detail"] == "short and stout"


def test_credential_evaluation_crash_still_produces_auth_401(client, monkeypatch):
    def _boom():
        raise RuntimeError("auth backend unavailable")

    monkeypatch.setattr(server_module, "get_api_key", _boom)
    response = client.get("/api/agent/info", headers={"X-API-Key": API_KEY})
    assert response.status_code == 401
    assert response.json()["detail"] == "Authentication failed"
    assert response.json()["error"]["code"] == "authentication_failed"
    assert response.json()["error"]["correlation_id"] == response.headers[
        "X-Correlation-ID"
    ]
