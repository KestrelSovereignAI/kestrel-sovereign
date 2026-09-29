"""Auth middleware status-correctness tests (#2490).

Authentication failures alone produce 401. Once authentication succeeds,
downstream failures follow FastAPI error handling: an unhandled route
exception yields the framework's generic 500 (never 401, never a
secret-bearing traceback body) and controlled ``HTTPException`` statuses
pass through intact.
"""

import os
import subprocess
import sys

import pytest
from fastapi import FastAPI, HTTPException, Request
from fastapi.testclient import TestClient
from starlette.middleware.sessions import SessionMiddleware

from kestrel_sovereign import server as server_module

API_KEY = "unit-test-key-2490"
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

    @app.get("/api/test/restart-authority-after-rotation")
    async def restart_authority_after_rotation(request: Request):
        from kestrel_sovereign.auth import caller_context_scope
        from kestrel_sovereign.features.restart_coordinator.authority import (
            RestartAuthorityError,
            require_restart_request_authority,
        )

        os.environ["KESTREL_API_KEY"] = "rotated-after-authentication"
        try:
            with caller_context_scope(request.state.caller):
                require_restart_request_authority()
        except RestartAuthorityError as error:
            return {"accepted": False, "error": str(error)}
        return {"accepted": True}

    return app


@pytest.fixture()
def client(monkeypatch):
    monkeypatch.setenv("KESTREL_API_KEY", API_KEY)
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


def test_temporary_sovereign_key_provenance_survives_a_child_process(monkeypatch):
    """A host child must not reinterpret its inherited bootstrap key as stable."""

    monkeypatch.delenv("KESTREL_API_KEY", raising=False)
    temporary = server_module.get_api_key()
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from kestrel_sovereign.security.sovereign_key import "
                "is_ephemeral_sovereign_key; "
                "import os; "
                "print(is_ephemeral_sovereign_key(os.environ['KESTREL_API_KEY']))"
            ),
        ],
        check=True,
        capture_output=True,
        text=True,
        env=dict(os.environ),
    )

    assert temporary == os.environ["KESTREL_API_KEY"]
    assert proc.stdout.strip() == "True"


def test_quoted_empty_sovereign_key_is_replaced_by_ephemeral_bootstrap(monkeypatch):
    """Docker-style quotes must not turn an absent secret into empty authority."""

    from kestrel_sovereign.security.sovereign_key import (
        is_ephemeral_sovereign_key,
    )

    monkeypatch.setenv("KESTREL_API_KEY", '""')

    selected = server_module.get_api_key()

    assert selected
    assert selected == os.environ["KESTREL_API_KEY"]
    assert selected != '""'
    assert is_ephemeral_sovereign_key(selected)


def test_restart_authority_is_bound_to_the_key_authenticated_at_entry(client):
    response = client.get(
        "/api/test/restart-authority-after-rotation",
        headers={"X-API-Key": API_KEY},
    )

    assert response.status_code == 200
    assert response.json()["accepted"] is False
    assert "authenticated credential" in response.json()["error"]
