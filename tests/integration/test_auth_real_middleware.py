"""Real-middleware-stack auth status-correctness tests (#2490).

Exercises the actual ``kestrel_sovereign.server`` application — the full
registered middleware stack (query redaction, deprecation shim, CORS,
session, auth, agent routing) — against temporary routes to prove the
invariant: authentication failures alone produce 401; once authentication
succeeds, downstream failures keep FastAPI's own status semantics (generic
sanitized 500 for unhandled exceptions, controlled ``HTTPException``
statuses intact) and are never translated into 401.
"""

from __future__ import annotations

import contextlib
import socket
import threading
import time
from contextlib import asynccontextmanager

import httpx
import pytest
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.testclient import TestClient
from starlette.middleware.sessions import SessionMiddleware

from kestrel_sovereign import server as server_module

API_KEY = "integration-key-2490"
SENTINEL = "sentinel downstream failure"

OK_PATH = "/api/_test_2490/ok"
FAILING_PATH = "/api/_test_2490/boom"
CONTROLLED_PATH = "/api/_test_2490/teapot"
LOGIN_PATH = "/api/_test_2490/login"

AUTH_LANES = ["header", "bearer", "session"]


@pytest.fixture()
def real_app(monkeypatch):
    """The real server app with a noop lifespan and temporary test routes."""
    monkeypatch.setenv("KESTREL_API_KEY", API_KEY)
    app = server_module.app

    @asynccontextmanager
    async def noop_lifespan(_app):
        yield

    original_lifespan = app.router.lifespan_context
    app.router.lifespan_context = noop_lifespan
    route_count = len(app.router.routes)

    @app.get(OK_PATH)
    async def _ok():
        return {"ok": True}

    @app.get(FAILING_PATH)
    async def _boom():
        raise RuntimeError(SENTINEL)

    @app.get(CONTROLLED_PATH)
    async def _teapot():
        raise HTTPException(status_code=418, detail="short and stout")

    @app.get(LOGIN_PATH)
    async def _login(request: Request):
        request.session["user_email"] = "operator@example.com"
        return {"ok": True}

    try:
        yield app
    finally:
        del app.router.routes[route_count:]
        app.router.lifespan_context = original_lifespan


@pytest.fixture()
def client(real_app):
    with TestClient(real_app, raise_server_exceptions=False) as test_client:
        yield test_client


def _lane_kwargs(client, lane, key=API_KEY):
    """Request kwargs for an authentication lane (logs in for the session lane)."""
    if lane == "header":
        return {"headers": {"X-API-Key": key}}
    if lane == "bearer":
        return {"headers": {"Authorization": f"Bearer {key}"}}
    if lane == "session":
        login = client.get(LOGIN_PATH, headers={"X-API-Key": API_KEY})
        assert login.status_code == 200
        return {}
    raise AssertionError(f"unknown lane {lane}")


@pytest.mark.parametrize("lane", AUTH_LANES)
def test_valid_credentials_reach_route(client, lane):
    response = client.get(OK_PATH, **_lane_kwargs(client, lane))
    assert response.status_code == 200
    assert response.json() == {"ok": True}


@pytest.mark.parametrize("lane", ["header", "bearer"])
def test_invalid_credentials_are_401(client, lane):
    response = client.get(OK_PATH, **_lane_kwargs(client, lane, key="wrong-key-2490"))
    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid or missing API Key"
    assert response.json()["error"]["code"] == "authentication_required"
    assert response.json()["error"]["correlation_id"] == response.headers[
        "X-Correlation-ID"
    ]


def test_missing_credentials_are_401(client):
    correlation_id = "auth-integration-2651"
    response = client.get(OK_PATH, headers={"X-Correlation-ID": correlation_id})
    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid or missing API Key"
    assert response.json()["error"] == {
        "code": "authentication_required",
        "message": "Invalid or missing API Key",
        "correlation_id": correlation_id,
    }
    assert response.headers["X-Correlation-ID"] == correlation_id


def test_cors_exposes_auth_error_correlation_id(client):
    response = client.get(
        OK_PATH,
        headers={"Origin": "http://localhost:3000"},
    )

    assert response.status_code == 401
    assert response.headers["Access-Control-Allow-Origin"] == "http://localhost:3000"
    exposed = {
        value.strip().lower()
        for value in response.headers["Access-Control-Expose-Headers"].split(",")
    }
    assert "x-correlation-id" in exposed
    assert response.json()["error"]["correlation_id"] == response.headers[
        "X-Correlation-ID"
    ]


@pytest.mark.parametrize("lane", AUTH_LANES)
def test_authenticated_downstream_failure_is_generic_500_not_401(client, lane):
    response = client.get(FAILING_PATH, **_lane_kwargs(client, lane))
    assert response.status_code == 500
    # Canonical generic body — no auth translation, traceback, or secrets.
    assert response.json()["error"]["code"] == "internal_error"
    assert response.json()["detail"] == "An internal error occurred."
    assert response.json()["error"]["correlation_id"] == response.headers[
        "X-Correlation-ID"
    ]
    assert SENTINEL not in response.text
    assert API_KEY not in response.text
    assert "Authentication failed" not in response.text


@pytest.mark.parametrize("lane", AUTH_LANES)
def test_authenticated_downstream_http_exception_status_is_preserved(client, lane):
    response = client.get(CONTROLLED_PATH, **_lane_kwargs(client, lane))
    assert response.status_code == 418
    assert response.json()["detail"] == "short and stout"


def _free_port() -> int:
    with contextlib.closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def test_live_server_auth_status_trio(monkeypatch):
    """Real-Uvicorn trio: invalid → 401, valid+ok → 200, valid+fault → sanitized 500."""
    monkeypatch.setenv("KESTREL_API_KEY", "live-secret-2490")
    app = FastAPI()
    # Canonical Kestrel auth middleware, same session layering as the real app.
    app.middleware("http")(server_module.auth_middleware)
    app.add_middleware(
        SessionMiddleware,
        secret_key="test-session-secret",
        session_cookie="kestrel_session",
    )

    @app.get("/api/ok")
    async def _ok():
        return {"ok": True}

    @app.get("/api/boom")
    async def _boom():
        raise RuntimeError(SENTINEL)

    port = _free_port()
    config = uvicorn.Config(
        app,
        host="127.0.0.1",
        port=port,
        log_config=None,
        access_log=False,
    )
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    try:
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                    break
            except OSError:
                time.sleep(0.05)
        else:
            raise RuntimeError(f"uvicorn failed to bind on port {port}")

        with httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=5.0) as client:
            unauthenticated = client.get("/api/ok")
            assert unauthenticated.status_code == 401

            wrong_key = client.get("/api/ok", headers={"X-API-Key": "wrong-key"})
            assert wrong_key.status_code == 401
            assert wrong_key.json()["detail"] == "Invalid or missing API Key"

            ok = client.get("/api/ok", headers={"X-API-Key": "live-secret-2490"})
            assert ok.status_code == 200
            assert ok.json() == {"ok": True}

            fault = client.get("/api/boom", headers={"X-API-Key": "live-secret-2490"})
            assert fault.status_code == 500
            assert fault.text == "Internal Server Error"
            assert SENTINEL not in fault.text
            assert "live-secret-2490" not in fault.text
    finally:
        server.should_exit = True
        thread.join(timeout=5.0)
