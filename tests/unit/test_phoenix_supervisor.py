"""Tests for the host-supervised Phoenix subprocess + reverse proxy (#2570).

No real Phoenix is needed: the subprocess is stubbed and the reverse proxy is
driven through an ``httpx.MockTransport`` upstream.
"""

from contextlib import asynccontextmanager

import httpx
import pytest
from fastapi.testclient import TestClient

from kestrel_sovereign import phoenix_supervisor as ps


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


class _StubPopen:
    """Minimal stand-in for ``subprocess.Popen`` — no real process."""

    instances: list = []

    def __init__(self, argv, **kwargs):
        self.argv = argv
        self.kwargs = kwargs
        self.pid = 4321
        self._alive = True
        self.signals: list = []
        _StubPopen.instances.append(self)

    def poll(self):
        return None if self._alive else 0

    def send_signal(self, sig):
        self.signals.append(sig)
        self._alive = False

    def terminate(self):
        self._alive = False

    def kill(self):
        self._alive = False

    def wait(self, timeout=None):
        self._alive = False
        return 0


def _running_supervisor(tmp_path, *, root_path="/phoenix", transport=None):
    """A supervisor that reports running, wired to a mock upstream if given."""
    sup = ps.PhoenixSupervisor(
        working_dir=tmp_path / "phoenix",
        port=6006,
        grpc_port=4317,
        root_path=root_path,
    )
    sup.process = _StubPopen(["stub"])
    if transport is not None:
        sup._client = httpx.AsyncClient(transport=transport, follow_redirects=False)
    return sup


# ---------------------------------------------------------------------------
# Config resolution
# ---------------------------------------------------------------------------


def test_defaults(monkeypatch):
    monkeypatch.delenv("KESTREL_PHOENIX_PORT", raising=False)
    monkeypatch.delenv("KESTREL_PHOENIX_GRPC_PORT", raising=False)
    assert ps.phoenix_port() == 6006
    assert ps.phoenix_grpc_port() == 4317
    # OTLP endpoint = the HTTP port (exporters are otlp-proto-http), NOT gRPC.
    assert ps.phoenix_otlp_endpoint() == "http://127.0.0.1:6006"


def test_port_overrides(monkeypatch):
    monkeypatch.setenv("KESTREL_PHOENIX_PORT", "7007")
    monkeypatch.setenv("KESTREL_PHOENIX_GRPC_PORT", "5555")
    assert ps.phoenix_port() == 7007
    assert ps.phoenix_grpc_port() == 5555
    assert ps.phoenix_otlp_endpoint() == "http://127.0.0.1:7007"


def test_enabled_requires_installed(monkeypatch):
    # Not importable in CI → disabled regardless of the flag.
    monkeypatch.setattr(ps, "phoenix_available", lambda: False)
    monkeypatch.delenv("KESTREL_PHOENIX_ENABLED", raising=False)
    assert ps.phoenix_enabled() is False


def test_enabled_when_installed(monkeypatch):
    monkeypatch.setattr(ps, "phoenix_available", lambda: True)
    monkeypatch.delenv("KESTREL_PHOENIX_ENABLED", raising=False)
    assert ps.phoenix_enabled() is True


def test_opt_out_flag(monkeypatch):
    monkeypatch.setattr(ps, "phoenix_available", lambda: True)
    monkeypatch.setenv("KESTREL_PHOENIX_ENABLED", "0")
    assert ps.phoenix_enabled() is False


def test_supervision_suppressed_under_pytest(monkeypatch):
    # Installed + not opted out, but we ARE under pytest → the lifespan must
    # NOT spawn a real Phoenix subprocess (issue #2570: no real Phoenix in CI).
    monkeypatch.setattr(ps, "phoenix_enabled", lambda: True)
    assert ps._running_under_pytest() is True
    assert ps.should_supervise_phoenix() is False


def test_supervision_active_outside_pytest(monkeypatch):
    # Simulate a real host process (not under pytest) with Phoenix installed.
    monkeypatch.setattr(ps, "phoenix_enabled", lambda: True)
    monkeypatch.setattr(ps, "_running_under_pytest", lambda: False)
    assert ps.should_supervise_phoenix() is True


def test_supervision_off_when_disabled_outside_pytest(monkeypatch):
    # Not under pytest, but Phoenix not enabled → still no supervision.
    monkeypatch.setattr(ps, "phoenix_enabled", lambda: False)
    monkeypatch.setattr(ps, "_running_under_pytest", lambda: False)
    assert ps.should_supervise_phoenix() is False


# ---------------------------------------------------------------------------
# OTLP autowiring (INV-SOLO)
# ---------------------------------------------------------------------------


def test_autowire_sets_endpoint_when_unset(monkeypatch):
    monkeypatch.delenv("KESTREL_PHOENIX_PORT", raising=False)
    env = {}
    result = ps.autowire_otlp_endpoint(env)
    assert result == "http://127.0.0.1:6006"
    assert env["OTEL_EXPORTER_OTLP_ENDPOINT"] == "http://127.0.0.1:6006"


def test_autowire_respects_operator_value():
    env = {"OTEL_EXPORTER_OTLP_ENDPOINT": "http://collector:4317"}
    assert ps.autowire_otlp_endpoint(env) is None
    assert env["OTEL_EXPORTER_OTLP_ENDPOINT"] == "http://collector:4317"


# ---------------------------------------------------------------------------
# build_env
# ---------------------------------------------------------------------------


def test_build_env_root_path(tmp_path):
    sup = ps.PhoenixSupervisor(
        working_dir=tmp_path, port=6006, grpc_port=4317, root_path="/phoenix"
    )
    env = sup.build_env(base_env={})
    assert env["PHOENIX_HOST"] == "127.0.0.1"
    assert env["PHOENIX_PORT"] == "6006"
    assert env["PHOENIX_GRPC_PORT"] == "4317"
    assert env["PHOENIX_WORKING_DIR"] == str(tmp_path)
    assert env["PHOENIX_HOST_ROOT_PATH"] == "/phoenix"
    assert env["PHOENIX_SQL_DATABASE_URL"].startswith("sqlite:///")


def test_build_env_fallback_drops_root_path(tmp_path):
    sup = ps.PhoenixSupervisor(working_dir=tmp_path, root_path="")
    env = sup.build_env(base_env={"PHOENIX_HOST_ROOT_PATH": "/stale"})
    assert "PHOENIX_HOST_ROOT_PATH" not in env


# ---------------------------------------------------------------------------
# Subprocess lifecycle (stubbed process)
# ---------------------------------------------------------------------------


def test_start_and_stop_lifecycle(tmp_path, monkeypatch):
    _StubPopen.instances.clear()
    monkeypatch.setattr(ps, "phoenix_enabled", lambda: True)
    monkeypatch.setattr(ps, "supports_host_root_path", lambda: True)
    monkeypatch.setattr(ps.subprocess, "Popen", _StubPopen)

    sup = ps.PhoenixSupervisor(working_dir=tmp_path / "phx", port=6006)
    # A pinned root_path was NOT given → resolved from supports_host_root_path.
    sup._root_path_override = None

    assert sup.start(wait_for_health=False) is True
    assert sup.running is True
    assert sup.root_path == "/phoenix"
    # Launched with the phoenix serve argv.
    assert _StubPopen.instances[-1].argv == sup.build_command()
    # PID file tracked on disk.
    assert sup.pid_file.exists()
    assert sup.pid_file.read_text() == "4321"

    sup.stop()
    assert sup.running is False
    assert not sup.pid_file.exists()


def test_start_falls_back_when_root_path_unsupported(tmp_path, monkeypatch):
    _StubPopen.instances.clear()
    monkeypatch.setattr(ps, "phoenix_enabled", lambda: True)
    monkeypatch.setattr(ps, "supports_host_root_path", lambda: False)
    monkeypatch.setattr(ps.subprocess, "Popen", _StubPopen)

    sup = ps.PhoenixSupervisor(working_dir=tmp_path / "phx")
    sup._root_path_override = None
    assert sup.start(wait_for_health=False) is True
    assert sup.root_path == ""


def test_start_skips_when_disabled(tmp_path, monkeypatch):
    monkeypatch.setattr(ps, "phoenix_enabled", lambda: False)
    called = {"popen": False}

    def _boom(*a, **k):
        called["popen"] = True

    monkeypatch.setattr(ps.subprocess, "Popen", _boom)

    sup = ps.PhoenixSupervisor(working_dir=tmp_path / "phx")
    assert sup.start(wait_for_health=False) is False
    assert sup.running is False
    assert called["popen"] is False


# ---------------------------------------------------------------------------
# upstream_url
# ---------------------------------------------------------------------------


def test_upstream_url_with_root_path(tmp_path):
    sup = _running_supervisor(tmp_path, root_path="/phoenix")
    assert (
        sup.upstream_url("v1/traces", "a=1")
        == "http://127.0.0.1:6006/phoenix/v1/traces?a=1"
    )
    assert sup.upstream_url("") == "http://127.0.0.1:6006/phoenix/"


def test_upstream_url_fallback(tmp_path):
    sup = _running_supervisor(tmp_path, root_path="")
    assert sup.upstream_url("foo") == "http://127.0.0.1:6006/foo"


# ---------------------------------------------------------------------------
# Embed cookie
# ---------------------------------------------------------------------------


def test_embed_cookie_roundtrip():
    secret = "super-secret"
    from starlette.requests import Request

    token = ps.mint_embed_token(secret, identity="jason")
    scope = {
        "type": "http",
        "headers": [(b"cookie", f"{ps.EMBED_COOKIE_NAME}={token}".encode())],
    }
    req = Request(scope)
    assert ps.verify_embed_cookie(req, secret) is True


def test_embed_cookie_rejects_wrong_secret():
    from starlette.requests import Request

    token = ps.mint_embed_token("secret-a")
    scope = {
        "type": "http",
        "headers": [(b"cookie", f"{ps.EMBED_COOKIE_NAME}={token}".encode())],
    }
    req = Request(scope)
    assert ps.verify_embed_cookie(req, "secret-b") is False


def test_embed_cookie_absent():
    from starlette.requests import Request

    req = Request({"type": "http", "headers": []})
    assert ps.verify_embed_cookie(req, "s") is False


# ---------------------------------------------------------------------------
# Reverse proxy unit (direct call)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_proxy_503_when_disabled():
    from starlette.requests import Request

    async def receive():
        return {"type": "http.request", "body": b""}

    scope = {
        "type": "http",
        "method": "GET",
        "path": "/phoenix/",
        "query_string": b"",
        "headers": [],
    }
    req = Request(scope, receive)
    resp = await ps.proxy_to_phoenix(req, None, "")
    assert resp.status_code == 503


@pytest.mark.asyncio
async def test_proxy_streams_upstream(tmp_path):
    from starlette.requests import Request

    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)

        async def body():
            yield b"<html>phoenix</html>"

        return httpx.Response(200, content=body())

    sup = _running_supervisor(
        tmp_path, root_path="/phoenix", transport=httpx.MockTransport(handler)
    )

    async def receive():
        return {"type": "http.request", "body": b""}

    scope = {
        "type": "http",
        "method": "GET",
        "path": "/phoenix/",
        "query_string": b"",
        "headers": [],
    }
    req = Request(scope, receive)
    resp = await ps.proxy_to_phoenix(req, sup, "")

    body = b""
    async for chunk in resp.body_iterator:
        body += chunk if isinstance(chunk, bytes) else chunk.encode()
    assert resp.status_code == 200
    assert b"phoenix" in body
    assert captured["url"] == "http://127.0.0.1:6006/phoenix/"
    await sup.aclose()


# ---------------------------------------------------------------------------
# Route-level: auth + 503 + embed cookie flow (through the real ASGI stack)
# ---------------------------------------------------------------------------


@asynccontextmanager
async def _noop_lifespan(_app):
    yield


def _client_with_state(phoenix_state, monkeypatch):
    """Return the shared host ``app`` wired for a route test.

    Uses ``monkeypatch`` so every mutation (the no-op lifespan, the app.state
    fields) is restored after the test. ``app`` is a process-wide singleton
    (``from server import app``); permanently swapping its lifespan_context here
    would leak a no-op lifespan into every later ``TestClient(app)`` in the same
    pytest session.
    """
    from server import app

    monkeypatch.setattr(app.router, "lifespan_context", _noop_lifespan)
    monkeypatch.setattr(app.state, "agent", None, raising=False)
    monkeypatch.setattr(app.state, "agent_manager", None, raising=False)
    monkeypatch.setattr(app.state, "startup_error", None, raising=False)
    monkeypatch.setattr(app.state, "phoenix", phoenix_state, raising=False)
    return app


def test_phoenix_route_requires_auth(monkeypatch):
    monkeypatch.setenv("KESTREL_API_KEY", "test-key-123")
    app = _client_with_state(None, monkeypatch)
    with TestClient(app) as client:
        # No auth at all → 401 (never leaks whether Phoenix is up).
        r = client.get("/phoenix/")
    assert r.status_code == 401


def test_phoenix_route_503_when_disabled(monkeypatch):
    monkeypatch.setenv("KESTREL_API_KEY", "test-key-123")
    app = _client_with_state(None, monkeypatch)
    with TestClient(app) as client:
        r = client.get("/phoenix/", headers={"X-API-Key": "test-key-123"})
    assert r.status_code == 503
    assert r.json()["detail"] == "Phoenix not enabled"


def test_mint_endpoint_requires_auth(monkeypatch):
    monkeypatch.setenv("KESTREL_API_KEY", "test-key-123")
    app = _client_with_state(None, monkeypatch)
    with TestClient(app) as client:
        r = client.post("/api/host/phoenix/session")
    assert r.status_code == 401


def test_real_lifespan_never_spawns_phoenix_under_pytest(monkeypatch):
    """End-to-end: the *real* host lifespan must not spawn a Phoenix subprocess
    while running under pytest, even when arize-phoenix looks installed.

    This is the guard that keeps ``pytest -q`` from launching (and then tearing
    down, SIGTERM→SIGKILL) a heavyweight Phoenix per ``TestClient`` startup.
    """
    monkeypatch.setenv("KESTREL_API_KEY", "test-key-123")
    # Make Phoenix look installed so only the pytest guard can stop a spawn.
    monkeypatch.setattr(ps, "phoenix_available", lambda: True)
    assert ps.should_supervise_phoenix() is False  # suppressed: we're in pytest

    # Trip on ANY subprocess spawn from the supervisor.
    def _tripwire(*_a, **_k):
        raise AssertionError("Phoenix subprocess spawned under pytest")

    monkeypatch.setattr(ps.subprocess, "Popen", _tripwire)

    from server import app

    # Use the REAL lifespan (do not swap in the no-op) so the guarded startup
    # path actually runs; just neutralise app.state afterwards via monkeypatch.
    with TestClient(app) as client:
        r = client.get("/phoenix/", headers={"X-API-Key": "test-key-123"})
    # Phoenix was never supervised → the proxy reports it disabled.
    assert getattr(app.state, "phoenix", None) is None
    assert r.status_code == 503


def test_mint_then_proxy_with_embed_cookie(tmp_path, monkeypatch):
    monkeypatch.setenv("KESTREL_API_KEY", "test-key-123")

    def handler(request: httpx.Request) -> httpx.Response:
        async def body():
            yield b"<html>phoenix-ui</html>"

        return httpx.Response(200, content=body())

    sup = _running_supervisor(
        tmp_path, root_path="/phoenix", transport=httpx.MockTransport(handler)
    )
    app = _client_with_state(sup, monkeypatch)

    with TestClient(app) as client:
        # Mint requires standard auth (API key here).
        minted = client.post(
            "/api/host/phoenix/session", headers={"X-API-Key": "test-key-123"}
        )
        assert minted.status_code == 200
        assert ps.EMBED_COOKIE_NAME in minted.cookies

        # Now the browser carries the embed cookie; NO api key header on the
        # iframe request. It must still be authorized (and proxied).
        client.headers.pop("X-API-Key", None)
        r = client.get("/phoenix/")
        assert r.status_code == 200
        assert b"phoenix-ui" in r.content
