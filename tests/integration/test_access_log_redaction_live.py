"""Real-Uvicorn access-log regression for SSE query credentials (#2429)."""

from __future__ import annotations

import contextlib
import io
import logging
import socket
import threading
import time
from types import SimpleNamespace

import httpx
import uvicorn
from fastapi import FastAPI, Request
from starlette.middleware.sessions import SessionMiddleware

from kestrel_sovereign.security.access_log import (
    SensitiveQueryStringRedactionMiddleware,
)
from kestrel_sovereign import server as server_module


def _free_port() -> int:
    with contextlib.closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def test_real_uvicorn_logs_redact_direct_and_multi_agent_sse_keys(monkeypatch):
    monkeypatch.setenv("KESTREL_API_KEY", "direct-secret-2429")
    app = FastAPI()
    observed_keys: list[str | None] = []
    observed_agents: list[object | None] = []

    routed_agent = SimpleNamespace(agent_id="did:test:kite")
    manager = SimpleNamespace(
        get_agent=lambda name: routed_agent if name.casefold() == "kite" else None,
    )
    app.state.agent_manager = manager

    # This is the real multi-agent ASGI router. It must sit inside auth and
    # query redaction, matching the main application's middleware order.
    app.add_middleware(server_module.MultiAgentAgentRoutingMiddleware)

    # Use the canonical Kestrel auth middleware rather than a stand-in. The
    # session/redaction ordering matches the real app's outer layers.
    app.middleware("http")(server_module.auth_middleware)
    app.add_middleware(
        SessionMiddleware,
        secret_key="test-session-secret",
        session_cookie="kestrel_session",
    )

    @app.get("/{path:path}")
    async def capture_query(request: Request, path: str):
        # Proves application/auth code still sees the real value.
        observed_keys.append(request.query_params.get("api_key"))
        observed_agents.append(getattr(request.state, "agent", None))
        return {
            "path": path,
            "authenticated": request.state.caller.is_sovereign,
        }

    app.add_middleware(SensitiveQueryStringRedactionMiddleware)

    access_logger = logging.getLogger("uvicorn.access")
    original_handlers = list(access_logger.handlers)
    original_level = access_logger.level
    original_propagate = access_logger.propagate
    output = io.StringIO()
    capture = logging.StreamHandler(output)
    capture.setFormatter(logging.Formatter("%(message)s"))
    access_logger.handlers = [capture]
    access_logger.setLevel(logging.INFO)
    access_logger.propagate = False

    port = _free_port()
    config = uvicorn.Config(
        app,
        host="127.0.0.1",
        port=port,
        log_config=None,
        access_log=True,
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

        paths_and_keys = [
            ("/api/agent/notifications/sse", "direct-secret-2429"),
            (
                "/api/agents/Kite/api/agent/notifications/sse",
                "direct-secret-2429",
            ),
        ]
        with httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=5.0) as client:
            for path, key in paths_and_keys:
                response = client.get(path, params={"api_key": key, "session_id": "s-1"})
                assert response.status_code == 200
                assert response.json()["authenticated"] is True

            # Actual Uvicorn must not log private metadata from artifact URLs,
            # whether the request succeeds or authentication refuses it.
            for artifact_path in (
                "/api/artifacts/publications/private-publication-key",
                "/api/agents/Kite/api/artifacts/private-artifact/revisions/private-revision/file",
            ):
                response = client.get(artifact_path, params={"path": "private-filename.md"},
                                      headers={"X-API-Key": "direct-secret-2429"})
                assert response.status_code == 200
                assert "private-" in response.json()["path"]
            assert client.get("/api/artifacts/publications/denied-private-key").status_code == 401

            for legacy_path in (
                "/api/workflows/runs/private-run/artifacts/private-artifact",
                "/api/agents/Kite/api/workflows/runs/private%3Frun/artifacts/private-artifact",
                "/%61pi/workflows/runs/private-run/%61rtifacts/private-artifact",
                "/api%2fworkflows%2fruns%2fprivate-run%2fartifacts%2fprivate-artifact",
            ):
                response = client.get(legacy_path, params={"signature": "private-signature"},
                                      headers={"X-API-Key": "direct-secret-2429"})
                assert response.status_code == 200
                assert "private-" in response.json()["path"]
            assert client.get("/api/workflows/runs/denied-private-run/artifacts/private-artifact").status_code == 401

        logs = output.getvalue()
        assert observed_keys[:2] == [key for _, key in paths_and_keys]
        assert observed_agents[:2] == [None, routed_agent]
        for _, key in paths_and_keys:
            assert key not in logs
        assert logs.count("api_key=redacted") == 2
        # The real multi-agent router strips the prefix before Uvicorn emits
        # the log record, proving the proxy-routing layer actually ran.
        assert logs.count("/api/agent/notifications/sse?") == 2
        assert "private-" not in logs
        assert logs.count('GET /api/artifacts HTTP/') == 3
        assert logs.count('GET /api/workflows/artifact-metadata HTTP/') == 5
    finally:
        server.should_exit = True
        thread.join(timeout=5.0)
        access_logger.handlers = original_handlers
        access_logger.setLevel(original_level)
        access_logger.propagate = original_propagate
