"""Integration test: out-of-tree feature UI assets bypass auth (#2043).

A pip-installed feature ships its frontend JS/CSS, mounted by the server at
``/features/{name}/static/``. The browser loads those assets with raw
mechanisms — ``<link href=...>`` and ``await import(mod)`` — which cannot
attach the ``X-API-Key`` header used by ``API.request()``. So the global
``auth_middleware`` must exempt the feature static mounts, exactly as it
already exempts the core ``/static`` and ``/js/`` trees, or the reference
feature 401s in API-key/local mode and never loads.

This boots a minimal app wired with the REAL ``auth_middleware`` from
``kestrel_sovereign.server`` (not a re-implementation) and a feature static
mount, then verifies the asset is reachable with no auth headers while a
protected API route still requires the key.
"""

import os
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.testclient import TestClient
from starlette.middleware.sessions import SessionMiddleware

API_KEY = "test-api-key-for-feature-ui-assets"


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("KESTREL_API_KEY", API_KEY)

    # Force SERVE_UI true on the imported module (it is read at import time).
    import kestrel_sovereign.server as server

    monkeypatch.setattr(server, "SERVE_UI", True, raising=False)

    # A feature's static asset dir mounted exactly like _mount_feature_ui_assets.
    asset_dir = tmp_path / "example_static"
    asset_dir.mkdir()
    (asset_dir / "ui.js").write_text("export function init() {}\n")
    (asset_dir / "ui.css").write_text(".kestrel-example { color: rebeccapurple; }\n")

    app = FastAPI()
    app.middleware("http")(server.auth_middleware)
    app.add_middleware(
        SessionMiddleware,
        secret_key="test-session-secret",
        session_cookie="kestrel_session",
    )

    app.mount(
        "/features/example/static",
        StaticFiles(directory=str(asset_dir)),
        name="feature-ui:/features/example/static",
    )

    @app.get("/api/conversations")
    def conversations():
        return {"conversations": []}

    # A non-static route under /features/ must remain protected — the bypass is
    # narrow to /features/{slug}/static/, not all of /features/.
    @app.get("/features/example/api/secret")
    def feature_api():
        return {"secret": "value"}

    return TestClient(app)


def test_feature_static_js_loads_without_api_key(client):
    """ES module import has no header path; the mount must be public like /js/."""
    resp = client.get("/features/example/static/ui.js")
    assert resp.status_code == 200
    assert "export function init" in resp.text


def test_feature_static_css_loads_without_api_key(client):
    """`<link href=...>` likewise cannot attach the API key."""
    resp = client.get("/features/example/static/ui.css")
    assert resp.status_code == 200
    assert "kestrel-example" in resp.text


def test_protected_api_still_requires_key(client):
    """The static bypass must not weaken auth on real API routes."""
    assert client.get("/api/conversations").status_code == 401
    assert (
        client.get("/api/conversations", headers={"X-API-Key": API_KEY}).status_code
        == 200
    )


def test_feature_non_static_route_still_requires_key(client):
    """Bypass is scoped to /features/{slug}/static/, not all /features/ paths."""
    assert client.get("/features/example/api/secret").status_code == 401
