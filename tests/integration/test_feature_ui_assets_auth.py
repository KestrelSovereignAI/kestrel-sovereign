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


# Parametrized over BOTH SERVE_UI states. The feature static mounts are installed
# unconditionally by _mount_feature_ui_assets(), so a host-managed agent (which
# runs with KESTREL_SERVE_UI=false) still serves them — and the browser reaches
# them header-less via the host's /api/agents/{id}/... proxy, which forwards no
# key. So the exemption must hold when SERVE_UI is False too; gating it on
# SERVE_UI (as the core /static exemption is) 401s the import() on every host
# agent (#2043 regression: the original test pinned SERVE_UI=True and missed it).
@pytest.fixture(params=[True, False], ids=["serve_ui", "host_agent_no_serve_ui"])
def client(tmp_path, monkeypatch, request):
    monkeypatch.setenv("KESTREL_API_KEY", API_KEY)

    # SERVE_UI is read at import time; override on the module for both modes.
    import kestrel_sovereign.server as server

    monkeypatch.setattr(server, "SERVE_UI", request.param, raising=False)

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

    # A feature API route whose path merely CONTAINS a later "static" segment must
    # still be protected — the exemption is anchored to /features/{slug}/static/,
    # not a substring match on "/static/".
    @app.get("/features/example/api/static/secret")
    def feature_api_static_lookalike():
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


def test_feature_api_path_with_static_segment_still_requires_key(client):
    """Exemption is anchored to /features/{slug}/static/, not a "/static/" substring.

    A feature API route like /features/foo/api/static/secret must NOT be
    unauthenticated just because a later path segment is literally "static".
    """
    assert client.get("/features/example/api/static/secret").status_code == 401


# --- Live host-agent path: real mount + real exemption, both under SERVE_UI=false --

class _FakeFeature:
    def __init__(self, contrib):
        self._contrib = contrib
        self.enabled = True

    def get_ui_contributions(self):
        return self._contrib


class _FakeAgent:
    def __init__(self, features):
        self.features = features


def test_host_agent_serves_feature_asset_without_key_end_to_end(tmp_path, monkeypatch):
    """The exact host-managed-agent scenario, exercising the REAL mount path.

    A host-managed agent runs with KESTREL_SERVE_UI=false. Both the asset MOUNT
    (_mount_feature_ui_assets) and the auth EXEMPTION (auth_middleware) must hold
    in that mode, or the browser's header-less import()  — proxied by the host to
    /features/{slug}/static/... — 404s (mount skipped) or 401s (auth). This test
    uses the real _mount_feature_ui_assets rather than a hand-mounted StaticFiles
    dir, so a regression in the mount gate is caught, not masked (#2048).
    """
    from kestrel_sovereign.features.base import UIContributions
    import kestrel_sovereign.server as server

    monkeypatch.setenv("KESTREL_API_KEY", API_KEY)
    monkeypatch.setattr(server, "SERVE_UI", False, raising=False)

    static_dir = tmp_path / "spawn_static"
    static_dir.mkdir()
    (static_dir / "spawn.js").write_text("export const x = 1;\n")

    app = FastAPI()
    app.middleware("http")(server.auth_middleware)
    app.add_middleware(
        SessionMiddleware,
        secret_key="test-session-secret",
        session_cookie="kestrel_session",
    )
    app.state.agent = _FakeAgent(
        {
            "SpawnFeature": _FakeFeature(
                UIContributions(
                    modules=["spawn.js"],
                    static_dir=str(static_dir),
                    capability="spawn",
                )
            )
        }
    )

    # The real server mount routine — must mount even though SERVE_UI is false.
    server._mount_feature_ui_assets(app)

    client = TestClient(app)
    resp = client.get("/features/spawnfeature/static/spawn.js")
    assert resp.status_code == 200, resp.status_code
    assert "export const x" in resp.text
