"""Runtime-enable serves a disabled feature's UI asset (#2048).

Regression for the runtime-enable 404: a feature that starts DISABLED but
declares ``get_ui_contributions()`` had its ``/features/{name}/static/`` dir
mounted only when enabled at startup. Enabling it from the Feature Store later
surfaced its contribution in ``/api/ui/contributions`` while its ``static_dir``
was never mounted, so the dynamic ``import()`` 404'd and the tab never appeared
until a restart.

This boots the REAL ``_mount_feature_ui_assets`` from ``kestrel_sovereign.server``
against an agent holding a DISABLED feature and asserts the asset is already
served — so flipping ``feature.enabled`` (which the manifest gates on) needs no
restart for the JS to load.
"""

from fastapi import FastAPI
from fastapi.testclient import TestClient

from kestrel_sovereign.features.base import UIContributions


class _FakeFeature:
    def __init__(self, contrib, enabled):
        self._contrib = contrib
        self.enabled = enabled

    def get_ui_contributions(self):
        return self._contrib


class _FakeAgent:
    def __init__(self, features):
        self.features = features


def test_disabled_feature_asset_served_so_runtime_enable_works(tmp_path, monkeypatch):
    import kestrel_sovereign.server as server

    monkeypatch.setattr(server, "SERVE_UI", True, raising=False)

    static_dir = tmp_path / "spawn_static"
    static_dir.mkdir()
    (static_dir / "spawn.js").write_text("export const x = 1;\n")

    contrib = UIContributions(
        modules=["spawn.js"], static_dir=str(static_dir), capability="spawn"
    )
    # The feature is present but DISABLED at startup.
    feature = _FakeFeature(contrib, enabled=False)
    agent = _FakeAgent({"SpawnFeature": feature})

    app = FastAPI()
    app.state.agent = agent
    app.state.agent_manager = None

    server._mount_feature_ui_assets(app)
    client = TestClient(app)

    # The asset is served even though the feature is disabled — so the moment it
    # is enabled (manifest gating only), the import() resolves with no restart.
    resp = client.get("/features/spawnfeature/static/spawn.js")
    assert resp.status_code == 200
    assert "export const x" in resp.text
