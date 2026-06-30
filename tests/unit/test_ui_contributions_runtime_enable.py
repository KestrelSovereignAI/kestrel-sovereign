"""Runtime-enable asset serving for feature UI contributions (#2048).

When a feature that declares ``get_ui_contributions()`` starts DISABLED and is
later enabled from the Feature Store, its ``/features/{name}/static/`` assets
must already be mounted — otherwise the dynamic ``import()`` 404s and the tab
never appears until a restart. The server mounts every declared ``static_dir``
at startup (``include_disabled=True``) while the manifest still advertises only
ENABLED features, so a disabled feature's mount is dormant until it is enabled.
"""

from pathlib import Path

from kestrel_sovereign.features.base import UIContributions
from kestrel_sovereign.ui_contributions import (
    compute_ui_manifest,
    feature_static_mounts,
)


class _FakeFeature:
    def __init__(self, contrib, enabled):
        self._contrib = contrib
        self.enabled = enabled

    def get_ui_contributions(self):
        return self._contrib


class _FakeAgent:
    def __init__(self, features):
        self.features = features


def _contrib(static_dir):
    return UIContributions(
        modules=["spawn.js"], static_dir=str(static_dir), capability="spawn"
    )


def _static_dir(tmp_path):
    static_dir = tmp_path / "spawn_static"
    static_dir.mkdir()
    (static_dir / "spawn.js").write_text("export const x = 1;\n")
    return static_dir


def test_disabled_feature_static_dir_mounted_when_include_disabled(tmp_path):
    static_dir = _static_dir(tmp_path)
    agent = _FakeAgent(
        {"SpawnFeature": _FakeFeature(_contrib(static_dir), enabled=False)}
    )

    # Default (manifest semantics): a disabled feature contributes no mount.
    assert feature_static_mounts(agent) == []

    # Server path: include_disabled mounts the asset dir so a later runtime enable
    # serves /features/spawnfeature/static/spawn.js without a restart.
    mounts = feature_static_mounts(agent, include_disabled=True)
    assert len(mounts) == 1
    mount_path, directory = mounts[0]
    assert mount_path == "/features/spawnfeature/static"
    assert Path(directory).resolve() == static_dir.resolve()


def test_manifest_excludes_disabled_then_surfaces_on_enable(tmp_path):
    static_dir = _static_dir(tmp_path)
    feature = _FakeFeature(_contrib(static_dir), enabled=False)
    agent = _FakeAgent({"SpawnFeature": feature})

    # Disabled: asset mounted (above) but NOT advertised in the manifest.
    assert all(e["feature"] != "SpawnFeature" for e in compute_ui_manifest(agent))

    # Runtime enable: the already-mounted asset now appears in the manifest, so
    # the boot loader import()s it and the panel surfaces with no restart.
    feature.enabled = True
    entry = next(
        e for e in compute_ui_manifest(agent) if e["feature"] == "SpawnFeature"
    )
    assert entry["capability"] == "spawn"
    assert any(m.endswith("/spawn.js") for m in entry["modules"])
