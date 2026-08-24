"""Unit-suite isolation from the host's real host-feature enablement (#3099).

A unit test that enters the server lifespan runs
``instantiate_host_features()``, which reads the host manifest from the
resolved project dir. On a developer machine that is the operator's own
``.kestrel-host-features.toml``; on a CI runner there is none. Both answer
the same way today, because a manifest naming no host-scoped entry enables
every discovered host feature — and so does no manifest at all.

That default fails *open*, in the direction of running more than intended:
the first time an operator disables a host feature in that manifest, the unit
suite starts it anyway, and a host feature installed later starts in tests
without anyone having written a line that says so.

The autouse fixture below points discovery at a manifest of this suite's own
that says ``default_enabled = false``. It deliberately names no slugs: a
seventh host feature added tomorrow is covered by the same file, unchanged —
which a hardcoded ``Claws = false, Eye = false, …`` list would not be.

A test that deliberately exercises manifest resolution, or that needs the
host features this machine actually has installed, opts out with
``@pytest.mark.owns_host_feature_manifest`` and names its own manifest.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from kestrel_sovereign.host_features import discovery

#: Marker name for tests that resolve the host manifest themselves.
OWNS_HOST_FEATURE_MANIFEST_MARKER = "owns_host_feature_manifest"

#: Namespaced container inside ``tmp_path``. Tests are free to create their
#: own ``tmp_path`` children, so the seeded manifest lives under a name no
#: test would pick.
ISOLATION_DIRNAME = "_kestrel_host_feature_isolation"


@pytest.fixture(autouse=True)
def _host_features_disabled_unless_named(request, tmp_path, monkeypatch):
    """Seed a disabling host manifest and make discovery read it.

    ``default_host_manifest_path`` is the single seam every reader resolves
    through (``read_host_manifest``, ``read_host_scoped_manifest``, and the
    ``instantiate_host_features()`` call the lifespan makes), so overriding it
    covers the whole surface rather than one call site.

    Returns the seeded manifest path for tests that want to assert on it.
    """
    if request.node.get_closest_marker(OWNS_HOST_FEATURE_MANIFEST_MARKER):
        return None

    manifest_dir = tmp_path / ISOLATION_DIRNAME
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest: Path = manifest_dir / discovery.HOST_MANIFEST_FILENAME
    manifest.write_text(
        f"[{discovery.HOST_SCOPE_TABLE}]\n"
        f"{discovery.HOST_SCOPE_DEFAULT_KEY} = false\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        discovery, "default_host_manifest_path", lambda: manifest
    )
    return manifest
