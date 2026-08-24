"""The unit suite starts no host features unless a test says so (#3099).

Every assertion here is about the autouse fixture in ``tests/unit/conftest.py``.
It exists because the widening it closes is invisible from inside the suite:
a missing manifest and a manifest naming no host-scoped entry both mean
"enable everything discovered", so on a machine where nothing is disabled the
isolated and un-isolated paths return the identical six host features. The
difference only appears the day an operator disables one — or the day a
seventh is installed.
"""

from __future__ import annotations

import pytest
from fastapi import APIRouter

from kestrel_sdk.features.host_base import HostFeature

from kestrel_sovereign import host_features as hf
from kestrel_sovereign.host_features import discovery

from tests.unit.conftest import ISOLATION_DIRNAME


class _HostFeatureAddedTomorrow(HostFeature):
    """Stands in for host feature number seven.

    The fixture must cover it without naming it — a seeded manifest that
    listed today's slugs explicitly would re-open the hole this class
    represents.
    """

    name = "host-feature-nobody-has-listed"

    def get_router(self) -> APIRouter:
        return APIRouter()


def test_discovery_reads_this_suites_own_manifest():
    resolved = discovery.default_host_manifest_path()
    assert ISOLATION_DIRNAME in resolved.parts
    assert resolved.name == discovery.HOST_MANIFEST_FILENAME


def test_the_seeded_manifest_names_no_host_feature():
    """It restricts by default, not by enumeration."""
    policy = hf.read_host_manifest()

    assert policy.default_enabled is False
    assert policy.enablement == {}


def test_a_host_feature_the_manifest_never_names_stays_off():
    assert (
        hf.instantiate_host_features(
            {"_HostFeatureAddedTomorrow": _HostFeatureAddedTomorrow}
        )
        == []
    )


def test_the_lifespan_call_starts_nothing_this_machine_has_installed():
    """The exact call the server lifespan makes: no classes, no manifest."""
    assert hf.instantiate_host_features() == []

    # And the emptiness is the manifest's doing, not an empty venv: feed
    # discovery's own answer back in, plus one class that certainly exists.
    installed = hf.discover_host_feature_classes()
    assert (
        hf.instantiate_host_features(
            {**installed, "_HostFeatureAddedTomorrow": _HostFeatureAddedTomorrow}
        )
        == []
    )


def test_a_test_can_still_name_the_enablement_it_wants(tmp_path):
    """The explicit opt-in: pass a manifest rather than inherit the suite's."""
    manifest = tmp_path / discovery.HOST_MANIFEST_FILENAME
    manifest.write_text(
        '[[feature]]\nname = "host-feature-nobody-has-listed"\n'
        "host_scoped = true\nenabled = true\n",
        encoding="utf-8",
    )

    features = hf.instantiate_host_features(
        {"_HostFeatureAddedTomorrow": _HostFeatureAddedTomorrow},
        manifest_path=manifest,
    )

    assert [f.name for f in features] == ["host-feature-nobody-has-listed"]


@pytest.mark.owns_host_feature_manifest
def test_the_opt_out_marker_actually_releases_the_override():
    """Otherwise a test could carry the marker and silently stay isolated.

    Deliberately does no filesystem I/O on the resolved path: this test proves
    the marker is wired up, and must not become the one unit test that reads
    the operator's real manifest.
    """
    assert ISOLATION_DIRNAME not in discovery.default_host_manifest_path().parts
