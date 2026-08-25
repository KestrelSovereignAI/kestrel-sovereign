"""The unit suite's function-based path resolution cannot reach the operator's
host-runtime state (#3087), and isolating it does not widen what runs (#3099).

Every assertion here is about the autouse fixture in ``tests/unit/conftest.py``.
It exists because the leak it closes was invisible from inside the suite: the
default host-feature database is ``~/.kestrel/host-data/host-features.db``, CI
runs with a fresh ``HOME``, and so a developer machine was the only place where
``pytest`` migrated a live database — silently, as a side effect of any test
that entered ``TestClient(server.app)`` without overriding the lifespan.

The second half of the file is the other direction. Redirecting ``KESTREL_HOME``
also hides the operator's ``.kestrel-host-features.toml``, and a missing
manifest means enable-all — so the isolation could start host features the
operator had disabled. The fixture seeds a manifest that disables by default;
these tests hold that seed to a policy rather than a list of today's slugs.

Scope matches ``tests/unit/conftest.py``: module-scope constants frozen at
import time are NOT covered here and are tracked in #3104.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from kestrel_sdk.features.host_base import HostFeature

from kestrel_sovereign import paths
from kestrel_sovereign.agent import token_counter
from kestrel_sovereign.host_features.context import build_host_context
from kestrel_sovereign.host_features.discovery import (
    HOST_MANIFEST_FILENAME,
    instantiate_host_features,
)
from kestrel_sovereign.host_features.storage import (
    HOST_DB_PATH_ENV,
    host_database_path,
)

from tests.unit.conftest import ISOLATION_DIRNAME


def test_resolved_host_database_is_inside_the_isolation_root(
    host_runtime_isolation_root,
):
    """The path the lifespan would resolve, without opening anything."""
    resolved, uses_default = host_database_path()

    assert uses_default is False
    assert host_runtime_isolation_root in resolved.parents


def test_every_default_branch_behind_the_override_also_lands_in_the_root(
    host_runtime_isolation_root, monkeypatch,
):
    """``KESTREL_HOST_DB_PATH`` is not load-bearing on its own.

    Both fallbacks — ``$KESTREL_HOME/host-data`` and ``~/.kestrel/host-data`` —
    must be closed too, so neither fallback branch resolves outside the
    isolation root.
    """
    assert host_runtime_isolation_root in paths.host_data_dir().parents
    assert host_runtime_isolation_root in Path.home().parents
    assert host_runtime_isolation_root in paths.project_dir().parents

    monkeypatch.delenv(HOST_DB_PATH_ENV)
    paths.reset_cache()
    fallback, uses_default = host_database_path()

    assert uses_default is True
    assert host_runtime_isolation_root in fallback.parents


def test_the_isolation_root_is_not_inside_the_test_tmp_path(
    host_runtime_isolation_root, tmp_path,
):
    """The fixture writes files, so it does not share the test's own directory.

    A test is entitled to assert that its ``tmp_path`` is empty — one already
    does — and the seeded host manifest would break that.
    """
    assert tmp_path not in host_runtime_isolation_root.parents
    assert list(tmp_path.iterdir()) == []


def test_discovered_context_limit_cache_is_redirected(host_runtime_isolation_root):
    """``token_counter`` freezes its cache path at import, so ``HOME`` misses it."""
    assert host_runtime_isolation_root in token_counter.CACHE_FILE.parents


@pytest.mark.asyncio
async def test_building_the_host_context_writes_only_under_the_root(
    host_runtime_isolation_root,
):
    """The exact call the server lifespan makes with no ``db_path``."""
    ctx = await build_host_context()
    try:
        assert ctx.db is not None, ctx.backend_error
        opened = Path(ctx.db.backend.db_path)
        assert host_runtime_isolation_root in opened.parents
        assert opened.exists()
    finally:
        if ctx.session_factory is not None:
            await ctx.session_factory.close()
        if ctx.db is not None:
            await ctx.db.close()


# ---------------------------------------------------------------------------
# Enablement: isolating the home must not widen what the suite starts (#3099)
# ---------------------------------------------------------------------------


class _StartsInUnitTests(HostFeature):
    """Stands in for whatever the entry-point group happens to hold."""

    name = "unit-suite-canary-host-feature"


class _AddedTomorrow(HostFeature):
    """The seventh host feature, which no seeded list could have named."""

    name = "unit-suite-host-feature-nobody-has-written-yet"


def _lifespan_manifest_path() -> Path:
    """Exactly what ``server.py`` passes to ``instantiate_host_features``."""
    return paths.project_dir() / HOST_MANIFEST_FILENAME


def test_the_suite_starts_no_host_features():
    """A test entering the lifespan runs the host scope empty."""
    assert _lifespan_manifest_path().is_file(), "the fixture seeds this manifest"

    features = instantiate_host_features(
        {"_StartsInUnitTests": _StartsInUnitTests},
        manifest_path=_lifespan_manifest_path(),
    )

    assert features == []


def test_a_host_feature_added_later_is_disabled_without_touching_the_fixture():
    """The seed is a default, not a list — the point of #3099.

    A fixture that wrote ``Claws = false, Eye = false, …`` would enable host
    feature number seven the day it appears, silently, which is the same shape
    as the hole being closed.
    """
    features = instantiate_host_features(
        {"_AddedTomorrow": _AddedTomorrow},
        manifest_path=_lifespan_manifest_path(),
    )

    assert features == []


@pytest.mark.owns_host_paths
def test_the_opt_out_hands_enablement_back_to_the_test(tmp_path, monkeypatch):
    """Opting out restores production resolution, not a second policy.

    The seeded manifest is gone, so the answer comes from whichever project dir
    resolves — and a test that owns that dir gets exactly the host features it
    declares. Deliberately does not assert against the operator's real
    manifest: that would make the suite's result depend on their machine.
    """
    assert ISOLATION_DIRNAME not in str(_lifespan_manifest_path())

    monkeypatch.setenv("KESTREL_HOME", str(tmp_path))
    paths.reset_cache()
    try:
        _lifespan_manifest_path().write_text(
            f'[[feature]]\nname = "{_StartsInUnitTests.name}"\n'
            f"host_scoped = true\nenabled = true\n",
            encoding="utf-8",
        )

        features = instantiate_host_features(
            {"_StartsInUnitTests": _StartsInUnitTests},
            manifest_path=_lifespan_manifest_path(),
        )
    finally:
        paths.reset_cache()

    assert [f.name for f in features] == [_StartsInUnitTests.name]


@pytest.mark.owns_host_paths
def test_the_opt_out_marker_actually_releases_the_override():
    """Otherwise the storage tests would silently stop testing the default.

    Deliberately does no filesystem I/O: this test proves the marker is wired
    up, and must not become the one unit test that touches a real host path.
    """
    assert ISOLATION_DIRNAME not in os.environ.get(HOST_DB_PATH_ENV, "")
    assert ISOLATION_DIRNAME not in os.environ.get("KESTREL_HOME", "")
