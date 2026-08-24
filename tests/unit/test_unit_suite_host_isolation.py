"""The unit suite cannot reach the operator's host-runtime state (#3087).

Every assertion here is about the autouse fixture in ``tests/unit/conftest.py``.
It exists because the leak it closes was invisible from inside the suite: the
default host-feature database is ``~/.kestrel/host-data/host-features.db``, CI
runs with a fresh ``HOME``, and so a developer machine was the only place where
``pytest`` migrated a live database — silently, as a side effect of any test
that entered ``TestClient(server.app)`` without overriding the lifespan.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from kestrel_sovereign import paths
from kestrel_sovereign.agent import token_counter
from kestrel_sovereign.host_features.context import build_host_context
from kestrel_sovereign.host_features.storage import (
    HOST_DB_PATH_ENV,
    host_database_path,
)

from tests.unit.conftest import ISOLATION_DIRNAME


def test_resolved_host_database_is_inside_this_test_tmp_path(tmp_path):
    """The path the lifespan would resolve, without opening anything."""
    resolved, uses_default = host_database_path()

    assert uses_default is False
    assert tmp_path in resolved.parents


def test_every_default_branch_behind_the_override_also_lands_in_tmp_path(
    tmp_path, monkeypatch,
):
    """``KESTREL_HOST_DB_PATH`` is not load-bearing on its own.

    Both fallbacks — ``$KESTREL_HOME/host-data`` and ``~/.kestrel/host-data`` —
    must be closed too, so a caller that ignores the override still cannot
    name the operator's data.
    """
    assert tmp_path in paths.host_data_dir().parents
    assert tmp_path in Path.home().parents
    assert tmp_path in paths.project_dir().parents

    monkeypatch.delenv(HOST_DB_PATH_ENV)
    paths.reset_cache()
    fallback, uses_default = host_database_path()

    assert uses_default is True
    assert tmp_path in fallback.parents


def test_discovered_context_limit_cache_is_redirected(tmp_path):
    """``token_counter`` freezes its cache path at import, so ``HOME`` misses it."""
    assert tmp_path in token_counter.CACHE_FILE.parents


@pytest.mark.asyncio
async def test_building_the_host_context_writes_only_under_tmp_path(tmp_path):
    """The exact call the server lifespan makes with no ``db_path``."""
    ctx = await build_host_context()
    try:
        assert ctx.db is not None, ctx.backend_error
        opened = Path(ctx.db.backend.db_path)
        assert tmp_path in opened.parents
        assert opened.exists()
    finally:
        if ctx.session_factory is not None:
            await ctx.session_factory.close()
        if ctx.db is not None:
            await ctx.db.close()


@pytest.mark.owns_host_paths
def test_the_opt_out_marker_actually_releases_the_override():
    """Otherwise the storage tests would silently stop testing the default.

    Deliberately does no filesystem I/O: this test proves the marker is wired
    up, and must not become the one unit test that touches a real host path.
    """
    assert ISOLATION_DIRNAME not in os.environ.get(HOST_DB_PATH_ENV, "")
    assert ISOLATION_DIRNAME not in os.environ.get("KESTREL_HOME", "")
