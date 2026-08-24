"""Unit-suite isolation from the operator's real host-runtime state (#3087).

Many tests here build ``TestClient(server.app)`` without overriding the
lifespan. That lifespan calls ``build_host_context()`` with no ``db_path``,
so the host-feature database resolves through the *production* precedence:
``$KESTREL_HOST_DB_PATH``, else ``$KESTREL_HOME/host-data``, else
``~/.kestrel/host-data``. On a developer machine that last branch is the
live fleet database — running the unit suite migrated its schema as a side
effect of ``pytest``, and the real host features named by the project's
``.kestrel-host-features.toml`` started (and recorded their start failures)
against it. CI never notices: the runner's ``HOME`` is fresh, so the same
code writes a throwaway file.

The autouse fixture below moves every one of those roots into the test's own
``tmp_path``, so a unit test cannot name the operator's data at all. Anything
narrower would leave the next test that boots a real lifespan free to do it
again.

Tests that deliberately exercise path resolution opt out with
``@pytest.mark.owns_host_paths`` and set the same variables themselves —
``tests/unit/test_host_feature_storage.py`` is the worked example.
"""

from __future__ import annotations

import pytest

from kestrel_sovereign import paths
from kestrel_sovereign.agent import token_counter
from kestrel_sovereign.host_features.storage import (
    HOST_DB_PATH_ENV,
    HOST_FEATURE_DB_FILENAME,
)

#: Marker name for tests that own host/home path resolution themselves.
OWNS_HOST_PATHS_MARKER = "owns_host_paths"

#: Namespaced container inside ``tmp_path``. Tests are free to create their
#: own ``tmp_path`` children (including ones called ``home``), so the
#: isolation roots live under a name no test would pick.
ISOLATION_DIRNAME = "_kestrel_host_runtime_isolation"


@pytest.fixture(autouse=True)
def _isolate_host_runtime_paths(request, tmp_path, monkeypatch):
    """Point every host-runtime root at this test's ``tmp_path``.

    ``KESTREL_HOST_DB_PATH`` is the authoritative override for the
    host-feature database. ``HOME`` and ``KESTREL_HOME`` close the two
    default branches behind it, so a code path that ignores the override —
    or resolves some *other* implicit host-runtime root, such as the Phoenix
    trace store, the host-feature manifest, or the ``~/.kestrel`` project
    fallback — still lands in the temporary directory rather than on the
    operator's disk.

    Nothing here is created eagerly. Every writer on these paths already
    creates what it needs — ``prepare_host_database`` even creates its own
    parent ``0700``, the same custody path production takes — and a test is
    entitled to assert that its ``tmp_path`` stayed empty.
    """
    if request.node.get_closest_marker(OWNS_HOST_PATHS_MARKER):
        yield
        return

    root = tmp_path / ISOLATION_DIRNAME

    monkeypatch.setenv("HOME", str(root / "home"))
    monkeypatch.setenv("KESTREL_HOME", str(root / "kestrel-home"))
    monkeypatch.setenv(
        HOST_DB_PATH_ENV,
        str(root / "host-data" / HOST_FEATURE_DB_FILENAME),
    )

    # ``token_counter`` freezes its cache path at import time, so ``HOME``
    # above cannot move it: without this, every unit test reads (and a
    # discovery run rewrites) the operator's real
    # ``~/.kestrel/discovered_context_limits.json``. Reset the one-time read
    # too, so the redirect is what the next lookup sees.
    monkeypatch.setattr(
        token_counter, "CACHE_FILE", root / "discovered_context_limits.json"
    )
    monkeypatch.setattr(token_counter, "_cached_limits", None)

    # ``project_dir`` memoizes on ``(KESTREL_HOME, cwd)``. The key changes
    # with the value so a stale answer is impossible, but the cache is small
    # and per-test temporary homes would otherwise evict real entries.
    paths.reset_cache()
    try:
        yield
    finally:
        paths.reset_cache()


@pytest.fixture
def kestrel_toml_catalog(request, monkeypatch):
    """Publish ``[llm.catalog.<section>]`` blocks into this test's project dir.

    The model catalog is a process-wide singleton loaded from
    ``project_dir()/kestrel.toml``. Since the isolation above resolves that
    directory inside ``tmp_path``, a test asserting catalog-driven behaviour
    has to write the catalog rather than read whichever ``kestrel.toml`` the
    machine happens to carry.

    Call it once per section, e.g.
    ``kestrel_toml_catalog("context_limits_override", {"gpt-4": 8192})``.
    Values are written verbatim, so they must be TOML integers. Repeated
    calls accumulate; the memoized services are dropped each time and
    restored at teardown.
    """
    from kestrel_sovereign.llm import model_catalog

    if request.node.get_closest_marker(OWNS_HOST_PATHS_MARKER):
        # Without the isolation above, ``project_dir()`` is the operator's own
        # project and this would overwrite their kestrel.toml. Refuse instead.
        pytest.fail(
            f"kestrel_toml_catalog writes project_dir()/kestrel.toml and is "
            f"unsafe under the {OWNS_HOST_PATHS_MARKER!r} opt-out; either drop "
            f"the marker or point the catalog somewhere explicit yourself."
        )

    sections: dict[str, dict] = {}

    def publish(section: str, entries: dict) -> None:
        sections[section] = dict(entries)
        home = paths.project_dir()
        home.mkdir(parents=True, exist_ok=True)
        home.joinpath("kestrel.toml").write_text(
            "".join(
                f"[llm.catalog.{name}]\n"
                + "".join(f'"{key}" = {value}\n' for key, value in items.items())
                for name, items in sections.items()
            ),
            encoding="utf-8",
        )
        # Both modules memoize the service; drop each so the next lookup
        # loads the file above, and let monkeypatch restore them after.
        monkeypatch.setattr(model_catalog, "_catalog_service", None)
        monkeypatch.setattr(token_counter, "_catalog_service", None)

    return publish
