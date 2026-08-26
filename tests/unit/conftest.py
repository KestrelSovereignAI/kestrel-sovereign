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

The autouse fixture below moves those roots into a temporary directory of its
own, and seeds that directory with a host manifest that starts no host features
(#3099). Isolating ``KESTREL_HOME`` alone would have *widened* enablement: the
manifest is read from the resolved project dir, and
``instantiate_host_features`` treats a missing one as enable-all, so hiding the
operator's manifest could start host features they had explicitly disabled.
The seeded manifest says ``[host_features] default_enabled = false``, which is a
policy rather than a list -- host feature number seven cannot appear in the
suite without an edit here that says so.

Scope, precisely: this isolates *function-based* path resolution --
``paths.host_data_dir()``, ``paths.project_dir()``, ``host_database_path()``
and anything else that reads the environment when called. It does **not**
isolate module-scope constants that build an absolute path at **import**
time, because collection imports them before any fixture runs. Five such
constants still name the operator's real home:
``cli_serve.STATE_DIR`` / ``STATE_FILE`` / ``LOG_DIR``,
``destructive_policy.DEFAULT_TRASH_DIR``, and
``local_mps_adapter.DEFAULT_WORKING_DIR``. Those are tracked in #3104; the
fix there is resolve-on-call in the modules, not a longer patch list here --
a fixture that enumerates names is a set that grows, and the sixth constant
would silently reopen the hole.

Note for anyone verifying this: a probe that imports inside a test body sees
everything clean, because the fixture has already run. Only a module-level
import reproduces what a real session does.

Tests that deliberately exercise path resolution opt out with
``@pytest.mark.owns_host_paths`` and set the same variables themselves —
``tests/unit/test_host_feature_storage.py`` is the worked example.
"""

from __future__ import annotations

import pytest

from kestrel_sovereign import paths
from kestrel_sovereign.agent import token_counter
from kestrel_sovereign.host_features.discovery import (
    DEFAULT_ENABLED_KEY,
    HOST_MANIFEST_FILENAME,
    HOST_SCOPE_TABLE,
)
from kestrel_sovereign.host_features.storage import (
    HOST_DB_PATH_ENV,
    HOST_FEATURE_DB_FILENAME,
)

#: Marker name for tests that own host/home path resolution themselves.
OWNS_HOST_PATHS_MARKER = "owns_host_paths"

#: Name of the fixture's own temporary root. It is a sibling of each test's
#: ``tmp_path``, not a child: the fixture writes a host manifest, and a
#: directory the test owns is the wrong place for the fixture's state — one
#: test asserts its ``tmp_path`` is empty, and every test is entitled to.
ISOLATION_DIRNAME = "_kestrel_host_runtime_isolation"

#: The seeded manifest. A *default*, deliberately not a list of slugs: a list
#: would name today's host features and silently miss tomorrow's, which is the
#: same silent widening this fixture exists to close (#3099).
HOST_FEATURES_DISABLED_MANIFEST = (
    "# Seeded by tests/unit/conftest.py — the unit suite starts no host\n"
    "# features. Opt out per test with @pytest.mark.owns_host_paths.\n"
    f"[{HOST_SCOPE_TABLE}]\n"
    f"{DEFAULT_ENABLED_KEY} = false\n"
)


@pytest.fixture(autouse=True)
def _isolate_host_runtime_paths(request, tmp_path_factory, monkeypatch):
    """Point every host-runtime root at a temporary directory of our own.

    ``KESTREL_HOST_DB_PATH`` is the authoritative override for the
    host-feature database. ``HOME`` and ``KESTREL_HOME`` close the two
    default branches behind it, so a code path that ignores the override —
    or resolves some *other* implicit host-runtime root, such as the Phoenix
    trace store, the host-feature manifest, or the ``~/.kestrel`` project
    fallback — still lands in the temporary directory rather than on the
    operator's disk.

    The one thing created eagerly is the host manifest, because
    ``instantiate_host_features`` reads it from the resolved project dir at
    lifespan time and cannot be told to look elsewhere. Everything else is
    left to its writer — ``prepare_host_database`` even creates its own parent
    ``0700``, the same custody path production takes.
    """
    if request.node.get_closest_marker(OWNS_HOST_PATHS_MARKER):
        yield
        return

    root = tmp_path_factory.mktemp(ISOLATION_DIRNAME)
    project_home = root / "kestrel-home"

    monkeypatch.setenv("HOME", str(root / "home"))
    monkeypatch.setenv("KESTREL_HOME", str(project_home))
    monkeypatch.setenv(
        HOST_DB_PATH_ENV,
        str(root / "host-data" / HOST_FEATURE_DB_FILENAME),
    )

    # Hiding the operator's manifest is not neutral: absent means enable-all,
    # so isolation without this file would start host features the operator
    # had turned off. Written before the first test line runs, since the
    # lifespan reads it during startup.
    project_home.mkdir(parents=True, exist_ok=True)
    project_home.joinpath(HOST_MANIFEST_FILENAME).write_text(
        HOST_FEATURES_DISABLED_MANIFEST, encoding="utf-8"
    )

    # ``token_counter`` freezes its cache path at import time, so ``HOME``
    # above cannot move it: without this, every unit test reads (and a
    # discovery run rewrites) the operator's real
    # ``~/.kestrel/discovered_context_limits.json``. Reset the one-time read
    # too, so the redirect is what the next lookup sees.
    #
    # This is a point fix for the one import-time constant this suite was
    # observed to write, NOT a general solution: five more are frozen the
    # same way and are deliberately left alone here (see the module
    # docstring and #3104). Do not grow this into a patch list.
    monkeypatch.setattr(
        token_counter, "CACHE_FILE", root / "discovered_context_limits.json"
    )
    monkeypatch.setattr(token_counter, "_cached_limits", None)

    # ``project_dir`` memoizes on ``(KESTREL_HOME, cwd)``. The key changes
    # with the value so a stale answer is impossible, but the cache is small
    # and per-test temporary homes would otherwise evict real entries.
    paths.reset_cache()
    try:
        yield root
    finally:
        paths.reset_cache()


@pytest.fixture
def host_runtime_isolation_root(_isolate_host_runtime_paths):
    """The temporary root this test's host-runtime paths were redirected into.

    Requesting it is how a test asserts *where* a resolved path landed without
    reconstructing the layout from the environment.
    """
    if _isolate_host_runtime_paths is None:
        pytest.fail(
            f"host-runtime isolation is off under the "
            f"{OWNS_HOST_PATHS_MARKER!r} opt-out, so there is no isolation "
            f"root; drop the marker or resolve the path yourself."
        )
    return _isolate_host_runtime_paths


@pytest.fixture
def kestrel_toml_catalog(request, monkeypatch):
    """Publish ``[llm.catalog.<section>]`` blocks into this test's project dir.

    The model catalog is a process-wide singleton loaded from
    ``project_dir()/kestrel.toml``. Since the isolation above resolves that
    directory into a temporary one, a test asserting catalog-driven behaviour
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
