"""Unit-suite isolation from the operator's real host-runtime state (#3087).

Many tests here build ``TestClient(server.app)`` without overriding the
lifespan. That lifespan calls ``build_host_context()`` with no ``db_path``,
so the host-feature database resolves through the *production* precedence:
``$KESTREL_HOST_DB_PATH``, else ``$KESTREL_DB_PATH/host-data``, else
``$KESTREL_HOME/host-data``, else ``~/.kestrel/host-data``. On a developer
machine that last branch is the live fleet database — running the unit suite
migrated its schema as a side effect of ``pytest``, and the real host features
named by the project's ``.kestrel-host-features.toml`` started (and recorded
their start failures) against it. CI never notices: the runner's ``HOME`` is
fresh, so the same code writes a throwaway file.

Every test in the repository already has every runtime-path variable
(``paths.RUNTIME_PATH_ENV_NAMES``) redirected by the suite-wide autouse
fixture in ``tests/conftest.py`` (#3286), and every resolver refuses a path outside
the test's temporary roots anyway (see
``tests/shared/host_runtime_isolation.py``). The autouse fixture below adds the
unit tier's ``HOME`` redirect inside that same temporary directory, and seeds
it with a host manifest that starts no host features (#3099).
Isolating ``KESTREL_HOME`` alone would have *widened* enablement: the
manifest is read from the resolved project dir, and
``instantiate_host_features`` treats a missing one as enable-all, so hiding the
operator's manifest could start host features they had explicitly disabled.
The seeded manifest says ``[host_features] default_enabled = false``, which is a
policy rather than a list -- host feature number seven cannot appear in the
suite without an edit here that says so.

Scope, precisely: the per-test redirect isolates *function-based* path
resolution -- ``paths.host_data_dir()``, ``paths.project_dir()``,
``host_database_path()`` and anything else that reads the environment when
called. A module-scope constant that builds a path at **import** time is
collected before any fixture runs. Those that read a registered runtime-path
variable (``cli_serve.STATE_DIR`` / ``STATE_FILE`` / ``LOG_DIR``,
``destructive_policy.DEFAULT_TRASH_DIR``, ``config.TRUSTED_AGENTS_DIR``)
resolve against the session pins ``tests/conftest.py`` installs before any
package import (#3286), so they name a session temporary root, shared by every
test, rather than the operator's home. Resolving on call in the modules
remains the fix (#3104), as ``local_mps_adapter.default_working_dir`` now does,
not a longer patch list here.

Note for anyone verifying this: a probe that imports inside a test body sees
everything clean, because the fixture has already run. Only a module-level
import reproduces what a real session does.

Tests that deliberately exercise path resolution opt out with
``@pytest.mark.owns_host_paths`` and set the same variables themselves —
``tests/unit/test_host_feature_storage.py`` is the worked example.
"""

from __future__ import annotations

import pytest
import pytest_asyncio

from kestrel_sovereign import paths
from kestrel_sovereign.agent import token_counter
from kestrel_sovereign.host_features.discovery import (
    DEFAULT_ENABLED_KEY,
    HOST_MANIFEST_FILENAME,
    HOST_SCOPE_TABLE,
)
from tests.shared.host_runtime_isolation import (
    OWNS_HOST_PATHS_MARKER,
    PROJECT_HOME_DIRNAME,
)
from tests.utils.ci_budget import refuse_unbudgeted_timeouts

#: The seeded manifest. A *default*, deliberately not a list of slugs: a list
#: would name today's host features and silently miss tomorrow's, which is the
#: same silent widening this fixture exists to close (#3099).
HOST_FEATURES_DISABLED_MANIFEST = (
    "# Seeded by tests/unit/conftest.py — the unit suite starts no host\n"
    "# features. Opt out per test with @pytest.mark.owns_host_paths.\n"
    f"[{HOST_SCOPE_TABLE}]\n"
    f"{DEFAULT_ENABLED_KEY} = false\n"
)


def pytest_collection_modifyitems(config, items):
    """Refuse a per-test timeout the tier's wall-clock budget cannot hold."""
    refuse_unbudgeted_timeouts(config, items, "unit")


@pytest.fixture(autouse=True)
def _isolate_unit_host_runtime_paths(_isolate_host_runtime_paths, monkeypatch):
    """Add ``HOME`` to the suite-wide redirect of the host-runtime roots.

    The suite-wide fixture already moved every runtime-path variable. ``HOME``
    closes the last default branch behind them, so a code path that ignores
    the variables — or resolves some *other* implicit host-runtime root, such
    as the Phoenix trace store or the ``~/.kestrel`` project fallback — still
    lands in the temporary directory rather than on the operator's disk.

    The one thing created eagerly is the host manifest, because
    ``instantiate_host_features`` reads it from the resolved project dir at
    lifespan time and cannot be told to look elsewhere. Everything else is
    left to its writer — ``prepare_host_database`` even creates its own parent
    ``0700``, the same custody path production takes.
    """
    root = _isolate_host_runtime_paths
    if root is None:  # the owns_host_paths opt-out
        yield
        return

    project_home = root / PROJECT_HOME_DIRNAME

    monkeypatch.setenv("HOME", str(root / "home"))

    # Hiding the operator's manifest is not neutral: absent means enable-all,
    # so isolation without this file would start host features the operator
    # had turned off. Written before the first test line runs, since the
    # lifespan reads it during startup.
    project_home.joinpath(HOST_MANIFEST_FILENAME).write_text(
        HOST_FEATURES_DISABLED_MANIFEST, encoding="utf-8"
    )

    yield root


@pytest.fixture(autouse=True)
def _seed_isolated_project_config():
    """Override the suite-wide seed: the unit tier starts from an empty project.

    A test that needs catalog-driven behaviour writes it with
    ``kestrel_toml_catalog`` rather than reading the checkout's configuration.
    """


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


@pytest.fixture
def new_york_clock(monkeypatch):
    """A non-UTC process zone, so a rule about naive or local time can be
    seen to fail on CI's UTC runners. Undone in the right order:
    ``monkeypatch`` restores ``TZ`` at teardown but ``tzset()`` is what the C
    library reads, and on glibc the cached zone outlives the variable."""
    import time

    monkeypatch.setenv("TZ", "America/New_York")
    time.tzset()
    yield
    monkeypatch.undo()
    time.tzset()


# ---------------------------------------------------------------------------
# self_followup test environment (#3101 / #3128)
#
# Lives here rather than in test_self_followup_schedule.py because a second
# test module needs it, and importing a fixture by name makes ruff read every
# test signature that takes it as an F811 redefinition of the import. The
# suppression for that is one noqa per test signature -- a list that grows an
# entry every time a test in either file uses the fixture. pytest discovers a
# conftest fixture with no import at all, so nothing shadows and nothing
# enumerates.
#
# The heavy scheduler/dispatcher imports are deliberately INSIDE the fixture
# body: this conftest is loaded for the whole unit suite, and a module-scope
# import here would pull the scheduler stack into every unit test's collection.
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def followup_env(tmp_path):
    """Real dispatcher + real scheduler runner + real SQLite, wired together."""
    import asyncio

    from kestrel_sovereign.agent.sleep import SleepMixin
    from kestrel_sovereign.agent.turn_lifecycle import TurnLifecycleMixin
    from kestrel_sovereign.features.scheduler.feature import SchedulerFeature
    from kestrel_sovereign.features.scheduler.runner import SchedulerRunner
    from kestrel_sovereign.kestrel_agent import KestrelAgent
    from kestrel_sovereign.signals import (
        OrderedLockManager,
        SignalDispatcher,
        SignalLogStore,
        SourceRegistry,
    )
    from kestrel_sovereign.signals.sources.scheduler import (
        build_cron_registrations,
    )
    from kestrel_sovereign.storage.async_database import AsyncDatabase
    from kestrel_sovereign.storage.db import SQLiteBackend

    class _FakeAgent(SleepMixin, TurnLifecycleMixin):
        """Minimal agent that records the turns a dispatch actually produced.

        Inherits the REAL :class:`TurnLifecycleMixin` rather than stubbing turn
        ownership. ``_owns_live_turn`` is the guard these tests exercise, so a
        double that simply answered True would assert the thing under test
        instead of exercising it; with the real mixin, ``owns_live_turn()`` is
        true only inside ``async with agent._turn_lifecycle()`` and false the
        instant that block exits, which is the actual production contract.
        """

        did = "did:test:self-followup"
        agent_name = "followup-test"

        def __init__(self):
            self.background_tasks = []
            self.sleep_hooks = []
            self.turn_prompts: list[str] = []
            self.turn_kwargs: list[dict] = []
            self.turn_session_id: str | None = None
            self._live_turn_id: str | None = None
            self._active_session_id: str | None = None

        async def process_input(self, prompt, pre_turn_guard=None, **kwargs):
            # Declares `pre_turn_guard` and evaluates it with the real agent's
            # evaluator before recording the turn, as `KestrelAgent` does first
            # thing inside its span (#3310). A refused guard therefore records
            # no turn, exactly like production.
            KestrelAgent._evaluate_pre_turn_guard(pre_turn_guard)
            self.turn_prompts.append(prompt)
            self.turn_kwargs.append(kwargs)
            return "follow-up handled"

        def get_turn_bound_session_id(self):
            return self.turn_session_id

        def _track_background_task(self, coro, *, name):
            task = asyncio.create_task(coro, name=name)
            self.background_tasks.append(task)
            return task

    backend = SQLiteBackend(str(tmp_path / "self_followup.db"))
    await backend.connect()
    store = SignalLogStore(backend)
    await store.initialize()

    registry = SourceRegistry()
    agent = _FakeAgent()
    dispatcher = SignalDispatcher(
        agent=agent,
        registry=registry,
        lock_manager=OrderedLockManager(),
        store=store,
    )
    agent.dispatcher = dispatcher
    agent.signal_registry = registry

    async def _lookup(name, args):  # no cron tool is exercised here
        raise AssertionError(f"unexpected tool lookup for {name}")

    for registration in build_cron_registrations(
        tool_lookup=_lookup,
        reason_codes_lookup=lambda _name: frozenset(),
        agent=agent,
    ):
        registry.register(registration)

    db = AsyncDatabase(backend)
    feature = SchedulerFeature(agent)
    feature._db = db
    feature._agent_id = agent.did

    runner = SchedulerRunner(db, agent.did, feature._dispatch_scheduled_task)
    await runner._ensure_tables()

    yield agent, feature, runner, db, backend

    pending = [t for t in agent.background_tasks if not t.done()]
    if pending:
        await asyncio.gather(*pending, return_exceptions=True)
    await backend.close()
