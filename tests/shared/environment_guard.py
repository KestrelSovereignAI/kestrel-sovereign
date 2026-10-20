"""Put back, and fail, a test or fixture that leaves ``os.environ`` changed (#3671).

A test that writes ``os.environ`` directly, or calls code that does, and never
puts it back hands every later test in its process an environment that test
did not ask for. Three ``test_host_features`` tests passed only because an
earlier test had left ``KESTREL_API_KEY`` set; with the unit tier split across
runners they failed in whichever shard held them.

So every test is checked. The environment is recorded when the test's first
function-scoped fixture is set up, which is after every module and session
fixture it uses and before any function fixture or the test body can write; a
test without one is recorded once its setup is done. It is compared after the
last function-scoped fixture is torn down. Every
``monkeypatch`` has been undone by then, whatever order its fixtures ran in,
so only a change that nothing would undo is reported. A module or session
fixture first set up during the test, through ``request.getfixturevalue``,
outlives it by design, so what its setup changed is added to the record. A leak
is put back first and then fails the test, so it can neither reach the next
test nor pass unnoticed.

Every class, module, package and session fixture is checked as it is torn
down, which happens after the last test that uses it: whatever that teardown
leaves changed, every later test inherits, the next module's included. The
environment is recorded as its teardown starts, after every fixture that
depends on it has gone, and each variable its setup changed is expected back as
it was before that setup. What differs afterwards is put back and fails the
test whose teardown retired the fixture, naming the fixture.

Importing a test module is checked the same way: a module-level write is put
back and fails that module's collection.

``register`` installs these hooks as a plugin. Imported into a conftest they
would see only the nodes below its directory, and the session sits at the
rootdir, above ``tests/``: no session fixture would reach them.

Not covered: ``os.putenv``, which bypasses ``os.environ``.
"""

from __future__ import annotations

from collections.abc import Mapping
import os
import sys

import pytest

#: pytest rewrites this for each phase of each test, so it differs between a
#: test's setup and its teardown by design.
PHASE_VARIABLE = "PYTEST_CURRENT_TEST"

PLUGIN_NAME = "kestrel-environment-guard"

_SNAPSHOT_KEY = pytest.StashKey[dict[str, str]]()
#: The test being checked, so a fixture set up during it can find its record.
_WATCHED_KEY = pytest.StashKey[pytest.Item]()

_REMEDY = (
    "Set a variable with monkeypatch.setenv. To start without one that the "
    "code under test may then write, call "
    "tests.shared.environment_guard.unset_environment(monkeypatch, NAME): "
    "monkeypatch.delenv records nothing for a variable that is already absent."
)

_FIXTURE_REMEDY = (
    "A class, module, package or session fixture can hold a variable with "
    "pytest.MonkeyPatch.context() around its yield, which puts it back as the "
    "fixture is torn down."
)


def register(config: pytest.Config) -> None:
    """Install the guard for every node of the run, the session's included."""
    config.pluginmanager.register(sys.modules[__name__], PLUGIN_NAME)


def unset_environment(monkeypatch: pytest.MonkeyPatch, *names: str) -> None:
    """Remove ``names`` for this test, and have ``monkeypatch`` undo any later write.

    ``monkeypatch.delenv(name, raising=False)`` records nothing when ``name``
    is already absent, so a value the code under test then sets outlives the
    test. Setting the variable first makes ``monkeypatch`` record how it was.
    """
    for name in names:
        monkeypatch.setenv(name, "")
        monkeypatch.delenv(name)


def snapshot_environment() -> dict[str, str]:
    """A copy of ``os.environ``, without the variable pytest rewrites per phase."""
    environment = dict(os.environ)
    environment.pop(PHASE_VARIABLE, None)
    return environment


def restore_environment(expected: Mapping[str, str]) -> list[str]:
    """Put ``os.environ`` back to ``expected``; return one line per variable it changed.

    The lines name variables, never their values: a leaked variable is often a
    credential, and these lines go to the CI log.
    """
    current = snapshot_environment()
    changes = []
    for name in sorted(current.keys() | expected.keys()):
        if name not in expected:
            changes.append(f"  {name}: set")
            del os.environ[name]
        elif name not in current:
            changes.append(f"  {name}: removed")
            os.environ[name] = expected[name]
        elif current[name] != expected[name]:
            changes.append(f"  {name}: changed")
            os.environ[name] = expected[name]
    return changes


def _carry_fixture_changes(
    expected: dict[str, str], before: Mapping[str, str], after: Mapping[str, str]
) -> None:
    """Add what a module or session fixture's setup changed to a test's record."""
    for name in before.keys() | after.keys():
        if name not in after:
            expected.pop(name, None)
        elif before.get(name) != after[name]:
            expected[name] = after[name]


def _check_item(item: pytest.Item) -> None:
    expected = item.stash[_SNAPSHOT_KEY]
    del item.stash[_SNAPSHOT_KEY]
    if item.config.stash.get(_WATCHED_KEY, None) is item:
        del item.config.stash[_WATCHED_KEY]
    changes = restore_environment(expected)
    if changes:
        pytest.fail(
            f"{item.nodeid} left os.environ changed. It has been put back; "
            "left in place, every later test in this process would have run "
            "with it (#3671):\n" + "\n".join(changes) + f"\n{_REMEDY}",
            pytrace=False,
        )


def _watch(item: pytest.Item) -> None:
    item.stash[_SNAPSHOT_KEY] = snapshot_environment()
    item.config.stash[_WATCHED_KEY] = item
    item.addfinalizer(lambda: _check_item(item))


class _FixtureRetirement:
    """Checks that one class, module, package or session fixture's teardown
    leaves the environment as that fixture's setup found it."""

    def __init__(self, fixturedef: pytest.FixtureDef, request: pytest.FixtureRequest):
        self._fixturedef = fixturedef
        self._owner = request.node
        self._before_setup = snapshot_environment()
        self._after_setup: dict[str, str] = {}
        self._expected: dict[str, str] = {}

    def setup_finished(self) -> None:
        self._after_setup = snapshot_environment()

    def carry_setup_into(self, record: dict[str, str]) -> None:
        """Add what this fixture's setup changed to the record of a test it outlives."""
        _carry_fixture_changes(record, self._before_setup, self._after_setup)

    def teardown_starts(self) -> None:
        """Record what the environment must be once this fixture is gone."""
        self._expected = snapshot_environment()
        for name in self._before_setup.keys() | self._after_setup.keys():
            if self._before_setup.get(name) == self._after_setup.get(name):
                continue
            if name in self._before_setup:
                self._expected[name] = self._before_setup[name]
            else:
                self._expected.pop(name, None)

    def teardown_finished(self) -> None:
        changes = restore_environment(self._expected)
        if changes:
            owner = self._owner.nodeid or "the session"
            pytest.fail(
                f"Tearing down the {self._fixturedef.scope}-scoped fixture "
                f"{self._fixturedef.argname!r} of {owner} left os.environ "
                "changed. It has been put back; left in place, every later "
                "test in this process would have run with it (#3671):\n"
                + "\n".join(changes) + f"\n{_FIXTURE_REMEDY}",
                pytrace=False,
            )


@pytest.hookimpl(hookwrapper=True)
def pytest_fixture_setup(fixturedef, request):
    """Record the environment as a test's first function-scoped fixture starts.

    The check is registered on the item before that fixture's own finalizer, so
    it runs after every function-scoped fixture of the test is torn down. A
    higher-scoped fixture's request belongs to a collector, not the item; one
    set up while a test is being checked adds its changes to that record.

    A higher-scoped fixture gets two finalizers of its own, one on each side of
    its setup. A fixture's finalizers run in reverse, so the check added before
    the setup runs after the fixture's teardown, and the record added after it
    runs before that teardown, once every fixture that depends on it has gone.
    """
    item = request.node
    if isinstance(item, pytest.Item):
        if _SNAPSHOT_KEY not in item.stash:
            _watch(item)
        yield
        return
    retirement = _FixtureRetirement(fixturedef, request)
    fixturedef.addfinalizer(retirement.teardown_finished)
    yield
    retirement.setup_finished()
    fixturedef.addfinalizer(retirement.teardown_starts)
    watched = request.config.stash.get(_WATCHED_KEY, None)
    if watched is not None:
        retirement.carry_setup_into(watched.stash[_SNAPSHOT_KEY])


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_setup(item):
    """Record the environment after setup for a test with no function-scoped fixture."""
    outcome = yield
    if outcome.exception is None and _SNAPSHOT_KEY not in item.stash:
        _watch(item)


@pytest.hookimpl(hookwrapper=True)
def pytest_make_collect_report(collector):
    """Fail the collection of a module whose import left the environment changed."""
    if not isinstance(collector, pytest.Module):
        yield
        return
    before = snapshot_environment()
    outcome = yield
    changes = restore_environment(before)
    if not changes:
        return
    message = (
        f"Importing {collector.nodeid} left os.environ changed. It has been put "
        "back; left in place, every test in this process would have run with it "
        "(#3671):\n" + "\n".join(changes) + f"\n{_REMEDY}"
    )
    report = outcome.get_result()
    if report.failed:
        # Keep the module's own error, and report the leak beside it.
        report.sections.append(("os.environ", message))
        return
    outcome.force_result(
        pytest.CollectReport(
            collector.nodeid, "failed", message, [], sections=report.sections
        )
    )
