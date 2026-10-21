"""Put back, and fail, whatever leaves ``os.environ`` changed (#3671).

A test that writes ``os.environ`` directly, or calls code that does, and never
puts it back hands every later test in its process an environment that test
did not ask for. Three ``test_host_features`` tests passed only because an
earlier test had left ``KESTREL_API_KEY`` set; with the unit tier split across
runners they failed in whichever shard held them.

The guard has one rule at each of three boundaries. Each puts a change back,
then fails, so no leak passes unnoticed; at a test and at an import that also
keeps the leak from every later test:

* **A test.** ``_environment_guard`` is an autouse fixture. pytest sets up a
  test's class, module, package and session fixtures before any of its
  function-scoped ones, and this is the first function-scoped fixture of every
  test, so it records the environment after the former and before the latter.
  Its teardown runs after every other function-scoped fixture's,
  ``monkeypatch``'s included, and before any higher-scoped teardown. What
  differs then is the test's, ``xfail`` mark or not. That includes a
  higher-scoped fixture first set up inside the test, through
  ``request.getfixturevalue`` or a function-scoped fixture that overrides it by
  name: request such a fixture as an argument instead.
* **A test module's import.** A module-level write fails that module's
  collection.
* **The run.** The environment is recorded as the session starts and compared
  once every fixture has been torn down, in the last test's teardown. A
  difference then was left by a class, module, package or session fixture, or
  by other code outside any test, and fails that last test. Every test after
  the leak ran with it, so this boundary reports a leak but cannot keep it
  from them. A run that stops after a test's teardown has left fixtures up (an
  interrupt, or a teardown error that reaches ``--maxfail``) tears them down
  as the session finishes. It is compared then, and with no test left to fail
  a leak is named in the warnings summary; such a run has already failed.

``register`` installs this module as a plugin, so the fixture is visible to
every test and the session hooks see the session.

Not covered: ``os.putenv``, which bypasses ``os.environ``.
"""

from __future__ import annotations

from collections.abc import Mapping
import os
import sys
from typing import NoReturn
import warnings

import pytest

#: pytest rewrites this for each phase of each test, so it differs between a
#: test's setup and its teardown by design.
PHASE_VARIABLE = "PYTEST_CURRENT_TEST"

PLUGIN_NAME = "kestrel-environment-guard"

_SESSION_START_KEY = pytest.StashKey[dict[str, str]]()
#: Set once a test of this process has been set up.
_RAN_KEY = pytest.StashKey[bool]()
#: Set once the run has been compared with the session start.
_COMPARED_KEY = pytest.StashKey[bool]()
#: Set on a test whose teardown reported a leak.
_LEAKED_KEY = pytest.StashKey[bool]()

_REMEDY = (
    "Set a variable with monkeypatch.setenv. To start without one that the "
    "code under test may then write, call "
    "tests.shared.environment_guard.unset_environment(monkeypatch, NAME): "
    "monkeypatch.delenv records nothing for a variable that is already absent."
)

_RUN_REMEDY = (
    "A class, module, package or session fixture can hold a variable with "
    "pytest.MonkeyPatch.context() around its yield, which puts it back as the "
    "fixture is torn down."
)


def register(config: pytest.Config) -> None:
    """Install the guard for every test of the run."""
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


@pytest.fixture(autouse=True)
def _environment_guard(request: pytest.FixtureRequest):
    """Fail a test that leaves ``os.environ`` changed, after putting it back."""
    request.session.stash[_RAN_KEY] = True
    before = snapshot_environment()
    yield
    changes = restore_environment(before)
    if changes:
        _fail(
            request.node,
            f"{request.node.nodeid} left os.environ changed. It has been put "
            "back; left in place, every later test in this process would have "
            "run with it (#3671):\n" + "\n".join(changes) + f"\n{_REMEDY}",
        )


def _leak_error(item: pytest.Item, message: str) -> BaseException:
    """The error that reports a leak, recorded on ``item`` for its report."""
    item.stash[_LEAKED_KEY] = True
    return pytest.fail.Exception(message, pytrace=False)


def _fail(item: pytest.Item, message: str) -> NoReturn:
    raise _leak_error(item, message)


@pytest.hookimpl(trylast=True)
def pytest_sessionstart(session: pytest.Session) -> None:
    session.stash[_SESSION_START_KEY] = snapshot_environment()


def _run_leak(session: pytest.Session) -> str | None:
    """Put back what the run left changed, and describe it."""
    session.stash[_COMPARED_KEY] = True
    changes = restore_environment(session.stash[_SESSION_START_KEY])
    if not changes:
        return None
    return (
        "The run left os.environ changed once every fixture had been torn "
        "down. It has been put back. A class, module, package or session "
        "fixture, or other code outside any test, changed it and nothing put "
        "it back; every test after that change ran with it (#3671):\n"
        + "\n".join(changes) + f"\n{_RUN_REMEDY}"
    )


@pytest.hookimpl(wrapper=True)
def pytest_runtest_teardown(item: pytest.Item, nextitem: pytest.Item | None):
    """After the last test every fixture is gone: compare with the session start."""
    if nextitem is not None:
        return (yield)
    try:
        result = yield
    except (pytest.exit.Exception, KeyboardInterrupt):
        # An interrupted run is compared as the session finishes.
        raise
    except BaseException as error:
        leak = _run_leak(item.session)
        if leak is None:
            raise
        # Both are reported: the teardown's own failure and the leak.
        raise BaseExceptionGroup(
            "the last test's teardown failed, and the run left os.environ changed",
            [error, _leak_error(item, leak)],
        ) from None
    leak = _run_leak(item.session)
    if leak is not None:
        _fail(item, leak)
    return result


@pytest.hookimpl(wrapper=True, trylast=True)
def pytest_sessionfinish(session: pytest.Session):
    """Compare a run whose fixtures are torn down here, after its last test.

    No test is left to fail, so the leak is named in the warnings summary. Only
    a run that has failed or been interrupted ends with fixtures still up.
    """
    try:
        return (yield)
    finally:
        ran = session.stash.get(_RAN_KEY, False)
        if ran and not session.stash.get(_COMPARED_KEY, False):
            leak = _run_leak(session)
            if leak is not None:
                warnings.warn(pytest.PytestWarning(leak))
                if session.exitstatus == pytest.ExitCode.OK:
                    session.exitstatus = pytest.ExitCode.TESTS_FAILED


@pytest.hookimpl(wrapper=True)
def pytest_runtest_makereport(item: pytest.Item, call: pytest.CallInfo[None]):
    """Keep a leak a failure: an ``xfail`` mark excuses the test, not its leak."""
    report = yield
    if call.when == "teardown" and item.stash.get(_LEAKED_KEY, False):
        if hasattr(report, "wasxfail"):
            del report.wasxfail
        report.outcome = "failed"
    return report


@pytest.hookimpl(wrapper=True)
def pytest_make_collect_report(collector: pytest.Collector):
    """Fail the collection of a module whose import left the environment changed."""
    if not isinstance(collector, pytest.Module):
        return (yield)
    before = snapshot_environment()
    report = yield
    changes = restore_environment(before)
    if not changes:
        return report
    message = (
        f"Importing {collector.nodeid} left os.environ changed. It has been put "
        "back; left in place, every test in this process would have run with it "
        "(#3671):\n" + "\n".join(changes) + f"\n{_REMEDY}"
    )
    if report.failed:
        # Keep the module's own error, and report the leak after it.
        message = f"{report.longrepr}\n\n{message}"
    return pytest.CollectReport(
        collector.nodeid, "failed", message, [], sections=report.sections
    )
