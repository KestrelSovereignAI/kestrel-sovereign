"""Whatever leaves ``os.environ`` changed is put back and fails (#3671).

Three boundaries: a test (from the guard fixture's setup to its teardown), a
test module's import, and the run (from the session's start until every
fixture has been torn down).
"""

import os
from pathlib import Path
import subprocess
import textwrap
import xml.etree.ElementTree as ElementTree

import pytest

from kestrel_sovereign.a2a.transport_auth import (
    A2A_TRANSPORT_KEY_ENV,
    ensure_a2a_transport_key,
)
from tests.shared import environment_guard
from tests.shared.child_pytest import run_child_pytest
from tests.shared.environment_guard import (
    PLUGIN_NAME,
    restore_environment,
    snapshot_environment,
    unset_environment,
)

#: A liveness bound on the child, not a measurement: it runs a few small files.
SUBPROCESS_TIMEOUT_SECONDS = 120

#: The child's tests live here, below its rootdir, as ``tests/`` sits below the
#: repository's.
SUITE = "suite"

#: Installed as ``tests/conftest.py`` installs it.
CHILD_CONFTEST = textwrap.dedent(
    """
    from tests.shared.environment_guard import register


    def pytest_configure(config):
        register(config)
    """
)

#: Every value below is one the guard must never print.
LEAKED_VALUES = (
    "leaked-value-3671", "changed-value-3671", "plain-value-3671",
    "fixture-value-3671", "teardown-value-3671", "kept-value-3671",
    "class-value-3671", "session-value-3671", "late-value-3671",
    "import-value-3671",
)

TEST_MESSAGE = "left os.environ changed. It has been put back"
RUN_MESSAGE = "The run left os.environ changed once every fixture had been torn down"

#: Each leaking test is followed by one that sees none of its leaks.
PROBE = textwrap.dedent(
    """
    import os

    import pytest

    from tests.shared.environment_guard import unset_environment


    @pytest.fixture(scope="module")
    def module_setting():
        with pytest.MonkeyPatch.context() as patch:
            patch.setenv("GUARD_PROBE_MODULE", "held for the module")
            yield


    @pytest.fixture(scope="module")
    def late_module_setting():
        with pytest.MonkeyPatch.context() as patch:
            patch.setenv("GUARD_PROBE_LATE", "late-value-3671")
            yield


    @pytest.fixture
    def leaky():
        os.environ["GUARD_PROBE_FIXTURE"] = "fixture-value-3671"


    @pytest.fixture(autouse=True)
    def autouse_monkeypatch(monkeypatch):
        # Undone only when monkeypatch tears down, after the test: a check run
        # before that undo would report it.
        monkeypatch.setenv("GUARD_PROBE_AUTOUSE", "set by an autouse fixture")


    def test_sets():
        os.environ["GUARD_PROBE_LEAK"] = "leaked-value-3671"


    def test_removes():
        del os.environ["GUARD_PROBE_PRESET"]


    def test_changes():
        os.environ["GUARD_PROBE_CHANGED"] = "changed-value-3671"


    def test_a_function_fixture_leaks(leaky):
        pass


    def test_sees_none_of_them():
        assert os.environ.get("GUARD_PROBE_LEAK") is None
        assert os.environ.get("GUARD_PROBE_PRESET") == "preset-value-3671"
        assert os.environ.get("GUARD_PROBE_CHANGED") == "original-value-3671"
        assert os.environ.get("GUARD_PROBE_FIXTURE") is None


    def test_monkeypatch_is_not_a_leak(monkeypatch):
        monkeypatch.setenv("GUARD_PROBE_SET", "by the test")
        monkeypatch.delenv("GUARD_PROBE_PRESET")


    def test_a_module_fixture_is_not_a_leak(module_setting):
        assert os.environ.get("GUARD_PROBE_MODULE") == "held for the module"


    def test_a_module_fixture_requested_inside_the_test_is_its_leak(request):
        request.getfixturevalue("late_module_setting")


    def test_unset_environment_owns_a_later_write(monkeypatch):
        unset_environment(monkeypatch, "GUARD_PROBE_WRITTEN_BY_CODE")
        os.environ["GUARD_PROBE_WRITTEN_BY_CODE"] = "as code under test would"


    def test_sees_the_original_environment():
        assert os.environ.get("GUARD_PROBE_SET") is None
        assert os.environ.get("GUARD_PROBE_LATE") is None
        assert os.environ.get("GUARD_PROBE_WRITTEN_BY_CODE") is None
    """
)

#: No fixture of its own: the guard is its only one.
PLAIN = textwrap.dedent(
    """
    import os


    def test_plain_leak():
        os.environ["GUARD_PROBE_PLAIN"] = "plain-value-3671"


    def test_plain_after():
        assert os.environ.get("GUARD_PROBE_PLAIN") is None
    """
)

#: A session fixture, visible to every module of the child.
SESSION_SETTING = textwrap.dedent(
    """


    import pytest


    @pytest.fixture(scope="session")
    def session_setting():
        with pytest.MonkeyPatch.context() as patch:
            patch.setenv("GUARD_PROBE_SESSION", "held for the session")
            yield
    """
)

#: A module fixture that requests the session fixture inside its own setup: the
#: session fixture's variable must outlive the module fixture.
NESTED = textwrap.dedent(
    """
    import os

    import pytest


    @pytest.fixture(scope="module")
    def module_using_the_session(request):
        request.getfixturevalue("session_setting")


    def test_uses_the_module_fixture(module_using_the_session):
        assert os.environ.get("GUARD_PROBE_SESSION") == "held for the session"
    """
)

NESTED_AFTER = textwrap.dedent(
    """
    import os


    def test_the_session_fixture_still_holds(session_setting):
        assert os.environ.get("GUARD_PROBE_SESSION") == "held for the session"
    """
)


def _outcomes(report: Path) -> dict[str, set[str]]:
    """Each test, as ``classname.name``, mapped to its junit error/failure tags."""
    outcomes: dict[str, set[str]] = {}
    for case in ElementTree.parse(report).iter("testcase"):
        tags = outcomes.setdefault(f"{case.get('classname')}.{case.get('name')}", set())
        tags.update(child.tag for child in case if child.tag in ("error", "failure"))
    return outcomes


def _run_suite(
    tmp_path: Path,
    modules: dict[str, str],
    args: list[str],
    *,
    fixtures: str = "",
) -> subprocess.CompletedProcess:
    """Run ``modules``, in order, from the child's ``suite`` directory.

    ``fixtures`` (fixtures or hooks) is added to the child's conftest.
    """
    suite = tmp_path / SUITE
    suite.mkdir()
    (suite / "conftest.py").write_text(CHILD_CONFTEST + fixtures, encoding="utf-8")
    *earlier, (last_name, last_source) = modules.items()
    for name, source in earlier:
        (suite / name).write_text(source, encoding="utf-8")
    return run_child_pytest(
        tmp_path,
        conftest="",
        test_file=f"{SUITE}/{last_name}",
        source=last_source,
        args=[*(f"{SUITE}/{name}" for name in modules), *args],
        timeout=SUBPROCESS_TIMEOUT_SECONDS,
    )


def _workers(workers: int) -> list[str]:
    distribution = ["--dist", "loadfile"] if workers else []
    return ["-p", "xdist", "-n", str(workers), *distribution]


@pytest.fixture
def inherited_environment(monkeypatch):
    """Variables the child inherits, for its tests to remove and change."""
    monkeypatch.setenv("GUARD_PROBE_PRESET", "preset-value-3671")
    monkeypatch.setenv("GUARD_PROBE_CHANGED", "original-value-3671")


@pytest.mark.timeout(SUBPROCESS_TIMEOUT_SECONDS + 15)
@pytest.mark.parametrize("workers", [0, 2], ids=["serial", "xdist"])
def test_a_leaking_test_is_put_back_and_fails(
    tmp_path, inherited_environment, workers
):
    completed = _run_suite(
        tmp_path,
        {
            "test_probe.py": PROBE,
            "test_plain.py": PLAIN,
            "test_nested.py": NESTED,
            "test_nested_after.py": NESTED_AFTER,
        },
        ["--junitxml=report.xml", *_workers(workers)],
        fixtures=SESSION_SETTING,
    )
    output = completed.stdout

    assert completed.returncode == pytest.ExitCode.TESTS_FAILED, output
    outcomes = _outcomes(tmp_path / "report.xml")
    failed = {
        "suite.test_probe.test_sets",
        "suite.test_probe.test_removes",
        "suite.test_probe.test_changes",
        "suite.test_probe.test_a_function_fixture_leaks",
        "suite.test_probe.test_a_module_fixture_requested_inside_the_test_is_its_leak",
        "suite.test_plain.test_plain_leak",
    }
    assert {name for name, tags in outcomes.items() if tags} == failed, output
    assert len(outcomes) == 14, outcomes
    for name in failed:
        assert outcomes[name] == {"error"}, (name, output)
    assert f"suite/test_probe.py::test_sets {TEST_MESSAGE}" in output, output
    for line in (
        "  GUARD_PROBE_LEAK: set",
        "  GUARD_PROBE_PRESET: removed",
        "  GUARD_PROBE_CHANGED: changed",
        "  GUARD_PROBE_FIXTURE: set",
        "  GUARD_PROBE_LATE: set",
        "  GUARD_PROBE_PLAIN: set",
    ):
        assert line in output, output
    # A held module or session fixture, monkeypatch and unset_environment are
    # never reported, and nothing is left for the run to report.
    for name in (
        "GUARD_PROBE_MODULE", "GUARD_PROBE_SESSION", "GUARD_PROBE_AUTOUSE",
        "GUARD_PROBE_SET", "GUARD_PROBE_WRITTEN_BY_CODE",
    ):
        assert name not in output, output
    assert RUN_MESSAGE not in output, output
    for value in LEAKED_VALUES:
        assert value not in output, output


@pytest.mark.timeout(SUBPROCESS_TIMEOUT_SECONDS + 15)
def test_a_leak_under_a_fixture_requested_late_is_still_the_tests(tmp_path):
    """The test leaks a variable, then requests a module fixture that holds the
    same variable. The test is reported; the fixture's teardown then puts the
    test's value back, which the run reports."""
    completed = _run_suite(
        tmp_path,
        {
            "test_late.py": textwrap.dedent(
                """
                import os

                import pytest


                @pytest.fixture(scope="module")
                def holds_it():
                    with pytest.MonkeyPatch.context() as patch:
                        patch.setenv("GUARD_PROBE_HELD", "held for the module")
                        yield


                def test_leaks_then_requests_a_fixture(request):
                    os.environ["GUARD_PROBE_HELD"] = "leaked-value-3671"
                    request.getfixturevalue("holds_it")
                """
            ),
            "test_last.py": "def test_last():\n    pass\n",
        },
        ["--junitxml=report.xml"],
    )
    output = completed.stdout

    assert completed.returncode == pytest.ExitCode.TESTS_FAILED, output
    assert _outcomes(tmp_path / "report.xml") == {
        "suite.test_late.test_leaks_then_requests_a_fixture": {"error"},
        "suite.test_last.test_last": {"error"},
    }, output
    assert (
        f"suite/test_late.py::test_leaks_then_requests_a_fixture {TEST_MESSAGE}"
    ) in output, output
    assert RUN_MESSAGE in output, output
    assert output.count("  GUARD_PROBE_HELD: set") == 2, output
    assert "leaked-value-3671" not in output, output


#: A fixture above function scope that leaves a variable changed, at its setup
#: or its teardown. No test is to blame; the run reports every one of them at
#: its last test. ``puts_back`` holds its variables properly.
RUN_MODULES = {
    "test_a_teardown_sets.py": textwrap.dedent(
        """
        import os

        import pytest


        @pytest.fixture(scope="module")
        def sets_on_teardown():
            yield
            os.environ["GUARD_PROBE_TEARDOWN"] = "teardown-value-3671"


        def test_uses_it(sets_on_teardown):
            assert os.environ.get("GUARD_PROBE_TEARDOWN") is None
        """
    ),
    "test_b_teardown_removes.py": textwrap.dedent(
        """
        import os

        import pytest


        @pytest.fixture(scope="module")
        def removes_on_teardown():
            yield
            del os.environ["GUARD_PROBE_PRESET"]


        def test_uses_it(removes_on_teardown):
            pass
        """
    ),
    "test_c_setup_kept.py": textwrap.dedent(
        """
        import os

        import pytest


        @pytest.fixture(scope="module")
        def never_puts_back():
            os.environ["GUARD_PROBE_KEPT"] = "kept-value-3671"
            yield


        @pytest.fixture(scope="module")
        def puts_back():
            with pytest.MonkeyPatch.context() as patch:
                patch.setenv("GUARD_PROBE_HELD", "held for the module")
                patch.setenv("GUARD_PROBE_CHANGED", "held for the module")
                yield


        def test_uses_them(never_puts_back, puts_back):
            assert os.environ.get("GUARD_PROBE_KEPT") == "kept-value-3671"
            assert os.environ.get("GUARD_PROBE_HELD") == "held for the module"


        def test_they_still_hold(never_puts_back, puts_back):
            assert os.environ.get("GUARD_PROBE_CHANGED") == "held for the module"
        """
    ),
    "test_d_class.py": textwrap.dedent(
        """
        import os

        import pytest


        class TestInAClass:
            @pytest.fixture(scope="class")
            def sets_on_teardown(self):
                yield
                os.environ["GUARD_PROBE_CLASS"] = "class-value-3671"

            def test_uses_it(self, sets_on_teardown):
                pass


        def test_after_the_class():
            pass
        """
    ),
    "test_e_session.py": textwrap.dedent(
        """
        import os

        import pytest


        @pytest.fixture(scope="session")
        def sets_on_session_teardown():
            yield
            os.environ["GUARD_PROBE_SESSION_END"] = "session-value-3671"


        def test_uses_it(sets_on_session_teardown):
            pass
        """
    ),
}

RUN_LINES = (
    "  GUARD_PROBE_TEARDOWN: set",
    "  GUARD_PROBE_PRESET: removed",
    "  GUARD_PROBE_KEPT: set",
    "  GUARD_PROBE_CLASS: set",
    "  GUARD_PROBE_SESSION_END: set",
)


@pytest.mark.timeout(SUBPROCESS_TIMEOUT_SECONDS + 15)
@pytest.mark.parametrize("workers", [0, 2], ids=["serial", "xdist"])
def test_a_fixture_that_leaves_the_run_changed_fails_its_last_test(
    tmp_path, inherited_environment, workers
):
    completed = _run_suite(
        tmp_path, RUN_MODULES, ["--junitxml=report.xml", *_workers(workers)]
    )
    output = completed.stdout

    assert completed.returncode == pytest.ExitCode.TESTS_FAILED, output
    outcomes = _outcomes(tmp_path / "report.xml")
    assert len(outcomes) == 7, outcomes
    errored = {name for name, tags in outcomes.items() if tags}
    if workers:
        # Each worker reports what its own tests' fixtures left, at its last test.
        assert 1 <= len(errored) <= workers, output
    else:
        assert errored == {"suite.test_e_session.test_uses_it"}, output
        assert output.count(RUN_MESSAGE) == 1, output
    assert TEST_MESSAGE not in output, output
    for line in RUN_LINES:
        assert output.count(line) == 1, output
    # What a fixture puts back is not a leak.
    assert "GUARD_PROBE_HELD" not in output, output
    assert "GUARD_PROBE_CHANGED" not in output, output
    for value in LEAKED_VALUES:
        assert value not in output, output


#: The last test's teardown fails on its own, and a session fixture's teardown
#: leaves a variable set. Both must be reported.
FAILING_LAST_TEARDOWNS = {
    "a fixture raises": (
        """
        import os

        import pytest


        @pytest.fixture(scope="session")
        def leaves_it_set():
            yield
            os.environ["GUARD_PROBE_SESSION_END"] = "session-value-3671"


        @pytest.fixture
        def fails_its_teardown():
            yield
            raise RuntimeError("the fixture's own error 3671")


        def test_last(leaves_it_set, fails_its_teardown):
            pass
        """,
        "the fixture's own error 3671",
    ),
    "the test leaks too": (
        """
        import os

        import pytest


        @pytest.fixture(scope="session")
        def leaves_it_set():
            yield
            os.environ["GUARD_PROBE_SESSION_END"] = "session-value-3671"


        def test_last(leaves_it_set):
            os.environ["GUARD_PROBE_LEAK"] = "leaked-value-3671"
        """,
        "  GUARD_PROBE_LEAK: set",
    ),
}


@pytest.mark.timeout(SUBPROCESS_TIMEOUT_SECONDS + 15)
@pytest.mark.parametrize("case", sorted(FAILING_LAST_TEARDOWNS))
def test_a_run_leak_is_reported_beside_a_failing_last_teardown(tmp_path, case):
    source, own_failure = FAILING_LAST_TEARDOWNS[case]
    completed = _run_suite(
        tmp_path, {"test_last.py": textwrap.dedent(source)}, ["--junitxml=report.xml"]
    )
    output = completed.stdout

    assert completed.returncode == pytest.ExitCode.TESTS_FAILED, output
    assert _outcomes(tmp_path / "report.xml") == {
        "suite.test_last.test_last": {"error"},
    }, output
    assert own_failure in output, output
    assert RUN_MESSAGE in output, output
    assert "  GUARD_PROBE_SESSION_END: set" in output, output


@pytest.mark.timeout(SUBPROCESS_TIMEOUT_SECONDS + 15)
@pytest.mark.parametrize("workers", [0, 2], ids=["serial", "xdist"])
def test_a_run_stopped_early_still_names_its_leak(tmp_path, workers):
    """``-x`` stops on a teardown error. The test it stopped after had a next
    test, so the module fixture is torn down as the session finishes, with no
    test left to fail."""
    completed = _run_suite(
        tmp_path,
        {
            "test_stops.py": textwrap.dedent(
                """
                import os

                import pytest


                @pytest.fixture(scope="module")
                def sets_on_teardown():
                    yield
                    os.environ["GUARD_PROBE_TEARDOWN"] = "teardown-value-3671"


                @pytest.fixture
                def fails_its_teardown():
                    yield
                    raise RuntimeError("the failure that stops the run 3671")


                def test_stops_the_run(sets_on_teardown, fails_its_teardown):
                    pass


                def test_never_runs(sets_on_teardown):
                    pass
                """
            ),
        },
        ["-x", *_workers(workers)],
    )
    output = completed.stdout

    # xdist reports a run it stopped as interrupted.
    expected = {pytest.ExitCode.TESTS_FAILED}
    if workers:
        expected.add(pytest.ExitCode.INTERRUPTED)
    assert completed.returncode in expected, output
    assert "the failure that stops the run 3671" in output, output
    assert "test_never_runs" not in output, output
    # Named once, as a warning: no test is left to fail.
    assert output.count(RUN_MESSAGE) == 1, output
    summary = output.split("warnings summary", 1)[-1]
    assert f"PytestWarning: {RUN_MESSAGE}" in summary, output
    assert "  GUARD_PROBE_TEARDOWN: set" in summary, output
    assert "teardown-value-3671" not in output, output


#: Runs after every test, on each xdist worker and on the controller, which
#: runs no test.
SESSION_FINISH_WRITER = textwrap.dedent(
    """


    import os


    def pytest_sessionfinish(session):
        os.environ["GUARD_PROBE_SESSION_FINISH"] = "after every test"
    """
)


@pytest.mark.timeout(SUBPROCESS_TIMEOUT_SECONDS + 15)
@pytest.mark.parametrize("workers", [0, 2], ids=["serial", "xdist"])
def test_code_after_the_last_test_is_not_compared(tmp_path, workers):
    completed = _run_suite(
        tmp_path,
        {"test_passes.py": "def test_passes():\n    pass\n"},
        _workers(workers),
        fixtures=SESSION_FINISH_WRITER,
    )
    output = completed.stdout

    assert completed.returncode == pytest.ExitCode.OK, output
    assert "GUARD_PROBE_SESSION_FINISH" not in output, output


@pytest.mark.timeout(SUBPROCESS_TIMEOUT_SECONDS + 15)
def test_an_xfail_mark_does_not_excuse_a_leak(tmp_path):
    completed = _run_suite(
        tmp_path,
        {
            "test_xfail.py": textwrap.dedent(
                """
                import os

                import pytest


                @pytest.fixture(scope="session")
                def leaves_it_set():
                    yield
                    os.environ["GUARD_PROBE_SESSION_END"] = "session-value-3671"


                @pytest.mark.xfail(reason="expected to fail 3671")
                def test_expected_to_fail_and_leaks():
                    os.environ["GUARD_PROBE_LEAK"] = "leaked-value-3671"
                    assert False


                @pytest.mark.xfail(reason="expected to fail 3671")
                def test_expected_to_fail_last(leaves_it_set):
                    assert False
                """
            ),
        },
        ["--junitxml=report.xml"],
    )
    output = completed.stdout

    assert completed.returncode == pytest.ExitCode.TESTS_FAILED, output
    assert _outcomes(tmp_path / "report.xml") == {
        "suite.test_xfail.test_expected_to_fail_and_leaks": {"error"},
        "suite.test_xfail.test_expected_to_fail_last": {"error"},
    }, output
    assert "2 xfailed, 2 errors" in output, output
    assert "  GUARD_PROBE_LEAK: set" in output, output
    assert "  GUARD_PROBE_SESSION_END: set" in output, output


@pytest.mark.timeout(SUBPROCESS_TIMEOUT_SECONDS + 15)
def test_a_module_whose_import_leaks_fails_its_collection(tmp_path):
    completed = _run_suite(
        tmp_path,
        {
            # Its import fails on its own; that error must survive the leak report.
            "test_broken.py": textwrap.dedent(
                """
                import os

                os.environ["GUARD_PROBE_BROKEN"] = "broken-value-3671"
                raise RuntimeError("the module's own error 3671")
                """
            ),
            "test_probe.py": textwrap.dedent(
                """
                import os

                os.environ["GUARD_PROBE_IMPORT"] = "import-value-3671"


                def test_probe():
                    pass
                """
            ),
            # Collected after both leaking modules; its import would fail if
            # either leak had not been put back.
            "test_zz_after.py": textwrap.dedent(
                """
                import os

                assert os.environ.get("GUARD_PROBE_IMPORT") is None
                assert os.environ.get("GUARD_PROBE_BROKEN") is None


                def test_after():
                    pass
                """
            ),
        },
        # The leak is reported in the error itself, not a captured section.
        ["--show-capture=no"],
    )
    output = completed.stdout

    assert completed.returncode == pytest.ExitCode.INTERRUPTED, output
    assert "Importing suite/test_probe.py left os.environ changed" in output, output
    assert "  GUARD_PROBE_IMPORT: set" in output, output
    assert "the module's own error 3671" in output, output
    assert "Importing suite/test_broken.py left os.environ changed" in output, output
    assert "  GUARD_PROBE_BROKEN: set" in output, output
    assert "2 errors during collection" in output, output
    assert "import-value-3671" not in output, output


def test_this_suite_runs_the_guard_as_a_plugin(pytestconfig, request):
    """Registered by ``tests/conftest.py``; its fixture is every test's first."""
    assert pytestconfig.pluginmanager.get_plugin(PLUGIN_NAME) is environment_guard
    function_scoped = [
        name for name in request.fixturenames
        if name in request._arg2fixturedefs
        and request._arg2fixturedefs[name][-1].scope == "function"
    ]
    assert function_scoped[0] == "_environment_guard", function_scoped
    assert function_scoped.index("monkeypatch") > 0, function_scoped


def test_restore_environment_puts_back_every_kind_of_change(monkeypatch):
    monkeypatch.setenv("GUARD_UNIT_REMOVED", "kept")
    monkeypatch.setenv("GUARD_UNIT_CHANGED", "before")
    expected = snapshot_environment()

    os.environ["GUARD_UNIT_SET"] = "new-value-3671"
    del os.environ["GUARD_UNIT_REMOVED"]
    os.environ["GUARD_UNIT_CHANGED"] = "after-value-3671"
    changes = restore_environment(expected)

    assert changes == [
        "  GUARD_UNIT_CHANGED: changed",
        "  GUARD_UNIT_REMOVED: removed",
        "  GUARD_UNIT_SET: set",
    ]
    assert snapshot_environment() == expected
    assert restore_environment(expected) == []


def test_restore_environment_ignores_the_variable_pytest_rewrites_per_phase(
    monkeypatch,
):
    expected = snapshot_environment()
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "another phase")

    assert restore_environment(expected) == []
    assert os.environ["PYTEST_CURRENT_TEST"] == "another phase"


def test_unset_environment_undoes_a_write_to_an_absent_variable():
    assert os.environ.get("GUARD_UNIT_ABSENT") is None
    with pytest.MonkeyPatch.context() as patch:
        unset_environment(patch, "GUARD_UNIT_ABSENT")
        os.environ["GUARD_UNIT_ABSENT"] = "written by the code under test"

    assert os.environ.get("GUARD_UNIT_ABSENT") is None


def test_unset_environment_removes_a_variable_and_restores_it(monkeypatch):
    monkeypatch.setenv("GUARD_UNIT_PRESENT", "before")
    with pytest.MonkeyPatch.context() as patch:
        unset_environment(patch, "GUARD_UNIT_PRESENT")
        assert os.environ.get("GUARD_UNIT_PRESENT") is None
        os.environ["GUARD_UNIT_PRESENT"] = "written by the code under test"

    assert os.environ.get("GUARD_UNIT_PRESENT") == "before"


@pytest.mark.parametrize("run", ["first", "second"])
def test_each_test_starts_without_an_a2a_transport_key(run):
    """The suite fixture drops the key an agent boot publishes, so the second
    run starts as the first did rather than with the first run's key."""
    assert os.environ.get(A2A_TRANSPORT_KEY_ENV) is None

    key = ensure_a2a_transport_key()

    assert os.environ[A2A_TRANSPORT_KEY_ENV] == key
