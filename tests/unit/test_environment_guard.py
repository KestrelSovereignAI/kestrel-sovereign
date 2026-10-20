"""A test, fixture or test module that leaves ``os.environ`` changed is put back and fails (#3671)."""

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
#: repository's: the session node is at the rootdir, so this directory's
#: conftest hooks never see a session-scoped fixture.
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
    "class-value-3671", "session-value-3671",
)

PROBE = textwrap.dedent(
    """
    import os

    import pytest

    from tests.shared.environment_guard import unset_environment


    @pytest.fixture(scope="module")
    def module_setting():
        os.environ["GUARD_PROBE_MODULE"] = "held for the module"
        yield
        del os.environ["GUARD_PROBE_MODULE"]


    @pytest.fixture(scope="module")
    def late_module_setting():
        os.environ["GUARD_PROBE_LATE"] = "held for the module"
        yield
        del os.environ["GUARD_PROBE_LATE"]


    @pytest.fixture(scope="session")
    def session_setting():
        os.environ["GUARD_PROBE_SESSION"] = "held for the session"
        yield
        del os.environ["GUARD_PROBE_SESSION"]


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


    def test_sees_none_of_them():
        assert "GUARD_PROBE_LEAK" not in os.environ
        assert os.environ["GUARD_PROBE_PRESET"] == "preset-value-3671"
        assert os.environ["GUARD_PROBE_CHANGED"] == "original-value-3671"


    def test_monkeypatch_is_not_a_leak(monkeypatch):
        monkeypatch.setenv("GUARD_PROBE_SET", "by the test")
        monkeypatch.delenv("GUARD_PROBE_PRESET")


    def test_a_module_fixture_is_not_a_leak(module_setting):
        assert os.environ["GUARD_PROBE_MODULE"] == "held for the module"


    def test_a_module_fixture_requested_during_the_test_is_not_a_leak(request):
        request.getfixturevalue("late_module_setting")


    def test_that_module_fixture_still_holds(late_module_setting):
        assert os.environ["GUARD_PROBE_LATE"] == "held for the module"


    def test_a_session_fixture_requested_during_the_test_is_not_a_leak(request):
        request.getfixturevalue("session_setting")


    def test_that_session_fixture_still_holds(session_setting):
        assert os.environ["GUARD_PROBE_SESSION"] == "held for the session"


    def test_unset_environment_owns_a_later_write(monkeypatch):
        unset_environment(monkeypatch, "GUARD_PROBE_WRITTEN_BY_CODE")
        os.environ["GUARD_PROBE_WRITTEN_BY_CODE"] = "as code under test would"
    """
)

#: No autouse fixture. ``test_plain_leak`` has no fixture at all, so only the
#: post-setup fallback records it; ``leaky`` is the first and only fixture of
#: its test, so only a record taken before that fixture runs sees its write.
PLAIN = textwrap.dedent(
    """
    import os

    import pytest


    @pytest.fixture
    def leaky():
        os.environ["GUARD_PROBE_FIXTURE"] = "fixture-value-3671"


    def test_plain_leak():
        os.environ["GUARD_PROBE_PLAIN"] = "plain-value-3671"


    def test_plain_fixture_leak(leaky):
        pass


    def test_plain_after():
        assert "GUARD_PROBE_PLAIN" not in os.environ
        assert "GUARD_PROBE_FIXTURE" not in os.environ
    """
)


def _outcomes(report: Path) -> dict[str, set[str]]:
    """Each test's name mapped to the junit elements (error, failure) it carries."""
    outcomes: dict[str, set[str]] = {}
    for case in ElementTree.parse(report).iter("testcase"):
        tags = outcomes.setdefault(case.get("name"), set())
        tags.update(child.tag for child in case if child.tag in ("error", "failure"))
    return outcomes


def _run_suite(
    tmp_path: Path, modules: dict[str, str], args: list[str]
) -> subprocess.CompletedProcess:
    """Run ``modules``, in order, from the child's ``suite`` directory."""
    suite = tmp_path / SUITE
    suite.mkdir()
    (suite / "conftest.py").write_text(CHILD_CONFTEST, encoding="utf-8")
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


def _run_items(tmp_path: Path, workers: int) -> subprocess.CompletedProcess:
    distribution = ["--dist", "loadfile"] if workers else []
    return _run_suite(
        tmp_path,
        {"test_probe.py": PROBE, "test_plain.py": PLAIN},
        ["--junitxml=report.xml", "-p", "xdist", "-n", str(workers), *distribution],
    )


@pytest.mark.timeout(SUBPROCESS_TIMEOUT_SECONDS + 15)
@pytest.mark.parametrize("workers", [0, 2], ids=["serial", "xdist"])
def test_a_leaking_test_is_put_back_and_fails(tmp_path, monkeypatch, workers):
    # Inherited by the child.
    monkeypatch.setenv("GUARD_PROBE_PRESET", "preset-value-3671")
    monkeypatch.setenv("GUARD_PROBE_CHANGED", "original-value-3671")

    completed = _run_items(tmp_path, workers)
    output = completed.stdout

    assert completed.returncode == pytest.ExitCode.TESTS_FAILED, output
    outcomes = _outcomes(tmp_path / "report.xml")
    assert {name for name, tags in outcomes.items() if tags} == {
        "test_sets", "test_removes", "test_changes", "test_plain_leak",
        "test_plain_fixture_leak",
    }, output
    assert outcomes["test_sees_none_of_them"] == set(), output
    assert outcomes["test_that_module_fixture_still_holds"] == set(), output
    assert outcomes["test_that_session_fixture_still_holds"] == set(), output
    assert outcomes["test_plain_after"] == set(), output
    assert len(outcomes) == 14, outcomes
    assert "suite/test_probe.py::test_sets left os.environ changed" in output, output
    for line in (
        "  GUARD_PROBE_LEAK: set",
        "  GUARD_PROBE_PRESET: removed",
        "  GUARD_PROBE_CHANGED: changed",
        "  GUARD_PROBE_PLAIN: set",
        "  GUARD_PROBE_FIXTURE: set",
    ):
        assert line in output, output
    for value in LEAKED_VALUES:
        assert value not in output, output


#: Each module's fixture is torn down as the next module starts, so the module
#: after it shows whether its leak was put back. ``test_z_after`` names each of
#: them; the session fixture's teardown is the run's last, at ``test_zz_last``.
TEARDOWN_MODULES = {
    "test_a_teardown_sets.py": textwrap.dedent(
        """
        import os

        import pytest


        @pytest.fixture(scope="module")
        def sets_on_teardown():
            yield
            os.environ["GUARD_PROBE_TEARDOWN"] = "teardown-value-3671"


        def test_uses_it(sets_on_teardown):
            assert "GUARD_PROBE_TEARDOWN" not in os.environ
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
            assert os.environ["GUARD_PROBE_KEPT"] == "kept-value-3671"
            assert os.environ["GUARD_PROBE_HELD"] == "held for the module"


        def test_they_still_hold(never_puts_back, puts_back):
            assert os.environ["GUARD_PROBE_CHANGED"] == "held for the module"
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
            assert "GUARD_PROBE_CLASS" not in os.environ
        """
    ),
    "test_z_after.py": textwrap.dedent(
        """
        import os


        def test_after():
            assert "GUARD_PROBE_TEARDOWN" not in os.environ
            assert os.environ["GUARD_PROBE_PRESET"] == "preset-value-3671"
            assert "GUARD_PROBE_KEPT" not in os.environ
            assert "GUARD_PROBE_HELD" not in os.environ
            assert os.environ["GUARD_PROBE_CHANGED"] == "original-value-3671"
            assert "GUARD_PROBE_CLASS" not in os.environ
        """
    ),
    "test_zz_last.py": textwrap.dedent(
        """
        import pytest


        @pytest.fixture(scope="session")
        def sets_on_session_teardown():
            yield
            import os

            os.environ["GUARD_PROBE_SESSION_END"] = "session-value-3671"


        def test_last(sets_on_session_teardown):
            pass
        """
    ),
}


@pytest.mark.timeout(SUBPROCESS_TIMEOUT_SECONDS + 15)
def test_a_fixture_whose_teardown_leaks_is_put_back_and_fails(tmp_path, monkeypatch):
    # Inherited by the child.
    monkeypatch.setenv("GUARD_PROBE_PRESET", "preset-value-3671")
    monkeypatch.setenv("GUARD_PROBE_CHANGED", "original-value-3671")

    completed = _run_suite(tmp_path, TEARDOWN_MODULES, ["--junitxml=report.xml"])
    output = completed.stdout

    assert completed.returncode == pytest.ExitCode.TESTS_FAILED, output
    report = ElementTree.parse(tmp_path / "report.xml")
    outcomes = {
        f"{case.get('classname')}.{case.get('name')}": {
            child.tag for child in case if child.tag in ("error", "failure")
        }
        for case in report.iter("testcase")
    }
    # Only the test whose teardown retired a leaking fixture carries an error;
    # every test after one passes, so each leak was put back.
    assert outcomes == {
        "suite.test_a_teardown_sets.test_uses_it": {"error"},
        "suite.test_b_teardown_removes.test_uses_it": {"error"},
        "suite.test_c_setup_kept.test_uses_them": set(),
        "suite.test_c_setup_kept.test_they_still_hold": {"error"},
        "suite.test_d_class.TestInAClass.test_uses_it": {"error"},
        "suite.test_d_class.test_after_the_class": set(),
        "suite.test_z_after.test_after": set(),
        "suite.test_zz_last.test_last": {"error"},
    }, output
    for fixture, owner, line in (
        ("module-scoped fixture 'sets_on_teardown'", "suite/test_a_teardown_sets.py",
         "  GUARD_PROBE_TEARDOWN: set"),
        ("module-scoped fixture 'removes_on_teardown'",
         "suite/test_b_teardown_removes.py", "  GUARD_PROBE_PRESET: removed"),
        ("module-scoped fixture 'never_puts_back'", "suite/test_c_setup_kept.py",
         "  GUARD_PROBE_KEPT: set"),
        ("class-scoped fixture 'sets_on_teardown'",
         "suite/test_d_class.py::TestInAClass", "  GUARD_PROBE_CLASS: set"),
        ("session-scoped fixture 'sets_on_session_teardown'", "the session",
         "  GUARD_PROBE_SESSION_END: set"),
    ):
        assert (
            f"Tearing down the {fixture} of {owner} left os.environ changed"
        ) in output, output
        assert line in output, output
    # What a fixture puts back is not a leak.
    assert "GUARD_PROBE_HELD" not in output, output
    assert "GUARD_PROBE_CHANGED" not in output, output
    for value in LEAKED_VALUES:
        assert value not in output, output


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

                assert "GUARD_PROBE_IMPORT" not in os.environ
                assert "GUARD_PROBE_BROKEN" not in os.environ


                def test_after():
                    pass
                """
            ),
        },
        [],
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


def test_this_suite_runs_the_guard_as_a_plugin(pytestconfig):
    """Registered by ``tests/conftest.py``, so its hooks see the session too."""
    assert pytestconfig.pluginmanager.get_plugin(PLUGIN_NAME) is environment_guard


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
    assert "GUARD_UNIT_ABSENT" not in os.environ
    with pytest.MonkeyPatch.context() as patch:
        unset_environment(patch, "GUARD_UNIT_ABSENT")
        os.environ["GUARD_UNIT_ABSENT"] = "written by the code under test"

    assert "GUARD_UNIT_ABSENT" not in os.environ


def test_unset_environment_removes_a_variable_and_restores_it(monkeypatch):
    monkeypatch.setenv("GUARD_UNIT_PRESENT", "before")
    with pytest.MonkeyPatch.context() as patch:
        unset_environment(patch, "GUARD_UNIT_PRESENT")
        assert "GUARD_UNIT_PRESENT" not in os.environ
        os.environ["GUARD_UNIT_PRESENT"] = "written by the code under test"

    assert os.environ["GUARD_UNIT_PRESENT"] == "before"


@pytest.mark.parametrize("run", ["first", "second"])
def test_each_test_starts_without_an_a2a_transport_key(run):
    """The suite fixture drops the key an agent boot publishes, so the second
    run starts as the first did rather than with the first run's key."""
    assert A2A_TRANSPORT_KEY_ENV not in os.environ

    key = ensure_a2a_transport_key()

    assert os.environ[A2A_TRANSPORT_KEY_ENV] == key
