"""A child pytest takes nothing from the session that starts it (#3669)."""

import pytest

from tests.shared.child_pytest import (
    CI_VARIABLES,
    PROJECT_ROOT,
    child_pytest_env,
    run_child_pytest,
)

#: A liveness bound on the child, not a measurement: it runs one small test.
SUBPROCESS_TIMEOUT_SECONDS = 120


def test_the_child_inherits_no_pytest_variable_of_the_session(monkeypatch):
    """The child's plugins, options and xdist role are its command line's alone."""
    monkeypatch.setenv("PYTEST_ADDOPTS", "-n 4 --timeout=0")
    monkeypatch.setenv("PYTEST_PLUGINS", "pytest_cov")
    monkeypatch.setenv("PYTEST_XDIST_WORKER", "gw3")
    monkeypatch.setenv("PYTEST_XDIST_WORKER_COUNT", "4")
    monkeypatch.setenv("PYTEST_TIMEOUT", "0")

    env = child_pytest_env()

    assert {name for name in env if name.startswith("PYTEST_")} == {
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD",
    }, sorted(env)
    assert env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] == "1"


@pytest.mark.timeout(SUBPROCESS_TIMEOUT_SECONDS + 15)
def test_the_child_does_not_run_as_ci(tmp_path, monkeypatch):
    """In CI pytest leaves summary lines untruncated, so every multi-line error
    appears twice in the child's output: once in full and once in the summary
    (#3671)."""
    for name in CI_VARIABLES:
        monkeypatch.setenv(name, "true")

    completed = run_child_pytest(
        tmp_path,
        conftest="",
        test_file="test_ci.py",
        source=(
            "from _pytest.compat import running_on_ci\n\n\n"
            "def test_not_on_ci():\n"
            "    assert not running_on_ci()\n"
        ),
        args=["test_ci.py"],
        timeout=SUBPROCESS_TIMEOUT_SECONDS,
    )

    assert completed.returncode == pytest.ExitCode.OK, completed.stdout


def test_the_child_imports_from_the_package_root_only(monkeypatch):
    monkeypatch.setenv("PYTHONPATH", "/somewhere/else")

    assert child_pytest_env()["PYTHONPATH"] == str(PROJECT_ROOT)
