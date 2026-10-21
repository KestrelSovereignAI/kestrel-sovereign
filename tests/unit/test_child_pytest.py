"""A child pytest takes nothing from the session that starts it (#3669)."""

from tests.shared.child_pytest import PROJECT_ROOT, child_pytest_env


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


def test_the_child_imports_from_the_package_root_only(monkeypatch):
    monkeypatch.setenv("PYTHONPATH", "/somewhere/else")

    assert child_pytest_env()["PYTHONPATH"] == str(PROJECT_ROOT)
