"""A hang pytest-timeout cannot end becomes a named failure with a stack dump (#3669).

Each case runs a child pytest under xdist, as CI does, on probe tests that
reproduce the hangs ``tests/shared/hang_backstop.py`` exists for. The child
loads only the backstop and the two plugins it needs, with a shrunken grace,
so a case costs seconds rather than the suite's 60s timeout plus its grace.
"""

import os
from pathlib import Path
import re
import subprocess
import sys
import textwrap

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]

#: The child's per-test timeout and backstop grace, in seconds.
CHILD_TIMEOUT = 1
CHILD_GRACE = 1.0
#: How long a probe's hang lasts: far past both, so only a timer can end it.
HANG_SECONDS = 600
#: A liveness bound on the child, not a measurement: a healthy child takes a
#: few seconds, and without the backstop it would run for HANG_SECONDS.
SUBPROCESS_TIMEOUT_SECONDS = 120

CHILD_CONFTEST = textwrap.dedent(
    f"""
    import tests.shared.hang_backstop as hang_backstop
    from tests.shared.hang_backstop import (  # noqa: F401 - exposed as pytest hooks
        pytest_enter_pdb,
        pytest_exception_interact,
        pytest_runtest_protocol,
    )

    hang_backstop.GRACE_SECONDS = {CHILD_GRACE!r}


    def pytest_configure(config):
        hang_backstop.configure(config)
    """
)

PROBES = textwrap.dedent(
    f"""
    import time

    import pytest


    @pytest.fixture
    def hanging_teardown():
        yield
        time.sleep({HANG_SECONDS})


    def test_timeout_is_swallowed():
        while True:
            try:
                time.sleep({HANG_SECONDS})
            except BaseException:
                continue


    def test_teardown_hangs_after_the_timeout(hanging_teardown):
        time.sleep({HANG_SECONDS})


    def test_teardown_hangs_after_a_failure(hanging_teardown):
        assert False, "an ordinary failure"


    def test_ordinary_timeout():
        time.sleep({HANG_SECONDS})


    def test_passes():
        pass
    """
)

HANGS = (
    "test_timeout_is_swallowed",
    "test_teardown_hangs_after_the_timeout",
    "test_teardown_hangs_after_a_failure",
)


def _run_child(
    tmp_path: Path, *, workers: int, timeout_plugin: bool = True, probes: str = PROBES,
) -> str:
    """The child's output, stdout and stderr in one stream as in a CI log."""
    (tmp_path / "pytest.ini").write_text("[pytest]\n", encoding="utf-8")
    (tmp_path / "conftest.py").write_text(CHILD_CONFTEST, encoding="utf-8")
    (tmp_path / "test_probes.py").write_text(probes, encoding="utf-8")
    env = os.environ.copy()
    # The package root only: forwarding sys.path lets a package directory
    # shadow a stdlib module in the child.
    env["PYTHONPATH"] = str(PROJECT_ROOT)
    env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    env.pop("PYTEST_PLUGINS", None)
    env.pop("PYTEST_ADDOPTS", None)
    completed = subprocess.run(
        [
            sys.executable, "-m", "pytest", "test_probes.py",
            "-c", str(tmp_path / "pytest.ini"), "--rootdir", str(tmp_path),
            "-p", "xdist", "-p", "no:cacheprovider", "-n", str(workers), "-v", "--tb=short",
            *(["-p", "timeout", f"--timeout={CHILD_TIMEOUT}"] if timeout_plugin else []),
        ],
        cwd=tmp_path,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=SUBPROCESS_TIMEOUT_SECONDS,
    )
    return f"exit status {completed.returncode}\n{completed.stdout}"


@pytest.fixture(scope="module")
def xdist_output(tmp_path_factory):
    return _run_child(tmp_path_factory.mktemp("hang_backstop"), workers=2)


@pytest.mark.timeout(SUBPROCESS_TIMEOUT_SECONDS + 15)
def test_each_hang_is_a_named_failure_with_its_stack_and_the_run_finishes(xdist_output):
    output = xdist_output

    assert output.startswith("exit status 1\n"), output
    for name in HANGS:
        # xdist names the test whose worker the backstop ended.
        assert re.search(
            rf"worker 'gw\d+' crashed while running '[^']*::{name}'", output,
        ), (name, output)
    # faulthandler's dump of every thread, down to the frame that hung.
    assert "Timeout (" in output, output
    assert "in test_timeout_is_swallowed" in output, output
    assert "in hanging_teardown" in output, output
    # The run went on with replacement workers and printed its summary.
    assert "replacing crashed worker" in output, output
    assert re.search(r"=+ .*\d+ failed, 1 passed.* in [\d.]+s", output), output


@pytest.mark.timeout(SUBPROCESS_TIMEOUT_SECONDS + 15)
def test_an_ordinary_timeout_keeps_its_report_and_its_worker(xdist_output):
    output = xdist_output

    assert re.search(
        r"_ test_ordinary_timeout _+\n(?:.*\n)*?E +Failed: Timeout", output,
    ), output
    assert "crashed while running 'test_probes.py::test_ordinary_timeout'" not in output, output
    assert re.search(r"PASSED \S*::test_passes", output), output


@pytest.mark.timeout(SUBPROCESS_TIMEOUT_SECONDS + 15)
def test_without_xdist_the_hang_ends_the_run_with_its_stack(tmp_path):
    output = _run_child(tmp_path, workers=0)

    # The first hang ends the process: its name is the line the dump follows.
    assert output.startswith("exit status 1\n"), output
    assert re.search(r"::test_timeout_is_swallowed Timeout \(", output), output
    assert "in test_timeout_is_swallowed" in output, output


@pytest.mark.timeout(SUBPROCESS_TIMEOUT_SECONDS + 15)
def test_without_pytest_timeout_loaded_the_backstop_stays_out_of_the_way(tmp_path):
    """No timeouts to back up, and their settings cannot be read: nothing arms."""
    output = _run_child(
        tmp_path, workers=2, timeout_plugin=False,
        probes="def test_passes():\n    pass\n",
    )

    assert output.startswith("exit status 0\n"), output
    assert "1 passed" in output, output
