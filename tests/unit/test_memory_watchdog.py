"""A test that runs away on memory ends its process by name (#4066).

Each child case runs a child pytest that loads only the watchdog, the way the
root ``tests/conftest.py`` does, and takes nothing from the session running
this file (``tests/shared/child_pytest.py``). Its ceiling is set far below the
suite's, so a runaway probe crosses it in about a second.
"""

import errno
import os
import re
import subprocess
import sys
import textwrap
import threading
import time
from unittest.mock import patch

import psutil
import pytest

from tests.shared.child_pytest import run_child_pytest
from tests.shared.environment_guard import unset_environment
from tests.shared.memory_watchdog import (
    CEILING_VARIABLE,
    EXIT_STATUS,
    PLUGIN_NAME,
    POLL_INTERVAL_SECONDS,
    MemoryWatchdog,
    ceiling_bytes,
)

GIB = 2**30
#: The child's ceiling, in GiB: an idle child pytest holds about 45 MiB.
CHILD_CEILING_GIB = "0.3"
#: A liveness bound on the child, not a measurement: a healthy child takes a
#: few seconds, and one the watchdog never ends holds its probe past this.
SUBPROCESS_TIMEOUT_SECONDS = 30
#: The memory of a GitHub-hosted runner, which the default must stay below.
RUNNER_MEMORY_BYTES = 16 * 10**9

CHILD_CONFTEST = textwrap.dedent(
    """
    from tests.shared.memory_watchdog import register


    def pytest_configure(config):
        register(config)
    """
)

#: The runaway grows about 400 MiB/s with no limit the watchdog could see. It
#: stops at a safety cap far past the child's ceiling and holds it longer than
#: the child's liveness bound, so a child the watchdog does not end is ended by
#: that bound rather than by the host running out of memory.
PROBES = textwrap.dedent(
    f"""
    import time

    import pytest

    from tests.shared.memory_watchdog import PLUGIN_NAME

    CHUNK_BYTES = 8 * 2**20
    SAFETY_CAP_BYTES = 2**30
    HOLD_SECONDS = {SUBPROCESS_TIMEOUT_SECONDS * 2}


    def test_well_behaved(pytestconfig):
        assert pytestconfig.pluginmanager.get_plugin(PLUGIN_NAME).running


    def test_runaway():
        held = []
        while len(held) * CHUNK_BYTES < SAFETY_CAP_BYTES:
            held.append(b"\\x01" * CHUNK_BYTES)
            time.sleep(0.02)
        time.sleep(HOLD_SECONDS)
        pytest.fail("nothing ended this process at its memory ceiling")


    def test_held_past_the_ceiling(pytestconfig):
        # Holds a fixed 384 MiB until the watchdog ends it or its thread dies.
        watchdog = pytestconfig.pluginmanager.get_plugin(PLUGIN_NAME)
        held = [b"\\x01" * CHUNK_BYTES for _ in range(48)]
        deadline = time.monotonic() + HOLD_SECONDS
        while watchdog.running and time.monotonic() < deadline:
            time.sleep(0.05)
        pytest.fail(
            f"this process held {{len(held) * CHUNK_BYTES}} bytes past its ceiling "
            f"and was not ended; watchdog thread running: {{watchdog.running}}"
        )
    """
)

RUNAWAY = "test_probe.py::test_runaway"
HELD = "test_probe.py::test_held_past_the_ceiling"
WATCHDOG_THREAD = "kestrel-test-memory-watchdog"


def _run_child(
    tmp_path, *names: str, workers: int = 0, stderr: int = subprocess.STDOUT,
) -> str:
    """The child's exit status and output, stdout and stderr as in a CI log.

    With ``stderr`` naming another destination, the output is stdout alone.
    """
    completed = run_child_pytest(
        tmp_path,
        conftest=CHILD_CONFTEST,
        test_file="test_probe.py",
        source=PROBES,
        args=[
            *(f"test_probe.py::{name}" for name in names),
            *(["-p", "xdist", "-n", str(workers)] if workers else []),
            "-v", "--tb=short",
        ],
        timeout=SUBPROCESS_TIMEOUT_SECONDS,
        stderr=stderr,
    )
    return f"exit status {completed.returncode}\n{completed.stdout}"


def _reports(output: str, nodeid: str) -> list[int]:
    """The RSS, in bytes, of each watchdog report naming ``nodeid``."""
    return [
        int(rss) for rss in re.findall(
            r"^\[memory watchdog\] pytest process \d+ reached RSS [\d.]+ GiB "
            r"\((\d+) bytes\), past its ceiling of 0\.30 GiB "
            rf"\({CEILING_VARIABLE}\), while running {re.escape(nodeid)}\. "
            rf"Ending it with exit status {EXIT_STATUS} ",
            output, re.M,
        )
    ]


def _runaway_frame() -> re.Pattern[str]:
    """One whole faulthandler line placing a thread in the runaway probe."""
    return re.compile(r'^  File "[^"\n]*test_probe\.py", line \d+ in test_runaway$', re.M)


@pytest.mark.timeout(SUBPROCESS_TIMEOUT_SECONDS + 15)
def test_a_runaway_test_ends_its_process_by_name_with_every_stack(tmp_path, monkeypatch):
    """The test before it passes; the runaway ends the run with the watchdog's status.

    Output capture is on, as in the suite: the report reaches the real stderr.
    """
    monkeypatch.setenv(CEILING_VARIABLE, CHILD_CEILING_GIB)

    output = _run_child(tmp_path, "test_well_behaved", "test_runaway")

    assert output.startswith(f"exit status {EXIT_STATUS}\n"), output
    assert re.search(r"::test_well_behaved PASSED", output), output
    (rss,) = _reports(output, RUNAWAY)
    assert rss > 0.3 * GIB, output
    # Every thread's stack: the main thread's, which the watchdog's is not,
    # down to the probe that ran away.
    assert _runaway_frame().search(output), output


@pytest.mark.timeout(SUBPROCESS_TIMEOUT_SECONDS + 15)
def test_a_report_that_cannot_be_written_still_ends_the_process(tmp_path, monkeypatch):
    """A stderr whose reader has gone fails the report's write, not the ceiling.

    The write raises ``BrokenPipeError``. Were the exit not unconditional, that
    would end only the watchdog's thread, and the test past its ceiling would
    run on, unguarded, to the end of its hold.
    """
    monkeypatch.setenv(CEILING_VARIABLE, CHILD_CEILING_GIB)
    read_end, write_end = os.pipe()
    os.close(read_end)
    try:
        output = _run_child(tmp_path, "test_held_past_the_ceiling", stderr=write_end)
    finally:
        os.close(write_end)

    assert output.startswith(f"exit status {EXIT_STATUS}\n"), output
    # The report went nowhere; the progress line on stdout names the test.
    assert "[memory watchdog]" not in output, output
    assert re.search(rf"^{re.escape(HELD)} $", output, re.M), output


@pytest.mark.timeout(SUBPROCESS_TIMEOUT_SECONDS + 15)
def test_under_xdist_the_runaway_is_a_named_crash_and_the_run_finishes(
    tmp_path, monkeypatch,
):
    """Each worker has its own watchdog; the controller names the item it ended."""
    monkeypatch.setenv(CEILING_VARIABLE, CHILD_CEILING_GIB)

    output = _run_child(tmp_path, "test_well_behaved", "test_runaway", workers=2)

    assert output.startswith("exit status 1\n"), output
    assert re.search(
        rf"^worker 'gw\d+' crashed while running '{re.escape(RUNAWAY)}'$", output, re.M,
    ), output
    assert len(_reports(output, RUNAWAY)) == 1, output
    assert "replacing crashed worker" in output, output
    assert re.search(r"=+ 1 failed, 1 passed.* in [\d.]+s", output), output


@pytest.mark.timeout(SUBPROCESS_TIMEOUT_SECONDS + 15)
def test_with_the_ceiling_unset_a_well_behaved_run_exits_0(tmp_path, monkeypatch):
    unset_environment(monkeypatch, CEILING_VARIABLE)

    output = _run_child(tmp_path, "test_well_behaved")

    assert output.startswith("exit status 0\n"), output
    assert "1 passed" in output, output
    assert "[memory watchdog]" not in output, output


@pytest.mark.timeout(SUBPROCESS_TIMEOUT_SECONDS + 15)
def test_a_malformed_ceiling_refuses_the_run_before_any_test(tmp_path, monkeypatch):
    monkeypatch.setenv(CEILING_VARIABLE, "eight")

    output = _run_child(tmp_path, "test_well_behaved")

    assert output.startswith("exit status 4\n"), output
    assert re.search(
        rf"^ERROR: {CEILING_VARIABLE} must be a finite, positive number of GiB .*, not 'eight'\.",
        output, re.M,
    ), output
    assert "passed" not in output, output


def test_this_process_runs_under_the_watchdog(request):
    """The root conftest started it here, and it knows which test is running.

    This runs in whatever process CI runs it in, so CI runs with it on.
    """
    watchdog = request.config.pluginmanager.get_plugin(PLUGIN_NAME)

    assert isinstance(watchdog, MemoryWatchdog)
    assert watchdog.running
    assert watchdog.nodeid == request.node.nodeid


def test_the_default_ceiling_is_8_gib_below_a_github_runner():
    assert ceiling_bytes({}) == 8 * GIB
    assert ceiling_bytes({}) < RUNNER_MEMORY_BYTES


def test_the_exit_status_is_the_documented_86():
    """TESTING_GUIDE.md names it; it must not read as any other way to end."""
    assert EXIT_STATUS == 86
    assert EXIT_STATUS not in {code.value for code in pytest.ExitCode}
    assert EXIT_STATUS < 128  # not a signal's 128 + N


@pytest.mark.parametrize(
    ("raw", "expected"), [("0.3", int(0.3 * GIB)), ("12", 12 * GIB), (" 2.5 ", int(2.5 * GIB))],
)
def test_the_ceiling_is_read_in_gib(raw, expected):
    assert ceiling_bytes({CEILING_VARIABLE: raw}) == expected


@pytest.mark.parametrize("raw", ["", "eight", "0", "-1", "nan", "inf", "1e300", "1e-12"])
def test_anything_but_a_finite_positive_ceiling_is_refused(raw):
    with pytest.raises(pytest.UsageError, match=re.escape(f"not {raw!r}")):
        ceiling_bytes({CEILING_VARIABLE: raw})


def test_the_rss_read_is_out_of_reach_of_a_test_patching_open():
    """On Linux psutil opens ``/proc/<pid>/statm`` with the builtin ``open``.

    ``tests/unit/test_reanchor_constitution.py`` patches it with a list of
    fake handles; a watchdog reading through psutil would take one of them.
    """
    def refused(*args, **kwargs):
        raise AssertionError("the watchdog's read reached a patched call")

    reader = MemoryWatchdog(GIB, 2).read_rss
    with (
        patch("builtins.open", side_effect=refused) as fake_open,
        patch.object(os, "open", refused),
        patch.object(os, "read", refused),
        patch.object(os, "close", refused),
        patch.object(psutil.Process, "memory_info", refused),
    ):
        rss = reader()

    assert fake_open.call_count == 0
    assert 0 < rss < 64 * GIB


def test_the_running_watchdog_survives_a_test_patching_open(request):
    """The thread keeps polling, and leaves the patch untouched, while it is in place.

    Only the watchdog thread's own calls count: another thread of this
    process may open a file meanwhile.
    """
    watchdog = request.config.pluginmanager.get_plugin(PLUGIN_NAME)
    callers = []

    def missing(*args, **kwargs):
        callers.append(threading.current_thread().name)
        raise FileNotFoundError(args[0] if args else None)

    with patch("builtins.open", side_effect=missing):
        time.sleep(POLL_INTERVAL_SECONDS * 2.5)

    assert WATCHDOG_THREAD not in callers
    assert watchdog.running


def test_a_failed_read_skips_a_poll_and_the_watchdog_goes_on(tmp_path):
    """While a test holds every descriptor, the Linux read fails with EMFILE."""
    reads = []

    def read_rss():
        reads.append(time.monotonic())
        if len(reads) < 3:
            raise OSError(errno.EMFILE, "Too many open files")
        return 1

    with open(tmp_path / "report", "wb") as report:
        watchdog = MemoryWatchdog(GIB, report.fileno())
        watchdog.read_rss = read_rss
        watchdog.start()
        try:
            deadline = time.monotonic() + POLL_INTERVAL_SECONDS * 10
            while len(reads) < 4 and time.monotonic() < deadline:
                time.sleep(POLL_INTERVAL_SECONDS / 5)
            assert watchdog.running
        finally:
            watchdog.stop()

    assert len(reads) >= 4, reads


@pytest.mark.skipif(
    not sys.platform.startswith("linux"),
    reason="only Linux reads the RSS without psutil",
)
def test_on_linux_the_rss_read_agrees_with_psutil():
    reader = MemoryWatchdog(GIB, 2).read_rss

    assert abs(reader() - psutil.Process().memory_info().rss) < 64 * 2**20
