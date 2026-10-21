"""A hang pytest-timeout cannot end becomes a named failure with a stack dump (#3669).

Each case runs a child pytest, with and without xdist, on probe tests that
reproduce the hangs ``tests/shared/hang_backstop.py`` exists for. The child
loads only the backstop and the plugins it names, with a shrunken grace, so a
case costs seconds rather than the suite's 60s timeout plus its grace. It
takes nothing from the session running this file (``tests/shared/child_pytest.py``).

Most probes hang the way pytest-timeout's ``signal`` method lets a test hang,
so their cases need that method, which Windows lacks. A teardown that hangs
after a failure is unbounded under either method, so its cases run everywhere
under the ``thread`` method too.
"""

import re
import signal
import textwrap

import pytest

from tests.shared.child_pytest import run_child_pytest

#: The child's per-test timeout and backstop grace, in seconds.
CHILD_TIMEOUT = 1
CHILD_GRACE = 1.0
#: How long a probe's hang lasts: far past both, so only a timer can end it.
HANG_SECONDS = 600
#: A liveness bound on the child, not a measurement: a healthy child takes a
#: few seconds, and without the backstop it would run for HANG_SECONDS.
SUBPROCESS_TIMEOUT_SECONDS = 120

#: pytest-timeout's ``signal`` method arms ``signal.SIGALRM`` with
#: ``signal.setitimer``. Without them (Windows) it fails setting its timer, and
#: pytest-timeout's default there is ``thread``.
HAVE_SIGNAL_METHOD = hasattr(signal, "SIGALRM") and hasattr(signal, "setitimer")
requires_signal_method = pytest.mark.skipif(
    not HAVE_SIGNAL_METHOD,
    reason="pytest-timeout's signal method needs signal.SIGALRM and "
    "signal.setitimer, which this platform (Windows) lacks",
)

CHILD_CONFTEST = textwrap.dedent(
    f"""
    import tests.shared.hang_backstop as hang_backstop
    from tests.shared.hang_backstop import (  # noqa: F401 - exposed as pytest hooks
        pytest_configure_node,
        pytest_enter_pdb,
        pytest_exception_interact,
        pytest_handlecrashitem,
        pytest_runtest_protocol,
    )

    hang_backstop.GRACE_SECONDS = {CHILD_GRACE!r}


    def pytest_configure(config):
        hang_backstop.configure(config)
    """
)

PROBES = textwrap.dedent(
    f"""
    import os
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


    def test_worker_exits_by_itself():
        os._exit(3)


    def test_passes():
        pass
    """
)

#: Each probe the backstop must end, and the frame its dump must show hung.
HANGS = {
    "test_timeout_is_swallowed": "test_timeout_is_swallowed",
    "test_teardown_hangs_after_the_timeout": "hanging_teardown",
    "test_teardown_hangs_after_a_failure": "hanging_teardown",
}
#: The one hang either timeout method leaves to the backstop: both cancel their
#: timer when a phase fails.
HANG_AFTER_A_FAILURE = "test_teardown_hangs_after_a_failure"


def _run_child(
    tmp_path,
    *,
    workers: int,
    method: str | None = "signal",
    probes: str = PROBES,
    select: tuple[str, ...] = (),
) -> str:
    """The child's output, stdout and stderr in one stream as in a CI log.

    ``method`` is the pytest-timeout method; None runs without pytest-timeout.
    ``select`` names the probes to run, all of them when empty.
    """
    completed = run_child_pytest(
        tmp_path,
        conftest=CHILD_CONFTEST,
        test_file="test_probes.py",
        source=probes,
        args=[
            *(
                [f"test_probes.py::{name}" for name in select]
                if select else ["test_probes.py"]
            ),
            "-p", "xdist", "-n", str(workers), "-v", "--tb=short",
            *(
                ["-p", "timeout", f"--timeout={CHILD_TIMEOUT}", f"--timeout-method={method}"]
                if method is not None else []
            ),
        ],
        timeout=SUBPROCESS_TIMEOUT_SECONDS,
    )
    return f"exit status {completed.returncode}\n{completed.stdout}"


def _hung_frame(frame: str) -> re.Pattern[str]:
    """One whole faulthandler line placing a thread in ``frame`` of the probes.

    A dump interleaved with another process's splits this line apart.
    """
    return re.compile(rf'^  File "[^"\n]*test_probes\.py", line \d+ in {frame}$', re.M)


def _crash_reports(output: str) -> dict[str, list[str]]:
    """The FAILURES section's crash reports, by the probe each names."""
    assert " FAILURES " in output, output
    failures = output.split(" FAILURES ", 1)[1].split("\n=", 1)[0]
    reports: dict[str, list[str]] = {}
    for section in re.split(r"^_+ .* _+$", failures, flags=re.M):
        named = re.search(r"^worker 'gw\d+' crashed while running '[^']*::(\w+)'$", section, re.M)
        if named:
            reports.setdefault(named.group(1), []).append(section)
    return reports


@pytest.fixture(scope="module")
def xdist_output(tmp_path_factory):
    return _run_child(tmp_path_factory.mktemp("hang_backstop"), workers=2)


@requires_signal_method
@pytest.mark.timeout(SUBPROCESS_TIMEOUT_SECONDS + 15)
def test_each_hang_is_a_named_failure_with_its_stack_and_the_run_finishes(xdist_output):
    """Under xdist each hang's crash report carries its own whole dump.

    Two workers start hanging together, so their deadlines expire within
    milliseconds. Written to the stderr they share, their dumps interleave and
    neither hung frame can be read: half the runs of this child failed so.
    """
    output = xdist_output

    assert output.startswith("exit status 1\n"), output
    reports = _crash_reports(output)
    for name, frame in HANGS.items():
        # xdist names the test whose worker the backstop ended, once.
        assert len(reports.get(name, [])) == 1, (name, output)
        report = reports[name][0]
        # faulthandler's dump of every thread, down to the frame that hung.
        assert report.count("Timeout (") == 1, (name, report)
        assert _hung_frame(frame).search(report), (name, report)
    # The run went on with replacement workers and printed its summary.
    assert "replacing crashed worker" in output, output
    assert re.search(r"=+ .*\d+ failed, 1 passed.* in [\d.]+s", output), output


@requires_signal_method
@pytest.mark.timeout(SUBPROCESS_TIMEOUT_SECONDS + 15)
def test_a_crash_the_backstop_did_not_cause_keeps_its_report(xdist_output):
    reports = _crash_reports(xdist_output)

    assert len(reports.get("test_worker_exits_by_itself", [])) == 1, xdist_output
    assert "Timeout (" not in reports["test_worker_exits_by_itself"][0], xdist_output


@requires_signal_method
@pytest.mark.timeout(SUBPROCESS_TIMEOUT_SECONDS + 15)
def test_an_ordinary_timeout_keeps_its_report_and_its_worker(xdist_output):
    output = xdist_output

    assert re.search(
        r"_ test_ordinary_timeout _+\n(?:.*\n)*?E +Failed: Timeout", output,
    ), output
    assert "test_ordinary_timeout" not in _crash_reports(output), output
    assert re.search(r"PASSED \S*::test_passes", output), output


@requires_signal_method
@pytest.mark.timeout(SUBPROCESS_TIMEOUT_SECONDS + 15)
@pytest.mark.parametrize("name", HANGS)
def test_without_xdist_each_hang_ends_the_run_with_its_stack(tmp_path, name):
    """A serial run has no controller to report a crash: its process ends.

    So each probe runs alone. The ``-v`` line naming the item is the line the
    dump follows.
    """
    output = _run_child(tmp_path, workers=0, select=(name,))

    assert output.startswith("exit status 1\n"), output
    # A teardown hang follows its call phase's FAILED, padded to the width.
    assert re.search(rf"::{name} (?:FAILED +\[ *\d+%\])?Timeout \(", output), output
    assert output.count("Timeout (0:") == 1, output
    assert _hung_frame(HANGS[name]).search(output), output


@pytest.mark.timeout(SUBPROCESS_TIMEOUT_SECONDS + 15)
def test_under_the_thread_method_a_hang_after_a_failure_is_a_named_failure_with_its_stack(
    tmp_path,
):
    """The thread method, Windows' only one, leaves this hang to the backstop too.

    Its timer, when it fires, dumps the stacks and exits by itself: the
    backstop is what ends a teardown that hangs once that timer is cancelled.
    """
    output = _run_child(
        tmp_path, workers=2, method="thread",
        select=(HANG_AFTER_A_FAILURE, "test_passes"),
    )

    assert output.startswith("exit status 1\n"), output
    reports = _crash_reports(output)
    assert set(reports) == {HANG_AFTER_A_FAILURE}, output
    (report,) = reports[HANG_AFTER_A_FAILURE]
    assert report.count("Timeout (") == 1, report
    assert _hung_frame(HANGS[HANG_AFTER_A_FAILURE]).search(report), report
    # The probe fails twice: its call phase's assertion, then its worker's crash.
    assert re.search(r"=+ 2 failed, 1 passed in [\d.]+s", output), output


@pytest.mark.timeout(SUBPROCESS_TIMEOUT_SECONDS + 15)
def test_under_the_thread_method_without_xdist_a_hang_after_a_failure_ends_the_run_with_its_stack(
    tmp_path,
):
    output = _run_child(
        tmp_path, workers=0, method="thread", select=(HANG_AFTER_A_FAILURE,),
    )

    assert output.startswith("exit status 1\n"), output
    assert re.search(
        rf"::{HANG_AFTER_A_FAILURE} FAILED +\[ *\d+%\]Timeout \(", output,
    ), output
    assert output.count("Timeout (0:") == 1, output
    assert _hung_frame(HANGS[HANG_AFTER_A_FAILURE]).search(output), output


@pytest.mark.timeout(SUBPROCESS_TIMEOUT_SECONDS + 15)
def test_without_pytest_timeout_loaded_the_backstop_stays_out_of_the_way(tmp_path):
    """No timeouts to back up, and their settings cannot be read: nothing arms."""
    output = _run_child(
        tmp_path, workers=2, method=None,
        probes="def test_passes():\n    pass\n",
    )

    assert output.startswith("exit status 0\n"), output
    assert "1 passed" in output, output
