"""End a test that its pytest-timeout did not end, naming it and dumping every stack (#3669).

pytest-timeout's ``signal`` method (its default on POSIX) raises ``Failed`` in
the main thread once, when a test's timeout expires, and it cancels its timer
as soon as any phase of a test fails: its ``pytest_exception_interact`` hook
runs for every failure, not only under ``--pdb``. A hang that outlives that one
exception, or that starts after a failure, stalls its worker with no FAILED
line, no stack and no summary until the runner kills the step. Three such
hangs were reproduced under the CI command:

* the test swallows the exception (an ``except BaseException`` retry loop);
* the exception ends the test body, then a fixture's teardown hangs;
* the exception ends an asyncio test, then closing its loop waits on an
  executor thread that never returns.

The ``thread`` method (its default where ``signal.SIGALRM`` is missing, as on
Windows) dumps the stacks and exits by itself when its timer fires, but it
cancels that timer on a failure too: a hang that starts after a failure is the
backstop's under either method.

This backstop bounds each item's whole runtest protocol (setup, call and
teardown) at the timeout pytest-timeout applies to it, markers included, plus
:data:`GRACE_SECONDS`. An item still running then has every thread's stack
dumped by :mod:`faulthandler`, which then exits the process.

Under xdist the controller reports ``worker 'gwN' crashed while running
'<nodeid>'`` as a failure and starts a replacement worker (up to its
``--max-worker-restart``), so the run goes on to its summary. Each worker
writes its dump to a file of its own, and the controller appends that file to
the worker's crash report, so the named failure carries its stack. Workers
whose items hang together fire within milliseconds of each other, and
faulthandler writes a dump in many small writes: on the stderr the workers
share, two dumps interleave byte by byte and neither stack can be read.
Without xdist the dump goes to the process's real stderr, after the ``-v``
line that names the item.

faulthandler's watchdog is a C thread. It needs no GIL, so it fires even when
the main thread never returns to the interpreter, and ``threading`` cannot see
it, so no test observes an extra thread. Only one such timer exists per
process, and pytest's own faulthandler plugin cancels it whenever a phase
fails, so :func:`pytest_exception_interact` arms it again for the rest of the
item's deadline. Setting ``faulthandler_timeout`` would replace it.
"""

from __future__ import annotations

import faulthandler
import os
from pathlib import Path
import shutil
import tempfile
import time

import pytest
import pytest_timeout
from pytest_timeout import _get_item_settings, is_debugging

#: How long past its timeout an item may still run before the backstop ends
#: the process. This leaves a test that timed out time to tear down, so it
#: still gets an ordinary FAILED report and keeps its worker.
GRACE_SECONDS = 30.0

#: The ``workerinput`` key naming the file an xdist worker writes its dump to.
_DUMP_PATH_INPUT = "hang_backstop_dump_path"

#: The descriptor this process's dump is written to.
_DUMP_FD_KEY = pytest.StashKey[int]()
#: On the xdist controller, the directory holding its workers' dump files.
_DUMP_DIR_KEY = pytest.StashKey[Path]()
#: The ``time.monotonic()`` at which the running item's protocol is a hang.
_DEADLINE_KEY = pytest.StashKey[float]()


def configure(config: pytest.Config) -> None:
    """Open the descriptor this process's dump will be written to.

    Called from ``pytest_configure``, when no capture is active. An xdist
    worker opens the file its controller named for it
    (:func:`pytest_configure_node`). Any other process keeps a descriptor for
    its real stderr: during a test, fd capture points descriptor 2 at a
    temporary file, which a process that exits from the watchdog never reads
    back.

    pytest-timeout is a declared test dependency and must be installed. A run
    that does not load it as a plugin (``-p no:timeout``, or plugin autoload
    off without ``-p timeout``) has no timeouts to back up, and its settings
    cannot be read, so the backstop stays unarmed.
    """
    if not config.pluginmanager.is_registered(pytest_timeout):
        return
    dump_path = getattr(config, "workerinput", {}).get(_DUMP_PATH_INPUT)
    if dump_path is None:
        fd = os.dup(2)
    else:
        fd = os.open(dump_path, os.O_WRONLY)
    config.stash[_DUMP_FD_KEY] = fd
    config.add_cleanup(lambda: os.close(fd))


@pytest.hookimpl(optionalhook=True)
def pytest_configure_node(node):
    """Create, on the xdist controller, the file a new worker writes its dump to.

    A replacement worker has an id of its own, so it gets a file of its own. A
    worker on another host cannot write to a file on this one: as xdist does
    for its ``basetemp``, only a local (``popen``) worker is given one, and any
    other keeps its stderr.
    """
    config = node.config
    if _DUMP_FD_KEY not in config.stash or not node.gateway.spec.popen:
        return
    if _DUMP_DIR_KEY not in config.stash:
        dump_dir = Path(tempfile.mkdtemp(prefix="hang-backstop-"))
        config.stash[_DUMP_DIR_KEY] = dump_dir
        config.add_cleanup(lambda: shutil.rmtree(dump_dir))
    dump_path = config.stash[_DUMP_DIR_KEY] / f"{node.gateway.id}.txt"
    dump_path.touch(mode=0o600, exist_ok=False)
    node.workerinput[_DUMP_PATH_INPUT] = str(dump_path)


@pytest.hookimpl(optionalhook=True)
def pytest_handlecrashitem(crashitem, report, sched):
    """Append a crashed worker's dump to the report naming the item it ran.

    A worker the backstop ended wrote its whole dump before it exited. One
    that crashed any other way left its file empty, and its report is left as
    xdist wrote it.
    """
    dump_path = report.node.workerinput.get(_DUMP_PATH_INPUT)
    if dump_path is None:
        return
    dump = Path(dump_path).read_text(encoding="utf-8", errors="replace")
    if dump:
        report.longrepr = f"{report.longrepr}\n\n{dump}"


def _backstop_after(item: pytest.Item) -> float | None:
    """Seconds after which ``item``'s protocol is a hang, or None to leave it.

    The settings come from pytest-timeout, as in ``tests/utils/ci_budget.py``,
    so the backstop follows every way a timeout can be set. An item with no
    timeout is left alone, and so is one whose timeout covers its body only
    (``func_only``): its fixtures are unbounded by choice.
    """
    settings = _get_item_settings(item)
    if settings.timeout is None or settings.timeout <= 0 or settings.func_only:
        return None
    if not settings.disable_debugger_detection and is_debugging():
        return None
    return settings.timeout + GRACE_SECONDS


def _arm(item: pytest.Item) -> None:
    """Arm the watchdog for what is left of ``item``'s deadline."""
    remaining = item.stash[_DEADLINE_KEY] - time.monotonic()
    # Past the deadline already: the item is overdue, so fire at once.
    faulthandler.dump_traceback_later(
        max(remaining, 0.001), exit=True, file=item.config.stash[_DUMP_FD_KEY],
    )


@pytest.hookimpl(wrapper=True, tryfirst=True)
def pytest_runtest_protocol(item, nextitem):
    """Bound the item's setup, call and teardown together."""
    after = (
        _backstop_after(item) if _DUMP_FD_KEY in item.config.stash else None
    )
    if after is None:
        return (yield)
    item.stash[_DEADLINE_KEY] = time.monotonic() + after
    try:
        _arm(item)
        return (yield)
    finally:
        faulthandler.cancel_dump_traceback_later()
        del item.stash[_DEADLINE_KEY]


@pytest.hookimpl(trylast=True)
def pytest_exception_interact(node, call, report):
    """Re-arm the watchdog that pytest's faulthandler plugin just cancelled.

    That plugin cancels it, ``tryfirst``, for every failed phase. Without this
    a failure would leave the item's remaining phases unbounded, which is
    exactly where a hanging teardown sits. Not under a debugger: after
    ``--pdb`` has run, pytest-timeout reports one, and the session is the
    developer's.
    """
    if _DEADLINE_KEY not in node.stash:
        return
    if is_debugging():
        return
    _arm(node)


def pytest_enter_pdb(config, pdb):
    """Leave a pdb session entered through pytest running past the timeout.

    That is ``--pdb`` or ``breakpoint()``; pytest-timeout stops firing once pdb
    is entered, and so does the backstop. A debugger attached before the item
    starts leaves it unarmed (:func:`_backstop_after`). One entered any other
    way mid-test (``ipdb.set_trace()``, a debugger attaching then) is ended
    once the item's deadline passes, because the C watchdog cannot see it:
    run with ``--timeout=0`` to debug that way.
    """
    faulthandler.cancel_dump_traceback_later()
