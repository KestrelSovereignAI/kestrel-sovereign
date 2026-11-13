"""End a pytest process whose memory crosses a ceiling, naming the test that ran it there (#4066).

A test that runs away on memory does not fail by itself: its process grows
until the host has nothing left. On 2026-11-13 seven xdist workers of one
unit run reached 272 GiB together, growing about 0.3 GB/s each, and the host
kernel-panicked. pytest-timeout does not bound this: at that rate one test
reaches about 90 GB before a 300 s timeout fires.

:func:`register` starts a daemon thread in each pytest process, the xdist
controller and every worker alike. About every :data:`POLL_INTERVAL_SECONDS`
it reads the process's resident set size. Once that exceeds the ceiling, it
writes to the process's real stderr the test the process is running (tracked
from ``pytest_runtest_protocol``), the RSS, and every thread's stack, then
ends the process with :data:`EXIT_STATUS`, even when that report cannot be
written (a stderr pipe whose reader has gone). Under xdist the controller reports
``worker 'gwN' crashed while running '<nodeid>'`` and starts a replacement
worker, so the run goes on to its summary with the test named as a failure.

The ceiling is ``KESTREL_TEST_MAX_RSS_GB`` GiB (2**30 bytes), read from the
environment when the root conftest configures the process,
:data:`DEFAULT_MAX_RSS_GB` when unset. The default sits below the roughly
16 GB of a GitHub-hosted runner.

The thread runs beside every test, so a test's patches must not reach it: the
calls that read the RSS and end the process are bound when this module is
imported or when the watchdog starts (:func:`_rss_reader`).

The thread is a Python thread: a single C call that holds the GIL while it
allocates (one ``b"x" * n`` of many gigabytes) is seen only once it returns.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
import faulthandler
import math
import os
import sys
import threading

import psutil
import pytest

PLUGIN_NAME = "kestrel-memory-watchdog"

#: The variable naming the ceiling, in GiB.
CEILING_VARIABLE = "KESTREL_TEST_MAX_RSS_GB"
#: The ceiling when the variable is unset, in GiB.
DEFAULT_MAX_RSS_GB = 8.0
#: How often the thread reads the process's RSS.
POLL_INTERVAL_SECONDS = 0.5
#: The status a process the watchdog ended exits with. It is none of pytest's
#: own (0-5), faulthandler's (1), a signal's (128 + N) or a sysexits code.
EXIT_STATUS = 86

_BYTES_PER_GIB = 2**30

# Bound at import: a test may patch any of these on the ``os`` or
# ``faulthandler`` module while the thread is running beside it.
_os_open = os.open
_os_read = os.read
_os_close = os.close
_os_write = os.write
_os_exit = os._exit
_dump_traceback = faulthandler.dump_traceback


def ceiling_bytes(environ: Mapping[str, str]) -> int:
    """The RSS ceiling ``environ`` sets, in bytes.

    Raises:
        pytest.UsageError: the variable is set to anything but a number of
            GiB that comes to a finite size of at least one byte.
    """
    raw = environ.get(CEILING_VARIABLE)
    if raw is None:
        return int(DEFAULT_MAX_RSS_GB * _BYTES_PER_GIB)
    try:
        size = float(raw) * _BYTES_PER_GIB
    except ValueError:
        size = math.nan
    # A positive value under one byte would truncate to a ceiling of 0.
    if not math.isfinite(size) or size < 1:
        raise pytest.UsageError(
            f"{CEILING_VARIABLE} must be a finite, positive number of GiB "
            f"(at least one byte), not {raw!r}. Unset, the ceiling is "
            f"{DEFAULT_MAX_RSS_GB:g} GiB."
        )
    return int(size)


def _rss_reader(pid: int) -> Callable[[], int]:
    """A function returning process ``pid``'s RSS in bytes, out of a test's reach.

    psutil reads it with C calls on every platform but Linux. On Linux
    ``psutil.Process().memory_info()`` opens ``/proc/<pid>/statm`` with the
    builtin ``open``, so a test that patches ``builtins.open`` would receive
    the watchdog's read: ``tests/unit/test_reanchor_constitution.py`` hands it
    a list of fake handles, one of which the read would consume, or
    ``FileNotFoundError``, which would end the thread. There this reads the
    same field psutil does, the resident page count, through descriptor calls
    bound at import.
    """
    if sys.platform.startswith("linux"):
        path = f"/proc/{pid}/statm"
        page_size = os.sysconf("SC_PAGE_SIZE")

        def read_statm() -> int:
            fd = _os_open(path, os.O_RDONLY)
            try:
                statm = _os_read(fd, 4096)
            finally:
                _os_close(fd)
            return int(statm.split()[1]) * page_size

        return read_statm
    # Bound once: a test that patches ``psutil.Process`` or one of its
    # methods, as tests of process handling do, does not reach this read.
    memory_info = psutil.Process(pid).memory_info
    return lambda: memory_info().rss


def _gib(size: int) -> str:
    return f"{size / _BYTES_PER_GIB:.2f} GiB"


class MemoryWatchdog:
    """The ceiling, the test this process is running, and the thread that polls."""

    def __init__(self, ceiling: int, report_fd: int) -> None:
        self.ceiling = ceiling
        #: The nodeid of the item whose runtest protocol is running, if any.
        self.nodeid: str | None = None
        self._pid = os.getpid()
        self.read_rss = _rss_reader(self._pid)
        self._report_fd = report_fd
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._poll, name="kestrel-test-memory-watchdog", daemon=True,
        )

    @property
    def running(self) -> bool:
        return self._thread.is_alive()

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join()

    def _poll(self) -> None:
        # ``Event.wait``, not ``time.sleep``: a test that patches the latter
        # to return at once would turn this loop into a spin.
        while not self._stop.wait(POLL_INTERVAL_SECONDS):
            try:
                rss = self.read_rss()
            except OSError:
                # On Linux the read opens a descriptor. While a test has them
                # all (EMFILE), the poll is skipped and the next one retries.
                continue
            if rss > self.ceiling:
                self._end_process(rss)

    def _end_process(self, rss: int) -> None:
        try:
            self._report(rss)
        finally:
            # Whatever the report met. Its write raises when the stderr it
            # duplicated cannot take it (``BrokenPipeError`` once the reader
            # has gone), and an exception that ended only this thread would
            # leave the process running with no ceiling at all.
            _os_exit(EXIT_STATUS)

    def _report(self, rss: int) -> None:
        nodeid = self.nodeid
        running = (
            f"while running {nodeid}" if nodeid is not None
            else "while running no test (collection or session start/finish)"
        )
        header = (
            f"\n[memory watchdog] pytest process {self._pid} reached RSS "
            f"{_gib(rss)} ({rss} bytes), past its ceiling of "
            f"{_gib(self.ceiling)} ({CEILING_VARIABLE}), {running}. "
            f"Ending it with exit status {EXIT_STATUS} (#4066). "
            "Stacks of every thread follow.\n"
        )
        _os_write(self._report_fd, header.encode("utf-8", errors="replace"))
        _dump_traceback(file=self._report_fd, all_threads=True)

    @pytest.hookimpl(wrapper=True, tryfirst=True)
    def pytest_runtest_protocol(self, item, nextitem):
        """Record the item for the report while its setup, call and teardown run."""
        self.nodeid = item.nodeid
        try:
            return (yield)
        finally:
            self.nodeid = None


def register(config: pytest.Config) -> MemoryWatchdog:
    """Start this process's watchdog and install it as a plugin.

    Called from ``pytest_configure``, when no capture is active, so the report
    descriptor is a duplicate of the real stderr: during a test, fd capture
    points descriptor 2 at a temporary file, which a process ended from the
    watchdog never reads back. The thread is stopped and the descriptor closed
    as the config is cleaned up.
    """
    ceiling = ceiling_bytes(os.environ)
    report_fd = os.dup(2)
    watchdog = MemoryWatchdog(ceiling, report_fd)
    config.pluginmanager.register(watchdog, PLUGIN_NAME)
    watchdog.start()

    def cleanup() -> None:
        watchdog.stop()
        config.pluginmanager.unregister(watchdog)
        os.close(report_fd)

    config.add_cleanup(cleanup)
    return watchdog
