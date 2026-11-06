"""Run a file read against a FIFO without letting a regression hang the suite.

``open()`` on a FIFO blocks until something opens it for writing. A reader
that should refuse a FIFO is run in a daemon thread and given a bounded time
to answer; if it has not, the FIFO is opened for writing once, which releases
the blocked reader, and the test fails rather than hanging.
"""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Callable

import pytest

if not hasattr(os, "mkfifo"):  # pragma: no cover - POSIX only
    pytest.skip("named pipes are POSIX-only", allow_module_level=True)


def call_against_fifo(fifo: Path, read: Callable[[], object], *, timeout: float = 5.0):
    """Create ``fifo`` and return the exception ``read`` raises, or fail."""
    os.mkfifo(fifo)
    outcome: dict[str, BaseException | object] = {}

    def run() -> None:
        try:
            outcome["result"] = read()
        except BaseException as error:  # noqa: BLE001 - handed back to the test
            outcome["error"] = error

    reader = threading.Thread(target=run, daemon=True)
    reader.start()
    reader.join(timeout)
    if reader.is_alive():
        writer = os.open(fifo, os.O_WRONLY | os.O_NONBLOCK)
        os.close(writer)
        reader.join(timeout)
        pytest.fail(f"reading {fifo} blocked instead of refusing a FIFO")
    if "error" not in outcome:
        pytest.fail(f"reading {fifo} returned {outcome.get('result')!r} instead of refusing")
    return outcome["error"]
