"""Shared subprocess helpers for the host-side bash-to-Python ports
(epic #1050).

The verify-install / demo-runner / docker-remote / agent-docker CLI
modules all spawn long-running subprocesses (uvicorn, docker run,
playwright test) and need the same three primitives:

- :func:`run_streaming` — ``subprocess.run`` with streamed stdout/stderr
  (no ``capture_output=True``; codex's Tier 1.3 lesson — buffered output
  makes long-running installs/builds look hung).
- :func:`start_background_process` — ``subprocess.Popen`` with
  ``start_new_session=True`` on POSIX and ``CREATE_NEW_PROCESS_GROUP``
  on Windows so SIGTERM/Ctrl+Break reaches worker children.
- :func:`stop_process` — best-effort termination that walks the process
  tree (``os.killpg`` on POSIX, ``taskkill /F /T`` on Windows). Cleanup,
  not assertion — failures swallowed.
- :func:`wait_for_health` — poll ``http://127.0.0.1:<port>/health``
  until 200 or timeout. Same idiom as
  :class:`kestrel_sovereign.multi_agent.process_manager.ProcessManager`.

``cli_verify_install.py`` predates this module and keeps its own
in-module copies (the extraction is risky for that file's
identity-bootstrap path). New modules in tier 3 use this shared
surface so the same idiom isn't copy-pasted three more times.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Mapping, Optional, Sequence


def is_windows() -> bool:
    """Single source of truth for the platform branch in this module."""
    return sys.platform == "win32"


def run_streaming(
    cmd: Sequence[str],
    *,
    cwd: Optional[Path] = None,
    env: Optional[Mapping[str, str]] = None,
    check: bool = False,
) -> int:
    """Run ``cmd`` with stdout/stderr streamed live to the parent
    console; return the exit code.

    Codex's Tier 1.3 review caught that ``capture_output=True``
    buffers everything until exit — for ``docker build``,
    ``npx playwright test``, or ``uvicorn`` boot the operator wants
    to see progress. We deliberately do *not* pass ``capture_output``.
    """
    completed = subprocess.run(
        list(cmd),
        cwd=str(cwd) if cwd else None,
        env=dict(env) if env is not None else None,
        check=check,
    )
    return completed.returncode


def start_background_process(
    cmd: Sequence[str],
    *,
    cwd: Optional[Path] = None,
    env: Optional[Mapping[str, str]] = None,
    stdout: Optional[int] = None,
    stderr: Optional[int] = None,
) -> "subprocess.Popen[bytes]":
    """Spawn ``cmd`` as a background process in a new session/process
    group, so :func:`stop_process` can later signal the whole tree.

    On POSIX we ``start_new_session=True`` (calls ``setsid()``); on
    Windows we use ``CREATE_NEW_PROCESS_GROUP`` so we can later send
    ``CTRL_BREAK_EVENT`` if needed. Same idiom as
    :class:`kestrel_sovereign.multi_agent.process_manager.ProcessManager`.

    ``stdout`` / ``stderr`` accept the usual ``subprocess`` sentinels
    (None = inherit, ``subprocess.DEVNULL``, an open file descriptor).
    Pass an open file FD if you want logs to a file rather than the
    parent terminal — e.g. uvicorn-as-demo-server.
    """
    kwargs: dict = {
        "cwd": str(cwd) if cwd else None,
        "env": dict(env) if env is not None else None,
        "stdout": stdout,
        "stderr": stderr,
    }
    if is_windows():
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        kwargs["start_new_session"] = True
    return subprocess.Popen(list(cmd), **kwargs)  # type: ignore[arg-type]


def stop_process(
    proc: "subprocess.Popen[bytes]",
    *,
    timeout: float = 10.0,
) -> None:
    """Terminate ``proc`` and its children best-effort; cleanup, not
    assertion — every OSError is swallowed.

    On POSIX we ``os.killpg(SIGTERM)`` so workers (uvicorn's children)
    don't outlive the parent. On Windows we use ``taskkill /F /T`` to
    walk the process tree.
    """
    if proc.poll() is not None:
        return
    try:
        if is_windows():
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                check=False,
            )
        else:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                proc.terminate()
    except OSError:
        pass
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        try:
            proc.kill()
        except OSError:
            pass


def wait_for_health(
    port: int,
    *,
    host: str = "127.0.0.1",
    timeout: float = 60.0,
    poll_interval: float = 1.0,
    proc: Optional["subprocess.Popen[bytes]"] = None,
) -> bool:
    """Poll ``http://<host>:<port>/health`` until it returns 200 or
    the deadline elapses. Returns True on health, False on timeout
    OR on subprocess exit (when ``proc`` is supplied and dies before
    becoming healthy).

    The bash predecessors ``sleep``ed unconditionally; we poll because
    cold uvicorn boot can be slower under heavy load and a fixed sleep
    is racy. ``proc`` lets the caller fail-fast: if uvicorn died the
    log is more useful than another 50 seconds of polling.
    """
    url = f"http://{host}:{port}/health"
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc is not None and proc.poll() is not None:
            return False
        try:
            with urllib.request.urlopen(url, timeout=2) as resp:
                if resp.status == 200:
                    return True
        except (urllib.error.URLError, urllib.error.HTTPError, OSError):
            pass
        time.sleep(poll_interval)
    return False


__all__ = [
    "is_windows",
    "run_streaming",
    "start_background_process",
    "stop_process",
    "wait_for_health",
]
