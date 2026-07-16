"""Hostile lifecycle tests for the shared bounded async subprocess runner."""

from __future__ import annotations

import asyncio
import os
import sys
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from kestrel_sovereign import _async_process as process_helpers
from kestrel_sovereign import _bounded_subprocess as bounded_helpers
from kestrel_sovereign._async_process import start_async_process
from kestrel_sovereign._bounded_subprocess import run_bounded_subprocess


async def _wait_for_file(path: Path, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while not path.exists():
        if time.monotonic() >= deadline:
            raise AssertionError(f"timed out waiting for {path}")
        await asyncio.sleep(0.01)


def _pid_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


async def _assert_pid_gone(pid: int, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while _pid_exists(pid):
        if time.monotonic() >= deadline:
            raise AssertionError(f"descendant process {pid} is still alive")
        await asyncio.sleep(0.02)


def _descendant_script(*, ignore_sigterm: bool) -> str:
    signal_setup = (
        "signal.signal(signal.SIGTERM, signal.SIG_IGN)" if ignore_sigterm else "pass"
    )
    return f"import signal, time\n{signal_setup}\ntime.sleep(60)\n"


def _parent_script(
    *,
    ignore_sigterm: bool,
    detach_child_output: bool = False,
    exit_after_spawn: bool = False,
) -> str:
    signal_setup = (
        "signal.signal(signal.SIGTERM, signal.SIG_IGN)" if ignore_sigterm else "pass"
    )
    child_stdio = (
        ", stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL"
        if detach_child_output
        else ""
    )
    epilogue = "sys.exit(0)" if exit_after_spawn else "time.sleep(60)"
    return (
        "import pathlib, signal, subprocess, sys, time\n"
        f"{signal_setup}\n"
        "child = subprocess.Popen([sys.executable, '-c', sys.argv[2]]"
        f"{child_stdio})\n"
        "pathlib.Path(sys.argv[1]).write_text(str(child.pid), encoding='utf-8')\n"
        "print(child.pid, flush=True)\n"
        f"{epilogue}\n"
    )


@pytest.mark.asyncio
@pytest.mark.skipif(sys.platform == "win32", reason="POSIX process-group assertion")
async def test_timeout_terminates_descendant_group_and_reaps_leader(tmp_path):
    pid_file = tmp_path / "child.pid"

    started = time.monotonic()
    result = await run_bounded_subprocess(
        [
            sys.executable,
            "-c",
            _parent_script(ignore_sigterm=False, detach_child_output=True),
            str(pid_file),
            _descendant_script(ignore_sigterm=True),
        ],
        timeout=0.2,
        terminate_grace=0.2,
        reap_timeout=1.0,
    )

    assert result.timed_out is True
    assert time.monotonic() - started < 3.0
    await _wait_for_file(pid_file)
    await _assert_pid_gone(int(pid_file.read_text(encoding="utf-8")))


@pytest.mark.asyncio
@pytest.mark.skipif(sys.platform == "win32", reason="POSIX process-group assertion")
async def test_success_sweeps_detached_descendant_after_root_exit(tmp_path):
    pid_file = tmp_path / "child.pid"

    result = await run_bounded_subprocess(
        [
            sys.executable,
            "-c",
            _parent_script(
                ignore_sigterm=False,
                detach_child_output=True,
                exit_after_spawn=True,
            ),
            str(pid_file),
            _descendant_script(ignore_sigterm=True),
        ],
        timeout=30,
        terminate_grace=0.2,
        reap_timeout=1.0,
    )

    assert result.timed_out is False
    assert result.returncode == 0
    await _wait_for_file(pid_file)
    await _assert_pid_gone(int(pid_file.read_text(encoding="utf-8")))


@pytest.mark.asyncio
@pytest.mark.skipif(sys.platform == "win32", reason="POSIX process-group assertion")
async def test_repeated_cancellation_waits_for_forced_tree_cleanup(tmp_path):
    pid_file = tmp_path / "child.pid"
    task = asyncio.create_task(
        run_bounded_subprocess(
            [
                sys.executable,
                "-c",
                _parent_script(ignore_sigterm=True),
                str(pid_file),
                _descendant_script(ignore_sigterm=True),
            ],
            timeout=30,
            terminate_grace=0.2,
            reap_timeout=1.0,
        )
    )
    await _wait_for_file(pid_file)

    task.cancel()
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    await _assert_pid_gone(int(pid_file.read_text(encoding="utf-8")))


@pytest.mark.asyncio
async def test_output_flood_is_drained_but_only_tail_is_retained():
    limit = 4096
    script = (
        "import os\n"
        "os.write(1, b'A' * 2_000_000 + b'OUT-END')\n"
        "os.write(2, b'B' * 2_000_000 + b'ERR-END')\n"
    )

    result = await run_bounded_subprocess(
        [sys.executable, "-c", script],
        timeout=10,
        max_output_bytes=limit,
    )

    assert result.returncode == 0
    assert result.timed_out is False
    assert result.stdout_truncated is True
    assert result.stderr_truncated is True
    assert len(result.stdout) == limit
    assert len(result.stderr) == limit
    assert result.stdout.endswith(b"OUT-END")
    assert result.stderr.endswith(b"ERR-END")


@pytest.mark.asyncio
async def test_launch_failure_preserves_exception_type(tmp_path):
    missing = tmp_path / "definitely-not-an-executable"
    with pytest.raises(FileNotFoundError):
        await run_bounded_subprocess([str(missing)], timeout=1)


@pytest.mark.asyncio
async def test_internal_collection_failure_still_terminates_process_tree(monkeypatch):
    proc = MagicMock(pid=4242, returncode=None)
    recorded: dict = {}

    async def fail_collection(*_args, **_kwargs):
        raise RuntimeError("pipe reader failed")

    async def spy_terminate_and_await(
        proc_arg, completion, *, terminate_grace, reap_timeout
    ):
        recorded["proc"] = proc_arg
        recorded["terminate_grace"] = terminate_grace
        recorded["reap_timeout"] = reap_timeout
        return await completion

    monkeypatch.setattr(
        bounded_helpers,
        "start_async_process",
        AsyncMock(return_value=proc),
    )
    monkeypatch.setattr(bounded_helpers, "_collect_process", fail_collection)
    monkeypatch.setattr(
        bounded_helpers, "_terminate_and_await", spy_terminate_and_await
    )

    with pytest.raises(RuntimeError, match="pipe reader failed"):
        await run_bounded_subprocess(
            ["fake-command"],
            timeout=1,
            terminate_grace=0.25,
            reap_timeout=0.5,
        )

    assert recorded == {
        "proc": proc,
        "terminate_grace": 0.25,
        "reap_timeout": 0.5,
    }


@pytest.mark.asyncio
async def test_cancellation_during_launch_cleans_eventual_process(monkeypatch):
    launch_entered = asyncio.Event()
    release_launch = asyncio.Event()
    proc = MagicMock(pid=4242, returncode=None)
    cleanup = AsyncMock()
    captured: dict = {}

    async def delayed_launch(*args, **kwargs):
        captured.update(kwargs)
        launch_entered.set()
        await release_launch.wait()
        return proc

    monkeypatch.setattr(asyncio, "create_subprocess_exec", delayed_launch)
    monkeypatch.setattr(process_helpers, "terminate_process_tree", cleanup)
    task = asyncio.create_task(start_async_process(["fake-command"]))
    await launch_entered.wait()

    task.cancel()
    release_launch.set()
    with pytest.raises(asyncio.CancelledError):
        await task

    cleanup.assert_awaited_once_with(proc)
    if os.name == "nt":
        assert captured["creationflags"]
    else:
        assert captured["start_new_session"] is True
