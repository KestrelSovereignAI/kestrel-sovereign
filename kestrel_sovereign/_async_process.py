"""Private-group ownership for async subprocess launches.

This module owns the detached-process contract: cancellation-safe launch,
whole-tree termination, and bounded reaping. Captured output and command result
semantics live separately in :mod:`kestrel_sovereign._bounded_subprocess`.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import signal
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, TypeVar, cast

from kestrel_sovereign._async_ownership import (
    await_owned_task,
    raise_owned_outcome,
)
from kestrel_sovereign._subprocess_helpers import (
    is_windows,
    new_process_group_kwargs,
)


DEFAULT_TERMINATE_GRACE_SECONDS = 2.0
DEFAULT_REAP_TIMEOUT_SECONDS = 5.0
_DRAIN_CHUNK_BYTES = 65_536
_GROUP_POLL_SECONDS = 0.05
_T = TypeVar("_T")


class SubprocessCleanupError(RuntimeError):
    """A child tree could not be reaped inside the bounded cleanup window."""


def _signal_process_tree(
    proc: asyncio.subprocess.Process,
    process_signal: signal.Signals,
) -> None:
    """Signal a private POSIX group, falling back to the direct child."""

    try:
        if is_windows():
            if process_signal == signal.SIGTERM:
                proc.terminate()
            else:
                proc.kill()
            return
        os.killpg(proc.pid, process_signal)
    except OSError:
        with contextlib.suppress(OSError):
            proc.send_signal(process_signal)


def _process_tree_alive(proc: asyncio.subprocess.Process) -> bool:
    """Whether the retained private group still has a live member.

    Windows limitation: there is no process-group probe analogous to POSIX
    ``killpg(pid, 0)``, so liveness degrades to the root's return code. A
    descendant that outlives a normally-exited root is invisible here and to
    ``taskkill /T`` (whose tree walk needs the root PID alive). Closing that
    gap requires assigning the child to a Job Object with kill-on-close at
    launch; until then Windows descendant cleanup is best-effort and bounded
    to the root's lifetime.
    """

    if is_windows():
        return proc.returncode is None
    try:
        os.killpg(proc.pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


async def _force_kill_windows_tree(
    proc: asyncio.subprocess.Process,
    timeout: float,
) -> None:
    """Use bounded ``taskkill /T`` before the Windows root disappears."""

    try:
        killer = await asyncio.create_subprocess_exec(
            "taskkill",
            "/F",
            "/T",
            "/PID",
            str(proc.pid),
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
    except (FileNotFoundError, OSError):
        with contextlib.suppress(ProcessLookupError):
            proc.kill()
        return

    try:
        await asyncio.wait_for(killer.wait(), timeout=max(timeout, 0.01))
    except TimeoutError:
        with contextlib.suppress(ProcessLookupError):
            killer.kill()
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(killer.wait(), timeout=max(timeout, 0.01))
    with contextlib.suppress(ProcessLookupError):
        proc.kill()


async def _wait_for_completion_and_tree_exit(
    proc: asyncio.subprocess.Process,
    completion: asyncio.Task[_T],
    timeout: float,
) -> bool:
    """Wait until the owned completion settles and the tree is extinct."""

    deadline = time.monotonic() + max(timeout, 0.0)
    while True:
        if completion.done() and not _process_tree_alive(proc):
            return True
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return False
        if completion.done():
            await asyncio.sleep(min(_GROUP_POLL_SECONDS, remaining))
        else:
            await asyncio.wait(
                {completion},
                timeout=min(_GROUP_POLL_SECONDS, remaining),
            )


async def _cancel_task(task: asyncio.Task[Any]) -> None:
    if not task.done():
        task.cancel()
    await asyncio.gather(task, return_exceptions=True)


async def _terminate_and_await(
    proc: asyncio.subprocess.Process,
    completion: asyncio.Task[_T],
    *,
    terminate_grace: float,
    reap_timeout: float,
) -> _T:
    """Terminate a private tree and settle its caller-owned completion task."""

    if completion.done() and not _process_tree_alive(proc):
        return completion.result()

    if is_windows():
        # CTRL_BREAK is not reliable for detached/no-console descendants.
        await _force_kill_windows_tree(proc, reap_timeout)
        if await _wait_for_completion_and_tree_exit(proc, completion, reap_timeout):
            return completion.result()
    else:
        _signal_process_tree(proc, signal.SIGTERM)
        if await _wait_for_completion_and_tree_exit(proc, completion, terminate_grace):
            return completion.result()

        _signal_process_tree(proc, signal.SIGKILL)
        if await _wait_for_completion_and_tree_exit(proc, completion, reap_timeout):
            return completion.result()

    # The group deadline expired. Give the direct child one final bounded reap;
    # a descendant that inherited no pipes may otherwise hide behind a settled
    # completion task while the leader remains a zombie.
    try:
        await asyncio.wait_for(proc.wait(), timeout=max(reap_timeout, 0.01))
    except TimeoutError as error:
        await _cancel_task(completion)
        raise SubprocessCleanupError(
            f"subprocess {proc.pid} did not reap within {reap_timeout:.3g}s"
        ) from error

    if not is_windows() and _process_tree_alive(proc):
        await _cancel_task(completion)
        raise SubprocessCleanupError(
            f"subprocess {proc.pid} reaped but its descendant process group "
            f"survived SIGKILL for {reap_timeout:.3g}s"
        )

    if not completion.done():
        await asyncio.wait({completion}, timeout=max(reap_timeout, 0.01))
    if completion.done():
        return completion.result()

    await _cancel_task(completion)
    raise SubprocessCleanupError(
        f"subprocess {proc.pid} reaped but its owned completion did not settle "
        f"within {reap_timeout:.3g}s"
    )


async def _discard_stream(stream: asyncio.StreamReader | None) -> None:
    if stream is None:
        return
    while await stream.read(_DRAIN_CHUNK_BYTES):
        pass


async def _drain_and_wait(proc: asyncio.subprocess.Process) -> None:
    """Discard inherited pipes and reap a detached process."""

    stdout_result, stderr_result, wait_result = await asyncio.gather(
        _discard_stream(getattr(proc, "stdout", None)),
        _discard_stream(getattr(proc, "stderr", None)),
        proc.wait(),
        return_exceptions=True,
    )
    del stdout_result, stderr_result
    if isinstance(wait_result, BaseException):
        raise RuntimeError(f"failed to reap subprocess {proc.pid}") from wait_result


async def terminate_process_tree(
    proc: asyncio.subprocess.Process,
    *,
    terminate_grace: float = DEFAULT_TERMINATE_GRACE_SECONDS,
    reap_timeout: float = DEFAULT_REAP_TIMEOUT_SECONDS,
) -> None:
    """Terminate a privately-grouped async process and all descendants.

    This detached variant owns pipe discard and direct-child reaping. It is
    suitable for a durable job whose ownership transfer failed after launch.
    Caller cancellation is re-raised only after cleanup reaches a terminal
    state.
    """

    if terminate_grace < 0:
        raise ValueError("terminate_grace must not be negative")
    if reap_timeout <= 0:
        raise ValueError("reap_timeout must be positive")

    completion = asyncio.create_task(
        _drain_and_wait(proc),
        name=f"subprocess-{proc.pid}-detached-completion",
    )
    cleanup = asyncio.create_task(
        _terminate_and_await(
            proc,
            completion,
            terminate_grace=terminate_grace,
            reap_timeout=reap_timeout,
        ),
        name=f"subprocess-{proc.pid}-detached-cleanup",
    )
    outcome = await await_owned_task(cleanup)
    raise_owned_outcome(outcome, operation="detached subprocess cleanup")


async def start_async_process(
    cmd: Sequence[str],
    *,
    cwd: Path | None = None,
    env: Mapping[str, str] | None = None,
    stdout: Any = None,
    stderr: Any = None,
) -> asyncio.subprocess.Process:
    """Launch a privately-grouped process without a cancellation gap."""

    argv = tuple(str(part) for part in cmd)
    if not argv:
        raise ValueError("cmd must not be empty")

    launch = asyncio.create_task(
        asyncio.create_subprocess_exec(
            *argv,
            cwd=str(cwd) if cwd else None,
            env=dict(env) if env is not None else None,
            stdout=stdout,
            stderr=stderr,
            **new_process_group_kwargs(),
        ),
        name="subprocess-launch",
    )
    launch_outcome = await await_owned_task(launch)
    if launch_outcome.cancellation is None or launch_outcome.error is not None:
        return raise_owned_outcome(launch_outcome, operation="subprocess launch")

    proc = cast(asyncio.subprocess.Process, launch_outcome.result)
    cleanup = asyncio.create_task(
        terminate_process_tree(proc),
        name=f"subprocess-{proc.pid}-cancelled-launch-cleanup",
    )
    cleanup_outcome = await await_owned_task(
        cleanup,
        launch_outcome.cancellation,
    )
    return raise_owned_outcome(
        cleanup_outcome,
        operation="cancelled subprocess launch cleanup",
    )


__all__ = [
    "DEFAULT_REAP_TIMEOUT_SECONDS",
    "DEFAULT_TERMINATE_GRACE_SECONDS",
    "SubprocessCleanupError",
    "start_async_process",
    "terminate_process_tree",
]
