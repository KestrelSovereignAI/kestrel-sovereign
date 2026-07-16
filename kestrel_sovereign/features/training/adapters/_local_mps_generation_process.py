"""Bounded POSIX process-group ownership for Local MPS generation."""

from __future__ import annotations

import asyncio
import logging
import os
import signal
from collections.abc import Iterable
from dataclasses import dataclass
from typing import cast

from ._local_mps_generation_workspace import (
    GenerationWorkspaceLease,
    validate_generation_workspace,
)
from ._owned_async_task import await_owned_task, raise_owned_outcome


logger = logging.getLogger(__name__)

GENERATION_TERMINATION_GRACE_SECONDS = 10.0
GENERATION_KILL_REAP_TIMEOUT_SECONDS = 5.0
GENERATION_GROUP_POLL_SECONDS = 0.01
_PIPE_DRAIN_CHUNK_BYTES = 64 * 1024


@dataclass(frozen=True, slots=True)
class GenerationProcessLease:
    """A subprocess and the process-group identity assigned at creation."""

    process: asyncio.subprocess.Process
    process_group_id: int | None


def _generation_process_group_exists(lease: GenerationProcessLease) -> bool:
    """Return whether the retained group identity still has any members."""
    if os.name == "posix" and lease.process_group_id is not None:
        try:
            os.killpg(lease.process_group_id, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True
    return lease.process.returncode is None


def _signal_generation_process_group(
    lease: GenerationProcessLease,
    process_signal: signal.Signals,
) -> None:
    """Signal the retained POSIX process group, with direct-child fallback."""
    if os.name == "posix" and lease.process_group_id is not None:
        try:
            os.killpg(lease.process_group_id, process_signal)
            return
        except ProcessLookupError:
            return
        except OSError as error:
            logger.warning(
                "Could not signal generation process group %s: %s",
                lease.process_group_id,
                error,
            )

    process = lease.process
    if process.returncode is not None:
        return
    try:
        if process_signal == signal.SIGTERM:
            process.terminate()
        else:
            process.kill()
    except (ProcessLookupError, OSError):
        pass


async def _discard_stream(stream: asyncio.StreamReader | None) -> None:
    """Drain one subprocess pipe without retaining its output."""
    if stream is None:
        return
    while await stream.read(_PIPE_DRAIN_CHUNK_BYTES):
        pass


async def _drain_and_reap_generation_process(
    process: asyncio.subprocess.Process,
) -> None:
    """Drain inherited pipes and reap the direct child as one operation."""
    stdout_result, stderr_result, wait_result = await asyncio.gather(
        _discard_stream(getattr(process, "stdout", None)),
        _discard_stream(getattr(process, "stderr", None)),
        process.wait(),
        return_exceptions=True,
    )
    for stream_result in (stdout_result, stderr_result):
        if isinstance(stream_result, BaseException):
            logger.debug("Failed while draining generation output: %s", stream_result)
    if isinstance(wait_result, BaseException):
        raise RuntimeError("Failed to reap generation subprocess") from wait_result


async def _wait_for_generation_terminalization(
    lease: GenerationProcessLease,
    direct_child_task: asyncio.Task[None] | None,
    timeout: float,
) -> bool:
    """Wait until both the process group is empty and the child is reaped."""
    deadline = asyncio.get_running_loop().time() + max(timeout, 0.0)
    while True:
        group_empty = not _generation_process_group_exists(lease)
        child_reaped = direct_child_task is None or direct_child_task.done()
        if group_empty and child_reaped:
            if direct_child_task is not None:
                await direct_child_task
            return True

        remaining = deadline - asyncio.get_running_loop().time()
        if remaining <= 0:
            return False
        await asyncio.sleep(min(GENERATION_GROUP_POLL_SECONDS, remaining))


async def terminate_generation_process(
    lease: GenerationProcessLease,
    *,
    direct_child_reaped: bool,
) -> None:
    """Terminate, escalate, reap, and positively empty a generation group."""
    drain_task = (
        None
        if direct_child_reaped
        else asyncio.create_task(_drain_and_reap_generation_process(lease.process))
    )

    if await _wait_for_generation_terminalization(lease, drain_task, 0.0):
        return

    _signal_generation_process_group(lease, signal.SIGTERM)
    if await _wait_for_generation_terminalization(
        lease,
        drain_task,
        GENERATION_TERMINATION_GRACE_SECONDS,
    ):
        return

    _signal_generation_process_group(lease, signal.SIGKILL)
    if await _wait_for_generation_terminalization(
        lease,
        drain_task,
        GENERATION_KILL_REAP_TIMEOUT_SECONDS,
    ):
        return

    if drain_task is not None:
        drain_task.cancel()
        await asyncio.gather(drain_task, return_exceptions=True)
    raise RuntimeError(
        "Generation process group extinction could not be proved after SIGKILL; "
        "workspace preserved"
    )


async def start_generation_process(
    executable: str,
    script: str,
    payload: str,
    *,
    inherited_fds: Iterable[int] = (),
    workspace: GenerationWorkspaceLease,
) -> GenerationProcessLease:
    """Start a private process group without losing cancellation ownership."""
    await validate_generation_workspace(workspace)
    inherited_fd_tuple = tuple(inherited_fds)
    process_kwargs: dict[str, object] = {
        "stdout": asyncio.subprocess.PIPE,
        "stderr": asyncio.subprocess.PIPE,
        "env": {**os.environ, "TOKENIZERS_PARALLELISM": "false"},
    }
    if os.name == "posix":
        process_kwargs["start_new_session"] = True
        process_kwargs["pass_fds"] = inherited_fd_tuple
    elif inherited_fd_tuple:
        raise RuntimeError("Private generation artifacts require inherited POSIX fds")

    creation_task = asyncio.create_task(
        asyncio.create_subprocess_exec(
            executable,
            "-c",
            script,
            payload,
            **process_kwargs,
        )
    )
    creation = await await_owned_task(creation_task)
    if creation.cancellation is None or creation.error is not None:
        process = raise_owned_outcome(creation, operation="Generation process start")
        process_id = getattr(process, "pid", None)
        return GenerationProcessLease(
            process=process,
            process_group_id=(
                process_id if isinstance(process_id, int) and process_id > 0 else None
            ),
        )

    process = cast(asyncio.subprocess.Process, creation.result)
    process_id = getattr(process, "pid", None)
    process_lease = GenerationProcessLease(
        process=process,
        process_group_id=(
            process_id if isinstance(process_id, int) and process_id > 0 else None
        ),
    )
    termination_task = asyncio.create_task(
        terminate_generation_process(
            process_lease,
            direct_child_reaped=False,
        )
    )
    termination = await await_owned_task(termination_task, creation.cancellation)
    if termination.error is not None:
        workspace.preserve(termination.error)
    return raise_owned_outcome(
        termination,
        operation="Cancelled generation process start teardown",
    )
