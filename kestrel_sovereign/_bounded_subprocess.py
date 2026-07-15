"""Bounded captured execution for async subprocesses.

The runner retains only the tail of each output stream while continuously
draining both pipes. Process launch and tree cleanup are delegated to the
private-group ownership contract in :mod:`kestrel_sovereign._async_process`.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from kestrel_sovereign._async_ownership import (
    await_owned_task,
    raise_owned_outcome,
)
from kestrel_sovereign._async_process import (
    DEFAULT_REAP_TIMEOUT_SECONDS,
    DEFAULT_TERMINATE_GRACE_SECONDS,
    _terminate_and_await,
    start_async_process,
    terminate_process_tree,
)


DEFAULT_CAPTURE_LIMIT_BYTES = 1_048_576
_READ_CHUNK_BYTES = 65_536


@dataclass(frozen=True, slots=True)
class BoundedProcessResult:
    """Observed result from :func:`run_bounded_subprocess`.

    ``stdout`` and ``stderr`` contain at most ``max_output_bytes`` each. When
    a stream exceeds that cap, the runner keeps its tail while continuing to
    drain the pipe so a noisy child cannot deadlock on a full OS pipe.
    """

    argv: tuple[str, ...]
    returncode: int
    stdout: bytes
    stderr: bytes
    duration_ms: int
    timed_out: bool = False
    stdout_truncated: bool = False
    stderr_truncated: bool = False


async def _read_bounded_tail(
    stream: asyncio.StreamReader | None,
    limit: int,
) -> tuple[bytes, bool]:
    """Drain ``stream`` to EOF while retaining its final ``limit`` bytes."""

    if stream is None:
        return b"", False

    retained = bytearray()
    total = 0
    while chunk := await stream.read(_READ_CHUNK_BYTES):
        total += len(chunk)
        if limit == 0:
            continue
        if len(chunk) >= limit:
            retained[:] = chunk[-limit:]
            continue
        overflow = len(retained) + len(chunk) - limit
        if overflow > 0:
            del retained[:overflow]
        retained.extend(chunk)
    return bytes(retained), total > limit


async def _collect_process(
    proc: asyncio.subprocess.Process,
    capture_limit: int,
) -> tuple[int, bytes, bool, bytes, bool]:
    """Wait for ``proc`` and both pipe drains as one owned operation."""

    stdout_task = asyncio.create_task(
        _read_bounded_tail(proc.stdout, capture_limit),
        name=f"subprocess-{proc.pid}-stdout",
    )
    stderr_task = asyncio.create_task(
        _read_bounded_tail(proc.stderr, capture_limit),
        name=f"subprocess-{proc.pid}-stderr",
    )
    wait_task = asyncio.create_task(
        proc.wait(),
        name=f"subprocess-{proc.pid}-wait",
    )
    owned = (stdout_task, stderr_task, wait_task)
    try:
        (
            (stdout, stdout_truncated),
            (stderr, stderr_truncated),
            returncode,
        ) = await asyncio.gather(*owned)
        return returncode, stdout, stdout_truncated, stderr, stderr_truncated
    finally:
        for task in owned:
            if not task.done():
                task.cancel()
        await asyncio.gather(*owned, return_exceptions=True)


async def run_bounded_subprocess(
    cmd: Sequence[str],
    *,
    cwd: Path | None = None,
    env: Mapping[str, str] | None = None,
    timeout: float,
    max_output_bytes: int = DEFAULT_CAPTURE_LIMIT_BYTES,
    terminate_grace: float = DEFAULT_TERMINATE_GRACE_SECONDS,
    reap_timeout: float = DEFAULT_REAP_TIMEOUT_SECONDS,
) -> BoundedProcessResult:
    """Run an argv command with bounded output and process-tree cleanup.

    Launch errors retain their original exception types for caller-side error
    classification. Timeout is represented by ``timed_out``. Cancellation is
    propagated only after the complete private process group reaches a bounded
    terminal cleanup outcome.
    """

    argv = tuple(str(part) for part in cmd)
    if not argv:
        raise ValueError("cmd must not be empty")
    if timeout <= 0:
        raise ValueError("timeout must be positive")
    if max_output_bytes < 0:
        raise ValueError("max_output_bytes must not be negative")
    if terminate_grace < 0:
        raise ValueError("terminate_grace must not be negative")
    if reap_timeout <= 0:
        raise ValueError("reap_timeout must be positive")

    started = time.monotonic()
    proc = await start_async_process(
        argv,
        cwd=cwd,
        env=env,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    completion = asyncio.create_task(
        _collect_process(proc, max_output_bytes),
        name=f"subprocess-{proc.pid}-completion",
    )
    timed_out = False
    try:
        done, _pending = await asyncio.wait({completion}, timeout=timeout)
        if done:
            captured = completion.result()
    except asyncio.CancelledError as cancellation:
        cleanup = asyncio.create_task(
            _terminate_and_await(
                proc,
                completion,
                terminate_grace=terminate_grace,
                reap_timeout=reap_timeout,
            ),
            name=f"subprocess-{proc.pid}-cancellation-cleanup",
        )
        outcome = await await_owned_task(cleanup, cancellation)
        raise_owned_outcome(outcome, operation="cancelled subprocess cleanup")
        raise AssertionError("cancelled cleanup unexpectedly returned")
    except Exception as process_error:
        try:
            await terminate_process_tree(
                proc,
                terminate_grace=terminate_grace,
                reap_timeout=reap_timeout,
            )
        except Exception as cleanup_error:  # noqa: BLE001
            process_error.add_note(f"subprocess cleanup also failed: {cleanup_error}")
        raise

    if not done:
        timed_out = True
        cleanup = asyncio.create_task(
            _terminate_and_await(
                proc,
                completion,
                terminate_grace=terminate_grace,
                reap_timeout=reap_timeout,
            ),
            name=f"subprocess-{proc.pid}-timeout-cleanup",
        )
        outcome = await await_owned_task(cleanup)
        captured = raise_owned_outcome(
            outcome,
            operation="timed-out subprocess cleanup",
        )

    returncode, stdout, stdout_truncated, stderr, stderr_truncated = captured
    return BoundedProcessResult(
        argv=argv,
        returncode=returncode,
        stdout=stdout,
        stderr=stderr,
        duration_ms=int((time.monotonic() - started) * 1000),
        timed_out=timed_out,
        stdout_truncated=stdout_truncated,
        stderr_truncated=stderr_truncated,
    )


__all__ = [
    "BoundedProcessResult",
    "DEFAULT_CAPTURE_LIMIT_BYTES",
    "run_bounded_subprocess",
]
