"""Local sandbox backend — direct host execution.

This backend is deliberately less ergonomic than the Docker backend:
shell exec runs on the host. It refuses to construct unless the caller
declares it has both grants from Amendment IX:
``shell_execution_sandboxed`` *and* ``shell_execution_host``. (The
sandboxed grant is still required because, conceptually, anything the
sandbox can do, the host can do.)
"""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import signal
import subprocess
import time
from pathlib import Path

from kestrel_sovereign._subprocess_helpers import (
    is_windows,
    new_process_group_kwargs,
)
from kestrel_sovereign.security.subprocess_env import sanitized_subprocess_env

from .base import (
    CapabilityBlocked,
    CaptureTarget,
    CompletedRun,
    DirEntry,
    SandboxBackend,
    host_list,
    host_read,
    host_write,
)

logger = logging.getLogger(__name__)

_MAX_OUTPUT_BYTES = 1024 * 1024  # 1 MiB cap on stdout/stderr each

# Copy size for a captured stream. Bounds this process's memory regardless of
# how much the command produces.
_PUMP_CHUNK_BYTES = 64 * 1024

# How long the pumps get after the direct child exits before anything still
# holding the output descriptors is reported instead of waited on.
_DRAIN_GRACE = 0.5

# Poll interval for the child's exit status, and the bound on reaping it
# after a kill so a kill that does not land cannot hang the tool.
_EXIT_POLL_SECONDS = 0.02
_REAP_GRACE = 5.0

# Extra time for a pump that has the bytes and is only landing them. Waiting
# on a pipe and writing to a slow disk both leave a task pending; only the
# first means someone else can still write.
_FLUSH_GRACE = 5.0

# Bound on the platform kill itself, so a wedged terminator cannot outlast
# the timeout that invoked it.
_KILL_TIMEOUT = 10.0


class LocalSandboxBackend(SandboxBackend):
    """Host-process backend.

    Args:
        granted_capabilities: The subset of
            ``DANGEROUS_CAPABILITIES`` that the sovereign has granted via
            Amendment IX. Construction fails fast if
            ``shell_execution_host`` is missing.
    """

    name = "local"  # type: ignore[assignment]

    def __init__(self, granted_capabilities: frozenset[str] | set[str]):
        granted = frozenset(granted_capabilities)
        if "shell_execution_host" not in granted:
            raise CapabilityBlocked(
                "constitution",
                "local backend requires Amendment IX grant 'shell_execution_host'",
            )
        if "shell_execution_sandboxed" not in granted:
            raise CapabilityBlocked(
                "constitution",
                "local backend requires Amendment IX grant 'shell_execution_sandboxed'",
            )
        self._granted = granted

    async def read(self, path: Path, *, max_bytes: int) -> bytes:
        return await host_read(path, max_bytes)

    async def write(self, path: Path, data: bytes) -> int:
        return await host_write(path, data)

    async def list(self, path: Path) -> list[DirEntry]:
        return await host_list(path)

    async def exec(
        self,
        argv: list[str],
        *,
        cwd: Path | None,
        env: dict[str, str] | None,
        timeout: int,
        capture: CaptureTarget | None = None,
    ) -> CompletedRun:
        """Run ``argv`` on the host.

        Without a capture, output is buffered in memory and clipped at
        ``_MAX_OUTPUT_BYTES`` on the way back — with the clip declared.

        With a capture, both streams are piped and pumped to disk in chunks.
        Memory stays O(chunk) however large the output grows, so the 1 MiB
        cap does not apply and nothing is truncated; what the pipe buys over
        handing the child the file descriptor directly is that **EOF is a
        fact about the output descriptors themselves**. The capture is final
        exactly when nothing holds them open.

        That was arrived at twice. Handing over the descriptors could not
        tell whether a descendant still held them; a process-group probe
        missed anything that called ``setsid``; and a side-channel sentinel
        missed anything spawned through ``subprocess.Popen``, which closes
        non-stdio descriptors while inheriting stdout and stderr — the common
        case, not an exotic one. Only the streams being written to can answer
        a question about the streams being written to.

        After the direct child exits the pumps get a short grace. If they
        have not reached EOF by then something else holds the descriptors,
        which is reported rather than waited on: a command that legitimately
        daemonizes should not hold the tool open for its whole timeout, and a
        capture it may still append to must not be called final.

        Stdin is ``/dev/null``. Inherited, it is the server's, and a command
        that reads it waits for input no one will send: ``claude -p`` blocks
        forever on an inherited stdin, which is the trap the merge-gate
        doctrine spells ``</dev/null`` and which this surface cannot express,
        since the redirect is shell grammar. The reviewer this feature exists
        to run is the exact program that hangs.
        """
        if not argv:
            raise ValueError("empty argv")

        binary = shutil.which(argv[0]) or argv[0]
        full_argv = [binary, *argv[1:]]
        started = time.monotonic()

        close_errors: dict[str, OSError] = {}
        out_error = err_error = None
        spawn_failed = False
        spawn_error = ""
        out_fh = err_fh = None
        if capture is not None:
            try:
                out_fh, err_fh = await asyncio.to_thread(_open_capture, capture)
            except OSError as exc:
                duration_ms = int((time.monotonic() - started) * 1000)
                # Defaults here would have said "nothing truncated, no
                # writers, no paths" — and the feature would then write a
                # manifest calling the run complete while previewing a file
                # that was never opened. The failure is total loss of the
                # capture, so it is reported as such.
                return CompletedRun(
                    argv=list(argv),
                    returncode=-1,
                    stdout="",
                    stderr=f"could not open capture file: {exc}",
                    duration_ms=duration_ms,
                    truncated_stdout=True,
                    truncated_stderr=True,
                    writers_remaining=None,
                    cwd=str(cwd) if cwd else os.getcwd(),
                )

        try:
            try:
                proc = await asyncio.create_subprocess_exec(
                    *full_argv,
                    cwd=str(cwd) if cwd else None,
                    env=sanitized_subprocess_env(env),
                    stdin=asyncio.subprocess.DEVNULL,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    **new_process_group_kwargs(),
                )
            except FileNotFoundError as exc:
                duration_ms = int((time.monotonic() - started) * 1000)
                message = str(exc)
                # ONLY FileNotFoundError. A broad OSError here turned a
                # permission denial, a bad executable format or descriptor
                # exhaustion into rc=127 — which ``shell`` reports PARTIAL
                # and the wrapper publishes as success, for a command that
                # never ran. Everything else propagates, as it did before
                # this branch existed.
                spawn_failed = True
                if capture is not None and err_fh is not None:
                    # The diagnostic is the only useful thing this run
                    # produced. Written INTO the capture, because the caller
                    # reads the artifact when there is one and would
                    # otherwise get rc=127 beside an empty file.
                    err_fh.write((message + "\n").encode("utf-8"))
                # Deliberately NOT returned here. The handles close in the
                # ``finally`` below, and a close that fails under disk
                # pressure would have had nowhere to go in a result already
                # constructed — the diagnostic would be filed as fully
                # persisted when it was not.
                spawn_error = message

            # Only when there is a process to wait for. The spawn-failure
            # branch above no longer returns, so this has to be skipped
            # explicitly rather than by falling out of the function.
            if not spawn_failed:
                timed_out = False
                writers_remaining: bool | None = None
                if capture is None:
                    try:
                        stdout_bytes, stderr_bytes = await asyncio.wait_for(
                            proc.communicate(), timeout=timeout
                        )
                    except asyncio.TimeoutError:
                        timed_out = True
                        await _kill_tree(proc.pid)
                        try:
                            stdout_bytes, stderr_bytes = await proc.communicate()
                        except Exception:  # noqa: BLE001
                            stdout_bytes, stderr_bytes = b"", b""
                else:
                    # A pump reports whether it is waiting on the pipe or
                    # flushing bytes it already read. Pending means "still
                    # writing" as often as it means "someone still holds the
                    # pipe", and cancelling a flush throws away bytes that were
                    # read successfully — measured on a slow filesystem, a
                    # one-second final write left an empty file behind a
                    # ``writers_remaining`` flag and a clean truncation flag.
                    states: list[dict] = [{"flushing": False}, {"flushing": False}]
                    pumps = [
                        asyncio.create_task(_pump(proc.stdout, out_fh, states[0])),
                        asyncio.create_task(_pump(proc.stderr, err_fh, states[1])),
                    ]
                    # NOT ``proc.wait()``. asyncio finishes a subprocess only
                    # once the process has exited AND every pipe transport has
                    # closed, so awaiting it waits for the descendants too — the
                    # grace below would never apply and a command that
                    # legitimately daemonizes would hold the tool for its whole
                    # timeout. Measured: a forked child sleeping 3s kept
                    # ``proc.wait()`` pending for 3s and the capture was then
                    # reported final, because by then it was.
                    #
                    # ``returncode`` is set by the child watcher when the process
                    # itself exits, independent of the pipes, so polling it
                    # separates "the command finished" from "nothing can write
                    # any more" — the two facts this needs to tell apart.
                    try:
                        timed_out = not await _await_exit(proc, timeout, pumps)
                        if timed_out or proc.returncode is None:
                            # Either the deadline passed, or a pump failed while
                            # the child was still running. Both end the same way:
                            # the tree goes, and what is left is reported.
                            await _kill_tree(proc.pid)
                            await _await_exit(proc, _REAP_GRACE)
                        done, pending = await asyncio.wait(
                            pumps, timeout=_DRAIN_GRACE
                        )
                        # Anything still flushing gets the rest of the budget:
                        # it has the bytes and only needs to land them, which is
                        # a different condition from waiting on a pipe nobody
                        # has closed.
                        if any(
                            states[i]["flushing"]
                            for i, t in enumerate(pumps)
                            if t in pending
                        ):
                            done, pending = await asyncio.wait(
                                pumps, timeout=_FLUSH_GRACE
                            )
                    except asyncio.CancelledError:
                        # Every post-spawn await, not just the first. The drain
                        # is its own wait and can be cancelled in its own right —
                        # a descendant holding the pipes is exactly when it takes
                        # long enough to be — and teardown that guarded only the
                        # exit wait left the pumps running into handles the
                        # ``finally`` was about to close.
                        #
                        # A shutdown, a client disconnect, an outer deadline:
                        # without teardown the host process runs on with nobody
                        # waiting for it, which for the long side-effecting
                        # commands this feature exists to run is worse than the
                        # timeout it mirrors.
                        await _kill_tree(proc.pid)
                        for task in pumps:
                            task.cancel()
                        # The same abandonment as the drain's timeout, and it
                        # leaks the same two descriptors: killing the direct
                        # child does not close pipes a surviving descendant
                        # still holds.
                        _close_pipe_transports(proc)
                        raise
                    # A pump cancelled mid-write loses whatever that write held,
                    # so it is lost output rather than a writer still holding the
                    # pipe. The two are reported differently because they mean
                    # different things to a reader of the manifest.
                    cancelled_mid_write: list[int] = [
                        i for i, t in enumerate(pumps)
                        if t in pending and states[i]["flushing"]
                    ]
                    writers_remaining = bool(
                        [t for i, t in enumerate(pumps)
                         if t in pending and not states[i]["flushing"]]
                    )
                    for task in pending:
                        task.cancel()
                    if pending:
                        _close_pipe_transports(proc)
                    # A pump that raised — a full disk, a vanished directory —
                    # finishes and lands in ``done`` like any other. Not asking
                    # for its exception meant a capture missing everything after
                    # the failure was reported complete: silent loss wearing a
                    # clean result. asyncio only logs it, at teardown, to a place
                    # no caller reads.
                    # Per stream, because the streams fail independently and
                    # saying stderr was clipped when stdout's disk write failed
                    # is the false claim the per-stream contract exists to
                    # prevent — collapsed here once already after being split
                    # one layer up.
                    out_error = pumps[0].exception() if pumps[0] in done else None
                    err_error = pumps[1].exception() if pumps[1] in done else None
                    stdout_bytes = stderr_bytes = b""
        finally:
            # A close flushes, and a flush can fail — a full disk surfaces
            # here rather than at any write. Swallowing it discarded the
            # buffered tail of a capture and called the file complete, the
            # same shape as the unread pump exception.
            # Off the loop for the same reason the writes are: a close
            # flushes, and a flush on a slow or full filesystem blocks
            # everything else in the process — including the timeout that is
            # supposed to bound this very call.
            for slot, fh in (("out", out_fh), ("err", err_fh)):
                if fh is not None:
                    try:
                        await asyncio.to_thread(fh.close)
                    except OSError as exc:  # pragma: no cover - disk-full path
                        close_errors[slot] = exc

        duration_ms = int((time.monotonic() - started) * 1000)
        effective_cwd = str(cwd) if cwd else os.getcwd()
        if spawn_failed:
            failed_close = bool(close_errors)
            return CompletedRun(
                argv=list(argv),
                returncode=127,
                stdout="",
                stderr=(
                    spawn_error
                    + (
                        f"; capture close failed: "
                        f"{'; '.join(str(e) for e in close_errors.values())}"
                        if failed_close
                        else ""
                    )
                ),
                duration_ms=duration_ms,
                truncated_stdout="out" in close_errors,
                truncated_stderr="err" in close_errors,
                stdout_path=str(capture.stdout_path) if capture else None,
                stderr_path=str(capture.stderr_path) if capture else None,
                cwd=effective_cwd,
                writers_remaining=False if capture else None,
            )
        if capture is not None:
            # A failed pump is lost output, which is what ``truncated_*``
            # already means and already folds into completeness — a second
            # field for the same fact would be a second thing to forget.
            out_lost = (
                out_error is not None
                or "out" in close_errors
                or 0 in cancelled_mid_write
            )
            err_lost = (
                err_error is not None
                or "err" in close_errors
                or 1 in cancelled_mid_write
            )
            failures = [
                str(e)
                for e in (
                    out_error,
                    err_error,
                    close_errors.get("out"),
                    close_errors.get("err"),
                )
                if e is not None
            ]
            return CompletedRun(
                argv=list(argv),
                returncode=proc.returncode if proc.returncode is not None else -1,
                stdout="",
                stderr=(
                    "capture write failed: " + "; ".join(failures)
                    if failures
                    else ""
                ),
                duration_ms=duration_ms,
                truncated_stdout=out_lost,
                truncated_stderr=err_lost,
                timed_out=timed_out,
                stdout_path=str(capture.stdout_path),
                stderr_path=str(capture.stderr_path),
                cwd=effective_cwd,
                writers_remaining=writers_remaining,
            )

        out, out_trunc = _truncate(stdout_bytes)
        err, err_trunc = _truncate(stderr_bytes)
        return CompletedRun(
            argv=list(argv),
            returncode=proc.returncode if proc.returncode is not None else -1,
            stdout=out,
            stderr=err,
            duration_ms=duration_ms,
            truncated_stdout=out_trunc,
            truncated_stderr=err_trunc,
            timed_out=timed_out,
            cwd=effective_cwd,
        )


def _close_pipe_transports(proc) -> None:
    """Close the subprocess pipes whose pumps we are abandoning.

    Cancelling a pump task does not close the pipe. A descendant that
    escaped the process group still holds the other end, so the server keeps
    two descriptors for as long as that process lives — indefinitely for a
    daemon, and a few such captures exhaust the process's limit.

    This is a function rather than a loop at the call site because there are
    two places that abandon pumps — the drain's own timeout and a
    cancellation arriving during the drain — and round 12 fixed only the
    first. The second is the same defect one path over, which is how this
    branch has repeatedly lost a round: the fix went in beside the bug
    rather than at the thing both paths share.
    """
    for stream in (proc.stdout, proc.stderr):
        transport = getattr(stream, "_transport", None)
        if transport is None:
            continue
        try:
            transport.close()
        except Exception:  # noqa: BLE001 - cleanup
            pass


async def _await_exit(proc, timeout: float, pumps: list | None = None) -> bool:
    """Wait for the process itself to exit. True if it did, False on timeout.

    Polls ``returncode`` rather than awaiting ``proc.wait()``: see the call
    site for why the difference is load-bearing.

    A failed pump ends the wait early. Once a capture write has raised, the
    artifact is already unrecoverable, so running the command to its full
    timeout buys nothing — and with nothing draining the pipe a high-output
    command blocks on it, turning a disk error into a hang. Returning here
    lets the caller kill the tree and report the loss while it is still
    news.
    """
    deadline = time.monotonic() + timeout
    while proc.returncode is None:
        if pumps and any(
            t.done() and t.exception() is not None for t in pumps
        ):
            return True
        if time.monotonic() >= deadline:
            return False
        await asyncio.sleep(_EXIT_POLL_SECONDS)
    return True


async def _pump(reader, fh, state: dict | None = None) -> None:
    """Copy ``reader`` to ``fh`` until EOF, a chunk at a time.

    The loop ends only when every holder of the write end has closed it, so
    the task outliving the direct child IS the signal that something else is
    still able to write.
    """
    if reader is None:  # pragma: no cover - defensive
        return
    while True:
        chunk = await reader.read(_PUMP_CHUNK_BYTES)
        if not chunk:
            return
        # Off the loop: a high-output command writing to a slow or full
        # filesystem would otherwise block every other task in this process,
        # including the poll that enforces this command's own timeout — the
        # bound exceeded by the work it exists to bound.
        if state is not None:
            state["flushing"] = True
        try:
            await asyncio.to_thread(fh.write, chunk)
        finally:
            if state is not None:
                state["flushing"] = False



async def _kill_tree(pid: int) -> None:
    """Kill the process and everything under it, best effort.

    ``os.killpg`` does not exist on Windows, and reaching it unguarded raised
    ``AttributeError`` out of the timeout path on a platform this package
    declares support for. The split mirrors
    :func:`kestrel_sovereign._subprocess_helpers.stop_process`.
    """
    if is_windows():
        # In a worker and bounded: this runs on every timeout and every
        # cancellation, and a wedged ``taskkill`` on the event loop would
        # block the server indefinitely — defeating the very timeout that
        # called it.
        try:
            await asyncio.wait_for(
                asyncio.to_thread(
                    subprocess.run,
                    ["taskkill", "/F", "/T", "/PID", str(pid)],
                    check=False,
                ),
                timeout=_KILL_TIMEOUT,
            )
        except (OSError, asyncio.TimeoutError):  # pragma: no cover - windows
            pass
        return
    try:
        os.killpg(pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        # Already gone, or never became a group leader. Fall back to the
        # leader alone rather than letting the timeout path raise.
        try:
            os.kill(pid, signal.SIGKILL)
        except OSError:
            pass


def _open_capture(capture: CaptureTarget):
    """Open both capture files for writing, creating parents.

    Opened ``wb`` rather than appended: a capture path names one run's
    output, and a stale body under a fresh run's manifest would read as that
    run's output.
    """
    from ..capture import open_stream

    out_fh = open_stream(capture.stdout_path)
    try:
        err_fh = open_stream(capture.stderr_path)
    except OSError:
        out_fh.close()
        raise
    return out_fh, err_fh


def _truncate(data: bytes) -> tuple[str, bool]:
    if len(data) <= _MAX_OUTPUT_BYTES:
        return data.decode("utf-8", errors="replace"), False
    head = data[:_MAX_OUTPUT_BYTES]
    return head.decode("utf-8", errors="replace"), True
