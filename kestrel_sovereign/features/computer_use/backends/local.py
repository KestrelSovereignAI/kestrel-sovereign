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
import shutil
import time
from pathlib import Path

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

        With a capture, the child's stdout and stderr are handed the open
        files as their own descriptors, so the bytes go from the process to
        the disk without passing through this one. That is what makes the
        artifact durable rather than merely re-serialized: there is no cap to
        exceed, no buffer to exhaust, and nothing to truncate. It is also
        exactly what the shell redirect this surface cannot express would
        have done (#3243).
        """
        if not argv:
            raise ValueError("empty argv")

        binary = shutil.which(argv[0]) or argv[0]
        full_argv = [binary, *argv[1:]]
        started = time.monotonic()

        out_fh = err_fh = None
        if capture is not None:
            try:
                out_fh, err_fh = await asyncio.to_thread(_open_capture, capture)
            except OSError as exc:
                duration_ms = int((time.monotonic() - started) * 1000)
                return CompletedRun(
                    argv=list(argv),
                    returncode=-1,
                    stdout="",
                    stderr=f"could not open capture file: {exc}",
                    duration_ms=duration_ms,
                )

        try:
            try:
                proc = await asyncio.create_subprocess_exec(
                    *full_argv,
                    cwd=str(cwd) if cwd else None,
                    env=sanitized_subprocess_env(env),
                    stdout=out_fh if out_fh is not None else asyncio.subprocess.PIPE,
                    stderr=err_fh if err_fh is not None else asyncio.subprocess.PIPE,
                )
            except FileNotFoundError as exc:
                duration_ms = int((time.monotonic() - started) * 1000)
                return CompletedRun(
                    argv=list(argv),
                    returncode=127,
                    stdout="",
                    stderr=str(exc),
                    duration_ms=duration_ms,
                    stdout_path=str(capture.stdout_path) if capture else None,
                    stderr_path=str(capture.stderr_path) if capture else None,
                )

            timed_out = False
            try:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    proc.communicate(), timeout=timeout
                )
            except asyncio.TimeoutError:
                timed_out = True
                proc.kill()
                try:
                    stdout_bytes, stderr_bytes = await proc.communicate()
                except Exception:  # noqa: BLE001
                    stdout_bytes, stderr_bytes = b"", b""
        finally:
            # The child holds its own duplicated descriptors; closing ours
            # here neither truncates the file nor races the write.
            for fh in (out_fh, err_fh):
                if fh is not None:
                    try:
                        fh.close()
                    except OSError:  # pragma: no cover - defensive
                        pass

        duration_ms = int((time.monotonic() - started) * 1000)
        if capture is not None:
            # ``communicate`` returns None for a stream it did not pipe.
            # Nothing was buffered, so nothing could have been clipped.
            return CompletedRun(
                argv=list(argv),
                returncode=proc.returncode if proc.returncode is not None else -1,
                stdout="",
                stderr="",
                duration_ms=duration_ms,
                truncated_stdout=False,
                truncated_stderr=False,
                timed_out=timed_out,
                stdout_path=str(capture.stdout_path),
                stderr_path=str(capture.stderr_path),
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
        )


def _open_capture(capture: CaptureTarget):
    """Open both capture files for writing, creating parents.

    Opened ``wb`` rather than appended: a capture path names one run's
    output, and a stale body under a fresh run's manifest would read as that
    run's output.
    """
    capture.stdout_path.parent.mkdir(parents=True, exist_ok=True)
    capture.stderr_path.parent.mkdir(parents=True, exist_ok=True)
    out_fh = open(capture.stdout_path, "wb")
    try:
        err_fh = open(capture.stderr_path, "wb")
    except OSError:
        out_fh.close()
        raise
    return out_fh, err_fh


def _truncate(data: bytes) -> tuple[str, bool]:
    if len(data) <= _MAX_OUTPUT_BYTES:
        return data.decode("utf-8", errors="replace"), False
    head = data[:_MAX_OUTPUT_BYTES]
    return head.decode("utf-8", errors="replace"), True
