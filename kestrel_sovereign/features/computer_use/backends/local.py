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

from .base import (
    CapabilityBlocked,
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
    ) -> CompletedRun:
        if not argv:
            raise ValueError("empty argv")

        binary = shutil.which(argv[0]) or argv[0]
        full_argv = [binary, *argv[1:]]
        started = time.monotonic()

        try:
            proc = await asyncio.create_subprocess_exec(
                *full_argv,
                cwd=str(cwd) if cwd else None,
                env=env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except FileNotFoundError as exc:
            duration_ms = int((time.monotonic() - started) * 1000)
            return CompletedRun(
                argv=list(argv),
                returncode=127,
                stdout="",
                stderr=str(exc),
                duration_ms=duration_ms,
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

        duration_ms = int((time.monotonic() - started) * 1000)
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


def _truncate(data: bytes) -> tuple[str, bool]:
    if len(data) <= _MAX_OUTPUT_BYTES:
        return data.decode("utf-8", errors="replace"), False
    head = data[:_MAX_OUTPUT_BYTES]
    return head.decode("utf-8", errors="replace"), True
