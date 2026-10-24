"""Filesystem-only backend: the file tools, and no shell execution at all.

Selected with ``backend = "none"`` in ``[features.computer_use]``.

Both shell backends refuse to construct without a shell grant (docker needs
``shell_execution_sandboxed``; local needs that and ``shell_execution_host``),
and every tool call needs a backend. An agent granted only
``filesystem_read`` therefore had no working tool at all: ``fs_read`` was
refused at readiness because no backend could be built.

This backend serves ``fs_read``, ``fs_list``, ``fs_write`` and ``fs_edit`` on
the host, as the other two do, and cannot execute. Choosing it is the
operator's statement that this agent runs no commands, and it keeps holding
when the constitution changes: a shell grant added later, without the backend
being changed, still runs nothing.
"""

from __future__ import annotations

from pathlib import Path

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


class FilesystemOnlyBackend(SandboxBackend):
    """Host filesystem operations with no ``exec``."""

    name = "none"  # type: ignore[assignment]
    supports_exec = False

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
        # The shell gate refuses before reaching here; this is the backstop
        # for any caller that does not go through it.
        raise CapabilityBlocked(
            "backend", "backend 'none' has no shell execution"
        )
