"""Abstract sandbox backend.

A backend is the only object in the feature that actually touches the
operating system. Filesystem ops always happen on the host — that is the
whole point — so the backend split is really about *shell exec*: the
Docker backend ships shell into a per-call container reusing the existing
``ComputeFeature`` executor, while the local backend runs the command
directly on the host. The host route requires a second, explicit
constitutional grant.

Path safety and policy decisions are the caller's responsibility. By the
time the backend is invoked, the path has already been resolved and
matched against the allow/deny lists.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path


class CapabilityBlocked(Exception):
    """Raised when a tool call cannot proceed because a gate refused it.

    The ``gate`` attribute names which gate refused: one of
    ``"privacy"``, ``"constitution"``, ``"approval"``, or ``"policy"``.
    """

    def __init__(self, gate: str, message: str):
        self.gate = gate
        super().__init__(f"{gate}: {message}")


@dataclass(frozen=True)
class DirEntry:
    """One row in a directory listing."""

    name: str
    is_dir: bool
    is_symlink: bool
    size: int  # 0 for directories
    mtime: float


@dataclass(frozen=True)
class CompletedRun:
    """Result of a shell exec."""

    argv: list[str]
    returncode: int
    stdout: str
    stderr: str
    duration_ms: int
    truncated_stdout: bool = False
    truncated_stderr: bool = False
    timed_out: bool = False


class SandboxBackend(ABC):
    """Abstract over the OS-touching layer."""

    @property
    @abstractmethod
    def name(self) -> str:
        """One-word identifier — ``"docker"`` or ``"local"``."""

    @abstractmethod
    async def read(self, path: Path, *, max_bytes: int) -> bytes:
        """Read up to ``max_bytes`` from ``path``."""

    @abstractmethod
    async def write(self, path: Path, data: bytes) -> int:
        """Write ``data`` to ``path``, creating parents as needed. Returns bytes written."""

    @abstractmethod
    async def list(self, path: Path) -> list[DirEntry]:
        """Return entries at ``path`` (non-recursive)."""

    @abstractmethod
    async def exec(
        self,
        argv: list[str],
        *,
        cwd: Path | None,
        env: dict[str, str] | None,
        timeout: int,
    ) -> CompletedRun:
        """Run ``argv`` and return its result."""

    async def shutdown(self) -> None:
        """Optional cleanup. Default: no-op."""


# === Shared host-FS implementation ============================================


async def host_read(path: Path, max_bytes: int) -> bytes:
    def _read() -> bytes:
        with open(path, "rb") as f:
            return f.read(max_bytes)

    return await asyncio.to_thread(_read)


async def host_write(path: Path, data: bytes) -> int:
    def _write() -> int:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            return f.write(data)

    return await asyncio.to_thread(_write)


async def host_list(path: Path) -> list[DirEntry]:
    def _list() -> list[DirEntry]:
        entries: list[DirEntry] = []
        with os_scandir(path) as it:
            for de in it:
                try:
                    st = de.stat(follow_symlinks=False)
                    entries.append(
                        DirEntry(
                            name=de.name,
                            is_dir=de.is_dir(follow_symlinks=False),
                            is_symlink=de.is_symlink(),
                            size=0 if de.is_dir(follow_symlinks=False) else st.st_size,
                            mtime=st.st_mtime,
                        )
                    )
                except OSError:
                    continue
        entries.sort(key=lambda e: (not e.is_dir, e.name.lower()))
        return entries

    return await asyncio.to_thread(_list)


def os_scandir(path):
    """Indirection so tests can monkeypatch."""
    import os as _os

    return _os.scandir(path)
