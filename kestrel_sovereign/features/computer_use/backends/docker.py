"""Docker sandbox backend.

Filesystem ops run on the host (the agent is editing user files; that is
the whole point of the feature). Shell exec is wrapped through the
existing ``ComputeFeature`` ``DockerExecutor``: each call constructs a
one-shot ``ComputeScript`` and runs it in a fresh container with the
working directory mounted read-only by default. The reuse is deliberate
— the compute feature already has a vetted set of container security
flags (``--read-only``, ``--network=none``, ``--security-opt=no-new-privileges``,
memory and pid limits) and we don't want a second container runtime path
that could drift from those guarantees.
"""

from __future__ import annotations

import logging
import shlex
import time
import uuid
from pathlib import Path
from typing import Optional

from .base import (
    CompletedRun,
    DirEntry,
    SandboxBackend,
    host_list,
    host_read,
    host_write,
)

logger = logging.getLogger(__name__)


class DockerSandboxBackend(SandboxBackend):
    """Default backend: shell goes through DockerExecutor; fs stays on host."""

    name = "docker"  # type: ignore[assignment]

    def __init__(
        self,
        *,
        granted_capabilities: frozenset[str] | set[str] | None = None,
        memory_limit: str = "256m",
        cpu_quota: int = 50000,
        pids_limit: int = 50,
        max_output_bytes: int = 1024 * 1024,
    ) -> None:
        # The sandboxed grant is required even for the docker backend so
        # that the feature's gate logic stays uniform — the grant gates
        # the *capability*, the backend gates the *execution path*.
        granted = frozenset(granted_capabilities or ())
        if "shell_execution_sandboxed" not in granted:
            from .base import CapabilityBlocked

            raise CapabilityBlocked(
                "constitution",
                "docker backend requires Amendment IX grant 'shell_execution_sandboxed'",
            )

        # Lazy-import the compute executor so this module imports cleanly
        # even when the compute feature is disabled.
        from kestrel_sovereign.features.compute.executors.docker_executor import (
            DockerExecutor,
        )

        self._executor = DockerExecutor(
            default_memory_limit=memory_limit,
            default_cpu_quota=cpu_quota,
            default_pids_limit=pids_limit,
            max_output_bytes=max_output_bytes,
        )

    @property
    def is_available(self) -> bool:
        return self._executor.is_available

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
        cwd: Optional[Path],
        env: Optional[dict[str, str]],
        timeout: int,
    ) -> CompletedRun:
        if not argv:
            raise ValueError("empty argv")

        from kestrel_sovereign.features.compute.models import ComputeScript

        script = ComputeScript(
            id=str(uuid.uuid4()),
            name=f"computer-use:{argv[0]}",
            language="bash",
            content=" ".join(shlex.quote(a) for a in argv) + "\n",
            purpose="computer-use shell exec",
            timeout_seconds=timeout,
            environment=env or {},
        )

        started = time.monotonic()
        record = await self._executor.execute(
            script,
            working_dir=str(cwd) if cwd else None,
        )
        duration_ms = int((time.monotonic() - started) * 1000)
        return CompletedRun(
            argv=list(argv),
            returncode=record.exit_code if record.exit_code is not None else -1,
            stdout=record.stdout,
            stderr=record.stderr,
            duration_ms=duration_ms,
        )
