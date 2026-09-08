"""Docker sandbox backend.

Filesystem ops run on the host (the agent is editing user files; that is
the whole point of the feature). Shell exec is wrapped through the
existing ``ComputeFeature`` ``DockerExecutor``: each call constructs a
one-shot ``ComputeCommand`` and runs it in a fresh container with the
working directory mounted read-only by default. The reuse is deliberate
— the compute feature already has a vetted set of container security
flags (``--read-only``, ``--network=none``, ``--security-opt=no-new-privileges``,
memory and pid limits) and we don't want a second container runtime path
that could drift from those guarantees.

The command is an argv vector all the way down. This backend used to
quote the vector into a bash script and run that instead, which meant
the words were read a second time, by a shell, after the policy had
vetted them — so ``eval 'printf HACKED'`` ran ``printf`` having shown
the policy only ``eval``, which is not a program at all (#3187). A
method named ``exec(argv)`` now execs argv.

Position after the image is not what makes that true; the executor
names ``argv[0]`` to Docker with ``--entrypoint``. Words placed after
an image are appended to whatever ``ENTRYPOINT`` it declares, so on an
image that has one they are arguments to the image's program rather
than a program of their own — the same defect one layer down.
"""

from __future__ import annotations

import logging
import time
import uuid
from pathlib import Path
from typing import Optional

from ..capture import write_stream
from .base import (
    CaptureTarget,
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
        capture: Optional[CaptureTarget] = None,
    ) -> CompletedRun:
        """Run ``argv`` in a one-shot container.

        Two completeness facts used to be dropped on this path, and both are
        the shape #3243 is about — a result that reads whole when it is not.

        The executor caps output at ``max_output_bytes`` and marks the clip
        by appending a marker to the *text*, so ``truncated_stdout`` stayed
        ``False`` here no matter how much was thrown away. It is now read
        back off that marker, which is the only signal the executor gives.

        A timeout raised out of this method entirely, so ``timed_out``
        was likewise never ``True`` on this backend. It is now caught and
        reported as the outcome it is.

        A capture on this backend is weaker than on the local one and says
        so: the container's output has already been through the executor's
        cap by the time it gets here, so the file is written from what
        survived, and ``truncated_stdout`` rides along to say whether that
        was everything.
        """
        if not argv:
            raise ValueError("empty argv")

        from kestrel_sovereign.features.compute.models import ComputeCommand
        from kestrel_sovereign.features.compute.executors.base import (
            _OUTPUT_TRUNCATED_SUFFIX,
            ExecutionTimeoutError,
        )

        command = ComputeCommand(
            id=str(uuid.uuid4()),
            name=f"computer-use:{argv[0]}",
            argv=list(argv),
            purpose="computer-use shell exec",
            timeout_seconds=timeout,
            environment=env or {},
        )

        started = time.monotonic()
        try:
            record = await self._executor.execute_command(
                command,
                working_dir=str(cwd) if cwd else None,
            )
        except ExecutionTimeoutError:
            duration_ms = int((time.monotonic() - started) * 1000)
            # A capture was asked for, so the files must exist even though
            # the executor kept nothing from before the kill. Skipping them
            # left the feature writing a manifest that named paths which
            # were not there, and previews reading "[capture unreadable]" —
            # a missing artifact reported as a broken one.
            if capture is not None:
                await write_stream(capture.stdout_path, b"")
                await write_stream(
                    capture.stderr_path,
                    f"command exceeded its {timeout}s timeout; the "
                    f"container was killed and no output was preserved\n".encode(
                        "utf-8"
                    ),
                )
            return CompletedRun(
                argv=list(argv),
                returncode=-1,
                stdout="",
                stderr=f"command exceeded its {timeout}s timeout",
                duration_ms=duration_ms,
                timed_out=True,
                stdout_path=str(capture.stdout_path) if capture else None,
                stderr_path=str(capture.stderr_path) if capture else None,
                cwd=str(cwd) if cwd else None,
            )
        duration_ms = int((time.monotonic() - started) * 1000)

        stdout, out_trunc = _split_truncation_marker(
            record.stdout, _OUTPUT_TRUNCATED_SUFFIX
        )
        stderr, err_trunc = _split_truncation_marker(
            record.stderr, _OUTPUT_TRUNCATED_SUFFIX
        )

        stdout_path = stderr_path = None
        if capture is not None:
            # ``write_stream``, not ``host_write``: a capture is owner-only,
            # like the manifest and the audit log beside it.
            await write_stream(capture.stdout_path, stdout.encode("utf-8"))
            await write_stream(capture.stderr_path, stderr.encode("utf-8"))
            stdout_path = str(capture.stdout_path)
            stderr_path = str(capture.stderr_path)

        return CompletedRun(
            argv=list(argv),
            returncode=record.exit_code if record.exit_code is not None else -1,
            stdout=stdout,
            stderr=stderr,
            duration_ms=duration_ms,
            truncated_stdout=out_trunc,
            truncated_stderr=err_trunc,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            cwd=str(cwd) if cwd else None,
        )


def _split_truncation_marker(text: str, marker: str) -> tuple[str, bool]:
    """Recover the executor's truncation flag from the text it appended.

    The executor signals a clip by appending ``marker`` to the decoded
    output and keeps no boolean on the record, so this is the only place
    the fact survives. Reading it back is coupling — hence the shared
    constant rather than a copied literal — but a marker in prose is not
    a flag a caller can branch on, and #3243 turns on being able to.
    """
    if text.endswith(marker):
        return text[: -len(marker)], True
    return text, False
