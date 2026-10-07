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

import asyncio
import logging
import time
import uuid
from pathlib import Path
from typing import BinaryIO, Optional

from .base import (
    CaptureTarget,
    CompletedRun,
    DirEntry,
    SandboxBackend,
    close_capture,
    host_list,
    host_read,
    host_write,
    open_capture,
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

        The executor caps output at ``max_output_bytes`` and used to mark the
        clip only by appending a marker to the *text*, so ``truncated_stdout``
        stayed ``False`` here no matter how much was thrown away. Reading it
        back off the marker fixed the flag and introduced a different lie:
        the output is caller-controlled, so a command that printed that exact
        string — echoing a prior executor log, say — was reported truncated
        when it was whole. ``ExecutionRecord.output_truncated`` now carries
        the fact, and the marker is only stripped when the record says there
        was one.

        A timeout raised out of this method entirely, so ``timed_out``
        was likewise never ``True`` on this backend. It is now caught and
        reported as the outcome it is.

        With a capture, each stream is written to its file as the container
        produces it: the executor hands every chunk it reads from the pipe to
        a sink before clipping its own copy (#3277). The capture used to be
        written afterwards from ``record.stdout``/``record.stderr`` — strings
        already clipped at ``max_output_bytes`` and decoded with
        ``errors="replace"`` — so on this, the DEFAULT backend, a review longer
        than the ceiling came back PARTIAL: the cap #3243 was filed against,
        still in force. The files now hold the bytes as emitted, unclipped,
        and ``truncated_*`` speak for them. The returned strings stay the
        executor's bounded copy.

        A stream's file is called whole only when the executor drained that
        stream to EOF and every write and the close landed. The executor says
        so by handing the sink a final ``b""``; nothing else — not the exit
        code, not the record's flags — is evidence that the pipe was read to
        its end.
        """
        if not argv:
            raise ValueError("empty argv")

        from kestrel_sovereign.features.compute.models import ComputeCommand
        from kestrel_sovereign.features.compute.executors.base import (
            _OUTPUT_TRUNCATED_SUFFIX,
            ExecutionEnvironmentError,
            ExecutionTimeoutError,
            OutputSinks,
        )

        command = ComputeCommand(
            id=str(uuid.uuid4()),
            name=f"computer-use:{argv[0]}",
            argv=list(argv),
            purpose="computer-use shell exec",
            timeout_seconds=timeout,
            environment=env or {},
        )
        ran_in = str(cwd) if cwd else None

        started = time.monotonic()
        out_fh = err_fh = None
        if capture is not None:
            try:
                out_fh, err_fh = await asyncio.to_thread(open_capture, capture)
            except OSError as exc:
                # Defaults here would have said "nothing truncated, no
                # writers" — and the feature would then write a manifest
                # calling the run complete over a file that was never opened.
                # Nothing ran, and the whole capture is lost; said as such.
                return CompletedRun(
                    argv=list(argv),
                    returncode=-1,
                    stdout="",
                    stderr=f"could not open capture file: {exc}",
                    duration_ms=int((time.monotonic() - started) * 1000),
                    truncated_stdout=True,
                    truncated_stderr=True,
                    writers_remaining=None,
                    cwd=ran_in,
                )
        out_sink = _StreamCapture(out_fh) if out_fh is not None else None
        err_sink = _StreamCapture(err_fh) if err_fh is not None else None

        record = None
        timed_out = False
        never_ran = False
        try:
            try:
                record = await self._executor.execute_command(
                    command,
                    working_dir=ran_in,
                    output_sinks=(
                        OutputSinks(stdout=out_sink, stderr=err_sink)
                        if out_sink is not None and err_sink is not None
                        else None
                    ),
                )
            except ExecutionTimeoutError:
                timed_out = True
            except ExecutionEnvironmentError:
                # No ``docker`` to run: nothing ran, so there is no output and
                # no artifact, and the caller gets the error instead of
                # paths. The files opened for one would be stream files no
                # manifest names, and pruning retires a set by its manifest —
                # on a host without Docker, two more per call, forever.
                never_ran = True
                raise
            if record is not None and err_sink is not None and not err_sink.ended:
                # The executor returned without reading stderr to its end,
                # which it does only when the run failed under it — a
                # snapshot it could not take, a ``docker`` it could not
                # start. The record's stderr is then the runtime's diagnostic
                # rather than anything the container wrote, and it goes INTO
                # the artifact, as the local backend's spawn failure does:
                # the caller reads the file when there is one. It is decoded
                # text written into the capture, so the stream is already
                # reported lost and ``annotate`` keeps it that way.
                if record.stderr:
                    await err_sink.annotate(
                        (record.stderr + "\n").encode("utf-8", errors="replace")
                    )
        finally:
            # Retired before closing, so a drain the executor abandoned
            # cannot write into a file whose completeness is being decided.
            for sink in (out_sink, err_sink):
                if sink is not None:
                    sink.retire()
            close_errors = await close_capture(out_fh, err_fh)
            if never_ran and capture is not None:
                await asyncio.to_thread(_discard_capture, capture)
        duration_ms = int((time.monotonic() - started) * 1000)

        stdout_path = str(capture.stdout_path) if capture else None
        stderr_path = str(capture.stderr_path) if capture else None
        out_lost = out_sink is not None and (out_sink.lost or "out" in close_errors)
        err_lost = err_sink is not None and (err_sink.lost or "err" in close_errors)
        failures = [
            str(exc)
            for exc in (
                out_sink.error if out_sink else None,
                err_sink.error if err_sink else None,
                close_errors.get("out"),
                close_errors.get("err"),
            )
            if exc is not None
        ]
        write_failure = (
            "\ncapture write failed: " + "; ".join(failures) if failures else ""
        )

        if timed_out:
            # What the container wrote before the deadline is in the files;
            # what it had in the pipe when it was killed is not, which is why
            # a timed-out capture never reached EOF and is reported lost as
            # well as timed out.
            return CompletedRun(
                argv=list(argv),
                returncode=-1,
                stdout="",
                stderr=f"command exceeded its {timeout}s timeout" + write_failure,
                duration_ms=duration_ms,
                timed_out=True,
                truncated_stdout=out_lost,
                truncated_stderr=err_lost,
                stdout_path=stdout_path,
                stderr_path=stderr_path,
                cwd=ran_in,
            )

        stdout, out_clipped = _split_truncation_marker(
            record.stdout,
            _OUTPUT_TRUNCATED_SUFFIX,
            bool(getattr(record, "stdout_truncated", False)),
        )
        stderr, err_clipped = _split_truncation_marker(
            record.stderr,
            _OUTPUT_TRUNCATED_SUFFIX,
            bool(getattr(record, "stderr_truncated", False)),
        )

        return CompletedRun(
            argv=list(argv),
            returncode=record.exit_code if record.exit_code is not None else -1,
            stdout=stdout,
            stderr=stderr + write_failure,
            duration_ms=duration_ms,
            # With a capture the files are the output of record: the clip on
            # the strings is a bounded copy of a whole artifact, not loss.
            truncated_stdout=out_lost if capture is not None else out_clipped,
            truncated_stderr=err_lost if capture is not None else err_clipped,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            cwd=ran_in,
        )


class _StreamCapture:
    """One capture file, written as the container produces its stream.

    The executor's :data:`OutputSink`: it is handed every chunk read from the
    pipe, unclipped and undecoded, and then a final ``b""`` at EOF. The file
    therefore holds the stream's bytes as emitted — a non-UTF-8 byte survives
    as itself — and :attr:`ended` records whether the stream was read to its
    end, the one fact that lets the file be called whole.
    """

    def __init__(self, fh: BinaryIO) -> None:
        self._fh = fh
        self._retired = False
        self.ended = False
        self.error: Optional[OSError] = None

    async def __call__(self, chunk: bytes) -> None:
        if self._retired:
            return
        if not chunk:
            self.ended = True
            return
        await self.write(chunk)

    async def write(self, data: bytes) -> None:
        """Append ``data`` to the file, off the event loop.

        A failed write is recorded rather than raised. Raising would end the
        executor's drain, and a pipe nobody reads blocks the container on it;
        once a piece is missing, later writes are skipped too, since a file
        with a hole in the middle reads as whole more convincingly than one
        that stops.
        """
        if self._retired or self.error is not None:
            return
        try:
            await asyncio.to_thread(self._fh.write, data)
        except OSError as exc:
            self.error = exc

    async def annotate(self, data: bytes) -> None:
        """Append the runtime's own text and stop listening to the stream.

        Only for a stream the executor did not read to its end, whose file is
        therefore already incomplete. Retiring first freezes that verdict: a
        drain still running cannot splice its output around the note, or
        announce an EOF that would make the file look whole.
        """
        self._retired = True
        try:
            await asyncio.to_thread(self._fh.write, data)
        except OSError as exc:
            if self.error is None:
                self.error = exc

    def retire(self) -> None:
        """Accept nothing more: the file is about to be closed."""
        self._retired = True

    @property
    def lost(self) -> bool:
        """Whether the file may be missing bytes the stream carried."""
        return not self.ended or self.error is not None


def _discard_capture(capture: CaptureTarget) -> None:
    """Remove the stream files of a run that never started.

    Best effort: a file that cannot be removed is logged, never raised, so
    the error that explains why nothing ran is the one the caller sees.
    """
    for path in (capture.stdout_path, capture.stderr_path):
        try:
            path.unlink(missing_ok=True)
        except OSError as exc:
            logger.warning("could not remove unused capture file %s: %s", path, exc)


def _split_truncation_marker(
    text: str, marker: str, truncated: bool
) -> tuple[str, bool]:
    """Drop the executor's cosmetic marker when the record says it clipped.

    ``truncated`` is the authority; the marker is only presentation. The
    text alone cannot be, because it is whatever the command chose to
    print — a run that legitimately ends with that string is not a
    truncated run, and reporting it as one turns a clean pass into a
    caveated PARTIAL.
    """
    if not truncated:
        return text, False
    if text.endswith(marker):
        return text[: -len(marker)], True
    return text, True
