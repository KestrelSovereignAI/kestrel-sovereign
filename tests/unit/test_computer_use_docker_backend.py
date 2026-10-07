"""What the Docker sandbox backend hands to the compute executor (#3187).

``SandboxBackend.exec`` is documented as "run ``argv`` and return its
result". This backend used to quote the vector into a bash script and
run *that*, which meant the words were read a second time — by a shell,
after the policy had vetted them. ``eval 'printf HACKED'`` therefore ran
``printf`` while the policy had seen only ``eval``, which is not a
program at all.

These tests pin the translation at the seam: an argv vector goes in, an
argv vector comes out, and the script-shaped path is not taken.

The capture tests at the end (#3277) drive the real ``DockerExecutor`` and
fake only the ``docker run`` child at its pipes, so they need no daemon. That
is the boundary the defect lived on: the capture used to be written from the
executor's decoded string, already clipped at ``max_output_bytes``, so on the
default backend a review past the ceiling came back PARTIAL.
"""

from __future__ import annotations

import asyncio
import errno
import json
import time
from pathlib import Path
from typing import Optional

import pytest
from kestrel_sdk.tools.result import ToolResultStatus

from kestrel_sovereign.features.compute.executors.base import (
    ExecutionEnvironmentError,
    OutputSinks,
)
from kestrel_sovereign.features.compute.models import ComputeCommand, ExecutionRecord
from kestrel_sovereign.features.computer_use import capture
from kestrel_sovereign.features.computer_use.backends import docker as docker_backend
from kestrel_sovereign.features.computer_use.backends.base import (
    CapabilityBlocked,
    CaptureTarget,
)
from kestrel_sovereign.features.computer_use.backends.docker import (
    DockerSandboxBackend,
)
from kestrel_sovereign.features.computer_use.feature import ComputerUseFeature
from tests.unit.test_computer_use_durable_capture import (
    FakeAgent,
    FakeApprovalQueue,
    _config,
)


class _RecordingExecutor:
    """Stands in for ``DockerExecutor``, recording which mode was used."""

    def __init__(self, exit_code: Optional[int] = 0) -> None:
        self.commands: list[ComputeCommand] = []
        self.scripts: list[object] = []
        self.working_dirs: list[Optional[str]] = []
        self.output_sinks: list[Optional[OutputSinks]] = []
        self._exit_code = exit_code

    async def execute_command(
        self,
        command: ComputeCommand,
        working_dir: Optional[str] = None,
        *,
        output_sinks: Optional[OutputSinks] = None,
    ) -> ExecutionRecord:
        self.commands.append(command)
        self.working_dirs.append(working_dir)
        self.output_sinks.append(output_sinks)
        return ExecutionRecord(
            id="exec-1",
            script_id=command.id,
            exit_code=self._exit_code,
            stdout="out",
            stderr="err",
            executor="docker",
        )

    async def execute(self, script, working_dir: Optional[str] = None):
        self.scripts.append(script)
        raise AssertionError("the backend built a script instead of a command")


def _backend(executor: _RecordingExecutor) -> DockerSandboxBackend:
    backend = DockerSandboxBackend(
        granted_capabilities={"shell_execution_sandboxed"},
    )
    backend._executor = executor  # type: ignore[assignment]
    return backend


@pytest.mark.asyncio
async def test_exec_hands_the_vector_to_the_argv_mode_unchanged() -> None:
    """Every element survives, including ones a shell would have read.

    The vector is deliberately full of shell meaning. Under the old
    implementation each of these words was quoted into a script and
    read back by ``sh``; here they must arrive at the executor as the
    same list of strings that went in.
    """
    executor = _RecordingExecutor()
    argv = ["eval", "printf HACKED", ";", "$(id)", "*"]

    result = await (_backend(executor)).exec(argv, cwd=None, env=None, timeout=30)

    assert executor.scripts == []
    assert len(executor.commands) == 1
    assert executor.commands[0].argv == tuple(argv)
    # Nothing was asked to be captured, so nothing is streamed anywhere.
    assert executor.output_sinks == [None]
    assert result.argv == argv
    assert result.returncode == 0
    assert (result.stdout, result.stderr) == ("out", "err")


@pytest.mark.asyncio
async def test_exec_passes_the_timeout_environment_and_cwd_through() -> None:
    executor = _RecordingExecutor()

    await (_backend(executor)).exec(
        ["printf", "ok"],
        cwd=Path("/tmp/somewhere"),
        env={"TOKEN": "x"},
        timeout=17,
    )

    command = executor.commands[0]
    assert command.timeout_seconds == 17
    assert command.environment == {"TOKEN": "x"}
    assert executor.working_dirs == ["/tmp/somewhere"]


@pytest.mark.asyncio
async def test_exec_reports_a_missing_exit_code_as_a_failure() -> None:
    """A record with no exit code is not a success."""
    executor = _RecordingExecutor(exit_code=None)

    result = await (_backend(executor)).exec(
        ["printf", "ok"], cwd=None, env=None, timeout=5
    )

    assert result.returncode == -1


@pytest.mark.asyncio
async def test_exec_refuses_an_empty_vector() -> None:
    executor = _RecordingExecutor()

    with pytest.raises(ValueError, match="empty argv"):
        await (_backend(executor)).exec([], cwd=None, env=None, timeout=5)

    assert executor.commands == []


def test_the_backend_still_requires_the_sandboxed_grant() -> None:
    """The execution mode changed; the constitutional gate did not."""
    with pytest.raises(CapabilityBlocked):
        DockerSandboxBackend(granted_capabilities=set())


# ---------------------------------------------------------------------------
# The capture is the stream, not a transcription of the clipped string (#3277)
# ---------------------------------------------------------------------------

# Small, so a capture "past the ceiling" costs kilobytes rather than mebibytes;
# the property is the same at any size.
CEILING = 64 * 1024


def _fed(data: bytes) -> asyncio.StreamReader:
    reader = asyncio.StreamReader()
    reader.feed_data(data)
    reader.feed_eof()
    return reader


class _ContainerRun:
    """The ``docker run`` child, faked at its pipes.

    The streams are real ``asyncio.StreamReader`` objects holding what the
    container wrote, so the executor drains them exactly as it drains a
    real ``docker run``: 64 KiB reads until EOF.
    """

    def __init__(self, stdout: bytes, stderr: bytes, *, returncode: int = 0) -> None:
        self.stdout = _fed(stdout)
        self.stderr = _fed(stderr)
        self.returncode = returncode
        self.pid = None

    async def wait(self) -> int:
        return self.returncode

    def kill(self) -> None:
        pass


class _HungContainerRun:
    """A container that wrote a little and then went quiet until killed.

    What the kill flushes out of the pipe arrives only after the deadline.
    """

    def __init__(self, before: bytes, after_kill: bytes) -> None:
        self.stdout = asyncio.StreamReader()
        self.stdout.feed_data(before)
        self.stderr = asyncio.StreamReader()
        self._after_kill = after_kill
        self.returncode: Optional[int] = None
        self._exited = asyncio.Event()
        self.pid = None

    async def wait(self) -> int:
        await self._exited.wait()
        return self.returncode or 0

    def kill(self) -> None:
        if self.returncode is None:
            self.stdout.feed_data(self._after_kill)
            self.stdout.feed_eof()
            self.stderr.feed_eof()
            self.returncode = -9
            self._exited.set()


class _ControlCommand:
    """``docker rm -f`` / ``docker kill``: exits at once."""

    returncode = 0

    async def wait(self) -> int:
        return 0


def _fake_docker(monkeypatch: pytest.MonkeyPatch, backend, container) -> list[tuple]:
    """Route the executor's ``docker`` CLI calls to ``container``.

    ``run`` gets the container and ``kill`` kills it; every lifecycle
    control command exits 0. Returns the recorded argv of each call.
    """
    calls: list[tuple] = []

    async def create_subprocess_exec(*argv, **_kwargs):
        calls.append(argv)
        if argv[1] == "run":
            return container
        if argv[1] == "kill":
            container.kill()
        return _ControlCommand()

    monkeypatch.setattr(backend._executor, "_get_docker_path", lambda: "/fake/docker")
    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_subprocess_exec)
    return calls


def _real_backend() -> DockerSandboxBackend:
    return DockerSandboxBackend(
        granted_capabilities={"shell_execution_sandboxed"},
        max_output_bytes=CEILING,
    )


def _target(tmp_path: Path) -> CaptureTarget:
    bundle = capture.allocate(tmp_path / "captures")
    return CaptureTarget(stdout_path=bundle.stdout_path, stderr_path=bundle.stderr_path)


@pytest.mark.asyncio
async def test_a_capture_past_the_ceiling_holds_all_of_it_and_is_complete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The defect, through ``shell`` on the DEFAULT backend.

    Both streams run several times past the configured ceiling. Before
    #3277 the files held the first ``max_output_bytes`` of each and the run
    came back ``complete: false``, PARTIAL — the 1 MiB cap #3243 was filed
    against, still in force. Now the files hold every byte, and the result
    and the manifest both say ``complete: true``.
    """
    stdout = b"".join(b"line %06d of the review\n" % i for i in range(12_000))
    stdout += b"VERDICT: no defects found\n"
    stderr = b"progress\n" * 30_000
    assert len(stdout) > 3 * CEILING and len(stderr) > 3 * CEILING

    (tmp_path / "secret").mkdir()
    f = ComputerUseFeature(FakeAgent(queue=FakeApprovalQueue()))
    f._cfg = _config(
        tmp_path, backend="docker", docker={"max_output_bytes": CEILING}
    )
    await f.initialize()
    assert f._backend.name == "docker"
    assert f._backend._executor._max_output_bytes == CEILING
    calls = _fake_docker(monkeypatch, f._backend, _ContainerRun(stdout, stderr))

    env = await f.shell(command="echo hi", capture_output=True)

    assert [c for c in calls if c[1] == "run"], "the container never ran"
    assert env.status is ToolResultStatus.OK, env.error
    assert env.data["complete"] is True
    assert env.data["truncated_stdout"] is False
    assert env.data["truncated_stderr"] is False
    assert Path(env.data["stdout_path"]).read_bytes() == stdout
    assert Path(env.data["stderr_path"]).read_bytes() == stderr
    # The verdict is at the very end, past every ceiling, and the preview
    # shows it without opening the file.
    assert "VERDICT: no defects found" in env.data["stdout"]

    body = json.loads(Path(env.data["manifest_path"]).read_text())
    assert body["backend"] == "docker"
    assert body["complete"] is True
    assert body["stdout_bytes"] == len(stdout)
    assert body["stderr_bytes"] == len(stderr)


@pytest.mark.asyncio
async def test_the_returned_strings_stay_clipped_at_the_ceiling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The ceiling is pinned where it still applies: the copy in memory.

    With a capture, the strings are the executor's first ``max_output_bytes``
    of each stream and the files are the output of record, so the clip is
    not loss. Without one there is no artifact behind the strings, and the
    same clip is reported as truncation.
    """
    stdout = b"o" * (3 * CEILING + 7)
    stderr = b"e" * (2 * CEILING + 3)

    backend = _real_backend()
    _fake_docker(monkeypatch, backend, _ContainerRun(stdout, stderr))
    target = _target(tmp_path)
    captured = await backend.exec(
        ["echo", "hi"], cwd=None, env=None, timeout=30, capture=target
    )

    assert captured.stdout == "o" * CEILING
    assert captured.stderr == "e" * CEILING
    assert (captured.truncated_stdout, captured.truncated_stderr) == (False, False)
    assert target.stdout_path.read_bytes() == stdout
    assert target.stderr_path.read_bytes() == stderr

    backend = _real_backend()
    _fake_docker(monkeypatch, backend, _ContainerRun(stdout, stderr))
    inline = await backend.exec(["echo", "hi"], cwd=None, env=None, timeout=30)

    assert inline.stdout == "o" * CEILING
    assert inline.stderr == "e" * CEILING
    assert (inline.truncated_stdout, inline.truncated_stderr) == (True, True)
    assert inline.stdout_path is None


@pytest.mark.asyncio
async def test_non_utf8_bytes_survive_the_capture_byte_for_byte(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The capture used to be ``record.stdout.encode()`` — a string the
    executor had decoded with ``errors="replace"`` — so every invalid byte
    came back as U+FFFD, and round 9 had to call such a capture incomplete.

    The bytes here include an invalid start byte, a lone continuation byte,
    a NUL, and a multibyte character split across the executor's 64 KiB
    read boundary, which a per-chunk decode would also have mangled. All of
    it reaches the file as emitted, and none of it is reported lost.
    """
    split_char = "é".encode()  # two bytes
    stdout = (
        b"\xff\xfe start "
        + b"x" * (64 * 1024 - 15)
        + split_char  # straddles the first 64 KiB read
        + b" \x80 lone continuation \x00 nul "
        + b"\xc3"  # a truncated sequence at EOF
    )
    stderr = b"warn: \xe2\x82 incomplete euro\n"

    backend = _real_backend()
    _fake_docker(monkeypatch, backend, _ContainerRun(stdout, stderr))
    target = _target(tmp_path)
    result = await backend.exec(
        ["echo", "hi"], cwd=None, env=None, timeout=30, capture=target
    )

    assert target.stdout_path.read_bytes() == stdout
    assert target.stderr_path.read_bytes() == stderr
    assert (result.truncated_stdout, result.truncated_stderr) == (False, False)
    # The returned copy is still a decode, which is why it is not the record.
    assert "\ufffd" in result.stdout


@pytest.mark.asyncio
async def test_a_timed_out_capture_keeps_what_came_before_the_deadline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Through the real executor's timeout path. What the container wrote
    before the deadline is in the file — the old path replaced it with an
    empty one — and what the kill flushed out of the pipe afterwards is not,
    so the stream never reached EOF and is reported lost, not just late."""
    backend = _real_backend()
    _fake_docker(
        monkeypatch,
        backend,
        _HungContainerRun(before=b"partial review\n", after_kill=b"after the kill\n"),
    )
    target = _target(tmp_path)

    result = await backend.exec(
        ["echo", "hi"], cwd=None, env=None, timeout=1, capture=target
    )

    assert result.timed_out is True
    assert target.stdout_path.read_bytes() == b"partial review\n"
    assert (result.truncated_stdout, result.truncated_stderr) == (True, True)
    assert "1s timeout" in result.stderr


class _FailingHandle:
    """A capture handle whose ``write`` or ``close`` fails like a full disk.

    A write fails once, at the budget, and the disk then "recovers": a
    backend that kept writing would leave a hole in the middle of the file
    rather than a file that stops where the loss began.
    """

    def __init__(self, fh, *, fail_write_after: Optional[int], fail_close: bool) -> None:
        self._fh = fh
        self._budget = fail_write_after
        self._fail_close = fail_close

    def write(self, data: bytes) -> int:
        if self._budget is not None:
            if self._budget < len(data):
                self._budget = None
                raise OSError(errno.ENOSPC, "No space left on device")
            self._budget -= len(data)
        return self._fh.write(data)

    def close(self) -> None:
        self._fh.close()
        if self._fail_close:
            raise OSError(errno.EIO, "Input/output error")


@pytest.mark.parametrize("failing", ["out", "err"])
@pytest.mark.parametrize("how", ["write", "close"])
@pytest.mark.asyncio
async def test_a_failed_capture_write_is_lost_output_on_that_stream_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failing: str, how: str
) -> None:
    """A disk that fills mid-capture leaves a file missing its tail. That is
    lost output on that stream — and only that stream: the per-stream
    contract #3243 split the flags for. The drain still reads to the end, so
    the container is never left blocked on a pipe nobody empties, and its
    exit code still comes back."""
    real_open = docker_backend.open_capture

    def open_failing(target):
        out_fh, err_fh = real_open(target)
        wrap = lambda fh: _FailingHandle(
            fh,
            fail_write_after=CEILING if how == "write" else None,
            fail_close=how == "close",
        )
        if failing == "out":
            return wrap(out_fh), err_fh
        return out_fh, wrap(err_fh)

    monkeypatch.setattr(docker_backend, "open_capture", open_failing)
    stdout = b"o" * (4 * CEILING)
    stderr = b"e" * (4 * CEILING)
    backend = _real_backend()
    _fake_docker(monkeypatch, backend, _ContainerRun(stdout, stderr, returncode=3))
    target = _target(tmp_path)

    result = await backend.exec(
        ["echo", "hi"], cwd=None, env=None, timeout=30, capture=target
    )

    assert result.returncode == 3
    assert result.truncated_stdout is (failing == "out")
    assert result.truncated_stderr is (failing == "err")
    assert "capture write failed" in result.stderr
    intact = target.stderr_path if failing == "out" else target.stdout_path
    assert intact.read_bytes() == (stderr if failing == "out" else stdout)
    if how == "write":
        broken = target.stdout_path if failing == "out" else target.stderr_path
        emitted = stdout if failing == "out" else stderr
        # Everything before the failed write, and nothing after it.
        assert broken.read_bytes() == emitted[:CEILING]


@pytest.mark.asyncio
async def test_an_unopenable_capture_is_reported_lost_and_runs_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Defaults would have said "nothing truncated, no writers" over a file
    that was never opened, and the run would have gone ahead with nowhere to
    put its output."""

    def refuse(_target):
        raise OSError(errno.EACCES, "Permission denied")

    monkeypatch.setattr(docker_backend, "open_capture", refuse)
    backend = _real_backend()
    calls = _fake_docker(monkeypatch, backend, _ContainerRun(b"x", b""))

    result = await backend.exec(
        ["echo", "hi"], cwd=None, env=None, timeout=30, capture=_target(tmp_path)
    )

    assert calls == []
    assert (result.truncated_stdout, result.truncated_stderr) == (True, True)
    assert result.writers_remaining is None
    assert "could not open capture file" in result.stderr


@pytest.mark.asyncio
async def test_a_drain_that_outlives_the_run_cannot_write_into_the_artifact(
    tmp_path: Path,
) -> None:
    """The files are closed when ``exec`` returns, and their completeness is
    decided then. A drain the executor abandoned — it keeps no promise to
    stop at that moment — must be refused rather than write into a closed
    file, or flip a stream to "ended" after the verdict on it was given."""
    stash: list[OutputSinks] = []

    class _Executor:
        async def execute_command(self, command, working_dir=None, *, output_sinks=None):
            stash.append(output_sinks)
            await output_sinks.stdout(b"early\n")
            await output_sinks.stdout(b"")
            await output_sinks.stderr(b"")
            return ExecutionRecord(
                id="exec-1", script_id=command.id, exit_code=0, executor="docker"
            )

    backend = DockerSandboxBackend(granted_capabilities={"shell_execution_sandboxed"})
    backend._executor = _Executor()  # type: ignore[assignment]
    target = _target(tmp_path)

    result = await backend.exec(
        ["echo", "hi"], cwd=None, env=None, timeout=30, capture=target
    )
    await stash[0].stdout(b"late\n")
    await stash[0].stderr(b"late\n")

    assert result.truncated_stdout is False
    assert target.stdout_path.read_bytes() == b"early\n"
    assert target.stderr_path.read_bytes() == b""


@pytest.mark.asyncio
async def test_an_eof_while_the_files_close_does_not_make_them_whole(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The narrow window behind the retire guard. Completeness is read after
    the close, and a close is a flush that can take a while. A drain the
    executor abandoned can deliver more output and then EOF inside that
    wait: the output is refused (the file is closing), so the EOF must be
    too, or a stream missing its tail is filed as one read to the end."""
    closing = asyncio.Event()
    loop = asyncio.get_running_loop()
    real_open = docker_backend.open_capture

    class _SlowClose:
        def __init__(self, fh) -> None:
            self._fh = fh

        def write(self, data: bytes) -> int:
            return self._fh.write(data)

        def close(self) -> None:
            loop.call_soon_threadsafe(closing.set)
            time.sleep(0.2)
            self._fh.close()

    def open_slow(target):
        out_fh, err_fh = real_open(target)
        return out_fh, _SlowClose(err_fh)

    monkeypatch.setattr(docker_backend, "open_capture", open_slow)
    late: list[asyncio.Task] = []

    class _Executor:
        async def execute_command(self, command, working_dir=None, *, output_sinks=None):
            await output_sinks.stdout(b"")
            await output_sinks.stderr(b"early\n")

            async def abandoned_drain() -> None:
                await closing.wait()
                await output_sinks.stderr(b"late\n")
                await output_sinks.stderr(b"")

            late.append(asyncio.create_task(abandoned_drain()))
            return ExecutionRecord(
                id="exec-1", script_id=command.id, exit_code=0, executor="docker"
            )

    backend = DockerSandboxBackend(granted_capabilities={"shell_execution_sandboxed"})
    backend._executor = _Executor()  # type: ignore[assignment]
    target = _target(tmp_path)

    result = await backend.exec(
        ["echo", "hi"], cwd=None, env=None, timeout=30, capture=target
    )
    try:
        assert closing.is_set(), "the stderr file was never closed"
        await late[0]
    finally:
        late[0].cancel()

    assert target.stderr_path.read_bytes() == b"early\n"
    assert result.truncated_stderr is True
    assert result.truncated_stdout is False


@pytest.mark.asyncio
async def test_no_docker_raises_and_leaves_no_capture_files_behind(
    tmp_path: Path,
) -> None:
    """With no ``docker`` binary nothing runs and ``exec`` raises, as before
    #3277. The capture files are opened before the run now, so they must not
    outlive it: no manifest will ever name them, and pruning retires a set by
    its manifest — on a host without Docker every captured call would leave
    two more empty files, forever."""
    backend = _real_backend()
    backend._executor._get_docker_path = lambda: None  # type: ignore[method-assign]
    target = _target(tmp_path)

    with pytest.raises(ExecutionEnvironmentError, match="Docker not found"):
        await backend.exec(
            ["echo", "hi"], cwd=None, env=None, timeout=30, capture=target
        )

    assert not target.stdout_path.exists()
    assert not target.stderr_path.exists()


@pytest.mark.asyncio
async def test_a_stream_annotated_with_a_diagnostic_stays_lost(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Found by review. The diagnostic goes into the stderr file only when
    the executor gave up on that stream, so the file is already short. A
    drain still running while the note is written must not then deliver its
    tail and an EOF around it: the stream would read as ended, and a file
    with runtime text spliced into the container's output would be filed as
    whole."""
    note = b"Docker cleanup failed\n"
    writing_note = asyncio.Event()
    loop = asyncio.get_running_loop()
    real_open = docker_backend.open_capture

    class _SlowNote:
        def __init__(self, fh) -> None:
            self._fh = fh

        def write(self, data: bytes) -> int:
            if data == note:
                loop.call_soon_threadsafe(writing_note.set)
                time.sleep(0.2)
            return self._fh.write(data)

        def close(self) -> None:
            self._fh.close()

    def open_slow(target):
        out_fh, err_fh = real_open(target)
        return out_fh, _SlowNote(err_fh)

    monkeypatch.setattr(docker_backend, "open_capture", open_slow)
    late: list[asyncio.Task] = []

    class _Executor:
        async def execute_command(self, command, working_dir=None, *, output_sinks=None):
            await output_sinks.stdout(b"")
            await output_sinks.stderr(b"early\n")

            async def abandoned_drain() -> None:
                await writing_note.wait()
                await output_sinks.stderr(b"tail\n")
                await output_sinks.stderr(b"")

            late.append(asyncio.create_task(abandoned_drain()))
            return ExecutionRecord(
                id="exec-1",
                script_id=command.id,
                exit_code=-1,
                stderr=note.decode().rstrip("\n"),
                executor="docker",
            )

    backend = DockerSandboxBackend(granted_capabilities={"shell_execution_sandboxed"})
    backend._executor = _Executor()  # type: ignore[assignment]
    target = _target(tmp_path)

    result = await backend.exec(
        ["echo", "hi"], cwd=None, env=None, timeout=30, capture=target
    )
    try:
        assert writing_note.is_set(), "the diagnostic never reached the file"
        await late[0]
    finally:
        late[0].cancel()

    assert result.truncated_stderr is True
    assert target.stderr_path.read_bytes() == b"early\n" + note
