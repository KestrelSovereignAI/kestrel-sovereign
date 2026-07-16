"""Focused lifecycle tests for the built-in compute executors."""

import asyncio
import logging
import os
import signal
from pathlib import Path
from typing import Optional

import pytest

from kestrel_sovereign.features.compute.executors import (
    BaseExecutor,
    DockerExecutor,
    ExecutionTimeoutError,
    LocalExecutor,
    UvExecutor,
)
from kestrel_sovereign.features.compute.executors import base as executor_base
from kestrel_sovereign.features.compute.executors import (
    docker_executor as docker_executor_module,
)
from kestrel_sovereign.features.compute.models import ComputeScript, ExecutionRecord


EXECUTOR_NAMES = ("local", "uv", "docker")
TRUNCATED_SUFFIX = "\n... [output truncated]"


class _CoordinatedStream:
    """A finite stream that proves stdout and stderr drains start together."""

    def __init__(
        self,
        content: bytes,
        started: asyncio.Event,
        peer_started: asyncio.Event,
    ) -> None:
        self._content = content
        self._offset = 0
        self._started = started
        self._peer_started = peer_started
        self.read_limits: list[int] = []

    async def read(self, limit: int) -> bytes:
        self.read_limits.append(limit)
        if not self._started.is_set():
            self._started.set()
            await self._peer_started.wait()
        if self._offset >= len(self._content):
            return b""

        chunk = self._content[self._offset : self._offset + limit]
        self._offset += len(chunk)
        return chunk


class _SuccessfulProcess:
    def __init__(self, stdout: bytes, stderr: bytes) -> None:
        stdout_started = asyncio.Event()
        stderr_started = asyncio.Event()
        self.stdout = _CoordinatedStream(stdout, stdout_started, stderr_started)
        self.stderr = _CoordinatedStream(stderr, stderr_started, stdout_started)
        self.returncode: Optional[int] = 0
        self.kill_calls = 0
        self.wait_calls = 0

    async def wait(self) -> int:
        self.wait_calls += 1
        return self.returncode or 0

    def kill(self) -> None:
        self.kill_calls += 1
        self.returncode = -9


class _BlockingStream:
    def __init__(self, released: asyncio.Event, started: asyncio.Event) -> None:
        self._released = released
        self.started = started

    async def read(self, _limit: int) -> bytes:
        self.started.set()
        await self._released.wait()
        return b""


class _BlockingProcess:
    def __init__(self) -> None:
        self.released = asyncio.Event()
        self.stdout_started = asyncio.Event()
        self.stderr_started = asyncio.Event()
        self.stdout = _BlockingStream(self.released, self.stdout_started)
        self.stderr = _BlockingStream(self.released, self.stderr_started)
        self.returncode: Optional[int] = None
        self._pending_returncode: Optional[int] = None
        self.kill_calls = 0
        self.wait_calls = 0

    async def wait(self) -> int:
        self.wait_calls += 1
        await self.released.wait()
        if self.returncode is None:
            self.returncode = self._pending_returncode or 0
        return self.returncode or 0

    def kill(self) -> None:
        self.kill_calls += 1
        self._pending_returncode = -9
        self.released.set()

    def container_stopped(self) -> None:
        self.returncode = 137
        self.released.set()


class _NeverEndingStream:
    """A pipe held open by a descendant after the direct child exits."""

    async def read(self, _limit: int) -> bytes:
        await asyncio.Event().wait()
        raise AssertionError("unreachable")


class _StubbornPipeProcess:
    def __init__(self, *, pid: Optional[int] = None) -> None:
        self.pid = pid
        self.stdout = _NeverEndingStream()
        self.stderr = _NeverEndingStream()
        self.returncode: Optional[int] = None
        self._exited = asyncio.Event()
        self.kill_calls = 0
        self.wait_calls = 0

    async def wait(self) -> int:
        self.wait_calls += 1
        await self._exited.wait()
        self.returncode = -9
        return self.returncode

    def kill(self) -> None:
        self.kill_calls += 1
        self._exited.set()


class _CompletedProcess:
    returncode = 0

    async def wait(self) -> int:
        return self.returncode


class _ControlledCommandProcess:
    """A Docker control process completed explicitly or by a forced kill."""

    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.released = asyncio.Event()
        self.returncode: Optional[int] = None
        self.kill_calls = 0

    async def wait(self) -> int:
        self.started.set()
        await self.released.wait()
        return self.returncode or 0

    def complete(self, returncode: int = 0) -> None:
        self.returncode = returncode
        self.released.set()

    def kill(self) -> None:
        self.kill_calls += 1
        self.complete(-9)


class _CompatibilityExecutor(BaseExecutor):
    """Small public-style subclass used to pin the base constructor API."""

    @property
    def name(self) -> str:
        return "compatibility"

    @property
    def is_available(self) -> bool:
        return True

    async def execute(
        self,
        script: ComputeScript,
        working_dir: Optional[str] = None,
    ) -> ExecutionRecord:
        raise NotImplementedError


def _make_executor(monkeypatch: pytest.MonkeyPatch, name: str, max_bytes: int = 4):
    if name == "local":
        return LocalExecutor(
            max_output_bytes=max_bytes,
            require_env_flag=False,
        )
    if name == "uv":
        executor = UvExecutor(max_output_bytes=max_bytes)
        monkeypatch.setattr(executor, "_get_uv_path", lambda: "/fake/uv")
        return executor
    if name == "docker":
        executor = DockerExecutor(max_output_bytes=max_bytes)
        monkeypatch.setattr(executor, "_get_docker_path", lambda: "/fake/docker")
        return executor
    raise AssertionError(f"Unknown executor: {name}")


def _script(
    *,
    timeout_seconds: int = 1,
    environment: Optional[dict[str, str]] = None,
) -> ComputeScript:
    return ComputeScript(
        id="executor-lifecycle-test",
        name="executor lifecycle",
        language="python",
        content="print('ok')",
        purpose="exercise the shared executor lifecycle",
        timeout_seconds=timeout_seconds,
        environment=environment or {},
    )


def _track_temp_dirs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> list[Path]:
    created: list[Path] = []

    def make_temp_dir(*, prefix: str) -> str:
        path = tmp_path / f"{prefix}{len(created)}"
        path.mkdir()
        created.append(path)
        return str(path)

    monkeypatch.setattr(executor_base.tempfile, "mkdtemp", make_temp_dir)
    return created


def _is_docker_command(command: tuple[object, ...], action: str) -> bool:
    return len(command) > 1 and command[1] == action


def test_base_executor_preserves_no_argument_subclass_construction() -> None:
    executor = _CompatibilityExecutor()

    assert executor._max_output_bytes == 1024 * 1024


@pytest.mark.asyncio
async def test_owned_task_failure_during_repeated_cancellation_keeps_cancellation(
    caplog: pytest.LogCaptureFixture,
) -> None:
    started = asyncio.Event()
    release = asyncio.Event()
    failure = RuntimeError("owned task failed after cancellation")
    unhandled: list[dict[str, object]] = []
    loop = asyncio.get_running_loop()
    prior_handler = loop.get_exception_handler()

    async def fail_after_release() -> None:
        started.set()
        await release.wait()
        raise failure

    owned = asyncio.create_task(fail_after_release())
    waiter = asyncio.create_task(BaseExecutor._await_owned_task(owned))
    await asyncio.wait_for(started.wait(), timeout=1)
    loop.set_exception_handler(lambda _loop, context: unhandled.append(context))
    caplog.set_level(logging.WARNING)
    try:
        waiter.cancel("first caller cancellation")
        await asyncio.sleep(0)
        waiter.cancel("second caller cancellation")
        await asyncio.sleep(0)
        assert not waiter.done()

        release.set()
        with pytest.raises(asyncio.CancelledError) as exc_info:
            await asyncio.wait_for(waiter, timeout=1)
        await asyncio.sleep(0)
    finally:
        loop.set_exception_handler(prior_handler)

    assert waiter.cancelled()
    assert exc_info.value.__cause__ is failure
    assert owned.done()
    assert unhandled == []
    assert "owned task failed after cancellation" in caplog.text


@pytest.mark.asyncio
async def test_capture_failure_during_repeated_cancellation_keeps_cancellation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executor = _CompatibilityExecutor()
    process = _BlockingProcess()
    teardown_started = asyncio.Event()
    release_teardown = asyncio.Event()
    failure = RuntimeError("simulated teardown failure")

    async def fail_teardown(*_args: object) -> None:
        teardown_started.set()
        await release_teardown.wait()
        raise failure

    async def terminate() -> None:
        raise AssertionError("replacement teardown owns this path")

    monkeypatch.setattr(executor, "_bounded_terminate_and_reap", fail_teardown)
    capture = asyncio.create_task(
        executor._capture_process_output(
            process,
            timeout_seconds=60,
            terminate=terminate,
        )
    )
    await asyncio.wait_for(
        asyncio.gather(
            process.stdout_started.wait(),
            process.stderr_started.wait(),
        ),
        timeout=1,
    )
    capture.cancel("first caller cancellation")
    await asyncio.wait_for(teardown_started.wait(), timeout=1)
    capture.cancel("second caller cancellation")
    await asyncio.sleep(0)
    assert not capture.done()

    release_teardown.set()
    with pytest.raises(asyncio.CancelledError) as exc_info:
        await asyncio.wait_for(capture, timeout=1)

    assert capture.cancelled()
    assert exc_info.value.__cause__ is failure


@pytest.mark.parametrize("executor_name", EXECUTOR_NAMES)
@pytest.mark.asyncio
async def test_success_uses_concurrent_byte_bounded_capture_and_cleans_workdir(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    executor_name: str,
) -> None:
    executor = _make_executor(monkeypatch, executor_name)
    created = _track_temp_dirs(monkeypatch, tmp_path)
    process = _SuccessfulProcess(
        b"A" + "é".encode() * 40_000,
        b"HEAD" + b"x" * 70_000 + b"TAIL",
    )
    commands: list[tuple[object, ...]] = []

    async def create_subprocess(*command: object, **_kwargs: object):
        commands.append(command)
        if _is_docker_command(command, "rm"):
            return _CompletedProcess()
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_subprocess)

    record = await asyncio.wait_for(executor.execute(_script()), timeout=2)

    assert isinstance(record, ExecutionRecord)
    assert record.script_id == "executor-lifecycle-test"
    assert record.executor == executor_name
    assert record.exit_code == 0
    assert record.stdout == f"Aé�{TRUNCATED_SUFFIX}"
    assert record.stderr == f"HEAD{TRUNCATED_SUFFIX}"
    if executor_name == "docker":
        assert record.container_id is not None
    else:
        assert record.container_id is None
    assert record.started_at <= record.completed_at
    assert record.workdir == str(created[0])
    assert not created[0].exists()
    assert process.wait_calls == 1
    assert len(process.stdout.read_limits) >= 3
    assert len(process.stderr.read_limits) >= 3
    assert set(process.stdout.read_limits) == {64 * 1024}
    assert set(process.stderr.read_limits) == {64 * 1024}
    assert any(_is_docker_command(command, "rm") for command in commands) == (
        executor_name == "docker"
    )


@pytest.mark.parametrize("executor_name", EXECUTOR_NAMES)
@pytest.mark.asyncio
async def test_launch_failure_returns_typed_record_and_cleans_workdir(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    executor_name: str,
) -> None:
    executor = _make_executor(monkeypatch, executor_name)
    created = _track_temp_dirs(monkeypatch, tmp_path)

    async def create_subprocess(*command: object, **_kwargs: object):
        if _is_docker_command(command, "rm"):
            return _CompletedProcess()
        raise OSError("simulated subprocess launch failure")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_subprocess)

    record = await asyncio.wait_for(executor.execute(_script()), timeout=2)

    assert isinstance(record, ExecutionRecord)
    assert record.executor == executor_name
    assert record.exit_code == -1
    assert record.stdout == ""
    assert record.stderr == "simulated subprocess launch failure"
    assert record.container_id is None
    assert record.workdir == str(created[0])
    assert not created[0].exists()


@pytest.mark.parametrize("executor_name", EXECUTOR_NAMES)
@pytest.mark.asyncio
async def test_timeout_terminates_reaps_propagates_and_cleans_workdir(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    executor_name: str,
) -> None:
    executor = _make_executor(monkeypatch, executor_name)
    created = _track_temp_dirs(monkeypatch, tmp_path)
    process = _BlockingProcess()
    commands: list[tuple[object, ...]] = []

    async def create_subprocess(*command: object, **_kwargs: object):
        commands.append(command)
        if _is_docker_command(command, "kill"):
            process.container_stopped()
            return _CompletedProcess()
        if _is_docker_command(command, "rm"):
            return _CompletedProcess()
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_subprocess)

    with pytest.raises(ExecutionTimeoutError) as exc_info:
        await asyncio.wait_for(
            executor.execute(_script(timeout_seconds=1)),
            timeout=2,
        )

    assert exc_info.value.script_id == "executor-lifecycle-test"
    assert exc_info.value.timeout_seconds == 1
    assert not created[0].exists()
    assert process.wait_calls >= 1
    if executor_name == "docker":
        assert any(_is_docker_command(command, "kill") for command in commands)
        assert any(_is_docker_command(command, "rm") for command in commands)
        assert process.kill_calls == 0
    else:
        assert process.kill_calls >= 1


@pytest.mark.parametrize("executor_name", EXECUTOR_NAMES)
@pytest.mark.asyncio
async def test_cancellation_terminates_reaps_and_cleans_workdir(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    executor_name: str,
) -> None:
    executor = _make_executor(monkeypatch, executor_name)
    created = _track_temp_dirs(monkeypatch, tmp_path)
    process = _BlockingProcess()
    commands: list[tuple[object, ...]] = []

    async def create_subprocess(*command: object, **_kwargs: object):
        commands.append(command)
        if _is_docker_command(command, "kill"):
            process.container_stopped()
            return _CompletedProcess()
        if _is_docker_command(command, "rm"):
            return _CompletedProcess()
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_subprocess)

    execution = asyncio.create_task(executor.execute(_script(timeout_seconds=60)))
    await asyncio.wait_for(
        asyncio.gather(
            process.stdout_started.wait(),
            process.stderr_started.wait(),
        ),
        timeout=1,
    )
    execution.cancel()

    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(execution, timeout=2)

    assert not created[0].exists()
    assert process.wait_calls >= 2
    if executor_name == "docker":
        assert any(_is_docker_command(command, "kill") for command in commands)
        assert any(_is_docker_command(command, "rm") for command in commands)
        assert process.kill_calls == 0
    else:
        assert process.kill_calls >= 1


@pytest.mark.asyncio
async def test_repeated_docker_cancellation_waits_for_removal_then_propagates(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    executor = _make_executor(monkeypatch, "docker")
    created = _track_temp_dirs(monkeypatch, tmp_path)
    run_process = _SuccessfulProcess(b"ok", b"")
    remove_process = _ControlledCommandProcess()

    async def create_subprocess(*command: object, **_kwargs: object):
        if _is_docker_command(command, "rm"):
            return remove_process
        return run_process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_subprocess)

    execution = asyncio.create_task(executor.execute(_script(timeout_seconds=60)))
    await asyncio.wait_for(remove_process.started.wait(), timeout=1)
    execution.cancel()
    await asyncio.sleep(0)

    assert not execution.done()
    execution.cancel()
    await asyncio.sleep(0)

    assert not execution.done()
    remove_process.complete()

    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(execution, timeout=1)

    assert remove_process.kill_calls == 0
    assert not created[0].exists()


@pytest.mark.asyncio
async def test_hung_docker_remove_is_bounded_and_reaps_control_process(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    executor = _make_executor(monkeypatch, "docker")
    created = _track_temp_dirs(monkeypatch, tmp_path)
    run_process = _SuccessfulProcess(b"ok", b"")
    remove_process = _ControlledCommandProcess()

    async def create_subprocess(*command: object, **_kwargs: object):
        if _is_docker_command(command, "rm"):
            return remove_process
        return run_process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_subprocess)
    monkeypatch.setattr(docker_executor_module, "SUBPROCESS_TIMEOUT_SHORT", 0.01)
    monkeypatch.setattr(
        docker_executor_module,
        "_DOCKER_CONTROL_REAP_TIMEOUT_SECONDS",
        0.01,
    )

    record = await asyncio.wait_for(executor.execute(_script()), timeout=0.5)

    assert record.exit_code == 0
    assert remove_process.kill_calls == 1
    assert remove_process.returncode == -9
    assert not created[0].exists()


@pytest.mark.asyncio
async def test_hung_docker_kill_is_bounded_and_preserves_primary_timeout(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    executor = _make_executor(monkeypatch, "docker")
    created = _track_temp_dirs(monkeypatch, tmp_path)
    run_process = _BlockingProcess()
    kill_process = _ControlledCommandProcess()

    async def create_subprocess(*command: object, **_kwargs: object):
        if _is_docker_command(command, "kill"):
            return kill_process
        if _is_docker_command(command, "rm"):
            return _CompletedProcess()
        return run_process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_subprocess)
    monkeypatch.setattr(docker_executor_module, "SUBPROCESS_TIMEOUT_SHORT", 0.01)
    monkeypatch.setattr(
        docker_executor_module,
        "_DOCKER_CONTROL_REAP_TIMEOUT_SECONDS",
        0.01,
    )

    with pytest.raises(ExecutionTimeoutError):
        await asyncio.wait_for(
            executor.execute(_script(timeout_seconds=0.01)),
            timeout=0.5,
        )

    assert kill_process.kill_calls == 1
    assert kill_process.returncode == -9
    assert run_process.kill_calls >= 1
    assert not created[0].exists()


@pytest.mark.asyncio
async def test_docker_cleanup_failure_does_not_mask_primary_timeout(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    executor = _make_executor(monkeypatch, "docker")
    created = _track_temp_dirs(monkeypatch, tmp_path)
    run_process = _BlockingProcess()

    async def create_subprocess(*command: object, **_kwargs: object):
        if _is_docker_command(command, "kill"):
            run_process.container_stopped()
            return _CompletedProcess()
        return run_process

    async def fail_cleanup(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("simulated cleanup failure")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_subprocess)
    monkeypatch.setattr(executor, "_remove_container", fail_cleanup)

    with pytest.raises(ExecutionTimeoutError):
        await asyncio.wait_for(
            executor.execute(_script(timeout_seconds=0.01)),
            timeout=0.5,
        )

    assert not created[0].exists()


@pytest.mark.parametrize("executor_name", ("local", "uv"))
@pytest.mark.asyncio
async def test_host_executor_timeout_is_bounded_when_descendant_holds_pipes_open(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    executor_name: str,
) -> None:
    executor = _make_executor(monkeypatch, executor_name)
    created = _track_temp_dirs(monkeypatch, tmp_path)
    process = _StubbornPipeProcess()
    subprocess_options: dict[str, object] = {}

    async def create_subprocess(*_command: object, **kwargs: object):
        subprocess_options.update(kwargs)
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_subprocess)
    monkeypatch.setattr(executor_base, "_TERMINATION_REAP_TIMEOUT_SECONDS", 0.01)

    with pytest.raises(ExecutionTimeoutError):
        await asyncio.wait_for(
            executor.execute(_script(timeout_seconds=0.01)),
            timeout=0.5,
        )

    assert subprocess_options["start_new_session"] is (os.name == "posix")
    assert process.returncode == -9
    assert process.kill_calls >= 1
    assert process.wait_calls >= 1
    assert not created[0].exists()


@pytest.mark.skipif(os.name != "posix", reason="process groups are POSIX-only")
@pytest.mark.asyncio
async def test_host_executor_timeout_kills_the_child_process_group(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    executor = _make_executor(monkeypatch, "local")
    created = _track_temp_dirs(monkeypatch, tmp_path)
    process = _StubbornPipeProcess(pid=4242)
    killpg_calls: list[tuple[int, signal.Signals]] = []

    async def create_subprocess(*_command: object, **_kwargs: object):
        return process

    def killpg(process_group_id: int, sig: signal.Signals) -> None:
        killpg_calls.append((process_group_id, sig))

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_subprocess)
    monkeypatch.setattr(os, "killpg", killpg)
    monkeypatch.setattr(executor_base, "_TERMINATION_REAP_TIMEOUT_SECONDS", 0.01)

    with pytest.raises(ExecutionTimeoutError):
        await asyncio.wait_for(
            executor.execute(_script(timeout_seconds=0.01)),
            timeout=0.5,
        )

    assert killpg_calls == [(4242, signal.SIGKILL)]
    assert process.kill_calls >= 1
    assert not created[0].exists()


@pytest.mark.asyncio
async def test_docker_debug_log_redacts_script_environment_values(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    executor = _make_executor(monkeypatch, "docker", max_bytes=128)
    _track_temp_dirs(monkeypatch, tmp_path)
    process = _SuccessfulProcess(b"ok", b"")
    commands: list[tuple[object, ...]] = []

    async def create_subprocess(*command: object, **_kwargs: object):
        commands.append(command)
        if _is_docker_command(command, "rm"):
            return _CompletedProcess()
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_subprocess)
    caplog.set_level(logging.DEBUG)

    record = await asyncio.wait_for(
        executor.execute(
            _script(environment={"COMPUTE_TOKEN": "do-not-log-this-secret"})
        ),
        timeout=2,
    )

    assert record.stdout == "ok"
    assert "do-not-log-this-secret" not in caplog.text
    assert "COMPUTE_TOKEN=<redacted>" in caplog.text
    run_command = next(
        command for command in commands if _is_docker_command(command, "run")
    )
    assert "COMPUTE_TOKEN=do-not-log-this-secret" in run_command


class TestDockerTrashStaging:
    """The Docker executor mounts a per-execution staging dir, never the
    shared trash root (#2485 review P1); staged entries are promoted
    host-side after the run."""

    def test_promote_moves_entries_and_removes_staging(self, tmp_path):
        from kestrel_sovereign.features.compute.executors.docker_executor import (
            DockerExecutor,
        )

        trash_root = tmp_path / "trash"
        trash_root.mkdir()
        staging = trash_root / ".staging-x"
        (staging / "rm_aaaa1111").mkdir(parents=True)
        (staging / "rm_aaaa1111" / "victim.txt").write_text("v")

        DockerExecutor._promote_staged_trash(staging, trash_root)

        assert not staging.exists()
        assert (trash_root / "rm_aaaa1111" / "victim.txt").read_text() == "v"

    def test_promote_suffixes_on_collision(self, tmp_path):
        from kestrel_sovereign.features.compute.executors.docker_executor import (
            DockerExecutor,
        )

        trash_root = tmp_path / "trash"
        (trash_root / "rm_aaaa1111").mkdir(parents=True)
        (trash_root / "rm_aaaa1111" / "old.txt").write_text("old")
        staging = trash_root / ".staging-y"
        (staging / "rm_aaaa1111").mkdir(parents=True)
        (staging / "rm_aaaa1111" / "new.txt").write_text("new")

        DockerExecutor._promote_staged_trash(staging, trash_root)

        assert (trash_root / "rm_aaaa1111" / "old.txt").read_text() == "old"
        assert (trash_root / "rm_aaaa1111.1" / "new.txt").read_text() == "new"
        assert not staging.exists()

    def test_promote_missing_staging_is_a_noop(self, tmp_path):
        from kestrel_sovereign.features.compute.executors.docker_executor import (
            DockerExecutor,
        )

        DockerExecutor._promote_staged_trash(
            tmp_path / "trash" / ".staging-gone", tmp_path / "trash"
        )
