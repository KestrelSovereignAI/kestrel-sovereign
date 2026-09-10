"""Focused lifecycle tests for the built-in compute executors."""

import asyncio
import hashlib
import json
import logging
import os
import shutil
import signal
import socket
import sqlite3
import subprocess
import sys
import tempfile
import venv
import zipfile
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
from kestrel_sovereign.features.compute.executors import (
    base as executor_base,
)
from kestrel_sovereign.features.compute.executors import (
    docker_executor as docker_executor_module,
)
from kestrel_sovereign.features.compute.executors import (
    uv_executor as uv_executor_module,
)
from kestrel_sovereign.features.compute.models import (
    ComputeCommand,
    ComputeScript,
    ExecutionRecord,
)

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


@pytest.fixture(autouse=True)
def _isolated_trash_root(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Every executor in this module trashes into a per-test root.

    ``DEFAULT_TRASH_DIR`` is frozen at import (#3104), so without this the
    Docker executor tests staged into the operator's real ``~/.kestrel/trash``
    and left owner records there; the next test's sweep then found those
    dead-owner records and asked the (faked) docker client about their
    containers, consuming the test's own process doubles (#3117).
    """
    from kestrel_sovereign.features.compute import destructive_policy, trash_manager

    root = tmp_path / "trash"
    monkeypatch.setattr(destructive_policy, "DEFAULT_TRASH_DIR", root)
    monkeypatch.setattr(trash_manager, "DEFAULT_TRASH_DIR", root)
    yield root


def _make_executor(monkeypatch: pytest.MonkeyPatch, name: str, max_bytes: int = 4):
    if name == "local":
        return LocalExecutor(
            max_output_bytes=max_bytes,
            require_env_flag=False,
        )
    if name == "uv":
        executor = UvExecutor(max_output_bytes=max_bytes)
        monkeypatch.setattr(executor, "_get_uv_path", lambda: "/fake/uv")
        monkeypatch.setattr(
            executor,
            "_get_base_python_path",
            lambda: "/fake/base/python",
        )
        monkeypatch.setattr(
            executor,
            "_get_filesystem_sandbox_prefix",
            lambda _writable_workspace=None, **_kwargs: [
                "/fake/filesystem-sandbox",
                "--",
            ],
        )
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


def _write_test_wheel(
    wheel_dir: Path,
    distribution: str,
    module: str,
    value: str,
) -> Path:
    """Create a minimal local wheel without invoking a package index."""
    wheel_dir.mkdir(parents=True, exist_ok=True)
    normalized = distribution.replace("-", "_")
    wheel_path = wheel_dir / f"{normalized}-1.0.0-py3-none-any.whl"
    dist_info = f"{normalized}-1.0.0.dist-info"
    with zipfile.ZipFile(wheel_path, "w") as wheel:
        wheel.writestr(f"{module}.py", f"VALUE = {value!r}\n")
        wheel.writestr(
            f"{dist_info}/METADATA",
            "Metadata-Version: 2.1\n"
            f"Name: {distribution}\n"
            "Version: 1.0.0\n",
        )
        wheel.writestr(
            f"{dist_info}/WHEEL",
            "Wheel-Version: 1.0\n"
            "Generator: kestrel-test\n"
            "Root-Is-Purelib: true\n"
            "Tag: py3-none-any\n",
        )
        wheel.writestr(
            f"{dist_info}/RECORD",
            f"{module}.py,,\n"
            f"{dist_info}/METADATA,,\n"
            f"{dist_info}/WHEEL,,\n"
            f"{dist_info}/RECORD,,\n",
        )
    return wheel_path


def _tree_manifest(root: Path) -> list[tuple[str, str]]:
    """Capture paths and content to detect writes to a synthetic host venv."""
    manifest: list[tuple[str, str]] = []
    for path in sorted(root.rglob("*")):
        relative = str(path.relative_to(root))
        if path.is_symlink():
            manifest.append((relative, f"link:{os.readlink(path)}"))
        elif path.is_file():
            with path.open("rb") as file:
                digest = hashlib.file_digest(file, "sha256").hexdigest()
            manifest.append((relative, f"file:{digest}"))
        else:
            manifest.append((relative, "dir"))
    return manifest


def test_base_executor_preserves_no_argument_subclass_construction() -> None:
    executor = _CompatibilityExecutor()

    assert executor._max_output_bytes == 1024 * 1024


def test_uv_base_python_prefers_canonical_base_executable(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runtime_prefix = tmp_path / "runtime venv"
    base_prefix = tmp_path / "base install"
    executable_name = "python.exe" if os.name == "nt" else "python3.14"
    base_executable = tmp_path / "nonstandard bin" / executable_name
    base_executable.parent.mkdir()
    base_executable.write_text("")
    base_executable.chmod(0o700)

    monkeypatch.setattr(uv_executor_module.sys, "prefix", str(runtime_prefix))
    monkeypatch.setattr(uv_executor_module.sys, "base_prefix", str(base_prefix))
    monkeypatch.setattr(
        uv_executor_module.sys,
        "_base_executable",
        str(base_executable),
    )

    assert UvExecutor._get_base_python_path() == str(base_executable.resolve())


def test_uv_base_python_falls_back_to_versioned_posix_executable(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    if os.name == "nt":
        pytest.skip("POSIX executable layout test")

    runtime_prefix = tmp_path / "runtime venv"
    base_prefix = tmp_path / "base install"
    base_executable = (
        base_prefix
        / "bin"
        / f"python{sys.version_info.major}.{sys.version_info.minor}"
    )
    base_executable.parent.mkdir(parents=True)
    base_executable.write_text("")
    base_executable.chmod(0o700)

    monkeypatch.setattr(uv_executor_module.sys, "prefix", str(runtime_prefix))
    monkeypatch.setattr(uv_executor_module.sys, "base_prefix", str(base_prefix))
    monkeypatch.delattr(uv_executor_module.sys, "_base_executable", raising=False)

    assert UvExecutor._get_base_python_path() == str(base_executable.resolve())


def test_uv_base_python_fails_closed_outside_supported_virtual_environment(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(uv_executor_module.sys, "prefix", str(tmp_path))
    monkeypatch.setattr(uv_executor_module.sys, "base_prefix", str(tmp_path))

    with pytest.raises(
        executor_base.ExecutionEnvironmentError,
        match="requires Kestrel to run inside a Python venv or virtualenv",
    ) as exc_info:
        UvExecutor._get_base_python_path()

    assert "Conda environment alone is not sufficient" in str(exc_info.value)


def test_uv_path_resolves_package_manager_symlink_before_sandbox_mount(
    tmp_path: Path,
) -> None:
    """The command invoked in bwrap must be the mounted executable target."""

    installed_uv = tmp_path / "store" / "uv-1.2.3" / "bin" / "uv"
    installed_uv.parent.mkdir(parents=True)
    installed_uv.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    installed_uv.chmod(0o755)
    alias = tmp_path / "bin" / "uv"
    alias.parent.mkdir()
    alias.symlink_to(installed_uv)

    executor = UvExecutor(uv_path=str(alias))

    assert executor._get_uv_path() == str(installed_uv.resolve())


def test_uv_macos_sandbox_rejects_preexisting_hard_link_aliases(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    custody = tmp_path / "host-data"
    custody.mkdir()
    database = custody / "host-features.db"
    database.write_bytes(b"intact")
    external_alias = tmp_path / "outside-custody-alias"
    os.link(database, external_alias)
    monkeypatch.setenv("KESTREL_HOST_DB_PATH", str(custody / "host-features.db"))
    monkeypatch.setattr(uv_executor_module.sys, "platform", "darwin")
    executor = UvExecutor()
    monkeypatch.setattr(executor, "_get_uv_path", lambda: "/fake/uv")
    monkeypatch.setattr(executor, "_get_base_python_path", lambda: "/fake/python")

    assert database.stat().st_ino == external_alias.stat().st_ino
    with pytest.raises(
        executor_base.ExecutionEnvironmentError,
        match="pre-existing hard-link aliases",
    ):
        executor._get_filesystem_sandbox_prefix()

    assert executor.is_available is False


def test_uv_linux_sandbox_exposes_only_runtime_and_workspace(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    custody = tmp_path / "host-data"
    custody.mkdir()
    database = custody / "host-features.db"
    database.write_bytes(b"intact")
    external_alias = tmp_path / "outside-custody-alias"
    os.link(database, external_alias)
    workspace = tmp_path / "kestrel_compute_fresh"
    workspace.mkdir()
    base_runtime = tmp_path / "trusted-base-runtime"
    base_python = base_runtime / "bin" / "python"
    base_python.parent.mkdir(parents=True)
    base_python.touch()
    uv_path = tmp_path / "trusted-tools" / "uv"
    uv_path.parent.mkdir()
    uv_path.touch()
    monkeypatch.setenv("KESTREL_HOST_DB_PATH", str(custody / "host-features.db"))
    monkeypatch.setattr(uv_executor_module.sys, "platform", "linux")
    monkeypatch.setattr(uv_executor_module.sys, "base_prefix", str(base_runtime))
    monkeypatch.setattr(
        uv_executor_module.shutil,
        "which",
        lambda command: "/usr/bin/bwrap" if command == "bwrap" else None,
    )

    assert database.stat().st_ino == external_alias.stat().st_ino
    prefix = UvExecutor()._get_filesystem_sandbox_prefix(
        str(workspace),
        uv_path=str(uv_path),
        base_python_path=str(base_python),
    )

    assert prefix[:4] == [
        "/usr/bin/bwrap",
        "--die-with-parent",
        "--new-session",
        "--unshare-all",
    ]
    assert ["--tmpfs", "/"] == prefix[
        prefix.index("--tmpfs") : prefix.index("--tmpfs") + 2
    ]
    assert ["--bind", str(workspace.resolve()), str(workspace.resolve())] == (
        prefix[prefix.index("--bind") : prefix.index("--bind") + 3]
    )
    read_only_sources = {
        prefix[index + 1]
        for index, value in enumerate(prefix[:-2])
        if value == "--ro-bind"
    }
    assert "/" not in read_only_sources
    assert str(custody.resolve()) not in read_only_sources
    assert str(base_runtime.resolve()) in read_only_sources
    assert str(uv_path.resolve()) in read_only_sources
    assert prefix[-1] == "--"


def test_uv_sandbox_fails_closed_without_platform_boundary(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    custody = tmp_path / "host-data"
    custody.mkdir()
    monkeypatch.setenv("KESTREL_HOST_DB_PATH", str(custody / "host-features.db"))
    monkeypatch.setattr(uv_executor_module.sys, "platform", "linux")
    monkeypatch.setattr(uv_executor_module.shutil, "which", lambda _command: None)
    executor = UvExecutor()
    monkeypatch.setattr(executor, "_get_uv_path", lambda: "/fake/uv")
    monkeypatch.setattr(executor, "_get_base_python_path", lambda: "/fake/python")

    with pytest.raises(
        executor_base.ExecutionEnvironmentError,
        match="requires bubblewrap",
    ):
        executor._get_filesystem_sandbox_prefix()
    assert executor.is_available is False


def test_uv_sandbox_fails_closed_when_present_but_forbidden(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Availability must exercise sandbox creation, not just find its binary."""

    executor = UvExecutor()
    monkeypatch.setattr(executor, "_get_uv_path", lambda: "/fake/uv")
    monkeypatch.setattr(
        executor,
        "_get_base_python_path",
        lambda: "/fake/base/python",
    )
    monkeypatch.setattr(
        executor,
        "_get_filesystem_sandbox_prefix",
        lambda **_kwargs: ["/fake/sandbox-exec", "--"],
    )
    commands: list[list[str]] = []

    def reject_nested_sandbox(command: list[str], **_kwargs: object):
        commands.append(command)
        return subprocess.CompletedProcess(command, 71)

    monkeypatch.setattr(uv_executor_module.subprocess, "run", reject_nested_sandbox)

    assert executor.is_available is False
    assert commands == [
        [
            "/fake/sandbox-exec",
            "--",
            "/fake/base/python",
            "-I",
            "-S",
            "-c",
            "pass",
        ]
    ]


@pytest.mark.asyncio
async def test_uv_command_is_isolated_project_free_and_only_adds_declared_requirements(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    executor = _make_executor(monkeypatch, "uv", max_bytes=128)
    created = _track_temp_dirs(monkeypatch, tmp_path)
    processes = [_SuccessfulProcess(b"", b""), _SuccessfulProcess(b"ok", b"")]
    commands: list[tuple[object, ...]] = []
    subprocess_options: list[dict[str, object]] = []
    sandbox_workspaces: list[Optional[str]] = []

    def sandbox_prefix(
        writable_workspace: Optional[str] = None,
        **_kwargs: object,
    ) -> list[str]:
        sandbox_workspaces.append(writable_workspace)
        return ["/fake/filesystem-sandbox", "--"]

    async def create_subprocess(*args: object, **kwargs: object):
        commands.append(args)
        subprocess_options.append(kwargs)
        return processes[len(commands) - 1]

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_subprocess)
    monkeypatch.setattr(executor, "_get_filesystem_sandbox_prefix", sandbox_prefix)
    monkeypatch.setenv("UV_PROJECT_ENVIRONMENT", "/host/project/.venv")
    monkeypatch.setenv("UV_INDEX_URL", "https://host-index.invalid/simple")
    script = _script(
        environment={
            "PYTHONPATH": "/caller/ambient/site-packages",
            "UV_OFFLINE": "1",
        }
    )
    script.requirements = ["declared-one==1.0", "declared-two>=2"]

    record = await executor.execute(script)

    script_path = str(created[0] / "script.py")
    assert sandbox_workspaces == [str(created[0])]
    assert commands[0] == (
        "/fake/uv",
        "run",
        "--isolated",
        "--no-project",
        "--no-config",
        "--no-build",
        "--no-python-downloads",
        "--python",
        "/fake/base/python",
        "--with",
        "declared-one==1.0",
        "--with",
        "declared-two>=2",
        "--",
        "/fake/base/python",
        "-I",
        "-S",
        "-c",
        "pass",
    )
    assert commands[1] == (
        "/fake/filesystem-sandbox",
        "--",
        "/fake/uv",
        "run",
        "--isolated",
        "--no-project",
        "--no-config",
        "--no-build",
        "--no-python-downloads",
        "--python",
        "/fake/base/python",
        "--with",
        "declared-one==1.0",
        "--with",
        "declared-two>=2",
        "--offline",
        script_path,
    )
    resolver_env = subprocess_options[0]["env"]
    assert isinstance(resolver_env, dict)
    assert "PYTHONPATH" not in resolver_env
    assert "VIRTUAL_ENV" not in resolver_env
    assert "UV_OFFLINE" not in resolver_env
    assert resolver_env["HOME"] == str(created[0])
    assert resolver_env["UV_CACHE_DIR"] == str(created[0] / ".uv-cache")
    child_env = subprocess_options[1]["env"]
    assert isinstance(child_env, dict)
    assert "PYTHONPATH" not in child_env
    assert "UV_PROJECT_ENVIRONMENT" not in child_env
    assert "UV_INDEX_URL" not in child_env
    assert child_env["UV_OFFLINE"] == "1"
    assert child_env["HOME"] == str(created[0])
    assert child_env["TMPDIR"] == str(created[0] / "tmp")
    assert child_env["UV_CACHE_DIR"] == str(created[0] / ".uv-cache")
    assert child_env["UV_NO_CONFIG"] == "1"
    assert child_env["UV_NO_BUILD"] == "1"
    assert child_env["UV_PYTHON_DOWNLOADS"] == "never"
    assert record.stdout == "ok"
    assert record.exit_code == 0


@pytest.mark.asyncio
async def test_uv_rejects_ambient_requirement_sources_before_launch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    executor = _make_executor(monkeypatch, "uv", max_bytes=128)
    _track_temp_dirs(monkeypatch, tmp_path)
    launched = False

    async def reject_launch(*_args: object, **_kwargs: object):
        nonlocal launched
        launched = True
        raise AssertionError("untrusted requirement reached uv")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", reject_launch)
    script = _script()
    script.requirements = ["example @ file:///host/custody/package.whl"]

    with pytest.raises(
        executor_base.ExecutionEnvironmentError,
        match="direct URL, VCS, and file requirements are not permitted",
    ):
        await executor.execute(script)

    assert launched is False


@pytest.mark.asyncio
async def test_uv_strips_dynamic_loader_environment_before_sandbox_launch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Caller code cannot run in the loader before bwrap/Seatbelt starts."""

    executor = _make_executor(monkeypatch, "uv", max_bytes=128)
    _track_temp_dirs(monkeypatch, tmp_path)
    process = _SuccessfulProcess(b"ok", b"")
    subprocess_options: dict[str, object] = {}

    async def create_subprocess(*_args: object, **kwargs: object):
        subprocess_options.update(kwargs)
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_subprocess)
    script = _script(
        environment={
            "LD_PRELOAD": "/caller/preload.so",
            "LD_AUDIT": "/caller/audit.so",
            "LD_LIBRARY_PATH": "/caller/lib",
            "DYLD_INSERT_LIBRARIES": "/caller/preload.dylib",
            "DYLD_LIBRARY_PATH": "/caller/dylibs",
            "LIBPATH": "/caller/aix",
            "SHLIB_PATH": "/caller/hpux",
            "COMPUTE_SAFE_VALUE": "preserved",
        }
    )

    record = await executor.execute(script)

    child_env = subprocess_options["env"]
    assert isinstance(child_env, dict)
    assert child_env["COMPUTE_SAFE_VALUE"] == "preserved"
    assert not {
        "LD_PRELOAD",
        "LD_AUDIT",
        "LD_LIBRARY_PATH",
        "DYLD_INSERT_LIBRARIES",
        "DYLD_LIBRARY_PATH",
        "LIBPATH",
        "SHLIB_PATH",
    }.intersection(child_env)
    assert record.exit_code == 0


@pytest.mark.asyncio
async def test_uv_real_process_cannot_mutate_host_hold_sqlite(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    custody = tmp_path / "host-data"
    custody.mkdir(mode=0o700)
    database = custody / "host-features.db"
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE hold_latches(value TEXT)")
        connection.execute("INSERT INTO hold_latches VALUES ('intact')")
    database.chmod(0o600)
    monkeypatch.setenv("KESTREL_HOST_DB_PATH", str(database))
    executor = UvExecutor(max_output_bytes=64 * 1024)
    if not executor.is_available:
        pytest.skip("uv executor requires a verified OS filesystem sandbox")

    script = ComputeScript(
        id="uv-host-hold-custody",
        name="attempt direct SQLite Hold mutation",
        language="python",
        content=(
            "import sqlite3\n"
            f"with sqlite3.connect({str(database)!r}) as connection:\n"
            "    connection.execute('DELETE FROM hold_latches')\n"
        ),
        purpose="prove C-extension writes cannot bypass host Hold custody",
    )

    record = await executor.execute(script)

    with sqlite3.connect(database) as connection:
        rows = connection.execute("SELECT value FROM hold_latches").fetchall()
    assert record.exit_code != 0
    assert rows == [("intact",)]


@pytest.mark.asyncio
async def test_docker_rejects_writable_alias_of_host_hold_custody_before_launch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    custody = tmp_path / "host-data"
    custody.mkdir()
    custody_alias = tmp_path / "custody-alias"
    custody_alias.symlink_to(custody, target_is_directory=True)
    monkeypatch.setenv("KESTREL_HOST_DB_PATH", str(custody / "host-features.db"))
    executor = DockerExecutor()
    monkeypatch.setattr(executor, "_get_docker_path", lambda: "/fake/docker")
    launched = False

    async def reject_launch(*_args, **_kwargs):
        nonlocal launched
        launched = True
        raise AssertionError("unsafe Docker command reached process launch")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", reject_launch)

    with pytest.raises(
        executor_base.ExecutionEnvironmentError,
        match="additional Docker mounts.*host service sockets",
    ):
        await executor.execute(
            _script(),
            mounts=[
                {"src": str(custody_alias), "dst": "/data", "ro": False}
            ],
        )

    assert launched is False


@pytest.mark.asyncio
async def test_docker_rejects_writable_mount_with_hard_link_to_hold_before_launch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    custody = tmp_path / "host-data"
    custody.mkdir()
    database = custody / "host-features.db"
    database.write_bytes(b"intact")
    mount_source = tmp_path / "additional-mount"
    mount_source.mkdir()
    alias = mount_source / "unrelated-name"
    os.link(database, alias)
    monkeypatch.setenv("KESTREL_HOST_DB_PATH", str(database))
    executor = DockerExecutor()
    monkeypatch.setattr(executor, "_get_docker_path", lambda: "/fake/docker")
    launched = False

    async def reject_launch(*_args, **_kwargs):
        nonlocal launched
        launched = True
        raise AssertionError("unsafe Docker command reached process launch")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", reject_launch)

    with pytest.raises(
        executor_base.ExecutionEnvironmentError,
        match="additional Docker mounts.*host service sockets",
    ):
        await executor.execute(
            _script(),
            mounts=[
                {"src": str(mount_source), "dst": "/data", "ro": False}
            ],
        )

    assert database.stat().st_ino == alias.stat().st_ino
    assert launched is False


def test_docker_rejects_read_only_unix_service_socket_mount() -> None:
    with tempfile.TemporaryDirectory(prefix="ks-", dir="/tmp") as socket_root:
        socket_path = Path(socket_root) / "docker.sock"
        service = socket.socket(socket.AF_UNIX)
        service.bind(str(socket_path))
        executor = DockerExecutor()

        try:
            with pytest.raises(
                executor_base.ExecutionEnvironmentError,
                match="read-only bind mounts still expose host service sockets",
            ):
                executor._validate_additional_mounts(
                    [
                        {
                            "src": str(socket_path),
                            "dst": "/run/docker.sock",
                            "ro": True,
                        }
                    ]
                )
        finally:
            service.close()


def test_docker_rejects_read_only_directory_that_can_gain_service_socket(
    tmp_path: Path,
) -> None:
    mount_source = tmp_path / "runtime"
    mount_source.mkdir()

    with pytest.raises(
        executor_base.ExecutionEnvironmentError,
        match="cannot prove separation from host Hold custody",
    ):
        DockerExecutor()._validate_additional_mounts(
            [{"src": str(mount_source), "dst": "/run/host", "ro": True}]
        )


@pytest.mark.asyncio
async def test_docker_rejects_internal_trash_staging_inside_hold_custody(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    custody = tmp_path / "host-data"
    custody.mkdir()
    monkeypatch.setenv("KESTREL_HOST_DB_PATH", str(custody / "host-features.db"))
    monkeypatch.setenv("KESTREL_TRASH_DIR", str(custody))
    executor = _make_executor(monkeypatch, "docker")
    # DEFAULT_TRASH_DIR is intentionally captured at module import. Model a
    # process that started with this setting without reloading shared modules.
    executor._policy.trash_dir = custody
    launched = False

    async def reject_launch(*_args: object, **_kwargs: object):
        nonlocal launched
        launched = True
        raise AssertionError("unsafe Docker command reached process launch")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", reject_launch)

    with pytest.raises(
        executor_base.ExecutionEnvironmentError,
        match="trash staging overlaps host Hold custody",
    ):
        await executor.execute(_script())

    assert not any(path.name.startswith(".staging-") for path in custody.iterdir())
    assert launched is False


@pytest.mark.asyncio
async def test_uv_refuses_ambient_host_working_directory(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    executor = _make_executor(monkeypatch, "uv")
    working_dir = tmp_path / "host-project"
    working_dir.mkdir()
    (working_dir / ".env").write_text("HOLD_DSN=secret\n")

    with pytest.raises(
        executor_base.ExecutionEnvironmentError,
        match="does not expose host working directories",
    ):
        await executor.execute(_script(), working_dir=str(working_dir))


@pytest.mark.asyncio
async def test_uv_real_process_isolated_from_ambient_host_workspace(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    custody = tmp_path / "host-data"
    custody.mkdir()
    monkeypatch.setenv("KESTREL_HOST_DB_PATH", str(custody / "host-features.db"))
    executor = UvExecutor(max_output_bytes=64 * 1024)
    if not executor.is_available:
        pytest.skip("uv executor requires uv and a virtualized Kestrel runtime")

    workspace = tmp_path / "synthetic host workspace with spaces"
    member = workspace / "host member project"
    nested_working_dir = member / "nested working directory"
    wheel_dir = workspace / "local wheel files"
    nested_working_dir.mkdir(parents=True)
    ambient_wheel = _write_test_wheel(
        wheel_dir,
        "ambient-dependency",
        "ambient_dependency",
        "host-project-only",
    )
    explicit_wheel = _write_test_wheel(
        wheel_dir,
        "explicit-dependency",
        "explicit_dependency",
        "declared-requirement",
    )

    (workspace / "pyproject.toml").write_text(
        '[tool.uv.workspace]\nmembers = ["host member project"]\n'
    )
    member_pyproject = (
        "[project]\n"
        'name = "synthetic-host-member"\n'
        'version = "1.0.0"\n'
        f'dependencies = ["ambient-dependency @ {ambient_wheel.resolve().as_uri()}"]\n'
        "\n[tool.uv]\n"
        "package = false\n"
    )
    (member / "pyproject.toml").write_text(member_pyproject)

    host_venv = workspace / ".venv"
    venv.EnvBuilder(with_pip=False, symlinks=os.name != "nt").create(host_venv)
    host_python = host_venv / (
        "Scripts/python.exe" if os.name == "nt" else "bin/python"
    )
    site_result = subprocess.run(
        [
            str(host_python),
            "-c",
            "import site; print(site.getsitepackages()[0])",
        ],
        capture_output=True,
        text=True,
    )
    assert site_result.returncode == 0, site_result.stderr
    site_packages = Path(site_result.stdout.strip())
    (site_packages / "ambient_dependency.py").write_text(
        "VALUE = 'host-project-only'\n"
    )
    (site_packages / "base_only_dependency.py").write_text(
        "VALUE = 'pinned-interpreter-only'\n"
    )
    assert subprocess.run(
        [
            str(host_python),
            "-c",
            "import ambient_dependency; print(ambient_dependency.VALUE)",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip() == "host-project-only"
    assert subprocess.run(
        [
            str(host_python),
            "-c",
            "import base_only_dependency; print(base_only_dependency.VALUE)",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip() == "pinned-interpreter-only"

    host_venv_before = _tree_manifest(host_venv)
    workspace_pyproject_before = (workspace / "pyproject.toml").read_bytes()
    member_pyproject_before = (member / "pyproject.toml").read_bytes()
    execution_dir = workspace / "executor temporary directory with spaces"
    sandbox_wheel = execution_dir / explicit_wheel.name

    def make_temp_dir(*, prefix: str) -> str:
        assert prefix == "kestrel_compute_"
        execution_dir.mkdir()
        shutil.copy2(explicit_wheel, sandbox_wheel)
        return str(execution_dir)

    monkeypatch.setattr(executor_base.tempfile, "mkdtemp", make_temp_dir)
    monkeypatch.setenv("VIRTUAL_ENV", str(host_venv))
    monkeypatch.setenv("UV_PROJECT_ENVIRONMENT", str(host_venv))
    monkeypatch.setenv("UV_INDEX_URL", "https://host-index.invalid/simple")
    monkeypatch.setenv("PYTHONPATH", str(site_packages))
    monkeypatch.setattr(
        executor,
        "_get_base_python_path",
        lambda: str(host_python),
    )

    script = ComputeScript(
        id="uv-real-isolation-test",
        name="uv real isolation",
        language="python",
        content=(
            "import json, sys\n"
            "try:\n"
            "    import ambient_dependency\n"
            "except ImportError:\n"
            "    ambient_visible = False\n"
            "else:\n"
            "    ambient_visible = ambient_dependency.VALUE\n"
            "try:\n"
            "    import base_only_dependency\n"
            "except ImportError:\n"
            "    base_visible = False\n"
            "else:\n"
            "    base_visible = base_only_dependency.VALUE\n"
            "import explicit_dependency\n"
            "print(json.dumps({\n"
            "    'ambient_visible': ambient_visible,\n"
            "    'base_visible': base_visible,\n"
            "    'explicit_value': explicit_dependency.VALUE,\n"
            "    'prefix': sys.prefix,\n"
            "}))\n"
        ),
        purpose="prove the uv project and interpreter boundary",
        requirements=[str(sandbox_wheel)],
        environment={"UV_OFFLINE": "1"},
    )

    record = await executor.execute(script)

    assert _tree_manifest(host_venv) == host_venv_before
    assert (workspace / "pyproject.toml").read_bytes() == workspace_pyproject_before
    assert (member / "pyproject.toml").read_bytes() == member_pyproject_before
    assert not (workspace / "uv.lock").exists()
    assert not execution_dir.exists()
    if (
        sys.platform == "darwin"
        and record.exit_code == 101
        and "system-configuration" in record.stderr
        and "Attempted to create a NULL object" in record.stderr
    ):
        pytest.skip("uv cannot access the macOS Dynamic Store in this sandbox")
    assert record.exit_code == 0, record.stderr
    output = json.loads(record.stdout.strip())
    assert output["ambient_visible"] is False
    assert output["base_visible"] is False
    assert output["explicit_value"] == "declared-requirement"
    assert Path(output["prefix"]).resolve().is_relative_to(
        (execution_dir / ".uv-cache").resolve()
    )

    no_requirements_script = ComputeScript(
        id="uv-real-isolation-no-requirements-test",
        name="uv real isolation without requirements",
        language="python",
        content=(
            "import json, sys\n"
            "visible = {}\n"
            "for module in ('ambient_dependency', 'base_only_dependency'):\n"
            "    try:\n"
            "        __import__(module)\n"
            "    except ImportError:\n"
            "        visible[module] = False\n"
            "    else:\n"
            "        visible[module] = True\n"
            "print(json.dumps({'visible': visible, 'prefix': sys.prefix}))\n"
        ),
        purpose="prove a dependency-free run still receives a fresh environment",
        requirements=[],
        environment={"UV_OFFLINE": "1"},
    )

    no_requirements_record = await executor.execute(no_requirements_script)

    assert _tree_manifest(host_venv) == host_venv_before
    assert (workspace / "pyproject.toml").read_bytes() == workspace_pyproject_before
    assert (member / "pyproject.toml").read_bytes() == member_pyproject_before
    assert not (workspace / "uv.lock").exists()
    assert not execution_dir.exists()
    if (
        sys.platform == "darwin"
        and no_requirements_record.exit_code == 101
        and "system-configuration" in no_requirements_record.stderr
        and "Attempted to create a NULL object" in no_requirements_record.stderr
    ):
        pytest.skip("uv cannot access the macOS Dynamic Store in this sandbox")
    assert no_requirements_record.exit_code == 0, no_requirements_record.stderr
    no_requirements_output = json.loads(no_requirements_record.stdout.strip())
    assert no_requirements_output["visible"] == {
        "ambient_dependency": False,
        "base_only_dependency": False,
    }
    assert Path(no_requirements_output["prefix"]).resolve().is_relative_to(
        (execution_dir / ".uv-cache").resolve()
    )


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


# =============================================================================
# Argv execution mode (#3187)
# =============================================================================


def _command(
    *,
    argv: Optional[list[str]] = None,
    timeout_seconds: int = 1,
    environment: Optional[dict[str, str]] = None,
) -> ComputeCommand:
    return ComputeCommand(
        id="executor-command-test",
        name="executor command",
        argv=argv or ["printf", "ok"],
        purpose="exercise the argv execution mode",
        timeout_seconds=timeout_seconds,
        environment=environment or {},
    )


async def _capture_container_argv(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    run,
) -> tuple[list[str], list[list[Path]]]:
    """Run one executor call, returning the docker argv and the temp-dir contents.

    The directory listing is taken at the moment the subprocess starts,
    which is the only moment a written script would still exist: the
    lifecycle deletes the temp directory on the way out.
    """
    executor = _make_executor(monkeypatch, "docker", max_bytes=128)
    created = _track_temp_dirs(monkeypatch, tmp_path)
    process = _SuccessfulProcess(b"ok", b"")
    calls: list[list[str]] = []
    contents: list[list[Path]] = []

    async def snapshot_for_argv_test(source: str, destination: Path, **_kwargs) -> str:
        return DockerExecutor._snapshot_working_directory(source, destination)

    # This helper inspects Docker's container argv. The subprocess worker has
    # its own lifecycle regression below, so perform the tiny fixture snapshot
    # directly and leave the one captured subprocess slot for the container.
    monkeypatch.setattr(
        executor,
        "_bounded_working_directory_snapshot",
        snapshot_for_argv_test,
    )

    async def create_subprocess(*args: object, **kwargs: object):
        calls.append([str(arg) for arg in args])
        contents.append(
            [entry for directory in created for entry in directory.iterdir()]
        )
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_subprocess)
    await run(executor)
    # Later calls are the lifecycle's own container removal, not the run.
    assert calls, "the executor never started a container"
    return calls[0], contents[:1]


@pytest.mark.asyncio
async def test_docker_command_mode_execs_the_vector_and_writes_no_script(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """#3187: a backend named ``exec(argv)`` must exec argv.

    The vector here is chosen so that any re-reading would show: under
    the previous implementation these words were quoted into a bash
    script, and ``eval`` ran ``printf`` with only ``eval`` vetted.

    ``argv[0]`` is named to Docker with ``--entrypoint`` rather than
    positioned after the image. Position is not enough: words after the
    image are appended to whatever ``ENTRYPOINT`` the image declares,
    so on an entrypoint-bearing image the first word is demoted to an
    argument and the image's own program runs (codex round 1 P1,
    reproduced live — see the parity test for the measurement).
    """
    argv = ["eval", "printf HACKED", ";", "--privileged", "$(id)"]
    captured, contents = await _capture_container_argv(
        monkeypatch,
        tmp_path,
        run=lambda executor: executor.execute_command(_command(argv=argv)),
    )

    entrypoint_at = captured.index("--entrypoint")
    image_at = captured.index(docker_executor_module.DEFAULT_COMMAND_IMAGE)
    assert captured[entrypoint_at + 1] == argv[0]
    assert entrypoint_at < image_at, "--entrypoint is an option, not an argument"
    assert captured[image_at + 1 :] == argv[1:]

    # Nothing interprets it: no interpreter is named, and no script
    # file exists to be interpreted.
    flags = captured[:image_at]
    assert "sh" not in flags and "bash" not in flags and "python" not in flags
    assert contents == [[]], f"the executor wrote a file: {contents}"

    # No mounts that only a script would need — and in particular no
    # writable host bind, because the deletion rewriter that the trash
    # mount exists for only rewrites script text.
    binds = [captured[i + 1] for i, arg in enumerate(captured) if arg == "-v"]
    assert binds == [], binds


@pytest.mark.asyncio
async def test_docker_command_mode_runs_under_the_same_isolation_as_a_script(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The two modes must not drift apart on container hardening.

    Derived from the script path rather than restated, so a flag added
    to one mode and not the other fails here instead of being noticed
    later — the drift the sandbox backend reuses this executor to
    avoid.
    """
    script_cmd, _ = await _capture_container_argv(
        monkeypatch,
        tmp_path,
        run=lambda executor: executor.execute(
            ComputeScript(
                id="isolation-parity",
                name="isolation parity",
                language="bash",
                content="printf ok\n",
                purpose="compare container hardening across execution modes",
                timeout_seconds=1,
            )
        ),
    )
    command_cmd, _ = await _capture_container_argv(
        monkeypatch,
        tmp_path,
        run=lambda executor: executor.execute_command(_command()),
    )

    def isolation_flags(cmd: list[str]) -> set[str]:
        return {arg for arg in cmd if arg.startswith("--") and arg != "--name"}

    assert isolation_flags(command_cmd) == isolation_flags(script_cmd)
    assert "--read-only" in isolation_flags(command_cmd)
    assert "--network=none" in isolation_flags(command_cmd)


@pytest.mark.asyncio
async def test_docker_script_mode_mounts_the_script_and_the_trash_staging(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The shared builder has to attach the binds it is handed.

    Found by mutating it: with the bind loop removed, every other test
    here still passed, while a real container would have started with
    no script to run and nowhere to put a trashed file. The command
    mode above asserts these binds are *absent*, which is only half an
    invariant — this is the other half, and it belongs to the caller
    that needs them.
    """
    script_cmd, _ = await _capture_container_argv(
        monkeypatch,
        tmp_path,
        run=lambda executor: executor.execute(
            ComputeScript(
                id="bind-mounts",
                name="bind mounts",
                language="bash",
                content="printf ok\n",
                purpose="pin the mounts a script execution needs",
                timeout_seconds=1,
            )
        ),
    )

    binds = [
        script_cmd[i + 1] for i, arg in enumerate(script_cmd) if arg == "-v"
    ]
    assert any(bind.endswith(":/scripts:ro") for bind in binds), binds
    assert any(bind.endswith(":/kestrel-trash:rw") for bind in binds), binds


@pytest.mark.asyncio
async def test_docker_script_mode_mounts_a_private_working_directory_snapshot(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The post-staging launch must not reintroduce the live host directory."""

    workspace = tmp_path / "caller cwd"
    workspace.mkdir()
    (workspace / "input.txt").write_text("stable input", encoding="utf-8")

    script_cmd, _ = await _capture_container_argv(
        monkeypatch,
        tmp_path,
        run=lambda executor: executor.execute(
            _script(),
            working_dir=str(workspace),
        ),
    )

    workspace_binds = [
        script_cmd[index + 1]
        for index, argument in enumerate(script_cmd[:-1])
        if argument == "-v" and script_cmd[index + 1].endswith(":/workspace:ro")
    ]
    assert len(workspace_binds) == 1
    snapshot_source = Path(workspace_binds[0].removesuffix(":/workspace:ro"))
    assert snapshot_source != workspace
    assert snapshot_source.name == "workspace"
    assert script_cmd[script_cmd.index("-w") + 1] == "/workspace"


@pytest.mark.asyncio
async def test_docker_rejects_working_directory_that_contains_snapshot(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A broad source such as /tmp must not recursively copy its own output."""

    executor = _make_executor(monkeypatch, "docker")
    _track_temp_dirs(monkeypatch, tmp_path)
    copied = False

    def reject_copy(*_args: object, **_kwargs: object) -> None:
        nonlocal copied
        copied = True
        raise AssertionError("recursive snapshot reached copytree")

    monkeypatch.setattr(docker_executor_module.shutil, "copytree", reject_copy)

    # 60s, not the _script() default of 1s: this test's subject is the
    # snapshot REJECTION, and reaching it walks a real directory. My
    # TimeoutError fix (it subclasses OSError, so the snapshot deadline
    # used to be reported as tampering) means the deadline now surfaces
    # honestly -- which exposed that 1s is not enough on a 2-core runner
    # to reach the check at all.
    record = await executor.execute(
        _script(timeout_seconds=60), working_dir=str(tmp_path)
    )

    assert record.exit_code == -1
    assert "would contain its own destination" in record.stderr
    assert copied is False


@pytest.mark.skipif(os.name != "posix", reason="FIFO creation is POSIX-only")
@pytest.mark.asyncio
async def test_docker_rejects_special_source_entry_before_copying(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Source inspection must reject devices/FIFOs before opening their data."""

    executor = _make_executor(monkeypatch, "docker")
    source = tmp_path / "caller cwd"
    source.mkdir()
    os.mkfifo(source / "unbounded-device")
    execution_root = tmp_path / "execution roots"
    execution_root.mkdir()
    _track_temp_dirs(monkeypatch, execution_root)
    copied = False

    def reject_copy(*_args: object, **_kwargs: object) -> None:
        nonlocal copied
        copied = True
        raise AssertionError("special source entry reached copytree")

    monkeypatch.setattr(docker_executor_module.shutil, "copytree", reject_copy)

    # 60s, not the _script() default of 1s: this test's subject is the
    # snapshot REJECTION, and reaching it walks a real directory. My
    # TimeoutError fix (it subclasses OSError, so the snapshot deadline
    # used to be reported as tampering) means the deadline now surfaces
    # honestly -- which exposed that 1s is not enough on a 2-core runner
    # to reach the check at all.
    record = await executor.execute(
        _script(timeout_seconds=60), working_dir=str(source)
    )

    assert record.exit_code == -1
    assert "host service socket or other special file" in record.stderr
    assert copied is False


@pytest.mark.skipif(os.name != "posix", reason="FIFO creation is POSIX-only")
def test_docker_rechecks_regular_source_at_copy_boundary(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A file raced after preflight must not be opened as snapshot content."""

    source = tmp_path / "caller cwd"
    source.mkdir()
    raced = source / "input.txt"
    raced.write_text("regular during preflight", encoding="utf-8")
    destination = tmp_path / "isolated" / "workspace"
    original_open = docker_executor_module.os.open
    replaced = False

    def replace_before_open(
        path,
        flags,
        mode=0o777,
        *,
        dir_fd=None,
    ):
        nonlocal replaced
        if path == raced.name and dir_fd is not None and not replaced:
            replaced = True
            raced.unlink()
            os.mkfifo(raced)
        return original_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(
        docker_executor_module.os,
        "open",
        replace_before_open,
    )

    with pytest.raises(
        executor_base.ExecutionEnvironmentError,
        match="host service socket or other special file",
    ):
        DockerExecutor._snapshot_working_directory(str(source), destination)

    assert not destination.exists()


@pytest.mark.skipif(os.name != "posix", reason="dirfd traversal is POSIX-only")
def test_docker_snapshot_anchors_directory_replaced_by_symlink(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A raced child path cannot redirect descent outside the accepted root."""

    source = tmp_path / "caller cwd"
    child = source / "child"
    child.mkdir(parents=True)
    (child / "public.txt").write_text("public", encoding="utf-8")
    secret = tmp_path / "ambient secret"
    secret.mkdir()
    (secret / "secret.txt").write_text("SECRET", encoding="utf-8")
    destination = tmp_path / "isolated" / "workspace"
    original_open = docker_executor_module.os.open
    replaced = False

    def replace_child_after_open(
        path,
        flags,
        mode=0o777,
        *,
        dir_fd=None,
    ):
        nonlocal replaced
        if (
            path == child.name
            and dir_fd is not None
            and flags & os.O_DIRECTORY
            and not replaced
        ):
            descriptor = original_open(path, flags, mode, dir_fd=dir_fd)
            replaced = True
            child.rename(source / "child-before-race")
            child.symlink_to(secret, target_is_directory=True)
            return descriptor
        return original_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(
        docker_executor_module.os,
        "open",
        replace_child_after_open,
    )

    DockerExecutor._snapshot_working_directory(str(source), destination)

    assert replaced is True
    assert (destination / "child" / "public.txt").read_text(
        encoding="utf-8"
    ) == "public"
    assert not (destination / "child" / "secret.txt").exists()


def test_docker_snapshot_enforces_byte_budget(tmp_path: Path) -> None:
    """Regular files cannot make pre-container host storage grow without bound."""

    source = tmp_path / "caller cwd"
    source.mkdir()
    (source / "input.txt").write_bytes(b"12345")
    destination = tmp_path / "isolated" / "workspace"

    with pytest.raises(
        executor_base.ExecutionEnvironmentError,
        match="4-byte limit",
    ):
        DockerExecutor._snapshot_working_directory(
            str(source),
            destination,
            max_bytes=4,
        )

    assert not destination.exists()


def test_docker_snapshot_enforces_entry_budget(tmp_path: Path) -> None:
    """Tiny files cannot evade the workspace resource boundary."""

    source = tmp_path / "caller cwd"
    source.mkdir()
    (source / "one").touch()
    (source / "two").touch()
    destination = tmp_path / "isolated" / "workspace"

    with pytest.raises(
        executor_base.ExecutionEnvironmentError,
        match="1-entry limit",
    ):
        DockerExecutor._snapshot_working_directory(
            str(source),
            destination,
            max_entries=1,
        )

    assert not destination.exists()


@pytest.mark.asyncio
async def test_docker_snapshot_worker_copies_into_private_execution_root(
    tmp_path: Path,
) -> None:
    """The real isolated worker publishes success and leaves no protocol file."""

    source = tmp_path / "caller cwd"
    source.mkdir()
    (source / "input.txt").write_text("stable input", encoding="utf-8")
    execution_root = tmp_path / "execution"
    execution_root.mkdir()
    destination = execution_root / "workspace"
    executor = DockerExecutor()
    loop = asyncio.get_running_loop()

    result = await executor._bounded_working_directory_snapshot(
        str(source),
        destination,
        deadline=loop.time() + 5,
        subject_id="snapshot-success",
        timeout_seconds=5,
    )

    assert result == str(destination)
    assert (destination / "input.txt").read_text(encoding="utf-8") == "stable input"
    assert not (execution_root / ".docker-snapshot-result.json").exists()


@pytest.mark.asyncio
async def test_docker_snapshot_is_off_loop_and_deadline_bounded(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A stuck filesystem worker is killed without using the shared executor."""

    executor = DockerExecutor()
    process = _BlockingProcess()
    commands: list[tuple[object, ...]] = []
    options: list[dict[str, object]] = []

    async def create_subprocess(*args: object, **kwargs: object):
        commands.append(args)
        options.append(kwargs)
        return process

    async def reject_shared_thread(*_args: object, **_kwargs: object):
        raise AssertionError("snapshot entered asyncio's shared thread executor")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_subprocess)
    monkeypatch.setattr(asyncio, "to_thread", reject_shared_thread)
    loop = asyncio.get_running_loop()
    started = loop.time()
    with pytest.raises(ExecutionTimeoutError):
        await executor._bounded_working_directory_snapshot(
            str(tmp_path),
            tmp_path / "workspace",
            deadline=loop.time() + 0.03,
            subject_id="snapshot-timeout",
            timeout_seconds=0.03,
        )

    assert loop.time() - started < 0.2
    assert commands[0][:4] == (
        sys.executable,
        "-I",
        "-m",
        docker_executor_module._SNAPSHOT_WORKER_MODULE,
    )
    assert options[0]["start_new_session"] is (os.name == "posix")
    assert process.kill_calls >= 1
    assert process.wait_calls >= 1


@pytest.mark.asyncio
async def test_docker_command_mode_stands_in_the_directory_it_mounted(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A relative operand must resolve against the caller's directory.

    With a host directory mounted, the container's working directory is
    that mount; without one there is no workspace to stand in, so the
    read-only image root is used rather than whatever the last mount
    happened to leave behind.
    """
    workspace = tmp_path / "caller cwd"
    workspace.mkdir()

    with_cwd, _ = await _capture_container_argv(
        monkeypatch,
        tmp_path,
        run=lambda executor: executor.execute_command(
            _command(argv=["ls", "."]), working_dir=str(workspace)
        ),
    )
    without_cwd, _ = await _capture_container_argv(
        monkeypatch,
        tmp_path,
        run=lambda executor: executor.execute_command(_command(argv=["ls", "."])),
    )

    workspace_binds = [
        with_cwd[index + 1]
        for index, argument in enumerate(with_cwd[:-1])
        if argument == "-v" and with_cwd[index + 1].endswith(":/workspace:ro")
    ]
    assert len(workspace_binds) == 1
    snapshot_source = Path(workspace_binds[0].removesuffix(":/workspace:ro"))
    assert snapshot_source != workspace
    assert snapshot_source.name == "workspace"
    assert with_cwd[with_cwd.index("-w") + 1] == "/workspace"
    assert without_cwd[without_cwd.index("-w") + 1] == "/"
    assert "-v" not in without_cwd


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ("script", "command"))
async def test_docker_working_directory_socket_never_reaches_container(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    mode: str,
) -> None:
    executor = _make_executor(monkeypatch, "docker")
    launched = False

    async def snapshot_for_socket_test(
        source: str,
        destination: Path,
        **_kwargs,
    ) -> str:
        return DockerExecutor._snapshot_working_directory(source, destination)

    # Keep this assertion focused on the descriptor traversal refusing the
    # socket before Docker launch; worker teardown is covered independently.
    monkeypatch.setattr(
        executor,
        "_bounded_working_directory_snapshot",
        snapshot_for_socket_test,
    )

    async def reject_launch(*_args: object, **_kwargs: object):
        nonlocal launched
        launched = True
        raise AssertionError("unsafe Docker command reached process launch")

    async def cleanup_without_process(*_args: object, **_kwargs: object) -> None:
        return None

    with tempfile.TemporaryDirectory(prefix="ks-", dir="/tmp") as socket_root:
        _track_temp_dirs(monkeypatch, tmp_path)
        monkeypatch.setattr(asyncio, "create_subprocess_exec", reject_launch)
        monkeypatch.setattr(executor, "_remove_container", cleanup_without_process)
        service = socket.socket(socket.AF_UNIX)
        service.bind(str(Path(socket_root) / "docker.sock"))
        try:
            if mode == "script":
                record = await executor.execute(
                    _script(),
                    working_dir=socket_root,
                )
            else:
                record = await executor.execute_command(
                    _command(argv=["true"]),
                    working_dir=socket_root,
                )
        finally:
            service.close()

    assert record.exit_code == -1
    assert "host service socket" in record.stderr
    assert launched is False


@pytest.mark.asyncio
async def test_docker_command_mode_redacts_environment_values_from_the_log(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A secret passed as an env var must not reach the debug log."""
    caplog.set_level(logging.DEBUG, logger=docker_executor_module.logger.name)
    captured, _ = await _capture_container_argv(
        monkeypatch,
        tmp_path,
        run=lambda executor: executor.execute_command(
            _command(environment={"TOKEN": "s3cret-value"})
        ),
    )

    assert "-e" in captured and "TOKEN=s3cret-value" in captured
    logged = "\n".join(record.getMessage() for record in caplog.records)
    assert "TOKEN=<redacted>" in logged
    assert "s3cret-value" not in logged


@pytest.mark.asyncio
async def test_docker_command_mode_logs_how_many_arguments_not_which(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Arguments are where a caller's secrets are (codex round 2 P1).

    The environment values on this same log line are redacted for that
    reason, and the script path never logged arguments at all — they
    were inside a file, and the line ended at ``sh /scripts/script.sh``.
    The program still appears: it is what the policy vetted, and it is
    the one thing a reader of this line needs.
    """
    caplog.set_level(logging.DEBUG, logger=docker_executor_module.logger.name)
    argv = ["curl", "-H", "Authorization: Bearer s3cret-token", "https://x"]
    captured, _ = await _capture_container_argv(
        monkeypatch,
        tmp_path,
        run=lambda executor: executor.execute_command(_command(argv=argv)),
    )

    # The secret still reaches the container — this is about the log.
    assert "Authorization: Bearer s3cret-token" in captured

    logged = "\n".join(record.getMessage() for record in caplog.records)
    assert "s3cret-token" not in logged
    assert "3 argument(s) not logged" in logged
    assert "curl" in logged, "the program is what the reader needs"


@pytest.mark.asyncio
async def test_docker_command_mode_cannot_have_a_log_line_forged_through_it(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A newline in the program name must not become a new log line.

    ``shlex.join`` quotes a newline; it does not encode it, so the raw
    byte would still land in the log and everything after it would read
    as a separate record.
    """
    caplog.set_level(logging.DEBUG, logger=docker_executor_module.logger.name)
    forged = "prog\nERROR forged-by-the-caller"
    await _capture_container_argv(
        monkeypatch,
        tmp_path,
        run=lambda executor: executor.execute_command(_command(argv=[forged, "x"])),
    )

    container_lines = [
        record.getMessage()
        for record in caplog.records
        if "Container command" in record.getMessage()
    ]
    assert container_lines, "the container command was never logged"
    for line in container_lines:
        assert "\nERROR forged-by-the-caller" not in line
        assert "forged-by-the-caller" in line, "escaped, not dropped"


@pytest.mark.asyncio
@pytest.mark.parametrize("executor_name", ("local", "uv"))
async def test_a_script_executor_refuses_an_argv_vector(
    monkeypatch: pytest.MonkeyPatch,
    executor_name: str,
) -> None:
    """The default is a refusal, not a fallback.

    An executor that can only run scripts could always quote the vector
    into one — which is precisely the substitution #3187 was filed for.
    Refusing keeps "this executor execs argv" a claim a subclass has to
    make, rather than one every caller may assume.
    """
    executor = _make_executor(monkeypatch, executor_name)

    with pytest.raises(executor_base.CommandExecutionUnsupported) as exc_info:
        await executor.execute_command(_command())

    assert executor.name in str(exc_info.value)
    assert "shell grammar" in str(exc_info.value)


@pytest.mark.asyncio
async def test_the_base_executor_refuses_an_argv_vector_by_default() -> None:
    """Including for an executor written outside this package."""
    with pytest.raises(executor_base.CommandExecutionUnsupported):
        await _CompatibilityExecutor().execute_command(_command())
