"""Adversarial process-group and cancellation contracts for Local MPS."""

import asyncio
import json
import os
import shutil
import signal
import sys
import threading
from pathlib import Path

import pytest

from kestrel_sovereign.features.training.adapters import local_mps_adapter
from kestrel_sovereign.features.training.adapters import (
    _local_mps_generation_lifecycle as generation_lifecycle,
)
from kestrel_sovereign.features.training.adapters import (
    _local_mps_generation_process as generation_process,
)
from kestrel_sovereign.features.training.adapters.local_mps_adapter import (
    LocalMPSTrainingAdapter,
)
from kestrel_sovereign.features.training.types import GenerationConfig, GenerationState


_PROCESS_TREE_SCRIPT = r"""
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

payload = json.loads(sys.argv[1])
Path(payload["parent_pid"]).write_text(str(os.getpid()), encoding="utf-8")
descendant_source = r'''
import json
import os
import signal
import sys
import time
from pathlib import Path

payload = json.loads(sys.argv[1])
signal.signal(signal.SIGTERM, signal.SIG_IGN)
Path(payload["child_pid"]).write_text(str(os.getpid()), encoding="utf-8")
time.sleep(payload["write_delay"])
Path(payload["late_marker"]).write_text("escaped", encoding="utf-8")
time.sleep(30)
'''
subprocess.Popen(
    [sys.executable, "-c", descendant_source, json.dumps(payload)],
    stdin=subprocess.DEVNULL,
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
    close_fds=True,
)
deadline = time.monotonic() + 3
while not Path(payload["child_pid"]).exists():
    if time.monotonic() >= deadline:
        raise RuntimeError("descendant did not report its pid")
    time.sleep(0.01)
if payload["mode"] == "success":
    os.ftruncate(payload["output_fd"], 0)
    os.pwrite(payload["output_fd"], b"private-image", 0)
    raise SystemExit(0)
signal.signal(signal.SIGTERM, signal.SIG_IGN)
time.sleep(30)
"""


@pytest.fixture
def adapter(tmp_path):
    model_path = tmp_path / "model"
    model_path.mkdir()
    (model_path / "model_index.json").write_text("{}", encoding="utf-8")

    diffusers_path = tmp_path / "diffusers"
    python_path = diffusers_path / ".venv/bin/python3"
    python_path.parent.mkdir(parents=True)
    python_path.symlink_to(sys.executable)

    return LocalMPSTrainingAdapter(
        model_path=str(model_path),
        working_dir=str(tmp_path / "working"),
        diffusers_path=str(diffusers_path),
    )


def _configure_real_process_tree(
    adapter,
    tmp_path,
    monkeypatch,
    *,
    mode: str,
    write_delay: float = 0.8,
):
    parent_pid_path = tmp_path / f"{mode}-parent.pid"
    child_pid_path = tmp_path / f"{mode}-child.pid"
    late_marker = tmp_path / f"{mode}-descendant-escaped"

    def build_process_tree_script(**kwargs):
        return _PROCESS_TREE_SCRIPT, json.dumps(
            {
                "mode": mode,
                "parent_pid": str(parent_pid_path),
                "child_pid": str(child_pid_path),
                "late_marker": str(late_marker),
                "write_delay": write_delay,
                "output_fd": kwargs["output_fd"],
            }
        )

    monkeypatch.setattr(
        local_mps_adapter,
        "_build_generation_script",
        build_process_tree_script,
    )
    return parent_pid_path, child_pid_path, late_marker


async def _wait_for_path(path: Path, timeout: float = 3.0) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while not path.exists():
        if asyncio.get_running_loop().time() >= deadline:
            raise AssertionError(f"Timed out waiting for {path}")
        await asyncio.sleep(0.01)


async def _wait_for_process_group_exit(group_id: int, timeout: float = 3.0) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while True:
        try:
            os.killpg(group_id, 0)
        except ProcessLookupError:
            return
        if asyncio.get_running_loop().time() >= deadline:
            raise AssertionError(f"Process group {group_id} remained alive")
        await asyncio.sleep(0.02)


async def _wait_for_pid_exit(process_id: int, timeout: float = 3.0) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while True:
        try:
            os.kill(process_id, 0)
        except ProcessLookupError:
            return
        if asyncio.get_running_loop().time() >= deadline:
            raise AssertionError(f"Process {process_id} remained alive")
        await asyncio.sleep(0.02)


def _force_kill_test_process_group(parent_pid_path: Path) -> None:
    if not parent_pid_path.exists():
        return
    try:
        os.killpg(int(parent_pid_path.read_text(encoding="utf-8")), signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass


class _StubbornGenerationProcess:
    def __init__(self):
        self.returncode = None
        self.terminated = False
        self.killed = False
        self.waited = False
        self._exited = asyncio.Event()

    def terminate(self):
        self.terminated = True

    def kill(self):
        self.killed = True
        self.returncode = -9
        self._exited.set()

    async def wait(self):
        self.waited = True
        await self._exited.wait()
        return self.returncode


class _UnreapableGenerationProcess(_StubbornGenerationProcess):
    def kill(self):
        self.killed = True
        self.returncode = -9


@pytest.mark.asyncio
async def test_process_escalates_and_reaps_stubborn_child(adapter, monkeypatch):
    process = _StubbornGenerationProcess()
    process_lease = generation_lifecycle.GenerationProcessLease(process, None)
    monkeypatch.setattr(
        generation_process,
        "GENERATION_TERMINATION_GRACE_SECONDS",
        0.01,
    )
    monkeypatch.setattr(
        generation_process,
        "GENERATION_KILL_REAP_TIMEOUT_SECONDS",
        0.5,
    )

    await generation_lifecycle.finalize_generation_resources(
        process=process_lease,
        process_communicated=False,
        workspace=None,
    )

    assert process.terminated is True
    assert process.killed is True
    assert process.waited is True
    assert process.returncode == -9


@pytest.mark.asyncio
async def test_unreapable_process_preserves_workspace_after_bounded_teardown(
    adapter,
    monkeypatch,
):
    process = _UnreapableGenerationProcess()
    process_lease = generation_lifecycle.GenerationProcessLease(process, None)
    workspace = generation_lifecycle._create_generation_workspace(adapter.working_dir)
    workspace_path = workspace.path
    monkeypatch.setattr(
        generation_process,
        "GENERATION_TERMINATION_GRACE_SECONDS",
        0.01,
    )
    monkeypatch.setattr(
        generation_process,
        "GENERATION_KILL_REAP_TIMEOUT_SECONDS",
        0.01,
    )

    with pytest.raises(RuntimeError, match="workspace preserved"):
        await generation_lifecycle.finalize_generation_resources(
            process=process_lease,
            process_communicated=False,
            workspace=workspace,
        )

    assert process.terminated is True
    assert process.killed is True
    assert workspace_path.is_dir()
    shutil.rmtree(workspace_path)


@pytest.mark.skipif(os.name != "posix", reason="POSIX process-group contract")
@pytest.mark.asyncio
async def test_real_success_kills_surviving_descendant_before_cleanup(
    adapter,
    tmp_path,
    monkeypatch,
):
    parent_pid_path, child_pid_path, late_marker = _configure_real_process_tree(
        adapter,
        tmp_path,
        monkeypatch,
        mode="success",
    )
    monkeypatch.setattr(
        generation_process,
        "GENERATION_TERMINATION_GRACE_SECONDS",
        0.05,
    )
    monkeypatch.setattr(
        generation_process,
        "GENERATION_KILL_REAP_TIMEOUT_SECONDS",
        2.0,
    )

    try:
        result = await asyncio.wait_for(
            adapter.generate_image(
                config=GenerationConfig(prompt="success tree", lora_path=""),
                lora_bytes=b"success-tree-lora",
            ),
            timeout=4.0,
        )
        await _wait_for_path(parent_pid_path)
        await _wait_for_path(child_pid_path)
        group_id = int(parent_pid_path.read_text(encoding="utf-8"))
        child_pid = int(child_pid_path.read_text(encoding="utf-8"))

        assert result.state is GenerationState.COMPLETED
        assert not list(adapter.working_dir.glob(".generation-*"))
        await _wait_for_process_group_exit(group_id)
        await _wait_for_pid_exit(group_id)
        await _wait_for_pid_exit(child_pid)
        await asyncio.sleep(0.9)
        assert not late_marker.exists()
    finally:
        _force_kill_test_process_group(parent_pid_path)


@pytest.mark.skipif(os.name != "posix", reason="POSIX process-group contract")
@pytest.mark.asyncio
async def test_real_timeout_kills_stubborn_descendant_before_cleanup(
    adapter,
    tmp_path,
    monkeypatch,
):
    parent_pid_path, child_pid_path, late_marker = _configure_real_process_tree(
        adapter,
        tmp_path,
        monkeypatch,
        mode="timeout",
    )
    monkeypatch.setattr(local_mps_adapter, "GENERATION_TIMEOUT_SECONDS", 0.2)
    monkeypatch.setattr(
        generation_process,
        "GENERATION_TERMINATION_GRACE_SECONDS",
        0.05,
    )
    monkeypatch.setattr(
        generation_process,
        "GENERATION_KILL_REAP_TIMEOUT_SECONDS",
        2.0,
    )

    try:
        result = await asyncio.wait_for(
            adapter.generate_image(
                config=GenerationConfig(prompt="timeout tree", lora_path=""),
                lora_bytes=b"timeout-tree-lora",
            ),
            timeout=4.0,
        )
        await _wait_for_path(parent_pid_path)
        await _wait_for_path(child_pid_path)
        group_id = int(parent_pid_path.read_text(encoding="utf-8"))
        child_pid = int(child_pid_path.read_text(encoding="utf-8"))

        assert result.state is GenerationState.FAILED
        assert result.error == "Generation timed out (0.2s)"
        assert not list(adapter.working_dir.glob(".generation-*"))
        await _wait_for_process_group_exit(group_id)
        await _wait_for_pid_exit(group_id)
        await _wait_for_pid_exit(child_pid)
        await asyncio.sleep(0.9)
        assert not late_marker.exists()
    finally:
        _force_kill_test_process_group(parent_pid_path)


@pytest.mark.skipif(os.name != "posix", reason="POSIX process-group contract")
@pytest.mark.asyncio
async def test_real_repeated_cancellation_kills_descendant_before_cleanup(
    adapter,
    tmp_path,
    monkeypatch,
):
    parent_pid_path, child_pid_path, late_marker = _configure_real_process_tree(
        adapter,
        tmp_path,
        monkeypatch,
        mode="cancel",
    )
    monkeypatch.setattr(local_mps_adapter, "GENERATION_TIMEOUT_SECONDS", 30)
    monkeypatch.setattr(
        generation_process,
        "GENERATION_TERMINATION_GRACE_SECONDS",
        0.05,
    )
    monkeypatch.setattr(
        generation_process,
        "GENERATION_KILL_REAP_TIMEOUT_SECONDS",
        2.0,
    )

    task = asyncio.create_task(
        adapter.generate_image(
            config=GenerationConfig(prompt="cancel tree", lora_path=""),
            lora_bytes=b"cancel-tree-lora",
        )
    )
    try:
        await _wait_for_path(parent_pid_path)
        await _wait_for_path(child_pid_path)
        group_id = int(parent_pid_path.read_text(encoding="utf-8"))
        child_pid = int(child_pid_path.read_text(encoding="utf-8"))
        task.cancel("first cancellation")
        await asyncio.sleep(0.01)
        task.cancel("repeated cancellation")

        with pytest.raises(asyncio.CancelledError) as cancelled:
            await asyncio.wait_for(task, timeout=4.0)

        assert str(cancelled.value) == "first cancellation"
        assert not list(adapter.working_dir.glob(".generation-*"))
        await _wait_for_process_group_exit(group_id)
        await _wait_for_pid_exit(group_id)
        await _wait_for_pid_exit(child_pid)
        await asyncio.sleep(0.9)
        assert not late_marker.exists()
    finally:
        if not task.done():
            task.cancel()
        _force_kill_test_process_group(parent_pid_path)


@pytest.mark.asyncio
async def test_cancellation_wins_over_cleanup_error_without_orphan_warning(
    adapter,
    monkeypatch,
    caplog,
):
    workspace = generation_lifecycle._create_generation_workspace(adapter.working_dir)
    workspace_path = workspace.path
    cleanup_started = threading.Event()
    allow_cleanup_failure = threading.Event()
    loop = asyncio.get_running_loop()
    loop_errors: list[dict[str, object]] = []
    previous_handler = loop.get_exception_handler()

    def fail_cleanup(_workspace):
        cleanup_started.set()
        allow_cleanup_failure.wait(timeout=5)
        raise RuntimeError("demonstrated cleanup failure")

    monkeypatch.setattr(
        generation_lifecycle,
        "_cleanup_generation_workspace",
        fail_cleanup,
    )
    loop.set_exception_handler(lambda _loop, context: loop_errors.append(context))
    caplog.set_level("ERROR")
    task = asyncio.create_task(
        generation_lifecycle.finalize_generation_resources(
            process=None,
            process_communicated=False,
            workspace=workspace,
        )
    )
    try:
        assert await asyncio.to_thread(cleanup_started.wait, 5) is True
        task.cancel("first cancellation")
        await asyncio.sleep(0)
        task.cancel("repeated cancellation")
        allow_cleanup_failure.set()

        with pytest.raises(asyncio.CancelledError) as cancelled:
            await task

        assert str(cancelled.value) == "first cancellation"
        assert isinstance(cancelled.value.__cause__, RuntimeError)
        assert "demonstrated cleanup failure" in str(cancelled.value.__cause__)
        await asyncio.sleep(0)
        assert not loop_errors
        assert "failed while preserving caller cancellation" in caplog.text
    finally:
        allow_cleanup_failure.set()
        loop.set_exception_handler(previous_handler)
        generation_lifecycle._close_generation_workspace(workspace)
        if workspace_path.exists():
            shutil.rmtree(workspace_path)
