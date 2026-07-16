"""Contracts for Local MPS training adapter async offload boundaries."""

import asyncio
import base64
import json
import os
import stat
import threading
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from kestrel_sovereign.features.training.adapters import (
    _local_mps_generation_lifecycle as generation_lifecycle,
)
from kestrel_sovereign.features.training.adapters import (
    _local_mps_generation_workspace as generation_workspace,
)
from kestrel_sovereign.features.training.adapters.local_mps_adapter import (
    LocalMPSTrainingAdapter,
)
from kestrel_sovereign.features.training.types import (
    GenerationConfig,
    GenerationState,
    TrainingConfig,
)


MODULE = "kestrel_sovereign.features.training.adapters.local_mps_adapter"


def _read_fd(fd: int) -> bytes:
    size = os.fstat(fd).st_size
    return os.pread(fd, size, 0)


def _write_fd(fd: int, data: bytes) -> None:
    os.ftruncate(fd, 0)
    os.pwrite(fd, data, 0)


@pytest.fixture
def adapter(tmp_path):
    model_path = tmp_path / "model"
    model_path.mkdir()
    (model_path / "model_index.json").write_text("{}", encoding="utf-8")

    diffusers_path = tmp_path / "diffusers"
    training_script = (
        diffusers_path / "examples/text_to_image/train_text_to_image_lora_sdxl.py"
    )
    training_script.parent.mkdir(parents=True)
    training_script.write_text("# training script", encoding="utf-8")
    python_path = diffusers_path / ".venv/bin/python3"
    python_path.parent.mkdir(parents=True)
    python_path.write_text("#!/usr/bin/env python3", encoding="utf-8")

    return LocalMPSTrainingAdapter(
        model_path=str(model_path),
        working_dir=str(tmp_path / "working"),
        diffusers_path=str(diffusers_path),
    )


@pytest.fixture
def tracked_to_thread():
    real_to_thread = asyncio.to_thread
    calls = []

    async def tracking_to_thread(func, *args, **kwargs):
        calls.append(func)
        return await real_to_thread(func, *args, **kwargs)

    return calls, tracking_to_thread


@pytest.mark.asyncio
async def test_start_training_offloads_file_setup_and_process_launch(
    adapter, tracked_to_thread
):
    calls, tracking_to_thread = tracked_to_thread
    process = MagicMock(pid=1234)

    with (
        patch.object(adapter, "is_available", return_value=True),
        patch(
            f"{MODULE}.subprocess.Popen",
            return_value=process,
        ),
        patch(
            f"{MODULE}.asyncio.to_thread",
            side_effect=tracking_to_thread,
        ),
    ):
        job = await adapter.start_training(
            companion_id="companion-123",
            avatar_data=b"avatar",
            config=TrainingConfig(trigger_word="TOKTEST", steps=3),
        )

    assert job.state.value == "training"
    assert adapter._training_processes[job.job_id] is process
    call_names = [getattr(func, "__name__", repr(func)) for func in calls]
    assert "_prepare_training_files" in call_names
    assert "_start_training_process" in call_names


@pytest.mark.asyncio
async def test_failed_status_offloads_log_tail_read(adapter, tracked_to_thread):
    calls, tracking_to_thread = tracked_to_thread
    job_id = "job-failed"
    log_file = adapter.working_dir / "training.log"
    log_file.write_text("failure details", encoding="utf-8")
    process = MagicMock()
    process.poll.return_value = 2
    adapter._training_processes[job_id] = process
    adapter._active_jobs[job_id] = {
        "started_at": datetime.now(timezone.utc),
        "output_dir": adapter.output_dir / job_id,
        "log_file": log_file,
        "steps": 3,
    }

    with patch(f"{MODULE}.asyncio.to_thread", side_effect=tracking_to_thread):
        status = await adapter.get_status(job_id)

    assert status.state.value == "failed"
    assert "failure details" in status.error
    call_names = [getattr(func, "__name__", repr(func)) for func in calls]
    assert "_read_text_tail" in call_names


@pytest.mark.asyncio
async def test_status_without_process_offloads_output_scan(adapter, tracked_to_thread):
    calls, tracking_to_thread = tracked_to_thread
    job_id = "job-output-scan"
    output_dir = adapter.output_dir / job_id
    output_dir.mkdir(parents=True)
    (output_dir / "checkpoint-1").mkdir()
    (output_dir / "checkpoint-1" / "weights.safetensors").write_bytes(b"weights")
    adapter._active_jobs[job_id] = {
        "started_at": datetime.now(timezone.utc),
        "output_dir": output_dir,
        "steps": 3,
    }

    with patch(f"{MODULE}.asyncio.to_thread", side_effect=tracking_to_thread):
        status = await adapter.get_status(job_id)

    assert status.state.value == "completed"
    call_names = [getattr(func, "__name__", repr(func)) for func in calls]
    assert "_find_any_lora_files" in call_names


@pytest.mark.asyncio
async def test_download_weights_offloads_lora_read(adapter, tracked_to_thread):
    calls, tracking_to_thread = tracked_to_thread
    job_id = "job-complete"
    output_dir = adapter.output_dir / job_id
    output_dir.mkdir(parents=True)
    (output_dir / "weights.safetensors").write_bytes(b"weights")
    adapter._active_jobs[job_id] = {"output_dir": output_dir}

    with patch(f"{MODULE}.asyncio.to_thread", side_effect=tracking_to_thread):
        weights = await adapter.download_weights(job_id)

    assert weights == b"weights"
    call_names = [getattr(func, "__name__", repr(func)) for func in calls]
    assert "_read_latest_lora" in call_names


@pytest.mark.asyncio
async def test_cancel_offloads_process_termination(adapter, tracked_to_thread):
    calls, tracking_to_thread = tracked_to_thread
    process = MagicMock()
    adapter._training_processes["job-cancel"] = process

    with patch(f"{MODULE}.asyncio.to_thread", side_effect=tracking_to_thread):
        cancelled = await adapter.cancel("job-cancel")

    assert cancelled is True
    assert "job-cancel" not in adapter._training_processes
    call_names = [getattr(func, "__name__", repr(func)) for func in calls]
    assert "_terminate_process" in call_names


@pytest.mark.asyncio
async def test_cleanup_offloads_dataset_removal(adapter, tracked_to_thread):
    calls, tracking_to_thread = tracked_to_thread
    job_id = "job-cleanup"
    dataset_dir = adapter.datasets_dir / job_id
    dataset_dir.mkdir(parents=True)
    (dataset_dir / "avatar.png").write_bytes(b"avatar")
    adapter._active_jobs[job_id] = {"dataset_dir": dataset_dir}

    with patch(f"{MODULE}.asyncio.to_thread", side_effect=tracking_to_thread):
        await adapter.cleanup(job_id)

    assert job_id not in adapter._active_jobs
    assert not dataset_dir.exists()
    call_names = [getattr(func, "__name__", repr(func)) for func in calls]
    assert "_cleanup_dataset_dir" in call_names


@pytest.mark.asyncio
async def test_generate_image_offloads_temp_lora_and_output_file_io(
    adapter, tracked_to_thread
):
    calls, tracking_to_thread = tracked_to_thread
    observed_lora = None
    observed_fds = None

    class FakeProcess:
        returncode = 0

        def __init__(self, payload):
            self.payload = payload

        async def communicate(self):
            nonlocal observed_lora
            observed_lora = _read_fd(self.payload["lora_fd"])
            _write_fd(self.payload["output_fd"], b"png-bytes")
            return b"OK", b""

    async def fake_create_subprocess_exec(*args, **kwargs):
        nonlocal observed_fds
        payload = json.loads(args[3])
        observed_fds = tuple(kwargs["pass_fds"])
        return FakeProcess(payload)

    config = GenerationConfig(
        prompt="TOKTEST portrait",
        lora_path="",
        width=64,
        height=64,
        num_inference_steps=1,
    )

    with (
        patch(
            f"{MODULE}.asyncio.create_subprocess_exec",
            side_effect=fake_create_subprocess_exec,
        ),
        patch(f"{MODULE}.asyncio.to_thread", side_effect=tracking_to_thread),
    ):
        result = await adapter.generate_image(config=config, lora_bytes=b"lora")

    assert result.state is GenerationState.COMPLETED
    assert result.images == ["data:image/png;base64,cG5nLWJ5dGVz"]
    assert observed_lora == b"lora"
    assert observed_fds is not None
    assert len(observed_fds) == 2
    assert not list(adapter.working_dir.glob(".generation-*"))
    call_names = [getattr(func, "__name__", repr(func)) for func in calls]
    assert "_create_generation_workspace" in call_names
    assert "_path_exists" in call_names
    assert "_create_generation_artifact" in call_names
    assert "_read_generation_artifact" in call_names
    assert "_cleanup_generation_workspace" in call_names


@pytest.mark.asyncio
async def test_generate_image_cleans_temp_lora_on_validation_failure(
    adapter, tracked_to_thread
):
    calls, tracking_to_thread = tracked_to_thread
    (adapter.diffusers_path / ".venv/bin/python3").unlink()
    config = GenerationConfig(prompt="TOKTEST portrait", lora_path="")

    with patch(f"{MODULE}.asyncio.to_thread", side_effect=tracking_to_thread):
        result = await adapter.generate_image(config=config, lora_bytes=b"lora")

    assert result.state is GenerationState.FAILED
    assert "Diffusers Python not found" in result.error
    assert not list(adapter.working_dir.glob(".generation-*"))
    call_names = [getattr(func, "__name__", repr(func)) for func in calls]
    assert "_create_generation_artifact" in call_names
    assert "_cleanup_generation_workspace" in call_names


@pytest.mark.asyncio
async def test_generation_cancellation_during_workspace_creation_cleans_workspace(
    adapter,
    monkeypatch,
):
    real_create_workspace = generation_workspace._create_generation_workspace
    creation_started = threading.Event()
    allow_creation_to_finish = threading.Event()
    created_workspace = None

    def slow_create_workspace(working_dir):
        nonlocal created_workspace
        created_workspace = real_create_workspace(working_dir)
        creation_started.set()
        allow_creation_to_finish.wait(timeout=5)
        return created_workspace

    monkeypatch.setattr(
        generation_workspace,
        "_create_generation_workspace",
        slow_create_workspace,
    )

    task = asyncio.create_task(
        adapter.generate_image(
            config=GenerationConfig(prompt="cancel during creation", lora_path=""),
            lora_bytes=b"cancel-lora",
        )
    )
    assert await asyncio.to_thread(creation_started.wait, 5) is True
    task.cancel()
    allow_creation_to_finish.set()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert created_workspace is not None
    assert not created_workspace.path.exists()
    assert created_workspace._closed is True


@pytest.mark.asyncio
async def test_concurrent_generation_uses_private_lora_and_output_artifacts(
    adapter,
    monkeypatch,
):
    both_started = asyncio.Event()
    first_output_written = asyncio.Event()
    second_output_written = asyncio.Event()
    started = 0
    payloads = {}
    observed_lora = {}
    workspaces = []
    image_bytes = {
        "first": b"first-private-image",
        "second": b"second-private-image",
    }

    class FakeProcess:
        def __init__(self, payload):
            self.payload = payload
            self.returncode = None

        async def communicate(self):
            prompt = self.payload["prompt"]
            await both_started.wait()
            observed_lora[prompt] = _read_fd(self.payload["lora_fd"])

            if prompt == "first":
                _write_fd(self.payload["output_fd"], image_bytes[prompt])
                first_output_written.set()
                await second_output_written.wait()
            else:
                await first_output_written.wait()
                _write_fd(self.payload["output_fd"], image_bytes[prompt])
                second_output_written.set()

            self.returncode = 0
            return b"OK", b""

    async def fake_create_subprocess_exec(*args, **_kwargs):
        nonlocal started
        payload = json.loads(args[3])
        payloads[payload["prompt"]] = payload
        started += 1
        if started == 2:
            both_started.set()
        return FakeProcess(payload)

    real_create_workspace = generation_lifecycle.create_generation_workspace

    async def track_workspace(working_dir):
        lease = await real_create_workspace(working_dir)
        workspaces.append(
            (lease.path, stat.S_IMODE(os.fstat(lease.workspace_fd).st_mode))
        )
        return lease

    monkeypatch.setattr(
        f"{MODULE}.asyncio.create_subprocess_exec",
        fake_create_subprocess_exec,
    )
    monkeypatch.setattr(
        generation_lifecycle,
        "create_generation_workspace",
        track_workspace,
    )

    first_result, second_result = await asyncio.gather(
        adapter.generate_image(
            config=GenerationConfig(prompt="first", lora_path=""),
            lora_bytes=b"first-private-lora",
        ),
        adapter.generate_image(
            config=GenerationConfig(prompt="second", lora_path=""),
            lora_bytes=b"second-private-lora",
        ),
    )

    assert observed_lora == {
        "first": b"first-private-lora",
        "second": b"second-private-lora",
    }
    assert first_result.images == [
        f"data:image/png;base64,{base64.b64encode(image_bytes['first']).decode()}"
    ]
    assert second_result.images == [
        f"data:image/png;base64,{base64.b64encode(image_bytes['second']).decode()}"
    ]

    assert len(payloads) == 2
    assert len({payload["lora_fd"] for payload in payloads.values()}) == 2
    assert len({payload["output_fd"] for payload in payloads.values()}) == 2
    assert len({path for path, _mode in workspaces}) == 2
    for workspace_path, workspace_mode in workspaces:
        assert workspace_path.parent == adapter.working_dir.resolve()
        assert workspace_path.name.startswith(".generation-")
        assert workspace_mode == 0o700
        assert not workspace_path.exists()


@pytest.mark.asyncio
async def test_generation_cleans_workspace_after_subprocess_failure(
    adapter, monkeypatch
):
    workspaces = []

    class FakeProcess:
        returncode = 7

        async def communicate(self):
            return b"", b"diffusers failed"

    async def fake_create_subprocess_exec(*args, **_kwargs):
        return FakeProcess()

    real_create_workspace = generation_lifecycle.create_generation_workspace

    async def track_workspace(working_dir):
        lease = await real_create_workspace(working_dir)
        workspaces.append(lease.path)
        return lease

    monkeypatch.setattr(
        f"{MODULE}.asyncio.create_subprocess_exec",
        fake_create_subprocess_exec,
    )
    monkeypatch.setattr(
        generation_lifecycle,
        "create_generation_workspace",
        track_workspace,
    )

    result = await adapter.generate_image(
        config=GenerationConfig(prompt="failure", lora_path=""),
        lora_bytes=b"failure-lora",
    )

    assert result.state is GenerationState.FAILED
    assert "diffusers failed" in result.error
    assert len(workspaces) == 1
    assert not workspaces[0].exists()


class _UnfinishedGenerationProcess:
    def __init__(self):
        self.returncode = None
        self.started = asyncio.Event()
        self.terminated = False
        self.waited = False

    async def communicate(self):
        self.started.set()
        await asyncio.Event().wait()

    def terminate(self):
        self.terminated = True
        self.returncode = -15

    def kill(self):
        self.returncode = -9

    async def wait(self):
        self.waited = True
        return self.returncode


@pytest.mark.asyncio
async def test_generation_timeout_reaps_process_before_workspace_cleanup(
    adapter,
    monkeypatch,
):
    process = _UnfinishedGenerationProcess()
    workspace = None

    real_create_workspace = generation_lifecycle.create_generation_workspace

    async def track_workspace(working_dir):
        nonlocal workspace
        lease = await real_create_workspace(working_dir)
        workspace = lease.path
        return lease

    async def fake_create_subprocess_exec(*args, **_kwargs):
        return process

    monkeypatch.setattr(
        f"{MODULE}.asyncio.create_subprocess_exec",
        fake_create_subprocess_exec,
    )
    monkeypatch.setattr(
        generation_lifecycle,
        "create_generation_workspace",
        track_workspace,
    )
    monkeypatch.setattr(f"{MODULE}.GENERATION_TIMEOUT_SECONDS", 0.01)

    result = await adapter.generate_image(
        config=GenerationConfig(prompt="timeout", lora_path=""),
        lora_bytes=b"timeout-lora",
    )

    assert result.state is GenerationState.FAILED
    assert result.error == "Generation timed out (0.01s)"
    assert process.terminated is True
    assert process.waited is True
    assert workspace is not None
    assert not workspace.exists()


@pytest.mark.asyncio
async def test_generation_cancellation_reaps_process_before_workspace_cleanup(
    adapter,
    monkeypatch,
):
    process = _UnfinishedGenerationProcess()
    workspace = None

    real_create_workspace = generation_lifecycle.create_generation_workspace

    async def track_workspace(working_dir):
        nonlocal workspace
        lease = await real_create_workspace(working_dir)
        workspace = lease.path
        return lease

    async def fake_create_subprocess_exec(*args, **_kwargs):
        return process

    monkeypatch.setattr(
        f"{MODULE}.asyncio.create_subprocess_exec",
        fake_create_subprocess_exec,
    )
    monkeypatch.setattr(
        generation_lifecycle,
        "create_generation_workspace",
        track_workspace,
    )

    task = asyncio.create_task(
        adapter.generate_image(
            config=GenerationConfig(prompt="cancel", lora_path=""),
            lora_bytes=b"cancel-lora",
        )
    )
    await process.started.wait()
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert process.terminated is True
    assert process.waited is True
    assert workspace is not None
    assert not workspace.exists()


@pytest.mark.asyncio
async def test_generation_cancellation_during_process_creation_reaps_child(
    adapter,
    monkeypatch,
):
    process = _UnfinishedGenerationProcess()
    creation_started = asyncio.Event()
    allow_creation_to_finish = asyncio.Event()
    workspace = None

    real_create_workspace = generation_lifecycle.create_generation_workspace

    async def track_workspace(working_dir):
        nonlocal workspace
        lease = await real_create_workspace(working_dir)
        workspace = lease.path
        return lease

    async def fake_create_subprocess_exec(*args, **_kwargs):
        creation_started.set()
        await allow_creation_to_finish.wait()
        return process

    monkeypatch.setattr(
        f"{MODULE}.asyncio.create_subprocess_exec",
        fake_create_subprocess_exec,
    )
    monkeypatch.setattr(
        generation_lifecycle,
        "create_generation_workspace",
        track_workspace,
    )

    task = asyncio.create_task(
        adapter.generate_image(
            config=GenerationConfig(prompt="cancel during spawn", lora_path=""),
            lora_bytes=b"cancel-lora",
        )
    )
    await creation_started.wait()
    task.cancel()
    allow_creation_to_finish.set()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert process.terminated is True
    assert process.waited is True
    assert workspace is not None
    assert not workspace.exists()
