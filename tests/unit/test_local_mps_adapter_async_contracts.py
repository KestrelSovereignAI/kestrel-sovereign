"""Contracts for Local MPS training adapter async offload boundaries."""

import asyncio
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from kestrel_sovereign.features.training.adapters.local_mps_adapter import LocalMPSTrainingAdapter
from kestrel_sovereign.features.training.types import (
    GenerationConfig,
    GenerationState,
    TrainingConfig,
)


MODULE = "kestrel_sovereign.features.training.adapters.local_mps_adapter"


@pytest.fixture
def adapter(tmp_path):
    model_path = tmp_path / "model"
    model_path.mkdir()
    (model_path / "model_index.json").write_text("{}", encoding="utf-8")

    diffusers_path = tmp_path / "diffusers"
    training_script = diffusers_path / "examples/text_to_image/train_text_to_image_lora_sdxl.py"
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
async def test_start_training_offloads_file_setup_and_process_launch(adapter, tracked_to_thread):
    calls, tracking_to_thread = tracked_to_thread
    process = MagicMock(pid=1234)

    with patch.object(adapter, "is_available", return_value=True), patch(
        f"{MODULE}.subprocess.Popen",
        return_value=process,
    ), patch(
        f"{MODULE}.asyncio.to_thread",
        side_effect=tracking_to_thread,
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
async def test_generate_image_offloads_temp_lora_and_output_file_io(adapter, tracked_to_thread):
    calls, tracking_to_thread = tracked_to_thread
    output_path = adapter.working_dir / "generated_selfie.png"

    class FakeProcess:
        returncode = 0

        async def communicate(self):
            output_path.write_bytes(b"png-bytes")
            return b"OK", b""

    async def fake_create_subprocess_exec(*args, **kwargs):
        return FakeProcess()

    config = GenerationConfig(
        prompt="TOKTEST portrait",
        lora_path="",
        width=64,
        height=64,
        num_inference_steps=1,
    )

    with patch(
        f"{MODULE}.asyncio.create_subprocess_exec",
        side_effect=fake_create_subprocess_exec,
    ), patch(f"{MODULE}.asyncio.to_thread", side_effect=tracking_to_thread):
        result = await adapter.generate_image(config=config, lora_bytes=b"lora")

    assert result.state is GenerationState.COMPLETED
    assert result.images == ["data:image/png;base64,cG5nLWJ5dGVz"]
    assert not (adapter.working_dir / "temp_lora.safetensors").exists()
    assert not output_path.exists()
    call_names = [getattr(func, "__name__", repr(func)) for func in calls]
    assert "_path_exists" in call_names
    assert "write_bytes" in call_names
    assert "read_bytes" in call_names
    assert "unlink" in call_names


@pytest.mark.asyncio
async def test_generate_image_cleans_temp_lora_on_validation_failure(adapter, tracked_to_thread):
    calls, tracking_to_thread = tracked_to_thread
    (adapter.diffusers_path / ".venv/bin/python3").unlink()
    config = GenerationConfig(prompt="TOKTEST portrait", lora_path="")

    with patch(f"{MODULE}.asyncio.to_thread", side_effect=tracking_to_thread):
        result = await adapter.generate_image(config=config, lora_bytes=b"lora")

    assert result.state is GenerationState.FAILED
    assert "Diffusers Python not found" in result.error
    assert not (adapter.working_dir / "temp_lora.safetensors").exists()
    call_names = [getattr(func, "__name__", repr(func)) for func in calls]
    assert "write_bytes" in call_names
    assert "unlink" in call_names
