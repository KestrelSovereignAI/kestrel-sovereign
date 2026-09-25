import asyncio
import sys
import types
import zipfile
from datetime import datetime, timezone
from io import BytesIO
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from kestrel_sovereign.features.training.adapters.replicate_adapter import (
    ReplicateTrainingAdapter,
)
from kestrel_sovereign.features.training.adapters.runpod_adapter import (
    RunPodTrainingAdapter,
)
from kestrel_sovereign.features.training.types import (
    TrainingConfig,
    TrainingJob,
    TrainingState,
)


def _runpod_session(*, persistent: bool):
    return SimpleNamespace(
        pod_id="pod-123",
        backend_base_url="https://pod.example",
        profile=SimpleNamespace(
            persistent_pod_id="${RUNPOD_POD_ID}" if persistent else None,
        ),
    )


def _runpod_manager(session):
    async def never_submits(**_kwargs):
        await asyncio.Event().wait()

    return SimpleNamespace(
        _session=None,
        _lock=asyncio.Lock(),
        _expand_single_env_var=lambda value: "pod-123" if value else None,
        start_training_pod=AsyncMock(return_value=session),
        submit_training_job=never_submits,
        stop_session=AsyncMock(),
        terminate_session=AsyncMock(),
        provider=SimpleNamespace(
            stop_pod=Mock(return_value={"id": session.pod_id}),
            terminate_pod=Mock(return_value=None),
        ),
    )


@pytest.mark.asyncio
async def test_runpod_cleanup_pauses_the_managers_current_persistent_pod():
    session = _runpod_session(persistent=True)
    manager = _runpod_manager(session)
    manager._session = session
    adapter = RunPodTrainingAdapter(manager=manager)
    job = await adapter.start_training("companion", b"avatar")

    await adapter.cleanup(job.job_id)

    manager.stop_session.assert_awaited_once_with()
    manager.terminate_session.assert_not_awaited()
    manager.provider.terminate_pod.assert_not_called()
    assert job.job_id not in adapter._active_jobs


@pytest.mark.asyncio
async def test_runpod_cleanup_pauses_a_persistent_pod_the_manager_no_longer_holds():
    session = _runpod_session(persistent=True)
    manager = _runpod_manager(session)
    adapter = RunPodTrainingAdapter(manager=manager)
    job = await adapter.start_training("companion", b"avatar")

    await adapter.cleanup(job.job_id)

    manager.stop_session.assert_not_awaited()
    # terminate_session swallows stop failures, so the provider stop is used.
    manager.terminate_session.assert_not_awaited()
    manager.provider.stop_pod.assert_called_once_with("pod-123")
    manager.provider.terminate_pod.assert_not_called()
    assert job.job_id not in adapter._active_jobs


@pytest.mark.asyncio
async def test_runpod_cleanup_terminates_on_demand_pod():
    session = _runpod_session(persistent=False)
    manager = _runpod_manager(session)
    manager._session = session
    adapter = RunPodTrainingAdapter(manager=manager)
    job = await adapter.start_training("companion", b"avatar")

    await adapter.cleanup(job.job_id)

    manager.provider.terminate_pod.assert_called_once_with("pod-123")
    manager.provider.stop_pod.assert_not_called()
    manager.stop_session.assert_not_awaited()
    manager.terminate_session.assert_not_awaited()
    assert manager._session is None
    assert job.job_id not in adapter._active_jobs


def test_replicate_training_zip_contains_avatar_bytes():
    adapter = ReplicateTrainingAdapter()
    zip_bytes = adapter._build_training_zip(b"avatar-bytes")

    with zipfile.ZipFile(BytesIO(zip_bytes)) as zf:
        assert zf.namelist() == ["avatar_01.jpg"]
        assert zf.read("avatar_01.jpg") == b"avatar-bytes"


@pytest.mark.asyncio
async def test_replicate_training_uses_client_file_input(monkeypatch):
    captured = {}

    class _FakeTraining:
        id = "training-123"
        status = "succeeded"
        output = {
            "version": "model-version",
            "weights": "https://replicate.delivery/weights.tar",
        }

    class _FakeTrainings:
        def create(self, *, destination, version, input):
            training_file = input["input_images"]
            assert not isinstance(training_file, str)
            captured["filename"] = training_file.name
            captured["zip_bytes"] = training_file.read()
            return _FakeTraining()

        def get(self, training_id):
            assert training_id == "training-123"
            return _FakeTraining()

    monkeypatch.setitem(
        sys.modules,
        "replicate",
        types.SimpleNamespace(trainings=_FakeTrainings()),
    )

    adapter = ReplicateTrainingAdapter()
    job = TrainingJob(
        job_id="job-3",
        companion_id="companion",
        provider=adapter.provider_name,
        state=TrainingState.PENDING,
        trigger_word="TOKcompanion",
        created_at=datetime.now(timezone.utc),
        config=TrainingConfig(),
    )
    adapter._training_data[job.job_id] = {}

    await adapter._run_training(job, b"avatar-bytes", "TOKcompanion")

    assert captured["filename"].endswith(".zip")
    with zipfile.ZipFile(BytesIO(captured["zip_bytes"])) as zf:
        assert zf.read("avatar_01.jpg") == b"avatar-bytes"
    assert job.state == TrainingState.COMPLETED
    assert job.provider_job_id == "training-123"
    assert job.output_path == "https://replicate.delivery/weights.tar"
