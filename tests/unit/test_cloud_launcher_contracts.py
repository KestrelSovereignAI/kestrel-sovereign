"""Contracts for cloud launcher env handling and model identity."""

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

runpod = pytest.importorskip("runpod", reason="runpod not installed (cloud extras)")

from kestrel_sovereign.features.gcp_compute.core import (
    GCPComputeEngineManagerCore,
)
from kestrel_sovereign.features.gcp_compute.models import (
    GCPComputeSession,
    GPUProfile as GCPGPUProfile,
    InstanceStatus as GCPInstanceStatus,
)
from kestrel_sovereign.features.runpod.feature import RunPodFeature
from kestrel_sovereign.features.runpod.providers import DirectRunPodProvider
from kestrel_sovereign.features.vastai.feature import VastAIFeature
from kestrel_sovereign.features.vastai.models import (
    GPUProfile as VastGPUProfile,
    InstanceStatus as VastInstanceStatus,
    VastAISession,
)


def _gcp_profile(default_model=None):
    return GCPGPUProfile(
        id="budget",
        name="Budget",
        task_type="test",
        machine_type="n1-standard-4",
        gpu_type="nvidia-tesla-t4",
        gpu_count=1,
        boot_disk_size_gb=100,
        image_name="gcr.io/test/image:latest",
        default_model=default_model,
    )


def _vast_profile(default_model=None):
    return VastGPUProfile(
        id="budget",
        name="Budget",
        task_type="test",
        image_name="pytorch/pytorch",
        disk_gb=20,
        gpu_ram_min=8,
        default_model=default_model,
    )


def test_gcp_session_preserves_absent_model_name():
    now = datetime.now(timezone.utc)
    session = GCPComputeSession(
        instance_name="test-instance",
        zone="us-central1-a",
        profile=_gcp_profile(default_model=None),
        task_profile="budget",
        model_name=None,
        status=GCPInstanceStatus.RUNNING,
        ttl_seconds=3600,
        started_at=now,
        expires_at=now,
    )

    assert session.to_dict()["model_name"] is None


def test_vast_session_preserves_absent_model_name():
    now = datetime.now(timezone.utc)
    session = VastAISession(
        instance_id=123,
        profile=_vast_profile(default_model=None),
        task_profile="budget",
        model_name=None,
        status=VastInstanceStatus.RUNNING,
        ttl_seconds=3600,
        started_at=now,
        expires_at=now,
    )

    assert session.to_dict()["model_name"] is None


def test_gcp_startup_script_skips_unset_env_values():
    manager = GCPComputeEngineManagerCore(config={"manager": {}, "profiles": {}})

    script = manager._build_startup_script(
        _gcp_profile(default_model=None),
        {
            "KESTREL_PROFILE": "budget",
            "TARGET_MODEL": None,
        },
    )

    assert 'export KESTREL_PROFILE="budget"' in script
    assert "TARGET_MODEL" not in script


@pytest.mark.asyncio
async def test_vast_feature_omits_target_model_override_when_unset():
    feature = VastAIFeature(agent=None)
    feature.manager = SimpleNamespace(
        profiles={"budget": object()},
        start_session=AsyncMock(return_value={"active": False}),
    )
    feature.llm_service = None

    await feature._start(profile_name="budget", model_name="", ttl_seconds="")

    metadata = feature.manager.start_session.await_args.kwargs["metadata"]
    assert metadata["env_overrides"] == {"KESTREL_PROFILE": "budget"}


@pytest.mark.asyncio
async def test_runpod_feature_omits_target_model_override_when_unset():
    feature = RunPodFeature(agent=None)
    feature.manager = SimpleNamespace(
        profiles={"llm": object()},
        start_session=AsyncMock(return_value={"inference_url": None}),
    )
    feature.llm_service = None

    await feature._start(model_name="", task_profile="llm", ttl_seconds="", pod_type="")

    metadata = feature.manager.start_session.await_args.kwargs["metadata"]
    assert metadata["env_overrides"] == {"KESTREL_PROFILE": "llm"}


def test_runpod_provider_drops_none_env_values():
    profile = SimpleNamespace(
        id="llm",
        image_name="repo/image:latest",
        gpu_type_id="NVIDIA",
        container_disk_gb=20,
        ports=["8000/http"],
        env={"KESTREL_PROFILE": "llm"},
        network_volume_id=None,
        volume_mount_path=None,
        volume_gb=0,
        template_id=None,
    )

    with patch(
        "kestrel_sovereign.features.runpod.providers.runpod.create_pod",
        return_value={"id": "pod-123"},
    ) as create_pod:
        provider = DirectRunPodProvider(api_key="test-key")
        provider.start_pod(
            profile,
            {
                "name": "kestrel-llm",
                "env_overrides": {"TARGET_MODEL": None, "EXTRA_FLAG": "1"},
            },
        )

    pod_config = create_pod.call_args.kwargs
    assert pod_config["env"] == {
        "KESTREL_PROFILE": "llm",
        "EXTRA_FLAG": "1",
    }
