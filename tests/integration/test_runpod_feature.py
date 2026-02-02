from types import SimpleNamespace

import pytest

from kestrel_sovereign.features.runpod.feature import RunPodFeature
from kestrel_sovereign.llm.service import BackendType


class FakeRunPodManager:
    def __init__(self):
        profile = SimpleNamespace(max_context_window=32768)
        image_profile = SimpleNamespace(max_context_window=4096)
        self.profiles = {"llm": profile, "image": image_profile}
        self.default_ttl_seconds = 1800
        self.started = False
        self.start_calls = []
        self.stop_calls = 0

    async def start_session(self, **kwargs):
        self.started = True
        self.start_calls.append(kwargs)
        profile = kwargs["task_profile"]
        base_response = {
            "pod_id": "pod-123",
            "task_profile": profile,
            "model_name": kwargs.get("model_name") or "llama-3",
            "remaining_ttl_seconds": kwargs.get("ttl_seconds", 1800),
            "status": "ready",
        }
        if profile == "image":
            base_response["image_endpoint"] = "http://gpu:9000/invoke"
        else:
            base_response["inference_url"] = "http://gpu:8000/v1"
        return base_response

    async def get_status(self, **_):
        return {
            "pod_id": "pod-123",
            "task_profile": "llm",
            "model_name": "llama-3",
            "status": "ready",
            "remaining_ttl_seconds": 1700,
        }

    async def stop_session(self):
        self.stop_calls += 1
        self.started = False
        return {
            "pod_id": "pod-123",
            "task_profile": "llm",
            "status": "terminating",
            "remaining_ttl_seconds": 0,
        }


class DummyLLMService:
    """Fake LLMService that tracks backend switching calls."""

    def __init__(self):
        self.last_backend = BackendType.CLOUD
        self.switch_calls = []
        self.deactivate_reasons = []

    def switch_backend(self, backend, *, config):
        self.last_backend = backend
        self.switch_calls.append((backend, config))

    def _deactivate_remote_backend(self, reason=None):
        self.last_backend = BackendType.CLOUD
        self.deactivate_reasons.append(reason)

    def get_backend_status(self):
        return {
            "current_backend": self.last_backend.value,
            "remote_active": self.last_backend == BackendType.REMOTE_GPU,
        }


@pytest.fixture
async def runpod_feature(monkeypatch):
    fake_manager = FakeRunPodManager()
    monkeypatch.setattr("kestrel_sovereign.features.runpod.feature.RunPodManager", lambda: fake_manager)

    llm_service = DummyLLMService()
    agent = SimpleNamespace(llm_service=llm_service)

    feature = RunPodFeature(agent)
    await feature.initialize()
    feature._post_json = lambda url, payload: {"url": url, "payload": payload}

    return feature, fake_manager, llm_service


@pytest.mark.asyncio
async def test_manage_gpu_start_and_stop(runpod_feature):
    feature, manager, llm_service = runpod_feature

    start_result = await feature.manage_gpu(
        action="on",
        model_name="llama-3-70b",
        task_profile="llm",
        ttl_seconds="120",
        pod_type="h100-single",
    )

    assert manager.started is True
    assert llm_service.switch_calls
    assert start_result["session"]["status"] == "ready"

    stop_result = await feature.manage_gpu(action="off")
    assert stop_result["session"]["status"] == "terminating"
    assert manager.started is False
    assert llm_service.deactivate_reasons[-1] == "Requested via !gpu off"


@pytest.mark.asyncio
async def test_image_generation_tears_down_session(runpod_feature):
    """Test that image generation automatically tears down the GPU session.

    Note: Uses generate_image_on_runpod method (dream_image command was removed).
    """
    feature, manager, llm_service = runpod_feature

    # Use the internal method (dream_image was removed, see feature.py lines 65-68)
    image_result = await feature.generate_image_on_runpod(prompt="sunset beach in watercolor")

    assert "result" in image_result
    assert manager.started is False
    assert llm_service.deactivate_reasons[-1] == "image generation completed"
    assert llm_service.last_backend == BackendType.CLOUD
