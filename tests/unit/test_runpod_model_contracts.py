"""Contracts for RunPod profile-owned model defaults."""

from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from kestrel_sovereign.features.runpod.core import RunPodManagerCore
from kestrel_sovereign.features.runpod.models import PodStatus, RunPodManagerError
from kestrel_sovereign.features.runpod.ollama import RunPodOllamaMixin


class _ResumeHarness:
    resume_stopped_pod = RunPodManagerCore.resume_stopped_pod

    def __init__(self, profile):
        self.provider = SimpleNamespace(resume_pod=lambda pod_id, gpu_count: None)
        self._lock = None
        self._session = None
        self.wait_called = False
        self.profile = profile

    async def _wait_until_ready(self):
        self.wait_called = True


class _AsyncNullLock:
    async def __aenter__(self):
        return None

    async def __aexit__(self, exc_type, exc, tb):
        return False


class _OllamaHarness(RunPodOllamaMixin):
    def __init__(self, profile, stopped_pod=None):
        self.profile = profile
        self.stopped_pod = stopped_pod
        self.start_calls = []
        self.resume_calls = []
        self._lock = _AsyncNullLock()
        self._session = SimpleNamespace(task_profile="ollama", is_active=True, backend_base_url="http://gpu:11434")

    def _select_profile(self, profile_name):
        assert profile_name == "ollama"
        return self.profile

    async def find_stopped_pod(self, *args):
        return self.stopped_pod

    async def resume_stopped_pod(self, pod, profile, ttl_seconds):
        self.resume_calls.append((pod, profile, ttl_seconds))
        return "resumed"

    async def start_session(self, **kwargs):
        self.start_calls.append(kwargs)
        return {"status": "ready"}


@pytest.mark.asyncio
async def test_resume_stopped_pod_requires_profile_default_model():
    profile = SimpleNamespace(
        id="training",
        task_type="training",
        default_model=None,
        pod_type="a100",
    )
    harness = _ResumeHarness(profile)
    harness._lock = _AsyncNullLock()

    with pytest.raises(RunPodManagerError, match="has no default_model configured"):
        await harness.resume_stopped_pod({"id": "pod-123", "gpuCount": 1}, profile, 3600)


@pytest.mark.asyncio
async def test_start_ollama_pod_uses_profile_default_model_without_hidden_fallback():
    profile = SimpleNamespace(default_model="phi4")
    harness = _OllamaHarness(profile)

    await harness.start_ollama_pod(models_to_pull=["phi4"])

    assert harness.start_calls[0]["model_name"] == "phi4"


@pytest.mark.asyncio
async def test_start_ollama_pod_resumes_existing_pod_without_new_model_override():
    profile = SimpleNamespace(default_model="phi4")
    harness = _OllamaHarness(profile, stopped_pod={"id": "pod-123"})

    result = await harness.start_ollama_pod()

    assert result == "resumed"
    assert harness.start_calls == []
    assert harness.resume_calls
