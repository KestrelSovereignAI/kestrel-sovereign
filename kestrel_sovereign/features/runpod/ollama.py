"""
RunPod Ollama Cloud Server Methods.

Contains methods for running Ollama LLM servers on RunPod
for users without local GPU access.
"""

import logging
from typing import List, Optional

from kestrel_sovereign.kestrel_config.constants import (
    HTTP_TIMEOUT_QUICK,
    HTTP_TIMEOUT_DOWNLOAD,
)
from .models import RunPodSession

logger = logging.getLogger(__name__)


class RunPodOllamaMixin:
    """
    Mixin for Ollama cloud server operations on RunPod.

    Requires RunPodManagerCore as base class.
    """

    async def start_ollama_pod(self, models_to_pull: Optional[List[str]] = None) -> Optional[RunPodSession]:
        """
        Start an Ollama server pod on RunPod for users without local GPU.

        Uses existing switch_backend() mechanism in LLMService to route requests.
        Tries to resume a stopped pod first (~10-30s) before creating new (~2-5min).

        Args:
            models_to_pull: Optional list of models to pre-pull on startup.
                          Overrides OLLAMA_MODELS_PULL in profile env.

        Returns:
            RunPodSession with backend_base_url like http://pod-ip:11434
        """
        try:
            profile = self._select_profile("ollama")
            ttl_seconds = 3600  # 1 hour default for chat sessions

            # Build env overrides for model pre-pulling
            env_overrides = {}
            if models_to_pull:
                env_overrides["OLLAMA_MODELS_PULL"] = ",".join(models_to_pull)

            # Try to resume a stopped Ollama pod first (much faster)
            stopped_pod = await self.find_stopped_pod("ollama_server", "ollama")
            if stopped_pod:
                logger.info("Resuming stopped Ollama pod (10-30s vs 2-5min for new)")
                return await self.resume_stopped_pod(stopped_pod, profile, ttl_seconds)

            # No stopped pod found, create new one
            result = await self.start_session(
                task_profile="ollama",
                model_name=profile.default_model or "qwen2.5:7b",
                ttl_seconds=ttl_seconds,
                metadata={
                    "name": "kestrel-ollama",
                    "purpose": "ollama_server",
                    "env_overrides": env_overrides
                }
            )

            async with self._lock:
                return self._session

        except Exception as e:
            logger.error(f"Failed to start Ollama pod: {e}")
            return None

    async def get_ollama_base_url(self) -> Optional[str]:
        """
        Get the base URL for the Ollama API on the running pod.

        Returns URL like http://pod-ip:11434 for use with OllamaAdapter.
        """
        async with self._lock:
            session = self._session

        if not session or session.task_profile != "ollama":
            return None

        if not session.is_active:
            return None

        return session.backend_base_url

    async def check_ollama_health(self, timeout: float = HTTP_TIMEOUT_QUICK) -> bool:
        """
        Check if the Ollama pod is healthy and responding.

        Returns True if Ollama API is responding, False otherwise.
        """
        import httpx

        base_url = await self.get_ollama_base_url()
        if not base_url:
            return False

        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.get(f"{base_url}/api/tags")
                return response.status_code == 200
        except Exception:
            return False

    async def list_ollama_models(self) -> List[str]:
        """
        List available models on the running Ollama pod.

        Returns list of model names.
        """
        import httpx

        base_url = await self.get_ollama_base_url()
        if not base_url:
            return []

        try:
            async with httpx.AsyncClient(timeout=HTTP_TIMEOUT_QUICK) as client:
                response = await client.get(f"{base_url}/api/tags")
                response.raise_for_status()
                data = response.json()
                return [m["name"] for m in data.get("models", [])]
        except Exception as e:
            logger.error(f"Failed to list Ollama models: {e}")
            return []

    async def pull_ollama_model(self, model_name: str) -> bool:
        """
        Pull a model to the running Ollama pod.

        Args:
            model_name: Model to pull (e.g., "qwen2.5:7b")

        Returns True if pull started successfully.
        """
        import httpx

        base_url = await self.get_ollama_base_url()
        if not base_url:
            return False

        try:
            async with httpx.AsyncClient(timeout=HTTP_TIMEOUT_DOWNLOAD) as client:  # Long timeout for large models
                response = await client.post(
                    f"{base_url}/api/pull",
                    json={"name": model_name, "stream": False}
                )
                return response.status_code == 200
        except Exception as e:
            logger.error(f"Failed to pull Ollama model {model_name}: {e}")
            return False
