"""
Ollama GPU Manager - Cloud Ollama Pod Lifecycle Management

Manages Ollama instances on RunPod/Vast.ai for users without local GPU hardware.
Reuses the existing RunPodManager infrastructure from training.

Key Features:
- Start/stop Ollama pods on-demand
- Persistent pods for fast resume (~10-30s)
- Network volume for model caching
- Integration with LLMService.switch_backend()
- Model pre-pulling on startup
"""

import asyncio
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import httpx

from kestrel_sovereign.kestrel_config.constants import (
    HTTP_TIMEOUT_DEFAULT,
    HTTP_TIMEOUT_QUICK,
    HTTP_TIMEOUT_MODEL_PULL,
)

logger = logging.getLogger(__name__)


@dataclass
class OllamaSession:
    """Active Ollama GPU session."""

    pod_id: str
    profile_name: str
    ollama_url: str  # http://pod_ip:11434
    model: str  # Primary model (e.g., qwen2.5:7b)
    available_models: List[str] = field(default_factory=list)
    gpu_type: Optional[str] = None
    vram_gb: Optional[int] = None
    cost_per_hr: Optional[float] = None
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    ttl_seconds: int = 3600  # Default 1 hour
    expires_at: Optional[datetime] = None

    def __post_init__(self):
        if self.expires_at is None:
            self.expires_at = self.started_at + timedelta(seconds=self.ttl_seconds)

    @property
    def remaining_seconds(self) -> int:
        """Remaining TTL in seconds."""
        if self.expires_at is None:
            return self.ttl_seconds
        delta = (self.expires_at - datetime.now(timezone.utc)).total_seconds()
        return max(0, int(delta))

    @property
    def is_expired(self) -> bool:
        """Check if session has expired."""
        return self.remaining_seconds <= 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pod_id": self.pod_id,
            "profile_name": self.profile_name,
            "ollama_url": self.ollama_url,
            "model": self.model,
            "available_models": self.available_models,
            "gpu_type": self.gpu_type,
            "vram_gb": self.vram_gb,
            "cost_per_hr": self.cost_per_hr,
            "started_at": self.started_at.isoformat(),
            "remaining_seconds": self.remaining_seconds,
            "is_expired": self.is_expired,
        }


class OllamaGPUManagerError(Exception):
    """Custom exception for Ollama manager failures."""


class OllamaGPUManager:
    """
    Manages Ollama instances on cloud GPUs.

    Uses the existing RunPodManager infrastructure. The 'ollama' profile
    in runpod_config.toml defines:
    - GPU type (RTX 3090 by default)
    - Docker image (gcr.io/YOUR_PROJECT_ID/kestrel-ollama:latest)
    - Network volume for persistent models
    - Default models to pre-pull on startup

    Usage:
        manager = OllamaGPUManager()

        # Start an Ollama session
        session = await manager.start_session()
        print(f"Ollama ready at {session.ollama_url}")

        # List available models
        models = await manager.list_models(session)

        # Pull additional models
        await manager.pull_model(session, "llama3.2:7b")

        # Stop when done
        await manager.stop_session(session)
    """

    def __init__(self, profile_name: str = "ollama"):
        """
        Initialize the Ollama GPU manager.

        Args:
            profile_name: Profile name from runpod_config.toml (default: "ollama")
        """
        self.profile_name = profile_name
        self._runpod_manager = None
        self._active_session: Optional[OllamaSession] = None

    def _get_runpod_manager(self):
        """Lazy load RunPodManager."""
        if self._runpod_manager is None:
            try:
                from kestrel_sovereign.features.runpod.manager import RunPodManager

                self._runpod_manager = RunPodManager()
            except ImportError as e:
                raise OllamaGPUManagerError(
                    f"RunPod manager not available: {e}. "
                    "Install runpod package: pip install runpod"
                )
            except Exception as e:
                raise OllamaGPUManagerError(f"Failed to initialize RunPod manager: {e}")
        return self._runpod_manager

    def is_available(self) -> bool:
        """Check if Ollama GPU is available (RunPod configured)."""
        api_key = os.environ.get("RUNPOD_API_KEY")
        if not api_key:
            return False
        try:
            self._get_runpod_manager()
            return True
        except OllamaGPUManagerError:
            return False

    @property
    def session(self) -> Optional[OllamaSession]:
        """Get the active Ollama session."""
        return self._active_session

    async def start_session(
        self,
        model: str = "qwen2.5:7b",
        ttl_seconds: int = 3600,
        models_to_pull: Optional[List[str]] = None,
    ) -> OllamaSession:
        """
        Start an Ollama GPU session.

        Args:
            model: Primary model to use (default: qwen2.5:7b)
            ttl_seconds: Session time-to-live in seconds (default: 1 hour)
            models_to_pull: Additional models to pull on startup

        Returns:
            OllamaSession with connection details

        Raises:
            OllamaGPUManagerError: If session cannot be started
        """
        if self._active_session and not self._active_session.is_expired:
            logger.info(
                f"Reusing existing Ollama session: {self._active_session.pod_id}"
            )
            return self._active_session

        manager = self._get_runpod_manager()

        try:
            # Start session using RunPod manager with ollama profile
            logger.info(f"Starting Ollama GPU session with profile '{self.profile_name}'")

            # Set models to pull via environment variable override
            env_overrides = {}
            if models_to_pull:
                # Combine with primary model
                all_models = [model] + models_to_pull
                env_overrides["OLLAMA_MODELS_PULL"] = ",".join(all_models)
            else:
                env_overrides["OLLAMA_MODELS_PULL"] = model

            runpod_session = await manager.start_session(
                task_profile=self.profile_name,
                ttl_seconds=ttl_seconds,
                metadata={"env_overrides": env_overrides},
            )

            # Build Ollama URL from RunPod session
            # The ollama profile configures port 11434
            ollama_url = runpod_session.inference_url or f"http://{runpod_session.backend_base_url}"

            # Wait for Ollama to be ready
            logger.info("Waiting for Ollama to be ready...")
            await self._wait_for_ready(ollama_url, timeout_seconds=300)

            # Get list of available models
            available_models = await self._list_models_internal(ollama_url)

            # Create session
            profile = runpod_session.profile
            self._active_session = OllamaSession(
                pod_id=runpod_session.pod_id,
                profile_name=self.profile_name,
                ollama_url=ollama_url,
                model=model,
                available_models=available_models,
                gpu_type=profile.gpu_type_id if hasattr(profile, "gpu_type_id") else None,
                vram_gb=profile.vram_gb if hasattr(profile, "vram_gb") else None,
                cost_per_hr=profile.cost_per_hr if hasattr(profile, "cost_per_hr") else None,
                ttl_seconds=ttl_seconds,
            )

            logger.info(
                f"Ollama session ready at {ollama_url} with models: {available_models}"
            )
            return self._active_session

        except Exception as e:
            logger.error(f"Failed to start Ollama session: {e}")
            raise OllamaGPUManagerError(f"Failed to start Ollama session: {e}")

    async def stop_session(self, session: Optional[OllamaSession] = None) -> None:
        """
        Stop an Ollama GPU session.

        Args:
            session: Session to stop (uses active session if not provided)
        """
        session = session or self._active_session
        if not session:
            logger.warning("No active Ollama session to stop")
            return

        manager = self._get_runpod_manager()

        try:
            # Check if this is a persistent pod (stop vs terminate)
            profile = manager._profiles.get(self.profile_name)
            is_persistent = profile and profile.persistent_pod_id

            if is_persistent:
                logger.info(f"Stopping persistent Ollama pod {session.pod_id}")
                await manager.stop_session()  # Just stops, doesn't terminate
            else:
                logger.info(f"Terminating Ollama pod {session.pod_id}")
                await manager.stop_session()

            if session == self._active_session:
                self._active_session = None

        except Exception as e:
            logger.error(f"Failed to stop Ollama session: {e}")
            raise OllamaGPUManagerError(f"Failed to stop session: {e}")

    async def list_models(
        self, session: Optional[OllamaSession] = None
    ) -> List[str]:
        """
        List available models on the Ollama instance.

        Args:
            session: Session to query (uses active session if not provided)

        Returns:
            List of model names
        """
        session = session or self._active_session
        if not session:
            raise OllamaGPUManagerError("No active Ollama session")

        return await self._list_models_internal(session.ollama_url)

    async def pull_model(
        self,
        model: str,
        session: Optional[OllamaSession] = None,
    ) -> bool:
        """
        Pull a model to the Ollama instance.

        Args:
            model: Model to pull (e.g., "llama3.2:7b")
            session: Session to use (uses active session if not provided)

        Returns:
            True if model was pulled successfully
        """
        session = session or self._active_session
        if not session:
            raise OllamaGPUManagerError("No active Ollama session")

        try:
            async with httpx.AsyncClient(timeout=HTTP_TIMEOUT_MODEL_PULL) as client:
                response = await client.post(
                    f"{session.ollama_url}/api/pull",
                    json={"name": model, "stream": False},
                )
                if response.status_code == 200:
                    logger.info(f"Successfully pulled model: {model}")
                    # Update available models
                    session.available_models = await self._list_models_internal(
                        session.ollama_url
                    )
                    return True
                else:
                    logger.error(f"Failed to pull model {model}: {response.text}")
                    return False
        except Exception as e:
            logger.error(f"Error pulling model {model}: {e}")
            return False

    async def get_llm_config(
        self, session: Optional[OllamaSession] = None
    ) -> Dict[str, Any]:
        """
        Get configuration for LLMService.switch_backend().

        Args:
            session: Session to use (uses active session if not provided)

        Returns:
            Dict suitable for LLMService.switch_backend("remote_gpu", config)
        """
        session = session or self._active_session
        if not session:
            raise OllamaGPUManagerError("No active Ollama session")

        return {
            "base_url": f"{session.ollama_url}/v1",  # OpenAI-compatible endpoint
            "model": session.model,
            "api_key": "ollama",  # Ollama doesn't require real key
            "timeout_seconds": 120,
        }

    # ==================== Private Methods ====================

    async def _wait_for_ready(
        self, ollama_url: str, timeout_seconds: int = 300
    ) -> None:
        """Wait for Ollama to be ready to accept requests."""
        start_time = datetime.now(timezone.utc)
        check_interval = 5

        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT_QUICK) as client:
            while True:
                elapsed = (datetime.now(timezone.utc) - start_time).total_seconds()
                if elapsed > timeout_seconds:
                    raise OllamaGPUManagerError(
                        f"Ollama not ready after {timeout_seconds}s"
                    )

                try:
                    response = await client.get(f"{ollama_url}/api/tags")
                    if response.status_code == 200:
                        logger.info("Ollama is ready")
                        return
                except httpx.RequestError as e:
                    logger.debug(f"Ollama not ready yet: {e}")

                await asyncio.sleep(check_interval)

    async def _list_models_internal(self, ollama_url: str) -> List[str]:
        """Internal method to list models from Ollama API."""
        try:
            async with httpx.AsyncClient(timeout=HTTP_TIMEOUT_DEFAULT) as client:
                response = await client.get(f"{ollama_url}/api/tags")
                if response.status_code == 200:
                    data = response.json()
                    models = data.get("models", [])
                    return [m.get("name", m.get("model", "")) for m in models]
                return []
        except Exception as e:
            logger.error(f"Failed to list Ollama models: {e}")
            return []
