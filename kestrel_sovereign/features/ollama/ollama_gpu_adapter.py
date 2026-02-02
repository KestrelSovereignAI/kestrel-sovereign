"""
Ollama GPU Adapter - Remote Ollama Integration for LLMService

Provides a high-level interface for integrating remote Ollama (on RunPod/Vast.ai)
with the LLMService's backend switching mechanism.

This adapter handles:
- Automatic session management (start/stop)
- LLMService backend configuration
- Model selection and fallback
- Cost tracking integration

Usage:
    from kestrel_sovereign.features.ollama import OllamaGPUAdapter

    adapter = OllamaGPUAdapter()

    # Start remote Ollama and configure LLM service
    await adapter.activate(llm_service, model="qwen2.5:7b")

    # LLM requests now route to remote Ollama
    response = await llm_service.generate("Hello!")

    # When done
    await adapter.deactivate(llm_service)
"""

import logging
import os
from typing import Any, Dict, List, Optional

from .ollama_manager import OllamaGPUManager, OllamaSession, OllamaGPUManagerError

logger = logging.getLogger(__name__)


class OllamaGPUAdapter:
    """
    High-level adapter for remote Ollama integration with LLMService.

    Manages the lifecycle of remote Ollama sessions and configures
    the LLMService to route requests through the remote backend.
    """

    def __init__(self, profile_name: str = "ollama"):
        """
        Initialize the adapter.

        Args:
            profile_name: RunPod profile name (default: "ollama")
        """
        self._manager = OllamaGPUManager(profile_name=profile_name)
        self._is_active = False

    @property
    def is_available(self) -> bool:
        """Check if remote Ollama is available."""
        return self._manager.is_available()

    @property
    def is_active(self) -> bool:
        """Check if remote Ollama is currently active."""
        return self._is_active and self._manager.session is not None

    @property
    def session(self) -> Optional[OllamaSession]:
        """Get the current Ollama session."""
        return self._manager.session

    async def activate(
        self,
        llm_service: Any,
        model: str = "qwen2.5:7b",
        ttl_seconds: int = 3600,
        models_to_pull: Optional[List[str]] = None,
    ) -> OllamaSession:
        """
        Activate remote Ollama and configure LLMService.

        Args:
            llm_service: LLMService instance to configure
            model: Primary model to use
            ttl_seconds: Session time-to-live (default: 1 hour)
            models_to_pull: Additional models to pre-pull

        Returns:
            OllamaSession with connection details

        Raises:
            OllamaGPUManagerError: If activation fails
        """
        if not self.is_available:
            raise OllamaGPUManagerError(
                "Remote Ollama not available. Set RUNPOD_API_KEY environment variable."
            )

        try:
            # Start Ollama session
            session = await self._manager.start_session(
                model=model,
                ttl_seconds=ttl_seconds,
                models_to_pull=models_to_pull,
            )

            # Get LLM config and switch backend
            config = await self._manager.get_llm_config(session)

            # Import BackendType for proper type
            from kestrel_sovereign.llm.service import BackendType

            llm_service.switch_backend(BackendType.REMOTE_GPU, config)
            self._is_active = True

            logger.info(
                f"Remote Ollama activated: {session.ollama_url} "
                f"(model: {model}, TTL: {ttl_seconds}s)"
            )
            return session

        except Exception as e:
            logger.error(f"Failed to activate remote Ollama: {e}")
            raise OllamaGPUManagerError(f"Activation failed: {e}")

    async def deactivate(self, llm_service: Any, stop_pod: bool = False) -> None:
        """
        Deactivate remote Ollama and restore LLMService to default backend.

        Args:
            llm_service: LLMService instance to restore
            stop_pod: Whether to stop the pod (default: False, keeps running)
        """
        if not self._is_active:
            logger.warning("Remote Ollama is not active")
            return

        try:
            # Switch back to default backend
            from kestrel_sovereign.llm.service import BackendType

            llm_service.switch_backend(BackendType.CLOUD)
            self._is_active = False

            # Optionally stop the pod
            if stop_pod:
                await self._manager.stop_session()
                logger.info("Remote Ollama pod stopped")
            else:
                logger.info(
                    "Remote Ollama deactivated (pod still running for fast resume)"
                )

        except Exception as e:
            logger.error(f"Error deactivating remote Ollama: {e}")
            self._is_active = False

    async def list_models(self) -> List[str]:
        """List available models on the remote Ollama instance."""
        if not self.is_active:
            raise OllamaGPUManagerError("Remote Ollama is not active")
        return await self._manager.list_models()

    async def pull_model(self, model: str) -> bool:
        """Pull a model to the remote Ollama instance."""
        if not self.is_active:
            raise OllamaGPUManagerError("Remote Ollama is not active")
        return await self._manager.pull_model(model)

    def get_status(self) -> Dict[str, Any]:
        """
        Get current status of the remote Ollama.

        Returns:
            Dict with status information
        """
        session = self._manager.session
        if session:
            return {
                "active": self._is_active,
                "pod_id": session.pod_id,
                "ollama_url": session.ollama_url,
                "model": session.model,
                "available_models": session.available_models,
                "gpu_type": session.gpu_type,
                "cost_per_hr": session.cost_per_hr,
                "remaining_seconds": session.remaining_seconds,
            }
        return {
            "active": False,
            "available": self.is_available,
        }


# Convenience function for agent commands
async def start_remote_ollama(
    llm_service: Any,
    model: str = "qwen2.5:7b",
    ttl_minutes: int = 60,
) -> Dict[str, Any]:
    """
    Start remote Ollama and configure LLMService.

    This is a convenience function for agent commands like !gpu on.

    Args:
        llm_service: LLMService instance
        model: Model to use (default: qwen2.5:7b)
        ttl_minutes: Time-to-live in minutes (default: 60)

    Returns:
        Dict with session info
    """
    adapter = OllamaGPUAdapter()
    session = await adapter.activate(
        llm_service,
        model=model,
        ttl_seconds=ttl_minutes * 60,
    )
    return {
        "success": True,
        "pod_id": session.pod_id,
        "ollama_url": session.ollama_url,
        "model": session.model,
        "available_models": session.available_models,
        "cost_per_hr": session.cost_per_hr,
        "ttl_minutes": ttl_minutes,
    }


async def stop_remote_ollama(llm_service: Any, terminate: bool = False) -> Dict[str, Any]:
    """
    Stop remote Ollama.

    Args:
        llm_service: LLMService instance
        terminate: Whether to terminate the pod (vs just stop routing)

    Returns:
        Dict with result
    """
    adapter = OllamaGPUAdapter()
    await adapter.deactivate(llm_service, stop_pod=terminate)
    return {
        "success": True,
        "terminated": terminate,
    }
