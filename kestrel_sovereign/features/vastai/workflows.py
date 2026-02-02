"""
Vast.ai Convenience Workflow Methods.

Higher-level methods for common Vast.ai workflows like
starting training, inference, or Ollama instances.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional

from .models import VastAIManagerError, VastAISession

logger = logging.getLogger(__name__)


class VastAIWorkflowsMixin:
    """Convenience methods for common Vast.ai workflows."""

    _lock: asyncio.Lock
    _session: Optional[VastAISession]

    async def start_session(
        self,
        task_profile: str,
        model_name: Optional[str] = None,
        ttl_seconds: Optional[int] = None,
        offer_id: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Start session - implemented by VastAIManagerCore."""
        raise NotImplementedError("Must be mixed with VastAIManagerCore")

    async def start_training_instance(
        self,
        companion_id: str,
        ttl_seconds: int = 3600,
    ) -> Optional[VastAISession]:
        """
        Start an instance for LoRA training.

        Args:
            companion_id: Companion being trained (for labeling)
            ttl_seconds: Session TTL

        Returns:
            VastAISession if started successfully
        """
        try:
            await self.start_session(
                task_profile="training",
                ttl_seconds=ttl_seconds,
                metadata={
                    "label": f"kestrel-lora-{companion_id[:8]}",
                    "companion_id": companion_id,
                    "purpose": "lora_training",
                },
            )

            async with self._lock:
                return self._session

        except VastAIManagerError as e:
            logger.error(f"Failed to start training instance: {e}")
            return None

    async def start_inference_instance(
        self,
        companion_id: str,
        ttl_seconds: int = 600,
    ) -> Optional[VastAISession]:
        """
        Start an instance for inference.

        Args:
            companion_id: Companion for tracking
            ttl_seconds: Session TTL

        Returns:
            VastAISession if started successfully
        """
        try:
            await self.start_session(
                task_profile="inference",
                ttl_seconds=ttl_seconds,
                metadata={
                    "label": f"kestrel-infer-{companion_id[:8]}",
                    "companion_id": companion_id,
                    "purpose": "inference",
                },
            )

            async with self._lock:
                return self._session

        except VastAIManagerError as e:
            logger.error(f"Failed to start inference instance: {e}")
            return None

    async def start_ollama_instance(
        self,
        models_to_pull: Optional[List[str]] = None,
        ttl_seconds: int = 3600,
    ) -> Optional[VastAISession]:
        """
        Start an Ollama server instance on Vast.ai.

        Args:
            models_to_pull: Models to pre-pull on startup
            ttl_seconds: Session TTL

        Returns:
            VastAISession with backend_base_url for Ollama API
        """
        try:
            env_overrides = {}
            if models_to_pull:
                env_overrides["OLLAMA_MODELS_PULL"] = ",".join(models_to_pull)

            await self.start_session(
                task_profile="ollama",
                ttl_seconds=ttl_seconds,
                metadata={
                    "label": "kestrel-ollama",
                    "purpose": "ollama_server",
                    "env_overrides": env_overrides,
                },
            )

            async with self._lock:
                return self._session

        except VastAIManagerError as e:
            logger.error(f"Failed to start Ollama instance: {e}")
            return None
