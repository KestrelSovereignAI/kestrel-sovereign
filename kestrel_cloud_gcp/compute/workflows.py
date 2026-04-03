"""
GCP Compute Workflow Methods.

Contains convenience workflow methods for common operations
like starting training or inference instances.
"""

import logging
from typing import Optional

from .models import GCPComputeSession

logger = logging.getLogger(__name__)


class GCPWorkflowsMixin:
    """
    Mixin for convenience workflow methods.

    Requires GCPComputeEngineManagerCore as base class.
    """

    async def start_training_instance(
        self,
        companion_id: str,
        ttl_seconds: int = 3600,
    ) -> Optional[GCPComputeSession]:
        """
        Start an instance for LoRA training.

        Args:
            companion_id: Companion ID for labeling
            ttl_seconds: Session TTL

        Returns:
            GCPComputeSession or None if failed
        """
        try:
            result = await self.start_session(
                task_profile="training",
                ttl_seconds=ttl_seconds,
                metadata={"companion_id": companion_id, "purpose": "lora_training"},
            )

            return self._session

        except Exception as e:
            logger.error(f"Failed to start training instance: {e}")
            return None

    async def start_inference_instance(
        self,
        model_name: Optional[str] = None,
        ttl_seconds: int = 3600,
    ) -> Optional[GCPComputeSession]:
        """
        Start an instance for image inference.

        Args:
            model_name: Model to load (defaults to profile default)
            ttl_seconds: Session TTL

        Returns:
            GCPComputeSession or None if failed
        """
        try:
            result = await self.start_session(
                task_profile="inference",
                model_name=model_name,
                ttl_seconds=ttl_seconds,
                metadata={"purpose": "image_inference"},
            )

            return self._session

        except Exception as e:
            logger.error(f"Failed to start inference instance: {e}")
            return None

    async def start_compute_instance(
        self,
        task_profile: str = "compute",
        ttl_seconds: int = 3600,
        env_overrides: Optional[dict] = None,
    ) -> Optional[GCPComputeSession]:
        """
        Start a general-purpose GPU compute instance.

        Args:
            task_profile: Profile name (defaults to 'compute')
            ttl_seconds: Session TTL
            env_overrides: Environment variable overrides

        Returns:
            GCPComputeSession or None if failed
        """
        try:
            metadata = {"purpose": "gpu_compute"}
            if env_overrides:
                metadata["env_overrides"] = env_overrides

            result = await self.start_session(
                task_profile=task_profile,
                ttl_seconds=ttl_seconds,
                metadata=metadata,
            )

            return self._session

        except Exception as e:
            logger.error(f"Failed to start compute instance: {e}")
            return None
