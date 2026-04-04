"""
Vast.ai GPU Instance Manager.

Combines all Vast.ai functionality into a single manager class:
- Core SDK operations and session management
- SSH-based training methods
- HTTP API methods for SimpleTuner
- Convenience workflow methods

Usage:
    from kestrel_cloud_vastai import VastAIManager

    manager = VastAIManager()
    session = await manager.start_training_instance("companion-123")
    job_id = await manager.submit_training_job_http(session, avatar_data, "companion-123")
"""

from typing import Any, Dict, Optional

from kestrel_sdk.config.constants import VASTAI_POLL_INTERVAL_SECONDS

from .core import VastAIManagerCore
from .http_api import VastAIHTTPAPIMixin
from .ssh_training import VastAISSHTrainingMixin
from .workflows import VastAIWorkflowsMixin


class VastAIManager(
    VastAIManagerCore,
    VastAISSHTrainingMixin,
    VastAIHTTPAPIMixin,
    VastAIWorkflowsMixin,
):
    """
    Coordinates Vast.ai GPU instance lifecycles.

    This is the main entry point for Vast.ai integration. It combines:
    - Core: SDK operations, session management, profile configuration
    - SSH Training: Direct SSH-based training on Kohya instances
    - HTTP API: SimpleTuner container HTTP endpoints
    - Workflows: Convenience methods for common use cases

    Example:
        manager = VastAIManager()

        # Start a training instance
        session = await manager.start_training_instance("companion-123")

        # Submit training via HTTP API
        job_id = await manager.submit_training_job_http(
            session, avatar_data, "companion-123"
        )

        # Poll for completion
        while True:
            status = await manager.poll_training_status_http(session, job_id)
            if status["status"] == "completed":
                break
            await asyncio.sleep(VASTAI_POLL_INTERVAL_SECONDS)

        # Download trained LoRA
        lora_data = await manager.download_lora_http(session, job_id)

        # Stop the instance
        await manager.stop_session()
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Vast.ai Manager.

        Args:
            config: Optional configuration dict. If not provided, loads from
                    vastai_config.toml via load_config().
        """
        super().__init__(config)
