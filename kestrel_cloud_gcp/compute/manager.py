"""
GCP Compute Engine Manager - Combined Class.

Combines all GCP Compute functionality from the mixin classes
into a single manager class.
"""

from typing import Any, Dict, Optional

from .core import GCPComputeEngineManagerCore
from .ssh_training import GCPSSHTrainingMixin
from .workflows import GCPWorkflowsMixin


class GCPComputeEngineManager(
    GCPComputeEngineManagerCore,
    GCPSSHTrainingMixin,
    GCPWorkflowsMixin,
):
    """
    Full GCP Compute Engine GPU instance manager.

    Combines:
    - GCPComputeEngineManagerCore: SDK operations, authentication, session management
    - GCPSSHTrainingMixin: SSH-based LoRA training methods
    - GCPWorkflowsMixin: Convenience workflow methods

    Usage:
        manager = GCPComputeEngineManager()
        session = await manager.start_training_instance("companion-123")
        job_id = await manager.submit_training_job(session, image_url, "companion-123")
        status = await manager.poll_training_status(session, job_id)
        lora_data = await manager.download_lora(session, job_id)
        await manager.terminate_session(session)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the GCP Compute Engine manager."""
        super().__init__(config)


# Backward-compatible alias
GCPComputeManager = GCPComputeEngineManager
