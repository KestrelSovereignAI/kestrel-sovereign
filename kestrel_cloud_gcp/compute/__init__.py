"""
Kestrel Cloud GCP — Compute Engine GPU management.

Extracted from kestrel-sovereign as a standalone cloud provider package.

Modular structure for the GCP Compute Engine GPU instance manager:
- models.py: Data models, enums, exceptions
- core.py: Core SDK operations, authentication, session management
- ssh_training.py: SSH-based LoRA training methods
- workflows.py: Convenience workflow methods
- manager.py: Combined GCPComputeEngineManager class

Key advantages over Vast.ai/RunPod:
- Enterprise-grade reliability
- Persistent Disk for model caching (survives instance termination)
- Spot instances for 60-90% cost savings
- Shared GCP project (YOUR_PROJECT_ID)

Usage:
    from kestrel_cloud_gcp.compute import GCPComputeManager, GCPComputeEngineManager

    # Create manager
    manager = GCPComputeManager()  # or GCPComputeEngineManager()

    # Start training instance
    session = await manager.start_training_instance("companion-123")

    # Submit and monitor training
    job_id = await manager.submit_training_job(session, image_url, "companion-123")
    status = await manager.poll_training_status(session, job_id)

    # Download results
    lora_data = await manager.download_lora(session, job_id)

    # Clean up
    await manager.terminate_session(session)
"""

from .manager import GCPComputeEngineManager, GCPComputeManager
from .models import (
    GCPComputeManagerError,
    GCPComputeSession,
    GPUProfile,
    InstanceStatus,
)

__all__ = [
    "GCPComputeEngineManager",
    "GCPComputeManager",
    "GCPComputeManagerError",
    "GCPComputeSession",
    "GPUProfile",
    "InstanceStatus",
]
