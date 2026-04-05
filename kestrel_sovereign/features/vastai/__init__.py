"""
Vast.ai GPU Compute Feature for Kestrel Agents.

Modular structure for the Vast.ai GPU instance manager:
- models.py: Data models, enums, exceptions
- core.py: Core SDK operations and session management
- ssh_training.py: SSH-based training methods for Kohya instances
- http_api.py: HTTP API methods for SimpleTuner containers
- workflows.py: Convenience workflow methods
- manager.py: Combined VastAIManager class
- feature.py: Kestrel feature integration

Usage:
    from kestrel_sovereign.features.vastai import VastAIManager, VastAIFeature

    # Direct manager usage
    manager = VastAIManager()
    session = await manager.start_training_instance("companion-123")

    # Or as a Kestrel feature
    feature = VastAIFeature(agent)
"""

from .feature import VastAIFeature
from .manager import VastAIManager
from .models import (
    GPUProfile,
    InstanceStatus,
    VastAIManagerError,
    VastAISession,
)

__all__ = [
    "VastAIFeature",
    "VastAIManager",
    "VastAIManagerError",
    "VastAISession",
    "InstanceStatus",
    "GPUProfile",
]
