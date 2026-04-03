"""
Kestrel Cloud Vast.ai — GPU marketplace cloud provider.

Extracted from kestrel-sovereign as a standalone cloud provider package.
Registers via entry_points group ``kestrel_sovereign.features``.
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
