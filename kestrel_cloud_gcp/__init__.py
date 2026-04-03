"""
Kestrel Cloud GCP — GCP Compute Engine + Cloud Run providers.

Extracted from kestrel-sovereign as a standalone cloud provider package.
Registers GCPComputeFeature via ``kestrel_sovereign.features`` and
CloudRunProvider via ``kestrel_sovereign.cloud_providers``.
"""

from .cloudrun import CloudRunProvider
from .compute import (
    GCPComputeEngineManager,
    GCPComputeManager,
    GCPComputeManagerError,
    GCPComputeSession,
    GPUProfile,
    InstanceStatus,
)

__all__ = [
    "CloudRunProvider",
    "GCPComputeEngineManager",
    "GCPComputeManager",
    "GCPComputeManagerError",
    "GCPComputeSession",
    "GPUProfile",
    "InstanceStatus",
]
