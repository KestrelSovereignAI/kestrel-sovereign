"""
Vast.ai GPU Compute Feature for Kestrel Agents.

.. deprecated::
    This module re-exports from ``kestrel_cloud_vastai``.
    Import directly from ``kestrel_cloud_vastai`` instead.
"""

# Backward-compat re-exports from extracted package
from kestrel_cloud_vastai import (  # noqa: F401
    GPUProfile,
    InstanceStatus,
    VastAIFeature,
    VastAIManager,
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
