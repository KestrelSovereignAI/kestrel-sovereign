"""
GCP Compute Engine GPU management for Kestrel.

.. deprecated::
    This module re-exports from ``kestrel_cloud_gcp.compute``.
    Import directly from ``kestrel_cloud_gcp`` instead.
"""

# Backward-compat re-exports from extracted package
from kestrel_cloud_gcp import (  # noqa: F401
    GCPComputeEngineManager,
    GCPComputeManager,
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
