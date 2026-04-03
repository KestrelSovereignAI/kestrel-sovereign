"""
Deploy Providers.

Core keeps the DeployProvider interface only.
Concrete implementations are in external packages:
- kestrel-cloud-gcp (CloudRunProvider)
- kestrel-cloud-azure (AzureContainerProvider)

They register via the ``kestrel_sovereign.cloud_providers`` entry_point group.
"""

from .base import DeployProvider

# Backward-compat re-exports from extracted packages
from .cloudrun import CloudRunProvider  # noqa: F401
from .azure_container import AzureContainerProvider  # noqa: F401

__all__ = [
    "DeployProvider",
    "CloudRunProvider",
    "AzureContainerProvider",
]
