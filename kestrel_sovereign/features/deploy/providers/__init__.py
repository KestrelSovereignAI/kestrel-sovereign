"""
Deploy Providers.

Core keeps the DeployProvider interface only.
Concrete implementations are in external packages:
- kestrel-cloud-gcp (CloudRunProvider)
- kestrel-cloud-azure (AzureContainerProvider)

They register via the ``kestrel_sovereign.cloud_providers`` entry_point group.
"""

from .base import DeployProvider

__all__ = [
    "DeployProvider",
]


def __getattr__(name):
    """Lazy re-export from extracted cloud provider packages."""
    if name == "CloudRunProvider":
        from kestrel_cloud_gcp.cloudrun import CloudRunProvider
        return CloudRunProvider
    if name == "AzureContainerProvider":
        from kestrel_cloud_azure.azure_container import AzureContainerProvider
        return AzureContainerProvider
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
