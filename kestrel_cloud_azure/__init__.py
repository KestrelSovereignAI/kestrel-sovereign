"""
Kestrel Cloud Azure — Azure Container Apps deployment provider.

Extracted from kestrel-sovereign as a standalone cloud provider package.
Registers via entry_points group ``kestrel_sovereign.cloud_providers``.
"""

from .azure_container import AzureContainerProvider

__all__ = ["AzureContainerProvider"]
