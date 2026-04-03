"""
Azure Container Apps Deployment Provider.

.. deprecated::
    This module re-exports from ``kestrel_cloud_azure``.
    Import directly from ``kestrel_cloud_azure`` instead.
"""

from kestrel_cloud_azure.azure_container import AzureContainerProvider  # noqa: F401

__all__ = ["AzureContainerProvider"]
