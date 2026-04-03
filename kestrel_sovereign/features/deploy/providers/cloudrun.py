"""
Cloud Run Deployment Provider.

.. deprecated::
    This module re-exports from ``kestrel_cloud_gcp``.
    Import directly from ``kestrel_cloud_gcp`` instead.
"""

from kestrel_cloud_gcp.cloudrun import CloudRunProvider  # noqa: F401

__all__ = ["CloudRunProvider"]
