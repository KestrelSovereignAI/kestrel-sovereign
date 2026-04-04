"""
Cloud Run Deployment Provider.

.. deprecated::
    This module re-exports from ``kestrel_cloud_gcp``.
    Import directly from ``kestrel_cloud_gcp`` instead.
"""


def __getattr__(name):
    if name == "CloudRunProvider":
        from kestrel_cloud_gcp.cloudrun import CloudRunProvider
        return CloudRunProvider
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["CloudRunProvider"]
