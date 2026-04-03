"""
Kestrel Storage Lighthouse -- Lighthouse IPFS pinning storage provider.

Extracted from kestrel-sovereign as a standalone storage provider package.
Registers via entry_points group ``kestrel_sovereign.storage_providers``.
"""

from .lighthouse_provider import LighthouseProvider, create_lighthouse_provider

__all__ = ["LighthouseProvider", "create_lighthouse_provider"]
