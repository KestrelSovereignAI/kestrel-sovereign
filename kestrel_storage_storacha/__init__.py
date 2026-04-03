"""
Kestrel Storage Storacha -- Storacha (web3.storage) storage provider.

Extracted from kestrel-sovereign as a standalone storage provider package.
Registers via entry_points group ``kestrel_sovereign.storage_providers``.
"""

from .storacha_provider import StorachaProvider, create_storacha_provider

__all__ = ["StorachaProvider", "create_storacha_provider"]
