"""
Kestrel Storage Filebase -- Filebase S3-compatible IPFS storage provider.

Extracted from kestrel-sovereign as a standalone storage provider package.
Registers via entry_points group ``kestrel_sovereign.storage_providers``.
"""

from .filebase_provider import FilebaseProvider, create_filebase_provider

__all__ = ["FilebaseProvider", "create_filebase_provider"]
