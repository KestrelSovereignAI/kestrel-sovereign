"""
Spawn subsystem for Kestrel agent delegation.

Provides SpawnMandate for parent-child DID delegation chains.
"""

from .mandate import (
    SpawnMandate,
    sign_mandate,
    verify_mandate,
    create_child_did_document,
)

__all__ = [
    "SpawnMandate",
    "sign_mandate",
    "verify_mandate",
    "create_child_did_document",
]
