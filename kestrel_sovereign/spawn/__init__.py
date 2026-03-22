"""
Spawn subsystem for Kestrel agent delegation.

Provides SpawnMandate for parent-child DID delegation chains,
DelegatedWallet for budget ceiling enforcement,
and ScopedConstitution for constitutional narrowing.
"""

from .mandate import (
    SpawnMandate,
    sign_mandate,
    verify_mandate,
    create_child_did_document,
)
from .delegated_wallet import (
    BudgetAllocation,
    BudgetExceededError,
    DelegatedWallet,
    create_delegated_wallet,
    release_delegated_wallet,
)
from .scoped_constitution import ScopedConstitution

__all__ = [
    "SpawnMandate",
    "sign_mandate",
    "verify_mandate",
    "create_child_did_document",
    "BudgetAllocation",
    "BudgetExceededError",
    "DelegatedWallet",
    "create_delegated_wallet",
    "release_delegated_wallet",
    "ScopedConstitution",
]
