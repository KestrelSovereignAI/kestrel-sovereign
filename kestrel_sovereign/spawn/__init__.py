"""
Spawn subsystem for Kestrel agent delegation.

This module re-exports spawn primitives from the extracted
kestrel-feature-spawn package. If the package is not installed,
falls back to the bundled copy under kestrel-feature-spawn/.

Install the standalone package:
    pip install kestrel-feature-spawn
"""

try:
    from kestrel_feature_spawn.spawn import (
        SpawnMandate,
        sign_mandate,
        verify_mandate,
        create_child_did_document,
        BudgetAllocation,
        BudgetExceededError,
        DelegatedWallet,
        create_delegated_wallet,
        release_delegated_wallet,
        ScopedConstitution,
        SpawnedAgentLifecycle,
        SpawnResult,
        SpawnStatus,
        SpawnMode,
    )
except ImportError:
    import sys
    from pathlib import Path

    _pkg_src = Path(__file__).resolve().parents[2] / "kestrel-feature-spawn" / "src"
    if _pkg_src.exists() and str(_pkg_src) not in sys.path:
        sys.path.insert(0, str(_pkg_src))

    from kestrel_feature_spawn.spawn import (  # noqa: F811
        SpawnMandate,
        sign_mandate,
        verify_mandate,
        create_child_did_document,
        BudgetAllocation,
        BudgetExceededError,
        DelegatedWallet,
        create_delegated_wallet,
        release_delegated_wallet,
        ScopedConstitution,
        SpawnedAgentLifecycle,
        SpawnResult,
        SpawnStatus,
        SpawnMode,
    )

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
    "SpawnedAgentLifecycle",
    "SpawnResult",
    "SpawnStatus",
    "SpawnMode",
]
