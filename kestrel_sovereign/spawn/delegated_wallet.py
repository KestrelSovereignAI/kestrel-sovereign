"""
DelegatedWallet — re-exported from kestrel-feature-spawn.

Install the standalone package:
    pip install kestrel-feature-spawn
"""

try:
    from kestrel_feature_spawn.spawn.delegated_wallet import (
        BudgetAllocation,
        BudgetExceededError,
        DelegatedWallet,
        create_delegated_wallet,
        release_delegated_wallet,
    )
except ImportError:
    import sys
    from pathlib import Path

    _pkg_src = Path(__file__).resolve().parents[2] / "kestrel-feature-spawn" / "src"
    if _pkg_src.exists() and str(_pkg_src) not in sys.path:
        sys.path.insert(0, str(_pkg_src))

    from kestrel_feature_spawn.spawn.delegated_wallet import (  # noqa: F811
        BudgetAllocation,
        BudgetExceededError,
        DelegatedWallet,
        create_delegated_wallet,
        release_delegated_wallet,
    )

__all__ = [
    "BudgetAllocation",
    "BudgetExceededError",
    "DelegatedWallet",
    "create_delegated_wallet",
    "release_delegated_wallet",
]
