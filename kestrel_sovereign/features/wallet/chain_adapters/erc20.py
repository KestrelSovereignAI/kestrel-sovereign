"""Shim: re-exports from kestrel_feature_wallet.chain_adapters.erc20."""
from kestrel_feature_wallet.chain_adapters.erc20 import *  # noqa: F401,F403
try:
    from kestrel_feature_wallet.chain_adapters.erc20 import ERC20Adapter  # noqa: F401
except ImportError:
    pass
