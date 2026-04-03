"""Shim: re-exports from kestrel_feature_wallet.chain_adapters.evm_adapter."""
from kestrel_feature_wallet.chain_adapters.evm_adapter import *  # noqa: F401,F403
try:
    from kestrel_feature_wallet.chain_adapters.evm_adapter import EVMAdapter, MAINNET_CHAIN_IDS  # noqa: F401
except ImportError:
    pass
