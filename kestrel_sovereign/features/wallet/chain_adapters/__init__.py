"""Shim: re-exports from kestrel_feature_wallet.chain_adapters."""
from kestrel_feature_wallet.chain_adapters import *  # noqa: F401,F403
from kestrel_feature_wallet.chain_adapters import (  # noqa: F401
    ChainNetwork,
    ChainAdapter,
    TransactionRequest,
    TransactionResult,
    GasEstimate,
    NetworkConfig,
    TokenRegistry,
    TokenInfo,
    EVM_AVAILABLE,
)

if EVM_AVAILABLE:
    from kestrel_feature_wallet.chain_adapters import EVMAdapter, ERC20Adapter  # noqa: F401
