"""
Multi-chain adapter system for Kestrel wallet.

Provides unified interface for transaction signing across EVM-compatible chains:
- Filecoin FEVM (Calibration testnet / Mainnet)
- Ethereum (Sepolia testnet / Mainnet)
- Polygon (Amoy testnet / Mainnet)

NOTE: web3 is an optional dependency. If not installed, EVM adapters will be unavailable
but the rest of the wallet system will still work.
"""

from .base import (
    ChainNetwork,
    ChainAdapter,
    TransactionRequest,
    TransactionResult,
    GasEstimate,
    NetworkConfig,
)
from .token_registry import TokenRegistry, TokenInfo

# EVM adapters require web3 - make them optional
try:
    from .evm_adapter import EVMAdapter
    from .erc20 import ERC20Adapter
    EVM_AVAILABLE = True
except ImportError:
    EVMAdapter = None  # type: ignore
    ERC20Adapter = None  # type: ignore
    EVM_AVAILABLE = False

__all__ = [
    "ChainNetwork",
    "ChainAdapter",
    "TransactionRequest",
    "TransactionResult",
    "GasEstimate",
    "NetworkConfig",
    "EVMAdapter",
    "ERC20Adapter",
    "TokenRegistry",
    "TokenInfo",
    "EVM_AVAILABLE",
]
