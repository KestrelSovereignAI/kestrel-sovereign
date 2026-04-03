"""
Kestrel Feature Wallet — Multi-currency agent wallet with Stripe on-ramp.

Extracted from kestrel-sovereign as a standalone feature package.

NOTE: Some wallet features require optional dependencies:
- web3: Required for EVM chain support (Ethereum, Polygon, Filecoin FEVM)
- stripe: Required for fiat on-ramp

If these dependencies are not installed, the affected features will be unavailable
but the core wallet functionality will still work.
"""

from .feature import WalletAgent, Currency
from .wallet_feature import WalletFeature
from .economic_gates import EconomicGateMixin
from .filecoin_tools import FilecoinToolsMixin
from .multichain_tools import MultichainToolsMixin
from .filecoin_testnet import FilecoinTestnetAdapter, FilecoinNetwork
from .filecoin_keys import FilecoinKeyManager
from .transaction_manager import TransactionManager, TransactionAudit
from .transaction_hook import TransactionSecurityHook

# Import chain adapters (handles web3 optionality internally)
from .chain_adapters import (
    ChainNetwork,
    ChainAdapter,
    TokenRegistry,
    TokenInfo,
    NetworkConfig,
    TransactionRequest,
    TransactionResult,
    GasEstimate,
    EVM_AVAILABLE,
)

# EVM adapters are optional (require web3)
if EVM_AVAILABLE:
    from .chain_adapters import EVMAdapter, ERC20Adapter
else:
    EVMAdapter = None  # type: ignore
    ERC20Adapter = None  # type: ignore

# Stripe on-ramp is optional
try:
    from .onramp import StripeOnRamp, OnRampSession, OnRampStatus, StripeWebhookHandler
    STRIPE_AVAILABLE = True
except ImportError:
    StripeOnRamp = None  # type: ignore
    OnRampSession = None  # type: ignore
    OnRampStatus = None  # type: ignore
    StripeWebhookHandler = None  # type: ignore
    STRIPE_AVAILABLE = False

__all__ = [
    # Core wallet
    "WalletAgent",
    "WalletFeature",
    "Currency",
    # Mixins
    "EconomicGateMixin",
    "FilecoinToolsMixin",
    "MultichainToolsMixin",
    # Filecoin
    "FilecoinTestnetAdapter",
    "FilecoinNetwork",
    "FilecoinKeyManager",
    # Multi-chain
    "TransactionManager",
    "TransactionAudit",
    "TransactionSecurityHook",
    "ChainNetwork",
    "ChainAdapter",
    "EVMAdapter",
    "ERC20Adapter",
    "TokenRegistry",
    "TokenInfo",
    "NetworkConfig",
    "TransactionRequest",
    "TransactionResult",
    "GasEstimate",
    # Fiat on-ramp
    "StripeOnRamp",
    "OnRampSession",
    "OnRampStatus",
    "StripeWebhookHandler",
    # Availability flags
    "EVM_AVAILABLE",
    "STRIPE_AVAILABLE",
]
