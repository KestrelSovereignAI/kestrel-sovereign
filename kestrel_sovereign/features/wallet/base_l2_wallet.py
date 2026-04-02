"""
Base L2 Wallet — convenience wrapper for USDC payments on Base.

Wraps EVMAdapter + ERC20Adapter to provide a simple API for x402
Lighthouse payments. Uses the agent's existing secp256k1 private key
(same key that generates the DID), so one identity works across chains.

Usage:
    wallet = BaseL2Wallet(private_key_hex="0xabc...", testnet=True)
    balance = await wallet.get_usdc_balance()
    tx_hash = await wallet.transfer_usdc(to="0x...", amount=Decimal("1.50"))
"""

import logging
from decimal import Decimal
from typing import Optional

try:
    from eth_account import Account
except ImportError:
    Account = None  # type: ignore[assignment,misc]

from .chain_adapters.base import ChainNetwork, TransactionResult
from .chain_adapters.evm_adapter import EVMAdapter
from .chain_adapters.erc20 import ERC20Adapter
from .chain_adapters.token_registry import TokenRegistry

logger = logging.getLogger(__name__)


class BaseL2Wallet:
    """
    Agent wallet for Base L2 (EVM) transactions.

    Uses the same secp256k1 private key as the agent's DID,
    enabling multi-chain identity from a single key.
    """

    def __init__(self, private_key_hex: str, testnet: bool = True):
        """
        Initialize Base L2 wallet.

        Args:
            private_key_hex: Hex-encoded secp256k1 private key (with or without 0x prefix)
            testnet: If True, use Base Sepolia; if False, use Base Mainnet
        """
        # Normalize key
        if private_key_hex.startswith("0x"):
            private_key_hex = private_key_hex[2:]
        self._private_key = bytes.fromhex(private_key_hex)
        self._account = Account.from_key(self._private_key)

        self.testnet = testnet
        self.network = ChainNetwork.BASE_SEPOLIA if testnet else ChainNetwork.BASE_MAINNET

        self._evm = EVMAdapter(self.network)
        self._erc20 = ERC20Adapter(self._evm)

    @property
    def address(self) -> str:
        """EVM address derived from the agent's secp256k1 key."""
        return self._account.address

    @property
    def usdc_address(self) -> Optional[str]:
        """USDC contract address for the current network."""
        token = TokenRegistry.get_token("USDC", self.network)
        return token.address if token else None

    async def get_eth_balance(self) -> Decimal:
        """Get native ETH balance (needed for gas)."""
        return await self._evm.get_balance(self.address)

    async def get_usdc_balance(self) -> Decimal:
        """Get USDC balance on Base."""
        balance = await self._erc20.get_token_balance_by_symbol("USDC", self.address)
        if balance is None:
            logger.warning(f"USDC not registered for {self.network.display_name}")
            return Decimal("0")
        return balance

    async def transfer_usdc(
        self,
        to: str,
        amount: Decimal,
        gas_limit: Optional[int] = None,
    ) -> TransactionResult:
        """
        Transfer USDC on Base.

        Args:
            to: Recipient address (0x...)
            amount: Amount in USDC (e.g., Decimal("1.50"))
            gas_limit: Optional gas limit override

        Returns:
            TransactionResult with tx_hash or error
        """
        return await self._erc20.transfer_by_symbol(
            symbol="USDC",
            to_address=to,
            amount=amount,
            private_key=self._private_key,
            gas_limit=gas_limit,
        )

    async def wait_for_confirmation(
        self, tx_hash: str, timeout: int = 60
    ) -> TransactionResult:
        """Wait for a transaction to be confirmed."""
        return await self._evm.wait_for_transaction(tx_hash, timeout=timeout)

    async def is_connected(self) -> bool:
        """Check if connected to Base RPC."""
        return await self._evm.is_connected()

    async def close(self) -> None:
        """Clean up connections."""
        await self._evm.close()

    def __repr__(self) -> str:
        return (
            f"BaseL2Wallet(address={self.address}, "
            f"network={self.network.display_name})"
        )
