"""
EVM Chain Adapter using web3.py.

Provides unified transaction signing for all EVM-compatible chains:
- Filecoin FEVM (uses standard Ethereum JSON-RPC)
- Ethereum
- Polygon

Uses EIP-1559 transaction format where supported.
"""

import os
import logging
from decimal import Decimal
from typing import Optional

try:
    from web3 import Web3
    from web3.exceptions import TransactionNotFound
    from eth_account import Account
except ImportError:
    Web3 = None  # type: ignore[assignment,misc]
    TransactionNotFound = None  # type: ignore[assignment,misc]
    Account = None  # type: ignore[assignment,misc]

from .base import (
    ChainAdapter,
    ChainNetwork,
    NetworkConfig,
    TransactionRequest,
    TransactionResult,
    GasEstimate,
)

logger = logging.getLogger(__name__)

# Hard-coded mainnet chain IDs that are ALWAYS blocked unless explicitly enabled
# This cannot be overridden by configuration alone
MAINNET_CHAIN_IDS = {
    1,       # Ethereum Mainnet
    56,      # BSC Mainnet
    137,     # Polygon Mainnet
    42161,   # Arbitrum One
    10,      # Optimism Mainnet
    314,     # Filecoin Mainnet
    8453,    # Base Mainnet
}


class EVMAdapter(ChainAdapter):
    """
    EVM chain adapter using web3.py.

    Handles transaction signing and broadcasting for all EVM-compatible chains.
    Uses EIP-1559 (type 2) transactions where supported.
    """

    def __init__(self, network: ChainNetwork):
        """
        Initialize EVM adapter.

        Args:
            network: Target blockchain network
        """
        if Web3 is None:
            raise ImportError(
                "web3 package is required for EVMAdapter. "
                "Install it with: pip install kestrel-sovereign[wallet]"
            )
        super().__init__(network)
        self.w3 = Web3(Web3.HTTPProvider(self.config.rpc_url))
        logger.info(f"EVMAdapter initialized for {network.display_name}")

    async def is_connected(self) -> bool:
        """Check connection to the RPC endpoint."""
        try:
            return self.w3.is_connected()
        except (ConnectionError, TimeoutError, OSError) as e:
            logger.error(f"Connection check failed: {e}")
            return False
        except Exception as e:
            logger.error(f"Connection check failed: {e}", exc_info=True)
            return False

    async def get_balance(self, address: str) -> Decimal:
        """
        Get native token balance.

        Args:
            address: The address to query

        Returns:
            Balance in native token units (e.g., ETH, FIL)
        """
        if not self.validate_address(address):
            raise ValueError(f"Invalid address: {address}")

        checksum_address = Web3.to_checksum_address(address)
        balance_wei = self.w3.eth.get_balance(checksum_address)
        return Decimal(balance_wei) / Decimal(10**self.config.native_decimals)

    async def estimate_gas(self, request: TransactionRequest) -> GasEstimate:
        """
        Estimate gas for a transaction.

        Args:
            request: Transaction to estimate

        Returns:
            Gas estimation with EIP-1559 pricing if available
        """
        tx_dict = self._build_tx_dict(request, include_gas=False)

        # Estimate gas limit
        try:
            gas_limit = self.w3.eth.estimate_gas(tx_dict)
            # Add 20% buffer for safety
            gas_limit = int(gas_limit * 1.2)
        except (ConnectionError, TimeoutError, ValueError) as e:
            logger.warning(f"Gas estimation failed, using default: {e}")
            gas_limit = 21000 if not request.data else 100000
        except Exception as e:
            logger.warning(f"Gas estimation failed, using default: {e}", exc_info=True)
            gas_limit = 21000 if not request.data else 100000

        # Get current gas prices
        try:
            # Try EIP-1559 pricing
            latest_block = self.w3.eth.get_block("latest")
            base_fee = latest_block.get("baseFeePerGas")

            if base_fee:
                # EIP-1559 supported
                max_priority_fee = self.w3.eth.max_priority_fee
                max_fee = base_fee * 2 + max_priority_fee

                return GasEstimate(
                    gas_limit=gas_limit,
                    gas_price_wei=base_fee + max_priority_fee,
                    max_fee_per_gas=max_fee,
                    max_priority_fee_per_gas=max_priority_fee,
                )
            else:
                # Legacy pricing
                gas_price = self.w3.eth.gas_price
                return GasEstimate(
                    gas_limit=gas_limit,
                    gas_price_wei=gas_price,
                )
        except (ConnectionError, TimeoutError, ValueError, KeyError) as e:
            logger.warning(f"Gas price fetch failed: {e}")
            # Fallback to legacy
            gas_price = self.w3.eth.gas_price
            return GasEstimate(
                gas_limit=gas_limit,
                gas_price_wei=gas_price,
            )
        except Exception as e:
            logger.warning(f"Gas price fetch failed: {e}", exc_info=True)
            # Fallback to legacy
            gas_price = self.w3.eth.gas_price
            return GasEstimate(
                gas_limit=gas_limit,
                gas_price_wei=gas_price,
            )

    async def send_transaction(
        self, request: TransactionRequest, private_key: bytes
    ) -> TransactionResult:
        """
        Sign and broadcast a transaction.

        Args:
            request: Transaction details
            private_key: Raw private key bytes

        Returns:
            Transaction result with hash or error
        """
        # CRITICAL SECURITY CHECK: Block mainnet transactions unless explicitly allowed
        if self.config.chain_id in MAINNET_CHAIN_IDS and not os.environ.get("KESTREL_ALLOW_MAINNET"):
            return TransactionResult(
                success=False,
                error=(
                    f"Mainnet transactions blocked (chain_id={self.config.chain_id}). "
                    f"Set KESTREL_ALLOW_MAINNET=1 to enable."
                ),
                network=self.network,
            )

        try:
            # Derive address from private key
            account = Account.from_key(private_key)
            from_address = account.address

            # Validate addresses
            if not self.validate_address(request.to_address):
                return TransactionResult(
                    success=False,
                    error=f"Invalid to_address: {request.to_address}",
                    network=self.network,
                )

            # Get nonce
            nonce = request.nonce
            if nonce is None:
                nonce = await self.get_nonce(from_address)

            # Estimate gas if not provided
            if request.gas_limit is None:
                gas_estimate = await self.estimate_gas(request)
                gas_limit = gas_estimate.gas_limit
                max_fee = gas_estimate.max_fee_per_gas
                max_priority_fee = gas_estimate.max_priority_fee_per_gas
                gas_price = gas_estimate.gas_price_wei
            else:
                gas_limit = request.gas_limit
                max_fee = request.max_fee_per_gas
                max_priority_fee = request.max_priority_fee_per_gas
                gas_price = request.gas_price_wei

            # Build transaction
            to_address = Web3.to_checksum_address(request.to_address)
            value_wei = int(request.amount * Decimal(10**self.config.native_decimals))

            if max_fee and max_priority_fee:
                # EIP-1559 transaction
                tx_dict = {
                    "type": 2,
                    "chainId": self.config.chain_id,
                    "nonce": nonce,
                    "to": to_address,
                    "value": value_wei,
                    "gas": gas_limit,
                    "maxFeePerGas": max_fee,
                    "maxPriorityFeePerGas": max_priority_fee,
                }
            else:
                # Legacy transaction
                tx_dict = {
                    "chainId": self.config.chain_id,
                    "nonce": nonce,
                    "to": to_address,
                    "value": value_wei,
                    "gas": gas_limit,
                    "gasPrice": gas_price or self.w3.eth.gas_price,
                }

            # Add contract data if present
            if request.data:
                tx_dict["data"] = request.data

            # Sign transaction
            signed_tx = self.w3.eth.account.sign_transaction(tx_dict, private_key)

            # Broadcast
            tx_hash = self.w3.eth.send_raw_transaction(signed_tx.raw_transaction)
            tx_hash_hex = tx_hash.hex()

            logger.info(
                f"Transaction sent: {tx_hash_hex} on {self.network.display_name}"
            )

            return TransactionResult(
                success=True,
                tx_hash=tx_hash_hex,
                network=self.network,
            )

        except (ConnectionError, TimeoutError) as e:
            logger.error(f"Transaction failed: {e}")
            return TransactionResult(
                success=False,
                error=str(e),
                network=self.network,
            )
        except (ValueError, TypeError, KeyError) as e:
            logger.error(f"Transaction failed: {e}")
            return TransactionResult(
                success=False,
                error=str(e),
                network=self.network,
            )
        except Exception as e:
            logger.error(f"Transaction failed: {e}", exc_info=True)
            return TransactionResult(
                success=False,
                error=str(e),
                network=self.network,
            )

    def validate_address(self, address: str) -> bool:
        """
        Validate an EVM address.

        Args:
            address: Address to validate

        Returns:
            True if valid EVM address
        """
        if not address:
            return False

        # Check basic format (0x + 40 hex chars)
        if not address.startswith("0x"):
            return False
        if len(address) != 42:
            return False

        try:
            # Web3 checksum validation
            Web3.to_checksum_address(address)
            return True
        except (ValueError, TypeError):
            return False
        except Exception:
            return False

    async def get_nonce(self, address: str) -> int:
        """
        Get the next nonce for an address.

        Args:
            address: The address to query

        Returns:
            Next nonce value
        """
        checksum_address = Web3.to_checksum_address(address)
        return self.w3.eth.get_transaction_count(checksum_address, "pending")

    async def wait_for_transaction(
        self, tx_hash: str, timeout: int = 120
    ) -> TransactionResult:
        """
        Wait for a transaction to be confirmed.

        Args:
            tx_hash: Transaction hash to wait for
            timeout: Maximum seconds to wait

        Returns:
            Transaction result with confirmation details
        """
        try:
            receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash, timeout=timeout)

            return TransactionResult(
                success=receipt["status"] == 1,
                tx_hash=tx_hash,
                block_number=receipt["blockNumber"],
                gas_used=receipt["gasUsed"],
                effective_gas_price=receipt.get("effectiveGasPrice"),
                network=self.network,
                error=None if receipt["status"] == 1 else "Transaction reverted",
            )
        except TransactionNotFound:
            return TransactionResult(
                success=False,
                tx_hash=tx_hash,
                error="Transaction not found",
                network=self.network,
            )
        except TimeoutError as e:
            return TransactionResult(
                success=False,
                tx_hash=tx_hash,
                error=str(e),
                network=self.network,
            )
        except (ConnectionError, ValueError) as e:
            return TransactionResult(
                success=False,
                tx_hash=tx_hash,
                error=str(e),
                network=self.network,
            )
        except Exception as e:
            return TransactionResult(
                success=False,
                tx_hash=tx_hash,
                error=str(e),
                network=self.network,
            )

    def _build_tx_dict(
        self, request: TransactionRequest, include_gas: bool = True
    ) -> dict:
        """Build a transaction dictionary for estimation or signing."""
        to_address = Web3.to_checksum_address(request.to_address)
        value_wei = int(request.amount * Decimal(10**self.config.native_decimals))

        tx_dict = {
            "to": to_address,
            "value": value_wei,
            "chainId": self.config.chain_id,
        }

        if request.from_address:
            tx_dict["from"] = Web3.to_checksum_address(request.from_address)

        if request.data:
            tx_dict["data"] = request.data

        if include_gas and request.gas_limit:
            tx_dict["gas"] = request.gas_limit

        return tx_dict

    async def close(self):
        """Clean up web3 provider connection."""
        # HTTPProvider doesn't require explicit cleanup
        pass

    def get_address_from_private_key(self, private_key: bytes) -> str:
        """
        Derive EVM address from private key.

        Args:
            private_key: Raw private key bytes

        Returns:
            Checksummed EVM address (0x...)
        """
        account = Account.from_key(private_key)
        return account.address
