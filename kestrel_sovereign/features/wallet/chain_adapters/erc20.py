"""
ERC-20 Token Adapter for stablecoin transfers.

Builds on EVMAdapter to provide ERC-20 token transfers (USDC, USDT, etc.)
Uses minimal ABI for transfer, balanceOf, and decimals functions.
"""

import logging
from decimal import Decimal
from typing import Optional

try:
    from web3 import Web3
    from eth_account import Account
except ImportError:
    Web3 = None  # type: ignore[assignment,misc]
    Account = None  # type: ignore[assignment,misc]

from .base import (
    ChainNetwork,
    TransactionRequest,
    TransactionResult,
    GasEstimate,
)
from .evm_adapter import EVMAdapter
from .token_registry import TokenRegistry, TokenInfo

logger = logging.getLogger(__name__)


# Minimal ERC-20 ABI for transfer operations
ERC20_ABI = [
    {
        "name": "transfer",
        "type": "function",
        "stateMutability": "nonpayable",
        "inputs": [
            {"name": "to", "type": "address"},
            {"name": "amount", "type": "uint256"},
        ],
        "outputs": [{"name": "", "type": "bool"}],
    },
    {
        "name": "balanceOf",
        "type": "function",
        "stateMutability": "view",
        "inputs": [{"name": "account", "type": "address"}],
        "outputs": [{"name": "", "type": "uint256"}],
    },
    {
        "name": "decimals",
        "type": "function",
        "stateMutability": "view",
        "inputs": [],
        "outputs": [{"name": "", "type": "uint8"}],
    },
    {
        "name": "symbol",
        "type": "function",
        "stateMutability": "view",
        "inputs": [],
        "outputs": [{"name": "", "type": "string"}],
    },
    {
        "name": "approve",
        "type": "function",
        "stateMutability": "nonpayable",
        "inputs": [
            {"name": "spender", "type": "address"},
            {"name": "amount", "type": "uint256"},
        ],
        "outputs": [{"name": "", "type": "bool"}],
    },
    {
        "name": "allowance",
        "type": "function",
        "stateMutability": "view",
        "inputs": [
            {"name": "owner", "type": "address"},
            {"name": "spender", "type": "address"},
        ],
        "outputs": [{"name": "", "type": "uint256"}],
    },
]


class ERC20Adapter:
    """
    Adapter for ERC-20 token operations.

    Uses EVMAdapter for underlying transaction signing.
    """

    # Default gas limit for ERC-20 transfers
    DEFAULT_GAS_LIMIT = 100000

    def __init__(self, evm_adapter: EVMAdapter):
        """
        Initialize ERC-20 adapter.

        Args:
            evm_adapter: Underlying EVM adapter for the network
        """
        self.evm = evm_adapter
        self.w3 = evm_adapter.w3
        self.network = evm_adapter.network

    async def get_token_balance(
        self, token_address: str, holder_address: str
    ) -> Decimal:
        """
        Get token balance for an address.

        Args:
            token_address: ERC-20 contract address
            holder_address: Address to check balance

        Returns:
            Token balance in token units
        """
        contract = self.w3.eth.contract(
            address=Web3.to_checksum_address(token_address),
            abi=ERC20_ABI,
        )

        holder = Web3.to_checksum_address(holder_address)
        balance_wei = contract.functions.balanceOf(holder).call()
        decimals = contract.functions.decimals().call()

        return Decimal(balance_wei) / Decimal(10**decimals)

    async def get_token_balance_by_symbol(
        self, symbol: str, holder_address: str
    ) -> Optional[Decimal]:
        """
        Get token balance by symbol (e.g., "USDC").

        Args:
            symbol: Token symbol from registry
            holder_address: Address to check balance

        Returns:
            Token balance or None if token not found
        """
        token = TokenRegistry.get_token(symbol, self.network)
        if not token:
            logger.warning(f"Token {symbol} not found on {self.network.display_name}")
            return None

        return await self.get_token_balance(token.address, holder_address)

    async def estimate_transfer_gas(
        self,
        token_address: str,
        from_address: str,
        to_address: str,
        amount: Decimal,
    ) -> GasEstimate:
        """
        Estimate gas for a token transfer.

        Args:
            token_address: ERC-20 contract address
            from_address: Sender address
            to_address: Recipient address
            amount: Amount to transfer

        Returns:
            Gas estimation
        """
        contract = self.w3.eth.contract(
            address=Web3.to_checksum_address(token_address),
            abi=ERC20_ABI,
        )

        # Get decimals
        decimals = contract.functions.decimals().call()
        amount_wei = int(amount * Decimal(10**decimals))

        # Build transfer call data
        transfer_data = contract.functions.transfer(
            Web3.to_checksum_address(to_address),
            amount_wei,
        )._encode_transaction_data()

        # Create request for gas estimation
        request = TransactionRequest(
            to_address=token_address,
            amount=Decimal("0"),  # No native token transfer
            network=self.network,
            from_address=from_address,
            data=transfer_data,
        )

        estimate = await self.evm.estimate_gas(request)

        # Ensure minimum gas for ERC-20 transfers
        if estimate.gas_limit < self.DEFAULT_GAS_LIMIT:
            estimate.gas_limit = self.DEFAULT_GAS_LIMIT

        return estimate

    async def transfer(
        self,
        token_address: str,
        to_address: str,
        amount: Decimal,
        private_key: bytes,
        gas_limit: Optional[int] = None,
    ) -> TransactionResult:
        """
        Transfer ERC-20 tokens.

        Args:
            token_address: ERC-20 contract address
            to_address: Recipient address
            amount: Amount to transfer (in token units, e.g., 100 USDC)
            private_key: Sender's private key
            gas_limit: Optional gas limit override

        Returns:
            Transaction result
        """
        try:
            # Derive sender address
            account = Account.from_key(private_key)
            from_address = account.address

            # Get contract
            contract = self.w3.eth.contract(
                address=Web3.to_checksum_address(token_address),
                abi=ERC20_ABI,
            )

            # Get decimals and convert amount
            decimals = contract.functions.decimals().call()
            amount_wei = int(amount * Decimal(10**decimals))

            # Check balance
            balance = contract.functions.balanceOf(from_address).call()
            if balance < amount_wei:
                return TransactionResult(
                    success=False,
                    error=f"Insufficient token balance: have {Decimal(balance) / Decimal(10**decimals)}, need {amount}",
                    network=self.network,
                )

            # Build transfer call data
            transfer_data = contract.functions.transfer(
                Web3.to_checksum_address(to_address),
                amount_wei,
            )._encode_transaction_data()

            # Create transaction request
            request = TransactionRequest(
                to_address=token_address,  # Send to contract
                amount=Decimal("0"),  # No native token
                network=self.network,
                from_address=from_address,
                data=transfer_data,
                gas_limit=gas_limit or self.DEFAULT_GAS_LIMIT,
            )

            # Send via EVM adapter
            result = await self.evm.send_transaction(request, private_key)

            if result.success:
                logger.info(
                    f"Token transfer: {amount} to {to_address}, tx: {result.tx_hash}"
                )

            return result

        except (ConnectionError, TimeoutError) as e:
            logger.error(f"Token transfer failed: {e}")
            return TransactionResult(
                success=False,
                error=str(e),
                network=self.network,
            )
        except (ValueError, TypeError, KeyError) as e:
            logger.error(f"Token transfer failed: {e}")
            return TransactionResult(
                success=False,
                error=str(e),
                network=self.network,
            )
        except Exception as e:
            logger.error(f"Token transfer failed: {e}", exc_info=True)
            return TransactionResult(
                success=False,
                error=str(e),
                network=self.network,
            )

    async def transfer_by_symbol(
        self,
        symbol: str,
        to_address: str,
        amount: Decimal,
        private_key: bytes,
        gas_limit: Optional[int] = None,
    ) -> TransactionResult:
        """
        Transfer tokens by symbol (e.g., "USDC").

        Args:
            symbol: Token symbol from registry
            to_address: Recipient address
            amount: Amount to transfer
            private_key: Sender's private key
            gas_limit: Optional gas limit override

        Returns:
            Transaction result
        """
        token = TokenRegistry.get_token(symbol, self.network)
        if not token:
            return TransactionResult(
                success=False,
                error=f"Token {symbol} not found on {self.network.display_name}",
                network=self.network,
            )

        return await self.transfer(
            token_address=token.address,
            to_address=to_address,
            amount=amount,
            private_key=private_key,
            gas_limit=gas_limit,
        )

    async def get_token_info(self, token_address: str) -> Optional[TokenInfo]:
        """
        Get token information from contract.

        Args:
            token_address: ERC-20 contract address

        Returns:
            TokenInfo or None if not a valid ERC-20
        """
        try:
            contract = self.w3.eth.contract(
                address=Web3.to_checksum_address(token_address),
                abi=ERC20_ABI,
            )

            symbol = contract.functions.symbol().call()
            decimals = contract.functions.decimals().call()

            return TokenInfo(
                symbol=symbol,
                name=symbol,  # Name not in minimal ABI
                address=token_address,
                decimals=decimals,
                network=self.network,
            )
        except (ConnectionError, TimeoutError) as e:
            logger.warning(f"Failed to get token info for {token_address}: {e}")
            return None
        except (ValueError, TypeError) as e:
            logger.warning(f"Failed to get token info for {token_address}: {e}")
            return None
        except Exception as e:
            logger.warning(f"Failed to get token info for {token_address}: {e}", exc_info=True)
            return None

    def list_available_tokens(self) -> list[TokenInfo]:
        """List all registered tokens for this network."""
        return TokenRegistry.list_tokens(self.network)
