"""
Base classes and types for multi-chain transaction support.

Provides:
- ChainNetwork enum for supported networks
- NetworkConfig with RPC URLs and chain IDs
- TransactionRequest/Result dataclasses
- ChainAdapter ABC for implementing chain-specific logic
"""

import os
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


class ChainNetwork(Enum):
    """Supported blockchain networks."""

    # Filecoin FEVM (EVM-compatible layer)
    FILECOIN_CALIBRATION = "filecoin_calibration"  # Testnet
    FILECOIN_MAINNET = "filecoin_mainnet"

    # Ethereum
    ETHEREUM_SEPOLIA = "ethereum_sepolia"  # Testnet
    ETHEREUM_MAINNET = "ethereum_mainnet"

    # Polygon
    POLYGON_AMOY = "polygon_amoy"  # Testnet (formerly Mumbai)
    POLYGON_MAINNET = "polygon_mainnet"

    # Base (Coinbase L2) — required for x402 Lighthouse payments
    BASE_SEPOLIA = "base_sepolia"  # Testnet
    BASE_MAINNET = "base_mainnet"

    @property
    def is_testnet(self) -> bool:
        """Check if this is a testnet."""
        return self in (
            ChainNetwork.FILECOIN_CALIBRATION,
            ChainNetwork.ETHEREUM_SEPOLIA,
            ChainNetwork.POLYGON_AMOY,
            ChainNetwork.BASE_SEPOLIA,
        )

    @property
    def is_mainnet(self) -> bool:
        """Check if this is a mainnet."""
        return not self.is_testnet

    @property
    def display_name(self) -> str:
        """Human-readable network name."""
        names = {
            ChainNetwork.FILECOIN_CALIBRATION: "Filecoin Calibration (Testnet)",
            ChainNetwork.FILECOIN_MAINNET: "Filecoin Mainnet",
            ChainNetwork.ETHEREUM_SEPOLIA: "Ethereum Sepolia (Testnet)",
            ChainNetwork.ETHEREUM_MAINNET: "Ethereum Mainnet",
            ChainNetwork.POLYGON_AMOY: "Polygon Amoy (Testnet)",
            ChainNetwork.POLYGON_MAINNET: "Polygon Mainnet",
            ChainNetwork.BASE_SEPOLIA: "Base Sepolia (Testnet)",
            ChainNetwork.BASE_MAINNET: "Base Mainnet",
        }
        return names.get(self, self.value)


@dataclass
class NetworkConfig:
    """Configuration for a blockchain network."""

    network: ChainNetwork
    rpc_url: str
    chain_id: int
    native_token: str
    native_decimals: int = 18
    explorer_url: Optional[str] = None
    faucet_url: Optional[str] = None

    @classmethod
    def get_config(cls, network: ChainNetwork) -> "NetworkConfig":
        """Get configuration for a network, with env var overrides."""
        configs = {
            ChainNetwork.FILECOIN_CALIBRATION: cls(
                network=ChainNetwork.FILECOIN_CALIBRATION,
                rpc_url=os.environ.get(
                    "FILECOIN_CALIBRATION_RPC",
                    "https://api.calibration.node.glif.io/rpc/v1",
                ),
                chain_id=314159,
                native_token="tFIL",
                explorer_url="https://calibration.filfox.info/en",
                faucet_url="https://faucet.calibnet.chainsafe-fil.io/",
            ),
            ChainNetwork.FILECOIN_MAINNET: cls(
                network=ChainNetwork.FILECOIN_MAINNET,
                rpc_url=os.environ.get(
                    "FILECOIN_MAINNET_RPC",
                    "https://api.node.glif.io/rpc/v1",
                ),
                chain_id=314,
                native_token="FIL",
                explorer_url="https://filfox.info/en",
            ),
            ChainNetwork.ETHEREUM_SEPOLIA: cls(
                network=ChainNetwork.ETHEREUM_SEPOLIA,
                rpc_url=os.environ.get(
                    "ETHEREUM_SEPOLIA_RPC",
                    "https://rpc.sepolia.org",
                ),
                chain_id=11155111,
                native_token="ETH",
                explorer_url="https://sepolia.etherscan.io",
                faucet_url="https://sepoliafaucet.com/",
            ),
            ChainNetwork.ETHEREUM_MAINNET: cls(
                network=ChainNetwork.ETHEREUM_MAINNET,
                rpc_url=os.environ.get(
                    "ETHEREUM_MAINNET_RPC",
                    "https://eth.llamarpc.com",
                ),
                chain_id=1,
                native_token="ETH",
                explorer_url="https://etherscan.io",
            ),
            ChainNetwork.POLYGON_AMOY: cls(
                network=ChainNetwork.POLYGON_AMOY,
                rpc_url=os.environ.get(
                    "POLYGON_AMOY_RPC",
                    "https://rpc-amoy.polygon.technology/",
                ),
                chain_id=80002,
                native_token="MATIC",
                explorer_url="https://amoy.polygonscan.com",
                faucet_url="https://faucet.polygon.technology/",
            ),
            ChainNetwork.POLYGON_MAINNET: cls(
                network=ChainNetwork.POLYGON_MAINNET,
                rpc_url=os.environ.get(
                    "POLYGON_MAINNET_RPC",
                    "https://polygon-rpc.com/",
                ),
                chain_id=137,
                native_token="MATIC",
                explorer_url="https://polygonscan.com",
            ),
            ChainNetwork.BASE_SEPOLIA: cls(
                network=ChainNetwork.BASE_SEPOLIA,
                rpc_url=os.environ.get(
                    "BASE_SEPOLIA_RPC",
                    "https://sepolia.base.org",
                ),
                chain_id=84532,
                native_token="ETH",
                explorer_url="https://sepolia.basescan.org",
                faucet_url="https://www.coinbase.com/faucets/base-ethereum-goerli-faucet",
            ),
            ChainNetwork.BASE_MAINNET: cls(
                network=ChainNetwork.BASE_MAINNET,
                rpc_url=os.environ.get(
                    "BASE_MAINNET_RPC",
                    "https://mainnet.base.org",
                ),
                chain_id=8453,
                native_token="ETH",
                explorer_url="https://basescan.org",
            ),
        }
        return configs[network]

    def get_tx_url(self, tx_hash: str) -> str:
        """Get explorer URL for a transaction."""
        if self.explorer_url:
            return f"{self.explorer_url}/tx/{tx_hash}"
        return tx_hash

    def get_address_url(self, address: str) -> str:
        """Get explorer URL for an address."""
        if self.explorer_url:
            return f"{self.explorer_url}/address/{address}"
        return address


@dataclass
class GasEstimate:
    """Gas estimation for a transaction."""

    gas_limit: int
    gas_price_wei: int
    max_fee_per_gas: Optional[int] = None  # EIP-1559
    max_priority_fee_per_gas: Optional[int] = None  # EIP-1559
    estimated_cost_wei: int = 0
    estimated_cost_native: Decimal = Decimal("0")

    def __post_init__(self):
        if self.estimated_cost_wei == 0:
            if self.max_fee_per_gas:
                self.estimated_cost_wei = self.gas_limit * self.max_fee_per_gas
            else:
                self.estimated_cost_wei = self.gas_limit * self.gas_price_wei
        if self.estimated_cost_native == Decimal("0"):
            self.estimated_cost_native = Decimal(self.estimated_cost_wei) / Decimal(
                10**18
            )


@dataclass
class TransactionRequest:
    """Request to send a transaction."""

    to_address: str
    amount: Decimal
    network: ChainNetwork
    from_address: Optional[str] = None
    data: Optional[bytes] = None  # Contract call data
    gas_limit: Optional[int] = None
    gas_price_wei: Optional[int] = None
    max_fee_per_gas: Optional[int] = None
    max_priority_fee_per_gas: Optional[int] = None
    nonce: Optional[int] = None
    metadata: dict = field(default_factory=dict)


@dataclass
class TransactionResult:
    """Result of a transaction."""

    success: bool
    tx_hash: Optional[str] = None
    block_number: Optional[int] = None
    gas_used: Optional[int] = None
    effective_gas_price: Optional[int] = None
    error: Optional[str] = None
    network: Optional[ChainNetwork] = None

    def get_explorer_url(self) -> Optional[str]:
        """Get explorer URL for this transaction."""
        if self.tx_hash and self.network:
            config = NetworkConfig.get_config(self.network)
            return config.get_tx_url(self.tx_hash)
        return None


class ChainAdapter(ABC):
    """
    Abstract base class for blockchain adapters.

    Implementations provide chain-specific transaction signing and broadcasting.
    """

    def __init__(self, network: ChainNetwork):
        """
        Initialize adapter for a specific network.

        Args:
            network: The blockchain network to connect to
        """
        self.network = network
        self.config = NetworkConfig.get_config(network)

    @abstractmethod
    async def is_connected(self) -> bool:
        """Check if connected to the network."""
        pass

    @abstractmethod
    async def get_balance(self, address: str) -> Decimal:
        """
        Get native token balance for an address.

        Args:
            address: The address to query

        Returns:
            Balance in native token units (e.g., ETH, FIL)
        """
        pass

    @abstractmethod
    async def estimate_gas(self, request: TransactionRequest) -> GasEstimate:
        """
        Estimate gas for a transaction.

        Args:
            request: The transaction to estimate

        Returns:
            Gas estimation with costs
        """
        pass

    @abstractmethod
    async def send_transaction(
        self, request: TransactionRequest, private_key: bytes
    ) -> TransactionResult:
        """
        Sign and broadcast a transaction.

        Args:
            request: The transaction to send
            private_key: Raw private key bytes for signing

        Returns:
            Transaction result with hash or error
        """
        pass

    @abstractmethod
    def validate_address(self, address: str) -> bool:
        """
        Validate an address for this network.

        Args:
            address: The address to validate

        Returns:
            True if valid
        """
        pass

    async def get_nonce(self, address: str) -> int:
        """
        Get the next nonce for an address.

        Args:
            address: The address to query

        Returns:
            Next nonce value
        """
        raise NotImplementedError("Subclass must implement get_nonce")

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
        raise NotImplementedError("Subclass must implement wait_for_transaction")

    async def close(self):
        """Clean up any connections."""
        pass
