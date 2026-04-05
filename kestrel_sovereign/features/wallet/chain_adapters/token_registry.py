"""
Token Registry for ERC-20 tokens across networks.

Provides contract addresses for well-known tokens on each network.
"""

import logging
from dataclasses import dataclass
from typing import Dict, Optional

from .base import ChainNetwork

logger = logging.getLogger(__name__)


@dataclass
class TokenInfo:
    """Information about an ERC-20 token."""

    symbol: str
    name: str
    address: str
    decimals: int
    network: ChainNetwork

    def __str__(self) -> str:
        return f"{self.symbol} ({self.name}) on {self.network.display_name}"


class TokenRegistry:
    """
    Registry of known ERC-20 tokens by network.

    Includes mainnet and testnet addresses for common stablecoins.
    """

    # Token addresses by network
    _TOKENS: Dict[ChainNetwork, Dict[str, TokenInfo]] = {
        # Ethereum Sepolia Testnet
        ChainNetwork.ETHEREUM_SEPOLIA: {
            "USDC": TokenInfo(
                symbol="USDC",
                name="USD Coin",
                address="0x1c7D4B196Cb0C7B01d743Fbc6116a902379C7238",
                decimals=6,
                network=ChainNetwork.ETHEREUM_SEPOLIA,
            ),
            "USDT": TokenInfo(
                symbol="USDT",
                name="Tether USD",
                address="0xaA8E23Fb1079EA71e0a56F48a2aA51851D8433D0",
                decimals=6,
                network=ChainNetwork.ETHEREUM_SEPOLIA,
            ),
        },
        # Ethereum Mainnet
        ChainNetwork.ETHEREUM_MAINNET: {
            "USDC": TokenInfo(
                symbol="USDC",
                name="USD Coin",
                address="0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48",
                decimals=6,
                network=ChainNetwork.ETHEREUM_MAINNET,
            ),
            "USDT": TokenInfo(
                symbol="USDT",
                name="Tether USD",
                address="0xdAC17F958D2ee523a2206206994597C13D831ec7",
                decimals=6,
                network=ChainNetwork.ETHEREUM_MAINNET,
            ),
            "DAI": TokenInfo(
                symbol="DAI",
                name="Dai Stablecoin",
                address="0x6B175474E89094C44Da98b954EedeAC495271d0F",
                decimals=18,
                network=ChainNetwork.ETHEREUM_MAINNET,
            ),
        },
        # Polygon Amoy Testnet
        ChainNetwork.POLYGON_AMOY: {
            "USDC": TokenInfo(
                symbol="USDC",
                name="USD Coin (Test)",
                address="0x41E94cFAEd3F3B7e1b6195Cd2816830010854685",
                decimals=6,
                network=ChainNetwork.POLYGON_AMOY,
            ),
            "USDT": TokenInfo(
                symbol="USDT",
                name="Tether USD (Test)",
                address="0xc85F14b050B277c7aCB8FC96f26c8a9538EaB662",
                decimals=6,
                network=ChainNetwork.POLYGON_AMOY,
            ),
        },
        # Polygon Mainnet
        ChainNetwork.POLYGON_MAINNET: {
            "USDC": TokenInfo(
                symbol="USDC",
                name="USD Coin",
                address="0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174",
                decimals=6,
                network=ChainNetwork.POLYGON_MAINNET,
            ),
            "USDT": TokenInfo(
                symbol="USDT",
                name="Tether USD",
                address="0xc2132D05D31c914a87C6611C10748AEb04B58e8F",
                decimals=6,
                network=ChainNetwork.POLYGON_MAINNET,
            ),
            "DAI": TokenInfo(
                symbol="DAI",
                name="Dai Stablecoin",
                address="0x8f3Cf7ad23Cd3CaDbD9735AFf958023239c6A063",
                decimals=18,
                network=ChainNetwork.POLYGON_MAINNET,
            ),
        },
        # Base Sepolia Testnet
        ChainNetwork.BASE_SEPOLIA: {
            "USDC": TokenInfo(
                symbol="USDC",
                name="USD Coin (Test)",
                address="0x036CbD53842c5426634e7929541eC2318f3dCF7e",
                decimals=6,
                network=ChainNetwork.BASE_SEPOLIA,
            ),
        },
        # Base Mainnet
        ChainNetwork.BASE_MAINNET: {
            "USDC": TokenInfo(
                symbol="USDC",
                name="USD Coin",
                address="0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913",
                decimals=6,
                network=ChainNetwork.BASE_MAINNET,
            ),
            "USDT": TokenInfo(
                symbol="USDT",
                name="Tether USD",
                address="0xfde4C96c8593536E31F229EA8f37b2ADa2699bb2",
                decimals=6,
                network=ChainNetwork.BASE_MAINNET,
            ),
        },
        # Filecoin networks - no official stablecoins yet
        ChainNetwork.FILECOIN_CALIBRATION: {},
        ChainNetwork.FILECOIN_MAINNET: {},
    }

    @classmethod
    def get_token(
        cls, symbol: str, network: ChainNetwork
    ) -> Optional[TokenInfo]:
        """
        Get token info by symbol and network.

        Args:
            symbol: Token symbol (e.g., "USDC")
            network: Target network

        Returns:
            TokenInfo or None if not found
        """
        network_tokens = cls._TOKENS.get(network, {})
        return network_tokens.get(symbol.upper())

    @classmethod
    def get_token_by_address(
        cls, address: str, network: ChainNetwork
    ) -> Optional[TokenInfo]:
        """
        Get token info by contract address.

        Args:
            address: Token contract address
            network: Target network

        Returns:
            TokenInfo or None if not found
        """
        address_lower = address.lower()
        network_tokens = cls._TOKENS.get(network, {})

        for token in network_tokens.values():
            if token.address.lower() == address_lower:
                return token
        return None

    @classmethod
    def list_tokens(cls, network: ChainNetwork) -> list[TokenInfo]:
        """
        List all known tokens for a network.

        Args:
            network: Target network

        Returns:
            List of TokenInfo objects
        """
        return list(cls._TOKENS.get(network, {}).values())

    @classmethod
    def has_tokens(cls, network: ChainNetwork) -> bool:
        """Check if a network has any registered tokens."""
        return bool(cls._TOKENS.get(network, {}))

    @classmethod
    def get_all_networks_with_tokens(cls) -> list[ChainNetwork]:
        """Get list of networks that have registered tokens."""
        return [
            network
            for network, tokens in cls._TOKENS.items()
            if tokens
        ]

    @classmethod
    def register_token(cls, token: TokenInfo) -> None:
        """
        Register a custom token.

        Args:
            token: TokenInfo to register
        """
        if token.network not in cls._TOKENS:
            cls._TOKENS[token.network] = {}
        cls._TOKENS[token.network][token.symbol.upper()] = token
        logger.info(f"Registered token: {token}")
