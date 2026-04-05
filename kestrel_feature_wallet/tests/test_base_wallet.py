"""Tests for Base L2 wallet and chain adapter additions."""

import pytest

from kestrel_feature_wallet.chain_adapters.base import (
    ChainNetwork,
    NetworkConfig,
)
from kestrel_feature_wallet.chain_adapters.token_registry import TokenRegistry
from kestrel_feature_wallet.chain_adapters.evm_adapter import MAINNET_CHAIN_IDS


class TestBaseChainNetwork:
    """Test Base chain additions to ChainNetwork enum."""

    def test_base_sepolia_exists(self):
        assert ChainNetwork.BASE_SEPOLIA.value == "base_sepolia"

    def test_base_mainnet_exists(self):
        assert ChainNetwork.BASE_MAINNET.value == "base_mainnet"

    def test_base_sepolia_is_testnet(self):
        assert ChainNetwork.BASE_SEPOLIA.is_testnet is True
        assert ChainNetwork.BASE_SEPOLIA.is_mainnet is False

    def test_base_mainnet_is_mainnet(self):
        assert ChainNetwork.BASE_MAINNET.is_mainnet is True
        assert ChainNetwork.BASE_MAINNET.is_testnet is False

    def test_display_names(self):
        assert "Base Sepolia" in ChainNetwork.BASE_SEPOLIA.display_name
        assert "Base Mainnet" in ChainNetwork.BASE_MAINNET.display_name


class TestBaseNetworkConfig:
    """Test Base network configurations."""

    def test_base_sepolia_config(self):
        config = NetworkConfig.get_config(ChainNetwork.BASE_SEPOLIA)
        assert config.chain_id == 84532
        assert config.native_token == "ETH"
        assert "sepolia.base.org" in config.rpc_url

    def test_base_mainnet_config(self):
        config = NetworkConfig.get_config(ChainNetwork.BASE_MAINNET)
        assert config.chain_id == 8453
        assert config.native_token == "ETH"
        assert "mainnet.base.org" in config.rpc_url

    def test_base_sepolia_has_faucet(self):
        config = NetworkConfig.get_config(ChainNetwork.BASE_SEPOLIA)
        assert config.faucet_url is not None

    def test_base_mainnet_explorer(self):
        config = NetworkConfig.get_config(ChainNetwork.BASE_MAINNET)
        assert config.explorer_url == "https://basescan.org"


class TestBaseTokenRegistry:
    """Test Base token registrations."""

    def test_base_mainnet_usdc(self):
        token = TokenRegistry.get_token("USDC", ChainNetwork.BASE_MAINNET)
        assert token is not None
        assert token.address == "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913"
        assert token.decimals == 6

    def test_base_sepolia_usdc(self):
        token = TokenRegistry.get_token("USDC", ChainNetwork.BASE_SEPOLIA)
        assert token is not None
        assert token.address == "0x036CbD53842c5426634e7929541eC2318f3dCF7e"
        assert token.decimals == 6

    def test_base_mainnet_usdt(self):
        token = TokenRegistry.get_token("USDT", ChainNetwork.BASE_MAINNET)
        assert token is not None
        assert token.decimals == 6

    def test_base_networks_have_tokens(self):
        assert TokenRegistry.has_tokens(ChainNetwork.BASE_MAINNET)
        assert TokenRegistry.has_tokens(ChainNetwork.BASE_SEPOLIA)


class TestMainnetBlocklist:
    """Test that Base mainnet is in the safety blocklist."""

    def test_base_mainnet_blocked(self):
        assert 8453 in MAINNET_CHAIN_IDS


try:
    import eth_account as _eth_account
    _has_web3 = True
except ImportError:
    _has_web3 = False


@pytest.mark.skipif(not _has_web3, reason="web3 extras not installed")
class TestBaseL2Wallet:
    """Test BaseL2Wallet initialization and address derivation."""

    def test_address_derivation(self):
        """Test that wallet derives correct EVM address from known key."""
        from kestrel_feature_wallet.base_l2_wallet import BaseL2Wallet

        # Known test key (DO NOT use in production)
        test_key = "0x" + "ab" * 32
        wallet = BaseL2Wallet(private_key_hex=test_key, testnet=True)

        assert wallet.address.startswith("0x")
        assert len(wallet.address) == 42
        assert wallet.network == ChainNetwork.BASE_SEPOLIA

    def test_testnet_vs_mainnet(self):
        from kestrel_feature_wallet.base_l2_wallet import BaseL2Wallet

        test_key = "ab" * 32
        testnet_wallet = BaseL2Wallet(private_key_hex=test_key, testnet=True)
        mainnet_wallet = BaseL2Wallet(private_key_hex=test_key, testnet=False)

        # Same key → same address
        assert testnet_wallet.address == mainnet_wallet.address
        # Different networks
        assert testnet_wallet.network == ChainNetwork.BASE_SEPOLIA
        assert mainnet_wallet.network == ChainNetwork.BASE_MAINNET

    def test_usdc_address(self):
        from kestrel_feature_wallet.base_l2_wallet import BaseL2Wallet

        wallet = BaseL2Wallet(private_key_hex="ab" * 32, testnet=True)
        assert wallet.usdc_address == "0x036CbD53842c5426634e7929541eC2318f3dCF7e"

    def test_repr(self):
        from kestrel_feature_wallet.base_l2_wallet import BaseL2Wallet

        wallet = BaseL2Wallet(private_key_hex="ab" * 32, testnet=True)
        r = repr(wallet)
        assert "BaseL2Wallet" in r
        assert "0x" in r
