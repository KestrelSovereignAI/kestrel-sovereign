"""
Integration tests for EVM transaction signing.

Tests the multi-chain wallet system with real testnets:
- Filecoin Calibration (FEVM)
- Ethereum Sepolia
- Polygon Amoy

Run with: RUN_NETWORK_TESTS=true pytest tests/integration/test_evm_transactions_e2e.py -v

Note: These tests require:
1. RUN_NETWORK_TESTS=true environment variable
2. Funded test wallet (get testnet tokens from faucets)
3. Network connectivity to RPC endpoints
"""

import os
import pytest
from decimal import Decimal
from unittest.mock import patch, MagicMock

# Skip all tests if RUN_NETWORK_TESTS is not set
pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_NETWORK_TESTS", "").lower() != "true",
    reason="Set RUN_NETWORK_TESTS=true to run network tests",
)


class TestChainNetwork:
    """Test ChainNetwork enum and configuration."""

    def test_network_enum_values(self):
        """Test all network enum values exist."""
        from kestrel_sovereign.features.wallet.chain_adapters import ChainNetwork

        assert ChainNetwork.FILECOIN_CALIBRATION.value == "filecoin_calibration"
        assert ChainNetwork.FILECOIN_MAINNET.value == "filecoin_mainnet"
        assert ChainNetwork.ETHEREUM_SEPOLIA.value == "ethereum_sepolia"
        assert ChainNetwork.ETHEREUM_MAINNET.value == "ethereum_mainnet"
        assert ChainNetwork.POLYGON_AMOY.value == "polygon_amoy"
        assert ChainNetwork.POLYGON_MAINNET.value == "polygon_mainnet"

    def test_testnet_detection(self):
        """Test is_testnet property."""
        from kestrel_sovereign.features.wallet.chain_adapters import ChainNetwork

        assert ChainNetwork.FILECOIN_CALIBRATION.is_testnet is True
        assert ChainNetwork.ETHEREUM_SEPOLIA.is_testnet is True
        assert ChainNetwork.POLYGON_AMOY.is_testnet is True
        assert ChainNetwork.FILECOIN_MAINNET.is_testnet is False
        assert ChainNetwork.ETHEREUM_MAINNET.is_testnet is False
        assert ChainNetwork.POLYGON_MAINNET.is_testnet is False

    def test_mainnet_detection(self):
        """Test is_mainnet property."""
        from kestrel_sovereign.features.wallet.chain_adapters import ChainNetwork

        assert ChainNetwork.FILECOIN_MAINNET.is_mainnet is True
        assert ChainNetwork.ETHEREUM_MAINNET.is_mainnet is True
        assert ChainNetwork.POLYGON_MAINNET.is_mainnet is True
        assert ChainNetwork.FILECOIN_CALIBRATION.is_mainnet is False

    def test_display_names(self):
        """Test human-readable display names."""
        from kestrel_sovereign.features.wallet.chain_adapters import ChainNetwork

        assert "Testnet" in ChainNetwork.FILECOIN_CALIBRATION.display_name
        assert "Testnet" in ChainNetwork.ETHEREUM_SEPOLIA.display_name
        assert "Mainnet" in ChainNetwork.FILECOIN_MAINNET.display_name


class TestNetworkConfig:
    """Test NetworkConfig class."""

    def test_get_config_for_all_networks(self):
        """Test config exists for all networks."""
        from kestrel_sovereign.features.wallet.chain_adapters import ChainNetwork, NetworkConfig

        for network in ChainNetwork:
            config = NetworkConfig.get_config(network)
            assert config.network == network
            assert config.rpc_url is not None
            assert config.chain_id > 0
            assert config.native_token is not None

    def test_chain_ids_are_correct(self):
        """Test chain IDs match expected values."""
        from kestrel_sovereign.features.wallet.chain_adapters import ChainNetwork, NetworkConfig

        assert NetworkConfig.get_config(ChainNetwork.FILECOIN_CALIBRATION).chain_id == 314159
        assert NetworkConfig.get_config(ChainNetwork.FILECOIN_MAINNET).chain_id == 314
        assert NetworkConfig.get_config(ChainNetwork.ETHEREUM_SEPOLIA).chain_id == 11155111
        assert NetworkConfig.get_config(ChainNetwork.ETHEREUM_MAINNET).chain_id == 1
        assert NetworkConfig.get_config(ChainNetwork.POLYGON_AMOY).chain_id == 80002
        assert NetworkConfig.get_config(ChainNetwork.POLYGON_MAINNET).chain_id == 137

    def test_explorer_urls(self):
        """Test explorer URL generation."""
        from kestrel_sovereign.features.wallet.chain_adapters import ChainNetwork, NetworkConfig

        config = NetworkConfig.get_config(ChainNetwork.ETHEREUM_SEPOLIA)
        tx_url = config.get_tx_url("0x123abc")
        assert "sepolia.etherscan.io" in tx_url
        assert "0x123abc" in tx_url


class TestEVMAdapter:
    """Test EVMAdapter functionality."""

    @pytest.fixture
    def sepolia_adapter(self):
        """Create adapter for Ethereum Sepolia."""
        from kestrel_sovereign.features.wallet.chain_adapters import ChainNetwork, EVMAdapter

        return EVMAdapter(ChainNetwork.ETHEREUM_SEPOLIA)

    def test_adapter_initialization(self, sepolia_adapter):
        """Test adapter initializes correctly."""
        from kestrel_sovereign.features.wallet.chain_adapters import ChainNetwork

        assert sepolia_adapter.network == ChainNetwork.ETHEREUM_SEPOLIA
        assert sepolia_adapter.config.chain_id == 11155111

    @pytest.mark.asyncio
    async def test_connection_check(self, sepolia_adapter):
        """Test RPC connection check."""
        is_connected = await sepolia_adapter.is_connected()
        assert is_connected is True

    def test_address_validation_valid(self, sepolia_adapter):
        """Test valid address validation."""
        valid_address = "0x742d35Cc6634C0532925a3b844Bc9e7595f3E123"
        assert sepolia_adapter.validate_address(valid_address) is True

    def test_address_validation_invalid(self, sepolia_adapter):
        """Test invalid address validation."""
        assert sepolia_adapter.validate_address("") is False
        assert sepolia_adapter.validate_address("not-an-address") is False
        assert sepolia_adapter.validate_address("0x123") is False
        assert sepolia_adapter.validate_address("742d35Cc6634C0532925a3b844Bc9e7595f3E123") is False

    @pytest.mark.asyncio
    async def test_get_balance(self, sepolia_adapter):
        """Test balance query for a known address."""
        # Use Sepolia faucet address (usually has some ETH)
        address = "0x742d35Cc6634C0532925a3b844Bc9e7595f3E123"
        balance = await sepolia_adapter.get_balance(address)
        assert isinstance(balance, Decimal)
        assert balance >= Decimal("0")


class TestTransactionRequest:
    """Test TransactionRequest dataclass."""

    def test_create_transaction_request(self):
        """Test creating a transaction request."""
        from kestrel_sovereign.features.wallet.chain_adapters import ChainNetwork, TransactionRequest

        request = TransactionRequest(
            to_address="0x742d35Cc6634C0532925a3b844Bc9e7595f3E123",
            amount=Decimal("0.01"),
            network=ChainNetwork.ETHEREUM_SEPOLIA,
        )
        assert request.to_address == "0x742d35Cc6634C0532925a3b844Bc9e7595f3E123"
        assert request.amount == Decimal("0.01")
        assert request.network == ChainNetwork.ETHEREUM_SEPOLIA


class TestTransactionSecurityHook:
    """Test TransactionSecurityHook validation."""

    @pytest.fixture
    def hook(self):
        """Create transaction security hook."""
        from kestrel_sovereign.features.wallet.transaction_hook import TransactionSecurityHook

        return TransactionSecurityHook(daily_limit_usd=Decimal("100"))

    def test_hook_initialization(self, hook):
        """Test hook initializes with correct settings."""
        assert hook.daily_limit_usd == Decimal("100")
        assert hook.allow_mainnet is False  # Default is blocked
        assert hook.require_approval is True

    @pytest.mark.asyncio
    async def test_allows_non_transaction_tools(self, hook):
        """Test hook allows non-transaction tools."""
        from kestrel_sovereign.hooks import HookInput

        input = HookInput(tool_name="some_other_tool", tool_input={})
        result = await hook.execute(input)
        assert result.allow is True

    @pytest.mark.asyncio
    async def test_blocks_mainnet_by_default(self, hook):
        """Test mainnet transactions are blocked by default."""
        from kestrel_sovereign.hooks import HookInput

        input = HookInput(
            tool_name="wallet_send",
            tool_input={
                "to_address": "0x742d35Cc6634C0532925a3b844Bc9e7595f3E123",
                "amount": "0.01",
                "network": "ethereum_mainnet",
            },
        )
        result = await hook.execute(input)
        assert result.allow is False
        assert "Mainnet" in result.message

    @pytest.mark.asyncio
    async def test_allows_testnet_transactions(self, hook):
        """Test testnet transactions are allowed."""
        from kestrel_sovereign.hooks import HookInput

        input = HookInput(
            tool_name="wallet_send",
            tool_input={
                "to_address": "0x742d35Cc6634C0532925a3b844Bc9e7595f3E123",
                "amount": "0.01",
                "network": "ethereum_sepolia",
            },
        )
        result = await hook.execute(input)
        assert result.allow is True

    @pytest.mark.asyncio
    async def test_validates_address_format(self, hook):
        """Test invalid addresses are rejected."""
        from kestrel_sovereign.hooks import HookInput

        input = HookInput(
            tool_name="wallet_send",
            tool_input={
                "to_address": "invalid-address",
                "amount": "0.01",
                "network": "ethereum_sepolia",
            },
        )
        result = await hook.execute(input)
        assert result.allow is False
        assert "Invalid address" in result.message


class TestTokenRegistry:
    """Test TokenRegistry for ERC-20 tokens."""

    def test_get_known_tokens(self):
        """Test retrieving known token addresses."""
        from kestrel_sovereign.features.wallet.chain_adapters import ChainNetwork, TokenRegistry

        usdc = TokenRegistry.get_token("USDC", ChainNetwork.ETHEREUM_SEPOLIA)
        assert usdc is not None
        assert usdc.symbol == "USDC"
        assert usdc.decimals == 6
        assert usdc.address.startswith("0x")

    def test_unknown_token_returns_none(self):
        """Test unknown token returns None."""
        from kestrel_sovereign.features.wallet.chain_adapters import ChainNetwork, TokenRegistry

        result = TokenRegistry.get_token("FAKE_TOKEN", ChainNetwork.ETHEREUM_SEPOLIA)
        assert result is None

    def test_list_tokens_for_network(self):
        """Test listing all tokens for a network."""
        from kestrel_sovereign.features.wallet.chain_adapters import ChainNetwork, TokenRegistry

        tokens = TokenRegistry.list_tokens(ChainNetwork.ETHEREUM_SEPOLIA)
        assert len(tokens) > 0
        symbols = [t.symbol for t in tokens]
        assert "USDC" in symbols


class TestERC20Adapter:
    """Test ERC20Adapter for token transfers."""

    @pytest.fixture
    def erc20_adapter(self):
        """Create ERC-20 adapter for Sepolia."""
        from kestrel_sovereign.features.wallet.chain_adapters import ChainNetwork, ERC20Adapter

        return ERC20Adapter(ChainNetwork.ETHEREUM_SEPOLIA)

    def test_adapter_initialization(self, erc20_adapter):
        """Test adapter initializes correctly."""
        from kestrel_sovereign.features.wallet.chain_adapters import ChainNetwork

        assert erc20_adapter.network == ChainNetwork.ETHEREUM_SEPOLIA

    @pytest.mark.asyncio
    async def test_get_token_balance(self, erc20_adapter):
        """Test getting token balance."""
        from kestrel_sovereign.features.wallet.chain_adapters import ChainNetwork, TokenRegistry

        usdc = TokenRegistry.get_token("USDC", ChainNetwork.ETHEREUM_SEPOLIA)
        if usdc:
            # Use a known address (may have 0 balance, that's fine)
            address = "0x742d35Cc6634C0532925a3b844Bc9e7595f3E123"
            balance = await erc20_adapter.get_token_balance(usdc.address, address)
            assert isinstance(balance, Decimal)
            assert balance >= Decimal("0")


class TestTransactionManager:
    """Test TransactionManager orchestration."""

    @pytest.fixture
    def manager(self, tmp_path):
        """Create transaction manager with temp database."""
        from kestrel_sovereign.features.wallet.transaction_manager import TransactionManager

        db_path = str(tmp_path / "tx_audit.db")
        return TransactionManager(audit_db_path=db_path)

    def test_manager_initialization(self, manager):
        """Test manager initializes correctly."""
        assert manager.daily_limit_usd == Decimal("100")

    def test_spending_limit_check(self, manager):
        """Test spending limit checking."""
        # Should allow small amounts
        can_spend, remaining = manager.check_spending_limit(Decimal("10"))
        assert can_spend is True
        assert remaining == Decimal("100")

        # Should block amounts over limit
        can_spend, remaining = manager.check_spending_limit(Decimal("200"))
        assert can_spend is False


class TestStripeOnRamp:
    """Test Stripe on-ramp integration."""

    @pytest.fixture
    def onramp(self, tmp_path):
        """Create on-ramp instance with temp database."""
        from kestrel_sovereign.features.wallet.onramp import StripeOnRamp

        db_path = str(tmp_path / "onramp.db")
        return StripeOnRamp(db_path=db_path)

    @pytest.mark.asyncio
    async def test_create_session_without_stripe(self, onramp):
        """Test session creation falls back to demo mode without Stripe key."""
        session = await onramp.create_session(
            agent_did="did:pkh:eip155:1:0x123",
            wallet_address="0x742d35Cc6634C0532925a3b844Bc9e7595f3E123",
            destination_currency="ETH",
            fiat_amount=Decimal("100"),
        )
        assert session is not None
        assert session.session_id is not None
        assert session.destination_currency == "ETH"
        assert session.redirect_url is not None  # Demo URL

    def test_supported_currencies(self, onramp):
        """Test supported destination currencies."""
        assert "ETH" in onramp.SUPPORTED_CURRENCIES
        assert "MATIC" in onramp.SUPPORTED_CURRENCIES

    @pytest.mark.asyncio
    async def test_unsupported_currency_raises(self, onramp):
        """Test unsupported currency raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported currency"):
            await onramp.create_session(
                agent_did="did:pkh:eip155:1:0x123",
                wallet_address="0x742d35Cc6634C0532925a3b844Bc9e7595f3E123",
                destination_currency="BTC",  # Not supported
            )


class TestWebhookHandler:
    """Test Stripe webhook handler."""

    @pytest.fixture
    def handler(self, tmp_path):
        """Create webhook handler with temp database."""
        from kestrel_sovereign.features.wallet.onramp import StripeOnRamp, StripeWebhookHandler

        db_path = str(tmp_path / "onramp.db")
        onramp = StripeOnRamp(db_path=db_path)
        return StripeWebhookHandler(onramp)

    @pytest.mark.asyncio
    async def test_handle_unknown_event_type(self, handler):
        """Test handling unknown event types."""
        import json

        payload = json.dumps({"type": "unknown.event", "data": {}}).encode()
        result = await handler.handle_webhook(payload, None)
        assert result.success is True
        assert "ignored" in result.message.lower()

    @pytest.mark.asyncio
    async def test_handle_invalid_payload(self, handler):
        """Test handling invalid JSON payload."""
        result = await handler.handle_webhook(b"not-json", None)
        assert result.success is False


# Integration test that requires a funded wallet
@pytest.mark.skipif(
    not os.environ.get("KESTREL_TEST_PRIVATE_KEY"),
    reason="Set KESTREL_TEST_PRIVATE_KEY to run wallet tests",
)
class TestRealTransaction:
    """Real transaction tests (requires funded test wallet)."""

    @pytest.mark.asyncio
    async def test_send_sepolia_eth(self):
        """Test sending ETH on Sepolia (requires funded wallet)."""
        from kestrel_sovereign.features.wallet.chain_adapters import (
            ChainNetwork,
            EVMAdapter,
            TransactionRequest,
        )

        adapter = EVMAdapter(ChainNetwork.ETHEREUM_SEPOLIA)

        # Get private key from environment
        private_key_hex = os.environ.get("KESTREL_TEST_PRIVATE_KEY", "")
        if private_key_hex.startswith("0x"):
            private_key_hex = private_key_hex[2:]
        private_key = bytes.fromhex(private_key_hex)

        # Get sender address
        from_address = adapter.get_address_from_private_key(private_key)

        # Check balance first
        balance = await adapter.get_balance(from_address)
        if balance < Decimal("0.001"):
            pytest.skip(f"Insufficient balance: {balance} ETH")

        # Send tiny amount to self
        request = TransactionRequest(
            to_address=from_address,  # Send to self
            amount=Decimal("0.0001"),  # 0.0001 ETH
            network=ChainNetwork.ETHEREUM_SEPOLIA,
            from_address=from_address,
        )

        result = await adapter.send_transaction(request, private_key)
        assert result.success is True
        assert result.tx_hash is not None
        assert result.tx_hash.startswith("0x")

        print(f"Transaction sent: {result.get_explorer_url()}")
