"""
WalletFeature Integration Tests

Real integration tests using actual SQLite persistence.
Tests wallet operations end-to-end without mocking core functionality.

Run with: pytest tests/integration/test_wallet_feature_e2e.py -v
"""

import asyncio
import os
import tempfile
from decimal import Decimal
from pathlib import Path
from typing import AsyncGenerator
from unittest.mock import AsyncMock, MagicMock

import pytest

# Import the classes we're testing
from kestrel_sovereign.features.wallet.wallet_feature import WalletFeature
from kestrel_sovereign.features.wallet.feature import WalletAgent, Currency


# =============================================================================
# Fixtures
# =============================================================================

class MockAgent:
    """Mock agent for testing WalletFeature in isolation."""

    def __init__(self, db_path: str, agent_id: str = "test_agent"):
        self.did = agent_id
        self.db_path = db_path
        self.storage = MagicMock()
        self.storage.db_path = db_path
        self.wallet = None  # Will be set by WalletFeature.initialize()


@pytest.fixture
def temp_db_path() -> str:
    """Create a temporary database path for testing."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        yield f.name
    # Cleanup
    try:
        os.unlink(f.name)
    except OSError:
        pass


@pytest.fixture
async def mock_agent(temp_db_path: str) -> MockAgent:
    """Create a mock agent with a temporary database."""
    return MockAgent(db_path=temp_db_path, agent_id="test_wallet_agent")


@pytest.fixture
async def wallet_feature(mock_agent: MockAgent) -> AsyncGenerator[WalletFeature, None]:
    """Create and initialize a WalletFeature for testing."""
    feature = WalletFeature(mock_agent)
    await feature.initialize()
    yield feature
    await feature.shutdown()


# =============================================================================
# Initialization Tests
# =============================================================================

class TestWalletInitialization:
    """Test wallet initialization and persistence."""

    @pytest.mark.asyncio
    async def test_wallet_initializes_with_default_balance(self, wallet_feature: WalletFeature):
        """Wallet should initialize with 100 FIL split 90/10."""
        assert wallet_feature.wallet is not None

        # Check FIL balance (default 100 FIL, split 90/10)
        main = wallet_feature.wallet.get_balance(Currency.FIL, "main")
        audit = wallet_feature.wallet.get_balance(Currency.FIL, "audit")

        assert main == Decimal("90.0"), f"Expected 90.0 main, got {main}"
        assert audit == Decimal("10.0"), f"Expected 10.0 audit, got {audit}"

    @pytest.mark.asyncio
    async def test_wallet_attached_to_agent(self, wallet_feature: WalletFeature, mock_agent: MockAgent):
        """Wallet should be attached to agent for other features to use."""
        assert mock_agent.wallet is not None
        assert mock_agent.wallet is wallet_feature.wallet

    @pytest.mark.asyncio
    async def test_wallet_persists_across_restarts(self, temp_db_path: str):
        """Wallet state should persist to database and reload."""
        # Create first instance and make changes
        agent1 = MockAgent(db_path=temp_db_path, agent_id="persist_test")
        feature1 = WalletFeature(agent1)
        await feature1.initialize()

        # Make a transfer
        await feature1.wallet.transfer(Decimal("10.0"), "test transfer", Currency.FIL)
        original_balance = feature1.wallet.get_balance(Currency.FIL, "main")
        await feature1.shutdown()

        # Create second instance - should load persisted state
        agent2 = MockAgent(db_path=temp_db_path, agent_id="persist_test")
        feature2 = WalletFeature(agent2)
        await feature2.initialize()

        reloaded_balance = feature2.wallet.get_balance(Currency.FIL, "main")
        assert reloaded_balance == original_balance, (
            f"Expected {original_balance}, got {reloaded_balance}"
        )
        await feature2.shutdown()


# =============================================================================
# Balance Tool Tests
# =============================================================================

class TestWalletBalanceTool:
    """Test the wallet_balance tool."""

    @pytest.mark.asyncio
    async def test_balance_all_currencies(self, wallet_feature: WalletFeature):
        """!wallet-balance should show all currency balances."""
        result = await wallet_feature.wallet_balance("all")

        assert "Wallet Balances" in result
        assert "FIL" in result
        assert "90" in result  # Main balance
        assert "10" in result  # Audit balance
        assert "Total USD Value" in result

    @pytest.mark.asyncio
    async def test_balance_specific_currency(self, wallet_feature: WalletFeature):
        """!wallet-balance FIL should show only FIL balance."""
        result = await wallet_feature.wallet_balance("FIL")

        assert "FIL Balance" in result
        assert "Main: 90" in result
        assert "Audit: 10" in result

    @pytest.mark.asyncio
    async def test_balance_invalid_currency(self, wallet_feature: WalletFeature):
        """Invalid currency should return error message."""
        result = await wallet_feature.wallet_balance("INVALID")

        assert "Unknown currency" in result
        assert "FIL, USDC, USDT" in result


# =============================================================================
# Transfer Tool Tests
# =============================================================================

class TestWalletTransferTool:
    """Test the wallet_transfer tool."""

    @pytest.mark.asyncio
    async def test_transfer_success(self, wallet_feature: WalletFeature):
        """Valid transfer should succeed and update balance."""
        result = await wallet_feature.wallet_transfer("10.5", "FIL", "test memo")

        assert "Transfer Successful" in result
        assert "10.5 FIL" in result
        assert "test memo" in result

        # Verify balance updated
        balance = wallet_feature.wallet.get_balance(Currency.FIL, "main")
        assert balance == Decimal("79.5"), f"Expected 79.5, got {balance}"

    @pytest.mark.asyncio
    async def test_transfer_insufficient_funds(self, wallet_feature: WalletFeature):
        """Transfer should fail if insufficient funds."""
        result = await wallet_feature.wallet_transfer("1000.0", "FIL", "too much")

        assert "Insufficient funds" in result
        assert "Have 90" in result

    @pytest.mark.asyncio
    async def test_transfer_invalid_amount(self, wallet_feature: WalletFeature):
        """Invalid amount should return error."""
        result = await wallet_feature.wallet_transfer("not-a-number", "FIL")

        assert "Invalid amount" in result

    @pytest.mark.asyncio
    async def test_transfer_negative_amount(self, wallet_feature: WalletFeature):
        """Negative amount should return error."""
        result = await wallet_feature.wallet_transfer("-10", "FIL")

        assert "must be positive" in result

    @pytest.mark.asyncio
    async def test_transfer_records_history(self, wallet_feature: WalletFeature):
        """Transfer should be recorded in transaction history."""
        await wallet_feature.wallet_transfer("5.0", "FIL", "history test")

        history = wallet_feature.wallet.transaction_history
        assert len(history) > 0

        last_tx = history[-1]
        assert last_tx["memo"] == "history test"
        assert Decimal(last_tx["amount"]) == Decimal("5.0")


# =============================================================================
# Deposit Tool Tests
# =============================================================================

class TestWalletDepositTool:
    """Test the wallet_deposit tool."""

    @pytest.mark.asyncio
    async def test_deposit_splits_90_10(self, wallet_feature: WalletFeature):
        """Deposit should split 90% to main, 10% to audit."""
        result = await wallet_feature.wallet_deposit("100.0", "FIL", "test deposit")

        assert "Deposit Recorded" in result
        assert "Main (90%)" in result  # Main portion
        assert "Audit (10%)" in result  # Audit portion

        # Verify balances updated (original 90/10 + new 90/10)
        main = wallet_feature.wallet.get_balance(Currency.FIL, "main")
        audit = wallet_feature.wallet.get_balance(Currency.FIL, "audit")

        assert main == Decimal("180.0"), f"Expected 180.0 main, got {main}"
        assert audit == Decimal("20.0"), f"Expected 20.0 audit, got {audit}"

    @pytest.mark.asyncio
    async def test_deposit_usdc(self, wallet_feature: WalletFeature):
        """Deposit should work for USDC."""
        result = await wallet_feature.wallet_deposit("50.0", "USDC", "usdc deposit")

        assert "Deposit Recorded" in result
        assert "USDC" in result

        main = wallet_feature.wallet.get_balance(Currency.USDC, "main")
        assert main == Decimal("45.0"), f"Expected 45.0 USDC, got {main}"


# =============================================================================
# History Tool Tests
# =============================================================================

class TestWalletHistoryTool:
    """Test the wallet_history tool."""

    @pytest.mark.asyncio
    async def test_history_empty(self, wallet_feature: WalletFeature):
        """Empty history should show appropriate message."""
        result = await wallet_feature.wallet_history(10)

        assert "No transactions" in result

    @pytest.mark.asyncio
    async def test_history_shows_transactions(self, wallet_feature: WalletFeature):
        """History should show recent transactions."""
        # Make some transactions
        await wallet_feature.wallet_transfer("5.0", "FIL", "tx1")
        await wallet_feature.wallet_transfer("3.0", "FIL", "tx2")

        result = await wallet_feature.wallet_history(10)

        assert "Transaction History" in result
        assert "tx1" in result
        assert "tx2" in result

    @pytest.mark.asyncio
    async def test_history_respects_limit(self, wallet_feature: WalletFeature):
        """History should respect the limit parameter."""
        # Make 5 transactions
        for i in range(5):
            await wallet_feature.wallet_transfer("1.0", "FIL", f"tx{i}")

        result = await wallet_feature.wallet_history(2)

        # Should only show 2 transactions
        assert "last 2" in result


# =============================================================================
# Status Tool Tests
# =============================================================================

class TestWalletStatusTool:
    """Test the wallet_status tool."""

    @pytest.mark.asyncio
    async def test_status_shows_complete_info(self, wallet_feature: WalletFeature):
        """Status should show comprehensive wallet information."""
        result = await wallet_feature.wallet_status()

        assert "Wallet Status" in result
        assert "Agent ID" in result
        assert "Total USD Value" in result
        assert "Cryostasis" in result
        assert "Balances" in result
        assert "Exchange Rates" in result

    @pytest.mark.asyncio
    async def test_status_shows_healthy_cryostasis(self, wallet_feature: WalletFeature):
        """Status should show healthy when above threshold."""
        result = await wallet_feature.wallet_status()

        assert "Healthy" in result or "CRYOSTASIS WARNING" not in result


# =============================================================================
# Exchange Rate Tool Tests
# =============================================================================

class TestWalletExchangeRatesTool:
    """Test the wallet_exchange_rates tool."""

    @pytest.mark.asyncio
    async def test_view_all_rates(self, wallet_feature: WalletFeature):
        """Should display all exchange rates."""
        result = await wallet_feature.wallet_exchange_rates()

        assert "Exchange Rates" in result
        assert "FIL" in result
        assert "USDC" in result
        assert "USDT" in result

    @pytest.mark.asyncio
    async def test_view_single_rate(self, wallet_feature: WalletFeature):
        """Should display rate for specific currency."""
        result = await wallet_feature.wallet_exchange_rates("FIL")

        assert "FIL" in result
        assert "$" in result

    @pytest.mark.asyncio
    async def test_update_rate(self, wallet_feature: WalletFeature):
        """Should update exchange rate."""
        result = await wallet_feature.wallet_exchange_rates("FIL", "10.0")

        assert "Exchange Rate Updated" in result
        assert "FIL" in result
        assert "10.0" in result

        # Verify rate was updated
        new_rate = wallet_feature.wallet._exchange_rates[Currency.FIL]
        assert new_rate == Decimal("10.0")


# =============================================================================
# Multi-Currency Tests
# =============================================================================

class TestMultiCurrency:
    """Test multi-currency support."""

    @pytest.mark.asyncio
    async def test_transfer_different_currencies(self, wallet_feature: WalletFeature):
        """Should handle transfers in different currencies."""
        # Deposit USDC
        await wallet_feature.wallet_deposit("100.0", "USDC")

        # Transfer USDC
        result = await wallet_feature.wallet_transfer("10.0", "USDC", "usdc transfer")

        assert "Transfer Successful" in result
        assert "USDC" in result

    @pytest.mark.asyncio
    async def test_total_usd_value(self, wallet_feature: WalletFeature):
        """Total USD value should aggregate all currencies."""
        # Add some USDC (1:1 with USD)
        await wallet_feature.wallet_deposit("50.0", "USDC")

        status = wallet_feature.wallet.get_status()
        total_usd = Decimal(status["total_usd"])

        # Should include FIL (100 * 5.5 = 550) + USDC (50 * 1 = 50) = 600
        assert total_usd == Decimal("600.0"), f"Expected 600.0, got {total_usd}"


# =============================================================================
# Cryostasis Tests
# =============================================================================

class TestCryostasis:
    """Test cryostasis threshold monitoring."""

    @pytest.mark.asyncio
    async def test_cryostasis_warning_when_low(self, wallet_feature: WalletFeature):
        """Status should warn when below cryostasis threshold."""
        # Set a very high threshold
        wallet_feature.wallet.set_cryostasis_threshold(Decimal("10000.0"))

        result = await wallet_feature.wallet_status()

        assert "CRYOSTASIS WARNING" in result

    @pytest.mark.asyncio
    async def test_runway_estimate(self, wallet_feature: WalletFeature):
        """Should estimate runway in months."""
        # With default 100 FIL at $5.50 = $550, monthly cost $50
        months = wallet_feature.wallet.get_runway_estimate(Decimal("50.0"))

        # Should be about 10-11 months (550 - threshold / 50)
        assert months is not None
        assert months > 0


# =============================================================================
# Edge Cases
# =============================================================================

class TestEdgeCases:
    """Test edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_wallet_not_initialized_error(self, mock_agent: MockAgent):
        """Tools should handle uninitialized wallet gracefully."""
        feature = WalletFeature(mock_agent)
        # Don't call initialize()

        result = await feature.wallet_balance()
        assert "Wallet not initialized" in result

    @pytest.mark.asyncio
    async def test_zero_transfer(self, wallet_feature: WalletFeature):
        """Zero transfer should fail."""
        result = await wallet_feature.wallet_transfer("0", "FIL")

        assert "must be positive" in result

    @pytest.mark.asyncio
    async def test_very_small_amounts(self, wallet_feature: WalletFeature):
        """Should handle very small amounts correctly."""
        result = await wallet_feature.wallet_transfer("0.000001", "FIL", "tiny")

        assert "Transfer Successful" in result


# =============================================================================
# Filecoin Testnet Adapter Tests (when available)
# =============================================================================

class TestFilecoinTestnetAdapter:
    """Test Filecoin testnet adapter (network tests skipped by default)."""

    @pytest.mark.asyncio
    async def test_adapter_initialization(self):
        """Adapter should initialize with default calibration network."""
        from kestrel_sovereign.features.wallet.filecoin_testnet import FilecoinTestnetAdapter, FilecoinNetwork

        adapter = FilecoinTestnetAdapter()
        assert adapter.network == FilecoinNetwork.CALIBRATION
        assert "calibration" in adapter.rpc_url
        await adapter.close()

    @pytest.mark.asyncio
    async def test_mainnet_blocked_by_default(self):
        """Mainnet should be blocked unless explicitly enabled."""
        from kestrel_sovereign.features.wallet.filecoin_testnet import FilecoinTestnetAdapter

        # Clear any existing env var
        os.environ.pop("FILECOIN_MAINNET_ENABLED", None)

        with pytest.raises(ValueError) as exc_info:
            FilecoinTestnetAdapter(network="mainnet")

        assert "Mainnet is disabled" in str(exc_info.value)

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        os.environ.get("RUN_NETWORK_TESTS") != "true",
        reason="Network tests disabled. Set RUN_NETWORK_TESTS=true to enable."
    )
    @pytest.mark.asyncio
    async def test_real_balance_check(self):
        """Test real balance check on Calibration testnet."""
        from kestrel_sovereign.features.wallet.filecoin_testnet import FilecoinTestnetAdapter

        adapter = FilecoinTestnetAdapter()

        try:
            # Use a known faucet address for testing
            test_address = "t1d2xrzcslx7xlbbylc5c3d5lvandqw4iwl6epxba"
            balance = await adapter.get_balance(test_address)

            # Balance should be a Decimal >= 0
            assert isinstance(balance, Decimal)
            assert balance >= 0
        finally:
            await adapter.close()

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        os.environ.get("RUN_NETWORK_TESTS") != "true",
        reason="Network tests disabled"
    )
    @pytest.mark.asyncio
    async def test_network_connectivity(self):
        """Test that we can connect to the testnet RPC."""
        from kestrel_sovereign.features.wallet.filecoin_testnet import FilecoinTestnetAdapter

        adapter = FilecoinTestnetAdapter()

        try:
            connected = await adapter.is_connected()
            assert connected, "Should be able to connect to Calibration testnet"

            head = await adapter.get_chain_head()
            assert "Height" in head or "Cids" in head
        finally:
            await adapter.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
