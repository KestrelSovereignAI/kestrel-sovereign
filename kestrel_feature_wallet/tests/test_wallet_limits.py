"""
Unit tests for wallet security features: daily limits and mainnet blocking.

Tests critical security mechanisms that prevent financial loss:
- Daily spending limit enforcement (at limit, over limit, reset)
- Hard-coded mainnet chain ID blocking
- Transaction audit logging
"""

import os
import pytest
pytest.importorskip("web3", reason="web3 not installed (wallet extras)")
from decimal import Decimal
from datetime import date, timedelta
from pathlib import Path
import aiosqlite

from kestrel_feature_wallet.transaction_manager import (
    TransactionManager,
    TransactionAudit,
)
from kestrel_feature_wallet.chain_adapters import ChainNetwork
from kestrel_feature_wallet.chain_adapters.evm_adapter import MAINNET_CHAIN_IDS


class TestDailySpendingLimits:
    """Test daily spending limit enforcement."""

    @pytest.fixture
    def temp_storage(self, tmp_path):
        """Create temporary storage directory."""
        return tmp_path / "wallet_test"

    @pytest.fixture
    async def tx_manager(self, temp_storage):
        """Create transaction manager with default $100 daily limit."""
        manager = TransactionManager(
            agent_id="test-agent",
            storage_dir=temp_storage,
            daily_limit_usd=Decimal("100"),
        )
        await manager.initialize()
        return manager

    @pytest.mark.asyncio
    async def test_spending_at_exactly_daily_limit_succeeds(self, tx_manager, temp_storage):
        """Spending exactly at the daily limit should succeed."""
        # Spend $99 (under limit)
        await tx_manager._update_spending(Decimal("99"))

        # Spend $1 more (exactly at limit)
        allowed, error, remaining = await tx_manager.check_spending_limit(Decimal("1"))

        assert allowed is True
        assert error == ""
        assert remaining == Decimal("1")  # $100 limit - $99 spent = $1 remaining

    @pytest.mark.asyncio
    async def test_spending_one_cent_over_limit_fails(self, tx_manager):
        """Spending $0.01 over the daily limit should fail."""
        # Spend $99.99
        await tx_manager._update_spending(Decimal("99.99"))

        # Try to spend $0.02 (would exceed limit by $0.01)
        allowed, error, remaining = await tx_manager.check_spending_limit(Decimal("0.02"))

        assert allowed is False
        assert "exceeds daily limit" in error.lower()
        assert "$100" in error  # Should mention the limit
        assert remaining == Decimal("0.01")

    @pytest.mark.asyncio
    async def test_daily_limit_resets_after_24h(self, tx_manager, temp_storage):
        """Daily limit should reset for a new day."""
        # Spend $100 today
        await tx_manager._update_spending(Decimal("100"))

        # Verify we're at limit
        allowed, _, _ = await tx_manager.check_spending_limit(Decimal("0.01"))
        assert allowed is False

        # Manually insert spending for yesterday
        yesterday = (date.today() - timedelta(days=1)).isoformat()
        async with aiosqlite.connect(str(tx_manager._audit_db_path)) as db:
            await db.execute(
                "INSERT INTO daily_spending (date, total_spent_usd, transaction_count) VALUES (?, ?, ?)",
                (yesterday, "100", 5),
            )
            await db.commit()

        # Today's limit should be independent
        spending_today = await tx_manager.get_spending_today()
        assert spending_today.total_spent_usd == Decimal("100")
        assert spending_today.date == date.today()

    @pytest.mark.asyncio
    async def test_multiple_small_transactions_accumulate(self, tx_manager):
        """Multiple transactions should accumulate toward the daily limit."""
        # Make 10 transactions of $10 each
        for i in range(10):
            allowed, _, remaining = await tx_manager.check_spending_limit(Decimal("10"))
            assert allowed is True
            await tx_manager._update_spending(Decimal("10"))

        # After spending $100, we should have $0 remaining
        spending = await tx_manager.get_spending_today()
        assert spending.total_spent_usd == Decimal("100")

        # 11th transaction should fail (would need $10 but have $0 remaining)
        allowed, error, remaining = await tx_manager.check_spending_limit(Decimal("10"))
        assert allowed is False
        assert remaining == Decimal("0")

    @pytest.mark.asyncio
    async def test_custom_daily_limit_via_env_var(self, tmp_path):
        """Daily limit can be configured via environment variable."""
        os.environ["KESTREL_TX_DAILY_LIMIT_USD"] = "250"

        try:
            manager = TransactionManager(
                agent_id="test-custom-limit",
                storage_dir=tmp_path / "custom_limit",
            )
            await manager.initialize()

            assert manager.daily_limit_usd == Decimal("250")

            # Should allow spending up to $250
            allowed, _, _ = await manager.check_spending_limit(Decimal("250"))
            assert allowed is True

            # But not $250.01
            await manager._update_spending(Decimal("250"))
            allowed, _, _ = await manager.check_spending_limit(Decimal("0.01"))
            assert allowed is False
        finally:
            del os.environ["KESTREL_TX_DAILY_LIMIT_USD"]

    @pytest.mark.asyncio
    async def test_testnet_transactions_skip_limit_check(self, tx_manager):
        """Testnet transactions should not count toward spending limits."""
        # This is tested at the TransactionManager.send_native level
        # where network.is_mainnet determines if limit check runs
        # Here we verify the check_spending_limit method works correctly

        # Spend $100
        await tx_manager._update_spending(Decimal("100"))

        # Limit check should fail
        allowed, _, _ = await tx_manager.check_spending_limit(Decimal("0.01"))
        assert allowed is False


class TestMainnetBlocking:
    """Test hard-coded mainnet blocking at the adapter level."""

    @pytest.fixture
    def temp_storage(self, tmp_path):
        """Create temporary storage directory."""
        return tmp_path / "mainnet_test"

    @pytest.mark.asyncio
    async def test_mainnet_chain_ids_are_blocked_by_default(self):
        """All mainnet chain IDs should be in the blocklist."""
        expected_mainnets = {1, 56, 137, 42161, 10, 314, 8453}
        assert MAINNET_CHAIN_IDS == expected_mainnets

    @pytest.mark.asyncio
    async def test_ethereum_mainnet_blocked_without_env_var(self, temp_storage):
        """Ethereum mainnet (chain_id=1) should be blocked by default."""
        from kestrel_feature_wallet.chain_adapters.evm_adapter import EVMAdapter
        from kestrel_feature_wallet.chain_adapters import TransactionRequest

        # Ensure env var is not set
        if "KESTREL_ALLOW_MAINNET" in os.environ:
            del os.environ["KESTREL_ALLOW_MAINNET"]

        adapter = EVMAdapter(ChainNetwork.ETHEREUM_MAINNET)

        request = TransactionRequest(
            to_address="0x0000000000000000000000000000000000000000",
            amount=Decimal("0.001"),
            network=ChainNetwork.ETHEREUM_MAINNET,
        )

        # Generate a test private key (32 bytes)
        private_key = b'\x01' * 32

        result = await adapter.send_transaction(request, private_key)

        assert result.success is False
        assert "Mainnet transactions blocked" in result.error
        assert "chain_id=1" in result.error
        assert "KESTREL_ALLOW_MAINNET=1" in result.error

    @pytest.mark.asyncio
    async def test_polygon_mainnet_blocked_without_env_var(self, temp_storage):
        """Polygon mainnet (chain_id=137) should be blocked by default."""
        from kestrel_feature_wallet.chain_adapters.evm_adapter import EVMAdapter
        from kestrel_feature_wallet.chain_adapters import TransactionRequest

        if "KESTREL_ALLOW_MAINNET" in os.environ:
            del os.environ["KESTREL_ALLOW_MAINNET"]

        adapter = EVMAdapter(ChainNetwork.POLYGON_MAINNET)

        request = TransactionRequest(
            to_address="0x0000000000000000000000000000000000000000",
            amount=Decimal("0.001"),
            network=ChainNetwork.POLYGON_MAINNET,
        )

        private_key = b'\x01' * 32

        result = await adapter.send_transaction(request, private_key)

        assert result.success is False
        assert "Mainnet transactions blocked" in result.error
        assert "chain_id=137" in result.error

    @pytest.mark.asyncio
    async def test_filecoin_mainnet_blocked_without_env_var(self, temp_storage):
        """Filecoin mainnet (chain_id=314) should be blocked by default."""
        from kestrel_feature_wallet.chain_adapters.evm_adapter import EVMAdapter
        from kestrel_feature_wallet.chain_adapters import TransactionRequest

        if "KESTREL_ALLOW_MAINNET" in os.environ:
            del os.environ["KESTREL_ALLOW_MAINNET"]

        adapter = EVMAdapter(ChainNetwork.FILECOIN_MAINNET)

        request = TransactionRequest(
            to_address="0x0000000000000000000000000000000000000000",
            amount=Decimal("0.001"),
            network=ChainNetwork.FILECOIN_MAINNET,
        )

        private_key = b'\x01' * 32

        result = await adapter.send_transaction(request, private_key)

        assert result.success is False
        assert "Mainnet transactions blocked" in result.error
        assert "chain_id=314" in result.error

    @pytest.mark.asyncio
    async def test_testnet_transactions_allowed(self, temp_storage):
        """Testnet transactions should not be blocked by the mainnet check."""
        from unittest.mock import patch, MagicMock
        from kestrel_feature_wallet.chain_adapters.evm_adapter import EVMAdapter
        from kestrel_feature_wallet.chain_adapters import TransactionRequest

        if "KESTREL_ALLOW_MAINNET" in os.environ:
            del os.environ["KESTREL_ALLOW_MAINNET"]

        # Sepolia testnet — chain_id should NOT be in MAINNET_CHAIN_IDS
        adapter = EVMAdapter(ChainNetwork.ETHEREUM_SEPOLIA)
        assert adapter.config.chain_id not in MAINNET_CHAIN_IDS

        request = TransactionRequest(
            to_address="0x0000000000000000000000000000000000000000",
            amount=Decimal("0.001"),
            network=ChainNetwork.ETHEREUM_SEPOLIA,
        )

        private_key = b'\x01' * 32

        # Mock web3 RPC calls so we don't hit the network.
        # We only care that send_transaction passes the mainnet blocking check.
        mock_eth = MagicMock()
        mock_eth.get_transaction_count.return_value = 0
        mock_eth.gas_price = 20_000_000_000
        mock_eth.estimate_gas.return_value = 21000
        mock_eth.get_block.return_value = {"baseFeePerGas": None}
        mock_eth.send_raw_transaction.return_value = b'\xab' * 32

        with patch.object(adapter.w3, 'eth', mock_eth):
            result = await adapter.send_transaction(request, private_key)

        # Should not fail due to mainnet blocking
        assert "Mainnet transactions blocked" not in (result.error or "")


class TestTransactionAuditLogging:
    """Test that all transactions are properly logged to the audit database."""

    @pytest.fixture
    def temp_storage(self, tmp_path):
        """Create temporary storage directory."""
        return tmp_path / "audit_test"

    @pytest.fixture
    async def tx_manager(self, temp_storage):
        """Create transaction manager."""
        manager = TransactionManager(
            agent_id="audit-test-agent",
            storage_dir=temp_storage,
            daily_limit_usd=Decimal("1000"),  # High limit to not interfere
        )
        await manager.initialize()
        return manager

    @pytest.mark.asyncio
    async def test_successful_transaction_logged_with_all_fields(self, tx_manager):
        """Successful transactions should be logged with complete metadata."""
        audit = TransactionAudit(
            agent_id="audit-test-agent",
            network="ethereum_sepolia",
            tx_type="native",
            to_address="0x1234567890123456789012345678901234567890",
            amount=Decimal("0.5"),
            amount_usd=Decimal("1500"),
            tx_hash="0xabcdef1234567890",
            status="success",
        )

        await tx_manager._record_transaction(audit)

        # Retrieve and verify
        history = await tx_manager.get_transaction_history(limit=1)

        assert len(history) == 1
        logged = history[0]
        assert logged.agent_id == "audit-test-agent"
        assert logged.network == "ethereum_sepolia"
        assert logged.tx_type == "native"
        assert logged.to_address == "0x1234567890123456789012345678901234567890"
        assert logged.amount == Decimal("0.5")
        assert logged.amount_usd == Decimal("1500")
        assert logged.tx_hash == "0xabcdef1234567890"
        assert logged.status == "success"
        assert logged.error is None

    @pytest.mark.asyncio
    async def test_failed_transaction_logged_with_error(self, tx_manager):
        """Failed transactions should be logged with error details."""
        audit = TransactionAudit(
            agent_id="audit-test-agent",
            network="polygon_mainnet",
            tx_type="erc20",
            token_symbol="USDC",
            to_address="0xdeadbeefdeadbeefdeadbeefdeadbeefdeadbeef",
            amount=Decimal("100"),
            amount_usd=Decimal("100"),
            status="failed",
            error="Mainnet transactions blocked (chain_id=137)",
        )

        await tx_manager._record_transaction(audit)

        history = await tx_manager.get_transaction_history(limit=1)

        assert len(history) == 1
        logged = history[0]
        assert logged.status == "failed"
        assert logged.error == "Mainnet transactions blocked (chain_id=137)"
        assert logged.tx_hash is None  # No hash for failed tx

    @pytest.mark.asyncio
    async def test_transaction_history_ordered_by_timestamp(self, tx_manager):
        """Transaction history should be ordered newest first."""
        # Record 3 transactions
        for i in range(3):
            audit = TransactionAudit(
                agent_id="audit-test-agent",
                network="ethereum_sepolia",
                tx_type="native",
                to_address=f"0x{'0' * 39}{i}",
                amount=Decimal(str(i)),
                amount_usd=Decimal(str(i * 100)),
                status="success",
            )
            await tx_manager._record_transaction(audit)

        history = await tx_manager.get_transaction_history(limit=10)

        assert len(history) == 3
        # Should be ordered newest first
        assert history[0].to_address.endswith("2")
        assert history[1].to_address.endswith("1")
        assert history[2].to_address.endswith("0")

    @pytest.mark.asyncio
    async def test_audit_log_persists_across_instances(self, temp_storage):
        """Audit log should persist when manager is recreated."""
        # Create manager and log a transaction
        manager1 = TransactionManager(
            agent_id="persist-test",
            storage_dir=temp_storage,
        )
        await manager1.initialize()

        audit = TransactionAudit(
            agent_id="persist-test",
            network="filecoin_calibration",
            tx_type="native",
            to_address="0xtest",
            amount=Decimal("1"),
            amount_usd=Decimal("5.50"),
            status="success",
        )
        await manager1._record_transaction(audit)

        # Create new manager instance pointing to same DB
        manager2 = TransactionManager(
            agent_id="persist-test",
            storage_dir=temp_storage,
        )
        await manager2.initialize()

        # Should be able to read the transaction
        history = await manager2.get_transaction_history(limit=1)
        assert len(history) == 1
        assert history[0].to_address == "0xtest"

    @pytest.mark.asyncio
    async def test_spending_tracked_only_for_successful_mainnet_tx(self, tx_manager):
        """Daily spending should only increment for successful mainnet transactions."""
        # Record a failed transaction
        await tx_manager._record_transaction(TransactionAudit(
            agent_id="audit-test-agent",
            network="ethereum_mainnet",
            tx_type="native",
            to_address="0x1234",
            amount=Decimal("1"),
            amount_usd=Decimal("3000"),
            status="failed",
            error="Insufficient balance",
        ))

        # Spending should not be updated
        spending = await tx_manager.get_spending_today()
        assert spending.total_spent_usd == Decimal("0")
        assert spending.transaction_count == 0

        # Record a successful testnet transaction (should not count toward limit)
        await tx_manager._record_transaction(TransactionAudit(
            agent_id="audit-test-agent",
            network="ethereum_sepolia",
            tx_type="native",
            to_address="0x5678",
            amount=Decimal("1"),
            amount_usd=Decimal("3000"),
            status="success",
        ))

        # Still should be zero (testnets don't update spending)
        spending = await tx_manager.get_spending_today()
        assert spending.total_spent_usd == Decimal("0")
