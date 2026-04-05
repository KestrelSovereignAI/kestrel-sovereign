import pytest  # noqa: E402
pytest.importorskip('kestrel_feature_wallet', reason='Extracted package not installed')

"""
End-to-end tests for Filecoin Calibration testnet integration.

Tests cover:
- Address generation with secure key storage
- On-chain balance queries
- Balance sync between internal wallet and on-chain
- WalletFeature commands (!wallet-generate-address, !wallet-sync)

Network tests are opt-in via RUN_NETWORK_TESTS=true environment variable.
"""

import os
import pytest
import tempfile
from decimal import Decimal
from pathlib import Path

from kestrel_feature_wallet import (
    WalletAgent,
    WalletFeature,
    Currency,
    FilecoinTestnetAdapter,
    FilecoinKeyManager,
)


# Skip network tests by default - they require real network access
NETWORK_TESTS_ENABLED = os.environ.get("RUN_NETWORK_TESTS") == "true"


class TestFilecoinKeyManager:
    """Tests for Filecoin address generation."""

    @pytest.fixture
    def temp_storage_dir(self):
        """Create temporary storage directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def key_manager(self, temp_storage_dir, monkeypatch):
        """Create key manager with temp storage and test encryption key."""
        # Set a test encryption key so secure storage works
        monkeypatch.setenv("KESTREL_DATA_KEY", "test-key-for-filecoin-tests-only")
        return FilecoinKeyManager(storage_dir=temp_storage_dir)

    @pytest.mark.asyncio
    async def test_generate_address_format(self, key_manager):
        """Generated address should have correct format."""
        address, pub_key = await key_manager.generate_address("test_agent_1")

        # Should be testnet format (t1...)
        assert address.startswith("t1"), f"Expected t1 prefix, got {address}"

        # Should have reasonable length (40-50 chars)
        assert 30 < len(address) < 60, f"Unexpected address length: {len(address)}"

        # Public key should be uncompressed (65 bytes)
        assert len(pub_key) == 65, f"Expected 65 byte pubkey, got {len(pub_key)}"
        assert pub_key[0] == 0x04, "Public key should start with 04 (uncompressed)"

    @pytest.mark.asyncio
    async def test_generate_address_uniqueness(self, key_manager):
        """Each agent should get unique address."""
        addr1, _ = await key_manager.generate_address("agent_a")
        addr2, _ = await key_manager.generate_address("agent_b")

        assert addr1 != addr2, "Different agents should have different addresses"

    @pytest.mark.asyncio
    async def test_generate_address_raises_if_exists(self, key_manager):
        """Should raise if agent already has address (requires KESTREL_DATA_KEY)."""
        # Skip if no secure storage (can't track existing addresses)
        if not key_manager._secure_storage:
            pytest.skip("Requires KESTREL_DATA_KEY for duplicate detection")

        await key_manager.generate_address("duplicate_agent")

        with pytest.raises(ValueError, match="already has"):
            await key_manager.generate_address("duplicate_agent")

    @pytest.mark.asyncio
    async def test_get_address_after_generate(self, key_manager):
        """Should be able to retrieve address after generation."""
        generated_addr, _ = await key_manager.generate_address("retrievable_agent")

        retrieved_addr = key_manager.get_address("retrievable_agent")

        assert retrieved_addr == generated_addr

    def test_has_address_false_initially(self, key_manager):
        """Should return False for unknown agent."""
        assert not key_manager.has_address("nonexistent")

    @pytest.mark.asyncio
    async def test_has_address_true_after_generate(self, key_manager):
        """Should return True after generating."""
        await key_manager.generate_address("tracked_agent")
        assert key_manager.has_address("tracked_agent")

    def test_explorer_url_format(self, key_manager):
        """Explorer URL should point to calibration filfox."""
        url = key_manager.get_explorer_url("t1abc123")
        assert "calibration.filfox.info" in url
        assert "t1abc123" in url

    def test_faucet_url_available(self, key_manager):
        """Faucet URL should be available for testnet."""
        url = key_manager.get_faucet_url()
        assert url is not None
        assert "faucet.calibnet" in url or "chainsafe" in url


class TestFilecoinTestnetAdapter:
    """Tests for Filecoin RPC adapter."""

    @pytest.fixture
    def adapter(self):
        return FilecoinTestnetAdapter()

    def test_default_network_is_calibration(self, adapter):
        """Should default to calibration testnet."""
        assert adapter.network.value == "calibration"

    def test_rpc_url_is_calibration(self, adapter):
        """RPC URL should be calibration endpoint."""
        assert "calibration" in adapter.rpc_url

    def test_mainnet_blocked_without_env(self):
        """Should raise error when trying to use mainnet."""
        with pytest.raises(ValueError, match="Mainnet is disabled"):
            FilecoinTestnetAdapter(network="mainnet")

    @pytest.mark.skipif(not NETWORK_TESTS_ENABLED, reason="Network tests disabled")
    @pytest.mark.asyncio
    async def test_is_connected(self, adapter):
        """Should connect to calibration testnet."""
        try:
            connected = await adapter.is_connected()
            assert connected, "Should be able to connect to testnet"
        finally:
            await adapter.close()

    @pytest.mark.skipif(not NETWORK_TESTS_ENABLED, reason="Network tests disabled")
    @pytest.mark.asyncio
    async def test_get_chain_head(self, adapter):
        """Should get current chain head."""
        try:
            head = await adapter.get_chain_head()
            assert "Cids" in head or "Height" in head
        finally:
            await adapter.close()

    @pytest.mark.skipif(not NETWORK_TESTS_ENABLED, reason="Network tests disabled")
    @pytest.mark.asyncio
    async def test_get_balance_known_address(self, adapter):
        """Should get balance for a known testnet address."""
        try:
            # Use the known faucet address
            faucet_addr = "t1d2xrzcslx7xlbbylc5c3d5lvandqw4iwl6epxba"
            balance = await adapter.get_balance(faucet_addr)

            assert isinstance(balance, Decimal)
            assert balance >= Decimal("0")
        finally:
            await adapter.close()


class TestWalletFilecoinIntegration:
    """Tests for WalletAgent integration with Filecoin."""

    @pytest.fixture
    def temp_db_path(self):
        """Create temporary database."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            yield f.name
        # Cleanup
        try:
            os.unlink(f.name)
        except Exception:
            pass

    @pytest.fixture
    async def wallet(self, temp_db_path):
        """Create initialized wallet."""
        wallet = WalletAgent(
            agent_id="test_fil_wallet",
            initial_balance=Decimal("100.0"),
            db_path=temp_db_path
        )
        await wallet.initialize()
        yield wallet

    @pytest.mark.asyncio
    async def test_filecoin_address_initially_none(self, wallet):
        """Filecoin address should be None initially."""
        assert wallet.filecoin_address is None

    @pytest.mark.asyncio
    async def test_set_filecoin_address(self, wallet):
        """Should be able to set Filecoin address."""
        test_addr = "t1abc123xyz"
        await wallet.set_filecoin_address(test_addr)

        assert wallet.filecoin_address == test_addr

    @pytest.mark.asyncio
    async def test_filecoin_address_in_status(self, wallet):
        """Status should include Filecoin address when set."""
        # Without address
        status = wallet.get_status()
        assert "filecoin_address" not in status

        # With address
        await wallet.set_filecoin_address("t1testaddress")
        status = wallet.get_status()
        assert status["filecoin_address"] == "t1testaddress"

    @pytest.mark.asyncio
    async def test_sync_without_address_fails(self, wallet):
        """Sync should fail without address configured."""
        success = await wallet.sync_on_chain_balance()
        assert not success

    @pytest.mark.asyncio
    async def test_get_on_chain_balance_without_address(self, wallet):
        """Should return None without address."""
        balance = await wallet.get_on_chain_balance()
        assert balance is None

    @pytest.mark.skipif(not NETWORK_TESTS_ENABLED, reason="Network tests disabled")
    @pytest.mark.asyncio
    async def test_get_on_chain_balance_with_address(self, wallet):
        """Should query on-chain balance when address is set."""
        # Use a known address with balance
        await wallet.set_filecoin_address("t1d2xrzcslx7xlbbylc5c3d5lvandqw4iwl6epxba")

        balance = await wallet.get_on_chain_balance()
        assert balance is not None
        assert isinstance(balance, Decimal)


class TestWalletFeatureFilecoinCommands:
    """Tests for WalletFeature Filecoin commands."""

    @pytest.fixture
    def temp_db_path(self):
        """Create temporary database."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            yield f.name
        try:
            os.unlink(f.name)
        except Exception:
            pass

    @pytest.fixture
    async def wallet_feature(self, temp_db_path):
        """Create WalletFeature with mock agent."""

        class MockAgent:
            def __init__(self):
                self.did = "test_cmd_agent"
                self.wallet = None
                self.db_path = None

        agent = MockAgent()
        agent.db_path = temp_db_path

        # Create minimal storage mock
        class Storage:
            def __init__(self, path):
                self.db_path = path
        agent.storage = Storage(temp_db_path)

        feature = WalletFeature(agent)
        await feature.initialize()

        yield feature

    @pytest.mark.asyncio
    async def test_wallet_address_without_address(self, wallet_feature):
        """Should prompt to generate address."""
        result = await wallet_feature.wallet_address()
        assert "No Filecoin Address" in result
        assert "!wallet-generate-address" in result

    @pytest.mark.asyncio
    async def test_wallet_sync_without_address(self, wallet_feature):
        """Should prompt to generate address first."""
        result = await wallet_feature.wallet_sync()
        assert "No Filecoin Address" in result
        assert "!wallet-generate-address" in result


class TestFilecoinAddressPersistence:
    """Tests for Filecoin address persistence across restarts."""

    @pytest.fixture
    def temp_db_path(self):
        """Create temporary database path (not file)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir) / "test_wallet.db"

    @pytest.mark.asyncio
    async def test_address_persists_across_restarts(self, temp_db_path):
        """Filecoin address should survive wallet restart."""
        test_address = "t1persistent123abc"

        # Create and initialize first wallet
        wallet1 = WalletAgent(
            agent_id="persist_test",
            db_path=str(temp_db_path)
        )
        await wallet1.initialize()
        await wallet1.set_filecoin_address(test_address)

        # Verify address is set
        assert wallet1.filecoin_address == test_address

        # Create new wallet instance (simulating restart)
        wallet2 = WalletAgent(
            agent_id="persist_test",
            db_path=str(temp_db_path)
        )
        await wallet2.initialize()

        # Address should be loaded from DB
        assert wallet2.filecoin_address == test_address


class TestEconomicGateMethods:
    """Tests for is_paid_tier and has_revenue_share."""

    @pytest.fixture
    def temp_db_path(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            yield f.name
        try:
            os.unlink(f.name)
        except Exception:
            pass

    @pytest.mark.asyncio
    async def test_is_paid_tier_true_when_balance_high(self, temp_db_path):
        """Should return True when balance > $10."""

        class MockAgent:
            def __init__(self):
                self.did = "paid_tier_test"
                self.wallet = None
                self.storage = type('Storage', (), {'db_path': temp_db_path})()

        agent = MockAgent()
        feature = WalletFeature(agent)
        await feature.initialize()

        # Default balance is 100 FIL @ $5.50 = $550
        assert feature.is_paid_tier() is True

    @pytest.mark.asyncio
    async def test_is_paid_tier_false_when_balance_low(self, temp_db_path):
        """Should return False when balance < $10."""

        class MockAgent:
            def __init__(self):
                self.did = "poor_tier_test"
                self.wallet = None
                self.storage = type('Storage', (), {'db_path': temp_db_path})()

        agent = MockAgent()
        feature = WalletFeature(agent)
        await feature.initialize()

        # Drain the wallet below $10
        # Default: 90 FIL main + 10 FIL audit = 100 FIL at $5.50 = $550
        # Need to get under $10 USD = 1.8 FIL total
        # Transfer 89 FIL from main (leaves 1 FIL main)
        await feature.wallet.transfer(Decimal("89"), "drain", Currency.FIL)
        # Deduct all 10 FIL audit (leaves 0 audit)
        await feature.wallet.deduct_audit_fee(Decimal("10"), "drain audit", Currency.FIL)

        # Now should be 1 FIL total = $5.50 < $10
        assert feature.wallet.get_total_balance_usd() < Decimal("10")
        assert feature.is_paid_tier() is False

    @pytest.mark.asyncio
    async def test_has_revenue_share_false_by_default(self, temp_db_path):
        """Should return False when no revenue share configured."""

        class MockAgent:
            def __init__(self):
                self.did = "no_revenue_test"
                self.wallet = None
                self.metadata = None
                self.storage = type('Storage', (), {'db_path': temp_db_path})()

        agent = MockAgent()
        feature = WalletFeature(agent)
        await feature.initialize()

        assert feature.has_revenue_share() is False

    @pytest.mark.asyncio
    async def test_has_revenue_share_true_when_configured(self, temp_db_path):
        """Should return True when revenue share address is set."""

        class MockAgent:
            def __init__(self):
                self.did = "revenue_test"
                self.wallet = None
                self.metadata = {"revenue_share_address": "t1abc123"}
                self.storage = type('Storage', (), {'db_path': temp_db_path})()

        agent = MockAgent()
        feature = WalletFeature(agent)
        await feature.initialize()

        assert feature.has_revenue_share() is True
