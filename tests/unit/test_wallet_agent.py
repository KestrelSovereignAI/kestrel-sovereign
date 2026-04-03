import pytest
import pytest_asyncio
from decimal import Decimal
from kestrel_feature_wallet import WalletAgent, Currency


@pytest.mark.asyncio
async def test_wallet_agent_demonstration():
    """
    Tests and demonstrates the WalletAgent's functionality,
    providing a visible artifact of its operations.
    """
    print("\n--- WalletAgent Demonstration ---")

    # 1. Initialize the WalletAgent with 200 FIL
    initial_balance = Decimal('200.0')
    wallet = WalletAgent(initial_balance=initial_balance)
    await wallet.initialize()  # Required for async wallet

    main_balance = wallet.get_balance(Currency.FIL, "main")
    audit_balance = wallet.get_balance(Currency.FIL, "audit")

    print(f"\n1. Wallet initialized with {initial_balance} FIL.")
    print(f"   -> Main Balance: {main_balance} FIL (90%)")
    print(f"   -> Audit Balance: {audit_balance} FIL (10%)")

    assert main_balance == initial_balance * Decimal('0.9')
    assert audit_balance == initial_balance * Decimal('0.1')

    # 2. Perform a successful transfer from the main balance
    transfer_amount = Decimal('50.5')
    print(f"\n2. Attempting to transfer {transfer_amount} FIL from main balance...")
    success = await wallet.transfer(transfer_amount, memo="Purchase compute services")
    assert success is True
    main_balance = wallet.get_balance(Currency.FIL, "main")
    print(f"   -> Transfer successful. New Main Balance: {main_balance} FIL")
    assert main_balance == (initial_balance * Decimal('0.9')) - transfer_amount

    # 3. Deduct a successful audit fee from the audit balance
    audit_fee = Decimal('2.1')
    print(f"\n3. Attempting to deduct audit fee of {audit_fee} FIL...")
    success = await wallet.deduct_audit_fee(audit_fee, memo="Constitutional integrity check")
    assert success is True
    audit_balance = wallet.get_balance(Currency.FIL, "audit")
    print(f"   -> Audit fee deduction successful. New Audit Balance: {audit_balance} FIL")
    assert audit_balance == (initial_balance * Decimal('0.1')) - audit_fee

    # 4. Attempt a transfer that is too large (insufficient funds)
    large_transfer_amount = Decimal('500.0')
    print(f"\n4. Attempting to transfer {large_transfer_amount} FIL (insufficient funds)...")
    success = await wallet.transfer(large_transfer_amount, memo="This should fail")
    assert success is False
    main_balance = wallet.get_balance(Currency.FIL, "main")
    print(f"   -> Transfer failed as expected. Main Balance remains: {main_balance} FIL")
    assert main_balance == (initial_balance * Decimal('0.9')) - transfer_amount

    # 5. Print final state
    print("\n--- End of Demonstration ---")
    print(f"Final Main Balance: {wallet.get_balance(Currency.FIL, 'main')}")
    print(f"Final Audit Balance: {wallet.get_balance(Currency.FIL, 'audit')}")
    print("Transaction History:")
    for tx in wallet.transaction_history:
        print(f"  - {tx}")
    print("----------------------------\n")


class TestWalletPersistence:
    """Tests for wallet state persistence to SQLite database (async)."""

    @pytest.fixture
    def temp_db_path(self, tmp_path):
        """Create a temporary database path."""
        return str(tmp_path / "test_wallet.db")

    @pytest.mark.asyncio
    async def test_wallet_saves_initial_state_to_db(self, temp_db_path):
        """Wallet should persist initial state to database."""
        import aiosqlite

        agent_id = "test-agent-001"
        wallet = WalletAgent(
            agent_id=agent_id,
            initial_balance=Decimal('100.0'),
            db_path=temp_db_path
        )
        await wallet.initialize()

        # Verify state was saved
        async with aiosqlite.connect(temp_db_path) as db:
            cursor = await db.execute(
                "SELECT balance FROM wallet_balances WHERE agent_id = ? AND currency = 'FIL' AND balance_type = 'main'",
                (agent_id,)
            )
            row = await cursor.fetchone()
            assert row is not None
            assert Decimal(row[0]) == Decimal('90.0')  # 90% of 100

            cursor = await db.execute(
                "SELECT balance FROM wallet_balances WHERE agent_id = ? AND currency = 'FIL' AND balance_type = 'audit'",
                (agent_id,)
            )
            row = await cursor.fetchone()
            assert row is not None
            assert Decimal(row[0]) == Decimal('10.0')  # 10% of 100

    @pytest.mark.asyncio
    async def test_wallet_loads_existing_state(self, temp_db_path):
        """Wallet should load existing state from database."""
        import aiosqlite

        agent_id = "test-agent-002"

        # Pre-populate database with custom balances
        async with aiosqlite.connect(temp_db_path) as db:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS wallet_balances (
                    agent_id TEXT,
                    currency TEXT,
                    balance_type TEXT,
                    balance TEXT,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (agent_id, currency, balance_type)
                )
            """)
            await db.execute(
                "INSERT INTO wallet_balances (agent_id, currency, balance_type, balance) VALUES (?, ?, ?, ?)",
                (agent_id, "FIL", "main", "75.5")
            )
            await db.execute(
                "INSERT INTO wallet_balances (agent_id, currency, balance_type, balance) VALUES (?, ?, ?, ?)",
                (agent_id, "FIL", "audit", "12.3")
            )
            await db.commit()

        # Create wallet - should load from DB, not use initial_balance
        wallet = WalletAgent(
            agent_id=agent_id,
            initial_balance=Decimal('999.0'),  # This should be ignored
            db_path=temp_db_path
        )
        await wallet.initialize()

        assert wallet.get_balance(Currency.FIL, "main") == Decimal('75.5')
        assert wallet.get_balance(Currency.FIL, "audit") == Decimal('12.3')

    @pytest.mark.asyncio
    async def test_wallet_persists_transactions(self, temp_db_path):
        """Wallet should persist transactions to database."""
        import aiosqlite

        agent_id = "test-agent-003"
        wallet = WalletAgent(
            agent_id=agent_id,
            initial_balance=Decimal('100.0'),
            db_path=temp_db_path
        )
        await wallet.initialize()

        # Perform a transfer
        await wallet.transfer(Decimal('10.0'), memo="Test payment")

        # Verify transaction was recorded
        async with aiosqlite.connect(temp_db_path) as db:
            cursor = await db.execute(
                "SELECT transaction_type, amount, memo, new_balance FROM wallet_transactions WHERE agent_id = ?",
                (agent_id,)
            )
            rows = await cursor.fetchall()

        assert len(rows) == 1
        # Transaction type now includes currency (e.g., "transfer_FIL")
        assert "transfer" in rows[0][0].lower()
        assert Decimal(rows[0][1]) == Decimal('10.0')
        assert rows[0][2] == "Test payment"
        assert Decimal(rows[0][3]) == Decimal('80.0')  # 90 - 10

    @pytest.mark.asyncio
    async def test_wallet_survives_restart(self, temp_db_path):
        """Wallet state should survive a restart (new instance with same DB)."""
        agent_id = "test-agent-004"

        # Create wallet and make some transactions
        wallet1 = WalletAgent(
            agent_id=agent_id,
            initial_balance=Decimal('100.0'),
            db_path=temp_db_path
        )
        await wallet1.initialize()
        await wallet1.transfer(Decimal('15.0'), memo="First payment")
        await wallet1.transfer(Decimal('5.0'), memo="Second payment")
        await wallet1.deduct_audit_fee(Decimal('2.0'), memo="Audit")

        final_main = wallet1.get_balance(Currency.FIL, "main")
        final_audit = wallet1.get_balance(Currency.FIL, "audit")

        # Simulate restart - create new wallet instance
        wallet2 = WalletAgent(
            agent_id=agent_id,
            initial_balance=Decimal('999.0'),  # Ignored due to existing state
            db_path=temp_db_path
        )
        await wallet2.initialize()

        # Verify state was restored
        assert wallet2.get_balance(Currency.FIL, "main") == final_main
        assert wallet2.get_balance(Currency.FIL, "audit") == final_audit
        assert wallet2.get_balance(Currency.FIL, "main") == Decimal('70.0')  # 90 - 15 - 5
        assert wallet2.get_balance(Currency.FIL, "audit") == Decimal('8.0')   # 10 - 2

    @pytest.mark.asyncio
    async def test_wallet_deposit_persists(self, temp_db_path):
        """Deposit should persist to database."""
        import aiosqlite

        agent_id = "test-agent-005"
        wallet = WalletAgent(
            agent_id=agent_id,
            initial_balance=Decimal('100.0'),
            db_path=temp_db_path
        )
        await wallet.initialize()

        # Deposit to main balance
        await wallet.deposit(Decimal('50.0'), memo="Bonus")
        assert wallet.get_balance(Currency.FIL, "main") == Decimal('140.0')  # 90 + 50

        # Deposit to audit balance
        await wallet.deposit(Decimal('5.0'), to_audit=True, memo="Audit reserve")
        assert wallet.get_balance(Currency.FIL, "audit") == Decimal('15.0')  # 10 + 5

        # Verify state was persisted
        async with aiosqlite.connect(temp_db_path) as db:
            cursor = await db.execute(
                "SELECT balance FROM wallet_balances WHERE agent_id = ? AND currency = 'FIL' AND balance_type = 'main'",
                (agent_id,)
            )
            row = await cursor.fetchone()
            assert Decimal(row[0]) == Decimal('140.0')

            cursor = await db.execute(
                "SELECT balance FROM wallet_balances WHERE agent_id = ? AND currency = 'FIL' AND balance_type = 'audit'",
                (agent_id,)
            )
            row = await cursor.fetchone()
            assert Decimal(row[0]) == Decimal('15.0')

    @pytest.mark.asyncio
    async def test_wallet_without_db_works(self):
        """Wallet should work normally without a database (in-memory only)."""
        wallet = WalletAgent(
            agent_id="no-db-agent",
            initial_balance=Decimal('50.0'),
            db_path=None
        )
        await wallet.initialize()

        assert wallet.get_balance(Currency.FIL, "main") == Decimal('45.0')  # 90% of 50
        assert wallet.get_balance(Currency.FIL, "audit") == Decimal('5.0')  # 10% of 50

        # Transactions should still work
        await wallet.transfer(Decimal('10.0'), memo="Test")
        assert wallet.get_balance(Currency.FIL, "main") == Decimal('35.0')

    @pytest.mark.asyncio
    async def test_wallet_get_status(self, temp_db_path):
        """Test get_status returns complete wallet information."""
        agent_id = "test-agent-006"
        wallet = WalletAgent(
            agent_id=agent_id,
            initial_balance=Decimal('100.0'),
            db_path=temp_db_path
        )
        await wallet.initialize()
        await wallet.transfer(Decimal('10.0'), memo="Payment")

        status = wallet.get_status()

        assert status["agent_id"] == agent_id
        assert Decimal(status["main_balance"]) == Decimal('80')
        assert Decimal(status["audit_balance"]) == Decimal('10')
        assert Decimal(status["total_balance"]) == Decimal('90')
        assert len(status["recent_transactions"]) == 1
        assert status["transaction_count"] == 1
