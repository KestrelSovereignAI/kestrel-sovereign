"""
Multi-Currency Wallet Agent (Async)

A Feature Agent that manages the Kestrel agent's economic identity and transactions.
Supports multiple currencies (FIL, USDC, USDT) for Lighthouse storage payments.

Cryostasis Integration:
- Monitors balance against cryostasis threshold
- Can trigger dormancy when funds are low
- Supports wake-up funding
"""

from decimal import Decimal, getcontext
from typing import List, Dict, Optional
from datetime import datetime, timezone
from enum import Enum
import logging
import aiosqlite

# Set precision for Decimal calculations
getcontext().prec = 18

logger = logging.getLogger(__name__)


class Currency(Enum):
    """Supported currencies for wallet operations."""
    FIL = "FIL"      # Filecoin (primary)
    USDC = "USDC"    # USD Coin (stablecoin)
    USDT = "USDT"    # Tether (stablecoin)
    USD = "USD"      # Internal USD tracking


# Default exchange rates (FIL varies, stablecoins ~$1)
# In production, these would come from an oracle/API
DEFAULT_EXCHANGE_RATES = {
    Currency.FIL: Decimal("5.50"),   # 1 FIL = $5.50
    Currency.USDC: Decimal("1.00"),  # 1 USDC = $1
    Currency.USDT: Decimal("1.00"),  # 1 USDT = $1
    Currency.USD: Decimal("1.00"),   # 1 USD = $1 (reference)
}


class WalletAgent:
    """
    A Feature Agent that manages the Kestrel agent's economic identity and transactions.

    Supports:
    - Multiple currencies (FIL, USDC, USDT)
    - Main and audit balances
    - Cryostasis threshold monitoring
    - Async SQLite persistence via aiosqlite

    Architecture:
    - 90% of deposits go to main balance (operations)
    - 10% of deposits go to audit reserve (governance)
    """

    def __init__(
        self,
        agent_id: str = "default",
        initial_balance: Decimal = Decimal('100.0'),
        initial_currency: Currency = Currency.FIL,
        db_path: Optional[str] = None,
        cryostasis_threshold_usd: Optional[Decimal] = None,
    ):
        """
        Initialize the wallet agent.

        Args:
            agent_id: Unique agent identifier
            initial_balance: Starting balance
            initial_currency: Currency for initial balance (default FIL)
            db_path: Path to SQLite database for async persistence
            cryostasis_threshold_usd: Minimum USD balance before cryostasis
        """
        self.agent_id = agent_id
        self.db_path = db_path
        self._initial_balance = initial_balance
        self._initial_currency = initial_currency
        self.transaction_history: List[Dict] = []
        self._initialized = False

        # Multi-currency balances
        self._balances: Dict[Currency, Dict[str, Decimal]] = {
            currency: {"main": Decimal("0"), "audit": Decimal("0")}
            for currency in Currency
        }

        # Exchange rates (updateable)
        self._exchange_rates = DEFAULT_EXCHANGE_RATES.copy()

        # Cryostasis threshold
        self._cryostasis_threshold_usd = cryostasis_threshold_usd or Decimal("0.02")

    async def initialize(self) -> None:
        """Async initialization - load from database or set defaults."""
        if self._initialized:
            return

        # Initialize Filecoin-related attributes
        self._filecoin_address: Optional[str] = None
        self._filecoin_adapter = None
        self._last_sync: Optional[datetime] = None

        # Try to load from database first
        if self.db_path and await self._load_from_db():
            logger.info(f"WalletAgent loaded from database for agent {self.agent_id}")
            # Also try to load Filecoin address
            await self._load_filecoin_address_from_db()
        else:
            # Initialize with defaults (90% main, 10% audit)
            main_amount = self._initial_balance * Decimal('0.9')
            audit_amount = self._initial_balance * Decimal('0.1')

            self._balances[self._initial_currency]["main"] = main_amount
            self._balances[self._initial_currency]["audit"] = audit_amount

            logger.info(
                f"WalletAgent initialized: {main_amount} {self._initial_currency.value} main, "
                f"{audit_amount} {self._initial_currency.value} audit"
            )

            # Persist initial state if database available
            if self.db_path:
                await self._save_to_db()

        self._initialized = True

    # =========================================================================
    # Multi-Currency Balance Methods
    # =========================================================================

    def get_balance(self, currency: Currency = Currency.FIL, balance_type: str = "main") -> Decimal:
        """
        Get balance for a specific currency and type.

        Args:
            currency: Currency to check
            balance_type: "main" or "audit"

        Returns:
            Balance in specified currency
        """
        return self._balances[currency].get(balance_type, Decimal("0"))

    def get_total_balance(self, currency: Currency = Currency.FIL) -> Decimal:
        """Get total balance (main + audit) for a currency."""
        return self._balances[currency]["main"] + self._balances[currency]["audit"]

    def get_total_balance_usd(self) -> Decimal:
        """Get total balance across all currencies in USD."""
        total = Decimal("0")
        for currency in Currency:
            if currency == Currency.USD:
                continue
            currency_total = self.get_total_balance(currency)
            rate = self._exchange_rates.get(currency, Decimal("1"))
            total += currency_total * rate
        return total

    def get_audit_balance(self, currency: Currency = Currency.FIL) -> Decimal:
        """Get audit balance for a currency."""
        return self._balances[currency]["audit"]

    def get_all_balances(self) -> Dict[str, Dict[str, str]]:
        """
        Get all balances across all currencies.

        Returns:
            Dict mapping currency name to {main, audit} balances
        """
        return {
            currency.value: {
                "main": str(balances["main"]),
                "audit": str(balances["audit"]),
                "total": str(balances["main"] + balances["audit"]),
            }
            for currency, balances in self._balances.items()
        }

    # =========================================================================
    # Exchange Rate Management
    # =========================================================================

    def update_exchange_rate(self, currency: Currency, rate_usd: Decimal) -> None:
        """
        Update exchange rate for a currency.

        Args:
            currency: Currency to update
            rate_usd: New rate in USD per unit
        """
        self._exchange_rates[currency] = rate_usd
        logger.info(f"Exchange rate updated: 1 {currency.value} = ${rate_usd}")

    def convert_to_usd(self, amount: Decimal, currency: Currency) -> Decimal:
        """Convert amount from currency to USD."""
        rate = self._exchange_rates.get(currency, Decimal("1"))
        return amount * rate

    def convert_from_usd(self, amount_usd: Decimal, target_currency: Currency) -> Decimal:
        """Convert USD amount to target currency."""
        rate = self._exchange_rates.get(target_currency, Decimal("1"))
        if rate == 0:
            return Decimal("0")
        return amount_usd / rate

    # =========================================================================
    # Transaction Methods
    # =========================================================================

    def can_afford(
        self,
        amount: Decimal,
        currency: Currency = Currency.FIL,
        balance_type: str = "main",
    ) -> bool:
        """Check if the wallet has sufficient balance."""
        return self._balances[currency][balance_type] >= amount

    def can_afford_usd(self, amount_usd: Decimal) -> bool:
        """Check if total USD value across all currencies covers the amount."""
        return self.get_total_balance_usd() >= amount_usd

    def can_afford_audit(self, amount: Decimal, currency: Currency = Currency.FIL) -> bool:
        """Check if audit balance covers the amount."""
        return self._balances[currency]["audit"] >= amount

    async def transfer(
        self,
        amount: Decimal,
        memo: str,
        currency: Currency = Currency.FIL,
    ) -> bool:
        """
        Transfer from main balance.

        Args:
            amount: Amount to transfer
            memo: Transaction memo
            currency: Currency to use

        Returns:
            True if successful
        """
        if not self.can_afford(amount, currency, "main"):
            logger.warning(
                f"Transaction failed: Insufficient {currency.value} for {memo}. "
                f"Need {amount}, have {self._balances[currency]['main']}."
            )
            return False

        self._balances[currency]["main"] -= amount

        transaction = {
            "type": "transfer",
            "currency": currency.value,
            "amount": str(amount),
            "memo": memo,
            "new_balance": str(self._balances[currency]["main"]),
            "timestamp": self._get_timestamp(),
        }
        self.transaction_history.append(transaction)

        # Persist to database
        await self._record_transaction(
            f"transfer_{currency.value}",
            amount,
            memo,
            self._balances[currency]["main"],
            currency,
        )
        await self._save_to_db()

        logger.info(
            f"Transaction successful: {memo} for {amount} {currency.value}. "
            f"New balance: {self._balances[currency]['main']} {currency.value}"
        )
        return True

    async def deduct_audit_fee(
        self,
        amount: Decimal,
        memo: str,
        currency: Currency = Currency.FIL,
    ) -> bool:
        """Deduct from audit balance."""
        if not self.can_afford_audit(amount, currency):
            logger.warning(
                f"Audit fee failed: Insufficient {currency.value} for {memo}. "
                f"Need {amount}, have {self._balances[currency]['audit']}."
            )
            return False

        self._balances[currency]["audit"] -= amount

        transaction = {
            "type": "audit_fee",
            "currency": currency.value,
            "amount": str(amount),
            "memo": memo,
            "new_balance": str(self._balances[currency]["audit"]),
            "timestamp": self._get_timestamp(),
        }
        self.transaction_history.append(transaction)

        await self._record_transaction(
            f"audit_{currency.value}",
            amount,
            memo,
            self._balances[currency]["audit"],
            currency,
        )
        await self._save_to_db()

        logger.info(
            f"Audit fee deducted: {memo} for {amount} {currency.value}. "
            f"New audit balance: {self._balances[currency]['audit']} {currency.value}"
        )
        return True

    async def deposit(
        self,
        amount: Decimal,
        currency: Currency = Currency.FIL,
        to_audit: bool = False,
        memo: str = "deposit",
    ) -> bool:
        """
        Deposit funds into the wallet.

        Args:
            amount: Amount to deposit
            currency: Currency being deposited
            to_audit: If True, deposit to audit balance
            memo: Transaction memo

        Returns:
            True if successful
        """
        if amount <= 0:
            logger.warning(f"Deposit failed: Amount must be positive, got {amount}")
            return False

        balance_type = "audit" if to_audit else "main"
        self._balances[currency][balance_type] += amount
        new_balance = self._balances[currency][balance_type]

        transaction = {
            "type": f"deposit_{balance_type}",
            "currency": currency.value,
            "amount": str(amount),
            "memo": memo,
            "new_balance": str(new_balance),
            "timestamp": self._get_timestamp(),
        }
        self.transaction_history.append(transaction)

        await self._record_transaction(
            f"deposit_{balance_type}_{currency.value}",
            amount,
            memo,
            new_balance,
            currency,
        )
        await self._save_to_db()

        logger.info(
            f"Deposit successful: {amount} {currency.value} to {balance_type}. "
            f"New balance: {new_balance} {currency.value}"
        )
        return True

    async def pay_for_storage(
        self,
        amount_usd: Decimal,
        preferred_currency: Currency = Currency.FIL,
        memo: str = "storage payment",
    ) -> Optional[Dict]:
        """
        Pay for storage, preferring specified currency but falling back.

        Args:
            amount_usd: Amount in USD to pay
            preferred_currency: Preferred payment currency
            memo: Transaction memo

        Returns:
            Transaction details or None if insufficient funds
        """
        # Try preferred currency first
        amount_in_currency = self.convert_from_usd(amount_usd, preferred_currency)

        if self.can_afford(amount_in_currency, preferred_currency):
            if await self.transfer(amount_in_currency, memo, preferred_currency):
                return {
                    "success": True,
                    "currency": preferred_currency.value,
                    "amount": str(amount_in_currency),
                    "amount_usd": str(amount_usd),
                }

        # Try other currencies
        for currency in [Currency.USDC, Currency.USDT, Currency.FIL]:
            if currency == preferred_currency:
                continue

            amount_in_currency = self.convert_from_usd(amount_usd, currency)
            if self.can_afford(amount_in_currency, currency):
                if await self.transfer(amount_in_currency, memo, currency):
                    return {
                        "success": True,
                        "currency": currency.value,
                        "amount": str(amount_in_currency),
                        "amount_usd": str(amount_usd),
                    }

        logger.warning(f"Insufficient funds for storage payment: ${amount_usd}")
        return None

    # =========================================================================
    # Cryostasis Support
    # =========================================================================

    def check_cryostasis_threshold(self) -> bool:
        """
        Check if balance is below cryostasis threshold.

        Returns:
            True if wallet balance is below threshold (cryostasis needed)
        """
        total_usd = self.get_total_balance_usd()
        return total_usd < self._cryostasis_threshold_usd

    def set_cryostasis_threshold(self, threshold_usd: Decimal) -> None:
        """Set the cryostasis threshold in USD."""
        self._cryostasis_threshold_usd = threshold_usd
        logger.info(f"Cryostasis threshold set to: ${threshold_usd}")

    def get_runway_estimate(self, monthly_cost_usd: Decimal) -> Optional[int]:
        """
        Estimate how many months until cryostasis.

        Args:
            monthly_cost_usd: Estimated monthly operating cost

        Returns:
            Number of months until cryostasis, or None if infinite (no costs)
        """
        if monthly_cost_usd <= 0:
            return None  # Infinite runway with no costs

        total_usd = self.get_total_balance_usd()
        buffer = self._cryostasis_threshold_usd
        available = total_usd - buffer

        if available <= 0:
            return 0

        return int(available / monthly_cost_usd)

    # =========================================================================
    # Filecoin Testnet Integration
    # =========================================================================

    @property
    def filecoin_address(self) -> Optional[str]:
        """
        Get the on-chain Filecoin address for this wallet.

        Returns:
            Filecoin address (t1... for testnet) or None if not set
        """
        return getattr(self, '_filecoin_address', None)

    async def set_filecoin_address(self, address: str) -> None:
        """
        Configure on-chain Filecoin address for this wallet.

        This links the internal wallet to an on-chain address,
        enabling balance sync and transaction tracking.

        Args:
            address: Filecoin address (t1... or f1...)
        """
        self._filecoin_address = address
        self._filecoin_adapter = None  # Will be created on first sync

        logger.info(f"Filecoin address configured: {address}")

        # Persist to database
        if self.db_path:
            await self._save_filecoin_address_to_db()

    async def _save_filecoin_address_to_db(self) -> None:
        """Persist Filecoin address to database."""
        if not self.db_path or not self._filecoin_address:
            return
        try:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute("""
                    CREATE TABLE IF NOT EXISTS wallet_config (
                        agent_id TEXT PRIMARY KEY,
                        filecoin_address TEXT,
                        filecoin_network TEXT DEFAULT 'calibration',
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                await db.execute(
                    """INSERT OR REPLACE INTO wallet_config
                       (agent_id, filecoin_address, updated_at)
                       VALUES (?, ?, CURRENT_TIMESTAMP)""",
                    (self.agent_id, self._filecoin_address)
                )
                await db.commit()
        except (OSError, IOError) as e:
            logger.error(f"Failed to save Filecoin address: {e}")
        except Exception as e:
            logger.error(f"Failed to save Filecoin address: {e}", exc_info=True)

    async def _load_filecoin_address_from_db(self) -> bool:
        """Load Filecoin address from database."""
        if not self.db_path:
            return False
        try:
            async with aiosqlite.connect(self.db_path) as db:
                cursor = await db.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name='wallet_config'"
                )
                if not await cursor.fetchone():
                    return False

                cursor = await db.execute(
                    "SELECT filecoin_address FROM wallet_config WHERE agent_id = ?",
                    (self.agent_id,)
                )
                row = await cursor.fetchone()
                if row and row[0]:
                    self._filecoin_address = row[0]
                    return True
        except (OSError, IOError) as e:
            logger.warning(f"Failed to load Filecoin address: {e}")
        except Exception as e:
            logger.warning(f"Failed to load Filecoin address: {e}", exc_info=True)
        return False

    async def sync_on_chain_balance(self) -> bool:
        """
        Sync wallet FIL balance with on-chain Filecoin balance.

        Queries the Filecoin Calibration testnet for the actual balance
        and updates the internal wallet to match.

        Note: Only syncs FIL currency. Other currencies (USDC, USDT)
        remain internally tracked.

        Returns:
            True if sync successful
        """
        if not self._filecoin_address:
            logger.warning("No Filecoin address configured for sync")
            return False

        # Lazy-load the adapter
        if not hasattr(self, '_filecoin_adapter') or self._filecoin_adapter is None:
            from .filecoin_testnet import FilecoinTestnetAdapter
            self._filecoin_adapter = FilecoinTestnetAdapter(
                wallet_address=self._filecoin_address
            )

        try:
            on_chain_balance = await self._filecoin_adapter.get_balance()

            # Calculate difference from internal balance
            current_total = self.get_total_balance(Currency.FIL)
            difference = on_chain_balance - current_total

            if difference != Decimal("0"):
                if difference > 0:
                    # Deposit detected - record it (goes to main, not split)
                    self._balances[Currency.FIL]["main"] += difference
                    await self._save_to_db()

                    transaction = {
                        "type": "on_chain_sync_deposit",
                        "currency": Currency.FIL.value,
                        "amount": str(difference),
                        "memo": "on-chain balance sync",
                        "new_balance": str(self._balances[Currency.FIL]["main"]),
                        "timestamp": self._get_timestamp(),
                    }
                    self.transaction_history.append(transaction)

                    await self._record_transaction(
                        "sync_deposit_FIL",
                        difference,
                        "on-chain balance sync",
                        self._balances[Currency.FIL]["main"],
                        Currency.FIL,
                    )

                    logger.info(
                        f"On-chain sync: detected deposit of {difference} FIL. "
                        f"Balance: {current_total} -> {on_chain_balance}"
                    )
                else:
                    # Withdrawal detected (external send)
                    # This updates internal state to match chain
                    self._balances[Currency.FIL]["main"] += difference  # difference is negative
                    await self._save_to_db()

                    transaction = {
                        "type": "on_chain_sync_withdrawal",
                        "currency": Currency.FIL.value,
                        "amount": str(abs(difference)),
                        "memo": "external withdrawal detected",
                        "new_balance": str(self._balances[Currency.FIL]["main"]),
                        "timestamp": self._get_timestamp(),
                    }
                    self.transaction_history.append(transaction)

                    logger.info(
                        f"On-chain sync: detected withdrawal of {abs(difference)} FIL. "
                        f"Balance: {current_total} -> {on_chain_balance}"
                    )

            self._last_sync = datetime.now(timezone.utc)
            return True

        except (ConnectionError, TimeoutError, OSError) as e:
            logger.error(f"Failed to sync on-chain balance: {e}")
            return False
        except (ValueError, TypeError, KeyError) as e:
            logger.error(f"Failed to sync on-chain balance: {e}")
            return False
        except Exception as e:
            logger.error(f"Failed to sync on-chain balance: {e}", exc_info=True)
            return False

    async def get_on_chain_balance(self) -> Optional[Decimal]:
        """
        Query current on-chain FIL balance without updating internal state.

        Returns:
            On-chain balance in FIL or None if unavailable
        """
        if not self._filecoin_address:
            return None

        # Lazy-load the adapter
        if not hasattr(self, '_filecoin_adapter') or self._filecoin_adapter is None:
            from .filecoin_testnet import FilecoinTestnetAdapter
            self._filecoin_adapter = FilecoinTestnetAdapter(
                wallet_address=self._filecoin_address
            )

        try:
            return await self._filecoin_adapter.get_balance()
        except (ConnectionError, TimeoutError, OSError) as e:
            logger.error(f"Failed to query on-chain balance: {e}")
            return None
        except Exception as e:
            logger.error(f"Failed to query on-chain balance: {e}", exc_info=True)
            return None

    # =========================================================================
    # Status and Reporting
    # =========================================================================

    def get_status(self) -> Dict:
        """Get complete wallet status for reporting."""
        status = {
            "agent_id": self.agent_id,
            "balances": self.get_all_balances(),
            "total_usd": str(self.get_total_balance_usd()),
            "cryostasis_threshold_usd": str(self._cryostasis_threshold_usd),
            "below_cryostasis_threshold": self.check_cryostasis_threshold(),
            "exchange_rates": {
                currency.value: str(rate)
                for currency, rate in self._exchange_rates.items()
            },
            "recent_transactions": self.transaction_history[-10:],
            "transaction_count": len(self.transaction_history),
            "main_balance": str(self.get_balance(Currency.FIL, "main")),
            "audit_balance": str(self.get_balance(Currency.FIL, "audit")),
            "total_balance": str(self.get_total_balance(Currency.FIL)),
        }

        # Add Filecoin testnet info
        if hasattr(self, '_filecoin_address') and self._filecoin_address:
            status["filecoin_address"] = self._filecoin_address
            status["last_sync"] = (
                self._last_sync.isoformat() if hasattr(self, '_last_sync') and self._last_sync else None
            )

        return status

    # =========================================================================
    # Database Persistence (Async)
    # =========================================================================

    async def _load_from_db(self) -> bool:
        """Load wallet state from database."""
        if not self.db_path:
            return False
        try:
            async with aiosqlite.connect(self.db_path) as db:
                # Check if multi-currency table exists
                cursor = await db.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name='wallet_balances'"
                )
                if await cursor.fetchone():
                    # Load multi-currency balances
                    cursor = await db.execute(
                        """SELECT currency, balance_type, balance
                           FROM wallet_balances WHERE agent_id = ?""",
                        (self.agent_id,)
                    )
                    rows = await cursor.fetchall()
                    if rows:
                        for row in rows:
                            currency = Currency(row[0])
                            balance_type = row[1]
                            balance = Decimal(row[2])
                            self._balances[currency][balance_type] = balance
                        return True

                # Fall back to legacy table
                cursor = await db.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name='wallet_state'"
                )
                if await cursor.fetchone():
                    cursor = await db.execute(
                        "SELECT main_balance, audit_balance FROM wallet_state WHERE agent_id = ?",
                        (self.agent_id,)
                    )
                    row = await cursor.fetchone()
                    if row:
                        self._balances[Currency.FIL]["main"] = Decimal(row[0])
                        self._balances[Currency.FIL]["audit"] = Decimal(row[1])

                        # Load transaction history
                        cursor = await db.execute(
                            """SELECT transaction_type, amount, memo, new_balance, created_at
                               FROM wallet_transactions
                               WHERE agent_id = ?
                               ORDER BY created_at DESC LIMIT 100""",
                            (self.agent_id,)
                        )
                        rows = await cursor.fetchall()
                        self.transaction_history = [
                            {
                                "type": r[0],
                                "amount": r[1],
                                "memo": r[2],
                                "new_balance": r[3],
                                "timestamp": r[4],
                            }
                            for r in rows
                        ]
                        return True
        except (OSError, IOError) as e:
            logger.warning(f"Failed to load wallet state: {e}")
        except Exception as e:
            logger.warning(f"Failed to load wallet state: {e}", exc_info=True)
        return False

    async def _save_to_db(self) -> bool:
        """Persist wallet state to database."""
        if not self.db_path:
            return False
        try:
            async with aiosqlite.connect(self.db_path) as db:
                # Create multi-currency table if needed
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

                # Save all balances
                for currency, balances in self._balances.items():
                    for balance_type, amount in balances.items():
                        await db.execute(
                            """INSERT OR REPLACE INTO wallet_balances
                               (agent_id, currency, balance_type, balance, updated_at)
                               VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)""",
                            (self.agent_id, currency.value, balance_type, str(amount))
                        )

                # Also update legacy table for backward compatibility
                await db.execute("""
                    CREATE TABLE IF NOT EXISTS wallet_state (
                        agent_id TEXT PRIMARY KEY,
                        main_balance TEXT,
                        audit_balance TEXT,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                await db.execute(
                    """INSERT OR REPLACE INTO wallet_state
                       (agent_id, main_balance, audit_balance, updated_at)
                       VALUES (?, ?, ?, CURRENT_TIMESTAMP)""",
                    (
                        self.agent_id,
                        str(self._balances[Currency.FIL]["main"]),
                        str(self._balances[Currency.FIL]["audit"]),
                    )
                )

                await db.commit()
                return True
        except (OSError, IOError) as e:
            logger.error(f"Failed to save wallet state: {e}")
            return False
        except Exception as e:
            logger.error(f"Failed to save wallet state: {e}", exc_info=True)
            return False

    async def _record_transaction(
        self,
        tx_type: str,
        amount: Decimal,
        memo: str,
        new_balance: Decimal,
        currency: Currency = Currency.FIL,
    ) -> None:
        """Record a transaction to database."""
        if not self.db_path:
            return
        try:
            async with aiosqlite.connect(self.db_path) as db:
                # Create transactions table if needed
                await db.execute("""
                    CREATE TABLE IF NOT EXISTS wallet_transactions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        agent_id TEXT,
                        transaction_type TEXT,
                        currency TEXT DEFAULT 'FIL',
                        amount TEXT,
                        memo TEXT,
                        new_balance TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)

                await db.execute(
                    """INSERT INTO wallet_transactions
                       (agent_id, transaction_type, currency, amount, memo, new_balance)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    (self.agent_id, tx_type, currency.value, str(amount), memo, str(new_balance))
                )
                await db.commit()
        except (OSError, IOError) as e:
            logger.error(f"Failed to record transaction: {e}")
        except Exception as e:
            logger.error(f"Failed to record transaction: {e}", exc_info=True)

    def _get_timestamp(self) -> str:
        """Get current timestamp in ISO format."""
        return datetime.now(timezone.utc).isoformat()


# Convenience function for creating wallets
async def create_wallet(
    agent_id: str = "default",
    initial_balance: Decimal = Decimal("100.0"),
    currency: Currency = Currency.FIL,
    db_path: Optional[str] = None,
) -> WalletAgent:
    """
    Create a new WalletAgent.

    Args:
        agent_id: Agent identifier
        initial_balance: Starting balance
        currency: Initial currency
        db_path: Optional SQLite database path

    Returns:
        Configured WalletAgent
    """
    wallet = WalletAgent(
        agent_id=agent_id,
        initial_balance=initial_balance,
        initial_currency=currency,
        db_path=db_path,
    )
    await wallet.initialize()
    return wallet


# Import WalletFeature at bottom to avoid circular imports
# This allows feature discovery to find WalletFeature when importing features.wallet.feature
from .wallet_feature import WalletFeature

__all__ = ["WalletAgent", "Currency", "WalletFeature", "create_wallet"]
