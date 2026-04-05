"""
Transaction Manager for Kestrel Wallet.

Orchestrates transaction signing, key management, spending limits, and audit logging.
Enforces security policies including mainnet blocking and daily limits.
"""

import os
import logging
import aiosqlite
from datetime import datetime, date
from decimal import Decimal
from pathlib import Path
from typing import Optional, Dict
from dataclasses import dataclass, field

from .chain_adapters import (
    ChainNetwork,
    EVMAdapter,
    ERC20Adapter,
    TransactionRequest,
    TransactionResult,
    TokenRegistry,
)

logger = logging.getLogger(__name__)


@dataclass
class SpendingRecord:
    """Record of spending for limit tracking."""

    date: date
    total_spent_usd: Decimal = Decimal("0")
    transaction_count: int = 0


@dataclass
class TransactionAudit:
    """Audit record for a transaction."""

    id: Optional[int] = None
    agent_id: str = ""
    timestamp: datetime = field(default_factory=datetime.utcnow)
    network: str = ""
    tx_type: str = ""  # "native" or "erc20"
    token_symbol: Optional[str] = None
    to_address: str = ""
    amount: Decimal = Decimal("0")
    amount_usd: Decimal = Decimal("0")
    tx_hash: Optional[str] = None
    status: str = ""  # "pending", "success", "failed"
    error: Optional[str] = None
    gas_used: Optional[int] = None
    metadata: Dict = field(default_factory=dict)


class TransactionManager:
    """
    Manages transaction signing and enforcement of security policies.

    Features:
    - Loads private keys from SecureKeyStorage
    - Routes to correct chain adapter
    - Tracks daily spending limits
    - Records all transactions to audit log
    - Blocks mainnet by default
    """

    # Default daily spending limit in USD
    DEFAULT_DAILY_LIMIT_USD = Decimal("100")

    # Price estimates for spending limit calculations (conservative)
    TOKEN_PRICES_USD = {
        "ETH": Decimal("3000"),
        "FIL": Decimal("5.50"),
        "tFIL": Decimal("0"),  # Testnet has no value
        "MATIC": Decimal("0.80"),
        "USDC": Decimal("1"),
        "USDT": Decimal("1"),
        "DAI": Decimal("1"),
    }

    def __init__(
        self,
        agent_id: str,
        storage_dir: Optional[Path] = None,
        daily_limit_usd: Optional[Decimal] = None,
    ):
        """
        Initialize transaction manager.

        Args:
            agent_id: Agent identifier for key lookup
            storage_dir: Directory for key storage and audit DB
            daily_limit_usd: Daily spending limit in USD
        """
        self.agent_id = agent_id

        if storage_dir is None:
            db_path = os.environ.get("KESTREL_DB_PATH", "./agent_dbs")
            storage_dir = Path(db_path)
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        # Spending limits
        limit_env = os.environ.get("KESTREL_TX_DAILY_LIMIT_USD")
        if limit_env:
            self.daily_limit_usd = Decimal(limit_env)
        elif daily_limit_usd:
            self.daily_limit_usd = daily_limit_usd
        else:
            self.daily_limit_usd = self.DEFAULT_DAILY_LIMIT_USD

        # Security settings
        self.allow_mainnet = os.environ.get("KESTREL_ALLOW_MAINNET", "").lower() == "true"
        self.require_approval = os.environ.get("KESTREL_TX_REQUIRE_APPROVAL", "true").lower() != "false"

        # Adapters cache
        self._adapters: Dict[ChainNetwork, EVMAdapter] = {}
        self._erc20_adapters: Dict[ChainNetwork, ERC20Adapter] = {}

        # Audit database
        self._audit_db_path = self.storage_dir / f"{agent_id}_tx_audit.db"
        self._db_initialized = False

        logger.info(
            f"TransactionManager initialized: agent={agent_id}, "
            f"daily_limit=${self.daily_limit_usd}, "
            f"mainnet={'allowed' if self.allow_mainnet else 'BLOCKED'}"
        )

    async def initialize(self):
        """Initialize audit database."""
        if self._db_initialized:
            return

        async with aiosqlite.connect(str(self._audit_db_path)) as db:
            await db.execute(
                """
                CREATE TABLE IF NOT EXISTS tx_audit (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    agent_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    network TEXT NOT NULL,
                    tx_type TEXT NOT NULL,
                    token_symbol TEXT,
                    to_address TEXT NOT NULL,
                    amount TEXT NOT NULL,
                    amount_usd TEXT NOT NULL,
                    tx_hash TEXT,
                    status TEXT NOT NULL,
                    error TEXT,
                    gas_used INTEGER,
                    metadata TEXT
                )
            """
            )

            await db.execute(
                """
                CREATE TABLE IF NOT EXISTS daily_spending (
                    date TEXT PRIMARY KEY,
                    total_spent_usd TEXT NOT NULL,
                    transaction_count INTEGER NOT NULL
                )
            """
            )

            await db.commit()

        self._db_initialized = True
        logger.info(f"Transaction audit DB initialized: {self._audit_db_path}")

    def get_adapter(self, network: ChainNetwork) -> EVMAdapter:
        """Get or create EVM adapter for a network."""
        if network not in self._adapters:
            self._adapters[network] = EVMAdapter(network)
        return self._adapters[network]

    def get_erc20_adapter(self, network: ChainNetwork) -> ERC20Adapter:
        """Get or create ERC-20 adapter for a network."""
        if network not in self._erc20_adapters:
            evm = self.get_adapter(network)
            self._erc20_adapters[network] = ERC20Adapter(evm)
        return self._erc20_adapters[network]

    async def check_mainnet_allowed(self, network: ChainNetwork) -> tuple[bool, str]:
        """
        Check if mainnet transactions are allowed.

        Args:
            network: Target network

        Returns:
            (allowed, error_message)
        """
        if network.is_mainnet and not self.allow_mainnet:
            return False, (
                f"Mainnet transactions are blocked. "
                f"Set KESTREL_ALLOW_MAINNET=true to enable {network.display_name}."
            )
        return True, ""

    async def check_spending_limit(
        self, amount_usd: Decimal
    ) -> tuple[bool, str, Decimal]:
        """
        Check if transaction is within daily spending limit.

        Args:
            amount_usd: Transaction amount in USD

        Returns:
            (allowed, error_message, remaining_limit)
        """
        await self.initialize()

        today = date.today().isoformat()

        async with aiosqlite.connect(str(self._audit_db_path)) as db:
            cursor = await db.execute(
                "SELECT total_spent_usd FROM daily_spending WHERE date = ?",
                (today,),
            )
            row = await cursor.fetchone()

            spent_today = Decimal(row[0]) if row else Decimal("0")
            remaining = self.daily_limit_usd - spent_today

            if amount_usd > remaining:
                return False, (
                    f"Transaction exceeds daily limit. "
                    f"Limit: ${self.daily_limit_usd}, Spent today: ${spent_today}, "
                    f"Remaining: ${remaining}, Requested: ${amount_usd}"
                ), remaining

            return True, "", remaining

    def estimate_usd_value(self, amount: Decimal, token: str) -> Decimal:
        """
        Estimate USD value of a token amount.

        Args:
            amount: Token amount
            token: Token symbol

        Returns:
            Estimated USD value
        """
        price = self.TOKEN_PRICES_USD.get(token.upper(), Decimal("0"))
        return amount * price

    async def send_native(
        self,
        network: ChainNetwork,
        to_address: str,
        amount: Decimal,
        private_key: bytes,
    ) -> TransactionResult:
        """
        Send native tokens (ETH, FIL, MATIC).

        Args:
            network: Target network
            to_address: Recipient address
            amount: Amount to send
            private_key: Sender's private key

        Returns:
            Transaction result
        """
        await self.initialize()

        # Check mainnet
        allowed, error = await self.check_mainnet_allowed(network)
        if not allowed:
            return TransactionResult(success=False, error=error, network=network)

        # Estimate USD value
        config = self.get_adapter(network).config
        amount_usd = self.estimate_usd_value(amount, config.native_token)

        # Check spending limit (skip for testnets)
        if network.is_mainnet:
            allowed, error, _ = await self.check_spending_limit(amount_usd)
            if not allowed:
                return TransactionResult(success=False, error=error, network=network)

        # Create audit record
        audit = TransactionAudit(
            agent_id=self.agent_id,
            network=network.value,
            tx_type="native",
            to_address=to_address,
            amount=amount,
            amount_usd=amount_usd,
            status="pending",
        )

        try:
            # Send transaction
            adapter = self.get_adapter(network)
            request = TransactionRequest(
                to_address=to_address,
                amount=amount,
                network=network,
            )
            result = await adapter.send_transaction(request, private_key)

            # Update audit
            audit.tx_hash = result.tx_hash
            audit.status = "success" if result.success else "failed"
            audit.error = result.error

            # Record to DB
            await self._record_transaction(audit)

            # Update spending if mainnet
            if result.success and network.is_mainnet:
                await self._update_spending(amount_usd)

            return result

        except (ConnectionError, TimeoutError, OSError) as e:
            audit.status = "failed"
            audit.error = str(e)
            await self._record_transaction(audit)
            raise
        except (ValueError, TypeError, KeyError) as e:
            audit.status = "failed"
            audit.error = str(e)
            await self._record_transaction(audit)
            raise
        except Exception as e:
            audit.status = "failed"
            audit.error = str(e)
            await self._record_transaction(audit)
            logger.error(f"Transaction failed: {e}", exc_info=True)
            raise

    async def send_token(
        self,
        network: ChainNetwork,
        token_symbol: str,
        to_address: str,
        amount: Decimal,
        private_key: bytes,
    ) -> TransactionResult:
        """
        Send ERC-20 tokens (USDC, USDT, etc.).

        Args:
            network: Target network
            token_symbol: Token symbol (e.g., "USDC")
            to_address: Recipient address
            amount: Amount to send
            private_key: Sender's private key

        Returns:
            Transaction result
        """
        await self.initialize()

        # Check mainnet
        allowed, error = await self.check_mainnet_allowed(network)
        if not allowed:
            return TransactionResult(success=False, error=error, network=network)

        # Check token exists
        token = TokenRegistry.get_token(token_symbol, network)
        if not token:
            return TransactionResult(
                success=False,
                error=f"Token {token_symbol} not available on {network.display_name}",
                network=network,
            )

        # Estimate USD value
        amount_usd = self.estimate_usd_value(amount, token_symbol)

        # Check spending limit (skip for testnets)
        if network.is_mainnet:
            allowed, error, _ = await self.check_spending_limit(amount_usd)
            if not allowed:
                return TransactionResult(success=False, error=error, network=network)

        # Create audit record
        audit = TransactionAudit(
            agent_id=self.agent_id,
            network=network.value,
            tx_type="erc20",
            token_symbol=token_symbol,
            to_address=to_address,
            amount=amount,
            amount_usd=amount_usd,
            status="pending",
        )

        try:
            # Send transaction
            erc20 = self.get_erc20_adapter(network)
            result = await erc20.transfer_by_symbol(
                symbol=token_symbol,
                to_address=to_address,
                amount=amount,
                private_key=private_key,
            )

            # Update audit
            audit.tx_hash = result.tx_hash
            audit.status = "success" if result.success else "failed"
            audit.error = result.error

            # Record to DB
            await self._record_transaction(audit)

            # Update spending if mainnet
            if result.success and network.is_mainnet:
                await self._update_spending(amount_usd)

            return result

        except (ConnectionError, TimeoutError, OSError) as e:
            audit.status = "failed"
            audit.error = str(e)
            await self._record_transaction(audit)
            raise
        except (ValueError, TypeError, KeyError) as e:
            audit.status = "failed"
            audit.error = str(e)
            await self._record_transaction(audit)
            raise
        except Exception as e:
            audit.status = "failed"
            audit.error = str(e)
            await self._record_transaction(audit)
            logger.error(f"Token transaction failed: {e}", exc_info=True)
            raise

    async def get_balance(
        self, network: ChainNetwork, address: str
    ) -> Decimal:
        """Get native token balance."""
        adapter = self.get_adapter(network)
        return await adapter.get_balance(address)

    async def get_token_balance(
        self, network: ChainNetwork, token_symbol: str, address: str
    ) -> Optional[Decimal]:
        """Get ERC-20 token balance."""
        erc20 = self.get_erc20_adapter(network)
        return await erc20.get_token_balance_by_symbol(token_symbol, address)

    async def get_spending_today(self) -> SpendingRecord:
        """Get today's spending record."""
        await self.initialize()

        today = date.today().isoformat()

        async with aiosqlite.connect(str(self._audit_db_path)) as db:
            cursor = await db.execute(
                "SELECT total_spent_usd, transaction_count FROM daily_spending WHERE date = ?",
                (today,),
            )
            row = await cursor.fetchone()

            if row:
                return SpendingRecord(
                    date=date.today(),
                    total_spent_usd=Decimal(row[0]),
                    transaction_count=row[1],
                )
            return SpendingRecord(date=date.today())

    async def get_transaction_history(
        self, limit: int = 20
    ) -> list[TransactionAudit]:
        """Get recent transaction history."""
        await self.initialize()

        async with aiosqlite.connect(str(self._audit_db_path)) as db:
            cursor = await db.execute(
                """
                SELECT id, agent_id, timestamp, network, tx_type, token_symbol,
                       to_address, amount, amount_usd, tx_hash, status, error, gas_used
                FROM tx_audit
                WHERE agent_id = ?
                ORDER BY timestamp DESC
                LIMIT ?
            """,
                (self.agent_id, limit),
            )

            rows = await cursor.fetchall()
            return [
                TransactionAudit(
                    id=row[0],
                    agent_id=row[1],
                    timestamp=datetime.fromisoformat(row[2]),
                    network=row[3],
                    tx_type=row[4],
                    token_symbol=row[5],
                    to_address=row[6],
                    amount=Decimal(row[7]),
                    amount_usd=Decimal(row[8]),
                    tx_hash=row[9],
                    status=row[10],
                    error=row[11],
                    gas_used=row[12],
                )
                for row in rows
            ]

    async def _record_transaction(self, audit: TransactionAudit):
        """Record transaction to audit log."""
        async with aiosqlite.connect(str(self._audit_db_path)) as db:
            await db.execute(
                """
                INSERT INTO tx_audit (
                    agent_id, timestamp, network, tx_type, token_symbol,
                    to_address, amount, amount_usd, tx_hash, status, error, gas_used
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    audit.agent_id,
                    audit.timestamp.isoformat(),
                    audit.network,
                    audit.tx_type,
                    audit.token_symbol,
                    audit.to_address,
                    str(audit.amount),
                    str(audit.amount_usd),
                    audit.tx_hash,
                    audit.status,
                    audit.error,
                    audit.gas_used,
                ),
            )
            await db.commit()

    async def _update_spending(self, amount_usd: Decimal):
        """Update daily spending record."""
        today = date.today().isoformat()

        async with aiosqlite.connect(str(self._audit_db_path)) as db:
            cursor = await db.execute(
                "SELECT total_spent_usd, transaction_count FROM daily_spending WHERE date = ?",
                (today,),
            )
            row = await cursor.fetchone()

            if row:
                new_total = Decimal(row[0]) + amount_usd
                new_count = row[1] + 1
                await db.execute(
                    "UPDATE daily_spending SET total_spent_usd = ?, transaction_count = ? WHERE date = ?",
                    (str(new_total), new_count, today),
                )
            else:
                await db.execute(
                    "INSERT INTO daily_spending (date, total_spent_usd, transaction_count) VALUES (?, ?, ?)",
                    (today, str(amount_usd), 1),
                )

            await db.commit()

    async def close(self):
        """Clean up adapters."""
        for adapter in self._adapters.values():
            await adapter.close()
        self._adapters.clear()
        self._erc20_adapters.clear()
