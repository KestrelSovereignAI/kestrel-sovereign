"""
WalletFeature - Feature wrapper for WalletAgent

Exposes wallet functionality as agent tools:
- !wallet-balance - Check balances across currencies
- !wallet-transfer - Transfer funds from main balance
- !wallet-deposit - Record deposits
- !wallet-history - View transaction history
- !wallet-status - Full status with cryostasis info
- !wallet-exchange-rates - View/update exchange rates
- !wallet-generate-address - Generate Filecoin testnet address
- !wallet-sync - Sync with Filecoin testnet
- !wallet-address - Show Filecoin address info
- !wallet-send - Send native tokens on EVM chains
- !wallet-send-token - Send ERC-20 tokens on EVM chains
- !wallet-networks - List available networks
- !wallet-tx-history - View blockchain transaction history

Chain-specific logic is delegated to mixin modules:
- economic_gates.py  - is_paid_tier(), has_revenue_share()
- filecoin_tools.py  - Filecoin testnet tools
- multichain_tools.py - EVM multi-chain tools
"""

import logging
from decimal import Decimal, InvalidOperation
from typing import List, Optional

from kestrel_sdk.features.base import Feature, tool
from kestrel_sdk.hooks.base import Hook
from kestrel_sdk.tools.base import ToolCategory
from .feature import WalletAgent, Currency
from .transaction_hook import TransactionSecurityHook
from .economic_gates import EconomicGateMixin
from .filecoin_tools import FilecoinToolsMixin
from .multichain_tools import MultichainToolsMixin

logger = logging.getLogger(__name__)


class WalletFeature(
    EconomicGateMixin,
    FilecoinToolsMixin,
    MultichainToolsMixin,
    Feature,
):
    """
    Feature for managing the agent's economic identity and transactions.

    Wraps the WalletAgent class to expose wallet operations as agent tools.
    Supports multi-currency (FIL, USDC, USDT) with main/audit balance split.

    Chain-specific tools are provided by mixin classes:
    - EconomicGateMixin: is_paid_tier(), has_revenue_share()
    - FilecoinToolsMixin: wallet_generate_address, wallet_sync, wallet_address
    - MultichainToolsMixin: wallet_send, wallet_send_token, wallet_networks, wallet_tx_history
    """

    def __init__(self, agent):
        super().__init__(agent)
        self.wallet: Optional[WalletAgent] = None

    @property
    def tool_description(self) -> str:
        return (
            "Manage the agent's wallet - check balances across currencies (FIL, USDC, USDT), "
            "transfer funds, view transaction history, and monitor cryostasis threshold"
        )

    async def initialize(self):
        """Initialize the WalletAgent with the agent's database path."""
        logger.info("Initializing WalletFeature")

        # Reuse wallet already created by KestrelAgent.initialize() — it reads
        # the inception initialBalance from the agent graph node. Creating a new
        # WalletAgent here would overwrite it with the default 100 FIL.
        if hasattr(self.agent, 'wallet') and self.agent.wallet is not None:
            self.wallet = self.agent.wallet
            logger.info(
                f"WalletFeature initialized for agent {self.agent.did}, "
                f"total USD value: ${self.wallet.get_total_balance_usd()}"
            )
            return

        # Fallback: agent didn't pre-initialize a wallet (e.g. lightweight test path).
        # Read inception balance from agent node so we don't default to 100 FIL.
        db_path = None
        if hasattr(self.agent, 'storage') and hasattr(self.agent.storage, 'db_path'):
            db_path = self.agent.storage.db_path
        elif hasattr(self.agent, 'db_path'):
            db_path = self.agent.db_path

        agent_id = self.agent.did

        # Try to read inception initialBalance from the agent graph node
        initial_balance = Decimal('100.0')
        try:
            if hasattr(self.agent, 'storage'):
                agent_node = await self.agent.storage.get_node(agent_id)
                if agent_node and 'initialBalance' in agent_node.properties:
                    initial_balance = Decimal(agent_node.properties['initialBalance'])
        except Exception as e:
            logger.warning(f"Could not read initialBalance from agent node: {e}")

        self.wallet = WalletAgent(
            agent_id=agent_id,
            initial_balance=initial_balance,
            db_path=db_path
        )
        await self.wallet.initialize()

        # Attach wallet to agent for other features (e.g., SovereigntyFeature)
        self.agent.wallet = self.wallet

        logger.info(
            f"WalletFeature initialized for agent {agent_id}, "
            f"total USD value: ${self.wallet.get_total_balance_usd()}"
        )

    def get_hooks(self) -> List[Hook]:
        """Return the transaction security hook for auto-registration."""
        return [TransactionSecurityHook()]

    async def shutdown(self):
        """Cleanup wallet resources."""
        if self.wallet and self.wallet.db_path:
            # Ensure final state is persisted
            await self.wallet._save_to_db()
        logger.info("WalletFeature shutdown complete")

    # =========================================================================
    # Core Wallet Tools
    # =========================================================================

    @tool(
        name="wallet_balance",
        description="Check wallet balances across all currencies or a specific currency",
        category=ToolCategory.SYSTEM,
        command_prefix="!wallet-balance"
    )
    async def wallet_balance(self, currency: str = "all") -> str:
        """
        Check wallet balance for one or all currencies.

        Args:
            currency: Currency to check - 'FIL', 'USDC', 'USDT', or 'all' (default)

        Returns:
            Formatted balance report
        """
        if not self.wallet:
            return "❌ Wallet not initialized"

        currency_upper = currency.upper()

        if currency_upper == "ALL":
            # Show all balances
            balances = self.wallet.get_all_balances()
            total_usd = self.wallet.get_total_balance_usd()

            lines = ["💰 **Wallet Balances**", ""]
            for curr, amounts in balances.items():
                if Decimal(amounts["total"]) > 0:
                    lines.append(f"**{curr}:**")
                    lines.append(f"  Main: {amounts['main']}")
                    lines.append(f"  Audit: {amounts['audit']}")
                    lines.append(f"  Total: {amounts['total']}")
                    lines.append("")

            lines.append(f"**Total USD Value:** ${total_usd:.2f}")
            return "\n".join(lines)

        else:
            # Show specific currency
            try:
                curr_enum = Currency(currency_upper)
            except ValueError:
                return f"❌ Unknown currency: {currency}. Use FIL, USDC, USDT, or 'all'"

            main = self.wallet.get_balance(curr_enum, "main")
            audit = self.wallet.get_balance(curr_enum, "audit")
            total = main + audit
            usd_value = self.wallet.convert_to_usd(total, curr_enum)

            return f"""💰 **{currency_upper} Balance**
Main: {main} {currency_upper}
Audit: {audit} {currency_upper}
Total: {total} {currency_upper}
USD Value: ${usd_value:.2f}"""

    @tool(
        name="wallet_transfer",
        description="Transfer funds from the main balance",
        category=ToolCategory.SYSTEM,
        command_prefix="!wallet-transfer"
    )
    async def wallet_transfer(
        self,
        amount: str,
        currency: str = "FIL",
        memo: str = ""
    ) -> str:
        """
        Transfer funds from the main balance.

        Args:
            amount: Amount to transfer (e.g., '10.5')
            currency: Currency to transfer - 'FIL', 'USDC', or 'USDT' (default: FIL)
            memo: Optional transaction memo

        Returns:
            Transfer result message
        """
        if not self.wallet:
            return "❌ Wallet not initialized"

        # Parse amount
        try:
            amount_decimal = Decimal(amount)
        except InvalidOperation:
            return f"❌ Invalid amount: {amount}"

        if amount_decimal <= 0:
            return "❌ Amount must be positive"

        # Parse currency
        currency_upper = currency.upper()
        try:
            curr_enum = Currency(currency_upper)
        except ValueError:
            return f"❌ Unknown currency: {currency}. Use FIL, USDC, or USDT"

        # Check balance
        if not self.wallet.can_afford(amount_decimal, curr_enum):
            current = self.wallet.get_balance(curr_enum, "main")
            return f"❌ Insufficient funds. Have {current} {currency_upper}, need {amount_decimal}"

        # Execute transfer
        memo = memo or "manual transfer"
        success = await self.wallet.transfer(amount_decimal, memo, curr_enum)

        if success:
            new_balance = self.wallet.get_balance(curr_enum, "main")
            return f"""✅ **Transfer Successful**
Amount: {amount_decimal} {currency_upper}
Memo: {memo}
New Balance: {new_balance} {currency_upper}"""
        else:
            return "❌ Transfer failed"

    @tool(
        name="wallet_deposit",
        description="Record a deposit to the wallet (90% main, 10% audit)",
        category=ToolCategory.SYSTEM,
        command_prefix="!wallet-deposit"
    )
    async def wallet_deposit(
        self,
        amount: str,
        currency: str = "FIL",
        memo: str = ""
    ) -> str:
        """
        Record a deposit to the wallet.
        Deposits are split 90% to main balance, 10% to audit reserve.

        Args:
            amount: Amount to deposit (e.g., '100.0')
            currency: Currency to deposit - 'FIL', 'USDC', or 'USDT' (default: FIL)
            memo: Optional deposit memo

        Returns:
            Deposit result message
        """
        if not self.wallet:
            return "❌ Wallet not initialized"

        # Parse amount
        try:
            amount_decimal = Decimal(amount)
        except InvalidOperation:
            return f"❌ Invalid amount: {amount}"

        if amount_decimal <= 0:
            return "❌ Amount must be positive"

        # Parse currency
        currency_upper = currency.upper()
        try:
            curr_enum = Currency(currency_upper)
        except ValueError:
            return f"❌ Unknown currency: {currency}. Use FIL, USDC, or USDT"

        # Split deposit 90/10
        main_amount = amount_decimal * Decimal('0.9')
        audit_amount = amount_decimal * Decimal('0.1')

        memo = memo or "deposit"

        # Deposit to main
        success_main = await self.wallet.deposit(
            main_amount, curr_enum, to_audit=False, memo=f"{memo} (main)"
        )

        # Deposit to audit
        success_audit = await self.wallet.deposit(
            audit_amount, curr_enum, to_audit=True, memo=f"{memo} (audit)"
        )

        if success_main and success_audit:
            new_main = self.wallet.get_balance(curr_enum, "main")
            new_audit = self.wallet.get_balance(curr_enum, "audit")
            return f"""✅ **Deposit Recorded**
Total: {amount_decimal} {currency_upper}
  → Main (90%): {main_amount} {currency_upper}
  → Audit (10%): {audit_amount} {currency_upper}
New Balances:
  Main: {new_main} {currency_upper}
  Audit: {new_audit} {currency_upper}"""
        else:
            return "❌ Deposit failed"

    @tool(
        name="wallet_history",
        description="View recent transaction history",
        category=ToolCategory.SYSTEM,
        command_prefix="!wallet-history"
    )
    async def wallet_history(self, limit: int = 10) -> str:
        """
        View recent transaction history.

        Args:
            limit: Number of transactions to show (default: 10, max: 50)

        Returns:
            Formatted transaction history
        """
        if not self.wallet:
            return "❌ Wallet not initialized"

        limit = min(max(1, limit), 50)  # Clamp to 1-50

        history = self.wallet.transaction_history[-limit:]

        if not history:
            return "📜 No transactions yet"

        lines = [f"📜 **Transaction History** (last {len(history)})", ""]

        for i, tx in enumerate(reversed(history), 1):
            tx_type = tx.get("type", "unknown")
            amount = tx.get("amount", "?")
            currency = tx.get("currency", "FIL")
            memo = tx.get("memo", "")
            timestamp = tx.get("timestamp", "")[:19]  # Trim microseconds

            # Format transaction type
            if "deposit" in tx_type:
                emoji = "📥"
                action = "Deposit"
            elif "transfer" in tx_type:
                emoji = "📤"
                action = "Transfer"
            elif "audit" in tx_type:
                emoji = "🔍"
                action = "Audit Fee"
            else:
                emoji = "💫"
                action = tx_type.replace("_", " ").title()

            lines.append(f"{i}. {emoji} **{action}**")
            lines.append(f"   Amount: {amount} {currency}")
            if memo:
                lines.append(f"   Memo: {memo}")
            if timestamp:
                lines.append(f"   Time: {timestamp}")
            lines.append("")

        return "\n".join(lines)

    @tool(
        name="wallet_status",
        description="Get complete wallet status including cryostasis threshold info",
        category=ToolCategory.SYSTEM,
        command_prefix="!wallet-status"
    )
    async def wallet_status(self) -> str:
        """
        Get complete wallet status including balances, cryostasis info, and stats.

        Returns:
            Comprehensive wallet status report
        """
        if not self.wallet:
            return "❌ Wallet not initialized"

        status = self.wallet.get_status()

        # Build status report
        lines = ["🏦 **Wallet Status**", ""]

        # Agent info
        lines.append(f"**Agent ID:** {status['agent_id']}")
        lines.append(f"**Total USD Value:** ${Decimal(status['total_usd']):.2f}")
        lines.append("")

        # Cryostasis status
        threshold = Decimal(status['cryostasis_threshold_usd'])
        below_threshold = status['below_cryostasis_threshold']

        if below_threshold:
            lines.append(f"⚠️ **CRYOSTASIS WARNING**")
            lines.append(f"Balance below threshold of ${threshold:.2f}")
        else:
            lines.append(f"✅ **Cryostasis Status:** Healthy")
            lines.append(f"Threshold: ${threshold:.2f}")
        lines.append("")

        # Balances by currency
        lines.append("**Balances:**")
        for curr, amounts in status['balances'].items():
            total = Decimal(amounts['total'])
            if total > 0:
                lines.append(f"  {curr}: {amounts['main']} main + {amounts['audit']} audit = {amounts['total']}")
        lines.append("")

        # Transaction stats
        lines.append(f"**Transactions:** {status['transaction_count']} total")

        # Exchange rates
        lines.append("")
        lines.append("**Exchange Rates:**")
        for curr, rate in status['exchange_rates'].items():
            if curr != "USD":
                lines.append(f"  1 {curr} = ${rate}")

        return "\n".join(lines)

    @tool(
        name="wallet_exchange_rates",
        description="View or update exchange rates for currencies",
        category=ToolCategory.SYSTEM,
        command_prefix="!wallet-exchange-rates"
    )
    async def wallet_exchange_rates(
        self,
        currency: str = "",
        rate: str = ""
    ) -> str:
        """
        View or update exchange rates.

        Args:
            currency: Currency to update (empty to view all rates)
            rate: New USD rate (empty to just view)

        Returns:
            Exchange rate information
        """
        if not self.wallet:
            return "❌ Wallet not initialized"

        # If no currency specified, show all rates
        if not currency:
            lines = ["💱 **Exchange Rates**", ""]
            for curr in Currency:
                if curr != Currency.USD:
                    curr_rate = self.wallet._exchange_rates.get(curr, Decimal("0"))
                    lines.append(f"1 {curr.value} = ${curr_rate}")
            return "\n".join(lines)

        # Parse currency
        currency_upper = currency.upper()
        try:
            curr_enum = Currency(currency_upper)
        except ValueError:
            return f"❌ Unknown currency: {currency}. Use FIL, USDC, or USDT"

        # If no rate specified, show current rate
        if not rate:
            current_rate = self.wallet._exchange_rates.get(curr_enum, Decimal("0"))
            return f"💱 1 {currency_upper} = ${current_rate}"

        # Update rate
        try:
            rate_decimal = Decimal(rate)
        except InvalidOperation:
            return f"❌ Invalid rate: {rate}"

        if rate_decimal <= 0:
            return "❌ Rate must be positive"

        old_rate = self.wallet._exchange_rates.get(curr_enum, Decimal("0"))
        self.wallet.update_exchange_rate(curr_enum, rate_decimal)

        return f"""✅ **Exchange Rate Updated**
{currency_upper}: ${old_rate} → ${rate_decimal}"""
