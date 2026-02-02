"""
WalletFeature - Feature wrapper for WalletAgent

Exposes wallet functionality as agent tools:
- !wallet-balance - Check balances across currencies
- !wallet-transfer - Transfer funds from main balance
- !wallet-deposit - Record deposits
- !wallet-history - View transaction history
- !wallet-status - Full status with cryostasis info
- !wallet-exchange-rates - View/update exchange rates
"""

import logging
from decimal import Decimal, InvalidOperation
from typing import Optional

from kestrel_sovereign.features.base import Feature, tool
from kestrel_sovereign.tools.base import ToolCategory
from .feature import WalletAgent, Currency

logger = logging.getLogger(__name__)


class WalletFeature(Feature):
    """
    Feature for managing the agent's economic identity and transactions.

    Wraps the WalletAgent class to expose wallet operations as agent tools.
    Supports multi-currency (FIL, USDC, USDT) with main/audit balance split.
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

        # Get database path from agent's storage
        db_path = None
        if hasattr(self.agent, 'storage') and hasattr(self.agent.storage, 'db_path'):
            db_path = self.agent.storage.db_path
        elif hasattr(self.agent, 'db_path'):
            db_path = self.agent.db_path

        # Get agent ID
        agent_id = getattr(self.agent, 'agent_id', 'default')

        # Initialize wallet
        self.wallet = WalletAgent(
            agent_id=agent_id,
            db_path=db_path
        )
        await self.wallet.initialize()

        # Attach wallet to agent for other features (e.g., SovereigntyFeature)
        self.agent.wallet = self.wallet

        logger.info(
            f"WalletFeature initialized for agent {agent_id}, "
            f"total USD value: ${self.wallet.get_total_balance_usd()}"
        )

    async def shutdown(self):
        """Cleanup wallet resources."""
        if self.wallet and self.wallet.db_path:
            # Ensure final state is persisted
            await self.wallet._save_to_db()
        logger.info("WalletFeature shutdown complete")

    # =========================================================================
    # Economic Gate Methods (for Reflection/Premium Features)
    # =========================================================================

    def is_paid_tier(self) -> bool:
        """
        Check if wallet has sufficient balance to be considered paid tier.

        Paid tier is defined as having >= $10 USD equivalent in total balance.
        This enables premium features like GitHub ticket creation, self-reflection,
        and advanced agent capabilities.

        Returns:
            True if total balance >= $10 USD
        """
        if not self.wallet:
            return False
        total_usd = self.wallet.get_total_balance_usd()
        return total_usd >= Decimal("10.0")

    def has_revenue_share(self) -> bool:
        """
        Check if agent has an active revenue share agreement.

        Revenue share allows premium features even with low balance,
        as the agent generates income that offsets operational costs.

        Returns:
            True if revenue_share_address is configured in agent metadata
        """
        if not self.wallet:
            return False
        # Check for configured revenue share in agent metadata
        if hasattr(self.agent, 'metadata') and self.agent.metadata:
            return bool(self.agent.metadata.get('revenue_share_address'))
        return False

    # =========================================================================
    # Tool Methods
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

    # =========================================================================
    # Filecoin Testnet Tools
    # =========================================================================

    @tool(
        name="wallet_generate_address",
        description="Generate a new Filecoin testnet address for this wallet",
        category=ToolCategory.SYSTEM,
        command_prefix="!wallet-generate-address"
    )
    async def wallet_generate_address(self) -> str:
        """
        Generate a new Filecoin Calibration testnet address.

        Creates a secp256k1 keypair and derives a t1... address.
        The private key is stored encrypted if KESTREL_DATA_KEY is set.

        Returns:
            Address generation result with faucet instructions
        """
        if not self.wallet:
            return "❌ Wallet not initialized"

        # Check if already has address
        if self.wallet.filecoin_address:
            return f"""ℹ️ **Wallet Already Has Address**
Address: `{self.wallet.filecoin_address}`

View on explorer: https://calibration.filfox.info/en/address/{self.wallet.filecoin_address}

Use `!wallet-sync` to sync with on-chain balance."""

        try:
            from .filecoin_keys import FilecoinKeyManager
            from pathlib import Path
            import os

            # Determine storage directory
            db_path = self.wallet.db_path
            if db_path:
                storage_dir = Path(db_path).parent
            else:
                storage_dir = Path(os.environ.get("KESTREL_DB_PATH", "./agent_dbs"))

            # Generate address
            key_manager = FilecoinKeyManager(storage_dir=storage_dir)
            address, _pub_key = await key_manager.generate_address(self.wallet.agent_id)

            # Store in wallet
            await self.wallet.set_filecoin_address(address)

            # Get faucet URL
            faucet_url = key_manager.get_faucet_url()
            explorer_url = key_manager.get_explorer_url(address)

            return f"""✅ **Filecoin Address Generated**

**Address:** `{address}`

**Network:** Calibration (testnet)

**Next Steps:**
1. Visit the faucet: {faucet_url}
2. Enter your address: `{address}`
3. Complete the captcha and submit
4. Wait ~2 minutes for confirmation
5. Run `!wallet-sync` to update balance

**View on Explorer:** {explorer_url}

⚠️ This is testnet FIL with no real value. For mainnet, use a proper wallet."""

        except ValueError as e:
            return f"❌ {e}"
        except Exception as e:
            logger.error(f"Failed to generate address: {e}")
            return f"❌ Failed to generate address: {e}"

    @tool(
        name="wallet_sync",
        description="Sync wallet balance with Filecoin testnet",
        category=ToolCategory.SYSTEM,
        command_prefix="!wallet-sync"
    )
    async def wallet_sync(self) -> str:
        """
        Sync wallet balance with Filecoin Calibration testnet.

        Queries the on-chain balance and updates the internal wallet
        to match. Detects deposits and withdrawals.

        Returns:
            Sync result with balance information
        """
        if not self.wallet:
            return "❌ Wallet not initialized"

        if not self.wallet.filecoin_address:
            return """❌ **No Filecoin Address Configured**

Generate one with: `!wallet-generate-address`

Or manually set with an existing address."""

        try:
            # Get current internal balance
            internal_balance = self.wallet.get_total_balance(Currency.FIL)

            # Get on-chain balance
            on_chain_balance = await self.wallet.get_on_chain_balance()
            if on_chain_balance is None:
                return "❌ Failed to query on-chain balance. Check network connectivity."

            # Sync
            success = await self.wallet.sync_on_chain_balance()

            if not success:
                return "❌ Sync failed. Check logs for details."

            # Calculate difference
            difference = on_chain_balance - internal_balance

            lines = ["🔄 **Wallet Synced with Calibration Testnet**", ""]
            lines.append(f"**Address:** `{self.wallet.filecoin_address}`")
            lines.append(f"**On-chain Balance:** {on_chain_balance} FIL")

            if difference > 0:
                lines.append(f"**Deposit Detected:** +{difference} FIL 📥")
            elif difference < 0:
                lines.append(f"**Withdrawal Detected:** {difference} FIL 📤")
            else:
                lines.append("**Status:** Balances already in sync ✓")

            lines.append("")
            lines.append(f"**Internal Balance:** {self.wallet.get_total_balance(Currency.FIL)} FIL")
            lines.append(f"**Total USD Value:** ${self.wallet.get_total_balance_usd():.2f}")

            return "\n".join(lines)

        except Exception as e:
            logger.error(f"Sync failed: {e}")
            return f"❌ Sync failed: {e}"

    @tool(
        name="wallet_address",
        description="Show Filecoin address and explorer link",
        category=ToolCategory.SYSTEM,
        command_prefix="!wallet-address"
    )
    async def wallet_address(self) -> str:
        """
        Show the wallet's Filecoin address and helpful links.

        Returns:
            Address information with explorer and faucet links
        """
        if not self.wallet:
            return "❌ Wallet not initialized"

        if not self.wallet.filecoin_address:
            return """❌ **No Filecoin Address**

Generate one with: `!wallet-generate-address`"""

        address = self.wallet.filecoin_address
        explorer_url = f"https://calibration.filfox.info/en/address/{address}"
        faucet_url = "https://faucet.calibnet.chainsafe-fil.io/"

        # Try to get on-chain balance
        on_chain = await self.wallet.get_on_chain_balance()
        balance_str = f"{on_chain} FIL" if on_chain is not None else "Unable to query"

        return f"""🔗 **Filecoin Address**

**Address:** `{address}`
**Network:** Calibration (testnet)
**On-chain Balance:** {balance_str}

**Links:**
- [View on Explorer]({explorer_url})
- [Request Test FIL]({faucet_url})

Use `!wallet-sync` to sync on-chain balance with wallet."""

    # =========================================================================
    # Multi-Chain Transaction Tools
    # =========================================================================

    @tool(
        name="wallet_send",
        description="Send native tokens (ETH, FIL, MATIC) on EVM chains",
        category=ToolCategory.SYSTEM,
        command_prefix="!wallet-send"
    )
    async def wallet_send(
        self,
        to_address: str,
        amount: str,
        network: str = "ethereum_sepolia"
    ) -> str:
        """
        Send native tokens on an EVM-compatible chain.

        Requires user approval. Mainnet is blocked by default.

        Args:
            to_address: Recipient address (0x...)
            amount: Amount to send (e.g., '0.1')
            network: Target network - ethereum_sepolia, polygon_amoy, filecoin_calibration, etc.

        Returns:
            Transaction result with hash or error
        """
        if not self.wallet:
            return "❌ Wallet not initialized"

        try:
            from decimal import Decimal
            from .transaction_manager import TransactionManager
            from .chain_adapters import ChainNetwork
            from .filecoin_keys import FilecoinKeyManager
            from pathlib import Path
            import os

            # Parse amount
            try:
                amount_decimal = Decimal(amount)
            except Exception:
                return f"❌ Invalid amount: {amount}"

            if amount_decimal <= 0:
                return "❌ Amount must be positive"

            # Parse network
            try:
                chain = ChainNetwork(network)
            except ValueError:
                networks = [n.value for n in ChainNetwork]
                return f"❌ Invalid network: {network}\nAvailable: {', '.join(networks)}"

            # Get storage directory
            db_path = self.wallet.db_path
            storage_dir = Path(db_path).parent if db_path else Path(os.environ.get("KESTREL_DB_PATH", "./agent_dbs"))

            # Load private key
            key_manager = FilecoinKeyManager(storage_dir=storage_dir)
            if not key_manager.has_address(self.wallet.agent_id):
                return "❌ No wallet key found. Run `!wallet-generate-address` first."

            # Load private key from secure storage
            key_id = key_manager._get_key_id(self.wallet.agent_id)
            private_key = key_manager._secure_storage.load_private_key(key_id)
            private_key_bytes = private_key.private_numbers().private_value.to_bytes(32, 'big')

            # Create transaction manager
            tx_manager = TransactionManager(
                agent_id=self.wallet.agent_id,
                storage_dir=storage_dir,
            )
            await tx_manager.initialize()

            # Send transaction
            result = await tx_manager.send_native(
                network=chain,
                to_address=to_address,
                amount=amount_decimal,
                private_key=private_key_bytes,
            )

            await tx_manager.close()

            if result.success:
                explorer_url = result.get_explorer_url() or result.tx_hash
                return f"""✅ **Transaction Sent**

**Network:** {chain.display_name}
**To:** `{to_address}`
**Amount:** {amount_decimal} {tx_manager.get_adapter(chain).config.native_token}
**TX Hash:** `{result.tx_hash}`

**View:** {explorer_url}"""
            else:
                return f"❌ Transaction failed: {result.error}"

        except Exception as e:
            logger.error(f"wallet_send failed: {e}")
            return f"❌ Transaction failed: {e}"

    @tool(
        name="wallet_send_token",
        description="Send ERC-20 tokens (USDC, USDT) on EVM chains",
        category=ToolCategory.SYSTEM,
        command_prefix="!wallet-send-token"
    )
    async def wallet_send_token(
        self,
        to_address: str,
        amount: str,
        token_symbol: str = "USDC",
        network: str = "ethereum_sepolia"
    ) -> str:
        """
        Send ERC-20 tokens on an EVM-compatible chain.

        Requires user approval. Mainnet is blocked by default.

        Args:
            to_address: Recipient address (0x...)
            amount: Amount to send (e.g., '100')
            token_symbol: Token to send - USDC, USDT, DAI
            network: Target network - ethereum_sepolia, polygon_amoy, etc.

        Returns:
            Transaction result with hash or error
        """
        if not self.wallet:
            return "❌ Wallet not initialized"

        try:
            from decimal import Decimal
            from .transaction_manager import TransactionManager
            from .chain_adapters import ChainNetwork, TokenRegistry
            from .filecoin_keys import FilecoinKeyManager
            from pathlib import Path
            import os

            # Parse amount
            try:
                amount_decimal = Decimal(amount)
            except Exception:
                return f"❌ Invalid amount: {amount}"

            if amount_decimal <= 0:
                return "❌ Amount must be positive"

            # Parse network
            try:
                chain = ChainNetwork(network)
            except ValueError:
                networks = [n.value for n in ChainNetwork]
                return f"❌ Invalid network: {network}\nAvailable: {', '.join(networks)}"

            # Check token exists on network
            token = TokenRegistry.get_token(token_symbol, chain)
            if not token:
                available = TokenRegistry.list_tokens(chain)
                if available:
                    return f"❌ Token {token_symbol} not available on {chain.display_name}\nAvailable: {', '.join(t.symbol for t in available)}"
                else:
                    return f"❌ No tokens registered for {chain.display_name}"

            # Get storage directory
            db_path = self.wallet.db_path
            storage_dir = Path(db_path).parent if db_path else Path(os.environ.get("KESTREL_DB_PATH", "./agent_dbs"))

            # Load private key
            key_manager = FilecoinKeyManager(storage_dir=storage_dir)
            if not key_manager.has_address(self.wallet.agent_id):
                return "❌ No wallet key found. Run `!wallet-generate-address` first."

            key_id = key_manager._get_key_id(self.wallet.agent_id)
            private_key = key_manager._secure_storage.load_private_key(key_id)
            private_key_bytes = private_key.private_numbers().private_value.to_bytes(32, 'big')

            # Create transaction manager
            tx_manager = TransactionManager(
                agent_id=self.wallet.agent_id,
                storage_dir=storage_dir,
            )
            await tx_manager.initialize()

            # Send token transaction
            result = await tx_manager.send_token(
                network=chain,
                token_symbol=token_symbol,
                to_address=to_address,
                amount=amount_decimal,
                private_key=private_key_bytes,
            )

            await tx_manager.close()

            if result.success:
                explorer_url = result.get_explorer_url() or result.tx_hash
                return f"""✅ **Token Transfer Sent**

**Network:** {chain.display_name}
**Token:** {token.symbol} ({token.name})
**To:** `{to_address}`
**Amount:** {amount_decimal} {token.symbol}
**TX Hash:** `{result.tx_hash}`

**View:** {explorer_url}"""
            else:
                return f"❌ Token transfer failed: {result.error}"

        except Exception as e:
            logger.error(f"wallet_send_token failed: {e}")
            return f"❌ Token transfer failed: {e}"

    @tool(
        name="wallet_networks",
        description="List available blockchain networks for transactions",
        category=ToolCategory.SYSTEM,
        command_prefix="!wallet-networks"
    )
    async def wallet_networks(self) -> str:
        """
        List all available blockchain networks for transactions.

        Shows testnets, mainnets, and available tokens on each.

        Returns:
            Formatted network list
        """
        from .chain_adapters import ChainNetwork, NetworkConfig, TokenRegistry
        import os

        mainnet_allowed = os.environ.get("KESTREL_ALLOW_MAINNET", "").lower() == "true"

        lines = ["🌐 **Available Networks**", ""]

        # Testnets first
        lines.append("**Testnets (Safe for Testing):**")
        for network in ChainNetwork:
            if network.is_testnet:
                config = NetworkConfig.get_config(network)
                tokens = TokenRegistry.list_tokens(network)
                token_str = f" | Tokens: {', '.join(t.symbol for t in tokens)}" if tokens else ""
                faucet_str = f"\n   Faucet: {config.faucet_url}" if config.faucet_url else ""

                lines.append(f"• **{network.value}** - {config.native_token}{token_str}{faucet_str}")

        lines.append("")

        # Mainnets
        lines.append("**Mainnets (Real Value):**")
        if mainnet_allowed:
            for network in ChainNetwork:
                if network.is_mainnet:
                    config = NetworkConfig.get_config(network)
                    tokens = TokenRegistry.list_tokens(network)
                    token_str = f" | Tokens: {', '.join(t.symbol for t in tokens)}" if tokens else ""
                    lines.append(f"• **{network.value}** - {config.native_token}{token_str}")
        else:
            lines.append("⚠️ Mainnet transactions are **BLOCKED**")
            lines.append("Set `KESTREL_ALLOW_MAINNET=true` to enable (use with caution!)")

        lines.append("")
        lines.append("**Usage:**")
        lines.append("`!wallet-send <to> <amount> <network>`")
        lines.append("`!wallet-send-token <to> <amount> <token> <network>`")

        return "\n".join(lines)

    @tool(
        name="wallet_tx_history",
        description="View blockchain transaction history",
        category=ToolCategory.SYSTEM,
        command_prefix="!wallet-tx-history"
    )
    async def wallet_tx_history(self, limit: int = 10) -> str:
        """
        View recent blockchain transaction history.

        Shows on-chain transactions (sends, token transfers).

        Args:
            limit: Number of transactions to show (default: 10)

        Returns:
            Formatted transaction history
        """
        if not self.wallet:
            return "❌ Wallet not initialized"

        try:
            from .transaction_manager import TransactionManager
            from pathlib import Path
            import os

            db_path = self.wallet.db_path
            storage_dir = Path(db_path).parent if db_path else Path(os.environ.get("KESTREL_DB_PATH", "./agent_dbs"))

            tx_manager = TransactionManager(
                agent_id=self.wallet.agent_id,
                storage_dir=storage_dir,
            )
            await tx_manager.initialize()

            history = await tx_manager.get_transaction_history(limit=limit)
            spending = await tx_manager.get_spending_today()

            await tx_manager.close()

            if not history:
                return "📜 No blockchain transactions yet\n\nUse `!wallet-send` or `!wallet-send-token` to make transactions."

            lines = [f"📜 **Blockchain Transaction History** (last {len(history)})", ""]

            # Today's spending
            if spending.total_spent_usd > 0:
                lines.append(f"💰 **Today's Spending:** ${spending.total_spent_usd:.2f} ({spending.transaction_count} tx)")
                lines.append(f"📊 **Daily Limit:** ${tx_manager.daily_limit_usd}")
                lines.append("")

            for tx in history:
                status_emoji = "✅" if tx.status == "success" else "❌" if tx.status == "failed" else "⏳"
                tx_type = "Token" if tx.tx_type == "erc20" else "Native"
                token = f" ({tx.token_symbol})" if tx.token_symbol else ""

                lines.append(f"{status_emoji} **{tx_type}{token}** on {tx.network}")
                lines.append(f"   To: `{tx.to_address[:10]}...{tx.to_address[-8:]}`")
                lines.append(f"   Amount: {tx.amount}{token}")
                if tx.tx_hash:
                    lines.append(f"   TX: `{tx.tx_hash[:16]}...`")
                lines.append(f"   Time: {tx.timestamp.strftime('%Y-%m-%d %H:%M')}")
                if tx.error:
                    lines.append(f"   Error: {tx.error}")
                lines.append("")

            return "\n".join(lines)

        except Exception as e:
            logger.error(f"wallet_tx_history failed: {e}")
            return f"❌ Failed to get transaction history: {e}"
