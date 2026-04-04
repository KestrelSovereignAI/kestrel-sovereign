"""
Multi-Chain Transaction Tool Methods for WalletFeature.

Provides @tool-decorated methods for EVM chain operations:
- wallet_send: Send native tokens (ETH, FIL, MATIC)
- wallet_send_token: Send ERC-20 tokens (USDC, USDT, DAI)
- wallet_networks: List available blockchain networks
- wallet_tx_history: View blockchain transaction history

Used as a mixin by WalletFeature.
"""

import logging
import os
from decimal import Decimal, InvalidOperation
from pathlib import Path

from kestrel_sdk.features.base import tool
from kestrel_sdk.tools.base import ToolCategory

logger = logging.getLogger(__name__)


class MultichainToolsMixin:
    """
    Mixin providing multi-chain transaction tools for WalletFeature.

    Requires self.wallet (WalletAgent) to be available.
    """

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
            from .transaction_manager import TransactionManager
            from .chain_adapters import ChainNetwork
            from .filecoin_keys import FilecoinKeyManager

            # Parse amount
            try:
                amount_decimal = Decimal(amount)
            except (ValueError, InvalidOperation):
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

        except ValueError as e:
            logger.error(f"wallet_send failed (validation error): {e}", exc_info=True)
            return f"❌ Transaction failed: {e}"
        except (ConnectionError, TimeoutError) as e:
            logger.error(f"wallet_send failed (network error): {e}", exc_info=True)
            return f"❌ Transaction failed: {e}"
        except (ImportError, OSError) as e:
            logger.error(f"wallet_send failed (system error): {e}", exc_info=True)
            return f"❌ Transaction failed: {e}"
        except Exception as e:
            logger.error(f"wallet_send failed: {e}", exc_info=True)
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
            from .transaction_manager import TransactionManager
            from .chain_adapters import ChainNetwork, TokenRegistry
            from .filecoin_keys import FilecoinKeyManager

            # Parse amount
            try:
                amount_decimal = Decimal(amount)
            except (ValueError, InvalidOperation):
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

        except ValueError as e:
            logger.error(f"wallet_send_token failed (validation error): {e}", exc_info=True)
            return f"❌ Token transfer failed: {e}"
        except (ConnectionError, TimeoutError) as e:
            logger.error(f"wallet_send_token failed (network error): {e}", exc_info=True)
            return f"❌ Token transfer failed: {e}"
        except (ImportError, OSError) as e:
            logger.error(f"wallet_send_token failed (system error): {e}", exc_info=True)
            return f"❌ Token transfer failed: {e}"
        except Exception as e:
            logger.error(f"wallet_send_token failed: {e}", exc_info=True)
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

        except (ImportError, OSError) as e:
            logger.error(f"wallet_tx_history failed (system error): {e}", exc_info=True)
            return f"❌ Failed to get transaction history: {e}"
        except (KeyError, ValueError, AttributeError) as e:
            logger.error(f"wallet_tx_history failed (data error): {e}", exc_info=True)
            return f"❌ Failed to get transaction history: {e}"
        except Exception as e:
            logger.error(f"wallet_tx_history failed: {e}", exc_info=True)
            return f"❌ Failed to get transaction history: {e}"
