"""
Filecoin Testnet Tool Methods for WalletFeature.

Provides @tool-decorated methods for Filecoin Calibration testnet operations:
- wallet_generate_address: Generate a new testnet address
- wallet_sync: Sync on-chain balance
- wallet_address: Show address info and explorer links

Used as a mixin by WalletFeature.
"""

import logging
import os
from pathlib import Path

from kestrel_sdk.features.base import tool
from kestrel_sdk.tools.base import ToolCategory
from .feature import Currency

logger = logging.getLogger(__name__)


class FilecoinToolsMixin:
    """
    Mixin providing Filecoin testnet tools for WalletFeature.

    Requires self.wallet (WalletAgent) to be available.
    """

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
        except (ImportError, OSError) as e:
            logger.error(f"Failed to generate address: {e}", exc_info=True)
            return f"❌ Failed to generate address: {e}"
        except Exception as e:
            logger.error(f"Failed to generate address: {e}", exc_info=True)
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

        except (ConnectionError, TimeoutError) as e:
            logger.error(f"Sync failed (network error): {e}", exc_info=True)
            return f"❌ Sync failed: {e}"
        except (KeyError, ValueError, TypeError) as e:
            logger.error(f"Sync failed (data parsing error): {e}", exc_info=True)
            return f"❌ Sync failed: {e}"
        except Exception as e:
            logger.error(f"Sync failed: {e}", exc_info=True)
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
