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

from kestrel_sovereign.features.base import tool
from kestrel_sdk.tools.base import ToolCategory
from kestrel_sdk.tools.result import ToolResult
from .feature import Currency


def _is_error_string(text: str) -> bool:
    """Heuristic for the legacy stringly-typed ``❌``-prefix error format
    that wallet tools used before #1061 wave 32. We keep the strings
    intact (they're already user-facing) but route them to the right
    envelope status based on the prefix."""
    return text.lstrip().startswith("❌")

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
    async def wallet_generate_address(self) -> ToolResult:
        """
        Generate a new Filecoin Calibration testnet address.

        Creates a secp256k1 keypair and derives a t1... address.
        The private key is stored encrypted if KESTREL_DATA_KEY is set.

        Returns:
            ToolResult.ok with the generated address + faucet
            instructions; ToolResult.failed when the wallet isn't
            initialized or address generation raises.
        """
        if not self.wallet:
            return ToolResult.failed(error="❌ Wallet not initialized")

        # Check if already has address — this is OK (idempotent),
        # but we surface it through the OK confirmation so the LLM
        # tells the sovereign "you already have one" instead of
        # claiming a new address was generated.
        if self.wallet.filecoin_address:
            existing_address = self.wallet.filecoin_address
            return ToolResult.ok(
                f"""ℹ️ **Wallet Already Has Address**
Address: `{existing_address}`

View on explorer: https://calibration.filfox.info/en/address/{existing_address}

Use `!wallet-sync` to sync with on-chain balance.""",
                data={"already_existed": True, "address": existing_address},
            )

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

            return ToolResult.ok(
                f"""✅ **Filecoin Address Generated**

**Address:** `{address}`

**Network:** Calibration (testnet)

**Next Steps:**
1. Visit the faucet: {faucet_url}
2. Enter your address: `{address}`
3. Complete the captcha and submit
4. Wait ~2 minutes for confirmation
5. Run `!wallet-sync` to update balance

**View on Explorer:** {explorer_url}

⚠️ This is testnet FIL with no real value. For mainnet, use a proper wallet.""",
                data={
                    "address": address,
                    "already_existed": False,
                    "faucet_url": faucet_url,
                    "explorer_url": explorer_url,
                },
            )

        except ValueError as e:
            return ToolResult.failed(error=f"❌ {e}")
        except (ImportError, OSError) as e:
            logger.error(f"Failed to generate address: {e}", exc_info=True)
            return ToolResult.failed(error=f"❌ Failed to generate address: {e}")
        except Exception as e:
            logger.error(f"Failed to generate address: {e}", exc_info=True)
            return ToolResult.failed(error=f"❌ Failed to generate address: {e}")

    @tool(
        name="wallet_sync",
        description="Sync wallet balance with Filecoin testnet",
        category=ToolCategory.SYSTEM,
        command_prefix="!wallet-sync"
    )
    async def wallet_sync(self) -> ToolResult:
        """
        Sync wallet balance with Filecoin Calibration testnet.

        Queries the on-chain balance and updates the internal wallet
        to match. Detects deposits and withdrawals.

        Returns:
            ToolResult.ok on a clean sync (with deposit/withdrawal
            detection in the confirmation). ToolResult.failed for
            wallet-not-initialized, no-address, on-chain-query-
            failure, and the underlying ``sync_on_chain_balance``
            returning False.
        """
        if not self.wallet:
            return ToolResult.failed(error="❌ Wallet not initialized")

        if not self.wallet.filecoin_address:
            return ToolResult.failed(
                error="""❌ **No Filecoin Address Configured**

Generate one with: `!wallet-generate-address`

Or manually set with an existing address."""
            )

        try:
            # Get current internal balance
            internal_balance = self.wallet.get_total_balance(Currency.FIL)

            # Get on-chain balance
            on_chain_balance = await self.wallet.get_on_chain_balance()
            if on_chain_balance is None:
                return ToolResult.failed(
                    error="❌ Failed to query on-chain balance. Check network connectivity."
                )

            # Sync
            success = await self.wallet.sync_on_chain_balance()

            if not success:
                return ToolResult.failed(error="❌ Sync failed. Check logs for details.")

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

            return ToolResult.ok(
                "\n".join(lines),
                data={
                    "address": self.wallet.filecoin_address,
                    "on_chain_balance": str(on_chain_balance),
                    "internal_balance": str(self.wallet.get_total_balance(Currency.FIL)),
                    "difference": str(difference),
                    "total_usd": f"{self.wallet.get_total_balance_usd():.2f}",
                },
            )

        except (ConnectionError, TimeoutError) as e:
            logger.error(f"Sync failed (network error): {e}", exc_info=True)
            return ToolResult.failed(error=f"❌ Sync failed: {e}")
        except (KeyError, ValueError, TypeError) as e:
            logger.error(f"Sync failed (data parsing error): {e}", exc_info=True)
            return ToolResult.failed(error=f"❌ Sync failed: {e}")
        except Exception as e:
            logger.error(f"Sync failed: {e}", exc_info=True)
            return ToolResult.failed(error=f"❌ Sync failed: {e}")

    @tool(
        name="wallet_address",
        description="Show Filecoin address and explorer link",
        category=ToolCategory.SYSTEM,
        command_prefix="!wallet-address"
    )
    async def wallet_address(self) -> ToolResult:
        """
        Show the wallet's Filecoin address and helpful links.

        Returns:
            ToolResult.ok with address info and explorer/faucet
            links; PARTIAL when the on-chain balance query failed
            (the address is valid and the balance shown will say
            "Unable to query" — the LLM should not promise the
            user a balance number from this call); ToolResult.failed
            for wallet-not-initialized and no-address paths.
        """
        if not self.wallet:
            return ToolResult.failed(error="❌ Wallet not initialized")

        if not self.wallet.filecoin_address:
            return ToolResult.failed(
                error="""❌ **No Filecoin Address**

Generate one with: `!wallet-generate-address`"""
            )

        address = self.wallet.filecoin_address
        explorer_url = f"https://calibration.filfox.info/en/address/{address}"
        faucet_url = "https://faucet.calibnet.chainsafe-fil.io/"

        # Try to get on-chain balance
        on_chain = await self.wallet.get_on_chain_balance()
        balance_str = f"{on_chain} FIL" if on_chain is not None else "Unable to query"

        confirmation = f"""🔗 **Filecoin Address**

**Address:** `{address}`
**Network:** Calibration (testnet)
**On-chain Balance:** {balance_str}

**Links:**
- [View on Explorer]({explorer_url})
- [Request Test FIL]({faucet_url})

Use `!wallet-sync` to sync on-chain balance with wallet."""

        data = {
            "address": address,
            "explorer_url": explorer_url,
            "faucet_url": faucet_url,
            "on_chain_balance": str(on_chain) if on_chain is not None else None,
        }
        if on_chain is None:
            return ToolResult.partial(
                confirmation,
                (
                    "on-chain balance query failed (network/RPC issue) — "
                    "the address itself is valid but the balance shown is "
                    "stale. Retry !wallet-sync once connectivity is back."
                ),
                data=data,
            )
        return ToolResult.ok(confirmation, data=data)
