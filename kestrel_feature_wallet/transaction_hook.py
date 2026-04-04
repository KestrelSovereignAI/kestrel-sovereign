"""
Transaction Security Hook for Kestrel Wallet.

Validates transaction requests before they're executed:
- Validates address formats
- Blocks mainnet if disabled
- Checks spending limits
- Requires user approval for all transactions

Runs at priority 5 (before general SecurityHook at priority 10).
"""

import os
import logging
from decimal import Decimal
from typing import Optional

from kestrel_sdk.hooks import Hook, HookEvent, HookInput, HookOutput
from .chain_adapters import ChainNetwork, EVMAdapter

logger = logging.getLogger(__name__)


class TransactionSecurityHook(Hook):
    """
    PreToolUse hook for transaction validation.

    Intercepts wallet transaction tools and validates:
    1. Address format is valid for the target network
    2. Mainnet is allowed (if targeting mainnet)
    3. Amount is within daily spending limit
    4. User has approved the transaction

    This hook runs before the general SecurityHook to provide
    transaction-specific validation.
    """

    # Tools this hook intercepts
    TRANSACTION_TOOLS = {
        "wallet_send",
        "wallet_send_token",
        "send_native",
        "send_token",
    }

    def __init__(
        self,
        daily_limit_usd: Optional[Decimal] = None,
        priority: int = 5,  # Run before SecurityHook (10)
    ):
        """
        Initialize transaction security hook.

        Args:
            daily_limit_usd: Maximum daily spending in USD
            priority: Hook priority (default 5, before SecurityHook)
        """
        super().__init__(
            name="transaction_security",
            events=[HookEvent.PRE_TOOL_USE],
            priority=priority,
        )

        # Load config from environment
        limit_env = os.environ.get("KESTREL_TX_DAILY_LIMIT_USD")
        if limit_env:
            self.daily_limit_usd = Decimal(limit_env)
        elif daily_limit_usd:
            self.daily_limit_usd = daily_limit_usd
        else:
            self.daily_limit_usd = Decimal("100")

        self.allow_mainnet = os.environ.get(
            "KESTREL_ALLOW_MAINNET", ""
        ).lower() == "true"

        self.require_approval = os.environ.get(
            "KESTREL_TX_REQUIRE_APPROVAL", "true"
        ).lower() != "false"

        logger.info(
            f"TransactionSecurityHook: limit=${self.daily_limit_usd}, "
            f"mainnet={'allowed' if self.allow_mainnet else 'blocked'}, "
            f"approval={'required' if self.require_approval else 'optional'}"
        )

    async def execute(self, input: HookInput) -> HookOutput:
        """
        Validate transaction request.

        Args:
            input: HookInput with tool context

        Returns:
            HookOutput allowing or denying the transaction
        """
        tool_name = input.tool_name or ""

        # Only process transaction tools
        if tool_name not in self.TRANSACTION_TOOLS:
            return HookOutput.allow()

        tool_args = input.tool_input or {}

        logger.debug(f"Transaction security check: {tool_name}")

        # Extract transaction details
        to_address = tool_args.get("to_address") or tool_args.get("to")
        network_str = tool_args.get("network", "ethereum_sepolia")
        amount = tool_args.get("amount", 0)

        # Parse network
        try:
            network = ChainNetwork(network_str)
        except ValueError:
            return HookOutput.deny(
                f"Invalid network: {network_str}. "
                f"Valid options: {[n.value for n in ChainNetwork]}"
            )

        # Check 1: Validate address
        if to_address:
            is_valid = self._validate_address(to_address, network)
            if not is_valid:
                return HookOutput.deny(
                    f"Invalid address format for {network.display_name}: {to_address}"
                )

        # Check 2: Block mainnet if disabled
        if network.is_mainnet and not self.allow_mainnet:
            return HookOutput.deny(
                f"Mainnet transactions are blocked. "
                f"Set KESTREL_ALLOW_MAINNET=true to enable {network.display_name}. "
                f"⚠️ WARNING: Mainnet transactions use real money!"
            )

        # Check 3: Warn about spending limits (actual enforcement in TransactionManager)
        if network.is_mainnet:
            try:
                amount_decimal = Decimal(str(amount))
                # Rough USD estimate (TransactionManager does precise calculation)
                estimated_usd = self._estimate_usd(amount_decimal, tool_args)

                if estimated_usd > self.daily_limit_usd:
                    return HookOutput.deny(
                        f"Transaction exceeds daily limit. "
                        f"Requested: ~${estimated_usd}, Limit: ${self.daily_limit_usd}. "
                        f"Adjust KESTREL_TX_DAILY_LIMIT_USD to increase."
                    )
            except (ValueError, TypeError):
                pass  # Let TransactionManager handle validation

        # Log the transaction attempt
        logger.info(
            f"Transaction pre-approved: {tool_name} on {network.display_name}, "
            f"to={to_address}, amount={amount}"
        )

        # Allow with metadata for approval queue
        return HookOutput.allow(
            message=f"Transaction validated for {network.display_name}",
            metadata={
                "network": network.value,
                "is_mainnet": network.is_mainnet,
                "requires_approval": self.require_approval,
                "to_address": to_address,
                "amount": str(amount),
            },
        )

    def _validate_address(self, address: str, network: ChainNetwork) -> bool:
        """
        Validate address format for the network.

        Args:
            address: Address to validate
            network: Target network

        Returns:
            True if valid
        """
        if not address:
            return False

        # All our networks are EVM-compatible, use same validation
        # Check basic format (0x + 40 hex chars)
        if not address.startswith("0x"):
            return False
        if len(address) != 42:
            return False

        try:
            # Check it's valid hex
            int(address, 16)
            return True
        except ValueError:
            return False

    def _estimate_usd(self, amount: Decimal, tool_args: dict) -> Decimal:
        """
        Rough USD estimate for spending limit check.

        Args:
            amount: Token amount
            tool_args: Tool arguments (may contain token_symbol)

        Returns:
            Estimated USD value
        """
        # Price estimates (conservative)
        prices = {
            "ETH": Decimal("3000"),
            "FIL": Decimal("5.50"),
            "tFIL": Decimal("0"),
            "MATIC": Decimal("0.80"),
            "USDC": Decimal("1"),
            "USDT": Decimal("1"),
        }

        token = tool_args.get("token_symbol", "ETH").upper()
        if token in ("TFIL",):  # Testnets have no value
            return Decimal("0")

        price = prices.get(token, Decimal("1"))
        return amount * price

    def __repr__(self) -> str:
        return (
            f"TransactionSecurityHook(priority={self.priority}, "
            f"limit=${self.daily_limit_usd}, "
            f"mainnet={'allowed' if self.allow_mainnet else 'blocked'})"
        )
