"""
Economic Gate Methods for WalletFeature.

Provides is_paid_tier() and has_revenue_share() checks that gate
premium features (e.g., reflection, GitHub ticket creation).

Used as a mixin by WalletFeature.
"""

import logging
from decimal import Decimal

logger = logging.getLogger(__name__)


class EconomicGateMixin:
    """
    Mixin providing economic gate checks for WalletFeature.

    Requires self.wallet (WalletAgent) and self.agent to be available.
    """

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
