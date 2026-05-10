"""
Economic Gates for Reflection Feature.

Controls access to premium reflection features based on the agent's
economic status (paid tier, revenue share, etc.).
"""

import logging
from typing import Protocol

logger = logging.getLogger(__name__)


class EconomicWallet(Protocol):
    """Wallet feature methods used by reflection's economic gates."""

    def is_paid_tier(self) -> bool: ...

    def has_revenue_share(self) -> bool: ...


class ConfigurationError(Exception):
    """Raised when required configuration is missing."""
    pass


class EconomicGate:
    """Gates features based on economic status.

    This class controls access to premium reflection features:
    - GitHub ticket creation (requires paid tier OR revenue share)
    - Self-model updates in Filecoin (requires paid tier)

    All gate checks fail fast with clear error messages if the wallet
    feature is not properly configured.
    """

    def __init__(self, wallet_feature: EconomicWallet):
        """Initialize the economic gate.

        Args:
            wallet_feature: WalletFeature instance for economic checks

        Raises:
            ConfigurationError: If wallet_feature is None
        """
        # FAIL FAST - wallet required
        if wallet_feature is None:
            raise ConfigurationError(
                "WalletFeature required for economic gates"
            )
        self.wallet = wallet_feature

    def can_create_tickets(self) -> bool:
        """Check if ticket creation is economically allowed.

        Ticket creation requires either:
        - Paid tier subscription
        - Active revenue share agreement

        Returns:
            True if ticket creation is allowed
        """
        try:
            is_paid = self.wallet.is_paid_tier()
            has_share = self.wallet.has_revenue_share()
            return is_paid or has_share
        except Exception as e:
            logger.error(f"Economic gate check failed: {e}")
            return False

    def can_update_self_model(self) -> bool:
        """Check if self-model updates are allowed.

        Self-model updates to Filecoin require paid tier due to
        storage costs and the permanence of the data.

        Returns:
            True if self-model updates are allowed
        """
        try:
            return self.wallet.is_paid_tier()
        except Exception as e:
            logger.error(f"Economic gate check failed: {e}")
            return False

    def check_or_raise(self, operation: str) -> None:
        """Check if an operation is allowed, raise if not.

        Args:
            operation: One of 'ticket_creation' or 'self_model_update'

        Raises:
            PermissionError: If operation is not allowed
            ValueError: If operation is unknown
        """
        checks = {
            "ticket_creation": (self.can_create_tickets, "Paid tier or revenue share required for ticket creation"),
            "self_model_update": (self.can_update_self_model, "Paid tier required for self-model updates"),
        }

        if operation not in checks:
            raise ValueError(f"Unknown operation: {operation}. Valid: {list(checks.keys())}")

        check_fn, error_msg = checks[operation]
        if not check_fn():
            raise PermissionError(error_msg)
