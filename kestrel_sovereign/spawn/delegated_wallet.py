"""
DelegatedWallet: Budget delegation for spawned child agents.

Wraps a WalletAgent with ceiling enforcement so parent agents can allocate
spending limits to children using a hold/release pattern.

Hold/Release Lifecycle:
1. Parent spawns child with budget=N
2. Parent wallet debited by N (hold)
3. Child's DelegatedWallet ceiling set to N
4. On termination: unspent = N - spent credited back to parent
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from typing import Optional

from kestrel_feature_wallet.feature import Currency, WalletAgent

logger = logging.getLogger(__name__)


class BudgetExceededError(Exception):
    """Raised when a transaction would exceed the delegated budget ceiling."""


@dataclass
class BudgetAllocation:
    """Tracks a budget delegation from parent to child agent."""

    child_did: str
    amount: Decimal
    spent: Decimal = Decimal("0")
    parent_did: str = ""
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    @property
    def remaining(self) -> Decimal:
        """Budget remaining before ceiling is hit."""
        return self.amount - self.spent

    @property
    def is_exhausted(self) -> bool:
        """True if no budget remains."""
        return self.spent >= self.amount


class DelegatedWallet:
    """Wallet wrapper that enforces a spending ceiling from a parent allocation.

    Every transaction is checked against the ceiling: ``spent + cost <= amount``.
    Overspend attempts raise ``BudgetExceededError``.
    """

    def __init__(
        self,
        wallet: WalletAgent,
        allocation: BudgetAllocation,
    ):
        self._wallet = wallet
        self.allocation = allocation
        self.transactions: list[dict] = []

    @property
    def ceiling(self) -> Decimal:
        """The maximum amount this wallet is allowed to spend."""
        return self.allocation.amount

    @property
    def spent(self) -> Decimal:
        """Total amount spent so far."""
        return self.allocation.spent

    @property
    def remaining(self) -> Decimal:
        """Budget remaining."""
        return self.allocation.remaining

    def can_spend(self, cost: Decimal) -> bool:
        """Check whether *cost* fits within the remaining budget."""
        return self.allocation.spent + cost <= self.allocation.amount

    async def spend(
        self,
        cost: Decimal,
        memo: str,
        currency: Currency = Currency.FIL,
    ) -> bool:
        """Spend from the delegated budget.

        Args:
            cost: Amount to spend.
            memo: Human-readable description of the spend.
            currency: Currency to use.

        Returns:
            True on success.

        Raises:
            BudgetExceededError: If the spend would exceed the ceiling.
        """
        if cost <= 0:
            raise ValueError("Spend amount must be positive")

        if not self.can_spend(cost):
            raise BudgetExceededError(
                f"Transaction of {cost} {currency.value} would exceed budget ceiling. "
                f"Ceiling: {self.allocation.amount}, "
                f"Spent: {self.allocation.spent}, "
                f"Remaining: {self.allocation.remaining}, "
                f"Requested: {cost}"
            )

        success = await self._wallet.transfer(cost, memo, currency)
        if not success:
            return False

        self.allocation.spent += cost
        self.transactions.append({
            "type": "delegated_spend",
            "currency": currency.value,
            "amount": str(cost),
            "memo": memo,
            "spent_total": str(self.allocation.spent),
            "remaining": str(self.allocation.remaining),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

        logger.info(
            "Delegated spend: %s %s for '%s'. Spent: %s/%s",
            cost, currency.value, memo,
            self.allocation.spent, self.allocation.amount,
        )
        return True

    def get_status(self) -> dict:
        """Return a summary of the delegated budget status."""
        return {
            "child_did": self.allocation.child_did,
            "parent_did": self.allocation.parent_did,
            "ceiling": str(self.allocation.amount),
            "spent": str(self.allocation.spent),
            "remaining": str(self.allocation.remaining),
            "is_exhausted": self.allocation.is_exhausted,
            "transaction_count": len(self.transactions),
        }


async def create_delegated_wallet(
    parent_wallet: WalletAgent,
    parent_did: str,
    child_did: str,
    budget: Decimal,
    currency: Currency = Currency.FIL,
) -> DelegatedWallet:
    """Create a delegated wallet by holding budget from the parent.

    This implements the "hold" phase of the hold/release lifecycle:
    1. Validates the parent can afford the budget.
    2. Debits the parent wallet.
    3. Creates a child WalletAgent funded with the budget.
    4. Returns a DelegatedWallet wrapping the child wallet.

    Args:
        parent_wallet: The parent's WalletAgent (will be debited).
        parent_did: DID of the parent agent.
        child_did: DID of the child agent.
        budget: Amount to delegate.
        currency: Currency for the delegation.

    Returns:
        A DelegatedWallet for the child agent.

    Raises:
        ValueError: If budget is non-positive or parent has insufficient funds.
    """
    if budget <= 0:
        raise ValueError("Budget must be positive")

    if not parent_wallet.can_afford(budget, currency):
        parent_balance = parent_wallet.get_balance(currency, "main")
        raise ValueError(
            f"Parent has insufficient funds. "
            f"Need {budget} {currency.value}, have {parent_balance} {currency.value}"
        )

    # Hold: debit the parent
    success = await parent_wallet.transfer(
        budget,
        f"budget hold for child {child_did}",
        currency,
    )
    if not success:
        raise ValueError("Failed to debit parent wallet for budget hold")

    # Create child wallet with the delegated budget (all goes to main, no 90/10 split)
    child_wallet = WalletAgent(
        agent_id=child_did,
        initial_balance=Decimal("0"),
        initial_currency=currency,
    )
    await child_wallet.initialize()
    # Deposit full budget to main (bypass the 90/10 split used for normal deposits)
    child_wallet._balances[currency]["main"] = budget

    allocation = BudgetAllocation(
        child_did=child_did,
        amount=budget,
        parent_did=parent_did,
    )

    logger.info(
        "Created delegated wallet: child=%s, budget=%s %s, parent=%s",
        child_did, budget, currency.value, parent_did,
    )

    return DelegatedWallet(wallet=child_wallet, allocation=allocation)


async def release_delegated_wallet(
    delegated_wallet: DelegatedWallet,
    parent_wallet: WalletAgent,
    currency: Currency = Currency.FIL,
) -> Decimal:
    """Release a delegated wallet, returning unspent funds to the parent.

    This implements the "release" phase of the hold/release lifecycle:
    1. Calculates unspent = ceiling - spent.
    2. Credits the unspent amount back to the parent wallet.

    Args:
        delegated_wallet: The child's delegated wallet.
        parent_wallet: The parent's WalletAgent (will be credited).
        currency: Currency for the release.

    Returns:
        The amount returned to the parent.
    """
    unspent = delegated_wallet.remaining

    if unspent > 0:
        await parent_wallet.deposit(
            unspent,
            currency,
            to_audit=False,
            memo=f"budget release from child {delegated_wallet.allocation.child_did}",
        )

    logger.info(
        "Released delegated wallet: child=%s, returned=%s %s, spent=%s %s",
        delegated_wallet.allocation.child_did,
        unspent, currency.value,
        delegated_wallet.spent, currency.value,
    )

    return unspent
