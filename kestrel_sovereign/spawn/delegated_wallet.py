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

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Protocol

logger = logging.getLogger(__name__)


class WalletProtocol(Protocol):
    """Minimal wallet surface required for delegated budget accounting."""

    _balances: dict

    async def initialize(self) -> None: ...

    def can_afford(self, amount: Decimal, currency: Any) -> bool: ...

    def get_balance(self, currency: Any, balance_type: str = "main") -> Decimal: ...

    async def transfer(
        self, amount: Decimal, memo: str = "", currency: Any = None
    ) -> bool: ...

    async def deposit(
        self,
        amount: Decimal,
        currency: Any = None,
        to_audit: bool = False,
        memo: str = "",
    ) -> bool: ...


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
        wallet: WalletProtocol,
        allocation: BudgetAllocation,
    ):
        self._wallet = wallet
        self.allocation = allocation
        self.transactions: list[dict] = []
        # Removal can quarantine cancellation-resistant cognition and refund
        # this child's allocation before that cognition finishes.  The reaper
        # must not be able to spend funds after they return to the parent.
        self._revoked = False
        self._spend_lock = asyncio.Lock()
        # A refund may await arbitrary parent-wallet I/O. Record its attempt
        # under the same lock so a cancelled/ambiguous deposit is never blindly
        # replayed and credited twice.
        self._refund_attempted = False
        self._refund_completed = False

    def fence_spending(self) -> None:
        """Immediately refuse new spends without waiting for wallet I/O.

        A spend already holding ``_spend_lock`` may still be inside the wrapped
        wallet transfer.  The eventual refund waits for that exact transfer,
        so it snapshots the final amount and cannot refund before a late debit.
        This synchronous fence lets a bounded control-plane removal hand the
        remaining wait to a retained cleanup owner instead of blocking forever.
        """

        self._revoked = True

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
        return (
            not self._revoked
            and self.allocation.spent + cost <= self.allocation.amount
        )

    async def revoke_and_get_unspent(self) -> Decimal:
        """Fence future spending and atomically snapshot refundable funds.

        ``release_delegated_wallet`` shares this lock with ``spend``. A
        quarantined turn therefore either completes its debit before the
        refund is calculated or is refused after revocation; it cannot debit
        after the parent has recovered the unspent allocation.
        """
        self.fence_spending()
        async with self._spend_lock:
            return self.allocation.remaining

    async def refund_to_parent(
        self, parent_wallet: WalletProtocol, *, currency: Any
    ) -> Decimal:
        """Fence, serialize, and perform this allocation's one refund.

        The parent deposit intentionally occurs while holding ``_spend_lock``.
        That makes the final child debit/refund ordering atomic from Core's
        perspective: a spend completes before the snapshot, or it observes the
        fence and never debits after the refund.  If provider I/O reports an
        exception after an ambiguous side effect, later calls refuse to replay
        the credit; the caller retains an unsafe cleanup record for operators.
        """

        self.fence_spending()
        async with self._spend_lock:
            if self._refund_completed:
                return Decimal("0")
            if self._refund_attempted:
                raise RuntimeError(
                    "delegated wallet refund outcome is uncertain; refusing duplicate credit"
                )
            self._refund_attempted = True
            unspent = self.allocation.remaining
            if unspent <= 0:
                self._refund_completed = True
                return unspent
            deposited = await parent_wallet.deposit(
                unspent,
                currency,
                to_audit=False,
                memo=f"budget release from child {self.allocation.child_did}",
            )
            if deposited is not True:
                raise RuntimeError("parent wallet refused delegated budget refund")
            # If the parent is itself budgeted (a grandchild being released
            # back into its budgeted parent), restore headroom only after the
            # single confirmed credit.
            if isinstance(parent_wallet, DelegatedWallet):
                parent_wallet.restore_headroom(unspent)
            self._refund_completed = True
            return unspent

    # ------------------------------------------------------------------
    # WalletProtocol drop-in surface (#2113).
    #
    # A child's spend paths (backups/anchoring in ``agent/backup.py`` and
    # ``features/sovereignty``, metered LLM cost, etc.) call the wallet's
    # ``can_afford`` / ``transfer`` — NOT ``spend``. So for the budget to be
    # enforced, ``child.wallet`` is set to this DelegatedWallet and it must be a
    # drop-in: override the spend-affecting methods with the ceiling check and
    # delegate everything else (``deposit``/``_balances``/``initialize``/…) to the
    # wrapped funded wallet via ``__getattr__``.
    # ------------------------------------------------------------------

    def can_afford(self, amount: Decimal, currency: Any = None) -> bool:
        """True only if *amount* is within BOTH the remaining budget ceiling and
        the wrapped wallet's funds."""
        if not self.can_spend(amount):
            return False
        currency = currency or _default_currency_for(self._wallet)
        return self._wallet.can_afford(amount, currency)

    async def transfer(
        self, amount: Decimal, memo: str = "", currency: Any = None
    ) -> bool:
        """Ceiling-enforced transfer — the drop-in for ``WalletAgent.transfer``.

        Routes through :meth:`spend`, so an overspend raises
        ``BudgetExceededError`` and a successful debit advances ``allocation.spent``.
        """
        return await self.spend(amount, memo, currency)

    def get_balance(self, currency: Any = None, balance_type: str = "main") -> Decimal:
        """Spendable balance = the smaller of the wrapped wallet's balance and the
        remaining budget, so a child never appears to hold more than its ceiling."""
        currency = currency or _default_currency_for(self._wallet)
        wrapped = self._wallet.get_balance(currency, balance_type)
        if balance_type == "main":
            return min(wrapped, self.remaining)
        return wrapped

    def restore_headroom(self, amount: Decimal) -> None:
        """Reduce ``allocation.spent`` by *amount*, restoring budget headroom.

        Called ONLY for an explicit budget refund — when a budgeted grandchild's
        unspent hold is released back into this (budgeted) child
        (``release_delegated_wallet``). This is deliberately NOT wired into
        ``deposit``: a normal external top-up must not restore budget headroom, or
        the child could spend past its ceiling after any deposit. Bounded at 0 so
        a refund can never lift ``spent`` below zero / inflate the ceiling beyond
        ``allocation.amount``. ``deposit`` itself is delegated to the wrapped
        wallet unchanged (via ``__getattr__``)."""
        self.allocation.spent = max(
            Decimal("0"), self.allocation.spent - Decimal(amount)
        )

    def __getattr__(self, name: str) -> Any:
        """Delegate any attribute this wrapper doesn't define to the wrapped
        wallet, so DelegatedWallet is a transparent drop-in (deposit, _balances,
        initialize, provider-specific helpers, …). Only ``transfer`` /
        ``can_afford`` / ``get_balance`` are overridden above to enforce the
        ceiling. ``_wallet`` itself is a real instance attribute set in __init__,
        so this never recurses on it."""
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        wallet = self.__dict__.get("_wallet")
        if wallet is None:
            raise AttributeError(name)
        return getattr(wallet, name)

    async def spend(
        self,
        cost: Decimal,
        memo: str,
        currency: Any = None,
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
        currency = currency or _default_currency_for(self._wallet)

        if cost <= 0:
            raise ValueError("Spend amount must be positive")

        async with self._spend_lock:
            if not self.can_spend(cost):
                currency_value = _currency_value(currency)
                raise BudgetExceededError(
                    f"Transaction of {cost} {currency_value} would exceed budget ceiling. "
                    f"Ceiling: {self.allocation.amount}, "
                    f"Spent: {self.allocation.spent}, "
                    f"Remaining: {self.allocation.remaining}, "
                    f"Requested: {cost}"
                )

            success = await self._wallet.transfer(cost, memo, currency)
            if not success:
                return False

            self.allocation.spent += cost
            currency_value = _currency_value(currency)
            self.transactions.append({
                "type": "delegated_spend",
                "currency": currency_value,
                "amount": str(cost),
                "memo": memo,
                "spent_total": str(self.allocation.spent),
                "remaining": str(self.allocation.remaining),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })

            logger.info(
                "Delegated spend: %s %s for '%s'. Spent: %s/%s",
                cost, currency_value, memo,
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
    parent_wallet: WalletProtocol,
    parent_did: str,
    child_did: str,
    budget: Decimal,
    currency: Any = None,
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

    currency = currency or _default_currency_for(parent_wallet)

    if not parent_wallet.can_afford(budget, currency):
        parent_balance = parent_wallet.get_balance(currency, "main")
        currency_value = _currency_value(currency)
        raise ValueError(
            f"Parent has insufficient funds. "
            f"Need {budget} {currency_value}, have {parent_balance} {currency_value}"
        )

    # Hold: debit the parent
    success = await parent_wallet.transfer(
        budget,
        f"budget hold for child {child_did}",
        currency,
    )
    if not success:
        raise ValueError("Failed to debit parent wallet for budget hold")

    # Transactional after the debit: if child-wallet construction/init fails, the
    # parent has already been debited, so refund the hold before propagating —
    # otherwise the funds are stranded (#2113, codex).
    try:
        # Construct from the CONCRETE wallet class: when the parent's wallet is
        # itself a DelegatedWallet (a budgeted child spawning a budgeted
        # grandchild), unwrap to the underlying funded wallet — DelegatedWallet's
        # constructor is (wallet, allocation), not the funded-wallet constructor,
        # so nested budgeted spawns would otherwise raise. The hold above still
        # went through the parent's (delegated) ceiling.
        base_wallet = (
            parent_wallet._wallet
            if isinstance(parent_wallet, DelegatedWallet)
            else parent_wallet
        )
        wallet_cls = type(base_wallet)
        child_wallet = wallet_cls(
            agent_id=child_did,
            initial_balance=Decimal("0"),
            initial_currency=currency,
        )
        await child_wallet.initialize()
        # Deposit full budget to main (bypass the 90/10 split used for deposits)
        child_wallet._balances[currency]["main"] = budget
    except Exception:
        try:
            await parent_wallet.deposit(
                budget,
                currency,
                to_audit=False,
                memo=f"budget hold refund (child wallet setup failed) for {child_did}",
            )
            # If the parent is itself budgeted, the failed hold incremented its
            # allocation.spent — restore that headroom too, mirroring the normal
            # release path (otherwise a failed nested spawn permanently shrinks
            # the parent's budget until termination).
            if isinstance(parent_wallet, DelegatedWallet):
                parent_wallet.restore_headroom(budget)
        except Exception:  # noqa: BLE001
            logger.error(
                "CRITICAL: budget hold for child %s could not be refunded after a "
                "setup failure — parent funds may be stranded.", child_did,
            )
        raise

    allocation = BudgetAllocation(
        child_did=child_did,
        amount=budget,
        parent_did=parent_did,
    )

    logger.info(
        "Created delegated wallet: child=%s, budget=%s %s, parent=%s",
        child_did, budget, _currency_value(currency), parent_did,
    )

    return DelegatedWallet(wallet=child_wallet, allocation=allocation)


async def release_delegated_wallet(
    delegated_wallet: DelegatedWallet,
    parent_wallet: WalletProtocol,
    currency: Any = None,
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
    currency = currency or _default_currency_for(parent_wallet)
    unspent = await delegated_wallet.refund_to_parent(
        parent_wallet, currency=currency
    )

    logger.info(
        "Released delegated wallet: child=%s, returned=%s %s, spent=%s %s",
        delegated_wallet.allocation.child_did,
        unspent, _currency_value(currency),
        delegated_wallet.spent, _currency_value(currency),
    )

    return unspent


def _default_currency_for(wallet: WalletProtocol) -> Any:
    """Return the wallet's FIL-like currency key without importing wallet code."""
    balances = getattr(wallet, "_balances", {})
    for currency in balances:
        if getattr(currency, "value", currency) == "FIL":
            return currency
    if balances:
        return next(iter(balances))
    return "FIL"


def _currency_value(currency: Any) -> str:
    return str(getattr(currency, "value", currency))
