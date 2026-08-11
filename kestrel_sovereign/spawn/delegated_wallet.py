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
import hashlib
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
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


class DurableDebitProviderProtocol(Protocol):
    """Provider-owned, durable idempotent debit contract.

    ``prepare_debit_intent`` MUST durably record ``idempotency_key`` before
    any provider debit can occur. ``execute_debit_intent`` may be retried and
    ``resolve_debit_intent`` returns ``True`` (applied), ``False`` (known not
    applied), or ``None`` (still ambiguous).  The allocation keeps only a
    reference to this provider-owned record; an in-memory flag is never used
    as proof that an external debit did or did not happen.
    """

    async def prepare_debit_intent(
        self,
        *,
        idempotency_key: str,
        amount: Decimal,
        memo: str,
        currency: Any,
    ) -> str: ...

    async def execute_debit_intent(self, intent_id: str) -> bool: ...

    async def resolve_debit_intent(self, intent_id: str) -> bool | None: ...


class DurableDelegatedAllocationProviderProtocol(Protocol):
    """Durable parent-side hold/release contract for a delegated budget.

    Child spend intents protect individual work charges.  This separate
    provider contract protects the allocation itself, whose hold and later
    refund can otherwise be ambiguous across a process crash.  Both calls are
    idempotent by ``allocation_id`` and commit their balance mutation with the
    provider's allocation ledger.
    """

    async def reserve_delegated_allocation(
        self,
        *,
        allocation_id: str,
        child_did: str,
        amount: Decimal,
        memo: str,
        currency: Any,
    ) -> bool: ...

    async def release_delegated_allocation(
        self,
        *,
        allocation_id: str,
        amount: Decimal,
        currency: Any,
    ) -> bool: ...


class DurableDelegatedChildWalletProviderProtocol(Protocol):
    """Provider-owned atomic hold plus durable child-wallet provisioning.

    Core cannot infer a feature's storage layout or safely seed private wallet
    state.  Providers that support durable allocations therefore expose this
    optional seam: it commits the parent hold, child initial allocation, and
    replay record together, then returns an initialized child wallet bound to
    durable storage.  ``None`` means the atomic hold was refused for insufficient
    funds; parameter drift and storage failures raise.
    """

    async def reserve_and_provision_delegated_child_wallet(
        self,
        *,
        allocation_id: str,
        child_did: str,
        amount: Decimal,
        memo: str,
        currency: Any,
    ) -> WalletProtocol | None: ...


class DurableDelegatedChildWalletReleaseProviderProtocol(Protocol):
    """Provider-owned terminal release for a durably provisioned child.

    The provider reads its authoritative child balance, fences that child
    against all future debits, credits the parent, and records the terminal
    allocation state in one atomic transaction.  The returned amount is the
    durable release result and is stable across retries; Core must not derive
    it from ``BudgetAllocation.spent``.
    """

    async def release_and_fence_delegated_child_wallet(
        self, *, allocation_id: str, currency: Any
    ) -> Decimal: ...


def has_durable_delegated_child_wallet_provisioning_contract(wallet: object) -> bool:
    """Whether *wallet* atomically reserves and provisions a durable child.

    This capability is the authority for a production spawn's affordability
    decision.  It is intentionally shared with ``AgentManager`` so a stale
    synchronous cache check cannot be reintroduced before the provider's
    idempotent reserve.
    """

    return callable(
        getattr(wallet, "reserve_and_provision_delegated_child_wallet", None)
    )


class BudgetExceededError(Exception):
    """Raised when a transaction would exceed the delegated budget ceiling."""


class DelegatedSpendOutcomeUnknown(RuntimeError):
    """A provider debit may have applied and must be reconciled before refund."""


@dataclass(frozen=True)
class DelegatedSpendIntent:
    """Serializable reference to a provider-owned durable debit intent."""

    intent_id: str
    idempotency_key: str
    amount: Decimal
    currency: str
    created_at: str


@dataclass
class BudgetAllocation:
    """Tracks a budget delegation from parent to child agent."""

    child_did: str
    amount: Decimal
    spent: Decimal = Decimal("0")
    parent_did: str = ""
    allocation_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    next_spend_sequence: int = 0
    # This is a reconciliation reference, not the source of durability.  The
    # provider contract above owns the durable debit intent and outcome.
    pending_spend_intent: DelegatedSpendIntent | None = None
    # ``True`` means the parent hold is provider-owned and its refund must use
    # the same durable allocation record rather than an ordinary deposit.
    parent_hold_durable: bool = False
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

    def _durable_debit_provider(self) -> DurableDebitProviderProtocol:
        provider = self._wallet
        required = (
            "prepare_debit_intent",
            "execute_debit_intent",
            "resolve_debit_intent",
        )
        if not all(callable(getattr(provider, name, None)) for name in required):
            raise RuntimeError(
                "delegated child spending requires a provider-owned durable "
                "idempotent debit intent contract"
            )
        return provider  # type: ignore[return-value]

    @staticmethod
    def _durable_allocation_provider(
        wallet: WalletProtocol,
    ) -> DurableDelegatedAllocationProviderProtocol | None:
        required = (
            "reserve_delegated_allocation",
            "release_delegated_allocation",
        )
        if all(callable(getattr(wallet, name, None)) for name in required):
            return wallet  # type: ignore[return-value]
        return None

    @staticmethod
    def _durable_child_wallet_provider(
        wallet: WalletProtocol,
    ) -> DurableDelegatedChildWalletProviderProtocol | None:
        if has_durable_delegated_child_wallet_provisioning_contract(wallet):
            return wallet  # type: ignore[return-value]
        return None

    @staticmethod
    def _durable_child_wallet_release_provider(
        wallet: WalletProtocol,
    ) -> DurableDelegatedChildWalletReleaseProviderProtocol | None:
        release = getattr(wallet, "release_and_fence_delegated_child_wallet", None)
        if callable(release):
            return wallet  # type: ignore[return-value]
        return None

    async def _reconcile_pending_spend(self) -> None:
        """Resolve any provider debit before another debit or a refund."""

        intent = self.allocation.pending_spend_intent
        if intent is None:
            return
        provider = self._durable_debit_provider()
        outcome = await provider.resolve_debit_intent(intent.intent_id)
        if outcome is None:
            raise DelegatedSpendOutcomeUnknown(
                "delegated child debit outcome is ambiguous; refusing refund or new spend"
            )
        if outcome is True:
            self.allocation.spent += intent.amount
            self._record_spend(intent.amount, intent.currency, "reconciled durable debit")
        self.allocation.pending_spend_intent = None

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
            await self._reconcile_pending_spend()
            if self._refund_completed:
                return Decimal("0")
            provider = self._durable_allocation_provider(parent_wallet)
            durable_hold = self.allocation.parent_hold_durable
            if durable_hold and provider is None:
                raise RuntimeError(
                    "delegated allocation requires its durable parent provider for release"
                )
            if self._refund_attempted and not durable_hold:
                raise RuntimeError(
                    "delegated wallet refund outcome is uncertain; refusing duplicate credit"
                )
            self._refund_attempted = True
            unspent = self.allocation.remaining
            child_release_provider = self._durable_child_wallet_release_provider(
                parent_wallet
            )
            if child_release_provider is not None:
                released_amount = await child_release_provider.release_and_fence_delegated_child_wallet(
                    allocation_id=self.allocation.allocation_id,
                    currency=currency,
                )
                try:
                    unspent = Decimal(released_amount)
                except (TypeError, ValueError, InvalidOperation) as exc:
                    raise RuntimeError(
                        "durable child release provider returned an invalid amount"
                    ) from exc
                if not unspent.is_finite() or unspent < 0:
                    raise RuntimeError(
                        "durable child release provider returned an invalid amount"
                    )
                # The allocation record is a local presentation cache only.
                # Synchronize it from the provider's authoritative child
                # balance after release so status/logging cannot claim the
                # old process's volatile spend total.
                self.allocation.spent = max(
                    Decimal("0"), self.allocation.amount - unspent
                )
            elif self._durable_child_wallet_provider(parent_wallet) is not None:
                raise RuntimeError(
                    "durably provisioned delegated child requires a provider-owned "
                    "child-fencing release contract"
                )
            elif unspent <= 0:
                self._refund_completed = True
                return unspent
            elif durable_hold:
                released = await provider.release_delegated_allocation(  # type: ignore[union-attr]
                    allocation_id=self.allocation.allocation_id,
                    amount=unspent,
                    currency=currency,
                )
                if released is not True:
                    raise RuntimeError("parent wallet refused durable delegated budget refund")
            else:
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
            await self._reconcile_pending_spend()
            if not self.can_spend(cost):
                currency_value = _currency_value(currency)
                raise BudgetExceededError(
                    f"Transaction of {cost} {currency_value} would exceed budget ceiling. "
                    f"Ceiling: {self.allocation.amount}, "
                    f"Spent: {self.allocation.spent}, "
                    f"Remaining: {self.allocation.remaining}, "
                    f"Requested: {cost}"
                )

            provider = self._durable_debit_provider()
            self.allocation.next_spend_sequence += 1
            idempotency_key = (
                f"delegated-spend:{self.allocation.allocation_id}:"
                f"{self.allocation.next_spend_sequence}"
            )
            intent_id = await provider.prepare_debit_intent(
                idempotency_key=idempotency_key,
                amount=cost,
                memo=memo,
                currency=currency,
            )
            if not isinstance(intent_id, str) or not intent_id:
                raise RuntimeError("provider returned no durable delegated debit intent ID")
            intent = DelegatedSpendIntent(
                intent_id=intent_id,
                idempotency_key=idempotency_key,
                amount=cost,
                currency=_currency_value(currency),
                created_at=datetime.now(timezone.utc).isoformat(),
            )
            # The durable provider intent is prepared before any debit I/O.
            # Keep its reference until the provider proves the debit outcome.
            self.allocation.pending_spend_intent = intent
            success = await provider.execute_debit_intent(intent_id)
            if not success:
                self.allocation.pending_spend_intent = None
                return False

            self.allocation.spent += cost
            self.allocation.pending_spend_intent = None
            self._record_spend(cost, _currency_value(currency), memo)

            logger.info(
                "Delegated spend: %s %s for '%s'. Spent: %s/%s",
                cost, _currency_value(currency), memo,
                self.allocation.spent, self.allocation.amount,
            )
            return True

    def _record_spend(self, cost: Decimal, currency_value: str, memo: str) -> None:
        """Record only a confirmed/reconciled debit; never an ambiguous one."""

        self.transactions.append({
                "type": "delegated_spend",
                "currency": currency_value,
                "amount": str(cost),
                "memo": memo,
                "spent_total": str(self.allocation.spent),
                "remaining": str(self.allocation.remaining),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })

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
            "pending_spend_intent": (
                self.allocation.pending_spend_intent.intent_id
                if self.allocation.pending_spend_intent is not None
                else None
            ),
        }


async def create_delegated_wallet(
    parent_wallet: WalletProtocol,
    parent_did: str,
    child_did: str,
    budget: Decimal,
    currency: Any = None,
) -> DelegatedWallet:
    """Create a delegated wallet by holding budget from the parent.

    This implements the "hold" phase of the hold/release lifecycle.  A modern
    durable provider atomically holds and provisions the child through its own
    storage contract. Legacy providers retain the historical affordability /
    transfer / in-memory-child path.

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

    # The child DID is minted before this call and is the durable identity of a
    # single spawn.  Derive the hold key from it rather than creating a fresh
    # random key here: after a crash between parent debit and child setup, the
    # manager can repeat the same child creation without taking a second hold.
    allocation_key = hashlib.sha256(
        f"delegated-allocation:v1\x00{parent_did}\x00{child_did}".encode("utf-8")
    ).hexdigest()
    allocation = BudgetAllocation(
        child_did=child_did,
        amount=budget,
        parent_did=parent_did,
        allocation_id=allocation_key,
    )
    # A direct durable parent can atomically persist the hold and the
    # allocation identity before any child construction.  Keep nested
    # DelegatedWallet parents on their ceiling-enforced transfer path: bypassing
    # that wrapper would debit the root wallet without consuming the parent's
    # delegated headroom.
    allocation_provider = (
        None
        if isinstance(parent_wallet, DelegatedWallet)
        else DelegatedWallet._durable_allocation_provider(parent_wallet)
    )
    child_wallet_provider = (
        None
        if isinstance(parent_wallet, DelegatedWallet)
        else DelegatedWallet._durable_child_wallet_provider(parent_wallet)
    )
    child_wallet: WalletProtocol | None = None
    success: bool | None = None
    if child_wallet_provider is not None:
        # This is deliberately before any synchronous can_afford preflight.
        # A retry can arrive after the original durable hold made the cache look
        # insufficient; the provider's allocation key is the authority and
        # returns the existing child without taking another hold.
        child_wallet = await child_wallet_provider.reserve_and_provision_delegated_child_wallet(
            allocation_id=allocation.allocation_id,
            child_did=child_did,
            amount=budget,
            memo=f"budget hold for child {child_did}",
            currency=currency,
        )
        if child_wallet is None:
            raise ValueError("Failed to debit parent wallet for budget hold")
        allocation.parent_hold_durable = True
    elif allocation_provider is not None:
        # The durable idempotency path must precede stale in-memory affordability
        # checks. Older durable providers may not yet provision children, but
        # their reserve operation still atomically decides whether a hold exists
        # or can be afforded.
        success = await allocation_provider.reserve_delegated_allocation(
            allocation_id=allocation.allocation_id,
            child_did=child_did,
            amount=budget,
            memo=f"budget hold for child {child_did}",
            currency=currency,
        )
        allocation.parent_hold_durable = True
    else:
        # Preserve the legacy provider path, which has no durable allocation
        # record to reconcile and therefore needs the historical local preflight.
        if not parent_wallet.can_afford(budget, currency):
            parent_balance = parent_wallet.get_balance(currency, "main")
            currency_value = _currency_value(currency)
            raise ValueError(
                f"Parent has insufficient funds. "
                f"Need {budget} {currency_value}, have {parent_balance} {currency_value}"
            )
        success = await parent_wallet.transfer(
            budget,
            f"budget hold for child {child_did}",
            currency,
        )
    if child_wallet is None and success is not True:
        raise ValueError("Failed to debit parent wallet for budget hold")

    # Transactional after the debit: if child-wallet construction/init fails, the
    # parent has already been debited, so refund the hold before propagating —
    # otherwise the funds are stranded (#2113, codex).
    try:
        if child_wallet is not None:
            return DelegatedWallet(wallet=child_wallet, allocation=allocation)
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
            if allocation.parent_hold_durable:
                if allocation_provider is None or not await allocation_provider.release_delegated_allocation(
                    allocation_id=allocation.allocation_id,
                    amount=budget,
                    currency=currency,
                ):
                    raise RuntimeError("durable budget hold release was not acknowledged")
            else:
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
