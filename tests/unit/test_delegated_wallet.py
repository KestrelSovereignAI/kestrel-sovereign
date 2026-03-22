"""Tests for DelegatedWallet budget delegation and ceiling enforcement."""

import pytest
from decimal import Decimal

from kestrel_sovereign.features.wallet.feature import Currency, WalletAgent
from kestrel_sovereign.spawn.delegated_wallet import (
    BudgetAllocation,
    BudgetExceededError,
    DelegatedWallet,
    create_delegated_wallet,
    release_delegated_wallet,
)


# ---------------------------------------------------------------------------
# BudgetAllocation dataclass
# ---------------------------------------------------------------------------


class TestBudgetAllocation:
    def test_remaining(self):
        alloc = BudgetAllocation(child_did="did:child:1", amount=Decimal("50"))
        assert alloc.remaining == Decimal("50")
        alloc.spent = Decimal("20")
        assert alloc.remaining == Decimal("30")

    def test_is_exhausted(self):
        alloc = BudgetAllocation(child_did="did:child:1", amount=Decimal("10"))
        assert not alloc.is_exhausted
        alloc.spent = Decimal("10")
        assert alloc.is_exhausted

    def test_created_at_populated(self):
        alloc = BudgetAllocation(child_did="did:child:1", amount=Decimal("1"))
        assert alloc.created_at  # non-empty ISO string


# ---------------------------------------------------------------------------
# Ceiling enforcement
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ceiling_enforcement_blocks_overspend():
    """Child cannot spend more than the allocated ceiling."""
    wallet = WalletAgent(
        agent_id="child-1",
        initial_balance=Decimal("100"),
    )
    await wallet.initialize()

    alloc = BudgetAllocation(
        child_did="did:child:1",
        amount=Decimal("30"),
        parent_did="did:parent:1",
    )
    dw = DelegatedWallet(wallet=wallet, allocation=alloc)

    # Spend within ceiling
    ok = await dw.spend(Decimal("20"), "first task")
    assert ok is True
    assert dw.spent == Decimal("20")
    assert dw.remaining == Decimal("10")

    # Attempt to overspend
    with pytest.raises(BudgetExceededError):
        await dw.spend(Decimal("15"), "too expensive")

    # Spend exact remainder
    ok = await dw.spend(Decimal("10"), "second task")
    assert ok is True
    assert dw.remaining == Decimal("0")
    assert dw.allocation.is_exhausted


@pytest.mark.asyncio
async def test_ceiling_enforcement_zero_budget():
    """Zero-budget wallet blocks all spending."""
    wallet = WalletAgent(agent_id="child-0", initial_balance=Decimal("100"))
    await wallet.initialize()

    alloc = BudgetAllocation(child_did="did:child:0", amount=Decimal("0"))
    dw = DelegatedWallet(wallet=wallet, allocation=alloc)

    with pytest.raises(BudgetExceededError):
        await dw.spend(Decimal("1"), "any task")


@pytest.mark.asyncio
async def test_spend_rejects_non_positive():
    """Negative and zero spend amounts are rejected."""
    wallet = WalletAgent(agent_id="child-neg", initial_balance=Decimal("100"))
    await wallet.initialize()

    alloc = BudgetAllocation(child_did="did:child:neg", amount=Decimal("50"))
    dw = DelegatedWallet(wallet=wallet, allocation=alloc)

    with pytest.raises(ValueError, match="positive"):
        await dw.spend(Decimal("0"), "zero spend")

    with pytest.raises(ValueError, match="positive"):
        await dw.spend(Decimal("-5"), "negative spend")


# ---------------------------------------------------------------------------
# Transaction logging
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_transaction_logging():
    """All child spending is tracked in the transactions list."""
    wallet = WalletAgent(agent_id="child-log", initial_balance=Decimal("100"))
    await wallet.initialize()

    alloc = BudgetAllocation(child_did="did:child:log", amount=Decimal("50"))
    dw = DelegatedWallet(wallet=wallet, allocation=alloc)

    await dw.spend(Decimal("10"), "task A")
    await dw.spend(Decimal("5"), "task B")

    assert len(dw.transactions) == 2
    assert dw.transactions[0]["memo"] == "task A"
    assert dw.transactions[0]["amount"] == "10"
    assert dw.transactions[1]["memo"] == "task B"
    assert dw.transactions[1]["amount"] == "5"

    # Status reflects transactions
    status = dw.get_status()
    assert status["transaction_count"] == 2
    assert status["spent"] == "15"
    assert status["remaining"] == "35"


# ---------------------------------------------------------------------------
# Hold/Release lifecycle
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_hold_release_lifecycle():
    """Parent balance reduced at spawn, remainder returned at termination."""
    parent_wallet = WalletAgent(
        agent_id="parent-1",
        initial_balance=Decimal("100"),
    )
    await parent_wallet.initialize()

    parent_main_before = parent_wallet.get_balance(Currency.FIL, "main")
    assert parent_main_before == Decimal("90")  # 90% of 100

    # Hold: create delegated wallet with budget=30
    dw = await create_delegated_wallet(
        parent_wallet=parent_wallet,
        parent_did="did:parent:1",
        child_did="did:child:1",
        budget=Decimal("30"),
    )

    # Parent debited
    parent_main_after_hold = parent_wallet.get_balance(Currency.FIL, "main")
    assert parent_main_after_hold == Decimal("60")  # 90 - 30

    # Child has the budget
    assert dw.ceiling == Decimal("30")
    assert dw.remaining == Decimal("30")

    # Child spends some
    await dw.spend(Decimal("12"), "compute")
    await dw.spend(Decimal("8"), "storage")
    assert dw.spent == Decimal("20")

    # Release: return unspent to parent
    returned = await release_delegated_wallet(dw, parent_wallet)
    assert returned == Decimal("10")  # 30 - 20

    # Parent credited back
    parent_main_after_release = parent_wallet.get_balance(Currency.FIL, "main")
    assert parent_main_after_release == Decimal("70")  # 60 + 10


@pytest.mark.asyncio
async def test_hold_release_full_spend():
    """When child spends entire budget, nothing is returned to parent."""
    parent_wallet = WalletAgent(
        agent_id="parent-full",
        initial_balance=Decimal("100"),
    )
    await parent_wallet.initialize()

    dw = await create_delegated_wallet(
        parent_wallet=parent_wallet,
        parent_did="did:parent:full",
        child_did="did:child:full",
        budget=Decimal("20"),
    )

    await dw.spend(Decimal("20"), "all in")

    returned = await release_delegated_wallet(dw, parent_wallet)
    assert returned == Decimal("0")

    # Parent only got back 0
    parent_main = parent_wallet.get_balance(Currency.FIL, "main")
    assert parent_main == Decimal("70")  # 90 - 20 + 0


@pytest.mark.asyncio
async def test_hold_release_no_spend():
    """When child spends nothing, full budget is returned to parent."""
    parent_wallet = WalletAgent(
        agent_id="parent-none",
        initial_balance=Decimal("100"),
    )
    await parent_wallet.initialize()

    dw = await create_delegated_wallet(
        parent_wallet=parent_wallet,
        parent_did="did:parent:none",
        child_did="did:child:none",
        budget=Decimal("25"),
    )

    returned = await release_delegated_wallet(dw, parent_wallet)
    assert returned == Decimal("25")

    parent_main = parent_wallet.get_balance(Currency.FIL, "main")
    assert parent_main == Decimal("90")  # 90 - 25 + 25 = original


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_parent_insufficient_funds():
    """Creating a delegated wallet fails if parent cannot afford the budget."""
    parent_wallet = WalletAgent(
        agent_id="parent-poor",
        initial_balance=Decimal("10"),
    )
    await parent_wallet.initialize()
    # Main balance is 9 (90% of 10)

    with pytest.raises(ValueError, match="insufficient funds"):
        await create_delegated_wallet(
            parent_wallet=parent_wallet,
            parent_did="did:parent:poor",
            child_did="did:child:poor",
            budget=Decimal("20"),
        )


@pytest.mark.asyncio
async def test_zero_budget_delegation():
    """Zero budget delegation raises ValueError."""
    parent_wallet = WalletAgent(
        agent_id="parent-zero",
        initial_balance=Decimal("100"),
    )
    await parent_wallet.initialize()

    with pytest.raises(ValueError, match="positive"):
        await create_delegated_wallet(
            parent_wallet=parent_wallet,
            parent_did="did:parent:zero",
            child_did="did:child:zero",
            budget=Decimal("0"),
        )


@pytest.mark.asyncio
async def test_budget_exhaustion_then_release():
    """After exhausting budget, release returns zero."""
    parent_wallet = WalletAgent(
        agent_id="parent-exhaust",
        initial_balance=Decimal("100"),
    )
    await parent_wallet.initialize()

    dw = await create_delegated_wallet(
        parent_wallet=parent_wallet,
        parent_did="did:parent:exhaust",
        child_did="did:child:exhaust",
        budget=Decimal("15"),
    )

    await dw.spend(Decimal("15"), "exhaust budget")
    assert dw.allocation.is_exhausted

    # Further spending blocked
    with pytest.raises(BudgetExceededError):
        await dw.spend(Decimal("1"), "one more")

    returned = await release_delegated_wallet(dw, parent_wallet)
    assert returned == Decimal("0")


@pytest.mark.asyncio
async def test_can_spend_check():
    """can_spend returns correct boolean without side effects."""
    wallet = WalletAgent(agent_id="child-check", initial_balance=Decimal("100"))
    await wallet.initialize()

    alloc = BudgetAllocation(child_did="did:child:check", amount=Decimal("10"))
    dw = DelegatedWallet(wallet=wallet, allocation=alloc)

    assert dw.can_spend(Decimal("10")) is True
    assert dw.can_spend(Decimal("10.01")) is False
    assert dw.can_spend(Decimal("0.01")) is True

    # No side effects
    assert dw.spent == Decimal("0")
