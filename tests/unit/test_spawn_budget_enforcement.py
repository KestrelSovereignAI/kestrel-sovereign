"""Per-child spawn budget enforcement via DelegatedWallet spend routing (#2113).

The DelegatedWallet machinery existed but was never wired into the child's spend
paths, so a `budget` was a no-op (later an interim rejection). These tests cover:
  * DelegatedWallet as a WalletProtocol drop-in (ceiling on transfer/can_afford,
    delegation of everything else),
  * the hold/release lifecycle,
  * AgentManager wiring: hold on spawn, ceiling'd child wallet, release on
    terminate, and refusal of a budget with no funded parent wallet.
"""

from decimal import Decimal
from types import SimpleNamespace

import pytest

from kestrel_sovereign.spawn.delegated_wallet import (
    BudgetAllocation,
    BudgetExceededError,
    DelegatedWallet,
    create_delegated_wallet,
    release_delegated_wallet,
)


class FakeWallet:
    """Minimal WalletProtocol implementation for tests (stands in for the
    external WalletAgent)."""

    def __init__(self, agent_id=None, initial_balance=Decimal("0"),
                 initial_currency="FIL"):
        self.agent_id = agent_id
        self._currency = initial_currency
        self._balances = {initial_currency: {"main": Decimal(initial_balance)}}
        self.deposits = []

    async def initialize(self):
        return None

    def _bal(self, currency, balance_type="main"):
        return self._balances.get(currency, {}).get(balance_type, Decimal("0"))

    def can_afford(self, amount, currency=None):
        return self._bal(currency or self._currency) >= amount

    def get_balance(self, currency=None, balance_type="main"):
        return self._bal(currency or self._currency, balance_type)

    async def transfer(self, amount, memo="", currency=None):
        currency = currency or self._currency
        if self._bal(currency) < amount:
            return False
        self._balances[currency]["main"] = self._bal(currency) - amount
        return True

    async def deposit(self, amount, currency=None, to_audit=False, memo=""):
        currency = currency or self._currency
        self._balances.setdefault(currency, {}).setdefault("main", Decimal("0"))
        self._balances[currency]["main"] += amount
        self.deposits.append((amount, memo))
        return True


# --------------------------- DelegatedWallet drop-in ---------------------------

@pytest.mark.asyncio
async def test_transfer_enforces_ceiling():
    funded = FakeWallet(initial_balance=Decimal("100"))
    dw = DelegatedWallet(funded, BudgetAllocation(child_did="c", amount=Decimal("10")))

    assert await dw.transfer(Decimal("4"), "a") is True
    assert dw.spent == Decimal("4")
    assert funded.get_balance() == Decimal("96")  # wrapped wallet debited

    assert dw.can_afford(Decimal("6")) is True     # exactly at ceiling
    assert dw.can_afford(Decimal("7")) is False     # over ceiling

    with pytest.raises(BudgetExceededError):
        await dw.transfer(Decimal("7"), "over")
    assert dw.spent == Decimal("4")                 # overspend didn't debit


@pytest.mark.asyncio
async def test_get_balance_capped_at_remaining():
    funded = FakeWallet(initial_balance=Decimal("100"))
    dw = DelegatedWallet(funded, BudgetAllocation(child_did="c", amount=Decimal("10")))
    # Wrapped wallet holds 100, but the child may never see more than its ceiling.
    assert dw.get_balance() == Decimal("10")
    await dw.transfer(Decimal("4"), "a")
    assert dw.get_balance() == Decimal("6")


@pytest.mark.asyncio
async def test_delegates_unoverridden_attrs():
    funded = FakeWallet(agent_id="c", initial_balance=Decimal("5"))
    dw = DelegatedWallet(funded, BudgetAllocation(child_did="c", amount=Decimal("10")))
    # _balances / agent_id / deposit are not overridden → delegated to wrapped.
    assert dw._balances is funded._balances
    assert dw.agent_id == "c"
    assert await dw.deposit(Decimal("3")) is True
    assert funded.get_balance() == Decimal("8")


# ----------------------------- hold / release ---------------------------------

@pytest.mark.asyncio
async def test_nested_budgeted_spawn_unwraps_delegated_parent():
    """A budgeted child (its wallet already a DelegatedWallet) spawning a
    budgeted grandchild: the hold goes through the child's ceiling, and the
    grandchild's funded wallet is built from the concrete class, not
    DelegatedWallet."""
    real = FakeWallet(initial_balance=Decimal("100"))
    parent_dw = DelegatedWallet(
        real, BudgetAllocation(child_did="c", amount=Decimal("50"))
    )
    gc = await create_delegated_wallet(parent_dw, "did:c", "did:gc", Decimal("20"))

    assert parent_dw.spent == Decimal("20")        # held through child's ceiling
    assert real.get_balance() == Decimal("80")
    assert gc.remaining == Decimal("20")
    assert isinstance(gc._wallet, FakeWallet)      # not a nested DelegatedWallet


@pytest.mark.asyncio
async def test_create_holds_and_release_returns_unspent():
    parent = FakeWallet(initial_balance=Decimal("100"))
    dw = await create_delegated_wallet(parent, "did:p", "did:c", Decimal("30"))
    assert parent.get_balance() == Decimal("70")   # 30 held from parent

    await dw.transfer(Decimal("10"), "work")
    assert dw.spent == Decimal("10")

    returned = await release_delegated_wallet(dw, parent)
    assert returned == Decimal("20")               # unspent
    assert parent.get_balance() == Decimal("90")   # 70 + 20 back


# ----------------------------- AgentManager wiring ----------------------------

def _mgr_with_mock_child(child):
    from kestrel_sovereign.multi_agent.agent_manager import AgentManager

    mgr = AgentManager()

    async def fake_create_agent(name, parent_did=None, features=None, mandate=None):
        return child

    async def fake_remove_agent(name):
        return True

    mgr.create_agent = fake_create_agent
    mgr.remove_agent = fake_remove_agent
    return mgr


@pytest.mark.asyncio
async def test_spawn_holds_budget_and_terminate_releases():
    from kestrel_sovereign.multi_agent.agent_manager import AgentManager
    from kestrel_sovereign.spawn.mandate import SpawnMandate

    parent = SimpleNamespace(
        _private_key=None, identity=None, agent_id="did:p", features={},
        wallet=FakeWallet(initial_balance=Decimal("100")),
    )
    child = SimpleNamespace(agent_id="did:c", wallet=None, wallet_agent=None)
    mgr = _mgr_with_mock_child(child)

    mandate = SpawnMandate(parent_did="did:p", purpose="x", budget_allocation=Decimal("30"))
    result = await mgr.spawn_agent("Kid", parent, mandate)

    assert result is child
    assert isinstance(child.wallet, DelegatedWallet)
    assert child.wallet_agent is child.wallet
    assert child._delegated_wallet is child.wallet          # spawn-status endpoint reads this
    assert parent.wallet.get_balance() == Decimal("70")     # held

    await child.wallet.transfer(Decimal("10"), "work")       # spend within ceiling

    assert await mgr.terminate_child("did:p", "Kid") is True
    assert parent.wallet.get_balance() == Decimal("90")      # 20 unspent released


@pytest.mark.asyncio
async def test_budget_refused_without_funded_parent_wallet():
    from kestrel_sovereign.spawn.mandate import SpawnMandate

    parent = SimpleNamespace(
        _private_key=None, identity=None, agent_id="did:p", features={}, wallet=None,
    )
    child = SimpleNamespace(agent_id="did:c", wallet=None, wallet_agent=None)
    mgr = _mgr_with_mock_child(child)

    mandate = SpawnMandate(parent_did="did:p", purpose="x", budget_allocation=Decimal("5"))
    with pytest.raises(ValueError, match="funded wallet"):
        await mgr.spawn_agent("Kid", parent, mandate)


@pytest.mark.asyncio
async def test_budget_refused_when_parent_cannot_afford():
    from kestrel_sovereign.spawn.mandate import SpawnMandate

    parent = SimpleNamespace(
        _private_key=None, identity=None, agent_id="did:p", features={},
        wallet=FakeWallet(initial_balance=Decimal("3")),
    )
    child = SimpleNamespace(agent_id="did:c", wallet=None, wallet_agent=None)
    mgr = _mgr_with_mock_child(child)

    mandate = SpawnMandate(parent_did="did:p", purpose="x", budget_allocation=Decimal("50"))
    with pytest.raises(ValueError, match="cannot afford"):
        await mgr.spawn_agent("Kid", parent, mandate)


@pytest.mark.asyncio
async def test_no_budget_leaves_wallet_untouched():
    from kestrel_sovereign.spawn.mandate import SpawnMandate

    original = FakeWallet(initial_balance=Decimal("100"))
    parent = SimpleNamespace(
        _private_key=None, identity=None, agent_id="did:p", features={}, wallet=original,
    )
    child = SimpleNamespace(agent_id="did:c", wallet="preexisting", wallet_agent=None)
    mgr = _mgr_with_mock_child(child)

    mandate = SpawnMandate(parent_did="did:p", purpose="x")  # budget defaults to 0
    await mgr.spawn_agent("Kid", parent, mandate)

    assert child.wallet == "preexisting"           # not replaced
    assert original.get_balance() == Decimal("100")  # nothing held
