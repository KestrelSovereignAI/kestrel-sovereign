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
from unittest.mock import AsyncMock

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


class _InitFailWallet(FakeWallet):
    async def initialize(self):
        raise RuntimeError("wallet provider init failed")


@pytest.mark.asyncio
async def test_create_refunds_parent_on_setup_failure():
    """If child-wallet setup fails AFTER the parent is debited, the hold is
    refunded rather than stranded."""
    parent = _InitFailWallet(initial_balance=Decimal("100"))
    with pytest.raises(RuntimeError, match="provider init"):
        await create_delegated_wallet(parent, "did:p", "did:c", Decimal("30"))
    assert parent.get_balance() == Decimal("100")  # debit refunded


@pytest.mark.asyncio
async def test_failed_nested_setup_restores_parent_headroom():
    """A failed budgeted grandchild spawn refunds the money AND restores the
    budgeted parent's headroom (not just the wrapped balance)."""
    real = _InitFailWallet(initial_balance=Decimal("100"))
    child_dw = DelegatedWallet(
        real, BudgetAllocation(child_did="c", amount=Decimal("50"))
    )
    with pytest.raises(RuntimeError, match="provider init"):
        await create_delegated_wallet(child_dw, "did:c", "did:gc", Decimal("20"))

    assert child_dw.spent == Decimal("0")          # headroom restored
    assert real.get_balance() == Decimal("100")    # funds refunded


@pytest.mark.asyncio
async def test_release_restores_parent_headroom():
    """Releasing a budgeted grandchild restores its budgeted parent's headroom
    (only the explicit release does this — see the deposit test below)."""
    real = FakeWallet(initial_balance=Decimal("100"))
    child_dw = DelegatedWallet(
        real, BudgetAllocation(child_did="c", amount=Decimal("50"))
    )
    gc = await create_delegated_wallet(child_dw, "did:c", "did:gc", Decimal("20"))
    assert child_dw.spent == Decimal("20")

    await gc.transfer(Decimal("5"), "gc work")
    returned = await release_delegated_wallet(gc, child_dw)
    assert returned == Decimal("15")

    # The 15 refunded restores the child's budget: net spent 20 - 15 = 5.
    assert child_dw.spent == Decimal("5")
    assert child_dw.remaining == Decimal("45")


@pytest.mark.asyncio
async def test_normal_deposit_does_not_restore_headroom():
    """A plain external top-up must NOT restore budget headroom, or a child could
    spend past its ceiling after any deposit."""
    real = FakeWallet(initial_balance=Decimal("100"))
    dw = DelegatedWallet(real, BudgetAllocation(child_did="c", amount=Decimal("10")))
    await dw.transfer(Decimal("6"), "spend")
    assert dw.spent == Decimal("6")

    await dw.deposit(Decimal("50"))                # external top-up, delegated
    assert dw.spent == Decimal("6")                # unchanged
    assert dw.remaining == Decimal("4")
    assert dw.can_afford(Decimal("5")) is False    # ceiling still enforced


@pytest.mark.asyncio
async def test_shutdown_all_releases_outstanding_holds():
    from kestrel_sovereign.spawn.mandate import SpawnMandate

    parent = SimpleNamespace(
        _private_key=None, identity=None, agent_id="did:p", features={},
        wallet=FakeWallet(initial_balance=Decimal("100")),
    )
    child = SimpleNamespace(agent_id="did:c", wallet=None, wallet_agent=None)
    mgr = _mgr_with_mock_child(child)

    mandate = SpawnMandate(parent_did="did:p", purpose="x", budget_allocation=Decimal("30"))
    await mgr.spawn_agent("Kid", parent, mandate)
    assert parent.wallet.get_balance() == Decimal("70")

    await mgr.shutdown_all()
    assert parent.wallet.get_balance() == Decimal("100")   # hold released on shutdown


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

    if not hasattr(child, "shutdown"):
        child.shutdown = AsyncMock()

    async def fake_create_agent(name, parent_did=None, features=None, mandate=None):
        # Mimic load_agent registering the child, so the REAL remove_agent (the
        # path that releases budget holds — #2113) finds and stops it.
        mgr._agents[name] = child
        mgr._agent_names[child.agent_id] = name
        return child

    mgr.create_agent = fake_create_agent
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
async def test_shutdown_releases_nested_budgets_leaf_first():
    """Graceful shutdown releases a budgeted grandchild before its budgeted
    parent, so ALL unspent funds flow back to the root (not stranded in an
    already-released parent wallet)."""
    from kestrel_sovereign.multi_agent.agent_manager import AgentManager

    root = FakeWallet(initial_balance=Decimal("100"))
    child_dw = await create_delegated_wallet(root, "did:root", "did:child", Decimal("30"))
    gc_dw = await create_delegated_wallet(child_dw, "did:child", "did:gc", Decimal("20"))
    assert root.get_balance() == Decimal("70")   # 30 held from root
    assert child_dw.spent == Decimal("20")        # 20 held for grandchild

    mgr = AgentManager()
    # Insertion order = spawn order (ancestor before descendant).
    mgr._child_budgets = {"child": (child_dw, root), "gc": (gc_dw, child_dw)}
    await mgr.shutdown_all()

    # gc released first (refunds child) → child released → full 30 back to root.
    assert root.get_balance() == Decimal("100")


@pytest.mark.asyncio
async def test_direct_remove_agent_releases_budget():
    """A budgeted child deleted through the generic remove_agent path (DELETE
    /api/agents/{name}) — not terminate_child — still releases its hold."""
    from kestrel_sovereign.multi_agent.agent_manager import AgentManager
    from kestrel_sovereign.spawn.mandate import SpawnMandate

    parent = SimpleNamespace(
        _private_key=None, identity=None, agent_id="did:p", features={},
        wallet=FakeWallet(initial_balance=Decimal("100")),
    )
    child = SimpleNamespace(agent_id="did:c", wallet=None, wallet_agent=None)
    mgr = AgentManager()

    async def fake_create_agent(name, parent_did=None, features=None, mandate=None):
        return child

    mgr.create_agent = fake_create_agent  # real remove_agent (the path under test)

    mandate = SpawnMandate(parent_did="did:p", purpose="x", budget_allocation=Decimal("30"))
    await mgr.spawn_agent("Kid", parent, mandate)
    assert parent.wallet.get_balance() == Decimal("70")

    await mgr.remove_agent("Kid")
    assert parent.wallet.get_balance() == Decimal("100")   # hold released on delete


@pytest.mark.asyncio
async def test_terminate_child_cascade_releases_nested_to_root():
    """terminate_child (the path budgeted children are torn down through) stops
    and releases a budgeted grandchild before its parent, so all held funds flow
    back to the root."""
    from kestrel_sovereign.multi_agent.agent_manager import AgentManager

    root = FakeWallet(initial_balance=Decimal("100"))
    child_dw = await create_delegated_wallet(root, "did:root", "did:child", Decimal("30"))
    gc_dw = await create_delegated_wallet(child_dw, "did:child", "did:gc", Decimal("20"))
    assert root.get_balance() == Decimal("70")

    mgr = AgentManager()
    mgr._agents = {
        "child": SimpleNamespace(agent_id="did:child", shutdown=AsyncMock()),
        "gc": SimpleNamespace(agent_id="did:gc", shutdown=AsyncMock()),
    }
    # root spawned child; child spawned gc.
    mgr._parent_children = {"did:root": ["child"], "did:child": ["gc"]}
    mgr._child_budgets = {"child": (child_dw, root), "gc": (gc_dw, child_dw)}

    await mgr.terminate_child("did:root", "child")
    assert root.get_balance() == Decimal("100")   # gc released into child, then child to root


@pytest.mark.asyncio
async def test_remove_agent_is_single_agent_release():
    """remove_agent is a leaf primitive: it releases only the NAMED agent's hold
    and does not cascade (nested teardown is terminate_child's job). A childless
    budgeted agent's hold returns to its parent."""
    from kestrel_sovereign.multi_agent.agent_manager import AgentManager

    parent = FakeWallet(initial_balance=Decimal("100"))
    dw = await create_delegated_wallet(parent, "did:p", "did:c", Decimal("30"))
    assert parent.get_balance() == Decimal("70")

    mgr = AgentManager()
    mgr._agents = {"c": SimpleNamespace(agent_id="did:c", shutdown=AsyncMock())}
    mgr._child_budgets = {"c": (dw, parent)}

    await mgr.remove_agent("c")
    assert parent.get_balance() == Decimal("100")   # own hold released


@pytest.mark.asyncio
async def test_budget_refused_for_persistent_child():
    """Budgets are in-process only, so they're restricted to ephemeral (TTL)
    children that can't be reloaded uncapped; a persistent spawn is refused."""
    from kestrel_sovereign.spawn.mandate import SpawnMandate

    parent = SimpleNamespace(
        _private_key=None, identity=None, agent_id="did:p", features={},
        wallet=FakeWallet(initial_balance=Decimal("100")),
    )
    child = SimpleNamespace(agent_id="did:c", wallet=None, wallet_agent=None)
    mgr = _mgr_with_mock_child(child)

    mandate = SpawnMandate(
        parent_did="did:p", purpose="x", budget_allocation=Decimal("30"), ttl_seconds=0,
    )
    with pytest.raises(ValueError, match="ephemeral"):
        await mgr.spawn_agent("Kid", parent, mandate)


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
