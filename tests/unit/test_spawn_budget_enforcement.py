"""Per-child spawn budget enforcement via DelegatedWallet spend routing (#2113).

The DelegatedWallet machinery existed but was never wired into the child's spend
paths, so a `budget` was a no-op (later an interim rejection). These tests cover:
  * DelegatedWallet as a WalletProtocol drop-in (ceiling on transfer/can_afford,
    delegation of everything else),
  * the hold/release lifecycle,
  * AgentManager wiring: hold on spawn, ceiling'd child wallet, release on
    terminate, and refusal of a budget with no funded parent wallet.
"""

import asyncio
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from kestrel_sovereign.inception_service import generate_secp256k1_keypair
from kestrel_sovereign.multi_agent.config import LocalAgentConfig
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
        self._debit_intents = {}

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

    async def prepare_debit_intent(self, *, idempotency_key, amount, memo, currency):
        self._debit_intents.setdefault(
            idempotency_key,
            {"amount": amount, "memo": memo, "currency": currency, "outcome": False},
        )
        return idempotency_key

    async def execute_debit_intent(self, intent_id):
        intent = self._debit_intents[intent_id]
        if intent["outcome"] is True:
            return True
        outcome = await self.transfer(intent["amount"], intent["memo"], intent["currency"])
        intent["outcome"] = outcome
        return outcome

    async def resolve_debit_intent(self, intent_id):
        return self._debit_intents[intent_id]["outcome"]

    async def deposit(self, amount, currency=None, to_audit=False, memo=""):
        currency = currency or self._currency
        self._balances.setdefault(currency, {}).setdefault("main", Decimal("0"))
        self._balances[currency]["main"] += amount
        self.deposits.append((amount, memo))
        return True


class DurableHoldOnlyWallet(FakeWallet):
    """An older durable allocation provider without child provisioning."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reserve_calls = []

    def can_afford(self, amount, currency=None):
        raise AssertionError("durable allocation retries must not use stale can_afford")

    async def reserve_delegated_allocation(
        self, *, allocation_id, child_did, amount, memo, currency
    ):
        self.reserve_calls.append(allocation_id)
        record = getattr(self, "_allocations", {}).get(allocation_id)
        if record is not None:
            return True
        self._allocations = getattr(self, "_allocations", {})
        self._allocations[allocation_id] = amount
        self._balances[currency]["main"] -= amount
        return True

    async def release_delegated_allocation(self, *, allocation_id, amount, currency):
        self._balances[currency]["main"] += amount
        return True


class DurableProvisioningWallet(DurableHoldOnlyWallet):
    """Modern provider seam: Core never constructs or funds this child itself."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.provision_calls = []
        self.release_calls = []
        self.children = {}

    async def reserve_and_provision_delegated_child_wallet(
        self, *, allocation_id, child_did, amount, memo, currency
    ):
        self.provision_calls.append(allocation_id)
        child = self.children.get(allocation_id)
        if child is not None:
            return child
        await self.reserve_delegated_allocation(
            allocation_id=allocation_id,
            child_did=child_did,
            amount=amount,
            memo=memo,
            currency=currency,
        )
        child = FakeWallet(
            agent_id=child_did,
            initial_balance=Decimal("0"),
            initial_currency=currency,
        )
        child.db_path = "provider-owned-durable-child-storage"
        await child.initialize()
        child._balances[currency]["main"] = amount
        self.children[allocation_id] = child
        return child

    async def release_and_fence_delegated_child_wallet(
        self, *, allocation_id, currency
    ):
        self.release_calls.append(allocation_id)
        child = self.children[allocation_id]
        unspent = child.get_balance(currency, "main")
        child._balances[currency]["main"] = Decimal("0")
        self._balances[currency]["main"] += unspent
        return unspent


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
async def test_durable_allocation_retry_precedes_stale_affordability_for_legacy_provider():
    """Old durable providers still own idempotent hold retries before preflight."""

    parent = DurableHoldOnlyWallet(initial_balance=Decimal("100"))
    child = await create_delegated_wallet(parent, "did:p", "did:c", Decimal("30"))

    assert child.remaining == Decimal("30")
    assert parent.get_balance() == Decimal("70")
    # A re-run after the original hold must ask the allocation authority, not
    # the now-stale/insufficient cache, and must not debit twice.
    retried = await create_delegated_wallet(parent, "did:p", "did:c", Decimal("30"))
    assert retried.remaining == Decimal("30")
    assert parent.get_balance() == Decimal("70")
    assert len(parent.reserve_calls) == 2


@pytest.mark.asyncio
async def test_durable_provider_provisions_and_persists_child_without_core_wallet_internals():
    """Core uses the provider seam for storage and initial allocation atomically."""

    parent = DurableProvisioningWallet(initial_balance=Decimal("100"))
    delegated = await create_delegated_wallet(parent, "did:p", "did:c", Decimal("30"))

    assert delegated._wallet.db_path == "provider-owned-durable-child-storage"
    assert delegated.get_balance() == Decimal("30")
    assert parent.get_balance() == Decimal("70")
    retried = await create_delegated_wallet(parent, "did:p", "did:c", Decimal("30"))
    assert retried._wallet is delegated._wallet
    assert parent.get_balance() == Decimal("70")
    assert len(parent.provision_calls) == 2


@pytest.mark.asyncio
async def test_durable_child_release_uses_provider_balance_not_volatile_allocation_spend():
    """A reconstructed Core wrapper must not refund its stale local counter."""

    parent = DurableProvisioningWallet(initial_balance=Decimal("100"))
    delegated = await create_delegated_wallet(parent, "did:p", "did:c", Decimal("30"))
    assert await delegated.transfer(Decimal("7"), "durable child work") is True

    # Simulate process restart: this allocation presentation has no confirmed
    # spend history, but the provider-owned child wallet has only 23 left.
    restarted = DelegatedWallet(
        delegated._wallet,
        BudgetAllocation(
            child_did="did:c",
            parent_did="did:p",
            amount=Decimal("30"),
            allocation_id=delegated.allocation.allocation_id,
            parent_hold_durable=True,
        ),
    )
    assert await release_delegated_wallet(restarted, parent) == Decimal("23")
    assert parent.get_balance() == Decimal("93")
    assert parent.release_calls == [delegated.allocation.allocation_id]
    assert await release_delegated_wallet(restarted, parent) == Decimal("0")
    assert parent.get_balance() == Decimal("93")

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
async def test_nested_legacy_release_does_not_proxy_root_child_release_contract():
    """A grandchild's legacy hold returns through its immediate delegated parent."""

    class RootWithLegacyNestedAffordability(DurableProvisioningWallet):
        # The outer provider is durable, while its manually wrapped delegated
        # child follows the legacy nested-transfer path.
        can_afford = FakeWallet.can_afford

    root = RootWithLegacyNestedAffordability(initial_balance=Decimal("100"))
    child = DelegatedWallet(
        root, BudgetAllocation(child_did="did:child", amount=Decimal("50"))
    )
    grandchild = await create_delegated_wallet(
        child, "did:child", "did:grandchild", Decimal("20")
    )
    assert await grandchild.transfer(Decimal("5"), "grandchild work") is True

    # ``child`` proxies normal deposits to ``root``, but its legacy grandchild
    # allocation has no root-owned child-release ledger entry. Looking through
    # __getattr__ would invoke the root protocol and strand the refund.
    assert await release_delegated_wallet(grandchild, child) == Decimal("15")
    assert root.get_balance() == Decimal("95")
    assert child.spent == Decimal("5")
    assert root.release_calls == []


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
async def test_shutdown_all_releases_outstanding_holds(tmp_path):
    from kestrel_sovereign.spawn.mandate import SpawnMandate

    parent = SimpleNamespace(
        _private_key=generate_secp256k1_keypair()[0], identity=None, agent_id="did:p", features={},
        wallet=FakeWallet(initial_balance=Decimal("100")),
    )
    child = SimpleNamespace(agent_id="did:c", wallet=None, wallet_agent=None)
    mgr = _mgr_with_mock_child(child, base_data_dir=tmp_path)

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

def _mgr_with_mock_child(child, *, base_data_dir=None):
    from kestrel_sovereign.multi_agent.agent_manager import AgentManager

    mgr = AgentManager(base_data_dir=base_data_dir)

    if not hasattr(child, "shutdown"):
        child.shutdown = AsyncMock()

    async def fake_create_agent(name, parent_did=None, features=None, mandate=None):
        # Mimic load_agent registering the child, so the REAL remove_agent (the
        # path that releases budget holds — #2113) finds and stops it.
        child._raw_storage = SimpleNamespace(
            graph=SimpleNamespace(add_trusted_cross_agent_edge=AsyncMock())
        )
        admission = mgr._agent_operations[mgr._canonical_agent_name(name)]
        assert admission.before_publish is not None
        admission.spawn_candidate_config = LocalAgentConfig(
            data_dir=Path("agent_data") / name,
            port=8802,
        )
        pending = mgr._spawn_authority_registry.reserve_pending(
            child_name=name,
            parent_did=parent_did,
            mandate=mandate,
            config=admission.spawn_candidate_config,
        )
        admission.spawn_authority_pending_id = pending.reservation_id
        await admission.before_publish(child)
        mgr._agents[name] = child
        mgr._agent_names[child.agent_id] = name
        return child

    mgr.create_agent = fake_create_agent
    return mgr


@pytest.mark.asyncio
async def test_spawn_holds_budget_and_terminate_releases(tmp_path):
    from kestrel_sovereign.spawn.mandate import SpawnMandate

    parent = SimpleNamespace(
        _private_key=generate_secp256k1_keypair()[0], identity=None, agent_id="did:p", features={},
        wallet=FakeWallet(initial_balance=Decimal("100")),
    )
    child = SimpleNamespace(agent_id="did:c", wallet=None, wallet_agent=None)
    mgr = _mgr_with_mock_child(child, base_data_dir=tmp_path)

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
async def test_spawn_cancellation_after_provider_allocation_refunds_tracked_hold(
    tmp_path,
) -> None:
    """A cancellation blocked on lifecycle admission cannot leak a new hold."""

    from kestrel_sovereign.multi_agent.agent_manager import AgentManager
    from kestrel_sovereign.spawn.mandate import SpawnMandate

    manager = AgentManager(base_data_dir=tmp_path)
    provider_returned_with_lifecycle_lock = asyncio.Event()

    class LockHoldingProvisioner(DurableProvisioningWallet):
        async def reserve_and_provision_delegated_child_wallet(self, **kwargs):
            child_wallet = await super().reserve_and_provision_delegated_child_wallet(
                **kwargs
            )
            await manager._a2a_lifecycle_lock.acquire()
            provider_returned_with_lifecycle_lock.set()
            return child_wallet

    parent = SimpleNamespace(
        _private_key=generate_secp256k1_keypair()[0],
        identity=None,
        agent_id="did:test:provider-parent",
        features={},
        wallet=LockHoldingProvisioner(initial_balance=Decimal("100")),
    )
    child = SimpleNamespace(
        agent_id="did:test:provider-child", wallet=None, wallet_agent=None,
        shutdown=AsyncMock(),
    )

    async def fake_create_agent(name, **kwargs):
        child._raw_storage = SimpleNamespace(
            graph=SimpleNamespace(add_trusted_cross_agent_edge=AsyncMock())
        )
        admission = manager._agent_operations[manager._canonical_agent_name(name)]
        assert admission.before_publish is not None
        admission.spawn_candidate_config = LocalAgentConfig(
            data_dir=Path("agent_data") / name,
            port=8802,
        )
        pending = manager._spawn_authority_registry.reserve_pending(
            child_name=name,
            parent_did=kwargs["parent_did"],
            mandate=kwargs["mandate"],
            config=admission.spawn_candidate_config,
        )
        admission.spawn_authority_pending_id = pending.reservation_id
        try:
            await admission.before_publish(child)
        except BaseException:
            # Match the real create/load contract: revoke prepublication
            # authority while storage is live, then close the private child.
            assert admission.before_publish_rollback is not None
            await admission.before_publish_rollback(child)
            await child.shutdown()
            raise
        manager._agents[name] = child
        manager._agent_names[child.agent_id] = name
        return child

    manager.create_agent = fake_create_agent
    mandate = SpawnMandate(
        parent_did=parent.agent_id,
        purpose="allocation cancellation regression",
        budget_allocation=Decimal("30"),
        ttl_seconds=60,
    )
    spawn = asyncio.create_task(manager.spawn_agent("Kid", parent, mandate))
    await asyncio.wait_for(provider_returned_with_lifecycle_lock.wait(), timeout=1.0)
    try:
        # The provider has returned a durable positive allocation, and the
        # spawn is now waiting to re-enter the lifecycle writer. Tracking must
        # already be visible to rollback before this cancellation is delivered.
        await asyncio.sleep(0)
        assert "Kid" in manager._child_budgets
        spawn.cancel()
        await asyncio.sleep(0)
    finally:
        manager._a2a_lifecycle_lock.release()

    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(spawn, timeout=1.0)

    assert parent.wallet.get_balance() == Decimal("100")
    assert manager._child_budgets == {}
    assert manager.get_agent("Kid") is None
    assert manager.get_children(parent.agent_id) == []
    assert manager.get_mandate("Kid") is None
    child.shutdown.assert_awaited_once()


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
async def test_direct_remove_agent_releases_budget(tmp_path):
    """A budgeted child deleted through the generic remove_agent path (DELETE
    /api/agents/{name}) — not terminate_child — still releases its hold."""
    from kestrel_sovereign.multi_agent.agent_manager import AgentManager
    from kestrel_sovereign.spawn.mandate import SpawnMandate

    parent = SimpleNamespace(
        _private_key=generate_secp256k1_keypair()[0], identity=None, agent_id="did:p", features={},
        wallet=FakeWallet(initial_balance=Decimal("100")),
    )
    child = SimpleNamespace(
        agent_id="did:c", wallet=None, wallet_agent=None, shutdown=AsyncMock()
    )
    mgr = AgentManager(base_data_dir=tmp_path)

    async def fake_create_agent(name, parent_did=None, features=None, mandate=None):
        # Match the public create/load contract: a spawn may only commit after
        # its exact child is published to both routing maps.
        child._raw_storage = SimpleNamespace(
            graph=SimpleNamespace(add_trusted_cross_agent_edge=AsyncMock())
        )
        admission = mgr._agent_operations[mgr._canonical_agent_name(name)]
        assert admission.before_publish is not None
        admission.spawn_candidate_config = LocalAgentConfig(
            data_dir=Path("agent_data") / name,
            port=8802,
        )
        pending = mgr._spawn_authority_registry.reserve_pending(
            child_name=name,
            parent_did=parent_did,
            mandate=mandate,
            config=admission.spawn_candidate_config,
        )
        admission.spawn_authority_pending_id = pending.reservation_id
        await admission.before_publish(child)
        mgr._agents[name] = child
        mgr._agent_names[child.agent_id] = name
        return child

    mgr.create_agent = fake_create_agent  # real remove_agent (the path under test)

    mandate = SpawnMandate(parent_did="did:p", purpose="x", budget_allocation=Decimal("30"))
    await mgr.spawn_agent("Kid", parent, mandate)
    assert parent.wallet.get_balance() == Decimal("70")

    await mgr.remove_agent("Kid")
    assert parent.wallet.get_balance() == Decimal("100")   # hold released on delete


@pytest.mark.asyncio
async def test_budget_allocation_helper_leaves_child_cleanup_to_spawn_owner():
    """Allocation failure alone must not bypass receipt-first spawn rollback."""

    from kestrel_sovereign.multi_agent.agent_manager import AgentManager
    from kestrel_sovereign.spawn.mandate import SpawnMandate

    manager = AgentManager()
    parent = SimpleNamespace(
        agent_id="did:test:budget-parent",
        wallet=FakeWallet(initial_balance=Decimal("100")),
    )
    child = SimpleNamespace(agent_id="did:test:budget-child")
    mandate = SpawnMandate(
        parent_did=parent.agent_id,
        purpose="rollback",
        budget_allocation=Decimal("30"),
        ttl_seconds=60,
    )
    manager.remove_agent = AsyncMock()
    allocation_failure = RuntimeError("allocation failed")

    with patch(
        "kestrel_sovereign.multi_agent.agent_manager.create_delegated_wallet",
        new=AsyncMock(side_effect=allocation_failure),
    ):
        with pytest.raises(RuntimeError) as raised:
            await manager._apply_delegated_budget(
                "Child", parent, child, mandate
            )

    assert raised.value is allocation_failure
    manager.remove_agent.assert_not_awaited()


@pytest.mark.asyncio
async def test_budget_allocation_failure_revokes_receipt_before_child_shutdown(
    tmp_path,
):
    """The outer spawn owner revokes authority while child graph storage is live."""

    from kestrel_sovereign.multi_agent.agent_manager import AgentManager
    from kestrel_sovereign.spawn.mandate import SpawnMandate

    manager = AgentManager(base_data_dir=tmp_path)
    events: list[str] = []

    class ClosingGraph:
        closed = False

        async def add_trusted_cross_agent_edge(
            self, _source, _target, _relation, *, properties
        ) -> None:
            if self.closed:
                raise RuntimeError("receipt graph is already closed")
            events.append(
                "signed-receipt"
                if properties.get("parent_signature")
                else "unsigned-lineage"
            )

    graph = ClosingGraph()
    parent = SimpleNamespace(
        _private_key=generate_secp256k1_keypair()[0],
        identity=None,
        agent_id="did:test:budget-parent-non-hosted",
        features={},
        wallet=FakeWallet(initial_balance=Decimal("100")),
    )

    async def close_child() -> None:
        events.append("child-shutdown")
        graph.closed = True

    child = SimpleNamespace(
        agent_id="did:test:budget-child-non-hosted",
        wallet=None,
        wallet_agent=None,
        shutdown=AsyncMock(side_effect=close_child),
        _raw_storage=SimpleNamespace(graph=graph),
    )

    async def create_and_publish(name, **kwargs):
        admission = manager._agent_operations[manager._canonical_agent_name(name)]
        assert admission.before_publish is not None
        admission.spawn_candidate_config = LocalAgentConfig(
            data_dir=Path("agent_data") / name,
            port=8802,
        )
        pending = manager._spawn_authority_registry.reserve_pending(
            child_name=name,
            parent_did=kwargs["parent_did"],
            mandate=kwargs["mandate"],
            config=admission.spawn_candidate_config,
        )
        admission.spawn_authority_pending_id = pending.reservation_id
        try:
            await admission.before_publish(child)
        except BaseException:
            # Match load_agent's prepublication ownership: once the receipt
            # inverse succeeds, the nested loader closes its private child
            # before the outer spawn operation resumes rollback.
            assert admission.before_publish_rollback is not None
            await admission.before_publish_rollback(child)
            await child.shutdown()
            raise
        manager._agents[name] = child
        manager._agent_names[child.agent_id] = name
        return child

    manager.create_agent = create_and_publish
    mandate = SpawnMandate(
        parent_did=parent.agent_id,
        purpose="rollback",
        budget_allocation=Decimal("30"),
        ttl_seconds=60,
    )
    allocation_failure = RuntimeError("allocation failed")

    with patch(
        "kestrel_sovereign.multi_agent.agent_manager.create_delegated_wallet",
        new=AsyncMock(side_effect=allocation_failure),
    ):
        with pytest.raises(RuntimeError) as raised:
            await manager.spawn_agent("Child", parent, mandate)

    assert raised.value is allocation_failure
    assert events == ["signed-receipt", "unsigned-lineage", "child-shutdown"]
    child.shutdown.assert_awaited_once_with()
    assert manager.get_agent("Child") is None
    assert child.agent_id not in manager._agent_names


@pytest.mark.asyncio
async def test_casefolded_delegated_hold_blocks_new_child_admission() -> None:
    """A restored Foo hold must reserve foo before any new child initializes."""

    from kestrel_sovereign.multi_agent.agent_manager import AgentManager
    from kestrel_sovereign.multi_agent.config import LocalAgentConfig

    manager = AgentManager()
    manager._child_budgets["Foo"] = (object(), object())
    initialize = AsyncMock()
    manager._initialize_agent = initialize

    with pytest.raises(RuntimeError, match="unresolved delegated budget cleanup"):
        await manager.load_agent("foo", LocalAgentConfig(data_dir="foo", port=8801))

    initialize.assert_not_awaited()


@pytest.mark.asyncio
async def test_terminate_child_keeps_retry_tracking_after_refund_failure() -> None:
    """A failed stop-then-refund retains the parent edge and mandate for retry."""

    from kestrel_sovereign.multi_agent.agent_manager import AgentManager
    from kestrel_sovereign.spawn.mandate import SpawnMandate

    manager = AgentManager()
    child = SimpleNamespace(agent_id="did:test:retry-child", shutdown=AsyncMock())
    entry = (object(), object())
    mandate = SpawnMandate(parent_did="did:test:retry-parent", purpose="retry")
    manager._agents["retry-child"] = child
    manager._agent_names[child.agent_id] = "retry-child"
    manager._child_budgets["retry-child"] = entry
    manager._parent_children["did:test:retry-parent"] = ["retry-child"]
    manager._child_mandates["retry-child"] = mandate

    async def fail_refund(name: str) -> bool:
        assert name == "retry-child"
        raise RuntimeError("refund provider failed")

    manager._release_child_budget_cancellation_safe = fail_refund
    with pytest.raises(RuntimeError, match="refund provider failed"):
        await manager.terminate_child("did:test:retry-parent", "retry-child")

    # remove_agent has withdrawn the stopped child, but it retained the exact
    # hold.  terminate_child must preserve the relation that permits the
    # normal retry to find that hold and its governance mandate.
    assert manager.get_children("did:test:retry-parent") == ["retry-child"]
    assert manager.get_mandate("retry-child") is mandate
    assert manager._child_budgets["retry-child"] is entry

    async def release_refund(name: str) -> bool:
        assert name == "retry-child"
        assert manager._child_budgets.pop(name) is entry
        return False

    manager._release_child_budget_cancellation_safe = release_refund
    assert await manager.terminate_child("did:test:retry-parent", "retry-child")
    assert manager.get_children("did:test:retry-parent") == []
    assert manager.get_mandate("retry-child") is None


@pytest.mark.asyncio
async def test_terminate_child_prunes_tracking_after_completed_removal_cancellation() -> None:
    """A terminal DELETE cancellation must not consume a spawn-cap slot forever."""

    from kestrel_sovereign.multi_agent.agent_manager import AgentManager
    from kestrel_sovereign.spawn.mandate import SpawnMandate

    manager = AgentManager()
    child = SimpleNamespace(
        agent_id="did:test:cancelled-child", shutdown=AsyncMock()
    )
    entry = (object(), object())
    mandate = SpawnMandate(parent_did="did:test:cancelled-parent", purpose="retry")
    manager._agents["cancelled-child"] = child
    manager._agent_names[child.agent_id] = "cancelled-child"
    manager._child_budgets["cancelled-child"] = entry
    manager._parent_children["did:test:cancelled-parent"] = ["cancelled-child"]
    manager._child_mandates["cancelled-child"] = mandate

    async def refund_then_report_cancellation(name: str) -> bool:
        assert name == "cancelled-child"
        assert manager._child_budgets.pop(name) is entry
        # This is the remove_agent contract after shutdown and refund have
        # completed while the caller's cancellation is still pending.
        return True

    manager._release_child_budget_cancellation_safe = refund_then_report_cancellation
    with pytest.raises(asyncio.CancelledError):
        await manager.terminate_child("did:test:cancelled-parent", "cancelled-child")

    child.shutdown.assert_awaited_once()
    assert manager.get_agent("cancelled-child") is None
    assert "cancelled-child" not in manager._child_budgets
    assert manager.get_children("did:test:cancelled-parent") == []
    assert manager.get_mandate("cancelled-child") is None
    assert manager._pending_spawns == 0


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
async def test_remove_agent_refuses_budgeted_parent():
    """remove_agent (single-agent primitive) refuses to delete an agent that has
    budgeted descendants — that would strand their holds; use terminate_child."""
    from kestrel_sovereign.multi_agent.agent_manager import AgentManager

    root = FakeWallet(initial_balance=Decimal("100"))
    child_dw = await create_delegated_wallet(root, "did:root", "did:child", Decimal("30"))
    gc_dw = await create_delegated_wallet(child_dw, "did:child", "did:gc", Decimal("20"))

    mgr = AgentManager()
    mgr._agents = {
        "child": SimpleNamespace(agent_id="did:child", shutdown=AsyncMock()),
        "gc": SimpleNamespace(agent_id="did:gc", shutdown=AsyncMock()),
    }
    mgr._parent_children = {"did:child": ["gc"]}
    mgr._child_budgets = {"child": (child_dw, root), "gc": (gc_dw, child_dw)}

    with pytest.raises(ValueError, match="budgeted child agents"):
        await mgr.remove_agent("child")


@pytest.mark.asyncio
async def test_budget_refused_for_persistent_child():
    """Budgets are in-process only, so they're restricted to ephemeral (TTL)
    children that can't be reloaded uncapped; a persistent spawn is refused."""
    from kestrel_sovereign.spawn.mandate import SpawnMandate

    parent = SimpleNamespace(
        _private_key=generate_secp256k1_keypair()[0], identity=None, agent_id="did:p", features={},
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
        _private_key=generate_secp256k1_keypair()[0], identity=None, agent_id="did:p", features={}, wallet=None,
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
        _private_key=generate_secp256k1_keypair()[0], identity=None, agent_id="did:p", features={},
        wallet=FakeWallet(initial_balance=Decimal("3")),
    )
    child = SimpleNamespace(agent_id="did:c", wallet=None, wallet_agent=None)
    mgr = _mgr_with_mock_child(child)

    mandate = SpawnMandate(parent_did="did:p", purpose="x", budget_allocation=Decimal("50"))
    with pytest.raises(ValueError, match="cannot afford"):
        await mgr.spawn_agent("Kid", parent, mandate)


@pytest.mark.asyncio
async def test_agent_manager_bypasses_stale_preflight_for_durable_provisioner():
    """Only the provider's atomic reserve may decide a durable spawn budget."""

    from kestrel_sovereign.multi_agent.agent_manager import AgentManager
    from kestrel_sovereign.spawn.mandate import SpawnMandate

    # ``DurableProvisioningWallet.can_afford`` deliberately raises: it models
    # a cross-process stale snapshot, while its reserve/provision seam is the
    # authoritative transaction exercised by the wallet package regression.
    parent = SimpleNamespace(
        agent_id="did:p",
        wallet=DurableProvisioningWallet(initial_balance=Decimal("30")),
    )
    mandate = SpawnMandate(
        parent_did="did:p",
        purpose="x",
        budget_allocation=Decimal("30"),
        ttl_seconds=60,
    )

    AgentManager()._validate_budget_precondition(parent, mandate)


@pytest.mark.asyncio
async def test_no_budget_leaves_wallet_untouched(tmp_path):
    from kestrel_sovereign.spawn.mandate import SpawnMandate

    original = FakeWallet(initial_balance=Decimal("100"))
    parent = SimpleNamespace(
        _private_key=generate_secp256k1_keypair()[0], identity=None, agent_id="did:p", features={}, wallet=original,
    )
    child = SimpleNamespace(agent_id="did:c", wallet="preexisting", wallet_agent=None)
    mgr = _mgr_with_mock_child(child, base_data_dir=tmp_path)

    mandate = SpawnMandate(parent_did="did:p", purpose="x")  # budget defaults to 0
    await mgr.spawn_agent("Kid", parent, mandate)

    assert child.wallet == "preexisting"           # not replaced
    assert original.get_balance() == Decimal("100")  # nothing held
