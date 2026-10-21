"""Whole refund transfer admission/publication with an actual local PG peer."""
import asyncio
import inspect
from decimal import Decimal

import pytest

from kestrel_sovereign._async_ownership import await_owned_task, create_owned_task
from kestrel_sovereign.execution_custody import (
    _explicit_control_chain, bind_execution_custody, execution_cleanup_debts, is_execution_control_error,
)
from kestrel_sovereign.multi_agent.agent_manager import AgentManager
from kestrel_sovereign.spawn.delegated_wallet import BudgetAllocation, DelegatedWallet
from tests.integration.test_execution_authority_postgres import GenerationFence, native_pg as native_pg
from tests.integration.test_execution_custody_review30_pg import lose_ack
from tests.unit.test_host_context_native_cleanup import _actual_operation


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["quarantine-admission", "quarantine-publication", "inner-admission"])
async def test_refund_transfer_never_drops_original_hold_or_accepted_owner(native_pg, tmp_path, monkeypatch, mode):
    backend, observer, schema = native_pg
    credit_entered, credit_release = asyncio.Event(), asyncio.Event()

    class LocalLedger:
        _balances = {"FIL": {"main": Decimal("0")}}

        async def deposit(self, amount, currency=None, to_audit=False, memo=""):
            assert amount == Decimal("10") and currency == "FIL" and not to_audit
            with bind_execution_custody(GenerationFence()):
                await backend.execute("INSERT INTO effects VALUES (220, 'original native refund transfer')")
            return True

    parent = LocalLedger()
    delegated = DelegatedWallet(parent, BudgetAllocation(child_did="did:test:refund-transfer", amount=Decimal("10")))
    entry = (delegated, parent)
    manager = AgentManager(base_data_dir=tmp_path)
    manager._child_budgets["child"] = entry
    loop, previous = asyncio.get_running_loop(), asyncio.get_running_loop().get_task_factory()
    refused, accepted = [], []
    refusal = RuntimeError("actual original refund transfer publication refused")
    owner = None

    def factory(loop, coroutine, **kwargs):
        actual = _actual_operation(coroutine)
        code = getattr(actual, "cr_code", None)
        selected = code is not None and (
            (mode == "quarantine-admission" and code.co_qualname.endswith("_handoff_child_budget_release_to_quarantined_reaper.<locals>.release"))
            or (mode == "inner-admission" and code.co_qualname == "AgentManager._release_child_budget")
        )
        if selected:
            refused.append((coroutine, actual))
            raise refusal
        return previous(loop, coroutine, **kwargs) if previous else asyncio.Task(coroutine, loop=loop, **kwargs)

    def refuse_publication(*, task, **_kwargs):
        accepted.append(task)
        raise refusal

    async def run():
        async with manager._lock:
            if mode != "inner-admission":
                return manager._handoff_child_budget_release_to_quarantined_reaper(
                    "child", agent_id="did:test:refund-transfer",
                )
            record = manager._start_child_budget_release("child")
        return await manager._await_child_budget_release_cancellation_safe(record.task)

    try:
        with monkeypatch.context() as ack:
            lose_ack(ack, cancelled=False, entered=credit_entered, release=credit_release)
            loop.set_task_factory(factory)
            if mode == "quarantine-publication":
                monkeypatch.setattr(manager, "_retain_quarantined_cleanup", refuse_publication)
            owner = create_owned_task(run(), retain_originals=True)
            outcome = await await_owned_task(owner)
            assert refusal in tuple(_explicit_control_chain(outcome.error))
            assert manager._child_budgets.get("child") is entry, "refund transfer withdrew its original hold before accepted publication"
            if mode == "quarantine-publication":
                assert len(accepted) == 1 and not accepted[0].done()
                assert any(record.task is accepted[0] for record in manager._inflight_removal_budget_releases.values()), "accepted native refund owner escaped after reaper publication refusal"
                await asyncio.wait_for(credit_entered.wait(), 5)
                assert await observer.fetchval(f'SELECT count(*) FROM "{schema}".effects WHERE id=220') == 1
                credit_release.set()
                joined = await await_owned_task(accepted[0])
                assert is_execution_control_error(joined.error)
                await asyncio.sleep(0)
                assert is_execution_control_error(manager._quarantined_shutdown_control), "late native refund uncertainty escaped its original manager"
            else:
                assert refused and all(inspect.getcoroutinestate(coro) == inspect.CORO_CLOSED for pair in refused for coro in pair)
                assert any(resource is entry and task is None for resource, task in execution_cleanup_debts(outcome.error)), "refused refund admission lost original hold debt"
                assert await observer.fetchval(f'SELECT count(*) FROM "{schema}".effects WHERE id=220') == 0
    finally:
        loop.set_task_factory(previous)
        credit_release.set()
        if owner is not None:
            await await_owned_task(owner)
        for task in accepted:
            await await_owned_task(task)
        for pair in refused:
            for coroutine in pair:
                coroutine.close()
