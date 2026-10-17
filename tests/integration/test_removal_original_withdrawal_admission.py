"""Actual registered removal preserves earlier withdrawal at later admission."""
import asyncio
from decimal import Decimal

import pytest

from kestrel_sovereign._async_ownership import await_owned_task, create_owned_task
from kestrel_sovereign.execution_custody import _explicit_control_chain, execution_cleanup_debts
from kestrel_sovereign.multi_agent.agent_manager import AgentManager
from kestrel_sovereign.spawn.delegated_wallet import BudgetAllocation, DelegatedWallet
from tests.integration.test_budget_refund_original_control import registered_cleanup_component
from tests.integration.test_execution_authority_postgres import native_pg as native_pg
from tests.unit.test_host_context_native_cleanup import _actual_operation


@pytest.mark.asyncio
async def test_public_registered_removal_retains_withdrawal_before_refund_refusal(native_pg, tmp_path, monkeypatch):
    backend, observer, schema = native_pg
    manager = AgentManager(base_data_dir=tmp_path)
    child, database = await registered_cleanup_component(manager, schema, tmp_path, monkeypatch)
    entered, release = asyncio.Event(), asyncio.Event()
    actual_close, actual_wait = child._raw_storage.close, asyncio.wait_for
    withdrawals, refused = [], []
    owner = None
    refusal = RuntimeError("actual quarantined refund Task admission refused after caller withdrawal")

    class Parent:
        _balances = {"FIL": {"main": Decimal("0")}}

        async def deposit(self, *args, **kwargs):
            await backend.execute("INSERT INTO effects VALUES (223, 'unaccepted refund')")
            return True

    parent = Parent()
    delegated = DelegatedWallet(parent, BudgetAllocation(child_did=child.did, amount=Decimal("10")))
    entry = (delegated, parent)
    manager._child_budgets["child"] = entry
    loop, previous = asyncio.get_running_loop(), asyncio.get_running_loop().get_task_factory()

    async def held_close():
        entered.set()
        await release.wait()
        return await actual_close()

    async def wait(awaitable, *args, **kwargs):
        try:
            return await actual_wait(awaitable, *args, **kwargs)
        except asyncio.CancelledError as cancellation:
            if asyncio.current_task() is owner:
                withdrawals.append(cancellation)
            raise

    def factory(loop, coroutine, **kwargs):
        operation = _actual_operation(coroutine)
        code = getattr(operation, "cr_code", None)
        if code is not None and code.co_qualname.endswith("_handoff_child_budget_release_to_quarantined_reaper.<locals>.release"):
            refused.append((coroutine, operation))
            raise refusal
        return previous(loop, coroutine, **kwargs) if previous else asyncio.Task(coroutine, loop=loop, **kwargs)

    try:
        monkeypatch.setattr(child._raw_storage, "close", held_close)
        monkeypatch.setattr(asyncio, "wait_for", wait)
        loop.set_task_factory(factory)
        owner = create_owned_task(manager.remove_agent("child", offboard_runtime=False), retain_originals=True)
        await actual_wait(entered.wait(), 5)
        owner.cancel("actual registered removal caller withdrew")
        outcome = await await_owned_task(owner)
        chain = tuple(_explicit_control_chain(outcome.error))
        assert refusal in chain and refused
        assert len(withdrawals) == 1 and withdrawals[0] in chain, "registered removal discarded its actual earlier caller cancellation"
        assert withdrawals[0] not in tuple(_explicit_control_chain(manager._quarantined_shutdown_control)), "private withdrawal contaminated shared refund source"
        assert manager._child_budgets["child"] is entry
        assert any(resource is entry and task is None for resource, task in execution_cleanup_debts(outcome.error))
        assert manager.get_agent("child") is None and not delegated.can_spend(Decimal("1"))
        assert await observer.fetchval(f'SELECT count(*) FROM "{schema}".effects WHERE id=223') == 0
    finally:
        loop.set_task_factory(previous)
        release.set()
        if owner is not None:
            await await_owned_task(owner)
        for record in tuple(manager._quarantined_shutdown_reapers.values()):
            await await_owned_task(record.task)
        await await_owned_task(create_owned_task(child.wait_for_shutdown_completion(), retain_originals=True))
        await actual_close()
        await database.close()
        for pair in refused:
            for coroutine in pair:
                coroutine.close()
