"""Actual delegated refund custody over a disposable PostgreSQL protocol ledger.

The ledger is a local wallet-protocol peer, not a production wallet provider.
No agent birth, funds, provider request or application READY is manufactured.
"""

import asyncio
import os
from decimal import Decimal

import pytest

from kestrel_sovereign._async_ownership import await_owned_task, create_owned_task
from kestrel_sovereign.execution_custody import (
    _explicit_control_chain, bind_execution_custody,
    execution_commit_outcome, is_execution_control_error,
)
from kestrel_sovereign.multi_agent.agent_manager import AgentManager
from kestrel_sovereign.spawn.delegated_wallet import BudgetAllocation, DelegatedWallet
from tests.integration.test_execution_authority_postgres import GenerationFence, native_pg as native_pg
from tests.integration.test_execution_custody_review30_pg import lose_ack
from tests.unit.test_execution_custody_review35_native import traceback_size


@pytest.mark.asyncio
@pytest.mark.parametrize("route", ["join", "completed", "remove", "registered-component", "drain", "drain-live", "drain-waiter", "drain-quarantine", "fleet-waiter", "drain-quarantine-unseal-refusal"])
@pytest.mark.parametrize("source_cancel", [False, True])
@pytest.mark.parametrize("caller_cancel", [False, True])
async def test_original_native_refund_survives_join_callback_and_terminal_drain(
    native_pg, tmp_path, monkeypatch, route, source_cancel, caller_cancel,
):
    backend, observer, schema = native_pg
    drain_route = ("drain-waiter" if route == "fleet-waiter" else
                   "drain-quarantine" if route == "drain-quarantine-unseal-refusal" else route)
    entered, release = asyncio.Event(), asyncio.Event()
    originals, graphs, deposit_owners = [], [], []

    class LocalPostgresWalletLedger:
        _balances = {"FIL": {"main": Decimal("0")}}

        async def deposit(self, amount, currency=None, to_audit=False, memo=""):
            assert amount == Decimal("10") and currency == "FIL" and not to_audit
            deposit_owners.append(asyncio.current_task())
            try:
                await backend.execute("INSERT INTO effects VALUES (184, 'original refund credit')")
            except BaseException as error:
                if source_cancel:
                    # Preserve a genuine Task cancellation as the refund's
                    # outer carrier, with its actual native ACK failure. This
                    # exercises cancelled-Task harvesting, not just an ordinary
                    # QueryError which happens to contain a cancellation below.
                    asyncio.current_task().cancel("actual original refund withdrawal after native receipt")
                    try:
                        await asyncio.sleep(0)
                    except asyncio.CancelledError as withdrawal:
                        withdrawal.__cause__ = error
                        error = withdrawal
                originals.append(error)
                graphs.append(tuple(_explicit_control_chain(error)))
                raise error
            return True

    parent = LocalPostgresWalletLedger()
    delegated = DelegatedWallet(parent, BudgetAllocation(child_did="did:test:unpublished-child", amount=Decimal("10")))
    manager = AgentManager(base_data_dir=tmp_path)
    entry = (delegated, parent)
    manager._child_budgets["child"] = entry
    records = []
    drain_join = asyncio.Event()
    drain_waiter = asyncio.Event()
    competing_drains = []
    withdrawals = []
    loop, previous_factory = asyncio.get_running_loop(), asyncio.get_running_loop().get_task_factory()
    refused = []
    refusal = RuntimeError("actual final handoff-unseal Task admission refused")
    if route == "drain-quarantine-unseal-refusal":
        from tests.unit.test_host_context_native_cleanup import _actual_operation

        def refuse_unseal(loop, coroutine, **kwargs):
            actual = _actual_operation(coroutine)
            code, frame = getattr(actual, "cr_code", None), getattr(actual, "cr_frame", None)
            if (not refused and manager._quarantined_shutdown_handoffs_sealed and code is not None
                    and code.co_qualname == "Lock.acquire" and frame.f_locals.get("self") is manager._lock):
                refused.append((coroutine, actual))
                raise refusal
            return previous_factory(loop, coroutine, **kwargs) if previous_factory else asyncio.Task(coroutine, loop=loop, **kwargs)

        loop.set_task_factory(refuse_unseal)
    if drain_route in {"drain-live", "drain-waiter", "drain-quarantine"}:
        from kestrel_sovereign.multi_agent import agent_manager as manager_module
        from kestrel_sovereign import _async_ownership as ownership_module

        actual_owned_join = manager_module.await_owned_task
        actual_legacy_join = manager_module.await_lifecycle_task_completion
        actual_acquire = manager._acquire_quarantined_shutdown_drain
        acquisitions = []

        async def observed_acquire(*args, **kwargs):
            acquisitions.append(asyncio.current_task())
            if drain_route == "drain-waiter" and len(acquisitions) == 2:
                assert manager._quarantined_shutdown_drain_lock.locked()
                drain_waiter.set()
            return await actual_acquire(*args, **kwargs)

        async def observed_owned_join(child):
            selected = bool(records) and child is records[0].task
            if selected:
                if drain_route != "drain-waiter" or asyncio.current_task() in competing_drains:
                    assert not child.done()
                    drain_join.set()
            outcome = await actual_owned_join(child)
            if outcome.cancellation is not None:
                withdrawals.append(outcome.cancellation)
            return outcome

        async def observed_legacy_join(child, *args, **kwargs):
            if records and child is records[0].task:
                assert not child.done()
                drain_join.set()
            return await actual_legacy_join(child, *args, **kwargs)

        monkeypatch.setattr(manager_module, "await_owned_task", observed_owned_join)
        monkeypatch.setattr(ownership_module, "await_owned_task", observed_owned_join)
        monkeypatch.setattr(manager_module, "await_lifecycle_task_completion", observed_legacy_join)
        monkeypatch.setattr(manager, "_acquire_quarantined_shutdown_drain", observed_acquire)
    child = child_db = None
    if route == "registered-component":
        from kestrel_sovereign.kestrel_agent import KestrelAgent
        from kestrel_sovereign.llm.service import LLMService
        from kestrel_sovereign.storage.async_database import AsyncDatabase
        from kestrel_sovereign.storage.async_storage import AsyncStorage
        from kestrel_sovereign.storage.db.postgres import PostgresBackend
        from tests.utils.postgres_schema import with_search_path

        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("PROJECT_DIR", str(tmp_path))
        monkeypatch.setenv("KESTREL_HOME", str(tmp_path / "home"))
        (tmp_path / "kestrel.toml").write_text('[llm]\nroute_priority = []\n[llm.vendors]\n', encoding="utf-8")
        # A declared empty installation inventory, not a substituted provider.
        monkeypatch.setattr("kestrel_sovereign.entrypoints.discover_entry_point_classes", lambda *_args, **_kwargs: {})
        child_backend = PostgresBackend(dsn=with_search_path(os.environ["TEST_POSTGRES_URL"], schema),
                                        min_pool_size=0, max_pool_size=2, advisory_max_pool_size=1)
        await child_backend.connect()
        child_db = AsyncDatabase(child_backend)
        service = LLMService(usage_db=child_db)
        assert not service.providers
        child = KestrelAgent(did="did:test:refund-component", storage_path=str(tmp_path / "unborn.db"), llm_service=service)
        storage = AsyncStorage(backend=child_backend)
        storage.db = child_db
        child.storage = child._raw_storage = storage
        # Inject only this cleanup component's map slot. It was NOT incepted,
        # initialized, authenticated or marked READY by this fixture.
        assert not child._runtime_publication_ready.is_set()
        manager._agents["child"] = child
        manager._agent_names[child.did] = "child"

    async def consume():
        with bind_execution_custody(GenerationFence()):
            if route in {"remove", "registered-component"}:
                return await manager.remove_agent("child", offboard_runtime=False)
            record = manager._start_child_budget_release("child")
            records.append(record)
            if drain_route == "drain-quarantine":
                manager._retain_quarantined_cleanup(name="child", agent_id="did:test:unpublished-child", task=record.task)
            if drain_route == "drain-waiter":
                competing_drains.append(create_owned_task(manager.drain_quarantined_shutdowns(), retain_originals=True))
                await drain_join.wait()
            if route in {"completed", "drain"}:
                await await_owned_task(record.task)
                await asyncio.sleep(0)  # Actual completion callback retires metadata.
            if route == "fleet-waiter":
                return await manager.shutdown_all()
            if route == "drain" or drain_route in {"drain-live", "drain-waiter", "drain-quarantine"}:
                return await manager.drain_quarantined_shutdowns()
            return await manager._await_child_budget_release_cancellation_safe(record.task)

    task = None
    try:
        with monkeypatch.context() as ack:
            lose_ack(ack, cancelled=False, entered=entered, release=release)
            task = create_owned_task(consume(), retain_originals=True)
            await asyncio.wait_for(entered.wait(), 5)
            assert await observer.fetchval(f'SELECT count(*) FROM "{schema}".effects WHERE id=184') == 1
            if route in {"remove", "registered-component"}:
                records.append(manager._inflight_removal_budget_releases_by_child["child"])
            if drain_route in {"drain-live", "drain-waiter", "drain-quarantine"}:
                await asyncio.wait_for((drain_waiter if drain_route == "drain-waiter" else drain_join).wait(), 5)
            if source_cancel:
                deposit_owners[0].cancel("actual original refund COMMIT observer withdrawal")
            if caller_cancel:
                task.cancel("actual refund caller withdrawal")
            release.set()
            outcome = await await_owned_task(task)
        assert len(originals) == 1 and execution_commit_outcome(originals[0]) == "unknown"
        if source_cancel:
            assert isinstance(originals[0], asyncio.CancelledError) and records[0].task.cancelled()
        assert outcome.error is not None and is_execution_control_error(outcome.error), "refund completion lost original native execution control"
        assert execution_commit_outcome(outcome.error) == "unknown", "refund completion erased original COMMIT uncertainty"
        assert originals[0] in tuple(_explicit_control_chain(outcome.error)), "refund completion discarded its original native carrier"
        if route == "drain-quarantine-unseal-refusal":
            import inspect
            assert refused and refusal in tuple(_explicit_control_chain(outcome.error))
            assert all(inspect.getcoroutinestate(coro) == inspect.CORO_CLOSED for pair in refused for coro in pair)
        if drain_route in {"drain-live", "drain-waiter", "drain-quarantine"} and caller_cancel:
            chain = tuple(_explicit_control_chain(outcome.error))
            assert any(isinstance(error, asyncio.CancelledError) and error.args == ("actual refund caller withdrawal",)
                       for error in chain), "live terminal refund drain lost its actual caller withdrawal"
            assert len(withdrawals) == 1 and withdrawals[0] in chain
            assert withdrawals[0] not in tuple(_explicit_control_chain(manager._quarantined_shutdown_control)), "finite drain withdrawal leaked into shared source latch"
        assert tuple(_explicit_control_chain(originals[0])) == graphs[0], "refund completion mutated its original native source graph"
        assert records[0].task.done() and manager._child_budgets["child"] is entry
        if child_db is not None:
            assert not child_db.backend.is_connected and not child._runtime_publication_ready.is_set()
        assert delegated._refund_attempted and not delegated._refund_completed
        assert not manager._inflight_removal_budget_releases
        assert manager.unsafe_removal_budget_release_failures()
        original_depth = traceback_size(originals[0])
        for attempt in range(20):
            if attempt == 1:
                for release_id in manager.unsafe_removal_budget_release_failures():
                    assert manager.acknowledge_unsafe_removal_budget_release_failure(release_id)
            caught = None
            try:
                await manager.drain_quarantined_shutdowns()
            except BaseException as error:
                caught = error
            assert is_execution_control_error(caught), "completed refund callback reduced native control to metadata-only failure"
            assert execution_commit_outcome(caught) == "unknown"
            assert originals[0] in tuple(_explicit_control_chain(caught))
            assert all(withdrawal not in tuple(_explicit_control_chain(caught)) for withdrawal in withdrawals), "later drain inherited an earlier caller's private withdrawal"
            assert tuple(_explicit_control_chain(originals[0])) == graphs[0]
            assert traceback_size(originals[0]) <= original_depth + 4, "refund receipt retained unbounded terminal-reader traceback frames"
        # Acknowledging public metadata cannot settle UNKNOWN or authorize a
        # duplicate legacy credit. The original delegated guard refuses it.
        repeated = manager._start_child_budget_release("child")
        records.append(repeated)
        caught = None
        try:
            await manager._await_child_budget_release_cancellation_safe(repeated.task)
        except BaseException as error:
            caught = error
        assert is_execution_control_error(caught) and execution_commit_outcome(caught) == "unknown", "repeated refund lost original unresolved native control"
        assert originals[0] in tuple(_explicit_control_chain(caught))
        assert len(deposit_owners) == 1
        assert await observer.fetchval(f'SELECT count(*) FROM "{schema}".effects WHERE id=184') == 1
    finally:
        loop.set_task_factory(previous_factory)
        for pair in refused:
            for coroutine in pair:
                coroutine.close()
        release.set()
        if task is not None:
            await await_owned_task(task)
        for drainer in competing_drains:
            await await_owned_task(drainer)
        for record in records:
            await await_owned_task(record.task)
        if child is not None:
            await child.shutdown()
        if child_db is not None:
            await child_db.close()
