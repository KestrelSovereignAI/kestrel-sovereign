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


async def registered_cleanup_component(manager, schema, tmp_path, monkeypatch):
    """Actual constructor/storage component; never born/initialized/READY."""
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
    monkeypatch.setattr("kestrel_sovereign.entrypoints.discover_entry_point_classes", lambda *_args, **_kwargs: {})
    backend = PostgresBackend(dsn=with_search_path(os.environ["TEST_POSTGRES_URL"], schema),
                              min_pool_size=0, max_pool_size=2, advisory_max_pool_size=1)
    await backend.connect()
    db = AsyncDatabase(backend)
    service = LLMService(usage_db=db)
    assert not service.providers
    child = KestrelAgent(did="did:test:refund-component", storage_path=str(tmp_path / "unborn.db"), llm_service=service)
    storage = AsyncStorage(backend=backend)
    storage.db = db
    child.storage = child._raw_storage = storage
    assert not child._runtime_publication_ready.is_set()
    manager._agents["child"] = child
    manager._agent_names[child.did] = "child"
    return child, db


@pytest.mark.asyncio
@pytest.mark.parametrize("route", ["join", "completed", "remove", "registered-component", "drain", "drain-live", "drain-waiter", "drain-quarantine", "fleet-waiter", "drain-quarantine-unseal-refusal", "remove-reconcile", "registered-component-reconcile", "registered-component-completion-observer"])
@pytest.mark.parametrize("source_cancel", [False, True])
@pytest.mark.parametrize("caller_cancel", [False, True])
async def test_original_native_refund_survives_join_callback_and_terminal_drain(
    native_pg, tmp_path, monkeypatch, route, source_cancel, caller_cancel,
):
    backend, observer, schema = native_pg
    removal_route = "registered-component" if route.startswith("registered-component-") else route.removesuffix("-reconcile")
    reconcile_route = route.endswith("-reconcile")
    earlier_observer_route = route.endswith("-observer")
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
    reconcile_join = asyncio.Event()
    completion_entered, completion_release = asyncio.Event(), asyncio.Event()
    reconcile_lock_owned = False
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
    if drain_route in {"drain-live", "drain-waiter", "drain-quarantine"} or reconcile_route or earlier_observer_route:
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

        async def observed_owned_join(child, *args, **kwargs):
            selected = bool(records) and child is records[0].task
            if selected:
                if drain_route != "drain-waiter" or asyncio.current_task() in competing_drains:
                    assert not child.done()
                    drain_join.set()
            outcome = await actual_owned_join(child, *args, **kwargs)
            if outcome.cancellation is not None:
                withdrawals.append(outcome.cancellation)
            return outcome

        async def observed_legacy_join(child, *args, **kwargs):
            if reconcile_route and child.get_name() == "agent_manager:reconcile_removed_child_tracking":
                assert manager._lock.locked() and not child.done()
                reconcile_join.set()
            if records and child is records[0].task:
                assert not child.done()
                drain_join.set()
            return await actual_legacy_join(child, *args, **kwargs)

        monkeypatch.setattr(manager_module, "await_owned_task", observed_owned_join)
        monkeypatch.setattr(ownership_module, "await_owned_task", observed_owned_join)
        monkeypatch.setattr(manager_module, "await_lifecycle_task_completion", observed_legacy_join)
        monkeypatch.setattr(manager, "_acquire_quarantined_shutdown_drain", observed_acquire)
    child = child_db = None
    if removal_route == "registered-component":
        child, child_db = await registered_cleanup_component(manager, schema, tmp_path, monkeypatch)

    task = None
    if earlier_observer_route:
        shutdown_tasks = []

        def observe_shutdown(loop, coroutine, **kwargs):
            code = getattr(coroutine, "cr_code", None)
            child_task = previous_factory(loop, coroutine, **kwargs) if previous_factory else asyncio.Task(coroutine, loop=loop, **kwargs)
            if code is not None and code.co_qualname == "KestrelAgent.shutdown":
                shutdown_tasks.append(child_task)
            return child_task

        loop.set_task_factory(observe_shutdown)
        actual_completion = child.wait_for_shutdown_completion

        async def held_completion():
            if asyncio.current_task() not in shutdown_tasks:
                # Delay only the manager's separate completion observer, not
                # the real shutdown's own internal native completion join.
                completion_entered.set()
                await completion_release.wait()
            return await actual_completion()

        monkeypatch.setattr(child, "wait_for_shutdown_completion", held_completion)

    async def consume():
        with bind_execution_custody(GenerationFence()):
            if removal_route in {"remove", "registered-component"}:
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

    try:
        with monkeypatch.context() as ack:
            lose_ack(ack, cancelled=False, entered=entered, release=release)
            task = create_owned_task(consume(), retain_originals=True)
            if earlier_observer_route:
                await asyncio.wait_for(completion_entered.wait(), 5)
                if route == "registered-component-completion-observer" and caller_cancel:
                    task.cancel("actual earlier removal observer withdrawal")
                completion_release.set()
            await asyncio.wait_for(entered.wait(), 5)
            assert await observer.fetchval(f'SELECT count(*) FROM "{schema}".effects WHERE id=184') == 1
            if removal_route in {"remove", "registered-component"}:
                records.append(manager._inflight_removal_budget_releases_by_child["child"])
            if drain_route in {"drain-live", "drain-waiter", "drain-quarantine"}:
                await asyncio.wait_for((drain_waiter if drain_route == "drain-waiter" else drain_join).wait(), 5)
            if source_cancel:
                deposit_owners[0].cancel("actual original refund COMMIT observer withdrawal")
            if reconcile_route:
                await manager._lock.acquire()
                reconcile_lock_owned = True
                release.set()
                await asyncio.wait_for(reconcile_join.wait(), 5)
            if caller_cancel and not earlier_observer_route:
                task.cancel("actual refund caller withdrawal")
            release.set()
            if reconcile_lock_owned:
                manager._lock.release()
                reconcile_lock_owned = False
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
        if (drain_route in {"drain-live", "drain-waiter", "drain-quarantine"} or reconcile_route or earlier_observer_route) and caller_cancel:
            chain = tuple(_explicit_control_chain(outcome.error))
            message = "actual earlier removal observer withdrawal" if earlier_observer_route else "actual refund caller withdrawal"
            assert any(isinstance(error, asyncio.CancelledError) and error.args == (message,)
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
        completion_release.set()
        if reconcile_lock_owned:
            manager._lock.release()
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


@pytest.mark.asyncio
async def test_real_shutdown_observer_withdrawal_is_not_reconstructed(native_pg, tmp_path, monkeypatch):
    _backend, observer, schema = native_pg
    manager = AgentManager(base_data_dir=tmp_path)
    child, db = await registered_cleanup_component(manager, schema, tmp_path, monkeypatch)
    actual_wait_for = asyncio.wait_for
    withdrawals = []
    entered, release = asyncio.Event(), asyncio.Event()
    actual_close = db.close
    owner = None

    async def close():
        entered.set()
        try:
            await release.wait()
        finally:
            await actual_close()

    async def wait_for(awaitable, *args, **kwargs):
        try:
            return await actual_wait_for(awaitable, *args, **kwargs)
        except asyncio.CancelledError as cancellation:
            if asyncio.current_task() is owner:
                withdrawals.append(cancellation)
            raise

    try:
        monkeypatch.setattr(asyncio, "wait_for", wait_for)
        monkeypatch.setattr(db, "close", close)
        owner = create_owned_task(manager.remove_agent("child", offboard_runtime=False), retain_originals=True)
        await asyncio.wait_for(entered.wait(), 5)
        assert not owner.done() and db.backend.is_connected
        owner.cancel("actual original shutdown wait_for observer withdrawal")
        release.set()
        outcome = await await_owned_task(owner)
        assert len(withdrawals) == 1 and outcome.error is withdrawals[0], "removal replaced the actual shutdown observer withdrawal"
        await manager.drain_quarantined_shutdowns()
        assert not db.backend.is_connected and manager.get_agent("child") is None
        assert await observer.fetchval(f'SELECT count(*) FROM "{schema}".effects') == 0
    finally:
        release.set()
        if owner is not None:
            await await_owned_task(owner)
        await manager.drain_quarantined_shutdowns()
        try:
            # Rejoining an already-cancelled shutdown replays its retained
            # control. Collect that terminal result without bypassing the
            # independent fixture database retirement below.
            await await_owned_task(create_owned_task(
                child.shutdown(), retain_originals=True,
            ))
        finally:
            await db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("caller_cancel", [False, True])
async def test_completion_join_keeps_prior_native_receipt_and_original_close_failure(
    native_pg, tmp_path, monkeypatch, caller_cancel,
):
    from kestrel_sovereign import _async_ownership as ownership
    from kestrel_sovereign.execution_custody import execution_cleanup_debts
    from kestrel_sovereign.kestrel_agent import await_agent_shutdown_completion

    backend, observer, schema = native_pg
    manager = AgentManager(base_data_dir=tmp_path)
    child, database = await registered_cleanup_component(manager, schema, tmp_path, monkeypatch)
    entered, release = asyncio.Event(), asyncio.Event()

    async def effect():
        with bind_execution_custody(GenerationFence()):
            await backend.execute("INSERT INTO effects VALUES (195, 'prior completion receipt')")

    with monkeypatch.context() as ack:
        lose_ack(ack, cancelled=False, entered=entered, release=release)
        source = create_owned_task(effect(), retain_originals=True)
        try:
            await asyncio.wait_for(entered.wait(), 5)
            release.set()
            original = (await await_owned_task(source)).error
        finally:
            release.set()
            await await_owned_task(source)
    assert original is not None and execution_commit_outcome(original) == "unknown"
    graph, debts = tuple(_explicit_control_chain(original)), tuple(execution_cleanup_debts(original))
    actual_wait, actual_close = child.wait_for_shutdown_completion, database.backend.close
    actual_join = ownership.await_owned_task
    close_entered, close_release = asyncio.Event(), asyncio.Event()
    close_ack = OSError("actual completed native close acknowledgement lost")
    withdrawals, owner = [], None

    async def close():
        close_entered.set()
        await close_release.wait()
        await actual_close()
        raise close_ack

    async def completion():
        # Transparent completion component: real agent waiter, then its real
        # database closer. No Agent initialization/READY is manufactured.
        await actual_wait()
        await database.close()

    async def join(task, *args, **kwargs):
        outcome = await actual_join(task, *args, **kwargs)
        if asyncio.current_task() is owner and outcome.cancellation is not None:
            withdrawals.append(outcome.cancellation)
        return outcome

    try:
        monkeypatch.setattr(database.backend, "close", close)
        monkeypatch.setattr(child, "wait_for_shutdown_completion", completion)
        monkeypatch.setattr(ownership, "await_owned_task", join)
        owner = create_owned_task(await_agent_shutdown_completion(child, prior_error=original), retain_originals=True)
        await asyncio.wait_for(close_entered.wait(), 5)
        if caller_cancel:
            owner.cancel("actual completion observer withdrawal beside prior receipt")
        close_release.set()
        outcome = await actual_join(owner)
        chain = tuple(_explicit_control_chain(outcome.error))
        assert original in chain and close_ack in chain, "completion join dropped an independent original close failure"
        assert execution_commit_outcome(outcome.error) == "unknown"
        assert tuple(_explicit_control_chain(original)) == graph
        assert tuple(execution_cleanup_debts(original)) == debts, "completion join mutated shared native receipt debt"
        assert bool(withdrawals) is caller_cancel and all(error in chain for error in withdrawals)
        assert not database.backend.is_connected
        assert await observer.fetchval(f'SELECT count(*) FROM "{schema}".effects WHERE id=195') == 1
    finally:
        close_release.set()
        if owner is not None:
            await actual_join(owner)
        monkeypatch.setattr(child, "wait_for_shutdown_completion", actual_wait)
        monkeypatch.setattr(database.backend, "close", actual_close)
        await await_owned_task(create_owned_task(child.shutdown(), retain_originals=True))
        await actual_close()


@pytest.mark.asyncio
async def test_bounded_native_offboarding_preserves_its_actual_observer(tmp_path, monkeypatch):
    import threading
    from kestrel_sovereign.features.isolated_runtime import RuntimeNamespaceCleanupOutcome
    from kestrel_sovereign.multi_agent.agent_manager import InflightRuntimeOffboarding, RuntimeOffboardingRetainedError
    from kestrel_sovereign.endpoints.models import _grouped_runtime_retained_detail

    manager = AgentManager(base_data_dir=tmp_path)
    artifact = tmp_path / "fixture-owned-offboarding"
    artifact.write_text("fixture-owned cleanup", encoding="utf-8")
    entered, release = threading.Event(), threading.Event()
    actual_wait = asyncio.wait
    withdrawals = []
    withdrawal_graphs = []
    observing = asyncio.Event()
    owner = None

    def remove():
        entered.set()
        assert release.wait(5)
        artifact.unlink()
        return RuntimeNamespaceCleanupOutcome.REMOVED

    async def wait(*args, **kwargs):
        if asyncio.current_task() is owner:
            observing.set()
        try:
            return await actual_wait(*args, **kwargs)
        except asyncio.CancelledError as cancellation:
            if asyncio.current_task() is owner:
                withdrawals.append(cancellation)
                withdrawal_graphs.append(tuple(_explicit_control_chain(cancellation)))
            raise

    worker = create_owned_task(asyncio.to_thread(remove), retain_originals=True)
    record = InflightRuntimeOffboarding("fixture-child", "did:test:fixture-offboarding", artifact, worker)
    try:
        monkeypatch.setattr(asyncio, "wait", wait)
        owner = create_owned_task(manager._finish_agent_runtime_offboarding(record, cancellation_already_observed=False), retain_originals=True)
        async with asyncio.timeout(5):
            while not entered.is_set():
                await asyncio.sleep(0.005)
        await asyncio.wait_for(observing.wait(), 5)
        owner.cancel("actual bounded offboarding observer withdrawal")
        outcome = await await_owned_task(owner)
        assert outcome.error is None
        cancelled, failure = outcome.result
        chain = tuple(_explicit_control_chain(failure))
        pending = [error for error in chain if isinstance(error, RuntimeOffboardingRetainedError)]
        assert cancelled and len(pending) == 1 and pending[0].cleanup_pending
        assert not worker.done() and artifact.exists()
        assert len(withdrawals) == 1 and withdrawals[0] in chain, "offboarding metadata discarded actual observer withdrawal"
        assert tuple(_explicit_control_chain(withdrawals[0])) == withdrawal_graphs[0], "composition mutated the actual observer"
        detail = _grouped_runtime_retained_detail(failure)
        assert detail["runtime_cleanup_pending"] is True and detail["runtime_cleanup_state"] == "pending"
        assert detail["retained_outcome_count"] == 1 and detail["additional_outcome_count"] == 1
    finally:
        release.set()
        if owner is not None:
            await await_owned_task(owner)
        await await_owned_task(worker)
        await manager.drain_quarantined_shutdowns()
