"""Normal renewal teardown joins real PostgreSQL COMMIT acknowledgements."""
import asyncio
from dataclasses import replace
from datetime import datetime, timedelta, timezone

import asyncpg
import pytest
from kestrel_sovereign._async_ownership import await_owned_task, create_owned_task
from kestrel_sovereign.execution_custody import ExecutionCustody, ProcessRuntimeExecutionFence
from kestrel_sovereign.execution_custody import ExecutionAuthorityError
from kestrel_sovereign.features.scheduler.runner import SchedulerRunner, SCHEDULER_PROTOCOL_VERSION
from kestrel_sovereign.signals import SignalDispatcher, SignalLogStore, SourceRegistry, OrderedLockManager
from kestrel_sovereign.signals.durable import DurableSignalStore
from kestrel_sovereign.signals.sources.channels import DURABLE_COGNITION_CONSUMER_ID
from kestrel_sovereign.signals.sources.channels import build_channel_message_registration
from kestrel_sovereign.storage.async_database import AsyncDatabase
from tests.integration.test_execution_authority_postgres import native_pg as native_pg
from tests.integration.test_execution_custody_review13_pg import _seed_owned_deliveries
from tests.unit.test_durable_signal_delivery import _Agent
from tests.utils.scheduler_ticks import tick_and_settle

pytestmark = [pytest.mark.asyncio, pytest.mark.integration]


@pytest.mark.parametrize("lane", ["scheduler", "cognition"])
async def test_normal_renewal_stop_cannot_interrupt_original_native_commit(native_pg, monkeypatch, lane):
    backend, observer, schema = native_pg
    entered, release, finish = asyncio.Event(), asyncio.Event(), asyncio.Event()
    owner = scheduler = dispatcher = None
    commits = []
    matching_commits = 0
    runtime = ExecutionCustody(ProcessRuntimeExecutionFence("original"))
    native_exit = asyncpg.transaction.Transaction.__aexit__
    async def held_commit(transaction, error_type, error, traceback):
        nonlocal matching_commits
        result = await native_exit(transaction, error_type, error, traceback)
        name = asyncio.current_task().get_name()
        if error_type is None and not transaction._nested and not commits and (
            name.startswith("scheduler-lease:") if lane == "scheduler" else name.startswith("durable_cognition_lease_renewal:")):
            matching_commits += 1
            # Cognition first reads its consumer in a guarded transaction;
            # its second native COMMIT is the original lease UPDATE itself.
            if matching_commits == (1 if lane == "scheduler" else 2):
                commits.append(transaction)
                entered.set()
                await release.wait()
        return result
    try:
        if lane == "scheduler":
            async def effect(name, args):
                await finish.wait()
            scheduler = SchedulerRunner(AsyncDatabase(backend), "original", effect, lease_seconds=1)
            await scheduler._ensure_tables()
            due = (datetime.now(timezone.utc) - timedelta(seconds=2)).isoformat()
            await backend.execute("""INSERT INTO scheduled_tasks
                (id,agent_id,task_name,cron_expression,args_json,enabled,next_run_at,created_at,
                 schedule_kind,run_at,timezone_name,misfire_policy,idempotency_key,scheduler_protocol_version)
                VALUES ('original','original','offline','* * * * *','{}',1,?,?,'one_shot',?,'UTC','skip','native-renewal',?)""",
                (due, due, due, SCHEDULER_PROTOCOL_VERSION))
            backend._execution_custody = runtime
            monkeypatch.setattr(asyncpg.transaction.Transaction, "__aexit__", held_commit)
            owner = create_owned_task(tick_and_settle(scheduler))
        else:
            await _seed_owned_deliveries(backend, observer, schema)
            signal_store = SignalLogStore(backend)
            await signal_store.initialize()
            registry = SourceRegistry()
            registry.register(build_channel_message_registration())
            dispatcher = SignalDispatcher(agent=_Agent("original"), registry=registry,
                lock_manager=OrderedLockManager(), store=signal_store)
            dispatcher._durable_delivery_owner = "dispatcher:original"
            await dispatcher.initialize_durable_delivery()
            await dispatcher._stop_runtime_owner_heartbeat()
            # The shared retirement fixture deliberately leaves expiry NULL.
            # This healthy renewal scenario needs the original token's live
            # persisted lease, not just an altered timer on its local snapshot.
            await observer.execute("UPDATE durable_signal_deliveries SET lease_expires_at=clock_timestamp()+interval '30 minutes' WHERE delivery_id='delivery1'")
            delivery = await DurableSignalStore(backend).get_delivery(agent_id="original",
                consumer_id=DURABLE_COGNITION_CONSUMER_ID, delivery_id="delivery1")
            assert delivery.status == "leased" and delivery.lease_token is not None, delivery
            delivery = replace(delivery, lease_expires_at=datetime.now(timezone.utc)+timedelta(seconds=.09))
            backend._execution_custody = runtime
            monkeypatch.setattr(asyncpg.transaction.Transaction, "__aexit__", held_commit)
            async def turn():
                async with dispatcher._renew_durable_cognition_lease(delivery) as lost:
                    await finish.wait()
                assert not lost.done(), (lost.result() if lost.done() else None,
                    await observer.fetchrow("SELECT status,lease_token,lease_expires_at FROM durable_signal_deliveries WHERE delivery_id='delivery1'"),
                    await observer.fetchrow("SELECT active,lease_seconds FROM durable_signal_consumers WHERE agent_id='original' AND consumer_id=$1", DURABLE_COGNITION_CONSUMER_ID))
            owner = create_owned_task(turn())
        await asyncio.wait_for(entered.wait(), 3)
        finish.set()
        done, _ = await asyncio.wait({owner}, timeout=.05)
        assert not done, "normal completion abandoned its original native renewal COMMIT"
        release.set()
        assert (await await_owned_task(owner)).error is None
        runtime.require_work()
        assert len(commits) == 1
        if scheduler is not None:
            assert scheduler._execution_control_error is None
        await backend.execute("CREATE TABLE healthy_after_renewal(id INTEGER)")
    finally:
        finish.set()
        release.set()
        if owner is not None:
            await await_owned_task(owner)
        monkeypatch.undo()
        backend._execution_custody = None
        if scheduler is not None:
            await scheduler.stop()
        if dispatcher is not None:
            await dispatcher.shutdown_durable_delivery()


async def test_concurrent_native_cognition_terminalizers_share_original_success_receipt(native_pg, monkeypatch):
    backend, observer, schema = native_pg
    await _seed_owned_deliveries(backend, observer, schema)
    signal_store = SignalLogStore(backend)
    await signal_store.initialize()
    dispatcher = SignalDispatcher(agent=_Agent("original"), registry=SourceRegistry(),
        lock_manager=OrderedLockManager(), store=signal_store)
    dispatcher._durable_delivery_owner = "dispatcher:original"
    await dispatcher.initialize_durable_delivery()
    await dispatcher._stop_runtime_owner_heartbeat()
    delivery = await DurableSignalStore(backend).get_delivery(agent_id="original",
        consumer_id=DURABLE_COGNITION_CONSUMER_ID, delivery_id="delivery1")
    runtime = ExecutionCustody(ProcessRuntimeExecutionFence("original"))
    runtime.revoke("original runtime stopped")
    backend._execution_custody = runtime
    error = ExecutionAuthorityError("original cognition control receipt")
    entered, release = asyncio.Event(), asyncio.Event()
    native_exit = asyncpg.transaction.Transaction.__aexit__
    commits = []
    async def held_commit(transaction, error_type, failure, traceback):
        result = await native_exit(transaction, error_type, failure, traceback)
        if error_type is None and not transaction._nested and asyncio.current_task().get_name().startswith("durable_cognition_control_terminal:"):
            commits.append(transaction)
            entered.set()
            await release.wait()
        return result
    monkeypatch.setattr(asyncpg.transaction.Transaction, "__aexit__", held_commit)
    callers = []
    try:
        callers.append(create_owned_task(dispatcher._terminalize_failed_cognition(delivery, error)))
        await asyncio.wait_for(entered.wait(), 2)
        callers.append(create_owned_task(dispatcher._terminalize_failed_cognition(delivery, error)))
        done, _ = await asyncio.wait(set(callers), timeout=.05)
        assert not done, "a concurrent terminalizer bypassed the original physical commit owner"
        release.set()
        for caller in callers:
            assert (await await_owned_task(caller)).error is None
        # A later retained-route callback joins the successful SAME token
        # receipt, not another UPDATE against an already-cleared token.
        await dispatcher._terminalize_failed_cognition(delivery, error)
        assert len(commits) == 1 and not dispatcher._retained_cognition_control_debt
        row = await observer.fetchrow("SELECT status,lease_owner,lease_token FROM durable_signal_deliveries WHERE delivery_id='delivery1'")
        assert tuple(row) == ("failed", None, None)
        with pytest.raises(ExecutionAuthorityError):
            await backend.execute("CREATE TABLE forbidden_fresh_work(id INTEGER)")
    finally:
        release.set()
        for caller in callers:
            await await_owned_task(caller)
        monkeypatch.undo()
        backend._execution_custody = None
        await dispatcher._stop_runtime_owner_heartbeat()
        await dispatcher.shutdown_durable_delivery()
