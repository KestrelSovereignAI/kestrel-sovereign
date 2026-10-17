"""Restart ACK component ownership with actual native PostgreSQL effects.

Actual Core handles and real Agent task publication are used; the dispatch-body
fault fixture runs SQL, not cognition. The constructor component is never born,
initialized as an Agent, or published READY. No host restart or provider call.
"""
import asyncio

import pytest
from kestrel_sdk.signals import Signal, SignalMode, SignalResult, Status

from kestrel_sovereign._async_ownership import await_owned_task, create_owned_task
from kestrel_sovereign.execution_custody import (
    _explicit_control_chain, bind_execution_custody, execution_cleanup_debts,
    execution_commit_outcome, is_execution_control_error,
)
from kestrel_sovereign.features.restart_coordinator.feature import RestartCoordinatorFeature
from kestrel_sovereign.features.restart_coordinator.store import (
    ensure_restart_requests_table, get_request, insert_request, update_status,
)
from kestrel_sovereign.multi_agent.agent_manager import AgentManager
from kestrel_sovereign.signals import (
    OrderedLockManager, SignalDispatcher, SignalLogStore, SourceRegistry,
)
from tests.integration.test_budget_refund_original_control import registered_cleanup_component
from tests.integration.test_execution_authority_postgres import GenerationFence, native_pg as native_pg
from tests.integration.test_execution_custody_review56_pg import lose_native_ack


@pytest.mark.asyncio
@pytest.mark.parametrize("route", ["enqueue", "ack", "terminal", "observer", "admission-observer", "late-ack"])
@pytest.mark.parametrize("source_cancel", [False, True])
async def test_restart_ack_keeps_original_control_and_borrowed_dispatch_owner(
    native_pg, tmp_path, monkeypatch, route, source_cancel,
):
    backend, observer, schema = native_pg
    child, db = await registered_cleanup_component(
        AgentManager(base_data_dir=tmp_path), schema, tmp_path, monkeypatch,
    )
    feature = RestartCoordinatorFeature(child)
    # Explicit private feature-component setup, not successful Agent startup.
    feature._db = db
    feature._inflight_restart_acks = set()
    # Disposable local host seal only; never provider credentials or authority
    # to execute a host restart. The Agent-request bounds remain enforced.
    monkeypatch.setenv("KESTREL_API_KEY", "disposable-native-restart-ack-component-host-seal")
    await ensure_restart_requests_table(db)
    request = await insert_request(
        db, requested_by_agent=child.did, reason="native ACK component",
        allow_agent_request=True,
    )
    await update_status(db, request.id, status="completed", expected_current_status="pending")
    row = await get_request(db, request.id)
    dispatcher = SignalDispatcher(
        agent=child, registry=SourceRegistry(), lock_manager=OrderedLockManager(),
        store=SignalLogStore(backend),
    )
    child.dispatcher = dispatcher
    entered, release = asyncio.Event(), asyncio.Event()
    originals, graphs, debts, calls = [], [], [], []
    handle = watcher = ack = pending_replacement = None

    async def native_effect():
        calls.append(asyncio.current_task())
        entered.set()
        await release.wait()
        try:
            with bind_execution_custody(GenerationFence()), monkeypatch.context() as fault:
                if route not in {"observer", "admission-observer"}:
                    committed = lose_native_ack(
                        fault, boundary="commit", original=OSError("actual restart ACK COMMIT acknowledgement lost"),
                    )
                await backend.execute("INSERT INTO effects VALUES (223, 'original restart wake component')")
        except BaseException as error:
            assert committed == [True]
            if source_cancel:
                asyncio.current_task().cancel("actual original restart dispatch withdrawal")
                try:
                    await asyncio.sleep(0)
                except asyncio.CancelledError as cancellation:
                    cancellation.__cause__ = error
                    error = cancellation
            originals.append(error)
            graphs.append(tuple(_explicit_control_chain(error)))
            debts.append(tuple(execution_cleanup_debts(error)))
            raise error
        return SignalResult(signal_id="native-restart", status=Status.OK, mode=SignalMode.COGNITION, duration_ms=0)

    async def native_dispatch(*args, **kwargs):
        return await native_effect()

    async def refused_enqueue(*args, **kwargs):
        return await native_effect()

    monkeypatch.setattr(dispatcher, "dispatch_signal", native_dispatch)
    try:
        if route == "enqueue":
            monkeypatch.setattr(dispatcher, "enqueue_signal", refused_enqueue)
            operation = feature._deliver_restart_completed(row, child.did, dispatcher, True)
            watcher = create_owned_task(operation, retain_originals=True)
            await asyncio.wait_for(entered.wait(), 5)
            release.set()
            outcome = await await_owned_task(watcher)
        else:
            handle = await dispatcher.enqueue_signal(Signal(
                source="restart.completed", kind="native-component",
                mode=SignalMode.COGNITION, target_agent=child.did, payload={},
            ))
            await asyncio.wait_for(entered.wait(), 5)
            if route in {"ack", "observer", "late-ack"}:
                feature._inflight_restart_acks.add(row.id)
                ack = feature._spawn_ack_supervisor(row, handle)
                # Queue a completion marker behind the observer's initial run;
                # no time-based CPU load or in-band cancellation timeout.
                started = asyncio.get_running_loop().create_future()
                asyncio.get_running_loop().call_soon(started.set_result, None)
                await started
                watcher = ack
            else:
                operation = (handle.wait_for_durable_admission() if route == "admission-observer"
                             else handle.wait())
                watcher = create_owned_task(operation, retain_originals=True)
                started = asyncio.get_running_loop().create_future()
                asyncio.get_running_loop().call_soon(started.set_result, None)
                await started
            if route in {"observer", "admission-observer", "late-ack"}:
                watcher.cancel("actual borrowed ACK observer withdrew")
                outcome = await await_owned_task(watcher)
                assert isinstance(outcome.error, asyncio.CancelledError)
                assert not handle.task.done() and not handle.task.cancelling(), "ACK observer cancelled borrowed dispatcher owner"
                assert not handle.durable_admission.cancelled()
                withdrawn = outcome.error
                if route == "late-ack":
                    assert row.id in feature._inflight_restart_acks, "withdrawn ACK released retry guard while original dispatch still owns work"
                    await feature.shutdown()
                    assert row.id in feature._inflight_restart_acks, "feature disable released borrowed dispatch retry guard"
                    replacement = RestartCoordinatorFeature(child)
                    replacement._db = db
                    replacement._inflight_restart_acks = set()
                    pending_replacement = await replacement._deliver_restart_completed(row, child.did, dispatcher, True)
                    assert pending_replacement is None, "replacement admitted second wake while original dispatch pending"
                    assert len(calls) == 1 and not handle.task.done() and not handle.task.cancelling()
                release.set()
                completed = await await_owned_task(handle.task)
                if route == "late-ack":
                    assert originals and originals[0] in tuple(_explicit_control_chain(completed.error))
                    assert originals[0] in tuple(_explicit_control_chain(vars(feature).get("_owned_background_control"))), "original producer completion did not reach Feature retry fence"
                    assert withdrawn not in tuple(_explicit_control_chain(vars(feature).get("_owned_background_control")))
                    outcome = completed
                else:
                    assert completed.error is None
            else:
                release.set()
                outcome = await await_owned_task(watcher)
        if route not in {"observer", "admission-observer"}:
            assert originals and originals[0] in tuple(_explicit_control_chain(outcome.error)), "restart ACK normalized original dispatch control into retryable absence"
            assert execution_commit_outcome(outcome.error) == "unknown"
            if route == "terminal":
                # A second observer of an already-cancelled Task must retrieve
                # the SAME original, not asyncio's fresh bare cancellation.
                for _ in range(3):
                    with pytest.raises(BaseException) as completed:
                        await handle.wait()
                    assert originals[0] in tuple(_explicit_control_chain(completed.value))
            if route in {"enqueue", "ack", "late-ack"}:
                for _ in range(3):
                    with pytest.raises(BaseException) as blocked:
                        await feature._deliver_restart_completed(row, child.did, dispatcher, True)
                    assert originals[0] in tuple(_explicit_control_chain(blocked.value)), "restart sweep retried original uncertain wake"
                if route == "late-ack":
                    replacement = RestartCoordinatorFeature(child)
                    replacement._db = db
                    replacement._inflight_restart_acks = set()
                    with pytest.raises(BaseException) as later_instance:
                        await replacement._deliver_restart_completed(row, child.did, dispatcher, True)
                    assert originals[0] in tuple(_explicit_control_chain(later_instance.value)), "new feature instance ignored Agent-owned original dispatch control"
            assert tuple(_explicit_control_chain(originals[0])) == graphs[0]
            assert tuple(execution_cleanup_debts(originals[0])) == debts[0]
        assert len(calls) == 1
        assert await observer.fetchval(f'SELECT count(*) FROM "{schema}".effects WHERE id=223') == 1
        assert not (await get_request(db, row.id)).wake_delivered, "withdrawn or uncertain ACK falsely marked wake delivered"
    finally:
        release.set()
        for task in (watcher, ack, pending_replacement, handle.task if handle else None, *tuple(child._background_tasks)):
            if task is not None:
                await await_owned_task(task)
        if handle is not None and not handle.durable_admission.done():
            handle.durable_admission.cancel()
        try:
            await feature.shutdown()
        except BaseException as error:
            assert is_execution_control_error(error)
        await db.close()
