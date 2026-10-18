"""Original native reservation CAS evidence, not cognition/Agent READY."""
import asyncio
import inspect

import pytest
from kestrel_sdk.signals import Signal, SignalMode

from kestrel_sovereign._async_ownership import await_owned_task, create_owned_task
from kestrel_sovereign.execution_custody import (
    _explicit_control_chain, bind_execution_custody, execution_cleanup_debts,
    execution_commit_outcome, is_execution_control_error,
)
from kestrel_sovereign.multi_agent.agent_manager import AgentManager
from kestrel_sovereign.signals import (
    DurableConsumerRegistration, DurableSignalStore, OrderedLockManager,
    SignalDispatcher, SignalLogStore, SourceRegistry,
)
from tests.integration.test_budget_refund_original_control import registered_cleanup_component
from tests.integration.test_execution_authority_postgres import GenerationFence, native_pg as native_pg
from tests.integration.test_execution_custody_review56_pg import lose_native_ack
from tests.unit.test_host_context_native_cleanup import _actual_operation
from kestrel_sovereign import _async_ownership as ownership


@pytest.mark.asyncio
@pytest.mark.parametrize(("route", "control"), [
    ("direct", "unknown"), ("owned", "unknown"),
    ("direct", "withdrawal"), ("owned", "withdrawal"),
    ("owned", "admission"), ("owned", "publication"),
    ("shutdown", "unknown"),
])
async def test_uncertain_original_reservation_repair_is_not_retried_or_normalized(
    native_pg, tmp_path, monkeypatch, route, control,
):
    backend, observer, schema = native_pg
    child, database = await registered_cleanup_component(AgentManager(base_data_dir=tmp_path), schema, tmp_path, monkeypatch)
    store = DurableSignalStore(backend)
    await store.initialize()
    dispatcher = SignalDispatcher(
        agent=child, registry=SourceRegistry(), lock_manager=OrderedLockManager(),
        store=SignalLogStore(backend), durable_store=store,
    )
    await store.register_runtime_owner(agent_id=child.did, owner_id=dispatcher._durable_delivery_owner)
    await store.register_consumer(DurableConsumerRegistration(
        consumer_id="original-native-repair", source="native.reservation", agent_id=child.did,
    ))
    persisted = await store.persist_signal(
        Signal(source="native.reservation", kind="component", mode=SignalMode.COGNITION,
               target_agent=child.did, payload={}),
        agent_id=child.did, source_event_id="original-native-cas", retention_days=1,
        transient_selector_payload={}, initial_lease_owner=dispatcher._durable_delivery_owner,
    )
    assert len(persisted.initial_reservations) == 1
    reservation = persisted.initial_reservations[0]
    actual_abandon = store.abandon_initial_reservation
    attempts, originals, withdrawals, source_graphs, source_debts = [], [], [], [], []
    owner = None
    loop, previous = asyncio.get_running_loop(), asyncio.get_running_loop().get_task_factory()
    accepted, refused = [], []
    allocation = RuntimeError("original native reservation repair admission/publication refused")
    admission_entered, admission_release, shutdown_entered = asyncio.Event(), asyncio.Event(), asyncio.Event()
    admitted = shutdown_observer = None
    shutdown_withdrawals = []
    actual_shield = asyncio.shield
    actual_join = ownership.await_owned_task

    async def observed_join(completion, *args, **kwargs):
        selected = asyncio.current_task() is shutdown_observer and completion is dispatcher._durable_shutdown_completion
        if selected:
            shutdown_entered.set()
        try:
            return await actual_join(completion, *args, **kwargs)
        except asyncio.CancelledError as cancellation:
            if selected:
                shutdown_withdrawals.append(cancellation)
            raise

    async def previously_admitted():
        async with dispatcher._admit_durable_operation():
            admission_entered.set()
            await admission_release.wait()

    def observed_shield(completion):
        future = actual_shield(completion)
        if asyncio.current_task() is shutdown_observer and completion is dispatcher._durable_shutdown_completion:
            async def observe():
                shutdown_entered.set()
                try:
                    return await future
                except asyncio.CancelledError as cancellation:
                    shutdown_withdrawals.append(cancellation)
                    raise
            return observe()
        return future

    class RefusedPublication(set):
        def add(self, task):
            accepted.append(task)
            raise allocation

    def factory(loop, coroutine, **kwargs):
        operation = _actual_operation(coroutine)
        if getattr(operation, "cr_code", None) is SignalDispatcher._requeue_committed_initial_reservations.__code__:
            refused.append((coroutine, operation))
            raise allocation
        return previous(loop, coroutine, **kwargs) if previous else asyncio.Task(coroutine, loop=loop, **kwargs)

    async def abandon(**kwargs):
        attempts.append(kwargs)
        if control == "withdrawal" and len(attempts) == 1:
            asyncio.current_task().cancel("actual ordinary reservation observer withdrew")
            try:
                await asyncio.sleep(0)
            except asyncio.CancelledError as cancellation:
                withdrawals.append(cancellation)
                raise
        try:
            return await actual_abandon(**kwargs)
        except BaseException as error:
            # A supported observer can withdraw while propagating an already
            # original native UNKNOWN. Preserve that actual causal carrier.
            assert execution_commit_outcome(error) == "unknown"
            originals.append(error)
            source_graphs.append(tuple(_explicit_control_chain(error)))
            source_debts.append(tuple(execution_cleanup_debts(error)))
            asyncio.current_task().cancel("actual original reservation repair observer withdrew")
            try:
                await asyncio.sleep(0)
            except asyncio.CancelledError as cancellation:
                withdrawals.append(cancellation)
                raise cancellation from error

    async def run():
        with bind_execution_custody(GenerationFence()):
            if route == "direct":
                await dispatcher._requeue_committed_initial_reservations(persisted)
            else:
                await dispatcher._repair_post_commit_reservations(persisted)

    try:
        if route == "shutdown":
            admitted = create_owned_task(previously_admitted(), retain_originals=True)
            await asyncio.wait_for(admission_entered.wait(), 5)
        if control == "admission":
            loop.set_task_factory(factory)
        elif control == "publication":
            dispatcher._post_commit_reservation_repairs = RefusedPublication()
        with monkeypatch.context() as ack:
            committed = (lose_native_ack(ack, boundary="commit", original=OSError("actual reservation CAS COMMIT ACK lost"))
                         if control in {"unknown", "publication"} else [])
            monkeypatch.setattr(store, "abandon_initial_reservation", abandon)
            owner = create_owned_task(run(), retain_originals=True)
            outcome = await await_owned_task(owner)
        loop.set_task_factory(previous)
        chain = tuple(_explicit_control_chain(outcome.error))
        if control == "admission":
            assert allocation in chain and not attempts and not committed
            assert refused and all(inspect.getcoroutinestate(coro) == inspect.CORO_CLOSED for pair in refused for coro in pair)
            assert any(resource is dispatcher and task is None for resource, task in execution_cleanup_debts(outcome.error))
            with pytest.raises(BaseException) as blocked:
                async with dispatcher._admit_durable_operation():
                    pytest.fail("unstarted original repair admitted new durable work")
            assert is_execution_control_error(blocked.value)
            assert await observer.fetchval(f'SELECT status FROM "{schema}".durable_signal_deliveries WHERE delivery_id=$1', reservation.delivery_id) == "initial_reserved"
            return
        if control == "publication":
            assert allocation in chain
            assert accepted and all(task.done() for task in accepted), "accepted original repair escaped refused publication"
            assert not dispatcher._post_commit_reservation_repairs
        assert committed == ([True] if control in {"unknown", "publication"} else [])
        assert len(attempts) == (2 if control == "withdrawal" else 1), "native UNKNOWN was retried or ordinary mandatory repair was abandoned"
        assert len(withdrawals) == 1 and withdrawals[0] in chain, "repair normalized its actual original cancellation"
        if control in {"unknown", "publication"}:
            assert len(originals) == 1 and originals[0] in chain
            assert execution_commit_outcome(outcome.error) == "unknown"
            for _ in range(20):
                with pytest.raises(BaseException) as blocked:
                    async with dispatcher._admit_durable_operation():
                        pytest.fail("unreconciled original CAS admitted more work")
                assert is_execution_control_error(blocked.value)
                # This is the ORIGINAL repair Task's cause-bearing native
                # carrier, not an unrelated later reader's cancellation.
                assert withdrawals[0] in tuple(_explicit_control_chain(blocked.value))
                assert tuple(_explicit_control_chain(originals[0])) == source_graphs[0]
                assert tuple(execution_cleanup_debts(originals[0])) == source_debts[0]
            with pytest.raises(BaseException) as drained:
                await dispatcher._drain_post_commit_reservation_repairs()
            assert is_execution_control_error(drained.value), "completed repair lost terminal control before owner release"
        else:
            assert outcome.error is withdrawals[0]
            assert not originals and dispatcher._outcome_log_control_error is None
        row = await observer.fetchrow(
            f'SELECT status,lease_owner,lease_token FROM "{schema}".durable_signal_deliveries WHERE delivery_id=$1',
            reservation.delivery_id,
        )
        assert tuple(row) == ("retry", None, None), "the original owner/token CAS did not physically commit"
        assert await observer.fetchval(f'SELECT count(*) FROM "{schema}".durable_signal_events') == 1
        if route == "shutdown":
            monkeypatch.setattr(asyncio, "shield", observed_shield)
            monkeypatch.setattr(ownership, "await_owned_task", observed_join)
            shutdown_observer = create_owned_task(dispatcher.shutdown_durable_delivery(), retain_originals=True)
            await asyncio.wait_for(shutdown_entered.wait(), 5)
            completion = dispatcher._durable_shutdown_completion
            assert completion is not None and not completion.done()
            shutdown_observer.cancel("actual outer repair-to-shutdown observer withdrew")
            observed = await await_owned_task(shutdown_observer)
            public_chain = tuple(_explicit_control_chain(observed.error))
            assert len(shutdown_withdrawals) == 1 and shutdown_withdrawals[0] in public_chain, "shutdown observer discarded its actual withdrawal beside retained CAS uncertainty"
            assert originals[0] in public_chain and execution_commit_outcome(observed.error) == "unknown"
            assert shutdown_withdrawals[0] not in tuple(_explicit_control_chain(dispatcher._outcome_log_control_error))
            assert not completion.done() and not completion.cancelled(), "public observer adopted the existing shutdown completion owner"
            admission_release.set()
            await await_owned_task(admitted)
            completed = await await_owned_task(completion)
            assert is_execution_control_error(completed.error)
            assert shutdown_withdrawals[0] not in tuple(_explicit_control_chain(completed.error))
            assert tuple(_explicit_control_chain(originals[0])) == source_graphs[0]
            assert tuple(execution_cleanup_debts(originals[0])) == source_debts[0]
            assert await observer.fetchval(f'SELECT stopped_at FROM "{schema}".durable_signal_runtime_owners WHERE agent_id=$1 AND owner_id=$2', child.did, dispatcher._durable_delivery_owner) is None
    finally:
        loop.set_task_factory(previous)
        admission_release.set()
        if owner is not None:
            await await_owned_task(owner)
        if admitted is not None:
            await await_owned_task(admitted)
        if shutdown_observer is not None:
            await await_owned_task(shutdown_observer)
        if dispatcher._durable_shutdown_completion is not None:
            await await_owned_task(dispatcher._durable_shutdown_completion)
        for task in accepted:
            await await_owned_task(task)
        for pair in refused:
            for coroutine in pair:
                coroutine.close()
        await dispatcher._stop_runtime_owner_heartbeat()
        await await_owned_task(create_owned_task(child.shutdown(), retain_originals=True))
        await database.close()
