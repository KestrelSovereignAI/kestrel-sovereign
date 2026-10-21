"""Actual partial Agent tail + real loopback snapshot; not Agent READY acceptance."""

import asyncio

import pytest
from tests.utils.original_failure import OriginalFailure

from kestrel_sovereign._async_ownership import await_owned_task, create_owned_task, owned_task_error
from kestrel_sovereign.execution_custody import (
    _explicit_control_chain,
    execution_cleanup_debts,
    is_execution_control_error,
)
from tests.integration.test_lighthouse_snapshot_original_control import (
    native_snapshot as native_snapshot,
)
from tests.unit.test_cleanup_task_admission import (
    _operation_name,
    _partial_native_agent,
)


# This is a pending-native-close ownership test, not a cold HTTP setup latency
# benchmark. Keep a finite observer budget that permits real archive/client
# setup on both interpreters before the deliberately held physical close.
SNAPSHOT_TAIL_OBSERVER_BUDGET_S = 1.0


async def _retire_partial_agent_fixture(agent, usage_db):
    # The finite tail may now retain native storage as well as snapshot owners.
    # Join those SAME owners before independently closing the fixture database;
    # a concurrent second physical close is not fixture isolation.
    await await_owned_task(create_owned_task(
        agent.wait_for_shutdown_completion(), retain_originals=True,
    ))
    await agent.llm_service.close()
    await usage_db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("withdrawal", [False, True])
@pytest.mark.parametrize("close_cancels", [False, True])
@pytest.mark.parametrize("completion", ["original-owners", "agent-wait"])
async def test_agent_bounded_tail_retains_actual_pending_snapshot_closer(
    native_snapshot, tmp_path, monkeypatch, withdrawal, close_cancels, completion
):
    (
        db,
        target,
        service,
        entered,
        close_entered,
        release_close,
        clients,
        source_errors,
        close_tasks,
        state,
        tasks,
    ) = native_snapshot
    agent, usage_db = await _partial_native_agent(tmp_path, monkeypatch)
    agent._sync_service = service
    agent._resource_custody().acquire("sync_service")
    if close_cancels:
        state["close_error"] = asyncio.CancelledError(
            "actual native close ACK cancelled"
        )
        state["close_error"].__cause__ = OSError("actual native close ACK failed")
    tail = create_owned_task(
        agent._run_durable_shutdown_tail(SNAPSHOT_TAIL_OBSERVER_BUDGET_S), retain_originals=True
    )
    tasks.append(tail)
    try:
        await asyncio.wait_for(entered.wait(), 3)
        if withdrawal:
            tail.cancel("actual Agent tail observer withdrew")
        await asyncio.wait_for(close_entered.wait(), 3)
        for _ in range(200):
            if tail.done():
                break
            await asyncio.sleep(0.01)
        assert tail.done(), "Agent tail escaped its original finite guard"
        assert not close_tasks[0].done(), "physical close ACK was not held at assertion"
        terminal = vars(agent).get("_shutdown_control")
        assert is_execution_control_error(terminal), (
            "bounded tail discarded its actual native close owner"
        )
        debts = execution_cleanup_debts(terminal)
        assert any(
            resource is service and owner is not None and not owner.done()
            for resource, owner in debts
        ), "same snapshot Task must retain its accepted native closer"
        waiter = None
        if completion == "agent-wait":
            from kestrel_sovereign import _async_ownership

            actual_join = _async_ownership.await_owned_task
            snapshot_owners = {
                owner for resource, owner in debts
                if resource is service and owner is not None
            }

            async def observe_original_join(owner, *args, **kwargs):
                # Assert the original dispatch before it could re-join a done
                # owner forever. No fake completion/ACK is supplied.
                if owner in snapshot_owners:
                    assert not owner.done(), (
                        "completed debt owner was re-admitted to the join loop"
                    )
                return await actual_join(owner, *args, **kwargs)

            monkeypatch.setattr(
                _async_ownership, "await_owned_task", observe_original_join
            )
            waiter = create_owned_task(
                agent.wait_for_shutdown_completion(), retain_originals=True
            )
            tasks.append(waiter)
            await asyncio.sleep(0.02)
            assert not waiter.done(), (
                "Agent completion returned before physical retirement"
            )
            if withdrawal:
                waiter.cancel("original completion observer withdrew")
                await asyncio.sleep(0.02)
                assert not waiter.done(), (
                    "completion withdrawal cancelled the retained native owner"
                )
        release_close.set()
        owners = {
            owner
            for resource, owner in debts
            if resource is service and owner is not None
        }
        await asyncio.gather(*owners, return_exceptions=True)
        if waiter is not None:
            await asyncio.gather(waiter, return_exceptions=True)
            assert is_execution_control_error(owned_task_error(waiter))
            if withdrawal:
                assert any(
                    isinstance(error, asyncio.CancelledError)
                    and error.args == ("original completion observer withdrew",)
                    for error in _explicit_control_chain(owned_task_error(waiter))
                )
        for _ in range(10):
            await asyncio.sleep(0)
        terminal = vars(agent).get("_shutdown_control")
        assert state["close_error"] in tuple(_explicit_control_chain(terminal)), (
            "eventual native failure did not reach original Agent lifetime"
        )
        assert any(
            resource is clients[0] and owner is close_tasks[0]
            for resource, owner in execution_cleanup_debts(terminal)
        )
        assert agent._resource_custody().is_held("sync_service")
    finally:
        release_close.set()
        if close_tasks:
            await asyncio.gather(*close_tasks, return_exceptions=True)
        await asyncio.gather(tail, return_exceptions=True)
        await _retire_partial_agent_fixture(agent, usage_db)


@pytest.mark.asyncio
async def test_agent_retains_periodic_stop_and_explicit_snapshot_owners(
    native_snapshot, tmp_path, monkeypatch
):
    (
        db,
        target,
        service,
        entered,
        close_entered,
        release_close,
        clients,
        source_errors,
        close_tasks,
        state,
        tasks,
    ) = native_snapshot
    agent, usage_db = await _partial_native_agent(tmp_path, monkeypatch)
    agent._sync_service = service
    agent._resource_custody().acquire("sync_service")
    await service.start()
    state["close_errors"] = [
        OSError("original explicit snapshot native close failed"),
        OSError("original periodic snapshot native close failed"),
    ]
    worker = service._poll_task
    tasks.append(worker)
    await asyncio.wait_for(entered.wait(), 3)
    tail = create_owned_task(
        agent._run_durable_shutdown_tail(SNAPSHOT_TAIL_OBSERVER_BUDGET_S), retain_originals=True
    )
    tasks.append(tail)
    try:
        for _ in range(200):
            if tail.done() and len(close_tasks) == 2:
                break
            await asyncio.sleep(0.01)
        assert tail.done() and len(close_tasks) == 2
        assert not worker.done() and all(not owner.done() for owner in close_tasks)
        debts = execution_cleanup_debts(vars(agent).get("_shutdown_control"))
        owners = {
            owner
            for resource, owner in debts
            if resource is service and owner is not None and not owner.done()
        }
        assert len(owners) == 2, (
            "snapshot and periodic-stop roots must BOTH remain owned"
        )
        release_close.set()
        await asyncio.gather(*owners, worker, return_exceptions=True)
        for _ in range(10):
            await asyncio.sleep(0)
        debts = execution_cleanup_debts(vars(agent).get("_shutdown_control"))
        assert all(
            any(
                resource is client and owner in close_tasks for resource, owner in debts
            )
            for client in clients
        )
        assert all(
            error
            in tuple(_explicit_control_chain(vars(agent).get("_shutdown_control")))
            for error in state["close_errors"]
        ), "one original native failure replaced another"
    finally:
        release_close.set()
        if close_tasks:
            await asyncio.gather(*close_tasks, return_exceptions=True)
        await asyncio.gather(tail, worker, return_exceptions=True)
        await _retire_partial_agent_fixture(agent, usage_db)


@pytest.mark.asyncio
async def test_tail_captures_original_cause_bearing_native_callback_cancellation(
    native_snapshot, tmp_path, monkeypatch
):
    (
        db,
        target,
        service,
        entered,
        close_entered,
        release_close,
        clients,
        source_errors,
        close_tasks,
        state,
        tasks,
    ) = native_snapshot
    state["upload_failure_as_cancel"] = True
    state["release_server"].set()
    state["close_error"] = None
    agent, usage_db = await _partial_native_agent(tmp_path, monkeypatch)
    agent._sync_service = service
    agent._resource_custody().acquire("sync_service")
    tail = create_owned_task(
        agent._run_durable_shutdown_tail(SNAPSHOT_TAIL_OBSERVER_BUDGET_S), retain_originals=True
    )
    tasks.append(tail)
    try:
        await asyncio.wait_for(close_entered.wait(), 3)
        assert source_errors and isinstance(source_errors[0], asyncio.CancelledError)
        import httpx

        assert isinstance(source_errors[0].__cause__, httpx.HTTPStatusError)
        for _ in range(200):
            if tail.done():
                break
            await asyncio.sleep(0.01)
        assert tail.done() and not close_tasks[0].done()
        debts = execution_cleanup_debts(vars(agent).get("_shutdown_control"))
        owners = {
            owner
            for resource, owner in debts
            if resource is service and owner is not None
        }
        assert owners
        release_close.set()
        await asyncio.gather(*owners, return_exceptions=True)
        for _ in range(10):
            await asyncio.sleep(0)
        terminal = vars(agent).get("_shutdown_control")
        assert source_errors[0] in tuple(_explicit_control_chain(terminal))
        assert source_errors[0].__cause__ in tuple(_explicit_control_chain(terminal))
        assert all(owner.done() for owner in close_tasks)
    finally:
        release_close.set()
        if close_tasks:
            await asyncio.gather(*close_tasks, return_exceptions=True)
        await asyncio.gather(tail, return_exceptions=True)
        await _retire_partial_agent_fixture(agent, usage_db)


@pytest.mark.asyncio
async def test_refused_snapshot_owner_preserves_unstarted_service_debt(
    native_snapshot, tmp_path, monkeypatch
):
    from inspect import CORO_CLOSED, getcoroutinestate

    from tests.unit.test_cleanup_task_admission import _operation_name

    (
        db,
        target,
        service,
        entered,
        close_entered,
        release_close,
        clients,
        source_errors,
        close_tasks,
        state,
        tasks,
    ) = native_snapshot
    agent, usage_db = await _partial_native_agent(tmp_path, monkeypatch)
    agent._sync_service = service
    agent._resource_custody().acquire("sync_service")
    loop, factory = (
        asyncio.get_running_loop(),
        asyncio.get_running_loop().get_task_factory(),
    )
    refusal = RuntimeError("actual final snapshot Task admission refused")
    source = OriginalFailure.capture(refusal)
    refused = []

    def refuse_snapshot(loop, coroutine, **kwargs):
        if _operation_name(coroutine) == "SyncService.force_snapshot":
            refused.append(coroutine)
            raise refusal
        return (
            factory(loop, coroutine, **kwargs)
            if factory
            else asyncio.Task(coroutine, loop=loop, **kwargs)
        )

    loop.set_task_factory(refuse_snapshot)
    try:
        with pytest.raises(BaseException) as result:
            await agent._run_durable_shutdown_tail(0.2)
        source.assert_retained(result.value)
        assert refused
        assert all(getcoroutinestate(coro) == CORO_CLOSED for coro in refused)
        assert not clients and not entered.is_set()
        assert (service, None) in execution_cleanup_debts(result.value)
        assert vars(agent).get("_shutdown_control") is result.value
        with pytest.raises(BaseException) as completion:
            await agent.wait_for_shutdown_completion()
        source.assert_retained(completion.value)
        assert completion.value is result.value, (
            "unstarted native debt was replaced or acknowledged"
        )
    finally:
        loop.set_task_factory(factory)
        await _retire_partial_agent_fixture(agent, usage_db)


@pytest.mark.asyncio
@pytest.mark.parametrize("caller", ["native-child", "snapshot-owner"])
async def test_completion_from_original_native_join_graph_keeps_owner_fenced(
    native_snapshot, tmp_path, monkeypatch, caller
):
    from kestrel_sovereign import _async_ownership

    (
        db,
        target,
        service,
        entered,
        close_entered,
        release_close,
        clients,
        source_errors,
        close_tasks,
        state,
        tasks,
    ) = native_snapshot
    agent, usage_db = await _partial_native_agent(tmp_path, monkeypatch)
    agent._sync_service = service
    agent._resource_custody().acquire("sync_service")
    observed = []

    async def completion_probe():
        try:
            await agent.wait_for_shutdown_completion()
        except BaseException as error:
            observed.append(error)

    if caller == "native-child":
        state["close_probe"] = completion_probe
    else:
        original_force = service.force_snapshot

        async def original_snapshot():
            try:
                return await original_force()
            finally:
                await completion_probe()

        monkeypatch.setattr(service, "force_snapshot", original_snapshot)
    tail = create_owned_task(
        agent._run_durable_shutdown_tail(SNAPSHOT_TAIL_OBSERVER_BUDGET_S), retain_originals=True
    )
    tasks.append(tail)
    try:
        await asyncio.wait_for(entered.wait(), 3)
        await asyncio.wait_for(close_entered.wait(), 3)
        for _ in range(200):
            if tail.done():
                break
            await asyncio.sleep(0.01)
        assert tail.done() and not close_tasks[0].done()
        debts = execution_cleanup_debts(vars(agent).get("_shutdown_control"))
        assert debts
        original_join = _async_ownership.await_owned_task

        async def observe_original_join(owner, *args, **kwargs):
            current = asyncio.current_task()
            # Check dispatch BEFORE a real cycle could hang test teardown.
            # Valid joins still forward to the original owner; no ACK is faked.
            assert owner is not current, "completion tried to join its own actual Task"
            assert owner not in _async_ownership.owned_task_join_ancestors(current), (
                "completion tried to join an actual waiting ancestor"
            )
            return await original_join(owner, *args, **kwargs)

        monkeypatch.setattr(_async_ownership, "await_owned_task", observe_original_join)
        release_close.set()
        owners = {
            owner
            for resource, owner in debts
            if resource is service and owner is not None
        }
        await asyncio.gather(*owners, return_exceptions=True)
        assert len(observed) == 1 and is_execution_control_error(observed[0])
        assert not isinstance(observed[0], AssertionError)
        assert agent._resource_custody().is_held("sync_service")
    finally:
        release_close.set()
        if close_tasks:
            await asyncio.gather(*close_tasks, return_exceptions=True)
        await asyncio.gather(tail, return_exceptions=True)
        await _retire_partial_agent_fixture(agent, usage_db)


# Timely native close failures at the real partial Agent boundary.


@pytest.mark.asyncio
async def test_timely_agent_snapshot_and_stop_retain_both_native_failures(
    native_snapshot, tmp_path, monkeypatch
):
    (
        db,
        target,
        service,
        entered,
        close_entered,
        release_close,
        clients,
        source_errors,
        close_tasks,
        state,
        tasks,
    ) = native_snapshot
    agent, usage_db = await _partial_native_agent(tmp_path, monkeypatch)
    agent._sync_service = service
    agent._resource_custody().acquire("sync_service")
    state["close_errors"] = [
        OSError("actual timely periodic native close ACK lost"),
        OSError("actual timely final snapshot native close ACK lost"),
    ]
    source_graphs = [tuple(_explicit_control_chain(error)) for error in state["close_errors"]]
    source_debts = [tuple(execution_cleanup_debts(error)) for error in state["close_errors"]]
    state["release_server"].set()
    release_close.set()
    await service.start()
    worker = service._poll_task
    tasks.append(worker)
    try:
        for _ in range(250):
            if worker.done():
                break
            await asyncio.sleep(0.01)
        assert worker.done() and state["close_errors"][0] in tuple(_explicit_control_chain(owned_task_error(worker)))
        assert source_errors and source_errors[0] in tuple(_explicit_control_chain(owned_task_error(worker)))
        await agent._run_durable_shutdown_tail(5)
        assert len(close_tasks) == 2 and all(owner.done() for owner in close_tasks)
        terminal = vars(agent).get("_shutdown_control")
        chain = tuple(_explicit_control_chain(terminal))
        assert all(error in chain for error in state["close_errors"]), (
            "timely harvest replaced an original native close error"
        )
        assert all(tuple(_explicit_control_chain(error)) == before for error, before in zip(state["close_errors"], source_graphs))
        assert all(tuple(execution_cleanup_debts(error)) == before for error, before in zip(state["close_errors"], source_debts))
        assert all(
            any(
                resource is client and owner in close_tasks
                for resource, owner in execution_cleanup_debts(terminal)
            )
            for client in clients
        )
    finally:
        release_close.set()
        await asyncio.gather(worker, return_exceptions=True)
        await _retire_partial_agent_fixture(agent, usage_db)


# Original actual tail observer cancellations survive native retirement.


@pytest.mark.asyncio
@pytest.mark.parametrize("repeat", [False, True])
@pytest.mark.parametrize("close_cancels", [False, True])
async def test_agent_retains_exact_original_tail_observer_cancellation(
    native_snapshot, tmp_path, monkeypatch, repeat, close_cancels
):
    (
        db,
        target,
        service,
        entered,
        close_entered,
        release_close,
        clients,
        source_errors,
        close_tasks,
        state,
        tasks,
    ) = native_snapshot
    agent, usage_db = await _partial_native_agent(tmp_path, monkeypatch)
    agent._sync_service = service
    agent._resource_custody().acquire("sync_service")
    if close_cancels:
        state["close_error"] = asyncio.CancelledError(
            "actual native close ACK cancellation"
        )
        state["close_error"].__cause__ = OSError(
            "actual original native close ACK failure"
        )
    actual_wait = asyncio.wait
    observed = []
    tail = None

    async def observe_original_withdrawal(*args, **kwargs):
        try:
            return await actual_wait(*args, **kwargs)
        except asyncio.CancelledError as withdrawal:
            if asyncio.current_task() is tail:
                observed.append(withdrawal)
            raise

    monkeypatch.setattr(asyncio, "wait", observe_original_withdrawal)
    tail = create_owned_task(
        agent._run_durable_shutdown_tail(SNAPSHOT_TAIL_OBSERVER_BUDGET_S), retain_originals=True
    )
    tasks.append(tail)
    try:
        await asyncio.wait_for(entered.wait(), 3)
        tail.cancel("original tail observer withdrawal")
        if repeat:
            await asyncio.sleep(0.02)
            tail.cancel("second original tail observer withdrawal")
        await asyncio.wait_for(close_entered.wait(), 3)
        for _ in range(200):
            if tail.done():
                break
            await asyncio.sleep(0.01)
        assert tail.done() and not close_tasks[0].done()
        assert len(observed) == (2 if repeat else 1)
        debts = execution_cleanup_debts(vars(agent).get("_shutdown_control"))
        owners = {
            owner
            for resource, owner in debts
            if resource is service and owner is not None
        }
        assert owners
        release_close.set()
        await asyncio.gather(*owners, return_exceptions=True)
        for _ in range(10):
            await asyncio.sleep(0)
        chain = tuple(_explicit_control_chain(vars(agent).get("_shutdown_control")))
        assert all(withdrawal in chain for withdrawal in observed), (
            "original observer object was reduced to a boolean/replaced by child cancellation"
        )
        assert state["close_error"] in chain
    finally:
        release_close.set()
        if close_tasks:
            await asyncio.gather(*close_tasks, return_exceptions=True)
        await asyncio.gather(tail, return_exceptions=True)
        await _retire_partial_agent_fixture(agent, usage_db)


# An actual earlier completion observer cannot erase native cancellation evidence.


@pytest.mark.asyncio
async def test_timely_native_cancel_receipt_survives_an_earlier_completion_reader(
    native_snapshot, tmp_path, monkeypatch
):
    (
        db,
        target,
        service,
        entered,
        close_entered,
        release_close,
        clients,
        source_errors,
        close_tasks,
        state,
        tasks,
    ) = native_snapshot
    agent, usage_db = await _partial_native_agent(tmp_path, monkeypatch)
    agent._sync_service = service
    agent._resource_custody().acquire("sync_service")
    state["close_error"] = asyncio.CancelledError(
        "original timely native close cancellation"
    )
    state["close_error"].__cause__ = OSError("actual original native close failure")
    source_graph = tuple(_explicit_control_chain(state["close_error"]))
    source_debts = tuple(execution_cleanup_debts(state["close_error"]))
    state["release_server"].set()
    release_close.set()
    loop, factory = (
        asyncio.get_running_loop(),
        asyncio.get_running_loop().get_task_factory(),
    )
    observed = []

    def observe_admitted_task(loop, coroutine, **kwargs):
        task = (
            factory(loop, coroutine, **kwargs)
            if factory
            else asyncio.Task(coroutine, loop=loop, **kwargs)
        )
        if _operation_name(coroutine) == "SyncService.force_snapshot":

            def first_reader(done):
                try:
                    error = done.exception()
                except BaseException as error:
                    observed.append(error)
                else:
                    observed.append(error)

            task.add_done_callback(first_reader)
        return task

    loop.set_task_factory(observe_admitted_task)
    try:
        await agent._run_durable_shutdown_tail(5)
        assert len(observed) == 1 and state["close_error"] in tuple(_explicit_control_chain(observed[0])), (
            "first native completion reader did not observe original cancellation"
        )
        assert is_execution_control_error(observed[0])
        assert source_errors and source_errors[0] in tuple(_explicit_control_chain(observed[0]))
        assert tuple(_explicit_control_chain(state["close_error"])) == source_graph
        assert tuple(execution_cleanup_debts(state["close_error"])) == source_debts
        assert len(close_tasks) == 1 and all(owner.done() for owner in close_tasks)
        terminal = vars(agent).get("_shutdown_control")
        assert state["close_error"] in tuple(_explicit_control_chain(terminal)), (
            "later harvest manufactured cause-free cancellation"
        )
        assert any(
            resource is clients[0] and owner is close_tasks[0]
            for resource, owner in execution_cleanup_debts(terminal)
        )
    finally:
        loop.set_task_factory(factory)
        await _retire_partial_agent_fixture(agent, usage_db)


@pytest.mark.asyncio
async def test_timely_original_callback_cancellation_is_raised_after_successful_close(
    native_snapshot, tmp_path, monkeypatch
):
    (
        db,
        target,
        service,
        entered,
        close_entered,
        release_close,
        clients,
        source_errors,
        close_tasks,
        state,
        tasks,
    ) = native_snapshot
    agent, usage_db = await _partial_native_agent(tmp_path, monkeypatch)
    agent._sync_service = service
    agent._resource_custody().acquire("sync_service")
    state["upload_failure_as_cancel"] = True
    state["close_error"] = None
    state["release_server"].set()
    release_close.set()
    try:
        cancelled, _ = await agent._run_durable_shutdown_tail(5)
        assert cancelled and source_errors
        original = source_errors[0]
        assert isinstance(original, asyncio.CancelledError) and original.__cause__
        assert all(owner.done() for owner in close_tasks)
        assert not execution_cleanup_debts(vars(agent).get("_shutdown_control"))
        with pytest.raises(asyncio.CancelledError) as result:
            agent._raise_shutdown_control()
        assert result.value is original, (
            "successful close replaced original cancellation with its cause"
        )
    finally:
        await _retire_partial_agent_fixture(agent, usage_db)


@pytest.mark.asyncio
@pytest.mark.parametrize("door", ["tail", "completion", "outer"])
@pytest.mark.parametrize("close_cancels", [False, True])
@pytest.mark.parametrize("refusal_cancels", [False, True])
@pytest.mark.parametrize("late_background", [False, True])
async def test_actual_snapshot_failure_and_stop_refusal_keep_both_originals(
    native_snapshot, tmp_path, monkeypatch, door, close_cancels, refusal_cancels,
    late_background,
):
    from inspect import CORO_CLOSED, getcoroutinestate

    from kestrel_sovereign.kestrel_agent import await_agent_shutdown_completion

    (
        db,
        target,
        service,
        entered,
        close_entered,
        release_close,
        clients,
        source_errors,
        close_tasks,
        state,
        tasks,
    ) = native_snapshot
    agent, usage_db = await _partial_native_agent(tmp_path, monkeypatch)
    agent._sync_service = service
    agent._resource_custody().acquire("sync_service")
    if close_cancels:
        state["close_error"] = asyncio.CancelledError(
            "actual original native close cancellation"
        )
        state["close_error"].__cause__ = OSError(
            "actual original physical close ACK lost"
        )
    original = state["close_error"]
    refusal = (
        asyncio.CancelledError("original refused sync-stop admission")
        if refusal_cancels
        else RuntimeError("original refused sync-stop admission")
    )
    if refusal_cancels:
        refusal.__cause__ = OSError("actual stop Task factory refusal")
    state["release_server"].set()
    release_close.set()
    loop, factory = (
        asyncio.get_running_loop(),
        asyncio.get_running_loop().get_task_factory(),
    )
    refused = []

    def refuse_stop(loop, coroutine, **kwargs):
        frame = coroutine.cr_frame
        operation = frame.f_locals.get("operation", coroutine) if frame else coroutine
        operation_frame = getattr(operation, "cr_frame", None)
        if (
            _operation_name(coroutine) == "ResourceCustody.release"
            and operation_frame
            and operation_frame.f_locals.get("name") == "sync_service"
        ):
            refused.append(coroutine)
            raise refusal
        return (
            factory(loop, coroutine, **kwargs)
            if factory
            else asyncio.Task(coroutine, loop=loop, **kwargs)
        )

    loop.set_task_factory(refuse_stop)
    try:
        tail_failure = None
        try:
            await agent._run_durable_shutdown_tail(5)
        except BaseException as error:
            tail_failure = error
        assert (
            tail_failure is not None
            and refused
            and len(close_tasks) == 1
            and close_tasks[0].done()
        )
        assert all(getcoroutinestate(coro) == CORO_CLOSED for coro in refused)
        if late_background:
            from kestrel_sovereign.execution_custody import ExecutionCommitOutcomeError

            # Declared stronger background input AFTER actual native close A
            # and actual stop-Task refusal B; not driver-generated UNKNOWN.
            late = ExecutionCommitOutcomeError("unknown")

            async def publish_late_control():
                raise late

            completed = asyncio.Event()
            worker = agent._track_background_task(
                publish_late_control(), name="late-background-control"
            )
            worker.add_done_callback(lambda done: completed.set())
            await asyncio.wait_for(completed.wait(), 5)
            assert worker.done()
            with pytest.raises(BaseException) as retained:
                agent._raise_shutdown_control()
            tail_failure = retained.value
        selected = tail_failure
        if door != "tail":
            selected = None
            try:
                if door == "completion":
                    await agent.wait_for_shutdown_completion()
                else:
                    await await_agent_shutdown_completion(
                        agent, prior_error=tail_failure
                    )
            except BaseException as error:
                selected = error
        chain = tuple(_explicit_control_chain(selected))
        assert original in chain and refusal in chain, (
            "accepted native failure and refused subsequent stop lost an original carrier"
        )
        assert any(
            resource is service and owner is None
            for resource, owner in execution_cleanup_debts(selected)
        )
        assert any(
            resource is clients[0] and owner is close_tasks[0]
            for resource, owner in execution_cleanup_debts(selected)
        )
    finally:
        loop.set_task_factory(factory)
        await _retire_partial_agent_fixture(agent, usage_db)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "boundary", ["physical-epilogue", "cancelled-owner", "failed-owner"]
)
async def test_native_retirement_control_cannot_be_demoted_to_observer_withdrawal(
    native_snapshot, tmp_path, monkeypatch, boundary
):
    from kestrel_sovereign._async_ownership import await_owned_task

    (
        db,
        target,
        service,
        entered,
        close_entered,
        release_close,
        clients,
        source_errors,
        close_tasks,
        state,
        tasks,
    ) = native_snapshot
    agent, usage_db = await _partial_native_agent(tmp_path, monkeypatch)
    agent._sync_service = service
    agent._resource_custody().acquire("sync_service")
    state["release_server"].set()
    release_close.set()
    forbidden = []
    original = state["close_error"]

    def forbid_serving_release():
        forbidden.append("serving-release")
        raise AssertionError("native retirement control reached serving-record release")

    async def forbid_replacement(*args):
        forbidden.append("replacement")
        raise AssertionError(
            "unresolved native retirement admitted a replacement cleanup"
        )

    try:
        await agent._run_durable_shutdown_tail(5)
        assert close_tasks and all(task.done() for task in close_tasks)
        assert original in tuple(_explicit_control_chain(agent._shutdown_control))
        monkeypatch.setattr(agent, "_release_serving_record", forbid_serving_release)
        if boundary == "physical-epilogue":
            operation = create_owned_task(
                agent._complete_durable_shutdown_continuation(None, None, None),
                retain_originals=True,
            )
            tasks.append(operation)
            failure = (await await_owned_task(operation)).error
        else:
            observer_entered = asyncio.Event()

            async def original_observer():
                observer_entered.set()
                if boundary == "failed-owner":
                    raise RuntimeError("declared original cleanup observer failure")
                await asyncio.Event().wait()

            observer = create_owned_task(original_observer(), retain_originals=True)
            tasks.append(observer)
            await asyncio.wait_for(observer_entered.wait(), 3)
            if boundary == "cancelled-owner":
                observer.cancel("withdraw cleanup observer, not native debt")
            await asyncio.gather(observer, return_exceptions=True)
            agent._durable_shutdown_continuation = observer
            monkeypatch.setattr(
                agent, "_ensure_durable_shutdown_continuation", forbid_replacement
            )
            failure = None
            try:
                await agent.wait_for_shutdown_completion()
            except BaseException as error:
                failure = error
        assert original in tuple(_explicit_control_chain(failure)), (
            "native control was classified as observer-only withdrawal"
        )
        assert execution_cleanup_debts(failure) and not forbidden
    finally:
        await _retire_partial_agent_fixture(agent, usage_db)
