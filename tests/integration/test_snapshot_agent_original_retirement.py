"""Actual partial Agent tail + real loopback snapshot; not Agent READY acceptance."""

import asyncio

import pytest

from kestrel_sovereign._async_ownership import create_owned_task, owned_task_error
from kestrel_sovereign.execution_custody import (
    _explicit_control_chain,
    execution_cleanup_debts,
    is_execution_control_error,
)
from tests.integration.test_lighthouse_snapshot_original_control import (
    native_snapshot as native_snapshot,
)
from tests.unit.test_cleanup_task_admission import _partial_native_agent


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
        agent._run_durable_shutdown_tail(0.2), retain_originals=True
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

            async def observe_original_join(owner, *args, **kwargs):
                # Assert the original dispatch before it could re-join a done
                # owner forever. No fake completion/ACK is supplied.
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
        await agent.llm_service.close()
        await usage_db.close()


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
        agent._run_durable_shutdown_tail(0.3), retain_originals=True
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
        await agent.llm_service.close()
        await usage_db.close()


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
        agent._run_durable_shutdown_tail(0.2), retain_originals=True
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
        await agent.llm_service.close()
        await usage_db.close()


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
        with pytest.raises(RuntimeError) as result:
            await agent._run_durable_shutdown_tail(0.2)
        assert result.value is refusal and refused
        assert all(getcoroutinestate(coro) == CORO_CLOSED for coro in refused)
        assert not clients and not entered.is_set()
        assert execution_cleanup_debts(refusal) == ((service, None),)
        assert vars(agent).get("_shutdown_control") is refusal
        with pytest.raises(RuntimeError) as completion:
            await agent.wait_for_shutdown_completion()
        assert completion.value is refusal, (
            "unstarted native debt was replaced or acknowledged"
        )
    finally:
        loop.set_task_factory(factory)
        await agent.llm_service.close()
        await usage_db.close()


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
        agent._run_durable_shutdown_tail(0.2), retain_originals=True
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
        await agent.llm_service.close()
        await usage_db.close()
