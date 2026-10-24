"""Original PostgreSQL control survives direct distributed close interruption."""

from tests.shared.native_shutdown_budget import shutdown_fixture_agent

import asyncio
import inspect

import asyncpg
import pytest

from kestrel_sovereign._async_ownership import await_owned_task, create_owned_task
from kestrel_sovereign.execution_custody import (
    ExecutionCustody,
    execution_cleanup_debts,
    execution_commit_outcome,
    is_execution_control_error,
)
from kestrel_sovereign.stop import (
    DistributedInvocationRegistry,
    DistributedInvocationStore,
)
from kestrel_sovereign.storage.async_database import AsyncDatabase
from tests.integration.test_execution_authority_postgres import GenerationFence
from tests.integration.test_execution_authority_postgres import native_pg as native_pg
from tests.unit.test_cleanup_task_admission import (
    _operation_name,
    _partial_native_agent,
)


@pytest.mark.asyncio
@pytest.mark.parametrize("interruption", ["withdrawal", "admission"])
async def test_direct_stop_close_retains_original_commit_and_accepted_children(
    native_pg,
    tmp_path,
    monkeypatch,
    interruption,
):
    from kestrel_sovereign.stop import invocation

    backend, observer, schema = native_pg
    db = AsyncDatabase(backend)
    store = DistributedInvocationStore(db)
    await store.ensure_schema()
    agent, agent_db = await _partial_native_agent(tmp_path, monkeypatch)
    registry = DistributedInvocationRegistry(store)
    registry.attach(agent)
    assert await registry.register(agent, "original-fenced-row", 1)
    registry._fail_closed_owner("native owner epoch retired for review")
    async with registry._registration_lock:
        await registry._reacquire_owner_lease()
    assert await registry.register(agent, "original-unknown", 1)
    assert await registry.register(agent, "original-pending", 1)
    assert await registry.register(agent, "original-current-row", 1)
    backend._execution_custody = ExecutionCustody(GenerationFence())
    native_exit = asyncpg.transaction.Transaction.__aexit__
    committed = [asyncio.Event(), asyncio.Event()]
    release = [asyncio.Event(), asyncio.Event()]
    abandon_entered, abandon_release = asyncio.Event(), asyncio.Event()
    native_abandon = store.abandon_owner
    abandonment_outcomes = []
    original = OSError("actual original Stop completion COMMIT acknowledgement lost")
    completion_owners, accepted, refused = [], [], []
    loop, factory = (
        asyncio.get_running_loop(),
        asyncio.get_running_loop().get_task_factory(),
    )
    refusal = RuntimeError("second original owner abandonment Task refused")
    close_task = None

    async def native_ack(transaction, error_type, error, tb):
        result = await native_exit(transaction, error_type, error, tb)
        current = asyncio.current_task()
        if error_type is None and not transaction._nested:
            matching = [
                index
                for index, owner in enumerate(completion_owners)
                if current.get_name()
                == owner.get_name() + ":postgres-original-transaction-exit"
            ]
            if matching:
                (index,) = matching
                if not committed[index].is_set():
                    committed[index].set()
                    await release[index].wait()
                    if index == 0:
                        raise original
        return result

    async def observed_original_abandon(owner_id):
        abandon_entered.set()
        await abandon_release.wait()
        try:
            await native_abandon(owner_id)
        except BaseException as error:
            abandonment_outcomes.append(error)
            raise
        else:
            abandonment_outcomes.append(None)

    def refuse_second(loop, coroutine, **kwargs):
        if _operation_name(coroutine).endswith("._abandon_owner_before"):
            if accepted and not refused:
                refused.append(coroutine)
                raise refusal
            task = (
                factory(loop, coroutine, **kwargs)
                if factory
                else asyncio.Task(coroutine, loop=loop, **kwargs)
            )
            accepted.append(task)
            return task
        return (
            factory(loop, coroutine, **kwargs)
            if factory
            else asyncio.Task(coroutine, loop=loop, **kwargs)
        )

    monkeypatch.setattr(asyncpg.transaction.Transaction, "__aexit__", native_ack)
    monkeypatch.setattr(invocation, "_DEFAULT_CLOSE_DRAIN_SECONDS", 0.15)
    try:
        for turn in ("original-unknown", "original-pending"):
            prior = set(registry._cleanup_tasks)
            registry.complete_soon(agent, turn, 1)
            (owner,) = set(registry._cleanup_tasks) - prior
            completion_owners.append(owner)
            index = len(completion_owners) - 1
            await asyncio.wait_for(committed[index].wait(), 5)
            if index == 0:
                release[0].set()
                first = await await_owned_task(owner)
                assert execution_commit_outcome(first.error) == "unknown"
        assert (
            await observer.fetchval(
                f'SELECT COUNT(*) FROM "{schema}".stop_active_invocations'
            )
            == 2
        )
        assert execution_commit_outcome(registry._terminal_control) == "unknown"
        if interruption == "admission":
            release[1].set()
            assert (await await_owned_task(completion_owners[1])).error is None
            monkeypatch.setattr(store, "abandon_owner", observed_original_abandon)
            loop.set_task_factory(refuse_second)
        close_operation = registry.close()
        close_task = create_owned_task(close_operation)
        if interruption == "withdrawal":
            async with asyncio.timeout(5):
                while (
                    close_operation.cr_frame is not None
                    and close_operation.cr_frame.f_locals.get("label") != "completion"
                ):
                    await asyncio.sleep(0)
            assert close_operation.cr_frame is not None and not close_task.done()
            close_task.cancel("direct public close observer withdrew")
            result = await await_owned_task(close_task)
            assert (
                not completion_owners[1].done()
                and completion_owners[1].cancelling() == 0
            )
            assert (
                result.error is not None
                and execution_commit_outcome(result.error) == "unknown"
            ), "direct close withdrawal replaced original native COMMIT UNKNOWN"
            assert any(
                resource is registry and closer is completion_owners[1]
                for resource, closer in execution_cleanup_debts(result.error)
            )
        else:
            async with asyncio.timeout(5):
                while not refused:
                    await asyncio.sleep(0)
            await asyncio.wait_for(abandon_entered.wait(), 5)
            assert not close_task.done(), (
                "partial abandonment admission returned before its original accepted native producer joined"
            )
            abandon_release.set()
            result = await await_owned_task(close_task)
            assert (
                result.error is not None
                and execution_commit_outcome(result.error) == "unknown"
            )
            assert all(task.done() for task in accepted)
            assert all(task.cancelling() == 0 for task in accepted)
            assert (registry, None) in execution_cleanup_debts(result.error)
            # The original backend still carries its actual UNKNOWN. Joining
            # its accepted producer cannot grant another write through that
            # loss. Assert fail-closed native entry, not a fabricated success.
            assert len(abandonment_outcomes) == 1
            assert execution_commit_outcome(abandonment_outcomes[0]) == "unknown"
            assert (
                await observer.fetchval(
                    f'SELECT COUNT(*) FROM "{schema}".stop_unresolved_invocations WHERE owner_id=$1',
                    registry._owner_id,
                )
                == 0
            )
            assert (
                await observer.fetchval(
                    f'SELECT COUNT(*) FROM "{schema}".stop_active_invocations WHERE owner_id=$1',
                    registry._owner_id,
                )
                == 1
            )
            assert all(
                inspect.getcoroutinestate(coro) == inspect.CORO_CLOSED
                for coro in refused
            )
        assert not agent._runtime_publication_ready.is_set()
    finally:
        loop.set_task_factory(factory)
        monkeypatch.setattr(store, "abandon_owner", native_abandon)
        for event in release:
            event.set()
        abandon_release.set()
        if close_task is not None:
            await await_owned_task(close_task)
        for task in completion_owners + accepted:
            await await_owned_task(task)
        for coroutine in refused:
            coroutine.close()
        try:
            await registry.close()
        except BaseException as error:
            if not is_execution_control_error(error):
                raise
        backend._execution_custody = None
        await shutdown_fixture_agent(agent)
        await agent_db.close()
        await db.close()
