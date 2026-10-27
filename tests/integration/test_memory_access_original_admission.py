"""Real loop admission and accepted native memory write owners, no fake Tasks."""

import asyncio
import inspect

import pytest

from kestrel_sovereign._async_ownership import await_owned_task
from kestrel_sovereign.agent.context_manager import ContextManager
from kestrel_sovereign.agent.context_stages import (
    ContextAssembly,
    ContextBuildMode,
    ContextBuildPlan,
)
from tests.integration.test_execution_authority_postgres import native_pg as native_pg
from tests.integration.test_saved_items_original_control import (
    native_recall as native_recall,
)
from tests.unit.test_host_context_native_cleanup import _actual_operation
from tests.shared.original_task_admission import assert_unknown_task_admission


@pytest.mark.asyncio
@pytest.mark.parametrize("door", ["record-accesses", "context-plan"])
async def test_partial_memory_task_admission_joins_accepted_children_and_fails_closed(
    native_recall, monkeypatch, door
):
    backend, storage, memory, agent, feature, store = native_recall
    rows = []
    for number in range(2):
        rows.append(
            await storage.db.fetchval(
                "INSERT INTO conversation_history (agent_id,role,content,metadata) VALUES (?,?,?,?) RETURNING id",
                (agent.did, "assistant", f"actual rehearsal {number}", "{}"),
            )
        )
    manager = ContextManager(
        storage, agent_id=agent.did, memory_retriever=memory.retriever
    )
    plan = ContextBuildPlan(
        mode=ContextBuildMode.LIVE,
        model="gpt-4o",
        assembly=ContextAssembly(),
        sections={},
        budget_summary={},
        context_limit=4096,
        response_reserve=512,
        total_budget=3584,
        total_tokens=0,
        memory_access_ids=tuple(rows),
    )
    original = RuntimeError("actual second memory access Task admission refused")
    loop, previous = (
        asyncio.get_running_loop(),
        asyncio.get_running_loop().get_task_factory(),
    )
    accepted, refused = [], []

    def task_factory(loop, coroutine, **kwargs):
        operation = _actual_operation(coroutine)
        code = getattr(operation, "cr_code", None)
        target = (
            code is not None and code.co_qualname == "MemoryRetriever.update_access"
        )
        if target and len(accepted) == 1:
            refused.append((coroutine, operation))
            raise original
        task = (
            previous(loop, coroutine, **kwargs)
            if previous
            else asyncio.Task(coroutine, loop=loop, **kwargs)
        )
        if target:
            accepted.append(task)
        return task

    failure = result = None
    try:
        loop.set_task_factory(task_factory)
        try:
            if door == "context-plan":
                result = await manager._commit_and_render_plan(plan)
            else:
                result = await memory.retriever.record_accesses(rows, agent.did)
        except BaseException as error:
            failure = error
        loop.set_task_factory(previous)
        assert len(accepted) == 1 and len(refused) == 1
        assert result is None
        assert_unknown_task_admission(failure, original)
        assert all(task.done() for task in accepted), (
            "refusal abandoned an accepted child"
        )
        assert all(
            inspect.getcoroutinestate(coro) == inspect.CORO_CLOSED
            for pair in refused
            for coro in pair
        ), "refused original coroutine was not closed"
    finally:
        loop.set_task_factory(previous)
        for task in accepted:
            if not task.done():
                task.cancel()
            await await_owned_task(task)
        for pair in refused:
            for coroutine in pair:
                coroutine.close()


@pytest.mark.asyncio
@pytest.mark.skipif(
    not hasattr(asyncio, "eager_task_factory"),
    reason="native eager Task execution requires Python 3.12+",
)
async def test_eager_native_admission_denial_is_harvested_before_context_returns(
    native_recall, monkeypatch
):
    backend, storage, memory, agent, feature, store = native_recall
    row = await storage.db.fetchval(
        "INSERT INTO conversation_history (agent_id,role,content,metadata) VALUES (?,?,?,?) RETURNING id",
        (agent.did, "assistant", "original eager denial", "{}"),
    )
    scope = backend._execution_custody
    scope.revoke("declared original lifetime revoked before eager rehearsal")
    actual, observed = scope.require_work, []

    def original_scope_check():
        try:
            return actual()
        except BaseException as error:
            observed.append(error)
            raise

    monkeypatch.setattr(scope, "require_work", original_scope_check)
    loop, previous = (
        asyncio.get_running_loop(),
        asyncio.get_running_loop().get_task_factory(),
    )
    failure = None
    try:
        # Native Python eager Task execution can settle before any done
        # callback is delivered. No fake Task/result or native I/O receipt.
        loop.set_task_factory(asyncio.eager_task_factory)
        try:
            await memory.retriever.record_accesses([row], agent.did)
        except BaseException as error:
            failure = error
    finally:
        loop.set_task_factory(previous)
    assert observed, "the original typed authority gate was not reached"
    from kestrel_sovereign.execution_custody import (
        _explicit_control_chain,
        is_execution_control_error,
    )

    assert failure is not None and observed[0] in tuple(
        _explicit_control_chain(failure)
    ), "eager native owner returned before harvesting its original denial"
    assert is_execution_control_error(failure)
    assert not memory.retriever._access_update_tasks
