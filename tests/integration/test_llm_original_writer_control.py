"""Bounded unpublished cleanup preserves original native writer control."""

import asyncio
import inspect

import asyncpg
import pytest

from kestrel_sovereign._async_ownership import await_owned_task
from kestrel_sovereign.execution_custody import (
    ExecutionCustody,
    execution_commit_outcome,
    is_execution_control_error,
)
from kestrel_sovereign.llm.lifecycle import retire_unpublished_llm
from tests.integration.test_execution_authority_postgres import GenerationFence
from tests.integration.test_execution_authority_postgres import native_pg as native_pg
from tests.unit.test_host_context_native_cleanup import _actual_operation


@pytest.mark.asyncio
async def test_default_unpublished_close_keeps_original_failed_writer_control(
    native_pg, tmp_path, monkeypatch
):
    from kestrel_sovereign.llm import service as service_module

    backend, observer, schema = native_pg
    monkeypatch.setattr(service_module, "load_section", lambda _name: {})
    monkeypatch.setattr(service_module, "CLIENT_CLOSE_TIMEOUT", 5.0)
    for name in ("DATABASE_URL", "KESTREL_DATABASE_URL"):
        monkeypatch.delenv(name, raising=False)
    service = service_module.LLMService(agent_data_dir=tmp_path)
    await service._ensure_db_initialized()
    usage = service._usage_db
    assert service._usage_db_owned and usage.backend_type == "sqlite"
    backend._execution_custody = ExecutionCustody(GenerationFence())
    native_entered, fallback_entered, release = (
        asyncio.Event(),
        asyncio.Event(),
        asyncio.Event(),
    )
    native_exit, commits = asyncpg.transaction.Transaction.__aexit__, []
    original = OSError("actual original preference writer COMMIT ACK lost")

    async def held_actual_ack(transaction, error_type, error, tb):
        result = await native_exit(transaction, error_type, error, tb)
        if error_type is None and not transaction._nested and not commits:
            commits.append(True)
            native_entered.set()
            await release.wait()
            raise original
        return result

    monkeypatch.setattr(asyncpg.transaction.Transaction, "__aexit__", held_actual_ack)

    async def persist(model, vendor, route):
        await backend.execute(
            "INSERT INTO effects VALUES (?, ?)", (137, "original preference write")
        )

    service.set_preference_persistence_callback(persist)
    service.clear_model_preference()
    (writer,) = tuple(service._preference_persistence_tasks)
    real_join = service_module.await_owned_task

    async def observe_original_join(task, *args, **kwargs):
        if task is writer:
            fallback_entered.set()
        return await real_join(task, *args, **kwargs)

    monkeypatch.setattr(service_module, "await_owned_task", observe_original_join)
    loop, previous, refused = (
        asyncio.get_running_loop(),
        asyncio.get_running_loop().get_task_factory(),
        [],
    )
    admission_error = RuntimeError("original preference drain admission refused")

    def factory(loop, coroutine, **kwargs):
        operation = _actual_operation(coroutine)
        code = getattr(operation, "cr_code", None)
        if (
            not refused
            and code is not None
            and code.co_qualname == "LLMService.drain_preference_persistence"
        ):
            refused.append((coroutine, operation))
            raise admission_error
        return (
            previous(loop, coroutine, **kwargs)
            if previous
            else asyncio.Task(coroutine, loop=loop, **kwargs)
        )

    retirement, owners = None, {}
    try:
        await asyncio.wait_for(native_entered.wait(), 5)
        loop.set_task_factory(factory)
        retirement = asyncio.create_task(
            retire_unpublished_llm(owners, "default-bounded", service, ())
        )
        await asyncio.wait_for(fallback_entered.wait(), 5)
        release.set()
        caught = None
        try:
            await asyncio.wait_for(retirement, 10)
        except BaseException as error:
            caught = error
        loop.set_task_factory(previous)
        assert (
            commits == [True]
            and await observer.fetchval(
                f'SELECT COUNT(*) FROM "{schema}".effects WHERE id=137'
            )
            == 1
        )
        assert (
            execution_commit_outcome(service._preference_persistence_control)
            == "unknown"
        )
        assert caught is not None and execution_commit_outcome(caught) == "unknown", (
            "bounded cleanup returned only factory refusal and discarded original native writer control"
        )
        assert owners["default-bounded"].provider is service
        assert (
            usage.backend._connection is None and not service.has_pending_close_owners()
        ), "original effect receipt is distinct from physically pending usage storage"
        assert refused and all(
            inspect.getcoroutinestate(coro) == inspect.CORO_CLOSED
            for pair in refused
            for coro in pair
        )
    finally:
        loop.set_task_factory(previous)
        release.set()
        await await_owned_task(writer)
        if retirement is not None:
            await await_owned_task(retirement)
        for pair in refused:
            for coroutine in pair:
                coroutine.close()
        try:
            await service.wait_for_close_completion()
        except BaseException as error:
            if not is_execution_control_error(error):
                raise
        await usage.close()
        backend._execution_custody = None
