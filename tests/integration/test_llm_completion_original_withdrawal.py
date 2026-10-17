"""Repeated observer withdrawal cannot erase original writer COMMIT control."""

import asyncio

import asyncpg
import pytest

from kestrel_sovereign._async_ownership import await_owned_task
from kestrel_sovereign.execution_custody import (
    ExecutionCustody,
    execution_commit_outcome,
    is_execution_control_error,
)
from tests.integration.test_execution_authority_postgres import GenerationFence
from tests.integration.test_execution_authority_postgres import native_pg as native_pg


@pytest.mark.asyncio
async def test_original_writer_control_survives_second_completion_checkpoint_cancel(
    native_pg, tmp_path, monkeypatch
):
    from kestrel_sovereign.llm import service as service_module

    backend, observer, schema = native_pg
    monkeypatch.setattr(service_module, "load_section", lambda _name: {})
    service = service_module.LLMService(agent_data_dir=tmp_path)
    backend._execution_custody = ExecutionCustody(GenerationFence())
    native_exit = asyncpg.transaction.Transaction.__aexit__
    committed, joined, release = asyncio.Event(), asyncio.Event(), asyncio.Event()
    commits, withdrawals = [], []
    original = OSError("original writer actual COMMIT ACK lost")

    async def native_ack(transaction, error_type, error, tb):
        result = await native_exit(transaction, error_type, error, tb)
        if error_type is None and not transaction._nested and not commits:
            commits.append(True)
            committed.set()
            await release.wait()
            raise original
        return result

    monkeypatch.setattr(asyncpg.transaction.Transaction, "__aexit__", native_ack)

    async def persist(model, vendor, route):
        await backend.execute(
            "INSERT INTO effects VALUES (?, ?)",
            (141, "original repeated observer cancellation"),
        )

    service.set_preference_persistence_callback(persist)
    service.clear_model_preference()
    (writer,) = tuple(service._preference_persistence_tasks)
    original_join, original_sleep = service_module.await_owned_task, asyncio.sleep
    waiter = None

    async def observed_join(task, *args, **kwargs):
        if task is writer and asyncio.current_task() is waiter:
            joined.set()
        return await original_join(task, *args, **kwargs)

    async def observed_checkpoint(delay, *args, **kwargs):
        if (
            delay == 0
            and asyncio.current_task() is waiter
            and writer.done()
            and withdrawals == ["first"]
        ):
            withdrawals.append("second")
            waiter.cancel()
        return await original_sleep(delay, *args, **kwargs)

    monkeypatch.setattr(service_module, "await_owned_task", observed_join)
    monkeypatch.setattr(asyncio, "sleep", observed_checkpoint)
    try:
        await asyncio.wait_for(committed.wait(), 5)
        waiter = asyncio.create_task(service.wait_for_close_completion())
        await asyncio.wait_for(joined.wait(), 5)
        withdrawals.append("first")
        waiter.cancel()
        await original_sleep(0)
        assert not writer.done() and writer.cancelling() == 0
        release.set()
        result = await await_owned_task(waiter)
        assert withdrawals == ["first", "second"]
        assert (
            await observer.fetchval(
                f'SELECT COUNT(*) FROM "{schema}".effects WHERE id=141'
            )
            == 1
        )
        assert (
            execution_commit_outcome(service._preference_persistence_control)
            == "unknown"
        )
        assert (
            result.error is not None
            and execution_commit_outcome(result.error) == "unknown"
        ), "second observer withdrawal masked already harvested original writer control"
        assert writer.done() and writer.cancelling() == 0
    finally:
        release.set()
        await await_owned_task(writer)
        if waiter is not None:
            await await_owned_task(waiter)
        monkeypatch.undo()
        try:
            await service.close()
        except BaseException as error:
            if not is_execution_control_error(error):
                raise
        backend._execution_custody = None
