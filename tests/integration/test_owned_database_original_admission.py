"""Refused original database retirement keeps its native commit receipt."""

import asyncio
import inspect

import asyncpg
import pytest

from kestrel_sovereign.execution_custody import (
    ExecutionCustody,
    execution_cleanup_debts,
    execution_commit_outcome,
)
from kestrel_sovereign.storage.async_database import (
    AsyncDatabase,
    DatabaseRetirementError,
    _close_owned_database,
)
from tests.integration.test_execution_authority_postgres import GenerationFence
from tests.integration.test_execution_authority_postgres import native_pg as native_pg
from tests.unit.test_host_context_native_cleanup import _actual_operation


@pytest.mark.asyncio
async def test_refused_original_database_close_keeps_actual_commit_and_owner(
    native_pg, monkeypatch
):
    backend, observer, schema = native_pg
    db = AsyncDatabase(backend)
    backend._execution_custody = ExecutionCustody(GenerationFence())
    native_exit, committed = asyncpg.transaction.Transaction.__aexit__, []
    original = OSError("original PostgreSQL COMMIT ACK lost")

    async def commit_then_lose_ack(transaction, error_type, error, tb):
        result = await native_exit(transaction, error_type, error, tb)
        if error_type is None and not transaction._nested and not committed:
            committed.append(True)
            raise original
        return result

    monkeypatch.setattr(
        asyncpg.transaction.Transaction, "__aexit__", commit_then_lose_ack
    )
    with pytest.raises(BaseException) as operation:
        await db.execute(
            "INSERT INTO effects VALUES (?, ?)",
            (140, "actual commit before closer refusal"),
        )
    assert execution_commit_outcome(operation.value) == "unknown"
    assert (
        await observer.fetchval(f'SELECT COUNT(*) FROM "{schema}".effects WHERE id=140')
        == 1
    )
    loop, previous, refused = (
        asyncio.get_running_loop(),
        asyncio.get_running_loop().get_task_factory(),
        [],
    )
    admission = RuntimeError("original database closer admission refused")

    def factory(loop, coroutine, **kwargs):
        actual = _actual_operation(coroutine)
        code, frame = (
            getattr(actual, "cr_code", None),
            getattr(actual, "cr_frame", None),
        )
        if (
            code is not None
            and code.co_qualname == "AsyncDatabase.close"
            and frame.f_locals.get("self") is db
        ):
            refused.append((coroutine, actual))
            raise admission
        return (
            previous(loop, coroutine, **kwargs)
            if previous
            else asyncio.Task(coroutine, loop=loop, **kwargs)
        )

    try:
        loop.set_task_factory(factory)
        caught = None
        try:
            await _close_owned_database(db, operation.value)
        except BaseException as error:
            caught = error
        loop.set_task_factory(previous)
        assert isinstance(caught, DatabaseRetirementError), (
            "Task refusal escaped the original database retirement contract"
        )
        assert (
            caught.retirement_owner is db and caught.operation_error is operation.value
        )
        assert execution_commit_outcome(caught) == "unknown"
        assert any(
            resource is db and owner is None
            for resource, owner in execution_cleanup_debts(caught)
        )
        assert refused and all(
            inspect.getcoroutinestate(coro) == inspect.CORO_CLOSED
            for pair in refused
            for coro in pair
        )
    finally:
        loop.set_task_factory(previous)
        for pair in refused:
            for coroutine in pair:
                coroutine.close()
        backend._execution_custody = None
        await db.close()
