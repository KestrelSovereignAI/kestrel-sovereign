"""An inherited, pre-observer operational lease cannot bypass eager admission."""
import asyncio

import asyncpg
import pytest

from kestrel_sovereign._async_ownership import await_owned_task, create_owned_task
from kestrel_sovereign.execution_custody import ExecutionAuthorityError, _explicit_control_chain
from kestrel_sovereign.native_retirement import NativeExecutionObserver, bind_native_execution_observer
from tests.integration.test_execution_authority_postgres import native_pg as native_pg


@pytest.mark.asyncio
@pytest.mark.parametrize("nested", [False, True])
async def test_eager_transaction_rejects_preexisting_unenrolled_lease_before_begin(native_pg, monkeypatch, nested):
    backend, observer, schema = native_pg
    begins, observations = [], []
    actual_begin = asyncpg.transaction.Transaction.__aenter__
    async def begin(transaction):
        begins.append((transaction, asyncio.current_task()))
        return await actual_begin(transaction)
    def eager_child(group, operation):
        loop = asyncio.get_running_loop()
        previous = loop.get_task_factory()
        loop.set_task_factory(asyncio.eager_task_factory)
        try:
            return group.create_task(operation)
        finally:
            loop.set_task_factory(previous)
    async def child():
        async with backend.transaction():
            await backend.execute("INSERT INTO effects VALUES (315, 'must not enter borrowed transaction')")
    async def parent():
        observation = NativeExecutionObserver(owner_task=asyncio.current_task(), trusted_phases_only=True)
        observations.append(observation)
        with bind_native_execution_observer(observation):
            async def inner():
                async with asyncio.TaskGroup() as group:
                    eager_child(group, child())
            async with asyncio.TaskGroup() as group:
                eager_child(group, inner() if nested else child())
    # Real lease is acquired BEFORE this consumer's observer exists. Inherited
    # context is not enrollment and may not retroactively adopt its connection.
    async with backend.operational_session():
        with monkeypatch.context() as fault:
            fault.setattr(asyncpg.transaction.Transaction, "__aenter__", begin)
            owner = create_owned_task(parent(), retain_originals=True)
            result = await await_owned_task(owner)
        assert any(isinstance(error, ExecutionAuthorityError) for error in _explicit_control_chain(result.error))
        assert not begins, "eager inherited lease entered native BEGIN before authenticating observer enrollment"
        assert observations and not observations[0]._resources and not observations[0].native_failures
    assert await observer.fetchval(f'SELECT count(*) FROM "{schema}".effects WHERE id=315') == 0
