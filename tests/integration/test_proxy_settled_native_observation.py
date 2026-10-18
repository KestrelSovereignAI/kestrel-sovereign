"""A real settled COMMIT observer is finite history, not failed retirement."""
import asyncio

import asyncpg
import pytest

from kestrel_sovereign._async_ownership import await_owned_task, create_owned_task
from kestrel_sovereign.execution_custody import (
    ExecutionCustody, ProcessRuntimeExecutionFence, bind_execution_custody_snapshot,
    execution_denial_scopes, execution_commit_outcome,
)
from kestrel_sovereign.features import isolated_runtime
from tests.integration.test_completion_observer_consumers import _publication_task_type
from tests.integration.test_execution_authority_postgres import native_pg as native_pg
from tests.unit.test_host_context_native_cleanup import _actual_operation
from tests.unit.test_isolated_feature_runtime import FakeIsolatedClient, _initialized_host_ingress_proxy
from tests.utils.original_failure import OriginalFailure


@pytest.mark.asyncio
@pytest.mark.parametrize("remove", [False, True])
async def test_proxy_settled_native_ack_observation_is_not_permanent_retirement_debt(
    native_pg, tmp_path, monkeypatch, remove,
):
    backend, observer, schema = native_pg
    source_owner = None
    seen = []
    native_exit = asyncpg.transaction.Transaction.__aexit__
    async def committed(transaction, error_type, error, tb):
        result = await native_exit(transaction, error_type, error, tb)
        if error is None and not transaction._nested and not seen:
            seen.append(transaction)
            source_owner.cancel("actual acknowledged native observer withdrew")
            await asyncio.sleep(0)
        return result
    async def original():
        escaped = None
        with bind_execution_custody_snapshot((ExecutionCustody(ProcessRuntimeExecutionFence("original-settled-proxy-ack")),)):
            async with backend.operational_session():
                try:
                    await backend.execute("INSERT INTO effects VALUES (246, 'actual acknowledged native effect')")
                except asyncio.CancelledError as error:
                    escaped = error
            assert escaped is not None and execution_denial_scopes(escaped) == ()
            return escaped
    with monkeypatch.context() as fault:
        fault.setattr(asyncpg.transaction.Transaction, "__aexit__", committed)
        source_owner = create_owned_task(original(), retain_originals=True)
        source_outcome = await await_owned_task(source_owner)
    assert source_outcome.error is None and execution_commit_outcome(source_outcome.result) == "committed"
    source = OriginalFailure.capture(source_outcome.result)
    assert await observer.fetchval(f'SELECT count(*) FROM "{schema}".effects WHERE id=246') == 1
    feature, _ = await _initialized_host_ingress_proxy(monkeypatch, tmp_path, FakeIsolatedClient)
    loop, previous = asyncio.get_running_loop(), asyncio.get_running_loop().get_task_factory()
    accepted = []
    task_type = _publication_task_type(source.error, remove=remove, after=True)
    def factory(loop, coroutine, **kwargs):
        operation = _actual_operation(coroutine)
        if getattr(operation, "cr_code", None) is isolated_runtime.ProxyFeature._terminal_cleanup_uninterrupted.__code__:
            task = task_type(coroutine, loop=loop, **kwargs)
            accepted.append(task)
            return task
        return previous(loop, coroutine, **kwargs) if previous else asyncio.Task(coroutine, loop=loop, **kwargs)
    try:
        loop.set_task_factory(factory)
        first = await await_owned_task(create_owned_task(feature.shutdown(), retain_originals=True))
        loop.set_task_factory(previous)
        source.assert_retained(first.error)
        assert accepted and all(task.done() for task in accepted)
        assert feature._terminal_cleanup_task is None
        assert vars(feature).get("_event_ingress_control") is None
        later = await await_owned_task(create_owned_task(feature.shutdown(), retain_originals=True))
        assert later.error is None and later.cancellation is None
        await feature.initialize()
        assert feature._client is not None
    finally:
        loop.set_task_factory(previous)
        await feature.shutdown()
