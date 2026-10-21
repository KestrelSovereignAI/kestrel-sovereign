"""Supported refund admission seams retain actual native control, not funds."""
import asyncio
import inspect

import pytest

from kestrel_sovereign._async_ownership import await_owned_task, create_owned_task
from kestrel_sovereign.execution_custody import (
    _explicit_control_chain, bind_execution_custody, execution_commit_outcome,
    is_execution_control_error,
)
from kestrel_sovereign.multi_agent.agent_manager import AgentManager
from tests.integration.test_execution_authority_postgres import GenerationFence, native_pg as native_pg
from tests.integration.test_execution_custody_review30_pg import lose_ack


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["synchronous", "admission"])
async def test_refund_admission_retains_original_control_bearing_cancellation(native_pg, tmp_path, monkeypatch, mode):
    backend, observer, schema = native_pg
    entered, release = asyncio.Event(), asyncio.Event()

    async def committed_source():
        try:
            with bind_execution_custody(GenerationFence()):
                await backend.execute("INSERT INTO effects VALUES (197, 'native admission evidence')")
        except BaseException as error:
            asyncio.current_task().cancel("actual original refund admission cancellation")
            try:
                await asyncio.sleep(0)
            except asyncio.CancelledError as cancellation:
                raise cancellation from error

    with monkeypatch.context() as ack:
        lose_ack(ack, cancelled=False, entered=entered, release=release)
        source = create_owned_task(committed_source(), retain_originals=True)
        try:
            await asyncio.wait_for(entered.wait(), 5)
            release.set()
            original = (await await_owned_task(source)).error
        finally:
            release.set()
            await await_owned_task(source)
    assert isinstance(original, asyncio.CancelledError) and execution_commit_outcome(original) == "unknown"
    source_graph = tuple(_explicit_control_chain(original))
    manager = AgentManager(base_data_dir=tmp_path)
    loop, previous = asyncio.get_running_loop(), asyncio.get_running_loop().get_task_factory()
    refused = []

    async def release_budget():
        await backend.execute("INSERT INTO effects VALUES (198, 'unaccepted refund must not execute')")

    def release_seam(_child):
        if mode == "synchronous":
            raise original
        return release_budget()

    def factory(loop, coroutine, **kwargs):
        frame = getattr(coroutine, "cr_frame", None)
        actual = frame.f_locals.get("operation", coroutine) if frame else coroutine
        if getattr(actual, "cr_code", None) is release_budget.__code__:
            refused.append((coroutine, actual))
            raise original
        return previous(loop, coroutine, **kwargs) if previous else asyncio.Task(coroutine, loop=loop, **kwargs)

    record = None
    try:
        monkeypatch.setattr(manager, "_release_child_budget_cancellation_safe", release_seam)
        loop.set_task_factory(factory)
        record = manager._start_child_budget_release("unpublished-child")
        loop.set_task_factory(previous)
        outcome = await await_owned_task(create_owned_task(
            manager._await_child_budget_release_cancellation_safe(record.task), retain_originals=True,
        ))
        chain = tuple(_explicit_control_chain(outcome.error))
        assert original in chain, "refund admission replaced its original cancellation"
        assert is_execution_control_error(manager._quarantined_shutdown_control), "refund admission erased its native unsafe-control latch"
        assert execution_commit_outcome(outcome.error) == "unknown"
        assert tuple(_explicit_control_chain(original)) == source_graph
        assert record.task.cancelled(), "legacy cancelled-Future compatibility changed"
        if mode == "admission":
            assert refused and all(inspect.getcoroutinestate(coro) == inspect.CORO_CLOSED for pair in refused for coro in pair)
        assert await observer.fetchval(f'SELECT count(*) FROM "{schema}".effects WHERE id=197') == 1
        assert await observer.fetchval(f'SELECT count(*) FROM "{schema}".effects WHERE id=198') == 0
    finally:
        loop.set_task_factory(previous)
        for pair in refused:
            for coroutine in pair:
                coroutine.close()
        if record is not None:
            await await_owned_task(record.task)
