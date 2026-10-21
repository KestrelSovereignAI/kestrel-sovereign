"""Actual SDK inner close doors join accepted publication-failed originals."""
import asyncio

import httpx2
import pytest

from kestrel_sovereign._async_ownership import await_owned_task, create_owned_task
from kestrel_sovereign.execution_custody import close_native_execution_stream, execution_cleanup_debts
from kestrel_sovereign.llm.anthropic_adapter import _anthropic_stream_with_retry
from tests.integration.test_anthropic_original_close_admission import native_sdk_peer as native_sdk_peer
from tests.integration.test_execution_authority_postgres import native_pg as native_pg
from tests.integration.test_maintenance_original_retirement import _original_control
from tests.unit.test_host_context_native_cleanup import _actual_operation
from tests.unit.test_cleanup_task_admission import _operation_name
from tests.utils.original_failure import OriginalFailure


@pytest.mark.asyncio
@pytest.mark.parametrize("door", ["response", "context"])
@pytest.mark.parametrize("after", [False, True, "refuse"])
async def test_inner_sdk_close_keeps_accepted_original_before_publication_failure(
    native_pg, native_sdk_peer, monkeypatch, door, after,
):
    source = OriginalFailure.capture(await _original_control(native_pg[0], monkeypatch))
    sdk, _, _ = native_sdk_peer
    accepted, fired, responses, resources = [], [], [], []
    entered, release = asyncio.Event(), asyncio.Event()
    loop, previous = asyncio.get_running_loop(), asyncio.get_running_loop().get_task_factory()
    native_close = httpx2.Response.aclose
    async def held_close(response):
        entered.set()
        await release.wait()
        await native_close(response)
    class PublicationTask(asyncio.Task):
        def add_done_callback(self, callback, *args, **kwargs):
            if not fired and getattr(callback, "__name__", "") == "close_unstarted":
                fired.append(True)
                if after:
                    super().add_done_callback(callback, *args, **kwargs)
                raise source.error
            return super().add_done_callback(callback, *args, **kwargs)
    def factory(loop, coroutine, **kwargs):
        actual = _actual_operation(coroutine)
        selected = (getattr(actual, "cr_code", None) is held_close.__code__ if door == "response"
                    else _operation_name(coroutine).endswith(".close_acquired.<locals>.owned_close"))
        if selected:
            if after == "refuse":
                fired.append(True)
                raise source.error
            task = PublicationTask(coroutine, loop=loop, **kwargs)
            accepted.append(task)
            return task
        return previous(loop, coroutine, **kwargs) if previous else asyncio.Task(coroutine, loop=loop, **kwargs)
    async def run():
        if door == "context":
            async with _anthropic_stream_with_retry(sdk,
                {"model": "local-protocol-input", "max_tokens": 1,
                 "messages": [{"role": "user", "content": "fixture"}]}, retry_open=False,
            ) as stream:
                responses.append(stream.response)
                resources.append(stream)
                loop.set_task_factory(factory)
                raise source.error
        else:
            manager = sdk.messages.stream(model="local-protocol-input", max_tokens=1,
                                           messages=[{"role": "user", "content": "fixture"}])
            stream = await manager.__aenter__()
            responses.append(stream.response)
            resources.append(stream.response)
            from kestrel_sovereign.execution_custody import retain_native_execution_stream_close
            retain_native_execution_stream_close(stream.response)
            loop.set_task_factory(factory)
            try:
                raise source.error
            finally:
                await close_native_execution_stream(stream.response)
    monkeypatch.setattr(httpx2.Response, "aclose", held_close)
    owner = create_owned_task(run(), retain_originals=True)
    try:
        outcome = await await_owned_task(owner)
        assert fired, "actual inner admission seam was not exercised"
        if after == "refuse":
            assert not accepted
            source.assert_retained(outcome.error)
            assert any(resource is resources[0] and task is None for resource, task in execution_cleanup_debts(outcome.error))
        else:
            assert len(accepted) == 1 and accepted[0].done(), "inner SDK boundary abandoned accepted publication-failed owner"
            source.assert_retained(outcome.error, resource=resources[0], task=accepted[0])
        assert not entered.is_set(), "publication-failed original ran physical close before its admission gate"
    finally:
        loop.set_task_factory(previous)
        release.set()
        await await_owned_task(owner)
        for task in accepted:
            await await_owned_task(task)
        for response in responses:
            await native_close(response)
