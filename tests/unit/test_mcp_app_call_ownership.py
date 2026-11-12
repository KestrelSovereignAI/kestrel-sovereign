"""The host owns the exact View child, its privacy reentry and physical join."""

import asyncio

import pytest

from kestrel_sdk.resources import ResourceConflict


@pytest.mark.asyncio
async def test_app_call_enters_the_original_native_privacy_span(native_tool_effect_agent):
    agent = await native_tool_effect_agent(did='did:test:owned-mcp-app')
    lock = agent._get_privacy_transition_lock()
    observed = []
    async with lock:
        original = lock.current_native_scope()

        async def operation():
            assert lock.current_native_scope() is original, 'App task lost its original native admission'
            async with lock:
                assert lock.current_native_scope() is original
                observed.append(asyncio.current_task())
                return 'native-result'

        async with agent.mcp_app_call_scope(operation, name='owned-app-call') as owner:
            assert owner.task is not asyncio.current_task()
            assert await owner.wait() == 'native-result'
            assert observed == [owner.task]
        assert owner.task.done()
        with pytest.raises(ResourceConflict, match='ownership ended'):
            await owner.wait()
    assert not lock.locked()


@pytest.mark.asyncio
async def test_app_call_preserves_the_original_failure(native_tool_effect_agent):
    agent = await native_tool_effect_agent(did='did:test:owned-mcp-app')
    failure = ValueError('original native App failure')

    async def operation():
        raise failure

    with pytest.raises(ValueError, match='original native App failure') as observed:
        async with agent.mcp_app_call_scope(operation, name='failed-app-call') as owner:
            await owner.wait()
    assert observed.value is failure
    assert owner.task.done()


@pytest.mark.asyncio
@pytest.mark.parametrize('completed', [False, True])
async def test_app_call_refuses_a_borrowed_task_without_cancelling_it(native_tool_effect_agent, completed):
    from kestrel_sovereign.callbacks import HostCallbackOwnershipError

    agent = await native_tool_effect_agent(did='did:test:owned-mcp-app')
    release = asyncio.Event()
    borrowed = asyncio.create_task(release.wait())
    if completed:
        release.set()
        await borrowed
    try:
        with pytest.raises(HostCallbackOwnershipError):
            async with agent.mcp_app_call_scope(lambda: borrowed, name='borrowed-app-call') as owner:
                await owner.wait()
        assert borrowed.done() is completed and borrowed.cancelling() == 0
    finally:
        release.set()
        await borrowed


@pytest.mark.asyncio
async def test_app_call_joins_its_original_epilogue_under_repeated_cancellation(native_tool_effect_agent):
    agent = await native_tool_effect_agent(did='did:test:owned-mcp-app')
    entered, cancelled, settle = asyncio.Event(), asyncio.Event(), asyncio.Event()
    admitted = []
    cancellations = []

    async def operation():
        entered.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancellations.append(True)
            cancelled.set()
            await settle.wait()
            raise

    async def run():
        async with agent._get_privacy_transition_lock():
            async with agent.mcp_app_call_scope(operation, name='cancelled-app-call') as owner:
                admitted.append(owner.task)
                await owner.wait()

    caller = asyncio.create_task(run())
    try:
        await asyncio.wait_for(entered.wait(), 5)
        caller.cancel()
        await asyncio.wait_for(cancelled.wait(), 5)
        caller.cancel()
        await asyncio.sleep(0)
        assert not caller.done() and not admitted[0].done() and cancellations == [True], (
            'The original App child lost physical custody or was cancelled twice')
        settle.set()
        result, = await asyncio.gather(caller, return_exceptions=True)
        assert isinstance(result, asyncio.CancelledError)
        assert admitted[0].done() and cancellations == [True]
    finally:
        settle.set()
        await asyncio.gather(caller, return_exceptions=True)
