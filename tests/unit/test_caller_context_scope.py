"""Authenticated caller authority stays task-local and clears on signal turns."""

import asyncio

import pytest

from kestrel_sovereign.agent.invocation import (
    bind_async_generator_invocation,
    bind_async_invocation,
)
from kestrel_sovereign.auth import (
    CallerContext,
    caller_context_scope,
    current_caller_context,
)


@bind_async_invocation("invocation_id")
async def _turn(*, caller=None, invocation_id=None):
    return current_caller_context()


@bind_async_generator_invocation("request_id")
async def _stream(*, caller=None, request_id=None):
    yield current_caller_context()


@pytest.mark.asyncio
async def test_async_invocation_binds_and_restores_endpoint_caller():
    sovereign = CallerContext.sovereign(identity="operator")

    assert current_caller_context() is None
    assert await _turn(caller=sovereign) is sovereign
    assert current_caller_context() is None


@pytest.mark.asyncio
async def test_absent_caller_clears_inherited_authority_for_unattended_turn():
    sovereign = CallerContext.sovereign(identity="operator")

    with caller_context_scope(sovereign):
        assert current_caller_context() is sovereign
        assert await _turn(caller=None) is None
        assert [item async for item in _stream(caller=None)] == [None]
        assert current_caller_context() is sovereign

    assert current_caller_context() is None


@pytest.mark.asyncio
async def test_detached_task_cannot_retain_caller_after_scope_exit():
    """A copied ContextVar must not outlive the endpoint-owned auth scope."""

    sovereign = CallerContext.sovereign(identity="operator")
    release = asyncio.Event()

    async def detached_reader():
        await release.wait()
        return current_caller_context()

    with caller_context_scope(sovereign):
        task = asyncio.create_task(detached_reader())
        assert current_caller_context() is sovereign

    release.set()
    assert await task is None
