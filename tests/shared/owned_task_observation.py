"""Refuse one real owned-task observation allocation at its original owner."""

import asyncio
import inspect
import sys

from kestrel_sovereign._async_ownership import await_owned_task


def refuse_owned_observation_allocation(monkeypatch, owner, *, allocation, error):
    """Keep allocation fault injection shared across all ownership consumers."""
    assert allocation in {'completion', 'observation'}
    refused = asyncio.Event()
    loop = asyncio.get_running_loop()
    original = loop.create_future
    owner_code = inspect.unwrap(await_owned_task).__code__
    allocation_code = (owner_code if allocation == 'completion'
        else asyncio.tasks._wait.__code__)

    def allocate():
        frame = sys._getframe(1)
        try:
            if not refused.is_set() and frame.f_code is allocation_code:
                while frame is not None and frame.f_code is not owner_code:
                    frame = frame.f_back
                target = owner()
                if target is not None and frame is not None and frame.f_locals.get('task') is target:
                    refused.set()
                    raise error
            return original()
        finally:
            del frame

    monkeypatch.setattr(loop, 'create_future', allocate)
    return refused


def observe_bounded_asyncio_wait(monkeypatch, before_wait):
    """Observe bounded resource joins while preserving the full wait API."""
    original = asyncio.wait

    async def wait(owners, *, timeout=None, return_when=asyncio.ALL_COMPLETED):
        if timeout is not None:
            await before_wait(owners, timeout)
        return await original(owners, timeout=timeout, return_when=return_when)

    monkeypatch.setattr(asyncio, 'wait', wait)
