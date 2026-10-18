"""Actual per-thread mutex ownership survives cancellation at handoff.

No app-server or provider request is made. The real Lock.acquire Task finishes
before cancellation is delivered to the actual Codex lock adapter.
"""

import asyncio
import inspect

import pytest

from kestrel_sovereign._async_ownership import await_owned_task
from kestrel_sovereign.llm.codex_adapter import _acquire_thread_turn_lock


@pytest.mark.asyncio
async def test_codex_acquired_mutex_release_does_not_need_cleanup_task_admission():
    loop, previous = (
        asyncio.get_running_loop(),
        asyncio.get_running_loop().get_task_factory(),
    )
    lock, acquired, refused = asyncio.Lock(), [], []
    owner = None

    def factory(loop, coroutine, **kwargs):
        code = getattr(coroutine, "cr_code", None)
        frame = getattr(coroutine, "cr_frame", None)
        inner = frame.f_locals.get("operation") if frame is not None else None
        candidate = inner if inspect.iscoroutine(inner) else coroutine
        candidate_code = getattr(candidate, "cr_code", None)
        if candidate_code is not None and candidate_code.co_qualname.endswith(
            "release_abandoned"
        ):
            refused.append((coroutine, candidate))
            raise RuntimeError("abandoned-result cleanup Task allocation refused")
        task = (
            previous(loop, coroutine, **kwargs)
            if previous
            else asyncio.Task(coroutine, loop=loop, **kwargs)
        )
        if code is asyncio.Lock.acquire.__code__:
            acquired.append(task)
            task.add_done_callback(
                lambda _: owner.cancel("original acquisition observer cancelled")
            )
        return task

    try:
        loop.set_task_factory(factory)
        owner = asyncio.create_task(_acquire_thread_turn_lock(lock, None))
        result = await await_owned_task(owner)
        assert (
            len(acquired) == 1 and acquired[0].done() and acquired[0].result() is True
        )
        assert not lock.locked(), (
            "successful original lock acquisition was abandoned before the caller's release block"
        )
        assert isinstance(result.error, asyncio.CancelledError), (
            "cleanup admission replaced original cancellation"
        )
        assert not refused, "a synchronous mutex release must not need another Task"
    finally:
        loop.set_task_factory(previous)
        if owner is not None:
            await await_owned_task(owner)
        for pair in refused:
            for coroutine in pair:
                coroutine.close()
        if lock.locked():
            lock.release()
