"""Actual join dependencies are distinct from creation/context ancestry."""

import asyncio

import pytest

from kestrel_sovereign._async_ownership import (
    await_owned_task,
    create_owned_task,
    owned_task_join_ancestors,
    owned_task_join_dependency,
)


@pytest.mark.asyncio
async def test_join_dependencies_count_nested_original_waiters():
    release = asyncio.Event()
    target = create_owned_task(release.wait())
    current = asyncio.current_task()
    try:
        with owned_task_join_dependency(target):
            with owned_task_join_dependency(target):
                assert owned_task_join_ancestors(target) == {current}
                assert target._kestrel_owned_join_waiters[current] == 2
            assert target._kestrel_owned_join_waiters[current] == 1
            assert owned_task_join_ancestors(target) == {current}
        assert not target._kestrel_owned_join_waiters
    finally:
        release.set()
        assert (await await_owned_task(target)).error is None


@pytest.mark.asyncio
async def test_completion_future_dependency_ends_before_reusable_host_task():
    release, host_release, entered = asyncio.Event(), asyncio.Event(), asyncio.Event()
    completion = asyncio.get_running_loop().create_future()
    seen = []

    async def original_child():
        await release.wait()
        seen.append(owned_task_join_ancestors())

    child = create_owned_task(original_child())

    async def reusable_host():
        with owned_task_join_dependency(asyncio.current_task(), waiter=completion):
            assert (await await_owned_task(child)).error is None
        completion.set_result(None)
        entered.set()
        await host_release.wait()

    host = create_owned_task(reusable_host())
    observer = create_owned_task(await_owned_task(completion))
    try:
        await asyncio.sleep(0)
        release.set()
        await entered.wait()
        assert (await await_owned_task(observer)).error is None
        assert seen == [{host, completion, observer}]
        assert not host.done() and not owned_task_join_ancestors(host)
        assert not getattr(child, "_kestrel_owned_join_waiters", ())
    finally:
        release.set()
        host_release.set()
        for task in (child, host, observer):
            await await_owned_task(task)
