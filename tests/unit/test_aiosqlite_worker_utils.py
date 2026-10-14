"""Tests for aiosqlite worker-lifecycle synchronization helpers."""

import asyncio
import threading

import pytest

import kestrel_sovereign.storage.db.sqlite as sqlite_module
from kestrel_sovereign.storage.db import SQLiteBackend
from tests.utils.aiosqlite_workers import (
    CloseStage,
    aiosqlite_worker,
    observe_bounded_close_stages,
    wait_for_lifecycle_checkpoint,
)


@pytest.mark.asyncio
async def test_lifecycle_checkpoint_wins_without_waiting_for_owner() -> None:
    checkpoint = asyncio.Event()
    owner_release = asyncio.Event()

    async def owner() -> None:
        await owner_release.wait()

    owner_task = asyncio.create_task(owner())
    checkpoint.set()
    try:
        await wait_for_lifecycle_checkpoint(
            checkpoint.wait(),
            owner_task,
            description="checkpoint reached",
            harness_timeout=0.1,
        )
        assert not owner_task.done()
    finally:
        owner_release.set()
        await owner_task


@pytest.mark.asyncio
async def test_lifecycle_checkpoint_rejects_premature_completion() -> None:
    async def owner() -> None:
        return None

    owner_task = asyncio.create_task(owner())
    await owner_task

    with pytest.raises(
        AssertionError,
        match="Lifecycle completed before checkpoint reached",
    ):
        await wait_for_lifecycle_checkpoint(
            asyncio.Event().wait(),
            owner_task,
            description="checkpoint reached",
            harness_timeout=0.1,
        )


@pytest.mark.asyncio
async def test_lifecycle_checkpoint_fails_closed_when_both_are_ready() -> None:
    checkpoint = asyncio.Event()

    async def owner() -> None:
        return None

    owner_task = asyncio.create_task(owner())
    checkpoint.set()
    await owner_task

    with pytest.raises(
        AssertionError,
        match="Lifecycle completed before checkpoint reached",
    ):
        await wait_for_lifecycle_checkpoint(
            checkpoint.wait(),
            owner_task,
            description="checkpoint reached",
            harness_timeout=0.1,
        )


@pytest.mark.asyncio
async def test_lifecycle_checkpoint_reports_owner_failure() -> None:
    failure = RuntimeError("owner failed")

    async def owner() -> None:
        raise failure

    owner_task = asyncio.create_task(owner())
    await asyncio.wait({owner_task})

    with pytest.raises(
        AssertionError,
        match="Lifecycle failed before checkpoint reached",
    ) as captured:
        await wait_for_lifecycle_checkpoint(
            asyncio.Event().wait(),
            owner_task,
            description="checkpoint reached",
            harness_timeout=0.1,
        )

    assert captured.value.__cause__ is failure


@pytest.mark.asyncio
async def test_lifecycle_checkpoint_reports_owner_cancellation() -> None:
    owner_task = asyncio.create_task(asyncio.Event().wait())
    owner_task.cancel()
    await asyncio.wait({owner_task})

    with pytest.raises(
        AssertionError,
        match="Lifecycle cancelled before checkpoint reached",
    ):
        await wait_for_lifecycle_checkpoint(
            asyncio.Event().wait(),
            owner_task,
            description="checkpoint reached",
            harness_timeout=0.1,
        )


@pytest.mark.asyncio
async def test_lifecycle_checkpoint_has_finite_deadlock_guard() -> None:
    owner_release = asyncio.Event()

    async def owner() -> None:
        await owner_release.wait()

    owner_task = asyncio.create_task(owner())
    try:
        with pytest.raises(
            AssertionError,
            match="made no progress within the 0.001s test-harness guard",
        ):
            await wait_for_lifecycle_checkpoint(
                asyncio.Event().wait(),
                owner_task,
                description="checkpoint reached",
                harness_timeout=0.001,
            )
    finally:
        owner_release.set()
        await owner_task


@pytest.mark.asyncio
async def test_close_stages_record_both_waits_and_the_deadline(tmp_path) -> None:
    backend = SQLiteBackend(str(tmp_path / "observed-close.db"))
    await backend.connect()
    connection = backend._connection
    assert connection is not None
    loop = asyncio.get_running_loop()

    with observe_bounded_close_stages() as stages:
        started = loop.time()
        await backend.close()
        finished = loop.time()

    await stages.reached(connection, CloseStage.CONNECTION_CLOSE)
    await stages.reached(connection, CloseStage.WORKER_EXIT)
    budget = sqlite_module._minimum_close_timeout_s()
    assert started + budget <= stages.deadlines[connection] <= finished + budget
    assert not aiosqlite_worker(connection).is_alive()


@pytest.mark.asyncio
async def test_close_stages_report_a_wait_while_it_is_armed(tmp_path) -> None:
    """``reached`` returns while the close is still parked in that wait."""
    backend = SQLiteBackend(str(tmp_path / "parked-close.db"))
    await backend.connect()
    connection = backend._connection
    assert connection is not None
    entered = threading.Event()
    release = threading.Event()

    def block() -> int:
        entered.set()
        release.wait()
        return 1

    await connection.create_function("block", 0, block)
    blocked = asyncio.create_task(connection.execute("SELECT block()"))
    close_task = None
    try:
        assert await asyncio.to_thread(entered.wait, 30.0)
        with observe_bounded_close_stages() as stages:
            close_task = asyncio.create_task(
                sqlite_module._close_aiosqlite_connection(
                    connection,
                    retained_closes=backend._retired_connection_closes,
                )
            )
            await stages.reached(connection, CloseStage.CONNECTION_CLOSE)
            assert not close_task.done()
            assert not stages.has_reached(connection, CloseStage.WORKER_EXIT)
            release.set()
            await close_task
            await stages.reached(connection, CloseStage.WORKER_EXIT)
    finally:
        release.set()
        await asyncio.gather(blocked, return_exceptions=True)
        if close_task is not None:
            await asyncio.gather(close_task, return_exceptions=True)
        await asyncio.to_thread(aiosqlite_worker(connection).join, 30.0)


def test_close_stages_restore_the_close_helpers() -> None:
    finish_close = sqlite_module._finish_aiosqlite_connection_close
    wait_for_worker_exit = sqlite_module._wait_for_aiosqlite_worker_shutdown

    with observe_bounded_close_stages():
        assert sqlite_module._finish_aiosqlite_connection_close is not finish_close

    assert sqlite_module._finish_aiosqlite_connection_close is finish_close
    assert (
        sqlite_module._wait_for_aiosqlite_worker_shutdown is wait_for_worker_exit
    )
