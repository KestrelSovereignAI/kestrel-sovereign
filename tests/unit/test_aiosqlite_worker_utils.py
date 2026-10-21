"""Tests for aiosqlite worker-lifecycle synchronization helpers."""

import asyncio
import threading
from contextlib import nullcontext

import pytest

import kestrel_sovereign.storage.db.sqlite as sqlite_module
from kestrel_sovereign.execution_custody import (
    ExecutionCommitOutcomeError,
    execution_commit_outcome,
)
from kestrel_sovereign.storage.db import SQLiteBackend
from tests.utils.aiosqlite_workers import (
    CloseStage,
    aiosqlite_worker,
    block_aiosqlite_native_close,
    delay_aiosqlite_worker_exit,
    normal_close_schedule,
    observe_bounded_close_stages,
    wait_for_lifecycle_checkpoint,
)
from tests.utils.original_failure import OriginalFailure
from tests.utils.virtual_clock import (
    HANG_GUARD_SECONDS,
    await_within_hang_guard,
    wait_for_thread_event,
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


@pytest.mark.asyncio
@pytest.mark.parametrize("stage", [CloseStage.CONNECTION_CLOSE, CloseStage.WORKER_EXIT])
async def test_normal_close_schedule_joins_the_exact_blocked_worker(tmp_path, stage):
    entered, release = threading.Event(), threading.Event()
    delayed_exit = (
        delay_aiosqlite_worker_exit(release, entered)
        if stage is CloseStage.WORKER_EXIT else nullcontext()
    )
    loop = asyncio.get_running_loop()
    real_clock = loop.time
    budget = sqlite_module.AIOSQLITE_WORKER_SHUTDOWN_TIMEOUT_S
    backend = SQLiteBackend(str(tmp_path / "normal-close.db"))
    owner = None
    with delayed_exit:
        await backend.connect()
        connection = backend._connection
        assert connection is not None
        worker = aiosqlite_worker(connection)
        if stage is CloseStage.CONNECTION_CLOSE:
            entered, release = block_aiosqlite_native_close(connection)
        try:
            with observe_bounded_close_stages() as stages:
                async with normal_close_schedule() as finish:
                    owner = asyncio.create_task(finish(
                        connection, backend.close(),
                        retained_closes=backend._retired_connection_closes,
                    ))
                    await wait_for_thread_event(entered, step=stage.value)
                    await stages.reached(connection, stage)
                    original_close = stages.close_tasks[connection]
                    assert worker.is_alive() and not owner.done()
                    if stage is CloseStage.CONNECTION_CLOSE:
                        assert not original_close.done()
                    release.set()
                    await await_within_hang_guard(owner, step="the original close completing")
                    assert original_close.done()
            assert loop.time == real_clock
            assert not worker.is_alive()
            assert not backend.connection_retirement_pending
            assert sqlite_module.AIOSQLITE_WORKER_SHUTDOWN_TIMEOUT_S == budget
        finally:
            release.set()
            if owner is not None:
                await await_within_hang_guard(
                    asyncio.gather(owner, return_exceptions=True),
                    step="joining the close schedule owner",
                )
            await asyncio.to_thread(worker.join, HANG_GUARD_SECONDS)
            assert not worker.is_alive()


@pytest.mark.asyncio
@pytest.mark.parametrize("control", [False, True])
async def test_normal_close_schedule_preserves_native_failure_and_restores_clock(
    tmp_path, monkeypatch, control,
):
    backend = SQLiteBackend(str(tmp_path / "failed-close.db"))
    await backend.connect()
    connection = backend._connection
    assert connection is not None
    worker = aiosqlite_worker(connection)
    close = connection.close
    failure = asyncio.CancelledError("native close control") if control else RuntimeError("native close failed")

    async def fail_after_real_close():
        await close()
        raise failure

    monkeypatch.setattr(connection, "close", fail_after_real_close)
    loop = asyncio.get_running_loop()
    real_clock = loop.time
    with pytest.raises(type(failure)) as caught:
        async with normal_close_schedule() as finish:
            await finish(
                connection, backend.close(),
                retained_closes=backend._retired_connection_closes,
            )
    assert caught.value is failure
    assert loop.time == real_clock
    assert not worker.is_alive()
    assert not backend.connection_retirement_pending


@pytest.mark.asyncio
@pytest.mark.parametrize("late_failure", [None, "ordinary", "unknown"])
@pytest.mark.parametrize("repeated", [False, True])
async def test_normal_close_schedule_joins_on_caller_cancellation_and_restores_clock(
    tmp_path, monkeypatch, late_failure, repeated,
):
    backend = SQLiteBackend(str(tmp_path / "cancelled-close.db"))
    await backend.connect()
    connection = backend._connection
    assert connection is not None
    worker = aiosqlite_worker(connection)
    entered, release = block_aiosqlite_native_close(connection)
    failure = (
        ExecutionCommitOutcomeError("unknown") if late_failure == "unknown" else
        OSError("original late native close failure") if late_failure == "ordinary" else None
    )
    native_original = OriginalFailure.capture(failure) if failure is not None else None
    if failure is not None:
        close = connection.close

        async def fail_after_real_close():
            await close()
            raise failure

        monkeypatch.setattr(connection, "close", fail_after_real_close)
    loop = asyncio.get_running_loop()
    real_clock = loop.time
    caller_originals = []
    caller_observed = asyncio.Event()

    async def retire():
        async with normal_close_schedule() as finish:
            try:
                await finish(
                    connection, backend.close(),
                    retained_closes=backend._retired_connection_closes,
                )
            except asyncio.CancelledError as error:
                caller_originals.append(OriginalFailure.capture(error))
                caller_observed.set()
                raise

    owner = asyncio.create_task(retire())
    try:
        await wait_for_thread_event(entered, step="native close entry")
        assert worker.is_alive() and not owner.done()
        owner.cancel("original caller")
        await await_within_hang_guard(caller_observed.wait(), step="original caller cancellation")
        if repeated:
            owner.cancel("repeated original caller")
            for _ in range(3):
                await asyncio.sleep(0)
            assert not owner.done(), "repeated caller C escaped before the original worker retired"
        release.set()
        with pytest.raises(BaseException) as caught:
            await await_within_hang_guard(owner, step="cancelled close schedule retiring")
        assert len(caller_originals) == 1
        caller_originals[0].assert_retained(caught.value)
        if native_original is not None:
            native_original.assert_retained(caught.value)
        assert execution_commit_outcome(caught.value) == ("unknown" if late_failure == "unknown" else None)
        assert loop.time == real_clock
        assert not worker.is_alive()
        assert not backend.connection_retirement_pending
    finally:
        release.set()
        await asyncio.gather(owner, return_exceptions=True)
        await asyncio.to_thread(worker.join, HANG_GUARD_SECONDS)
        assert not worker.is_alive()
