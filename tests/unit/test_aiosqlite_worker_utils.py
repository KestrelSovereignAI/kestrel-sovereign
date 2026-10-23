"""Tests for aiosqlite worker-lifecycle synchronization helpers."""

import asyncio
import threading
from contextlib import nullcontext

import pytest
import pytest_asyncio

import kestrel_sovereign.storage.db.sqlite as sqlite_module
from kestrel_sovereign._async_ownership import await_owned_operation, await_owned_task
from kestrel_sovereign.execution_custody import (
    ExecutionCommitOutcomeError,
    execution_cleanup_debts,
    execution_commit_outcome,
    execution_observation_error,
)
from tests.unit.test_host_context_native_cleanup import _actual_operation
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


@pytest_asyncio.fixture
async def prejoined_snapshot_case(tmp_path):
    """Real private snapshot; explicit test-owner reconciliation on refusal.

    This resource fixture never reports a failed close as successful. Its
    final raw close is ONLY reconciliation after assertions for an original
    unstarted/failed closer, as in test_sqlite_inner_close_admission. Normal
    successful cases must already have retired the worker before teardown.
    """
    backend = SQLiteBackend(str(tmp_path / "prejoined-snapshot.db"))
    await backend.connect()
    snapshot = await backend._open_snapshot_read_connection()
    worker = aiosqlite_worker(snapshot)
    # This is the actual coordinating test owner's retained map; the wake
    # adapter separately requires its exact backend-owned map, not this one.
    retained_closes = {}
    real_close = snapshot.close
    failures = []
    try:
        yield snapshot, worker, retained_closes
    except BaseException as error:  # noqa: BLE001 - retain the original test body alongside cleanup
        failures.append(error)
    finally:
        try:
            if worker.is_alive():
                # A different, explicit TEST cleanup after the assertions,
                # never an ACK/retry inside finish(). Preserve the original map.
                outcome = await await_within_hang_guard(
                    await_owned_operation(
                        real_close(), retirement_resource=snapshot,
                        retain_originals=True,
                    ),
                    step="test-owner reconciliation of the exact snapshot close",
                )
                failures.extend((outcome.error, outcome.cancellation))
            await await_within_hang_guard(
                asyncio.to_thread(worker.join, HANG_GUARD_SECONDS),
                step="test-owner reconciliation of the exact snapshot worker",
            )
            assert not worker.is_alive()
            for record in retained_closes.values():
                if record.retirement_task is not None:
                    outcome = await await_within_hang_guard(
                        await_owned_task(record.retirement_task),
                        step="the canonical original snapshot reaper settling",
                    )
                    failures.extend((outcome.error, outcome.cancellation))
        except BaseException as error:  # noqa: BLE001 - do not skip primary retirement on snapshot failure
            failures.append(error)
        finally:
            try:
                outcome = await await_within_hang_guard(
                    await_owned_operation(
                        backend.close(), retirement_resource=backend,
                        retain_originals=True,
                    ),
                    step="the actual primary backend closing after snapshot cleanup",
                )
                failures.extend((outcome.error, outcome.cancellation))
            except BaseException as error:  # noqa: BLE001 - compose every actual cleanup original
                failures.append(error)
        terminal = execution_observation_error(*failures)
        if terminal is not None:
            raise terminal


@pytest.mark.asyncio
async def test_prejoined_schedule_requires_original_close_task_done_before_ack(
    prejoined_snapshot_case, monkeypatch,
):
    connection, worker, retained = prejoined_snapshot_case
    real_close = connection.close
    close_epilogue = asyncio.Event()
    release_epilogue = asyncio.Event()
    originals, acknowledgements = [], []
    loop = asyncio.get_running_loop()
    original_clock = loop.time

    async def close_with_epilogue():
        originals.append(asyncio.current_task())
        await real_close()
        close_epilogue.set()
        await release_epilogue.wait()

    monkeypatch.setattr(connection, "close", close_with_epilogue)

    def acknowledge(original_task):
        assert original_task is originals[0]
        assert original_task.done(), "worker exit was mistaken for Task completion"
        assert not worker.is_alive()
        acknowledgements.append(original_task)
        return sqlite_module._close_aiosqlite_connection(
            connection, close_task=original_task, retained_closes=retained,
        )

    owner = None
    body_failure = None
    try:
        async with normal_close_schedule(prejoined=True) as finish:
            owner = asyncio.create_task(finish(
                connection, acknowledge, retained_closes=retained,
            ))
            await await_within_hang_guard(close_epilogue.wait(), step="real close epilogue")
            await await_within_hang_guard(
                asyncio.to_thread(worker.join, HANG_GUARD_SECONDS),
                step="the real worker exiting before its close Task",
            )
            assert not worker.is_alive() and not originals[0].done()
            assert not acknowledgements and not owner.done()
            release_epilogue.set()
            await await_within_hang_guard(owner, step="same close Task and ACK completing")
        assert len(originals) == len(acknowledgements) == 1
        assert not retained and loop.time == original_clock
    except BaseException as error:  # noqa: BLE001 - keep assertion/control and actual owner failure
        body_failure = error
    finally:
        release_epilogue.set()
        failures = [body_failure]
        if owner is not None:
            outcome = await await_owned_task(owner)
            failures.extend((outcome.error, outcome.cancellation))
        terminal = execution_observation_error(*failures)
        if terminal is not None:
            raise terminal


@pytest.mark.asyncio
@pytest.mark.parametrize("failure_kind", ["ordinary", "control", "unknown"])
@pytest.mark.parametrize("phase", ["native", "ack"])
async def test_prejoined_schedule_preserves_original_native_or_ack_failure(
    prejoined_snapshot_case, monkeypatch, failure_kind, phase,
):
    connection, worker, retained = prejoined_snapshot_case
    failure = (
        asyncio.CancelledError("original prejoined close control")
        if failure_kind == "control" else
        ExecutionCommitOutcomeError("unknown") if failure_kind == "unknown" else
        OSError("original prejoined close failure")
    )
    source = OriginalFailure.capture(failure)
    real_close = connection.close
    originals, acknowledgements = [], []

    async def fail_after_real_close():
        originals.append(asyncio.current_task())
        await real_close()
        if phase == "native":
            raise failure

    monkeypatch.setattr(connection, "close", fail_after_real_close)

    async def acknowledge(original_task):
        acknowledgements.append(original_task)
        await sqlite_module._close_aiosqlite_connection(
            connection, close_task=original_task, retained_closes=retained,
        )
        if phase == "ack":
            raise failure

    with pytest.raises(BaseException) as caught:
        async with normal_close_schedule(prejoined=True) as finish:
            await finish(connection, acknowledge, retained_closes=retained)
    source.assert_retained(caught.value)
    assert len(originals) == 1 and originals[0].done()
    assert not worker.is_alive()
    assert len(acknowledgements) == (1 if phase == "ack" else 0)
    assert execution_commit_outcome(caught.value) == (
        "unknown" if failure_kind == "unknown" else None
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("late_failure", [None, "ordinary", "unknown"])
@pytest.mark.parametrize("repeated", [False, True])
async def test_prejoined_schedule_retains_caller_and_late_close_errors(
    prejoined_snapshot_case, monkeypatch, late_failure, repeated,
):
    connection, worker, retained = prejoined_snapshot_case
    entered, release = block_aiosqlite_native_close(connection)
    failure = (
        ExecutionCommitOutcomeError("unknown") if late_failure == "unknown" else
        OSError("late actual prejoined close error") if late_failure == "ordinary" else None
    )
    source = OriginalFailure.capture(failure) if failure is not None else None
    real_close = connection.close
    originals, acknowledgements, callers = [], [], []
    caller_observed = asyncio.Event()

    async def close_with_late_error():
        originals.append(asyncio.current_task())
        await real_close()
        if failure is not None:
            raise failure

    monkeypatch.setattr(connection, "close", close_with_late_error)

    def acknowledge(original_task):
        acknowledgements.append(original_task)
        return sqlite_module._close_aiosqlite_connection(
            connection, close_task=original_task, retained_closes=retained,
        )

    async def retire():
        async with normal_close_schedule(prejoined=True) as finish:
            try:
                await finish(connection, acknowledge, retained_closes=retained)
            except asyncio.CancelledError as error:
                callers.append(OriginalFailure.capture(error))
                caller_observed.set()
                raise

    owner = asyncio.create_task(retire())
    body_failure = None
    owner_outcome_asserted = False
    try:
        await wait_for_thread_event(entered, step="actual prejoined native close held")
        owner.cancel("original prejoined caller")
        await await_within_hang_guard(caller_observed.wait(), step="caller C observed")
        if repeated:
            owner.cancel("repeated prejoined caller")
            for _ in range(3):
                await asyncio.sleep(0)
            assert not owner.done()
        assert not acknowledgements
        release.set()
        with pytest.raises(BaseException) as caught:
            await await_within_hang_guard(owner, step="same cancelled close owner retiring")
        assert len(callers) == 1
        callers[0].assert_retained(caught.value)
        if source is not None:
            source.assert_retained(caught.value)
        assert len(originals) == 1 and originals[0].done()
        assert not worker.is_alive() and not acknowledgements
        assert execution_commit_outcome(caught.value) == (
            "unknown" if late_failure == "unknown" else None
        )
        owner_outcome_asserted = True
    except BaseException as error:  # noqa: BLE001 - a failed assertion cannot discard a late original
        body_failure = error
    finally:
        release.set()
        outcome = await await_owned_task(owner)
        # On the ordinary expected-negative path the terminal was already
        # asserted above. Any earlier assertion failure must compose ALL
        # actual owner outcomes rather than discard a late UNKNOWN or C.
        terminal = execution_observation_error(
            body_failure,
            outcome.error if not owner_outcome_asserted else None,
            outcome.cancellation,
        )
        if terminal is not None:
            raise terminal


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["native", "ack"])
@pytest.mark.parametrize("accepted", [False, True])
async def test_prejoined_schedule_keeps_exact_admission_receipt(
    prejoined_snapshot_case, phase, accepted,
):
    connection, worker, retained = prejoined_snapshot_case
    loop, previous = asyncio.get_running_loop(), asyncio.get_running_loop().get_task_factory()
    failure = OSError("exact prejoined admission publication failure")
    source = OriginalFailure.capture(failure)
    original_handle = connection._connection
    accepted_tasks, attempted, acknowledgement_inputs = [], [], []
    native_code = connection.close.__func__.__code__

    async def acknowledge(original_task):
        await sqlite_module._close_aiosqlite_connection(
            connection, close_task=original_task, retained_closes=retained,
        )

    def acknowledgement_factory(original_task):
        assert original_task.done() and not worker.is_alive()
        acknowledgement_inputs.append(original_task)
        return acknowledge(original_task)

    class FailedPublicationTask(asyncio.Task):
        def add_done_callback(self, callback, *args, **kwargs):
            if getattr(callback, "__name__", "") == "close_unstarted" and not attempted:
                attempted.append(True)
                super().add_done_callback(callback, *args, **kwargs)
                raise failure
            return super().add_done_callback(callback, *args, **kwargs)

    def factory(loop, coroutine, **kwargs):
        operation = _actual_operation(coroutine)
        target = native_code if phase == "native" else acknowledge.__code__
        if getattr(operation, "cr_code", None) is target:
            if not accepted:
                attempted.append(True)
                raise failure
            task = FailedPublicationTask(coroutine, loop=loop, **kwargs)
            accepted_tasks.append(task)
            return task
        return previous(loop, coroutine, **kwargs) if previous else asyncio.Task(coroutine, loop=loop, **kwargs)

    try:
        loop.set_task_factory(factory)
        with pytest.raises(BaseException) as caught:
            async with normal_close_schedule(prejoined=True) as finish:
                await finish(connection, acknowledgement_factory, retained_closes=retained)
        loop.set_task_factory(previous)
        source.assert_retained(caught.value)
        assert len(attempted) == 1
        assert len(accepted_tasks) == int(accepted)
        assert all(task.done() for task in accepted_tasks)
        if phase == "native":
            expected_task = accepted_tasks[0] if accepted else None
            assert worker.is_alive() and not acknowledgement_inputs
            assert retained[connection].close_task is expected_task
            assert any(resource is connection and task is expected_task
                       for resource, task in execution_cleanup_debts(caught.value))
            assert connection._connection is original_handle
        else:
            assert len(acknowledgement_inputs) == 1
            assert acknowledgement_inputs[0].done() and not worker.is_alive()
            assert not retained
    finally:
        loop.set_task_factory(previous)


@pytest.mark.asyncio
async def test_prejoined_schedule_rejects_wrong_original_ack_task(prejoined_snapshot_case):
    connection, worker, retained = prejoined_snapshot_case
    foreign = asyncio.create_task(asyncio.sleep(0))
    await foreign
    originals = []

    def wrong_acknowledgement(original_task):
        assert original_task.done() and original_task is not foreign
        originals.append(original_task)
        return sqlite_module._close_aiosqlite_connection(
            connection, close_task=foreign, retained_closes=retained,
        )

    with pytest.raises(AssertionError, match="prejoined ACK substituted the original close Task"):
        async with normal_close_schedule(prejoined=True) as finish:
            await finish(connection, wrong_acknowledgement, retained_closes=retained)
    assert len(originals) == 1 and originals[0].done()
    assert not worker.is_alive()


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
