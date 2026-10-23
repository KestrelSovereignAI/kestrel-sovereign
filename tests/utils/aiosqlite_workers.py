"""Failure injection helpers for aiosqlite worker-lifecycle tests."""

from __future__ import annotations

import asyncio
import enum
import threading
import time
from collections.abc import Awaitable, Callable, Iterator
from contextlib import asynccontextmanager, contextmanager, nullcontext, suppress
from typing import Any
from unittest.mock import patch

import aiosqlite

import kestrel_sovereign.storage.db.sqlite as sqlite_module
from kestrel_sovereign._async_ownership import (
    OwnedTaskAdmission,
    create_owned_task,
    owned_task_admission_error,
    owned_task_error,
)
from kestrel_sovereign.execution_custody import execution_observation_error
from tests.utils.virtual_clock import (
    HANG_GUARD_SECONDS,
    await_within_hang_guard,
    pinned_loop_clock,
)


def aiosqlite_worker(connection: aiosqlite.Connection) -> threading.Thread:
    """Return the worker for either supported aiosqlite threading model."""
    return getattr(connection, "_thread", connection)


class _SlowCloseConnection:
    """Hold a real native close at the same boundary as stalled disk I/O."""

    def __init__(
        self, real, entered: threading.Event, release: threading.Event, *,
        after_native_close: bool = False, timeline: dict[str, float] | None = None,
    ):
        self._real = real
        self._entered = entered
        self._release = release
        self._after_native_close = after_native_close
        self._timeline = timeline

    def close(self) -> None:
        if not self._after_native_close:
            self._entered.set()
            self._release.wait(HANG_GUARD_SECONDS)
        if self._timeline is not None:
            self._timeline["native_close_started_at"] = time.monotonic()
        self._real.close()
        if self._timeline is not None:
            self._timeline["native_close_returned_at"] = time.monotonic()
        if self._after_native_close:
            self._entered.set()
            self._release.wait(HANG_GUARD_SECONDS)

    def __getattr__(self, name):
        return getattr(self._real, name)


def block_aiosqlite_native_close(
    connection: aiosqlite.Connection,
    *,
    after_native_close: bool = False,
    timeline: dict[str, float] | None = None,
) -> tuple[threading.Event, threading.Event]:
    """Hold before/after the same native close; optionally observe real time.

    Defaults preserve existing before-close injection. AFTER means sqlite's
    handle has closed, not that its aiosqlite Task/worker has acknowledged it.
    """
    entered = threading.Event()
    release = threading.Event()
    connection._connection = _SlowCloseConnection(
        connection._connection, entered, release,
        after_native_close=after_native_close, timeline=timeline,
    )
    return entered, release


class CloseStage(enum.Enum):
    """The two waits of one bounded aiosqlite close, in order."""

    CONNECTION_CLOSE = "Connection.close()"
    WORKER_EXIT = "worker exit"


class BoundedCloseStages:
    """Which bounded closes have reached which deadline wait.

    ``kestrel_sovereign.storage.db.sqlite`` closes a connection in two waits
    under one absolute deadline: for ``Connection.close()`` to finish, then
    for the worker thread to exit. A close arms each wait before it next
    yields, so once :meth:`reached` returns, moving the loop clock past the
    deadline expires exactly that wait. The worker-exit wait arms only for a
    worker still alive when the close reaches it.
    """

    def __init__(self) -> None:
        self._reached: dict[
            tuple[CloseStage, aiosqlite.Connection], asyncio.Event
        ] = {}
        #: The absolute deadline each connection's close was given.
        self.deadlines: dict[aiosqlite.Connection, float] = {}
        #: The original native close Task, including caller-supplied owners.
        self.close_tasks: dict[aiosqlite.Connection, asyncio.Task[None]] = {}
        self.entered_at: dict[tuple[CloseStage, aiosqlite.Connection], float] = {}

    def _event(
        self, stage: CloseStage, connection: aiosqlite.Connection
    ) -> asyncio.Event:
        return self._reached.setdefault((stage, connection), asyncio.Event())

    def _enter(
        self, stage: CloseStage, connection: aiosqlite.Connection
    ) -> None:
        self.entered_at[(stage, connection)] = time.monotonic()
        self._event(stage, connection).set()

    def has_reached(
        self, connection: aiosqlite.Connection, stage: CloseStage
    ) -> bool:
        """Whether the close of ``connection`` has entered ``stage`` yet."""
        return self._event(stage, connection).is_set()

    async def reached(
        self, connection: aiosqlite.Connection, stage: CloseStage
    ) -> None:
        """Return once the close of ``connection`` has entered ``stage``."""
        await await_within_hang_guard(
            self._event(stage, connection).wait(),
            step=f"the close reaching its {stage.value} wait",
        )


@contextmanager
def observe_bounded_close_stages() -> Iterator[BoundedCloseStages]:
    """Record each bounded close as it reaches each of its deadline waits."""
    stages = BoundedCloseStages()
    finish_close = sqlite_module._finish_aiosqlite_connection_close
    wait_for_worker_exit = sqlite_module._wait_for_aiosqlite_worker_shutdown

    async def observed_finish_close(connection, **kwargs):
        stages.deadlines[connection] = kwargs["deadline"]
        stages.close_tasks[connection] = kwargs["close_task"]
        stages._enter(CloseStage.CONNECTION_CLOSE, connection)
        return await finish_close(connection, **kwargs)

    async def observed_wait_for_worker_exit(connection, **kwargs):
        stages._enter(CloseStage.WORKER_EXIT, connection)
        return await wait_for_worker_exit(connection, **kwargs)

    with patch.object(
        sqlite_module,
        "_finish_aiosqlite_connection_close",
        observed_finish_close,
    ), patch.object(
        sqlite_module,
        "_wait_for_aiosqlite_worker_shutdown",
        observed_wait_for_worker_exit,
    ):
        yield stages


@asynccontextmanager
async def normal_close_schedule(*, prejoined: bool = False):
    """Exercise real closes before their deadlines, not a disk-speed claim.

    Pin before the enclosing operation captures its deadline. Each original
    close and worker must actually finish; only then wake its worker poll.
    Thread-based guards bound real hangs independently of the pinned clock.
    Production close budgets and deadline-expiry behavior are not changed.

    ``prejoined=True`` instead keeps REAL loop time and accepts a factory
    ``operation(original_close_task)``. The fixture admits connection.close
    exactly once, physically joins that same Task/worker, then passes the
    same terminal Task to the existing bounded closer for acknowledgement.
    This is deliberately normal functional-test scheduling BEFORE the Core
    ACK budget, not production close-admission/latency coverage or WORK rights.
    """
    owners = []
    workers = []
    retirement_maps = []
    failures = []
    unstarted_connections = set()

    async def settle(tasks, *, step):
        # The guard observes this exact completion Future, so withdrawing a
        # caller never cancels the original close/join owners. Repeated C does
        # not reset the independent real-time hang deadline.
        completion = asyncio.gather(*tasks, return_exceptions=True)
        deadline = time.monotonic() + HANG_GUARD_SECONDS
        try:
            while not completion.done():
                try:
                    await await_within_hang_guard(
                        completion, step=step,
                        seconds=max(0.0, deadline - time.monotonic()),
                    )
                except asyncio.CancelledError as error:
                    failures.append(error)
        finally:
            # Harvest the actual owners, not cancellation projections returned
            # by gather, and never discard a late native UNKNOWN or error.
            failures.extend(owned_task_error(task) for task in tasks if task.done())

    # Late cleanup still needs the real stage observer after clock restoration.
    with observe_bounded_close_stages() as stages:
        try:
            with (nullcontext(None) if prejoined else pinned_loop_clock()) as clock:
                async def finish(connection, operation, *, retained_closes):
                    retirement_maps.append(retained_closes)
                    worker = aiosqlite_worker(connection)
                    if prejoined:
                        native_admission = OwnedTaskAdmission()
                        try:
                            close_task = create_owned_task(
                                connection.close(), retain_originals=True,
                                retirement_resource=connection,
                                admission=native_admission,
                            )
                        except BaseException as error:  # noqa: BLE001 - retain the exact native admission and debt
                            # The failed start gate never entered close(). A
                            # live worker is unresolved original debt, not an
                            # operation this fixture can silently retry.
                            if native_admission.task is not None:
                                owners.append(native_admission.task)
                            unstarted_connections.add(connection)
                            terminal = owned_task_admission_error(
                                connection, native_admission, error
                            )
                            sqlite_module._retain_aiosqlite_close(
                                connection, native_admission.task,
                                retained_closes, control_error=terminal,
                            )
                            raise terminal
                    else:
                        workers.append(worker)
                        close_task = create_owned_task(operation, retain_originals=True)
                    owners.append(close_task)
                    if prejoined:
                        workers.append(worker)
                    if not prejoined:
                        await stages.reached(connection, CloseStage.CONNECTION_CLOSE)
                        deadline = stages.deadlines[connection]
                    else:
                        # Worker exit alone does not prove the actual close
                        # coroutine's epilogue finished. No ACK recipe may be
                        # constructed until this SAME Task is terminal.
                        await await_within_hang_guard(
                            close_task, step="the original native SQLite close Task completing"
                        )
                        assert close_task.done()
                    join_task = asyncio.create_task(
                        asyncio.to_thread(worker.join, HANG_GUARD_SECONDS)
                    )
                    owners.append(join_task)
                    await await_within_hang_guard(
                        join_task, step="the original SQLite worker retiring"
                    )
                    assert not worker.is_alive()
                    if prejoined:
                        acknowledgement_admission = OwnedTaskAdmission()
                        try:
                            acknowledgement = create_owned_task(
                                operation(close_task), retain_originals=True,
                                admission=acknowledgement_admission,
                            )
                        except BaseException as error:  # noqa: BLE001 - join an admitted failed ACK gate too
                            if acknowledgement_admission.task is not None:
                                owners.append(acknowledgement_admission.task)
                            raise owned_task_admission_error(
                                None, acknowledgement_admission, error
                            )
                        owners.append(acknowledgement)
                        await stages.reached(connection, CloseStage.CONNECTION_CLOSE)
                        assert stages.close_tasks[connection] is close_task, (
                            "prejoined ACK substituted the original close Task"
                        )
                        return await await_within_hang_guard(
                            acknowledgement,
                            step="the original prejoined SQLite close acknowledgement",
                        )
                    clock.advance(sqlite_module._AIOSQLITE_WORKER_SHUTDOWN_POLL_S * 1.1)
                    assert clock.time() < deadline
                    return await await_within_hang_guard(
                        close_task, step="the original SQLite close completing"
                    )

                yield finish
        except BaseException as error:  # noqa: BLE001 - rethrow every original body/control after joining
            failures.append(error)
        finally:
            # The caller first drains enclosing operations that can register
            # more closes. Keep observing while settling every original owner.
            try:
                if owners:
                    await settle(owners, step="settling the original SQLite close/join owners")
                worker_joins = [
                    asyncio.create_task(asyncio.to_thread(worker.join, HANG_GUARD_SECONDS))
                    for worker in workers
                ]
                await settle(worker_joins, step="joining the original SQLite worker on exceptional exit")
                for worker in workers:
                    assert not worker.is_alive(), "original SQLite worker did not retire"
                retirement_tasks = {
                    retained.retirement_task
                    for retained_closes in retirement_maps
                    for connection, retained in retained_closes.items()
                    if retained.retirement_task is not None
                    and not (
                        connection in unstarted_connections
                        and aiosqlite_worker(connection).is_alive()
                    )
                }
                await settle(retirement_tasks, step="joining original SQLite retirement tasks")
                # Known-unstarted native admissions are intentionally not a
                # normal worker join: the failed close never queued STOP.
                # Their original Task-or-None, worker and error remain rooted
                # in the supplied canonical retained map. The test owner must
                # explicitly reconcile that resource after observing failure;
                # there is no automatic replacement close or successful ACK.
            except BaseException as error:  # noqa: BLE001 - compose, never hide, physical cleanup failure
                failures.append(error)
            terminal = execution_observation_error(*failures)
            if terminal is not None:
                raise terminal


@contextmanager
def delay_aiosqlite_worker_exit(
    release_worker: threading.Event,
    worker_exit_delayed: threading.Event,
    *,
    should_delay: Callable[[threading.Thread], bool] | None = None,
) -> Iterator[list[threading.Thread]]:
    """Delay aiosqlite's final worker return on 0.21 and 0.22+.

    ``Connection.close()`` acknowledges its stop sentinel before this final
    return, which reproduces the event-loop teardown race without mocking the
    connection lifecycle itself.
    """
    workers: list[threading.Thread] = []

    def _delay_current_worker(worker: threading.Thread) -> None:
        if should_delay is not None and not should_delay(worker):
            return
        workers.append(worker)
        worker_exit_delayed.set()
        release_worker.wait()

    worker_target = getattr(aiosqlite.core, "_connection_worker_thread", None)

    if worker_target is not None:

        def delayed_worker(*args, **kwargs):
            try:
                return worker_target(*args, **kwargs)
            finally:
                _delay_current_worker(threading.current_thread())

        patcher = patch.object(
            aiosqlite.core, "_connection_worker_thread", delayed_worker,
        )
    else:
        original_run = aiosqlite.Connection.run

        def delayed_run(connection, *args, **kwargs):
            try:
                return original_run(connection, *args, **kwargs)
            finally:
                _delay_current_worker(connection)

        patcher = patch.object(aiosqlite.Connection, "run", delayed_run)

    with patcher:
        yield workers


async def wait_until_aiosqlite_worker_exit_is_delayed(
    worker_exit_delayed: threading.Event,
) -> None:
    """Yield until a failure-injected worker has reached its final return."""
    while not worker_exit_delayed.is_set():
        await asyncio.sleep(0)


async def wait_for_lifecycle_checkpoint(
    checkpoint: Awaitable[Any],
    lifecycle_task: asyncio.Task[Any],
    *,
    description: str,
    harness_timeout: float = 10.0,
) -> None:
    """Wait for a checkpoint unless the lifecycle terminates first.

    Lifecycle regressions should fail because the owner completed, failed, or
    cancelled before reaching the expected checkpoint—not because a busy CI
    runner did not schedule two cooperating tasks within a one-second wall
    clock window.  ``harness_timeout`` remains a generous deadlock guard; it
    is not part of the behavior being asserted.
    """
    checkpoint_task = asyncio.ensure_future(checkpoint)
    awaited: set[asyncio.Future[Any]] = {checkpoint_task, lifecycle_task}
    try:
        done, _pending = await asyncio.wait(
            awaited,
            timeout=harness_timeout,
            return_when=asyncio.FIRST_COMPLETED,
        )
        # Fail closed when both tasks become ready in the same event-loop
        # turn.  A completed lifecycle cannot prove that it was still live
        # when the checkpoint was reached, so owner termination takes
        # precedence over a simultaneously observed checkpoint.
        if lifecycle_task in done:
            if lifecycle_task.cancelled():
                raise AssertionError(
                    f"Lifecycle cancelled before {description}"
                )
            failure = lifecycle_task.exception()
            if failure is not None:
                raise AssertionError(
                    f"Lifecycle failed before {description}"
                ) from failure
            raise AssertionError(
                f"Lifecycle completed before {description}"
            )

        if checkpoint_task in done:
            await checkpoint_task
            return

        raise AssertionError(
            f"Lifecycle and {description} made no progress within the "
            f"{harness_timeout:g}s test-harness guard"
        )
    finally:
        if not checkpoint_task.done():
            checkpoint_task.cancel()
        with suppress(asyncio.CancelledError):
            await checkpoint_task
