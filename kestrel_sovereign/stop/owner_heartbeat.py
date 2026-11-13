"""The Distributed Stop owner's lease heartbeat, on a thread of its own (#3889).

The owner lease (``STOP_OWNER_LEASE_SECONDS``) is how other replicas know
this process still owns its in-flight generations. It used to be renewed by a
task on the event loop every agent's turn runs on, so a synchronous section
longer than the lease in one agent's turn let it lapse: the owner was fenced
and every agent's in-flight turn on the host was cancelled (#3745).

The heartbeat renews it from a daemon thread that runs its own event loop over
its own connection to the Stop database. It shares neither the server's loop
nor its connection, so a stalled loop no longer reaches the lease. What a
renewal proves is that this process is alive, not that any agent's loop is
responsive: a stall is reported by the event-loop stall watchdog and is never
a reason to fence.

When the process dies, or none of its threads can run, renewals stop and the
lease expires within its usual bound. So it does when the heartbeat thread
itself stops: the registry fences once its newest renewal is a lease old. The
heartbeat also stops once the thread that started it (the registry's loop
thread) has ended, closing its connection, so an interpreter that exits without
closing the registry is not kept alive by the connection's worker thread.

A heartbeat thread that exits without being stopped records why. The registry
reports the exit within one relay interval, whether or not any work is in
flight, and restarts the thread only for new work: an admission that finds it
exited starts a new one and admits nothing until that thread has renewed the
owner, so it never admits work that would be self-fenced one lease later. A
restarted thread that has not renewed the owner once it exited, or once a lease
has passed since it started, makes the owner's health unavailable, naming why.
That stands through every later restart, until the heartbeat renews the owner
(#3987). A thread that cannot be started at all counts as a launch whose thread
never ran, and is named the same way.

The heartbeat only publishes renewals. The registry folds each one into its
lease on its own loop, where its generations live, so the thread never touches
the loop's objects. Every renewal names the owner it renewed, and a renewal of
an owner the registry has since replaced describes nothing it holds. The newest
successful renewal is kept apart from the newest attempt: a stalled loop may
next look after a failed attempt, and the renewals made during the stall still
hold the lease.

The heartbeat changes where a renewal runs, not what its failure means. A
renewal error that carries execution control (``is_execution_control_error``),
such as a lost COMMIT acknowledgement, is never retried: the heartbeat keeps
the original exception on that attempt and renews nothing more. The registry
retains it on its own loop as terminal control, which fences the owner, refuses
new work and raises it at close, as when the relay renewed. Any other error is
retried, and the lease is judged only against the last successful renewal, so
a run of them fences once one lease has passed since it.

One coupling remains, in the database rather than the loop. A renewal is a
write, so a Stop-store write transaction that the server's loop holds open
across a stall (SQLite's database write lock, or a PostgreSQL row lock on one
of this owner's rows) holds the renewal back until that transaction ends.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass
from typing import Any, AsyncContextManager, AsyncIterator, Awaitable, Callable

from kestrel_sovereign.execution_custody import is_execution_control_error

logger = logging.getLogger(__name__)

THREAD_NAME = "kestrel-stop-owner-heartbeat"

# How often ``stop()`` looks whether the thread has exited. It looks from the
# registry's loop rather than joining in that loop's default executor: a join
# queued there starts its timeout only once a worker is free, so blocking work
# occupying every worker held close past its budget.
_EXIT_POLL_SECONDS = 0.01

DatabaseOpener = Callable[[], AsyncContextManager[Any]]
Renewal = Callable[[Any, str], Awaitable[Any]]


async def _no_schema(_database: Any) -> None:
    """The server's own connection already created the Stop schema."""


def independent_database_opener(database: Any) -> DatabaseOpener:
    """Open the Stop database again, over a connection of its own.

    Returns a factory for an async context manager that yields a fresh
    ``AsyncDatabase`` on whichever event loop enters it, and closes it on
    exit. Raises when the Stop database cannot be reached a second time.
    """

    from kestrel_sovereign.storage.async_database import AsyncDatabase
    from kestrel_sovereign.storage.db.sqlite import SQLiteBackend

    backend = getattr(database, "backend", None)
    if isinstance(backend, SQLiteBackend):
        path = backend.db_path
        if path == ":memory:":
            raise ValueError(
                "the Distributed Stop owner heartbeat needs a connection of "
                "its own, and an in-memory SQLite database has no second one"
            )

        @asynccontextmanager
        async def open_sqlite() -> AsyncIterator[AsyncDatabase]:
            opened = await AsyncDatabase.sqlite(path, schema_initializer=_no_schema)
            try:
                yield opened
            finally:
                await opened.close()

        return open_sqlite

    from kestrel_sovereign.storage.db.postgres import (
        NativeJournalPostgresConnection,
        PostgresBackend,
    )

    if isinstance(backend, PostgresBackend):
        import asyncpg

        args, kwargs = backend.independent_connection_recipe()
        kwargs["connection_class"] = NativeJournalPostgresConnection

        @asynccontextmanager
        async def open_postgres() -> AsyncIterator[AsyncDatabase]:
            pool = await asyncpg.create_pool(*args, min_size=0, max_size=1, **kwargs)
            try:
                opened = await AsyncDatabase.from_connected_backend(
                    PostgresBackend.from_pool(pool, advisory_max_pool_size=1),
                    schema_initializer=_no_schema,
                )
                try:
                    yield opened
                finally:
                    await opened.close()
            finally:
                await pool.close()

        return open_postgres
    raise TypeError(
        "the Distributed Stop owner heartbeat needs a SQLite file or PostgreSQL "
        f"Stop database, not {type(backend).__name__}"
    )


@dataclass(frozen=True, slots=True)
class OwnerRenewal:
    """One finished renewal attempt for one owner identity.

    ``started`` is ``time.monotonic()`` before the renewal's transaction
    began, so it is no later than the heartbeat it wrote. ``poll`` is the
    owner's renewed inventory, or ``None`` when the attempt failed with
    ``failure`` (the exception's type name). ``control`` is the original
    exception when it carried execution control; the heartbeat attempts no
    renewal after such an attempt.
    """

    owner_id: str
    started: float
    finished: float
    poll: Any | None
    failure: str | None
    control: BaseException | None = None


class OwnerHeartbeat:
    """Renew one owner identity every interval from a dedicated thread.

    ``start``, ``retarget``, ``ensure_running`` and ``stop`` are called on
    the registry's loop; the rest of the work happens on the thread.
    """

    def __init__(
        self,
        open_database: DatabaseOpener,
        renew: Renewal,
        *,
        interval_seconds: float,
    ) -> None:
        if interval_seconds <= 0:
            raise ValueError("owner heartbeat interval must be positive")
        self._open_database = open_database
        self._renew = renew
        self._interval = float(interval_seconds)
        self._stopping = threading.Event()
        self._thread: threading.Thread | None = None
        self._host: threading.Thread | None = None
        # Shared with the heartbeat thread.
        self._lock = threading.Lock()
        self._owner_id: str | None = None
        self._in_flight: tuple[str, float] | None = None
        self._latest: OwnerRenewal | None = None
        self._latest_renewed: OwnerRenewal | None = None
        # The type name of an exception a heartbeat connection's close
        # raised: that connection may still be open, so it stays owned.
        self._connection_failure: str | None = None
        # How many threads were started, when the newest one was, and why
        # the newest one exited without being stopped, or could not start.
        self._launches = 0
        self._launched_at: float | None = None
        self._exit_cause: str | None = None
        self._start_failure: str | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._wake: asyncio.Event | None = None

    @property
    def alive(self) -> bool:
        thread = self._thread
        return thread is not None and thread.is_alive()

    def start(self, owner_id: str) -> None:
        if self._thread is not None:
            raise RuntimeError("owner heartbeat was already started")
        self._host = threading.current_thread()
        self.retarget(owner_id)
        self._launch()

    def _launch(self) -> None:
        """Start a new thread; a start that raises still counts as a launch.

        The launch is counted, and timed, before ``Thread.start()``, so a
        thread that could not be started is a launch whose thread never ran:
        :meth:`describe` and :meth:`exit_cause` name why until the next
        launch. The exception is raised to the caller.
        """

        thread = threading.Thread(target=self._run, name=THREAD_NAME, daemon=True)
        with self._lock:
            # Anything the previous thread left in flight ended with it.
            self._in_flight = None
            self._exit_cause = None
            self._start_failure = None
            self._launches += 1
            self._launched_at = time.monotonic()
        try:
            thread.start()
        except BaseException as error:
            with self._lock:
                self._start_failure = f"{type(error).__name__}: {error}"
            raise
        self._thread = thread

    @property
    def launches(self) -> int:
        """How many heartbeat threads were started: one, until a restart."""

        with self._lock:
            return self._launches

    @property
    def launched_at(self) -> float | None:
        """``time.monotonic()`` just before the newest thread was started."""

        with self._lock:
            return self._launched_at

    def exit_cause(self) -> str | None:
        """Why the heartbeat thread exited without being stopped.

        ``None`` while it runs, before it was started, and once it was
        stopped.
        """

        if self._thread is None or self._stopping.is_set() or self.alive:
            return None
        with self._lock:
            cause = self._exit_cause
            start_failure = self._start_failure
        if start_failure is not None:
            return f"a new thread could not be started ({start_failure})"
        return cause or "it exited without recording why"

    def renewed_since_launch(self, owner_id: str) -> bool:
        """Whether the newest thread has renewed ``owner_id`` since it started."""

        with self._lock:
            renewed = self._latest_renewed
            launched_at = self._launched_at
        return (
            renewed is not None
            and launched_at is not None
            and renewed.owner_id == owner_id
            and renewed.started >= launched_at
        )

    def ensure_running(self) -> bool:
        """Start a new thread if the previous one exited without being stopped.

        Returns whether a new thread was started. ``False`` when there was
        nothing to restart (never started, stopped, or still running), and
        when ``Thread.start()`` raised ``RuntimeError`` (the runtime could not
        create a thread): that launch is logged and named by
        :meth:`describe` until the next one.
        """

        if self._thread is None or self._stopping.is_set() or self.alive:
            return False
        logger.warning(
            "Distributed Stop owner heartbeat thread had exited; starting a new one"
        )
        try:
            self._launch()
        except RuntimeError:
            logger.error(
                "Distributed Stop owner heartbeat thread could not be started; "
                "nothing renews the owner lease until one is",
                exc_info=True,
            )
            return False
        return True

    def retarget(self, owner_id: str) -> None:
        """Renew ``owner_id`` from the next renewal on; never the previous one."""

        with self._lock:
            self._owner_id = owner_id

    def latest(self) -> OwnerRenewal | None:
        """The most recent finished renewal attempt, of whichever owner.

        Once an attempt carries ``control``, it stays the most recent one.
        """

        with self._lock:
            return self._latest

    def latest_renewed(self) -> OwnerRenewal | None:
        """The most recent successful renewal, of whichever owner.

        A later failed attempt never replaces it, so the renewals made while
        the registry's loop was blocked still reach the lease once the loop
        runs again, whatever the attempt just before that was (#3889).
        """

        with self._lock:
            return self._latest_renewed

    def describe(self, now: float, lease_seconds: float) -> str:
        """Name what the heartbeat is doing, for a fence reason."""

        with self._lock:
            in_flight = self._in_flight
            latest = self._latest
            renewed = self._latest_renewed
            launched_at = self._launched_at
            start_failure = self._start_failure
        if self._thread is None:
            return "the heartbeat thread was never started"
        if not self.alive:
            if start_failure is not None and not self._stopping.is_set():
                return f"a new heartbeat thread could not be started ({start_failure})"
            cause = self.exit_cause()
            if cause is None:
                return "the heartbeat thread has exited"
            return f"the heartbeat thread has exited ({cause})"
        if in_flight is not None:
            flight = now - in_flight[1]
            if (
                latest is not None
                and latest.failure is not None
                and (launched_at is None or latest.started >= launched_at)
            ):
                # A retry: name the failure it retries, which may recur.
                return (
                    f"its last renewal failed ({latest.failure}); its retry "
                    f"has been in flight for {flight:.1f} s"
                )
            return f"its renewal has been in flight for {flight:.1f} s"
        if latest is None:
            return "it has not finished a renewal"
        if latest.control is not None:
            return (
                f"its last renewal raised execution control ({latest.failure}), "
                "which is never retried"
            )
        if launched_at is not None and latest.started < launched_at:
            return (
                f"it was restarted {now - launched_at:.1f} s ago and has not "
                "finished a renewal since"
            )
        if latest.failure is not None:
            if renewed is None:
                return f"its last renewal failed ({latest.failure})"
            return (
                f"its last renewal failed ({latest.failure}); its last "
                f"successful one started {now - renewed.started:.1f} s ago"
            )
        took = latest.finished - latest.started
        if took >= lease_seconds:
            return f"its last renewal took {took:.1f} s"
        return f"its last renewal finished {now - latest.finished:.1f} s ago"

    def unretired(self) -> str | None:
        """Why the heartbeat still holds a thread or connection, or ``None``."""

        if self.alive:
            return "its thread has not exited"
        with self._lock:
            failure = self._connection_failure
        if failure is not None:
            return f"its database connection did not close ({failure})"
        return None

    def retirement_error(self) -> TimeoutError | None:
        """Retirement debt for a heartbeat that has not released everything.

        ``None`` once its thread has exited and every connection it opened
        closed. Otherwise the error names this heartbeat and its thread as
        still owned, so its caller keeps them rather than reporting them
        closed.
        """

        reason = self.unretired()
        if reason is None:
            return None
        error = TimeoutError(
            f"Distributed Stop owner heartbeat has not retired: {reason}"
        )
        error._kestrel_native_retirement_debts = ((self, self._thread),)
        return error

    async def stop(self, *, timeout: float) -> bool:
        """Stop the thread and wait up to ``timeout`` for it to exit.

        A renewal in flight finishes, and the thread closes its connection,
        before it exits. Returns whether it retired: its thread exited in
        time and every connection it opened closed. When it did not, the
        caller still owns both; :meth:`retirement_error` names them, and a
        later ``stop`` waits again. The wait takes no worker from the loop's
        default executor, so blocking work occupying them cannot extend it.
        """

        thread = self._thread
        self._stopping.set()
        if thread is None:
            return True
        with self._lock:
            loop, wake = self._loop, self._wake
        if loop is not None and wake is not None:
            try:
                loop.call_soon_threadsafe(wake.set)
            except RuntimeError:
                pass  # its loop already closed: the thread is exiting
        await _wait_for_exit(thread, timeout)
        unretired = self.unretired()
        if unretired is not None:
            logger.warning(
                "Distributed Stop owner heartbeat did not retire within "
                "%.1f s: %s",
                timeout,
                unretired,
            )
            return False
        return True

    # -- heartbeat thread --------------------------------------------------

    def _run(self) -> None:
        try:
            asyncio.run(self._beat_until_stopped())
        except BaseException as error:
            if self._stopping.is_set():
                # stop() reports this through unretired().
                logger.error(
                    "Distributed Stop owner heartbeat failed while stopping",
                    exc_info=True,
                )
                return
            with self._lock:
                self._exit_cause = f"it raised {type(error).__name__}"
            # A thread's death is otherwise only printed to stderr. The
            # registry reports it too, and restarts it for new work.
            logger.critical(
                "Distributed Stop owner heartbeat thread died; nothing renews "
                "the owner lease until it is restarted",
                exc_info=True,
            )
        else:
            if not self._stopping.is_set():
                with self._lock:
                    self._exit_cause = "the thread that started it has ended"

    async def _beat_until_stopped(self) -> None:
        wake = asyncio.Event()
        with self._lock:
            self._loop = asyncio.get_running_loop()
            self._wake = wake
        failing = False
        host = self._host
        connection = AsyncExitStack()
        try:
            database = None
            while not self._stopping.is_set() and host.is_alive():
                with self._lock:
                    latest = self._latest
                    # A control-bearing failure is never retried, by this
                    # thread or one ensure_running() starts after it.
                    halted = latest is not None and latest.control is not None
                    owner_id = self._owner_id
                    started = time.monotonic()
                    if not halted:
                        self._in_flight = (owner_id, started)
                if not halted:
                    poll = failure = control = None
                    try:
                        if database is None:
                            database = await connection.enter_async_context(
                                self._open_database()
                            )
                        poll = await self._renew(database, owner_id)
                    except BaseException as error:
                        if is_execution_control_error(error):
                            control = error
                        elif not isinstance(error, Exception):
                            raise
                        failure = type(error).__name__
                        if control is not None:
                            logger.error(
                                "Distributed Stop owner heartbeat renewal raised "
                                "execution control (%s); it is never retried, and "
                                "the registry fences on it",
                                failure,
                                exc_info=(type(error), error, error.__traceback__),
                            )
                        elif not failing:
                            logger.warning(
                                "Distributed Stop owner heartbeat could not renew "
                                "the owner lease (%s); retrying every %.2f s",
                                failure,
                                self._interval,
                                exc_info=(type(error), error, error.__traceback__),
                            )
                        failing = True
                    else:
                        if failing:
                            logger.warning(
                                "Distributed Stop owner heartbeat renews the "
                                "owner lease again"
                            )
                        failing = False
                    renewal = OwnerRenewal(
                        owner_id, started, time.monotonic(), poll, failure, control
                    )
                    with self._lock:
                        self._in_flight = None
                        self._latest = renewal
                        if failure is None:
                            self._latest_renewed = renewal
                    if control is not None and not isinstance(control, Exception):
                        # Published first, so the registry still retains it.
                        raise control
                try:
                    await asyncio.wait_for(wake.wait(), timeout=self._interval)
                except TimeoutError:
                    pass
        finally:
            try:
                await connection.aclose()
            except BaseException as error:
                with self._lock:
                    self._connection_failure = type(error).__name__
                raise
            finally:
                with self._lock:
                    self._loop = None
                    self._wake = None


async def _wait_for_exit(thread: threading.Thread, timeout: float) -> None:
    """Wait up to ``timeout`` seconds for ``thread`` to exit, on this loop."""

    loop = asyncio.get_running_loop()
    deadline = loop.time() + max(timeout, 0.0)
    while thread.is_alive():
        remaining = deadline - loop.time()
        if remaining <= 0:
            return
        await asyncio.sleep(min(_EXIT_POLL_SECONDS, remaining))


__all__ = [
    "OwnerHeartbeat",
    "OwnerRenewal",
    "independent_database_opener",
]
