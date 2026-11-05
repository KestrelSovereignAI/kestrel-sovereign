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

The heartbeat only publishes renewals. The registry folds each one into its
lease on its own loop, where its generations live, so the thread never touches
the loop's objects. Every renewal names the owner it renewed, and a renewal of
an owner the registry has since replaced describes nothing it holds. The newest
successful renewal is kept apart from the newest attempt: a stalled loop may
next look after a failed attempt, and the renewals made during the stall still
hold the lease.

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

logger = logging.getLogger(__name__)

THREAD_NAME = "kestrel-stop-owner-heartbeat"

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
    ``failure`` (the exception's type name).
    """

    owner_id: str
    started: float
    finished: float
    poll: Any | None
    failure: str | None


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
        thread = threading.Thread(target=self._run, name=THREAD_NAME, daemon=True)
        thread.start()
        self._thread = thread

    def ensure_running(self) -> None:
        """Start a new thread if the previous one exited without being stopped."""

        if self._thread is None or self._stopping.is_set() or self.alive:
            return
        logger.warning(
            "Distributed Stop owner heartbeat thread had exited; starting a new one"
        )
        self._launch()

    def retarget(self, owner_id: str) -> None:
        """Renew ``owner_id`` from the next renewal on; never the previous one."""

        with self._lock:
            self._owner_id = owner_id

    def latest(self) -> OwnerRenewal | None:
        """The most recent finished renewal attempt, of whichever owner."""

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
        if self._thread is None:
            return "the heartbeat thread was never started"
        if not self.alive:
            return "the heartbeat thread has exited"
        if in_flight is not None:
            return f"its renewal has been in flight for {now - in_flight[1]:.1f} s"
        if latest is None:
            return "it has not finished a renewal"
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
        later ``stop`` waits again.
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
        await asyncio.to_thread(thread.join, max(timeout, 0.0))
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
        except BaseException:
            if self._stopping.is_set():
                # stop() reports this through unretired().
                logger.error(
                    "Distributed Stop owner heartbeat failed while stopping",
                    exc_info=True,
                )
                return
            # A thread's death is otherwise only printed to stderr. The
            # registry notices it as a lease that stopped being renewed.
            logger.critical(
                "Distributed Stop owner heartbeat thread died; the owner lease "
                "lapses within one lease and the registry fences",
                exc_info=True,
            )

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
                    owner_id = self._owner_id
                    started = time.monotonic()
                    self._in_flight = (owner_id, started)
                poll = failure = None
                try:
                    if database is None:
                        database = await connection.enter_async_context(
                            self._open_database()
                        )
                    poll = await self._renew(database, owner_id)
                except Exception as error:
                    failure = type(error).__name__
                    if not failing:
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
                            "Distributed Stop owner heartbeat renews the owner "
                            "lease again"
                        )
                    failing = False
                renewal = OwnerRenewal(
                    owner_id, started, time.monotonic(), poll, failure
                )
                with self._lock:
                    self._in_flight = None
                    self._latest = renewal
                    if failure is None:
                        self._latest_renewed = renewal
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


__all__ = [
    "OwnerHeartbeat",
    "OwnerRenewal",
    "independent_database_opener",
]
